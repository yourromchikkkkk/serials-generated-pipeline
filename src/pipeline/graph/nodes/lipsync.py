"""Lip-sync gate — merges each shot's validated video and audio via the configured lip-sync
provider, then verifies the merge with a sync-confidence score: mouth movement actually matching
the audio, not just each asset being individually correct. `lipsync_confidence_threshold`
defaults to null per the run-config spec, so by default this records a score without rejecting.

Shots with no dialogue (no voice generation record) have nothing to sync — they're carried
through untouched, without a LipsyncResult."""

from concurrent.futures import ThreadPoolExecutor

from langsmith import traceable

from pipeline.clients import fal_client, minio_client
from pipeline.clients.litellm_client import chat_completion
from pipeline.config import get_settings
from pipeline.graph.nodes.per_shot_generation import extract_frame, latest_for, select_attempt
from pipeline.graph.state import LipsyncResult, Run, ShotSpec

GATE = "lipsync_gate"

DEFAULT_LIPSYNC_PROVIDER = "act_two"


def _max_attempts(run: Run) -> int:
    retry_limit = run.parameters.get("lipsync_retry_limit")
    return (retry_limit + 1) if retry_limit is not None else 1


def _parse_score(content: str) -> float:
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return 0.0
    return 0.0


@traceable(run_type="llm", name="lipsync.confidence_score")
def _confidence_score(synced_bytes: bytes, duration_sec: float, transcript: str) -> float:
    """Heuristic proxy for real lip-sync detection: samples two frames and asks a vision model
    whether the mouth shapes look plausible for the transcribed speech. Not a specialized
    lip-sync-detection model — a pragmatic stand-in consistent with the other vision-judgment
    gates in per_shot_generation.py."""
    early_frame = extract_frame(synced_bytes, duration_sec * 0.25)
    late_frame = extract_frame(synced_bytes, duration_sec * 0.75)
    response = chat_completion(
        model=get_settings().vision_gate_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "These two frames are sampled from a lip-synced video clip. The speaker is "
                    "saying the given line. Score 0-1 how plausible the mouth shape/positioning "
                    "looks for natural speech (not judging content, just sync plausibility). "
                    "Reply with exactly two lines, no extra commentary:\nSCORE: <float between 0 and 1>\n"
                    "FEEDBACK: <one sentence critique>"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Line being spoken: {transcript!r}"},
                    {"type": "image_url", "image_url": {"url": fal_client.to_data_uri(early_frame, "image/jpeg")}},
                    {"type": "image_url", "image_url": {"url": fal_client.to_data_uri(late_frame, "image/jpeg")}},
                ],
            },
        ],
    )
    return _parse_score(response.choices[0].message.content or "")


def _generate_lipsync(run: Run, shot: ShotSpec, video_bytes: bytes, audio_bytes: bytes, transcript: str) -> LipsyncResult:
    provider = run.parameters.get("lipsync_provider", DEFAULT_LIPSYNC_PROVIDER)
    threshold = run.parameters.get("lipsync_confidence_threshold")
    max_attempts = _max_attempts(run)

    video_url = fal_client.upload_bytes(video_bytes, "video/mp4")
    audio_url = fal_client.upload_bytes(audio_bytes, "audio/mpeg")

    attempts: list[LipsyncResult] = []
    for attempt in range(max_attempts):
        media_url = fal_client.generate_lipsync(video_url, audio_url)
        synced_bytes = fal_client.download(media_url)

        score = _confidence_score(synced_bytes, shot.duration_sec, transcript) if threshold is not None else None
        status = "pass" if (threshold is None or (score is not None and score >= threshold)) else "retry"

        object_uri = minio_client.upload_bytes(f"lipsync/{shot.shot_id}_{attempt}.mp4", synced_bytes, "video/mp4")
        candidate = LipsyncResult(
            shot_id=shot.shot_id,
            lipsync_provider=provider,
            attempt_number=attempt,
            synced_video_uri=object_uri,
            sync_confidence_score=score,
            retry_count=attempt,
            status=status,
        )
        if status == "pass":
            return candidate
        attempts.append(candidate)

    result = select_attempt(run, attempts, key=lambda c: c.sync_confidence_score or 0)
    result.status = "flagged"
    return result


def _process_shot(run: Run, shot: ShotSpec) -> LipsyncResult | None:
    voice = latest_for(run.voice_generations, shot.shot_id)
    if voice is None:
        return None
    video = latest_for(run.video_generations, shot.shot_id)
    if video is None:
        return None

    video_bytes = fal_client.download(minio_client.presigned_url(video.video_uri))
    audio_bytes = fal_client.download(minio_client.presigned_url(voice.audio_uri))
    return _generate_lipsync(run, shot, video_bytes, audio_bytes, voice.stt_transcript)


def generate(run: Run) -> dict:
    """Runs the lip-sync gate for every shot in parallel."""
    if not run.shot_list:
        return {"status": "lipsync_complete"}

    with ThreadPoolExecutor(max_workers=min(4, len(run.shot_list))) as pool:
        results = list(pool.map(lambda shot: _process_shot(run, shot), run.shot_list))

    new_results = [r for r in results if r is not None]
    retries = sum(r.retry_count for r in new_results)

    return {
        "lipsync_results": [*run.lipsync_results, *new_results],
        "lipsync_retry_count": run.lipsync_retry_count + retries,
        "status": "lipsync_complete",
    }
