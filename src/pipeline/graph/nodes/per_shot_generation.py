"""Per-shot generation stage — produces character/video/voice assets for every shot, each
independently quality-gated and retried up to `per_asset_retry_limit`. Gate thresholds
(`character_consistency_threshold`, `video_content_score_threshold`) default to null per the
run-config spec: unset means the gate records a score but never rejects. `verbatim_match_required`
defaults to true, so the voice gate is the one gate that meaningfully retries out of the box.

Character generation is memoized per `character_id` across the whole shot list (cached and reused
rather than regenerated per shot), guarded by a lock since shots are processed concurrently."""

import base64
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ffmpeg
from langsmith import traceable

from pipeline.clients import fal_client, minio_client
from pipeline.clients.litellm_client import chat_completion, transcribe
from pipeline.config import get_settings
from pipeline.graph.state import CharacterReference, Run, ShotSpec, VideoGeneration, VoiceGeneration

GENERATE = "per_shot_generate"

DEFAULT_VIDEO_PROVIDER = "kling"
DEFAULT_VOICE_PROVIDER = "elevenlabs"


def latest_for(records: list, shot_id: str):
    for record in reversed(records):
        if record.shot_id == shot_id:
            return record
    return None


def _max_attempts(run: Run) -> int:
    retry_limit = run.parameters.get("per_asset_retry_limit")
    return (retry_limit + 1) if retry_limit is not None else 1


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _parse_score(content: str) -> float:
    for line in content.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return 0.0
    return 0.0


@traceable(run_type="llm", name="per_shot_generation.vision_score")
def _vision_score(image_bytes: bytes, mime_type: str, description: str) -> float:
    response = chat_completion(
        model=get_settings().vision_gate_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Score 0-1 how well the image matches the description. Reply with exactly "
                    "two lines, no extra commentary:\nSCORE: <float between 0 and 1>\n"
                    "FEEDBACK: <one sentence critique>"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": description},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{_b64(image_bytes)}"}},
                ],
            },
        ],
    )
    return _parse_score(response.choices[0].message.content or "")


@traceable(run_type="llm", name="per_shot_generation.consistency_score")
def _consistency_score(reference_bytes: bytes, frame_bytes: bytes) -> float:
    response = chat_completion(
        model=get_settings().vision_gate_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "The first image is a character reference. Score 0-1 how consistent the "
                    "character in the second image is with the reference. Reply with exactly "
                    "two lines, no extra commentary:\nSCORE: <float between 0 and 1>\n"
                    "FEEDBACK: <one sentence critique>"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(reference_bytes)}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(frame_bytes)}"}},
                ],
            },
        ],
    )
    return _parse_score(response.choices[0].message.content or "")


def _require_ffmpeg_binary(binary: str) -> None:
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"'{binary}' not found on PATH — ffmpeg-python is just Python bindings, it shells "
            f"out to the ffmpeg CLI for the video tier1 gate. Install it, e.g. "
            f"`brew install ffmpeg` on macOS, then retry."
        )


def _probe(video_bytes: bytes) -> dict:
    _require_ffmpeg_binary("ffprobe")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        path.write_bytes(video_bytes)
        return ffmpeg.probe(str(path))


def extract_frame(video_bytes: bytes, at_sec: float) -> bytes:
    _require_ffmpeg_binary("ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "clip.mp4"
        dst = Path(tmp) / "frame.jpg"
        src.write_bytes(video_bytes)
        ffmpeg.input(str(src), ss=max(at_sec, 0)).output(str(dst), vframes=1).overwrite_output().run(
            quiet=True, capture_stdout=True, capture_stderr=True
        )
        return dst.read_bytes()


def _tier1_check(video_bytes: bytes) -> str:
    """Cheap technical checks with no model calls: corruption, duration, and (as a
    dependency-free proxy for real motion detection) more than a single frame."""
    try:
        probe = _probe(video_bytes)
    except ffmpeg.Error:
        return "fail_corrupted"

    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        return "fail_corrupted"

    duration = float(probe.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        return "fail_duration"

    nb_frames = video_streams[0].get("nb_frames")
    if nb_frames is not None and int(nb_frames) <= 1:
        return "fail_motion"

    return "pass"


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def select_attempt(run: Run, attempts: list, key):
    """Picks which non-passing attempt to ship once `per_asset_retry_limit` is exhausted, per
    `exit_strategy` (best_scoring_attempt, the default, vs last_attempt)."""
    if run.parameters.get("exit_strategy") == "last_attempt":
        return attempts[-1]
    return max(attempts, key=key)


def _generate_character(run: Run, shot: ShotSpec, character_id: str, cache: dict, lock: threading.Lock) -> CharacterReference:
    with lock:
        cached = cache.get(character_id)
        if cached is not None:
            return CharacterReference(
                shot_id=shot.shot_id,
                character_id=character_id,
                image_uri=cached.image_uri,
                seed=cached.seed,
                character_gate_score=cached.character_gate_score,
                cached=True,
            )

    prompt = f"Character reference portrait, id '{character_id}', for scene: {shot.scene_description}"
    threshold = run.parameters.get("character_consistency_threshold")
    max_attempts = _max_attempts(run)

    attempts: list[CharacterReference] = []
    for attempt in range(max_attempts):
        url, seed = fal_client.generate_character_image(prompt)
        image_bytes = fal_client.download(url)
        score = _vision_score(image_bytes, "image/png", prompt) if threshold is not None else None
        object_uri = minio_client.upload_bytes(
            f"characters/{character_id}/{shot.shot_id}_{attempt}.png", image_bytes, "image/png"
        )
        candidate = CharacterReference(
            shot_id=shot.shot_id,
            character_id=character_id,
            image_uri=object_uri,
            seed=seed,
            character_gate_score=score,
            cached=False,
        )
        if threshold is None or (score is not None and score >= threshold):
            with lock:
                cache.setdefault(character_id, candidate)
            return candidate
        attempts.append(candidate)

    result = select_attempt(run, attempts, key=lambda c: c.character_gate_score or 0)
    with lock:
        cache.setdefault(character_id, result)
    return result


def _generate_video(
    run: Run, shot: ShotSpec, image_bytes: bytes, character_reference_bytes: bytes | None
) -> VideoGeneration:
    """`image_bytes` is the starting frame handed to the image-to-video provider — always
    present, since the provider requires one on every call. `character_reference_bytes` is only
    set when that starting frame is a real cached character reference; for b-roll/establishing
    shots with no character, it's None so the consistency check (which would otherwise compare
    the output against an arbitrary generated scene image) is skipped rather than run on
    meaningless data."""
    provider = run.parameters.get("video_provider", DEFAULT_VIDEO_PROVIDER)
    content_threshold = run.parameters.get("video_content_score_threshold")
    consistency_threshold = run.parameters.get("character_consistency_threshold")
    max_attempts = _max_attempts(run)

    image_data_uri = fal_client.to_data_uri(image_bytes, "image/png")

    attempts: list[VideoGeneration] = []
    for attempt in range(max_attempts):
        media_url = fal_client.generate_video(image_data_uri, shot.scene_description, shot.duration_sec)
        video_bytes = fal_client.download(media_url)

        tier1_result = _tier1_check(video_bytes)
        tier2_score = None
        consistency_score = None
        if tier1_result == "pass" and (content_threshold is not None or consistency_threshold is not None):
            frame_bytes = extract_frame(video_bytes, shot.duration_sec / 2)
            if content_threshold is not None:
                tier2_score = _vision_score(frame_bytes, "image/jpeg", shot.scene_description)
            if consistency_threshold is not None and character_reference_bytes is not None:
                consistency_score = _consistency_score(character_reference_bytes, frame_bytes)

        status = "pass"
        if tier1_result != "pass":
            status = "retry"
        elif content_threshold is not None and (tier2_score or 0) < content_threshold:
            status = "retry"
        elif consistency_threshold is not None and (consistency_score or 0) < consistency_threshold:
            status = "retry"

        object_uri = minio_client.upload_bytes(f"videos/{shot.shot_id}_{attempt}.mp4", video_bytes, "video/mp4")
        candidate = VideoGeneration(
            shot_id=shot.shot_id,
            provider=provider,
            attempt_number=attempt,
            video_uri=object_uri,
            tier1_result=tier1_result,
            tier2_score=tier2_score,
            character_consistency_score=consistency_score,
            retry_count=attempt,
            status=status,
        )
        if status == "pass":
            return candidate
        attempts.append(candidate)

    result = select_attempt(run, attempts, key=lambda c: (c.tier2_score or 0) + (c.character_consistency_score or 0))
    result.status = "flagged"
    return result


def _generate_voice(run: Run, shot: ShotSpec) -> VoiceGeneration:
    provider = run.parameters.get("voice_provider", DEFAULT_VOICE_PROVIDER)
    verbatim_required = run.parameters.get("verbatim_match_required", True)
    max_attempts = _max_attempts(run)

    attempts: list[VoiceGeneration] = []
    for attempt in range(max_attempts):
        media_url = fal_client.generate_voice(shot.dialogue_line)
        audio_bytes = fal_client.download(media_url)
        transcript = transcribe(get_settings().stt_model, audio_bytes)
        verbatim = _normalize(transcript) == _normalize(shot.dialogue_line)
        status = "pass" if (verbatim or not verbatim_required) else "retry"

        object_uri = minio_client.upload_bytes(f"voice/{shot.shot_id}_{attempt}.mp3", audio_bytes, "audio/mpeg")
        candidate = VoiceGeneration(
            shot_id=shot.shot_id,
            provider=provider,
            voice_id=None,
            audio_uri=object_uri,
            source_text=shot.dialogue_line,
            stt_transcript=transcript,
            verbatim_match=verbatim,
            retry_count=attempt,
            status=status,
        )
        if status == "pass":
            return candidate
        attempts.append(candidate)

    result = select_attempt(run, attempts, key=lambda c: 1 if c.verbatim_match else 0)
    result.status = "flagged"
    return result


def _scene_image_bytes(shot: ShotSpec) -> bytes:
    """The video provider is image-to-video and requires a starting frame on every call. Shots
    with no character_refs (pure b-roll/establishing shots) have no cached reference to reuse, so
    generate a one-off scene image straight from the scene description instead."""
    url, _ = fal_client.generate_character_image(f"Establishing shot, no characters in frame: {shot.scene_description}")
    return fal_client.download(url)


def regenerate_character(run: Run, shot: ShotSpec) -> dict:
    """Targeted regeneration for shot review — regenerates every character reference this shot
    uses, without touching other shots that may share the same character_id."""
    if not shot.character_refs:
        return {}
    cache: dict = {}
    lock = threading.Lock()
    new_refs = [_generate_character(run, shot, cid, cache, lock) for cid in shot.character_refs]
    return {"character_references": [*run.character_references, *new_refs]}


def regenerate_video(run: Run, shot: ShotSpec) -> dict:
    character_ref = latest_for(run.character_references, shot.shot_id)
    if character_ref is not None:
        character_bytes = fal_client.download(minio_client.presigned_url(character_ref.image_uri))
        new_video = _generate_video(run, shot, character_bytes, character_bytes)
    else:
        new_video = _generate_video(run, shot, _scene_image_bytes(shot), None)
    return {"video_generations": [*run.video_generations, new_video]}


def regenerate_voice(run: Run, shot: ShotSpec) -> dict:
    if not shot.dialogue_line:
        return {}
    new_voice = _generate_voice(run, shot)
    return {"voice_generations": [*run.voice_generations, new_voice]}


def _process_shot(run: Run, shot: ShotSpec, cache: dict, lock: threading.Lock) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        voice_future = pool.submit(_generate_voice, run, shot) if shot.dialogue_line else None

        character_refs = [_generate_character(run, shot, cid, cache, lock) for cid in shot.character_refs]
        if character_refs:
            character_bytes = fal_client.download(minio_client.presigned_url(character_refs[0].image_uri))
            video = _generate_video(run, shot, character_bytes, character_bytes)
        else:
            video = _generate_video(run, shot, _scene_image_bytes(shot), None)

        voice = voice_future.result() if voice_future else None

    retries = video.retry_count
    if voice is not None:
        retries += voice.retry_count

    return {
        "character_references": character_refs,
        "video_generations": [video],
        "voice_generations": [voice] if voice is not None else [],
        "retries": retries,
    }


def generate(run: Run) -> dict:
    """Produces character/video/voice assets for every shot in parallel. Character generation is
    memoized across shots via a shared cache; video and voice generation run concurrently within
    each shot's worker since voice has no dependency on the character/video pipeline."""
    if not run.shot_list:
        return {"status": "per_shot_generation_complete"}

    cache: dict = {ref.character_id: ref for ref in run.character_references if not ref.cached}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=min(4, len(run.shot_list))) as pool:
        results = list(pool.map(lambda shot: _process_shot(run, shot, cache, lock), run.shot_list))

    character_references = list(run.character_references)
    video_generations = list(run.video_generations)
    voice_generations = list(run.voice_generations)
    retries = 0
    for result in results:
        character_references.extend(result["character_references"])
        video_generations.extend(result["video_generations"])
        voice_generations.extend(result["voice_generations"])
        retries += result["retries"]

    return {
        "character_references": character_references,
        "video_generations": video_generations,
        "voice_generations": voice_generations,
        "per_asset_retry_count": run.per_asset_retry_count + retries,
        "status": "per_shot_generation_complete",
    }
