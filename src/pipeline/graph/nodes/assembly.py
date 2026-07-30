"""Assembly stage — the final step of the pipeline (CLAUDE.md 2.9). Stitches every shot's synced
clip into one deliverable, in shot order: normalizes codec/format/frame rate across shots (video
providers and lip-sync attempts can return different specs) and trims/pads to hit
`target_duration_sec`. Deterministic, FFmpeg-only — no model calls, no quality gate, retried
automatically on the environment/format failures that are the only way this step fails.

Shots with no dialogue never went through the lip-sync gate (see nodes/lipsync.py), so their
plain generated video is used instead; it's normalized the same way, with a silent audio track
added so every clip has matching video+audio streams before concatenation.

The stitched file is uploaded to MinIO and its `minio://` uri is what the CLI resolves to a
presigned URL and prints as the final deliverable."""

import shutil
import tempfile
from pathlib import Path

import ffmpeg

from pipeline.clients import fal_client, minio_client
from pipeline.graph.nodes.per_shot_generation import latest_for
from pipeline.graph.state import AssemblyResult, Run

ASSEMBLE = "assembly"

PIPELINE_VERSION = "0.1.0"
OUTPUT_FORMAT = "mp4"
CODEC = "libx264"
FRAME_RATE = 30
_ASPECT_RATIO_DIMENSIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}
_DEFAULT_ASPECT_RATIO = "9:16"


def _run_ffmpeg(output_stream) -> None:
    output_stream.overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)


def _has_audio(path: Path) -> bool:
    probe = ffmpeg.probe(str(path))
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _clip_duration(path: Path) -> float:
    probe = ffmpeg.probe(str(path))
    return float(probe["format"]["duration"])


def _normalize_clip(src: Path, dst: Path, width: int, height: int) -> None:
    """Scales/pads every clip to a common resolution and frame rate, re-encodes to a single
    codec, and guarantees an audio stream (adding silence for dialogue-less shots) so the
    concat step below never trips on mismatched specs between clips."""
    inp = ffmpeg.input(str(src))
    video = inp.video.filter("scale", width, height, force_original_aspect_ratio="decrease").filter(
        "pad", width, height, "(ow-iw)/2", "(oh-ih)/2"
    ).filter("fps", fps=FRAME_RATE)

    if _has_audio(src):
        audio = inp.audio
    else:
        duration = _clip_duration(src)
        audio = ffmpeg.input("anullsrc=channel_layout=stereo:sample_rate=44100", f="lavfi", t=duration).audio

    _run_ffmpeg(ffmpeg.output(video, audio, str(dst), vcodec=CODEC, acodec="aac", pix_fmt="yuv420p"))


def _concat_clips(normalized_paths: list[Path], concat_list_path: Path, dst: Path) -> None:
    concat_list_path.write_text("".join(f"file '{p}'\n" for p in normalized_paths))
    _run_ffmpeg(ffmpeg.input(str(concat_list_path), format="concat", safe=0).output(str(dst), c="copy"))


def _fit_duration(src: Path, dst: Path, target_duration_sec: float) -> None:
    """Trims (or pads, holding the last frame and extending silence) the assembled video to hit
    `target_duration_sec`, per the run config's per-episode duration target."""
    current_duration = _clip_duration(src)

    if current_duration > target_duration_sec:
        _run_ffmpeg(ffmpeg.input(str(src), t=target_duration_sec).output(str(dst), c="copy"))
        return

    pad_sec = target_duration_sec - current_duration
    if pad_sec <= 0.05:
        shutil.copyfile(src, dst)
        return

    inp = ffmpeg.input(str(src))
    video = inp.video.filter("tpad", stop_mode="clone", stop_duration=pad_sec)
    audio = inp.audio.filter("apad", pad_dur=pad_sec)
    _run_ffmpeg(ffmpeg.output(video, audio, str(dst), vcodec=CODEC, acodec="aac", pix_fmt="yuv420p"))


def _shot_clip_uri(run: Run, shot_id: str) -> str | None:
    lipsync_result = latest_for(run.lipsync_results, shot_id)
    if lipsync_result is not None:
        return lipsync_result.synced_video_uri
    video = latest_for(run.video_generations, shot_id)
    return video.video_uri if video is not None else None


def assemble(run: Run) -> dict:
    """Concatenates every shot's synced clip, in shot order, into the final deliverable and
    uploads it to MinIO."""
    if not run.shot_list:
        return {"status": "assembly_complete"}

    shot_ids = [shot.shot_id for shot in run.shot_list]
    clip_uris = [(shot_id, _shot_clip_uri(run, shot_id)) for shot_id in shot_ids]
    missing = [shot_id for shot_id, uri in clip_uris if uri is None]
    if missing:
        raise ValueError(f"assembly: no generated clip found for shot(s): {missing}")

    width, height = _ASPECT_RATIO_DIMENSIONS.get(
        run.parameters.get("aspect_ratio", _DEFAULT_ASPECT_RATIO), _ASPECT_RATIO_DIMENSIONS[_DEFAULT_ASPECT_RATIO]
    )
    # Only forces a duration when the run explicitly asked for one — actual generated clip
    # lengths (video snapped to the provider's allowed durations, voice length driven by the
    # dialogue text) routinely miss the planned per-shot durations, so falling back to the sum
    # of ShotSpec.duration_sec here padded the output with a long frozen last frame.
    target_duration_sec = run.parameters.get("target_duration_sec")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        normalized_paths = []
        for i, (shot_id, uri) in enumerate(clip_uris):
            clip_bytes = fal_client.download(minio_client.presigned_url(uri))
            src = tmp_path / f"src_{i}.mp4"
            src.write_bytes(clip_bytes)
            normalized = tmp_path / f"norm_{i}.mp4"
            _normalize_clip(src, normalized, width, height)
            normalized_paths.append(normalized)

        concatenated = tmp_path / "concatenated.mp4"
        _concat_clips(normalized_paths, tmp_path / "concat_list.txt", concatenated)

        if target_duration_sec is not None:
            final_path = tmp_path / "final.mp4"
            _fit_duration(concatenated, final_path, target_duration_sec)
        else:
            final_path = concatenated

        final_bytes = final_path.read_bytes()
        final_duration = _clip_duration(final_path)

    object_uri = minio_client.upload_bytes(f"final/{run.run_id}.{OUTPUT_FORMAT}", final_bytes, "video/mp4")
    result = AssemblyResult(
        run_id=run.run_id,
        video_uri=object_uri,
        shot_ids=shot_ids,
        duration_sec=final_duration,
        format=OUTPUT_FORMAT,
        codec=CODEC,
        pipeline_version=PIPELINE_VERSION,
    )

    return {"assembly_result": result, "status": "assembly_complete"}
