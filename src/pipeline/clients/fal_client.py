"""fal.ai client factory + thin generation wrappers (character/video/voice/lip-sync generation).

The exact argument/response shape of each fal.ai app varies per model — the wrappers below
assume the conventional shapes documented for FAL_CHARACTER_MODEL (flux-style text-to-image:
`images: [{url, seed}]`), FAL_VIDEO_MODEL (image-to-video: `video: {url}`), FAL_VOICE_MODEL
(TTS: `audio: {url}`), and FAL_LIPSYNC_MODEL (video+audio merge: `video: {url}`). Verify against
the fal.ai catalog entry for whichever model id is configured in .env if it's swapped for a
different provider."""

import base64
from functools import lru_cache
from typing import Any
from urllib.request import urlopen

import fal_client
from fal_client.client import FalClientHTTPError

from pipeline.config import get_settings

# fal.ai's kling-video image-to-video app only accepts these two literal duration values (as
# strings, not numbers) — see the 422 this snaps around. Specific to the default FAL_VIDEO_MODEL;
# revisit if that's swapped for a provider with different duration constraints.
_KLING_ALLOWED_DURATIONS_SEC = (5, 10)

_TRUNCATE_AFTER = 200


@lru_cache
def get_fal_client() -> fal_client.SyncClient:
    return fal_client.SyncClient(key=get_settings().fal_key)


def _truncate_long_strings(value: Any) -> Any:
    """fal.ai's 422 responses echo the full request body back in `detail` — including any
    inline `data:` URI we sent, which can be megabytes of base64. Recursively truncates long
    strings so the actual validation error (type/loc/msg) stays legible instead of being buried
    under a wall of base64."""
    if isinstance(value, str):
        if len(value) > _TRUNCATE_AFTER:
            return f"{value[:_TRUNCATE_AFTER]}...<{len(value) - _TRUNCATE_AFTER} more chars truncated>"
        return value
    if isinstance(value, dict):
        return {k: _truncate_long_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_long_strings(v) for v in value]
    return value


def _subscribe(model: str, arguments: dict) -> dict:
    try:
        return get_fal_client().subscribe(model, arguments=arguments)
    except FalClientHTTPError as exc:
        raise RuntimeError(
            f"fal.ai request to {model!r} failed ({exc.status_code}): {_truncate_long_strings(exc.message)}"
        ) from exc


def download(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def to_data_uri(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def upload_bytes(data: bytes, content_type: str) -> str:
    """Uploads to fal's storage and returns a real HTTPS url. Required for video/audio inputs —
    fal.ai enforces a request-size limit on inline `data:` URIs that a full video clip blows past
    (the request gets rejected with a 422 that echoes the oversized body back). Small images
    (a character reference, a vision-gate frame) stay under that limit and are fine as data URIs."""
    return get_fal_client().upload(data, content_type)


def generate_character_image(prompt: str, seed: int | None = None) -> tuple[str, int]:
    """Returns (media_url, seed_used)."""
    arguments: dict = {"prompt": prompt}
    if seed is not None:
        arguments["seed"] = seed
    result = _subscribe(get_settings().fal_character_model, arguments)
    image = result["images"][0]
    return image["url"], image.get("seed", seed or 0)


def generate_video(image_data_uri: str, prompt: str, duration_sec: float) -> str:
    """Returns the generated clip's media url. `image_data_uri` must be a `data:` URI, not a
    plain URL — fal.ai's servers can't reach a locally-hosted MinIO URL, and require HTTPS or a
    Data URI regardless."""
    nearest = min(_KLING_ALLOWED_DURATIONS_SEC, key=lambda d: abs(d - duration_sec))
    result = _subscribe(
        get_settings().fal_video_model,
        {"image_url": image_data_uri, "prompt": prompt, "duration": str(nearest)},
    )
    return result["video"]["url"]


def generate_voice(text: str, voice_id: str | None = None) -> str:
    """Returns the generated audio's media url."""
    arguments: dict = {"text": text}
    if voice_id:
        arguments["voice"] = voice_id
    result = _subscribe(get_settings().fal_voice_model, arguments)
    return result["audio"]["url"]


def generate_lipsync(video_url: str, audio_url: str) -> str:
    """Merges video and audio into a lip-synced clip; returns the result's media url. Both
    inputs must be URLs fal.ai's servers can reach — a local MinIO url won't work (see
    `generate_video`'s note), and inline `data:` URIs won't work either at this size (see
    `upload_bytes`). Upload the raw bytes to fal's storage first and pass the resulting url."""
    result = _subscribe(get_settings().fal_lipsync_model, {"video_url": video_url, "audio_url": audio_url})
    return result["video"]["url"]
