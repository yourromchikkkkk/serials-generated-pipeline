"""fal.ai client factory + thin generation wrappers (character/video/voice generation).

The exact argument/response shape of each fal.ai app varies per model — the wrappers below
assume the conventional shapes documented for FAL_CHARACTER_MODEL (flux-style text-to-image:
`images: [{url, seed}]`), FAL_VIDEO_MODEL (image-to-video: `video: {url}`), and FAL_VOICE_MODEL
(TTS: `audio: {url}`). Verify against the fal.ai catalog entry for whichever model id is
configured in .env if it's swapped for a different provider."""

import base64
from functools import lru_cache
from urllib.request import urlopen

import fal_client

from pipeline.config import get_settings

# fal.ai's kling-video image-to-video app only accepts these two literal duration values (as
# strings, not numbers) — see the 422 this snaps around. Specific to the default FAL_VIDEO_MODEL;
# revisit if that's swapped for a provider with different duration constraints.
_KLING_ALLOWED_DURATIONS_SEC = (5, 10)


@lru_cache
def get_fal_client() -> fal_client.SyncClient:
    return fal_client.SyncClient(key=get_settings().fal_key)


def download(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def to_data_uri(data: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def generate_character_image(prompt: str, seed: int | None = None) -> tuple[str, int]:
    """Returns (media_url, seed_used)."""
    arguments: dict = {"prompt": prompt}
    if seed is not None:
        arguments["seed"] = seed
    result = get_fal_client().subscribe(get_settings().fal_character_model, arguments=arguments)
    image = result["images"][0]
    return image["url"], image.get("seed", seed or 0)


def generate_video(image_data_uri: str, prompt: str, duration_sec: float) -> str:
    """Returns the generated clip's media url. `image_data_uri` must be a `data:` URI, not a
    plain URL — fal.ai's servers can't reach a locally-hosted MinIO URL, and require HTTPS or a
    Data URI regardless."""
    nearest = min(_KLING_ALLOWED_DURATIONS_SEC, key=lambda d: abs(d - duration_sec))
    result = get_fal_client().subscribe(
        get_settings().fal_video_model,
        arguments={"image_url": image_data_uri, "prompt": prompt, "duration": str(nearest)},
    )
    return result["video"]["url"]


def generate_voice(text: str, voice_id: str | None = None) -> str:
    """Returns the generated audio's media url."""
    arguments: dict = {"text": text}
    if voice_id:
        arguments["voice"] = voice_id
    result = get_fal_client().subscribe(get_settings().fal_voice_model, arguments=arguments)
    return result["audio"]["url"]
