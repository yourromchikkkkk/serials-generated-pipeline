"""Chat-completion client for the LiteLLM proxy in docker-compose.yml."""

from typing import Any

import litellm

from pipeline.config import get_settings


def chat_completion(model: str, messages: list[dict[str, str]], **kwargs: Any):
    """Call a model configured in config/litellm/config.yaml through the proxy."""
    settings = get_settings()
    return litellm.completion(
        model=model,
        messages=messages,
        api_base=settings.litellm_api_base,
        api_key=settings.litellm_master_key,
        **kwargs,
    )


def transcribe(model: str, audio_bytes: bytes, filename: str = "audio.mp3") -> str:
    """Speech-to-text on generated audio, used by the voice gate to verify verbatim delivery."""
    settings = get_settings()
    response = litellm.transcription(
        model=model,
        file=(filename, audio_bytes),
        api_base=settings.litellm_api_base,
        api_key=settings.litellm_master_key,
    )
    return response.get("text", "") if isinstance(response, dict) else response.text
