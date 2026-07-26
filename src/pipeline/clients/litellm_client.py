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
