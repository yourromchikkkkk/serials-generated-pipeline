"""Single point of truth for env loading and secrets. Every client reads config from here."""

import os
import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LiteLLM proxy
    litellm_api_base: str = "http://localhost:4000"
    litellm_master_key: str | None = None

    # LLM provider keys (consumed by the LiteLLM proxy)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # LLM models per pipeline stage (routed through the LiteLLM proxy, see config/litellm/config.yaml)
    script_enhancer_model: str = "anthropic/claude-haiku"
    tarantino_model: str = "anthropic/claude-haiku"
    shot_list_generation_model: str = "anthropic/claude-haiku"
    vision_gate_model: str = "openai/gpt-4o-mini"
    stt_model: str = "gpt-4o-mini-transcribe"

    # fal.ai
    fal_key: str | None = None
    fal_character_model: str = "fal-ai/flux/dev"
    fal_video_model: str = "fal-ai/kling-video/v1.6/standard/image-to-video"
    fal_voice_model: str = "fal-ai/elevenlabs/tts/turbo-v2.5"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "serials-generated-pipeline"
    langsmith_endpoint: str | None = None

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MongoDB
    mongo_url: str = "mongodb://admin:changeme@localhost:27017"
    mongo_database: str = "pipeline"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "pipeline-artifacts"
    minio_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_langsmith() -> None:
    """Export LangSmith env vars so langsmith/langchain-core (which read os.environ directly,
    not our Settings object) actually pick them up for tracing our own graph runs."""
    settings = get_settings()
    if not settings.langsmith_tracing:
        return
    if not settings.langsmith_api_key:
        print("warning: LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set, skipping trace export", file=sys.stderr)
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
