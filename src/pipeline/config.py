"""Single point of truth for env loading and secrets. Every client reads config from here."""

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

    # fal.ai
    fal_key: str | None = None

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "serials-generated-pipeline"

    # Redis
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
