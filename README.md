# serials-generated-pipeline

## Stack

LangGraph (orchestration) · LiteLLM (LLM routing) · fal.ai (image/video/voice/lip-sync
generation) · FFmpeg (assembly) · Redis (per-shot parallelism).

## Setup

```bash
uv sync                      # installs deps + dev tools into .venv
cp .env.example .env         # fill in FAL_KEY, OPENAI_API_KEY / ANTHROPIC_API_KEY, etc.
docker compose up -d         # starts the litellm proxy (:4000) and redis (:6379)
```

## Usage

```bash
uv run pipeline --help
```

## Project layout

```
config/litellm/   LiteLLM proxy model config
src/pipeline/      Application package (config, clients, CLI)
docker-compose.yml LiteLLM + Redis containers
```

## Status

Infra only so far: config loading, LiteLLM/Redis/fal.ai clients, and a CLI skeleton. No
pipeline stages are implemented yet.
