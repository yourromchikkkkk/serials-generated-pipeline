# serials-generated-pipeline

## Stack

LangGraph (orchestration) · LiteLLM (LLM routing) · fal.ai (image/video/voice/lip-sync
generation) · FFmpeg (assembly) · Redis (per-shot parallelism) · MongoDB (result storage) ·
MinIO (object storage).

## Setup

```bash
uv sync                      # installs deps + dev tools into .venv
cp .env.example .env         # fill in FAL_KEY, OPENAI_API_KEY / ANTHROPIC_API_KEY, etc.
docker compose up -d         # starts litellm (:4000), redis (:6379), mongodb (:27017), minio (:9000/:9001)
```

## Usage

```bash
uv run pipeline --help
```

## Project layout

```
config/litellm/   LiteLLM proxy model config
src/pipeline/      Application package (config, clients, CLI)
docker-compose.yml LiteLLM + Redis + MongoDB + MinIO containers
```

## Status

Infra only so far: config loading, LiteLLM/Redis/fal.ai clients, and a CLI skeleton. No
pipeline stages are implemented yet.
