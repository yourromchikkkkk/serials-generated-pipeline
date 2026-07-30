# serials-generated-pipeline

## Stack

LangGraph (orchestration) · LiteLLM (LLM routing) · fal.ai (image/video/voice/lip-sync
generation) · FFmpeg (assembly) · MongoDB (result storage) · MinIO (object storage).

## Setup

```bash
brew install ffmpeg          # ffmpeg-python is just bindings — the ffmpeg/ffprobe CLI binaries
                              # must be on PATH for the video gate and (later) assembly
uv sync                      # installs deps + dev tools into .venv
cp .env.example .env         # fill in FAL_KEY, OPENAI_API_KEY / ANTHROPIC_API_KEY, etc.
docker compose up -d         # starts litellm (:4000), mongodb (:27017), minio (:9000/:9001)
```

## Usage

```bash
uv run pipeline run "A detective finds a note that isn't from who she thinks." --shot-review
```

The run pauses at each human-in-the-loop checkpoint — clarifying questions, a Tarantino score
below threshold (or retries exhausted), shot-list approval, and (if `--shot-review` is set) each
shot's generated assets — and prompts interactively in the terminal.

### `pipeline run` options

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `script` (positional) | string | — | 1-3 sentence idea, or full script text. Mutually exclusive with `--file` |
| `--file`, `-f` | path | — | Read the script text from a file instead of the command line |
| `--enhance-script` | flag | off | Enable the optional script enhancer stage (asks clarifying questions before proceeding) |
| `--revision-limit` | int | unlimited | Max script enhancer rounds before proceeding with the latest version |
| `--tarantino-threshold` | float (0-1) | `0.7` | Minimum Tarantino quality-gate score required to pass |
| `--tarantino-retry-limit` | int | unlimited | Max Tarantino evaluation rounds before the run stops and hands the decision to you, instead of auto-shipping below threshold |
| `--enable-auto-rewriter` | flag | off | Auto-rewrite the script on a failed Tarantino gate instead of asking you first |
| `--skip-tarantino-evaluation` | flag | off | Skip the Tarantino quality-gate stage entirely and proceed straight to shot-list generation |
| `--per-asset-retry-limit` | int | 0 (single attempt) | Max retries per character/video/voice asset before flagging it for review |
| `--exit-strategy` | `best_scoring_attempt` \| `last_attempt` | `best_scoring_attempt` | Which attempt to keep once `--per-asset-retry-limit` is exhausted |
| `--character-consistency-threshold` | float (0-1) | unset (gate records score only, never rejects) | Cutoff for the character gate and the video character-consistency check |
| `--video-content-threshold` | float (0-1) | unset (gate records score only, never rejects) | Vision-model prompt-adherence cutoff for the video content gate |
| `--shot-review` | flag | off | Enable the optional shot review stage — inspect each shot's generated video/audio (via signed MinIO URLs) before assembly, and reject a single asset for targeted regeneration |
| `--lipsync-retry-limit` | int | 0 (single attempt) | Max retries for the lip-sync gate per shot before flagging it for review |
| `--lipsync-confidence-threshold` | float (0-1) | unset (gate records score only, never rejects) | Sync-confidence cutoff for the lip-sync gate |

Run `uv run pipeline run --help` for the same list from the CLI itself.

Note: the voice gate's verbatim-match requirement is always on (`verbatim_match_required` has no
CLI flag) — the spoken line matching the source dialogue word-for-word is a hard requirement of
this pipeline, not a configurable quality target.

## Project layout

```
config/litellm/           LiteLLM proxy model config
src/pipeline/clients/     LiteLLM / fal.ai / MinIO / MongoDB client factories
src/pipeline/graph/       LangGraph state (Run) and graph wiring (base.py)
src/pipeline/graph/nodes/ One module per pipeline stage
src/pipeline/cli.py       CLI entry point
docker-compose.yml        LiteLLM + MongoDB + MinIO containers
```

## Compromises

Pragmatic tradeoffs made to get an end-to-end pipeline working.

- **Vision-LLM gates instead of specialized models.** Video content-match, character-consistency,
  and lip-sync-confidence checks all ask a general vision model (`vision_gate_model`) for a 0-1
  score, rather than using embedding-similarity or a purpose-built lip-sync-detection model. Good
  enough to gate on, cheaper to build than integrating a dedicated model per check.
- **In-memory graph checkpointing.** `MemorySaver` gets human-in-the-loop interrupts working
  correctly within one process, but a run doesn't survive a crash or restart — there's no way to
  resume a `run_id` from a different invocation yet.
- **Thread pool, not a distributed queue.** Per-shot parallelism is a local `ThreadPoolExecutor`
  (max 4 workers), not the Redis/queue-based worker pool the original design sketch considered.
  Redis was removed from the stack rather than left half-wired for a capability nothing uses.
- **One reference image per shot, even with multiple characters.** Every character in a shot gets
  a generated/cached reference, but only the first is actually passed to the image-to-video call —
  simplest way to satisfy an image-to-video provider's single-image input, at the cost of
  consistency for every character after the first.
- **Characterless shots get an uncached, one-off scene image.** Shots with no named character
  generate a throwaway establishing image from the scene description to satisfy the video
  provider's image-to-video requirement, instead of a properly cached, reviewable asset type.
- **Exact-match verbatim checking.** The voice gate normalizes only case and whitespace before
  comparing the transcript to source dialogue — strict on purpose, since verbatim delivery is a
  hard requirement, but it can flag technically-correct audio as a retry over STT quirks (e.g.
  "5" vs. "five").
- **No retry backoff.** Generation retries fire immediately up to the configured limit, with no
  exponential backoff or special handling for rate-limit responses — simplest way to get the
  retry-until-limit loop working first.
