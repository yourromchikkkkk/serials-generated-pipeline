"""Shot-list generation stage — translates the approved script into a structured,
machine-actionable production plan: one ShotSpec per shot."""

from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError

from pipeline.clients.litellm_client import chat_completion
from pipeline.config import get_settings
from pipeline.graph.state import Run, ShotSpec

GENERATE = "shot_list_generate"

MAX_GENERATION_ATTEMPTS = 3

_SYSTEM_PROMPT = (
    "You are a shot-list planner for short-form vertical video. Break the given script into "
    "an ordered list of shots, each with a scene description, camera direction, the exact "
    "verbatim dialogue line spoken in that shot (empty string if none), a target duration in "
    "seconds, and any character ids required in the shot."
)


class _ShotDraft(BaseModel):
    """One shot as returned by the LLM, before run_id/shot_id/order_index are attached."""

    scene_description: str
    camera: str
    dialogue_line: str
    duration_sec: float
    character_refs: list[str] = Field(default_factory=list)


class _ShotListResponse(BaseModel):
    """Strict schema passed as `response_format` so the provider enforces it at generation time
    (OpenAI structured outputs / strict JSON-schema mode) instead of us hand-parsing free text."""

    shots: list[_ShotDraft]


@traceable(run_type="llm", name="shot_list_generation.generate")
def _generate_shots(script_text: str, num_shots: int | None, target_duration_sec: int | None) -> list[_ShotDraft]:
    constraints = []
    if num_shots is not None:
        constraints.append(f"Produce exactly {num_shots} shots.")
    if target_duration_sec is not None:
        constraints.append(f"Shot durations must sum to approximately {target_duration_sec} seconds total.")

    user_content = script_text
    if constraints:
        user_content = f"{script_text}\n\nConstraints:\n" + "\n".join(constraints)

    response = chat_completion(
        model=get_settings().shot_list_generation_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=_ShotListResponse,
    )
    content = response.choices[0].message.content or ""
    return _ShotListResponse.model_validate_json(content).shots


def generate(run: Run) -> dict:
    """Produces the ordered ShotSpec list for the episode. A schema-validation failure (including
    invalid JSON) is cheap to fix with an automatic re-prompt to the same LLM call — no human
    needed, unlike the mandatory shot-validation checkpoint that follows this stage."""
    num_shots = run.parameters.get("num_shots")
    target_duration_sec = run.parameters.get("target_duration_sec")

    last_error: Exception | None = None
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            drafts = _generate_shots(run.script_text, num_shots, target_duration_sec)
            if not drafts:
                raise ValueError("shot list must be non-empty")
            shot_list = [
                ShotSpec(
                    run_id=run.run_id,
                    order_index=index,
                    scene_description=draft.scene_description,
                    camera=draft.camera,
                    dialogue_line=draft.dialogue_line,
                    duration_sec=draft.duration_sec,
                    character_refs=draft.character_refs,
                )
                for index, draft in enumerate(drafts)
            ]
        except (ValidationError, ValueError) as exc:
            last_error = exc
            continue
        return {"shot_list": shot_list, "status": "shot_list_generated"}

    raise ValueError(f"shot-list generation failed after {MAX_GENERATION_ATTEMPTS} attempts") from last_error
