"""Shot validation stage — the first mandatory human checkpoint. Approves the production plan
before any expensive image/video/voice generation begins; unlike `shot_review` and `final_review`
downstream, this one cannot be disabled."""

from langgraph.graph import END
from langgraph.types import interrupt

from pipeline.graph.state import Run, ShotSpec, ShotValidation

AWAIT_APPROVAL = "shot_validation_await_approval"


def _apply_edits(run: Run, edits: list[dict]) -> list[ShotSpec]:
    return [
        ShotSpec(
            run_id=run.run_id,
            order_index=index,
            scene_description=edit["scene_description"],
            camera=edit["camera"],
            dialogue_line=edit["dialogue_line"],
            duration_sec=edit["duration_sec"],
            character_refs=edit.get("character_refs", []),
        )
        for index, edit in enumerate(edits)
    ]


def await_approval(run: Run) -> dict:
    """Bounded by `shot_validation_retry_limit`; once exhausted, proceeds with the latest shot
    list instead of blocking the run permanently."""
    retry_limit = run.parameters.get("shot_validation_retry_limit")
    if retry_limit is not None and run.shot_validation_retry_count >= retry_limit:
        return {"status": "shot_validation_limit_reached"}

    decision = interrupt({"shot_list": [shot.model_dump(mode="json") for shot in run.shot_list]})
    if not isinstance(decision, dict) or decision.get("decision") not in ("approve", "edit"):
        raise ValueError("shot validation expected {'decision': 'approve' | 'edit', ...}")

    version = len(run.shot_validations)
    approved_by = decision.get("approved_by")

    if decision["decision"] == "approve":
        validation = ShotValidation(
            run_id=run.run_id,
            shot_list_version=version,
            approved_by=approved_by,
            edits=None,
            shot_validation_retry_count=run.shot_validation_retry_count,
        )
        return {
            "shot_validations": [*run.shot_validations, validation],
            "status": "shot_validation_approved",
        }

    edits = decision.get("edits")
    if not isinstance(edits, list) or not edits:
        raise ValueError("shot validation 'edit' decision requires a non-empty 'edits' shot list")

    revised_shot_list = _apply_edits(run, edits)
    validation = ShotValidation(
        run_id=run.run_id,
        shot_list_version=version,
        approved_by=approved_by,
        edits=edits,
        shot_validation_retry_count=run.shot_validation_retry_count,
    )
    return {
        "shot_list": revised_shot_list,
        "shot_validations": [*run.shot_validations, validation],
        "shot_validation_retry_count": run.shot_validation_retry_count + 1,
        "status": "shot_validation_edited",
    }


def route_after_await_approval(run: Run) -> str:
    if run.status == "shot_validation_edited":
        return AWAIT_APPROVAL
    return END
