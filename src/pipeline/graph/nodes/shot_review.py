"""Shot review stage — optional second HITL checkpoint over generated video/audio per shot,
before they lock in for assembly. Unlike shot validation, a reviewer can reject a single asset
(character, video, or voice) and only that asset is regenerated — an approved video isn't
re-rolled because the voice was wrong."""

from langgraph.graph import END
from langgraph.types import interrupt

from pipeline.clients import minio_client
from pipeline.graph.nodes import per_shot_generation
from pipeline.graph.state import Run, ShotReview, ShotSpec

AWAIT_REVIEW = "shot_review_await_review"

_REGENERATORS = {
    "character": per_shot_generation.regenerate_character,
    "video": per_shot_generation.regenerate_video,
    "voice": per_shot_generation.regenerate_voice,
}


def _next_shot_to_review(run: Run) -> ShotSpec | None:
    latest_decision: dict[str, str] = {}
    for review in run.shot_reviews:
        latest_decision[review.shot_id] = review.decision
    for shot in run.shot_list:
        if latest_decision.get(shot.shot_id) != "approve":
            return shot
    return None


def _signed_url(uri: str | None) -> str | None:
    return minio_client.presigned_url(uri) if uri else None


def await_review(run: Run) -> dict:
    """Gated by `shot_review` (off by default). Loops over shots not yet approved, presenting
    signed MinIO URLs for the reviewer to open directly, rather than raw storage URIs."""
    if not run.parameters.get("shot_review", False):
        return {"status": "shot_review_skipped"}

    shot = _next_shot_to_review(run)
    if shot is None:
        return {"status": "shot_review_complete"}

    character = per_shot_generation.latest_for(run.character_references, shot.shot_id)
    video = per_shot_generation.latest_for(run.video_generations, shot.shot_id)
    voice = per_shot_generation.latest_for(run.voice_generations, shot.shot_id)

    payload = {
        "shot_id": shot.shot_id,
        "scene_description": shot.scene_description,
        "character_url": _signed_url(character.image_uri if character else None),
        "video_url": _signed_url(video.video_uri if video else None),
        "voice_url": _signed_url(voice.audio_uri if voice else None),
    }
    decision = interrupt(payload)
    if not isinstance(decision, dict) or decision.get("decision") not in ("approve", "reject"):
        raise ValueError("shot review expected {'decision': 'approve' | 'reject', ...}")

    reviewed_by = decision.get("reviewed_by")

    if decision["decision"] == "approve":
        review = ShotReview(shot_id=shot.shot_id, reviewed_by=reviewed_by, decision="approve", rejected_asset=None)
        return {"shot_reviews": [*run.shot_reviews, review], "status": "shot_review_in_progress"}

    rejected_asset = decision.get("rejected_asset")
    regenerate = _REGENERATORS.get(rejected_asset)
    if regenerate is None:
        raise ValueError("shot review 'reject' decision requires 'rejected_asset' in character|video|voice")

    review = ShotReview(shot_id=shot.shot_id, reviewed_by=reviewed_by, decision="reject", rejected_asset=rejected_asset)
    updates = regenerate(run, shot)
    updates["shot_reviews"] = [*run.shot_reviews, review]
    updates["status"] = "shot_review_in_progress"
    return updates


def route_after_await_review(run: Run) -> str:
    if run.status == "shot_review_in_progress":
        return AWAIT_REVIEW
    return END
