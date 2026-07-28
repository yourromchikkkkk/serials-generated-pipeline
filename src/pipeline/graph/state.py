"""Run record — the state object threaded through the graph."""

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ScriptRevision(BaseModel):
    """Record of one script-enhancer round."""

    run_id: str
    revision_number: int
    previous_text: str
    clarifying_questions: list[str]
    user_answers: list[str]
    revised_text: str
    revision_count: int


class TarantinoEvaluation(BaseModel):
    """Record of one Tarantino quality-gate scoring round."""

    run_id: str
    script_revision_id: str
    score: float
    feedback_text: str
    threshold_used: float
    decision: str
    tarantino_retry_count: int


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    script_text: str
    dialogue_lines: list[str] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    revision_count: int = 0
    script_revisions: list[ScriptRevision] = Field(default_factory=list)
    clarifying_questions: list[str] | None = None
    tarantino_evaluations: list[TarantinoEvaluation] = Field(default_factory=list)
    tarantino_retry_count: int = 0
    shot_validation_retry_count: int = 0
    per_asset_retry_count: int = 0
    lipsync_retry_count: int = 0
