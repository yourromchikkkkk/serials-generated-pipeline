"""Run record — the state object threaded through the graph. See CLAUDE.md section 2.1/3."""

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Run(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    script_text: str
    dialogue_lines: list[str] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "created"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    revision_count: int = 0
    tarantino_retry_count: int = 0
    shot_validation_retry_count: int = 0
    per_asset_retry_count: int = 0
    lipsync_retry_count: int = 0
