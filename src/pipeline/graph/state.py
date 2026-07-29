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


class ShotSpec(BaseModel):
    """One shot in the production plan produced by shot-list generation."""

    shot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    order_index: int
    scene_description: str
    camera: str
    dialogue_line: str
    duration_sec: float
    character_refs: list[str] = Field(default_factory=list)


class ShotValidation(BaseModel):
    """Record of one shot-list approval round — the first mandatory HITL checkpoint, sitting
    right before per-shot generation begins spending on paid media generation."""

    run_id: str
    shot_list_version: int
    approved_by: str | None
    edits: list[dict[str, Any]] | None = None
    shot_validation_retry_count: int


class CharacterReference(BaseModel):
    """Character reference image for one shot — generated once per character and cached/reused
    across the rest of the episode."""

    shot_id: str
    character_id: str
    image_uri: str
    seed: int
    character_gate_score: float | None
    cached: bool


class VideoGeneration(BaseModel):
    """Record of one video-generation attempt for a shot."""

    shot_id: str
    provider: str
    attempt_number: int
    video_uri: str
    tier1_result: str
    tier2_score: float | None
    character_consistency_score: float | None
    retry_count: int
    status: str


class VoiceGeneration(BaseModel):
    """Record of one voice-generation attempt for a shot."""

    shot_id: str
    provider: str
    voice_id: str | None
    audio_uri: str
    source_text: str
    stt_transcript: str
    verbatim_match: bool
    retry_count: int
    status: str


class ShotReview(BaseModel):
    """Record of one shot-review decision — the optional second HITL checkpoint, letting a human
    reject a single asset (character/video/voice) rather than the whole shot."""

    shot_id: str
    reviewed_by: str | None
    decision: str
    rejected_asset: str | None
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LipsyncResult(BaseModel):
    """Record of one lip-sync attempt for a shot — merges validated video and audio, then
    verifies the merge with a sync-confidence score."""

    shot_id: str
    lipsync_provider: str
    attempt_number: int
    synced_video_uri: str
    sync_confidence_score: float | None
    retry_count: int
    status: str


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
    shot_list: list[ShotSpec] = Field(default_factory=list)
    shot_validations: list[ShotValidation] = Field(default_factory=list)
    shot_validation_retry_count: int = 0
    character_references: list[CharacterReference] = Field(default_factory=list)
    video_generations: list[VideoGeneration] = Field(default_factory=list)
    voice_generations: list[VoiceGeneration] = Field(default_factory=list)
    per_asset_retry_count: int = 0
    shot_reviews: list[ShotReview] = Field(default_factory=list)
    lipsync_results: list[LipsyncResult] = Field(default_factory=list)
    lipsync_retry_count: int = 0
