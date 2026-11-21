import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VideoMetricsSchema(BaseModel):
    channel_name: str | None = Field(default=None, max_length=100)
    video_duration_sec: int | None = Field(default=None, ge=0)
    playback_rate_avg: float | None = Field(default=None, ge=0.25, le=4.0)
    rewind_count: int = Field(default=0, ge=0)
    pause_count: int = Field(default=0, ge=0)


class PassiveEventSchema(BaseModel):
    client_event_id: uuid.UUID
    source_domain: str = Field(min_length=1, max_length=255)
    source_url: str = Field(min_length=1, max_length=256)
    page_title: str = Field(min_length=1, max_length=160)
    media_type: Literal["article", "docs", "video", "pdf"] = "article"
    active_dwell_seconds: int = Field(ge=15, le=14400)
    estimated_coverage_pct: float = Field(ge=0.0, le=1.0)
    viewed_headings: list[str] = Field(default_factory=list, max_length=5)
    top_keywords: list[str] = Field(default_factory=list, max_length=10)
    is_revisit: bool = False
    days_since_last_visit: float | None = Field(default=None, ge=0.0)
    video_metrics: VideoMetricsSchema | None = None
    started_at: datetime
    ended_at: datetime

    @model_validator(mode="after")
    def clamp_timestamps_and_bounds(self) -> "PassiveEventSchema":
        """Clamps future timestamps to server UTC now and ensures ended_at >= started_at."""
        now = datetime.now(UTC)
        # Ensure timezone-aware UTC for comparison
        started = self.started_at if self.started_at.tzinfo else self.started_at.replace(tzinfo=UTC)
        ended = self.ended_at if self.ended_at.tzinfo else self.ended_at.replace(tzinfo=UTC)

        if started > now:
            started = now
        if ended > now:
            ended = now
        if ended < started:
            ended = started

        self.started_at = started
        self.ended_at = ended

        # Truncate string items in headings and keywords if necessary
        self.viewed_headings = [h[:100] for h in self.viewed_headings[:5]]
        self.top_keywords = [k[:64] for k in self.top_keywords[:10]]

        return self


class PassiveBatchRequest(BaseModel):
    events: list[PassiveEventSchema] = Field(min_length=1, max_length=50)


class PassiveIngestResponse(BaseModel):
    ingested: int
    duplicates_ignored: int


class PassiveExposureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    client_event_id: uuid.UUID
    concept_id: uuid.UUID | None
    source_domain: str
    source_url: str
    page_title: str
    media_type: str
    active_dwell_seconds: int
    estimated_coverage_pct: float
    viewed_headings: list[str]
    top_keywords: list[str]
    is_revisit: bool
    days_since_last_visit: float | None
    video_metrics: dict[str, Any] | None
    started_at: datetime
    ended_at: datetime
    received_at: datetime
