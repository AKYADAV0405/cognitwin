from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User

JSON_COMPAT = JSON().with_variant(JSONB, "postgresql")


class PassiveExposure(Base):
    """Passive study telemetry record captured by the Chrome extension."""

    __tablename__ = "passive_exposures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    client_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        unique=True,
        index=True,
        nullable=False,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    concept_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )  # Future Phase 4 Concept Graph Seam
    source_domain: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    source_url: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
    )
    page_title: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
    )
    media_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="article",
    )  # "article" | "docs" | "video" | "pdf"
    active_dwell_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    estimated_coverage_pct: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    viewed_headings: Mapped[list[str]] = mapped_column(
        JSON_COMPAT,
        nullable=False,
        default=list,
    )
    top_keywords: Mapped[list[str]] = mapped_column(
        JSON_COMPAT,
        nullable=False,
        default=list,
    )
    is_revisit: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    days_since_last_visit: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    video_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON_COMPAT,
        nullable=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    ended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    user: Mapped[User] = relationship(
        "User",
        back_populates="passive_exposures",
    )
