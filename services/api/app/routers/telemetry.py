from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_active_user, get_current_extension_user
from app.core.limiter import limiter
from app.models.extension_token import ExtensionToken
from app.models.passive_exposure import PassiveExposure
from app.models.user import User
from app.schemas.telemetry import (
    PassiveBatchRequest,
    PassiveExposureResponse,
    PassiveIngestResponse,
)

router = APIRouter(prefix="/telemetry", tags=["Passive Telemetry"])


@router.post(
    "/passive",
    response_model=PassiveIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest batched passive study telemetry events (idempotent)",
)
@limiter.limit("30/minute")
async def ingest_passive_telemetry(
    request: Request,
    batch: PassiveBatchRequest,
    auth_data: tuple[User, ExtensionToken] = Depends(get_current_extension_user),
    db: AsyncSession = Depends(get_db),
) -> PassiveIngestResponse:
    """Ingests batched passive study focus events from Chrome Extension.

    Guarantees idempotency via `client_event_id` unique index check. Discards
    duplicate client events without error, returning counts for ingested vs
    ignored records.
    """
    user, _ = auth_data

    # Extract all event IDs in incoming batch
    event_ids = [event.client_event_id for event in batch.events]

    # Query existing IDs to enforce idempotency
    existing_result = await db.execute(
        select(PassiveExposure.client_event_id).where(
            PassiveExposure.client_event_id.in_(event_ids)
        )
    )
    existing_ids = set(existing_result.scalars().all())

    new_events = [e for e in batch.events if e.client_event_id not in existing_ids]
    duplicates_count = len(batch.events) - len(new_events)

    for event in new_events:
        exposure = PassiveExposure(
            client_event_id=event.client_event_id,
            user_id=user.id,
            concept_id=None,  # Seam: mapped to Knowledge Graph nodes in Phase 4
            source_domain=event.source_domain,
            source_url=event.source_url,
            page_title=event.page_title,
            media_type=event.media_type,
            active_dwell_seconds=event.active_dwell_seconds,
            estimated_coverage_pct=event.estimated_coverage_pct,
            viewed_headings=event.viewed_headings,
            top_keywords=event.top_keywords,
            is_revisit=event.is_revisit,
            days_since_last_visit=event.days_since_last_visit,
            video_metrics=event.video_metrics.model_dump() if event.video_metrics else None,
            started_at=event.started_at,
            ended_at=event.ended_at,
        )
        db.add(exposure)

    if new_events:
        await db.commit()

    return PassiveIngestResponse(
        ingested=len(new_events),
        duplicates_ignored=duplicates_count,
    )


@router.get(
    "/passive/recent",
    response_model=list[PassiveExposureResponse],
    status_code=status.HTTP_200_OK,
    summary="Get recent passive study exposure sessions for current user",
)
async def get_recent_passive_telemetry(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[PassiveExposureResponse]:
    """Retrieves the student's 20 most recent passive study exposure sessions."""
    result = await db.execute(
        select(PassiveExposure)
        .where(PassiveExposure.user_id == current_user.id)
        .order_by(PassiveExposure.ended_at.desc())
        .limit(20)
    )
    records = result.scalars().all()
    return [PassiveExposureResponse.model_validate(r) for r in records]
