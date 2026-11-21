from datetime import UTC, datetime

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


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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

    # 1. Exact retry deduplication via client_event_id
    event_ids = [event.client_event_id for event in batch.events]
    existing_result = await db.execute(
        select(PassiveExposure.client_event_id).where(
            PassiveExposure.client_event_id.in_(event_ids)
        )
    )
    existing_ids = set(existing_result.scalars().all())

    new_events = [e for e in batch.events if e.client_event_id not in existing_ids]
    duplicates_count = len(batch.events) - len(new_events)

    if not new_events:
        return PassiveIngestResponse(
            ingested=0,
            duplicates_ignored=duplicates_count,
        )

    # 2. Fetch existing session buckets for this user and incoming URLs
    buckets_to_check = {e.started_at.strftime("%Y-%m-%dT%H") for e in new_events}
    urls_to_check = {e.source_url for e in new_events}

    existing_rows_res = await db.execute(
        select(PassiveExposure).where(
            PassiveExposure.user_id == user.id,
            PassiveExposure.source_url.in_(urls_to_check),
            PassiveExposure.session_bucket.in_(buckets_to_check),
        )
    )
    existing_rows = existing_rows_res.scalars().all()
    bucket_map: dict[tuple[str, str], PassiveExposure] = {
        (r.source_url, r.session_bucket): r for r in existing_rows
    }

    ingested_count = 0
    for event in new_events:
        session_bucket = event.started_at.strftime("%Y-%m-%dT%H")
        key = (event.source_url, session_bucket)

        if key in bucket_map:
            existing = bucket_map[key]
            # Coalesce: sum dwell seconds
            existing.active_dwell_seconds += event.active_dwell_seconds
            # Max coverage
            existing.estimated_coverage_pct = max(
                existing.estimated_coverage_pct, event.estimated_coverage_pct
            )
            # Union headings up to 6
            existing.viewed_headings = list(
                dict.fromkeys(existing.viewed_headings + event.viewed_headings)
            )[:6]
            # Union keywords up to 12
            existing.top_keywords = list(dict.fromkeys(existing.top_keywords + event.top_keywords))[
                :12
            ]
            # Update ended_at if newer
            if _to_utc(event.ended_at) > _to_utc(existing.ended_at):
                existing.ended_at = event.ended_at
            # Meta description fallback
            if event.meta_description and not existing.meta_description:
                existing.meta_description = event.meta_description
            # Revisit flag
            if event.is_revisit:
                existing.is_revisit = True
            # Merge video metrics if present
            if event.video_metrics:
                curr_vm = existing.video_metrics or {}
                new_vm = event.video_metrics.model_dump()
                existing.video_metrics = {
                    "channel_name": new_vm.get("channel_name") or curr_vm.get("channel_name"),
                    "video_duration_sec": (
                        new_vm.get("video_duration_sec") or curr_vm.get("video_duration_sec")
                    ),
                    "playback_rate_avg": (
                        new_vm.get("playback_rate_avg") or curr_vm.get("playback_rate_avg", 1.0)
                    ),
                    "rewind_count": curr_vm.get("rewind_count", 0) + new_vm.get("rewind_count", 0),
                    "pause_count": curr_vm.get("pause_count", 0) + new_vm.get("pause_count", 0),
                }
            ingested_count += 1
        else:
            # Create new row
            exposure = PassiveExposure(
                client_event_id=event.client_event_id,
                user_id=user.id,
                concept_id=None,
                session_bucket=session_bucket,
                source_domain=event.source_domain,
                source_url=event.source_url,
                page_title=event.page_title,
                meta_description=event.meta_description,
                mapping_confidence=None,
                media_type=event.media_type,
                active_dwell_seconds=event.active_dwell_seconds,
                estimated_coverage_pct=event.estimated_coverage_pct,
                viewed_headings=event.viewed_headings[:6],
                top_keywords=event.top_keywords[:12],
                is_revisit=event.is_revisit,
                days_since_last_visit=event.days_since_last_visit,
                video_metrics=event.video_metrics.model_dump() if event.video_metrics else None,
                started_at=event.started_at,
                ended_at=event.ended_at,
            )
            db.add(exposure)
            bucket_map[key] = exposure
            ingested_count += 1

    await db.commit()

    return PassiveIngestResponse(
        ingested=ingested_count,
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
