import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.database import Base, get_db
from app.core.limiter import limiter
from app.main import app

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(autouse=True)
async def setup_test_db() -> AsyncGenerator[None, None]:
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    limiter.enabled = False

    yield

    app.dependency_overrides.clear()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        cookies={},
        headers={"Origin": "http://testserver"},
    ) as c:
        yield c


@pytest.mark.asyncio
async def test_extension_token_creation_and_revocation(client: AsyncClient) -> None:
    # 1. Register student
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": "ext_student@example.com", "password": "Password123!@#", "role": "student"},
    )
    assert reg.status_code == 201

    # 2. Generate extension token
    token_resp = await client.post(
        "/api/v1/auth/extension-token",
        json={"name": "Work Laptop Chrome"},
    )
    assert token_resp.status_code == 201
    data = token_resp.json()
    assert "cgt_" in data["token"]
    assert data["name"] == "Work Laptop Chrome"
    raw_token = data["token"]

    # 3. Check status endpoint
    status_resp = await client.get("/api/v1/auth/extension-token/status")
    assert status_resp.status_code == 200
    statuses = status_resp.json()
    assert len(statuses) == 1
    assert statuses[0]["is_active"] is True

    # 4. Revoke token
    rev_resp = await client.delete("/api/v1/auth/extension-token")
    assert rev_resp.status_code == 200

    # 5. Check status is now inactive
    status_resp2 = await client.get("/api/v1/auth/extension-token/status")
    assert status_resp2.status_code == 200
    assert status_resp2.json()[0]["is_active"] is False

    # 6. Attempt to use revoked token -> 401
    ext_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
    failed_ingest = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {raw_token}"},
        json={
            "events": [
                {
                    "client_event_id": str(uuid.uuid4()),
                    "source_domain": "docs.python.org",
                    "source_url": "https://docs.python.org/3/tutorial/",
                    "page_title": "The Python Tutorial",
                    "media_type": "docs",
                    "active_dwell_seconds": 120,
                    "estimated_coverage_pct": 0.45,
                    "viewed_headings": ["Introduction", "Using Python"],
                    "top_keywords": ["python", "interpreter", "syntax"],
                    "is_revisit": False,
                    "started_at": datetime.now(UTC).isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                }
            ]
        },
    )
    assert failed_ingest.status_code == 401
    assert "Invalid or revoked extension token" in failed_ingest.json()["detail"]


@pytest.mark.asyncio
async def test_passive_telemetry_ingestion_and_idempotency(client: AsyncClient) -> None:
    # 1. Register and generate active extension token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "telemetry_test@example.com",
            "password": "Password123!@#",
            "role": "student",
        },
    )
    token_resp = await client.post("/api/v1/auth/extension-token")
    bearer_token = token_resp.json()["token"]

    # 2. Ingest batch of 2 passive events
    event_id_1 = str(uuid.uuid4())
    event_id_2 = str(uuid.uuid4())
    now = datetime.now(UTC)

    payload = {
        "events": [
            {
                "client_event_id": event_id_1,
                "source_domain": "react.dev",
                "source_url": "https://react.dev/learn",
                "page_title": "Quick Start – React",
                "media_type": "docs",
                "active_dwell_seconds": 180,
                "estimated_coverage_pct": 0.72,
                "viewed_headings": ["Creating components", "Writing markup"],
                "top_keywords": ["component", "jsx", "state", "props"],
                "is_revisit": False,
                "started_at": (now - timedelta(minutes=5)).isoformat(),
                "ended_at": (now - timedelta(minutes=2)).isoformat(),
            },
            {
                "client_event_id": event_id_2,
                "source_domain": "youtube.com",
                "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "page_title": "Distributed Systems Lecture 1",
                "media_type": "video",
                "active_dwell_seconds": 600,
                "estimated_coverage_pct": 0.85,
                "viewed_headings": ["System Architecture", "Failure Modes"],
                "top_keywords": ["consensus", "raft", "fault-tolerance"],
                "is_revisit": True,
                "days_since_last_visit": 2.5,
                "video_metrics": {
                    "channel_name": "MIT OpenCourseWare",
                    "video_duration_sec": 3600,
                    "playback_rate_avg": 1.25,
                    "rewind_count": 3,
                    "pause_count": 2,
                },
                "started_at": (now - timedelta(minutes=15)).isoformat(),
                "ended_at": (now - timedelta(minutes=5)).isoformat(),
            },
        ]
    }

    ext_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")
    res1 = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json=payload,
    )
    assert res1.status_code == 202
    assert res1.json()["ingested"] == 2
    assert res1.json()["duplicates_ignored"] == 0

    # 3. Idempotency test: Resend the exact same batch with duplicate event_id_1 and event_id_2
    res2 = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json=payload,
    )
    assert res2.status_code == 202
    assert res2.json()["ingested"] == 0
    assert res2.json()["duplicates_ignored"] == 2

    # 4. Partial duplicate test: Batch with 1 existing and 1 new event
    event_id_3 = str(uuid.uuid4())
    partial_payload = {
        "events": [
            payload["events"][0],  # duplicate
            {
                "client_event_id": event_id_3,
                "source_domain": "arxiv.org",
                "source_url": "https://arxiv.org/abs/1706.03762",
                "page_title": "Attention Is All You Need",
                "media_type": "article",
                "active_dwell_seconds": 300,
                "estimated_coverage_pct": 0.50,
                "viewed_headings": ["Transformer Architecture", "Multi-Head Attention"],
                "top_keywords": ["transformer", "attention", "encoder", "decoder"],
                "is_revisit": False,
                "started_at": (now - timedelta(minutes=30)).isoformat(),
                "ended_at": (now - timedelta(minutes=25)).isoformat(),
            },
        ]
    }
    res3 = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json=partial_payload,
    )
    assert res3.status_code == 202
    assert res3.json()["ingested"] == 1
    assert res3.json()["duplicates_ignored"] == 1

    # 5. Fetch recent passive telemetry via web authenticated session
    recent_res = await client.get("/api/v1/telemetry/passive/recent")
    assert recent_res.status_code == 200
    records = recent_res.json()
    assert len(records) == 3
    # Check that video metrics were stored and retrieved correctly
    video_rec = next(r for r in records if r["media_type"] == "video")
    assert video_rec["video_metrics"]["channel_name"] == "MIT OpenCourseWare"
    assert video_rec["video_metrics"]["rewind_count"] == 3


@pytest.mark.asyncio
async def test_telemetry_validation_and_timestamp_clamping(client: AsyncClient) -> None:
    # 1. Register and generate active extension token
    await client.post(
        "/api/v1/auth/register",
        json={"email": "clamp_test@example.com", "password": "Password123!@#", "role": "student"},
    )
    token_resp = await client.post("/api/v1/auth/extension-token")
    bearer_token = token_resp.json()["token"]

    ext_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")

    # 2. Validation failure: active_dwell_seconds < 15
    res_short = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json={
            "events": [
                {
                    "client_event_id": str(uuid.uuid4()),
                    "source_domain": "example.com",
                    "source_url": "https://example.com",
                    "page_title": "Example",
                    "media_type": "article",
                    "active_dwell_seconds": 10,  # Below minimum 15s
                    "estimated_coverage_pct": 0.2,
                    "started_at": datetime.now(UTC).isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                }
            ]
        },
    )
    assert res_short.status_code == 422

    # 3. Validation failure: estimated_coverage_pct > 1.0
    res_cov = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json={
            "events": [
                {
                    "client_event_id": str(uuid.uuid4()),
                    "source_domain": "example.com",
                    "source_url": "https://example.com",
                    "page_title": "Example",
                    "media_type": "article",
                    "active_dwell_seconds": 60,
                    "estimated_coverage_pct": 1.5,  # Exceeds 1.0
                    "started_at": datetime.now(UTC).isoformat(),
                    "ended_at": datetime.now(UTC).isoformat(),
                }
            ]
        },
    )
    assert res_cov.status_code == 422

    # 4. Future timestamp clamping test
    future_time = datetime.now(UTC) + timedelta(hours=2)
    event_id = str(uuid.uuid4())
    res_clamp = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json={
            "events": [
                {
                    "client_event_id": event_id,
                    "source_domain": "example.com",
                    "source_url": "https://example.com",
                    "page_title": "Future Event",
                    "media_type": "article",
                    "active_dwell_seconds": 60,
                    "estimated_coverage_pct": 0.5,
                    "started_at": future_time.isoformat(),
                    "ended_at": (future_time + timedelta(minutes=5)).isoformat(),
                }
            ]
        },
    )
    assert res_clamp.status_code == 202
    assert res_clamp.json()["ingested"] == 1

    # Check that it was clamped to server UTC now
    recent_res = await client.get("/api/v1/telemetry/passive/recent")
    assert recent_res.status_code == 200
    stored_event = next(r for r in recent_res.json() if r["client_event_id"] == event_id)
    stored_ended = datetime.fromisoformat(stored_event["ended_at"])
    if stored_ended.tzinfo is None:
        stored_ended = stored_ended.replace(tzinfo=UTC)
    now_utc = datetime.now(UTC)
    # Stored time should be <= now + 5 seconds (not 2 hours in future)
    assert stored_ended <= now_utc + timedelta(seconds=5)


@pytest.mark.asyncio
async def test_hourly_session_coalescing_and_meta_description(client: AsyncClient) -> None:
    # 1. Register student & extension token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "coalesce_test@example.com",
            "password": "Password123!@#",
            "role": "student",
        },
    )
    token_resp = await client.post("/api/v1/auth/extension-token")
    bearer_token = token_resp.json()["token"]

    ext_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")

    # 2. Ingest first session for a page
    event_id_1 = str(uuid.uuid4())
    start_t1 = datetime(2026, 1, 15, 10, 10, 0, tzinfo=UTC)
    end_t1 = datetime(2026, 1, 15, 10, 15, 0, tzinfo=UTC)

    res1 = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json={
            "events": [
                {
                    "client_event_id": event_id_1,
                    "source_domain": "geeksforgeeks.org",
                    "source_url": "https://geeksforgeeks.org/data-preprocessing",
                    "page_title": "Data Preprocessing in Python",
                    "meta_description": "Learn data preprocessing steps with python examples.",
                    "media_type": "article",
                    "active_dwell_seconds": 35,
                    "estimated_coverage_pct": 0.40,
                    "viewed_headings": ["Introduction", "Handling Missing Data"],
                    "top_keywords": ["data preprocessing", "missing values"],
                    "is_revisit": False,
                    "started_at": start_t1.isoformat(),
                    "ended_at": end_t1.isoformat(),
                }
            ]
        },
    )
    assert res1.status_code == 202
    assert res1.json()["ingested"] == 1
    assert res1.json()["duplicates_ignored"] == 0

    # 3. Ingest second session for same URL within same hour bucket with different client_event_id
    event_id_2 = str(uuid.uuid4())
    start_t2 = datetime(2026, 1, 15, 10, 25, 0, tzinfo=UTC)
    end_t2 = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)

    res2 = await ext_client.post(
        "/api/v1/telemetry/passive",
        headers={"Authorization": f"Bearer {bearer_token}"},
        json={
            "events": [
                {
                    "client_event_id": event_id_2,
                    "source_domain": "geeksforgeeks.org",
                    "source_url": "https://geeksforgeeks.org/data-preprocessing",
                    "page_title": "Data Preprocessing in Python",
                    "media_type": "article",
                    "active_dwell_seconds": 45,
                    "estimated_coverage_pct": 0.85,
                    "viewed_headings": ["Handling Missing Data", "Feature Scaling"],
                    "top_keywords": ["feature scaling", "normalization"],
                    "is_revisit": False,
                    "started_at": start_t2.isoformat(),
                    "ended_at": end_t2.isoformat(),
                }
            ]
        },
    )
    assert res2.status_code == 202
    assert res2.json()["ingested"] == 1
    assert res2.json()["duplicates_ignored"] == 0

    # 4. Verify only 1 coalesced record exists in DB for this page and session bucket
    recent_res = await client.get("/api/v1/telemetry/passive/recent")
    assert recent_res.status_code == 200
    records = [
        r
        for r in recent_res.json()
        if r["source_url"] == "https://geeksforgeeks.org/data-preprocessing"
    ]
    assert len(records) == 1
    record = records[0]

    # Verify coalesced metrics
    assert record["session_bucket"] == "2026-01-15T10"
    assert record["active_dwell_seconds"] == 35 + 45  # 80s summed
    assert record["estimated_coverage_pct"] == 0.85  # max coverage
    assert record["meta_description"] == "Learn data preprocessing steps with python examples."
    assert set(record["viewed_headings"]) == {
        "Introduction",
        "Handling Missing Data",
        "Feature Scaling",
    }
    assert set(record["top_keywords"]) == {
        "data preprocessing",
        "missing values",
        "feature scaling",
        "normalization",
    }
