import base64
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.database import Base, get_db
from app.core.limiter import limiter
from app.main import app

# Test in-memory SQLite database
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(autouse=True)
async def setup_test_db() -> AsyncGenerator[None, None]:
    """Recreates the database tables before each test for complete isolation."""
    # Temporarily disable rate limiting so unit tests remain deterministic
    limiter.enabled = False

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    async def _test_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = _test_get_db
    yield
    app.dependency_overrides.clear()
    limiter.enabled = True


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"origin": "http://localhost:5173"},
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_register_success_and_cookie_generation(client: AsyncClient) -> None:
    payload = {
        "email": "Student.One@Example.com",
        "password": "StrongPassword123!",
        "role": "student",
    }
    response = await client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201

    data = response.json()
    assert data["email"] == "student.one@example.com"  # Normalized lowercase
    assert data["role"] == "student"
    assert data["auth_provider"] == "local"

    # Verify HTTP-only cookies
    cookies = response.cookies
    assert "access_token" in cookies
    assert "refresh_token" in cookies


@pytest.mark.asyncio
async def test_register_weak_password_enforcement(client: AsyncClient) -> None:
    # Too short
    res_short = await client.post(
        "/api/v1/auth/register",
        json={"email": "u1@test.com", "password": "Short1!", "role": "student"},
    )
    assert res_short.status_code == 422 or res_short.status_code == 400

    # Missing uppercase
    res_no_upper = await client.post(
        "/api/v1/auth/register",
        json={"email": "u2@test.com", "password": "weakpassword123!", "role": "student"},
    )
    assert res_no_upper.status_code == 400
    assert "uppercase" in res_no_upper.json()["detail"].lower()

    # Missing special character
    res_no_sym = await client.post(
        "/api/v1/auth/register",
        json={"email": "u3@test.com", "password": "NoSpecialCharacter123", "role": "student"},
    )
    assert res_no_sym.status_code == 400
    assert "special character" in res_no_sym.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_duplicate_email_conflict(client: AsyncClient) -> None:
    payload = {
        "email": "duplicate@test.com",
        "password": "Password123!@#",
        "role": "student",
    }
    res1 = await client.post("/api/v1/auth/register", json=payload)
    assert res1.status_code == 201

    res2 = await client.post("/api/v1/auth/register", json=payload)
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"]


@pytest.mark.asyncio
async def test_login_success_and_enumeration_defense(client: AsyncClient) -> None:
    # Register user
    reg = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "authuser@test.com",
            "password": "ValidPassword999!",
            "role": "student",
        },
    )
    assert reg.status_code == 201

    # Login with wrong password -> Generic error
    bad_pw = await client.post(
        "/api/v1/auth/login",
        json={"email": "authuser@test.com", "password": "WrongPassword999!"},
    )
    assert bad_pw.status_code == 401
    assert bad_pw.json()["detail"] == "Invalid email or password."

    # Login with non-existent user -> Identical generic error
    no_user = await client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@test.com", "password": "SomePassword123!"},
    )
    assert no_user.status_code == 401
    assert no_user.json()["detail"] == "Invalid email or password."

    # Login with correct password
    good_login = await client.post(
        "/api/v1/auth/login",
        json={"email": "authuser@test.com", "password": "ValidPassword999!"},
    )
    assert good_login.status_code == 200
    assert good_login.json()["email"] == "authuser@test.com"
    assert "access_token" in good_login.cookies
    assert "refresh_token" in good_login.cookies


@pytest.mark.asyncio
async def test_account_lockout_after_five_failed_logins(client: AsyncClient) -> None:
    email = "lockout@test.com"
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "CorrectPassword123!", "role": "student"},
    )

    # 4 failed attempts
    for _ in range(4):
        res = await client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": "WrongPassword123!"},
        )
        assert res.status_code == 401
        assert res.json()["detail"] == "Invalid email or password."

    # 5th failed attempt triggers lockout
    res5 = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "WrongPassword123!"},
    )
    assert res5.status_code == 401

    # 6th attempt is blocked with 403 Account Locked
    res6 = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "CorrectPassword123!"},
    )
    assert res6.status_code == 403
    assert "temporarily locked" in res6.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_token_rotation_and_reuse_theft_detection(client: AsyncClient) -> None:
    # 1. Register to obtain initial token pair
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": "rotation@test.com", "password": "ValidPassword123!", "role": "student"},
    )
    assert reg.status_code == 201
    initial_refresh = reg.cookies.get("refresh_token")
    assert initial_refresh is not None

    # 2. Legitimate rotation: POST /auth/refresh with initial_refresh
    client.cookies.set("refresh_token", initial_refresh)
    ref1 = await client.post("/api/v1/auth/refresh")
    assert ref1.status_code == 200

    rotated_refresh = ref1.cookies.get("refresh_token")
    assert rotated_refresh is not None
    assert rotated_refresh != initial_refresh

    # 3. REUSE THEFT ATTACK: Malicious actor or stolen client replays the OLD initial_refresh
    client.cookies.set("refresh_token", initial_refresh)
    replay_attack = await client.post("/api/v1/auth/refresh")
    assert replay_attack.status_code == 401
    assert "reuse detected" in replay_attack.json()["detail"]

    # 4. Confirm family revocation: the legitimately rotated token is now ALSO revoked!
    client.cookies.set("refresh_token", rotated_refresh)
    post_theft_attempt = await client.post("/api/v1/auth/refresh")
    assert post_theft_attempt.status_code == 401


@pytest.mark.asyncio
async def test_rbac_student_vs_researcher_access(client: AsyncClient) -> None:
    # Register Student
    await client.post(
        "/api/v1/auth/register",
        json={"email": "student@test.com", "password": "Password123!@#", "role": "student"},
    )

    # Student accessing /student/ping -> 200
    res_student_ping = await client.get("/api/v1/student/ping")
    assert res_student_ping.status_code == 200
    assert res_student_ping.json()["role"] == "student"

    # Student attempting /research/ping -> 403 Access Denied
    res_research_denied = await client.get("/api/v1/research/ping")
    assert res_research_denied.status_code == 403
    assert "researcher role required" in res_research_denied.json()["detail"]

    # Register Researcher
    client.cookies.clear()
    await client.post(
        "/api/v1/auth/register",
        json={"email": "researcher@test.com", "password": "Password123!@#", "role": "researcher"},
    )

    # Researcher accessing /research/ping -> 200
    res_research_allowed = await client.get("/api/v1/research/ping")
    assert res_research_allowed.status_code == 200
    assert res_research_allowed.json()["role"] == "researcher"

    # Researcher accessing /student/ping -> 200 (researchers have universal access)
    res_student_also = await client.get("/api/v1/student/ping")
    assert res_student_also.status_code == 200


@pytest.mark.asyncio
async def test_auth_me_and_logout_lifecycle(client: AsyncClient) -> None:
    reg = await client.post(
        "/api/v1/auth/register",
        json={"email": "lifecycle@test.com", "password": "Password123!@#", "role": "student"},
    )
    assert reg.status_code == 201

    # Access /auth/me with active cookies
    me = await client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "lifecycle@test.com"

    # Logout
    logout = await client.post("/api/v1/auth/logout")
    assert logout.status_code == 200

    # /auth/me without cookies returns 401
    client.cookies.clear()
    me_after = await client.get("/api/v1/auth/me")
    assert me_after.status_code == 401


@pytest.mark.asyncio
async def test_google_oauth_pkce_and_callback_flow(client: AsyncClient) -> None:
    # 1. Initiate Google Login -> Expect 302 redirect and oauth_pkce cookie
    login_resp = await client.get("/api/v1/auth/google/login", follow_redirects=False)
    assert login_resp.status_code == 302
    assert "accounts.google.com" in login_resp.headers["location"]
    assert "code_challenge=" in login_resp.headers["location"]

    pkce_cookie = login_resp.cookies.get("oauth_pkce")
    assert pkce_cookie is not None
    raw_b64 = pkce_cookie.strip('"').encode("utf-8")
    decoded_cookie = base64.urlsafe_b64decode(raw_b64).decode("utf-8")
    pkce_data = json.loads(decoded_cookie)
    state = pkce_data["state"]

    # 2. Callback with state mismatch -> 400 CSRF
    client.cookies.set("oauth_pkce", pkce_cookie)
    bad_state_resp = await client.get(
        "/api/v1/auth/google/callback",
        params={"code": "mock-code", "state": "wrong-state"},
        follow_redirects=False,
    )
    assert bad_state_resp.status_code == 400
    assert "state mismatch" in bad_state_resp.json()["detail"].lower()

    # 3. Callback with valid mock OIDC token -> Creates user, sets cookies, redirects to /dashboard
    mock_id_payload = {
        "sub": "google-sub-unique-12345",
        "email": "googlestudent@example.com",
        "email_verified": True,
        "aud": settings.GOOGLE_CLIENT_ID,
        "iss": "https://accounts.google.com",
        "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
    }
    mock_id_token = jwt.encode(
        mock_id_payload,
        "mock-key-minimum-32-bytes-long-for-test!!",
        algorithm="HS256",
    )

    callback_resp = await client.get(
        "/api/v1/auth/google/callback",
        params={
            "code": "mock-code",
            "state": state,
            "mock_id_token": mock_id_token,
        },
        follow_redirects=False,
    )
    assert callback_resp.status_code == 302
    assert "/dashboard" in callback_resp.headers["location"]
    assert "access_token" in callback_resp.cookies
    assert "refresh_token" in callback_resp.cookies

    # 4. Verify user was created in database and can access /auth/me
    client.cookies.set("access_token", callback_resp.cookies["access_token"])
    me_resp = await client.get("/api/v1/auth/me")
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "googlestudent@example.com"
    assert me_resp.json()["auth_provider"] == "google"


@pytest.mark.asyncio
async def test_security_headers_present_on_all_responses(client: AsyncClient) -> None:
    res = await client.get("/")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


@pytest.mark.asyncio
async def test_rate_limiting_enforcement(client: AsyncClient) -> None:
    limiter.enabled = True
    try:
        # /api/v1/auth/login has a 5/minute limit per IP
        responses = []
        for i in range(7):
            res = await client.post(
                "/api/v1/auth/login",
                json={"email": f"rl{i}@test.com", "password": "Password123!"},
            )
            responses.append(res)

        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes
        rate_limited_resp = next(r for r in responses if r.status_code == 429)
        assert "Retry-After" in rate_limited_resp.headers
    finally:
        limiter.enabled = False
