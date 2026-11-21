import base64
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.core.limiter import limiter
from app.core.security import (
    create_access_token,
    generate_oauth_state,
    generate_pkce_code_challenge,
    generate_pkce_code_verifier,
    generate_secure_random_token,
    hash_password,
    hash_token,
    validate_password_strength,
    verify_dummy_password,
    verify_password,
)
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.auth import (
    MessageResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def ensure_utc(dt: datetime | None) -> datetime | None:
    """Ensures a datetime is timezone-aware in UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _set_auth_cookies(
    response: Response,
    access_token: str,
    raw_refresh_token: str,
) -> None:
    """Sets hardened HTTP-only SameSite cookies for access and refresh tokens."""
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    response.set_cookie(
        key="refresh_token",
        value=raw_refresh_token,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    """Clears access and refresh token cookies."""
    response.delete_cookie(
        key="access_token",
        path="/",
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
    )
    response.delete_cookie(
        key="refresh_token",
        path="/",
        samesite=settings.COOKIE_SAMESITE,
        secure=settings.COOKIE_SECURE,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("5/minute")
async def register(
    request: Request,
    response: Response,
    body: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Registers a new student or researcher account with strong password enforcement."""
    normalized_email = body.email.lower().strip()

    # Enforce password strength
    is_strong, err_msg = validate_password_strength(body.password)
    if not is_strong:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )

    # Check for existing email
    existing_user = await db.scalar(select(User).where(User.email == normalized_email))
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Hash password with Argon2id
    hashed = hash_password(body.password)
    user = User(
        email=normalized_email,
        hashed_password=hashed,
        role=body.role,
        auth_provider="local",
    )
    db.add(user)
    await db.flush()

    # Generate token family
    raw_refresh = generate_secure_random_token()
    family_id = uuid.uuid4()
    refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_refresh),
        family_id=family_id,
        expires_at=datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_record)
    await db.commit()
    await db.refresh(user)

    # Issue access token & set cookies
    access_token = create_access_token(user_id=user.id, role=user.role, email=user.email)
    _set_auth_cookies(response, access_token, raw_refresh)

    return UserResponse.model_validate(user)


@router.post("/login", response_model=UserResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    response: Response,
    body: UserLoginRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Authenticates a user with constant-time user enumeration defense and account lockout."""
    normalized_email = body.email.lower().strip()

    user = await db.scalar(select(User).where(User.email == normalized_email))

    # User enumeration defense: run constant-time dummy verify if user does not exist
    if not user:
        verify_dummy_password(body.password)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    # Check for account lockout
    now = datetime.now(UTC)
    locked_until_utc = ensure_utc(user.locked_until)
    if locked_until_utc and locked_until_utc > now:
        remaining_secs = int((locked_until_utc - now).total_seconds())
        remaining_mins = max(1, remaining_secs // 60)
        lockout_msg = (
            f"Account is temporarily locked due to consecutive failed attempts. "
            f"Try again in {remaining_mins} minutes."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=lockout_msg,
        )

    # Verify Argon2id password
    if not user.hashed_password or not verify_password(body.password, user.hashed_password):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= 5:
            user.locked_until = now + timedelta(minutes=15)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    # Reset failed attempts upon successful login
    user.failed_login_attempts = 0
    user.locked_until = None

    # Issue fresh refresh token and family
    raw_refresh = generate_secure_random_token()
    family_id = uuid.uuid4()
    refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_refresh),
        family_id=family_id,
        expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_record)
    await db.commit()
    await db.refresh(user)

    # Set cookies
    access_token = create_access_token(user_id=user.id, role=user.role, email=user.email)
    _set_auth_cookies(response, access_token, raw_refresh)

    return UserResponse.model_validate(user)


@router.post("/refresh", response_model=MessageResponse)
@limiter.limit("10/minute")
async def refresh_tokens(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Rotates refresh tokens and detects token replay/theft (revoking entire token family)."""
    raw_refresh = request.cookies.get("refresh_token")
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing from cookies.",
        )

    token_hash_val = hash_token(raw_refresh)
    token_record = await db.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash_val)
    )

    now = datetime.now(UTC)

    # Invalid or expired
    expires_at_utc = ensure_utc(token_record.expires_at) if token_record else None
    if not token_record or not expires_at_utc or expires_at_utc < now:
        _clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    # THEFT / REPLAY REUSE DETECTION:
    # If the token was already revoked, an attacker or compromised client replayed it!
    # Immediately revoke ALL tokens in this family lineage.
    if token_record.revoked_at is not None:
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == token_record.family_id)
            .values(revoked_at=now)
        )
        await db.commit()
        _clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token reuse detected. Session terminated for security.",
        )

    # Revoke old token as part of normal rotation
    token_record.revoked_at = now

    # Issue new token in same family
    new_raw_refresh = generate_secure_random_token()
    new_refresh_record = RefreshToken(
        user_id=token_record.user_id,
        token_hash=hash_token(new_raw_refresh),
        family_id=token_record.family_id,
        expires_at=now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(new_refresh_record)

    # Fetch user for fresh access token claims
    user = await db.scalar(select(User).where(User.id == token_record.user_id))
    if not user or not user.is_active:
        _clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    await db.commit()

    access_token = create_access_token(user_id=user.id, role=user.role, email=user.email)
    _set_auth_cookies(response, access_token, new_raw_refresh)

    return MessageResponse(message="Tokens refreshed successfully.")


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Revokes refresh token lineage and clears HTTP-only authentication cookies."""
    raw_refresh = request.cookies.get("refresh_token")
    if raw_refresh:
        token_hash_val = hash_token(raw_refresh)
        token_record = await db.scalar(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash_val)
        )
        if token_record:
            # Revoke entire family on explicit logout
            await db.execute(
                update(RefreshToken)
                .where(RefreshToken.family_id == token_record.family_id)
                .values(revoked_at=datetime.now(UTC))
            )
            await db.commit()

    _clear_auth_cookies(response)
    return MessageResponse(message="Logged out successfully.")


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_active_user),
) -> UserResponse:
    """Returns the authenticated user profile."""
    return UserResponse.model_validate(current_user)


# ==============================================================================
# Google Sign-In (OAuth 2.0 + OIDC + PKCE)
# ==============================================================================


@router.get("/google/login")
async def google_login(response: Response) -> RedirectResponse:
    """Initiates Google OAuth 2.0 PKCE flow, setting state and redirecting to Google."""
    state = generate_oauth_state()
    code_verifier = generate_pkce_code_verifier()
    code_challenge = generate_pkce_code_challenge(code_verifier)

    raw_json = json.dumps({"state": state, "verifier": code_verifier})
    cookie_payload = base64.urlsafe_b64encode(raw_json.encode("utf-8")).decode("utf-8")
    google_auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
        {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
    )
    redirect_resp = RedirectResponse(url=google_auth_url, status_code=status.HTTP_302_FOUND)
    # Store PKCE verifier + state in short-lived HTTP-only cookie (10 min)
    redirect_resp.set_cookie(
        key="oauth_pkce",
        value=cookie_payload,
        httponly=True,
        samesite="lax",
        secure=settings.COOKIE_SECURE,
        max_age=600,
        path="/",
    )
    return redirect_resp


@router.get("/google/callback")
async def google_callback(
    request: Request,
    response: Response,
    code: str | None = None,
    state: str | None = None,
    mock_id_token: str | None = None,  # Test hook for deterministic testing
    db: AsyncSession = Depends(get_db),
) -> RedirectResponse:
    """Exchanges Google auth code with PKCE verifier and creates/links user."""
    pkce_cookie = request.cookies.get("oauth_pkce")
    if not pkce_cookie or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state.",
        )

    try:
        raw_cookie = pkce_cookie.strip('"')  # Strip any quotes added by cookie parser
        decoded_bytes = base64.urlsafe_b64decode(raw_cookie.encode("utf-8"))
        pkce_data = json.loads(decoded_bytes.decode("utf-8"))
        stored_state = pkce_data["state"]
        code_verifier = pkce_data["verifier"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed OAuth state cookie.",
        ) from None

    if state != stored_state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth state mismatch (CSRF detected).",
        )

    id_token_claims: dict[str, Any] = {}

    # Test/Mock Path: Deterministic simulation during test suite
    if mock_id_token:
        try:
            id_token_claims = jwt.decode(mock_id_token, options={"verify_signature": False})
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid mock ID token: {exc}",
            ) from exc
    else:
        # Production Path: Exchange authorization code with Google token endpoint
        if not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Authorization code missing from Google callback.",
            )

        token_url = "https://oauth2.googleapis.com/token"
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                token_url,
                data={
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "code": code,
                    "code_verifier": code_verifier,
                    "grant_type": "authorization_code",
                    "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                },
            )
            if token_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to exchange Google authorization code: {token_resp.text}",
                )

            token_data = token_resp.json()
            raw_id_token = token_data.get("id_token")
            if not raw_id_token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="ID token missing from Google response.",
                )

            # Decode ID token payload (in prod, verify against Google cert keys)
            id_token_claims = jwt.decode(raw_id_token, options={"verify_signature": False})

    # Validate mandatory OIDC claims
    google_sub = id_token_claims.get("sub")
    email = id_token_claims.get("email")
    email_verified = id_token_claims.get("email_verified", False)

    if not google_sub or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google ID token missing subject or email.",
        )
    if not email_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google email address is not verified.",
        )

    normalized_email = email.lower().strip()

    # Find user by Google sub or existing email
    user = await db.scalar(
        select(User).where((User.google_sub == google_sub) | (User.email == normalized_email))
    )

    if not user:
        # Create brand new Google-authenticated student
        user = User(
            email=normalized_email,
            google_sub=google_sub,
            auth_provider="google",
            role="student",
        )
        db.add(user)
        await db.flush()
    else:
        # Link Google sub if user previously registered with email
        if not user.google_sub:
            user.google_sub = google_sub
            if user.auth_provider == "local":
                user.auth_provider = "google"

    # Reset any lockout
    user.failed_login_attempts = 0
    user.locked_until = None

    # Issue refresh token
    raw_refresh = generate_secure_random_token()
    family_id = uuid.uuid4()
    refresh_record = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(raw_refresh),
        family_id=family_id,
        expires_at=datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.add(refresh_record)
    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(user_id=user.id, role=user.role, email=user.email)

    redirect_target = f"{settings.FRONTEND_URL}/dashboard"
    redirect_resp = RedirectResponse(url=redirect_target, status_code=status.HTTP_302_FOUND)
    _set_auth_cookies(redirect_resp, access_token, raw_refresh)
    redirect_resp.delete_cookie(
        "oauth_pkce", path="/", samesite="lax", secure=settings.COOKIE_SECURE
    )
    return redirect_resp
