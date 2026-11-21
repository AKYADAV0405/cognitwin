import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token, hash_token
from app.models.extension_token import ExtensionToken
from app.models.user import User

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_active_user(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extracts access token from HTTP-only cookie or Authorization header.

    Loads the active user from the database.
    """
    token: str | None = None

    # Priority 1: HTTP-only cookie (Web client default)
    if "access_token" in request.cookies:
        token = request.cookies.get("access_token")
    # Priority 2: Authorization: Bearer token (CLI / Chrome Extension)
    elif bearer and bearer.credentials:
        token = bearer.credentials

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: No access token provided.",
        )

    try:
        payload = decode_access_token(token)
        user_id_str: str = payload.get("sub", "")
        user_id = uuid.UUID(user_id_str)
    except (jwt.PyJWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
        ) from None

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found or deactivated.",
        )

    return user


def require_role(required_role: str) -> Callable[..., User]:
    """Dependency factory enforcing Role-Based Access Control (RBAC).

    - 'student': accessible by students and researchers.
    - 'researcher': strictly accessible only by users with role 'researcher'.
    """

    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if required_role == "researcher" and current_user.role != "researcher":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: researcher role required.",
            )
        return current_user

    return role_checker


async def get_current_extension_user(
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> tuple[User, ExtensionToken]:
    """Validates scoped Bearer API token from Chrome extension against hashed tokens."""
    if not bearer or not bearer.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Extension authentication required: Missing Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_token = bearer.credentials
    token_hash_val = hash_token(raw_token)

    result = await db.execute(
        select(ExtensionToken, User)
        .join(User, ExtensionToken.user_id == User.id)
        .where(
            ExtensionToken.token_hash == token_hash_val,
            ExtensionToken.revoked_at.is_(None),
            User.is_active.is_(True),
        )
    )
    row = result.first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked extension token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    ext_token, user = row
    ext_token.last_used_at = datetime.now(UTC)
    await db.commit()

    return user, ext_token
