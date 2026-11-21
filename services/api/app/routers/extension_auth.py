import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.core.security import hash_token
from app.models.extension_token import ExtensionToken
from app.models.user import User
from app.schemas.auth import MessageResponse
from app.schemas.extension_token import (
    ExtensionTokenCreate,
    ExtensionTokenInfo,
    ExtensionTokenResponse,
)

router = APIRouter(prefix="/auth/extension-token", tags=["Extension Auth"])


@router.post(
    "",
    response_model=ExtensionTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate scoped Bearer token for Chrome Extension",
)
async def generate_extension_token(
    payload: ExtensionTokenCreate | None = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> ExtensionTokenResponse:
    """Generates a secure Bearer token for the Chrome Extension.

    Stores the SHA-256 hash in the database and returns the raw token to the
    authenticated web client once.
    """
    token_name = payload.name if payload and payload.name else "Chrome Extension"
    raw_token = f"cgt_{secrets.token_urlsafe(32)}"
    token_hash_val = hash_token(raw_token)

    now = datetime.now(UTC)
    token_record = ExtensionToken(
        user_id=current_user.id,
        token_hash=token_hash_val,
        name=token_name,
        created_at=now,
    )
    db.add(token_record)
    await db.commit()
    await db.refresh(token_record)

    return ExtensionTokenResponse(
        token=raw_token,
        name=token_record.name,
        created_at=token_record.created_at,
    )


@router.delete(
    "",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Revoke active Chrome Extension token(s)",
)
async def revoke_extension_tokens(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MessageResponse:
    """Revokes all active extension tokens for the authenticated user."""
    now = datetime.now(UTC)
    await db.execute(
        update(ExtensionToken)
        .where(
            ExtensionToken.user_id == current_user.id,
            ExtensionToken.revoked_at.is_(None),
        )
        .values(revoked_at=now)
    )
    await db.commit()

    return MessageResponse(message="Extension token(s) revoked successfully.")


@router.get(
    "/status",
    response_model=list[ExtensionTokenInfo],
    status_code=status.HTTP_200_OK,
    summary="Get status of extension tokens",
)
async def get_extension_token_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> list[ExtensionTokenInfo]:
    """Returns metadata for the user's extension tokens."""
    result = await db.execute(
        select(ExtensionToken)
        .where(ExtensionToken.user_id == current_user.id)
        .order_by(ExtensionToken.created_at.desc())
        .limit(10)
    )
    tokens = result.scalars().all()
    return [
        ExtensionTokenInfo(
            id=t.id,
            name=t.name,
            last_used_at=t.last_used_at,
            created_at=t.created_at,
            is_active=t.revoked_at is None,
        )
        for t in tokens
    ]
