import base64
import hashlib
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import argon2
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.core.config import settings

# Argon2id password hasher with production-recommended parameters
ph = PasswordHasher(
    time_cost=2,
    memory_cost=19456,  # 19 MiB
    parallelism=1,
    hash_len=32,
    type=argon2.Type.ID,
)

# Dummy hash for constant-time failed password timing defense (user enumeration mitigation)
DUMMY_HASH = ph.hash("CognitwinConstantTimeDummyPassword987!")


def hash_password(password: str) -> str:
    """Hashes a password using Argon2id."""
    return ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against an Argon2id hash in constant time."""
    try:
        return ph.verify(hashed_password, plain_password)
    except (VerifyMismatchError, Exception):
        return False


def verify_dummy_password(plain_password: str) -> None:
    """Consumes equivalent CPU time when an email does not exist to prevent timing attacks."""
    try:
        ph.verify(DUMMY_HASH, plain_password)
    except Exception:
        pass


def validate_password_strength(password: str) -> tuple[bool, str | None]:
    """Validates that a password satisfies strong complexity criteria.

    - Minimum 10 characters
    - At least one uppercase letter (A-Z)
    - At least one lowercase letter (a-z)
    - At least one numeric digit (0-9)
    - At least one special symbol
    """
    if len(password) < 10:
        return False, "Password must be at least 10 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r"\d", password):
        return False, "Password must include at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>\-_=+[\];/]", password):
        return False, "Password must include at least one special character."
    return True, None


def hash_token(raw_token: str) -> str:
    """Computes SHA-256 hash of a raw token for secure database storage."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_secure_random_token(nbytes: int = 32) -> str:
    """Generates a cryptographically strong URL-safe random string."""
    return secrets.token_urlsafe(nbytes)


def create_access_token(
    user_id: uuid.UUID | str,
    role: str,
    email: str,
    expires_delta: timedelta | None = None,
) -> str:
    """Creates a signed, short-lived JWT access token."""
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "email": email,
        "exp": expire,
        "iat": datetime.now(UTC),
        "type": "access",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decodes and cryptographically verifies an access token."""
    return jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub", "role"]},
    )


def generate_pkce_code_verifier() -> str:
    """Generates a cryptographic PKCE code_verifier (RFC 7636)."""
    return secrets.token_urlsafe(64)


def generate_pkce_code_challenge(verifier: str) -> str:
    """Computes the S256 PKCE code_challenge from code_verifier."""
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def generate_oauth_state() -> str:
    """Generates a secure state parameter to mitigate CSRF in OAuth flows."""
    return secrets.token_urlsafe(32)
