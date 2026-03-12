"""
Security utilities for authentication and authorization.
Includes JWT token handling and password hashing.
"""

import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import bcrypt
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

# Standard bcrypt context for backwards compatibility
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Password reset token expiration (in minutes)
PASSWORD_RESET_EXPIRE_MINUTES = 30


def _prehash_password(password: str) -> bytes:
    """
    Pre-hash password with SHA-256 to support passwords longer than 72 bytes.
    This allows users to have passwords of any length while keeping bcrypt's security.
    """
    return hashlib.sha256(password.encode('utf-8')).digest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    Uses SHA-256 pre-hashing to support passwords longer than 72 bytes.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        bool: True if password matches
    """
    # Check if this is a new-style hash (base64 encoded SHA-256 + bcrypt)
    if hashed_password.startswith('$sha256$'):
        # Format: $sha256$bcrypt_hash
        bcrypt_hash = hashed_password.split('$', 2)[2]
        prehashed = _prehash_password(plain_password)
        return bcrypt.checkpw(prehashed, bcrypt_hash.encode('utf-8'))
    else:
        # Legacy direct bcrypt (for backwards compatibility)
        try:
            # Try normal bcrypt first
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            # If that fails, try with pre-hashing for long passwords
            prehashed = base64.b64encode(_prehash_password(plain_password)).decode('utf-8')
            return pwd_context.verify(prehashed, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt with SHA-256 pre-hashing.
    This supports passwords of any length while maintaining security.

    Args:
        password: Plain text password

    Returns:
        str: Hashed password with prefix $sha256$
    """
    # Pre-hash with SHA-256 to handle any password length
    prehashed = _prehash_password(password)

    # Hash the pre-hash with bcrypt
    bcrypt_hash = bcrypt.hashpw(prehashed, bcrypt.gensalt(rounds=12))

    # Store with prefix to identify this hashing method
    return f"$sha256${bcrypt_hash.decode('utf-8')}"


def create_access_token(
    subject: str | int,
    claims: Optional[Dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: Subject (usually user_id)
        claims: Additional claims to include
        expires_delta: Custom expiration time

    Returns:
        str: Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode = {"sub": str(subject), "exp": expire, "iat": datetime.utcnow(), "type": "access"}

    if claims:
        to_encode.update(claims)

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

    return encoded_jwt


def create_refresh_token(subject: str | int) -> str:
    """
    Create a JWT refresh token.

    Args:
        subject: Subject (usually user_id)

    Returns:
        str: Encoded JWT refresh token
    """
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)

    to_encode = {"sub": str(subject), "exp": expire, "iat": datetime.utcnow(), "type": "refresh"}

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

    return encoded_jwt


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Optional[Dict]: Decoded payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError:
        return None


def verify_refresh_token(token: str) -> Optional[str]:
    """
    Verify a refresh token and return the user_id.

    Args:
        token: Refresh token

    Returns:
        Optional[str]: User ID if valid, None otherwise
    """
    payload = decode_token(token)
    if payload and payload.get("type") == "refresh":
        return payload.get("sub")
    return None


def create_tokens(
    user_id: int, role: str = "user", additional_claims: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Create both access and refresh tokens for a user.

    Args:
        user_id: User ID
        role: User role
        additional_claims: Additional claims to include

    Returns:
        Dict with access_token and refresh_token
    """
    claims = {"role": role}
    if additional_claims:
        claims.update(additional_claims)

    access_token = create_access_token(user_id, claims=claims)
    refresh_token = create_refresh_token(user_id)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": settings.access_token_expire_minutes * 60,
    }


def generate_password_reset_token() -> str:
    """
    Generate a secure random token for password reset.

    Returns:
        str: URL-safe random token (43 characters = 256 bits of entropy)
    """
    return secrets.token_urlsafe(32)


def create_password_reset_token_jwt(user_id: int) -> str:
    """
    Create a JWT token for password reset.

    Args:
        user_id: User ID requesting password reset

    Returns:
        str: Encoded JWT password reset token
    """
    expire = datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_EXPIRE_MINUTES)

    to_encode = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "password_reset",
    }

    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[int]:
    """
    Verify a password reset token and return the user_id.

    Args:
        token: Password reset token (JWT or random token)

    Returns:
        Optional[int]: User ID if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        if payload.get("type") == "password_reset":
            return int(payload.get("sub"))
    except JWTError:
        pass
    return None


def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """
    Validate password strength requirements.

    Args:
        password: Password to validate

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_errors)
    """
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if len(password) > 128:
        errors.append("Password must not exceed 128 characters")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")

    return (len(errors) == 0, errors)
