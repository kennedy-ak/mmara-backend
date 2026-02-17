"""
Security utilities for authentication and authorization.
Includes JWT token handling and password hashing.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password

    Returns:
        bool: True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.

    Args:
        password: Plain text password

    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


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

    return {"access_token": access_token, "refresh_token": refresh_token}
