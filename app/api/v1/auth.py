"""
Authentication API endpoints.
"""

from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.sql import func

from app.core.rate_limiter import auth_rate_limit, strict_rate_limit
from app.core.security import (
    create_password_reset_token_jwt,
    create_tokens,
    get_password_hash,
    verify_password,
    verify_password_reset_token,
)
from app.db.models import PasswordReset, User
from app.dependencies import CurrentUser, DBSession, RedisSvc, get_current_active_user
from app.models.user import (
    ChangePasswordRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    PasswordResetResponse,
    Token,
    User as UserResponse,
    UserCreate,
    UserLogin,
)
from app.services.email_service import email_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    user_data: UserCreate,
    db: DBSession,
    redis: RedisSvc,
):
    """
    Register a new user.

    - **email**: User email address
    - **password**: User password (min 8 characters, must contain uppercase, lowercase, and digit)
    - **full_name**: Optional full name
    - **phone**: Optional phone number
    """
    # Apply rate limiting
    from app.core.rate_limiter import RateLimiter
    limiter = RateLimiter(redis, max_requests=3, window_seconds=3600)
    await limiter.check(request)
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
        )

    # Create new user
    new_user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        phone=user_data.phone,
        role="user",
        is_active=True,
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DBSession,
    redis: RedisSvc,
):
    """
    Login with email and password (OAuth2 compatible).

    - **username**: Email address
    - **password**: User password
    """
    # Apply rate limiting
    from app.core.rate_limiter import RateLimiter
    limiter = RateLimiter(redis, max_requests=5, window_seconds=300)
    await limiter.check(request)
    # Find user
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    # Verify user and password
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    # Create tokens
    tokens = create_tokens(user.id, user.role)

    return tokens


@router.post("/login/json", response_model=Token)
async def login_json(
    request: Request,
    credentials: UserLogin,
    db: DBSession,
    redis: RedisSvc,
):
    """
    Login with JSON body.

    - **email**: User email address
    - **password**: User password
    """
    # Apply rate limiting
    from app.core.rate_limiter import RateLimiter
    limiter = RateLimiter(redis, max_requests=5, window_seconds=300)
    await limiter.check(request)
    # Find user
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()

    # Verify user and password
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    # Create tokens
    tokens = create_tokens(user.id, user.role)

    return tokens


class RefreshTokenRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=Token)
async def refresh_token(db: DBSession, body: RefreshTokenRequest):
    """
    Refresh access token using refresh token.

    - **refresh_token**: Valid refresh token
    """
    from app.core.security import verify_refresh_token

    # Verify refresh token
    user_id = verify_refresh_token(body.refresh_token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user")

    # Create new tokens
    tokens = create_tokens(user.id, user.role)

    return tokens


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: CurrentUser):
    """
    Get current user information.
    Requires authentication.
    """
    return current_user


@router.post("/logout")
async def logout():
    """
    Logout user.

    Note: With JWT tokens, logout is handled client-side by deleting the token.
    This endpoint is provided for future token blacklisting functionality.
    """
    return {"message": "Successfully logged out"}


@router.post("/password-reset/request", response_model=PasswordResetResponse)
async def request_password_reset(
    http_request: Request,
    request: PasswordResetRequest,
    db: DBSession,
    redis: RedisSvc,
):
    """
    Request a password reset email.

    - **email**: Email address associated with the account

    Security: Always returns success to prevent email enumeration.
    """
    # Apply rate limiting (3 attempts per hour)
    from app.core.rate_limiter import RateLimiter
    limiter = RateLimiter(redis, max_requests=3, window_seconds=3600)
    await limiter.check(http_request)
    # Find user by email
    result = await db.execute(select(User).where(User.email == request.email))
    user = result.scalar_one_or_none()

    if not user:
        # Don't reveal if email exists or not (security best practice)
        return PasswordResetResponse(
            message="If an account with this email exists, a password reset link has been sent."
        )

    if not user.is_active:
        # Same message for inactive users
        return PasswordResetResponse(
            message="If an account with this email exists, a password reset link has been sent."
        )

    # Check if there's a recent unused reset token (prevent spam)
    recent_time = datetime.utcnow() - timedelta(minutes=5)
    recent_reset = await db.execute(
        select(PasswordReset)
        .where(
            PasswordReset.user_id == user.id,
            PasswordReset.used == False,
            PasswordReset.created_at >= recent_time,
        )
        .order_by(PasswordReset.created_at.desc())
    )
    existing_reset = recent_reset.scalar_one_or_none()

    if existing_reset:
        # Token exists and hasn't expired yet, don't create a new one
        await email_service.send_password_reset_email(
            email=user.email,
            reset_token=existing_reset.token,
            user_name=user.full_name,
        )
        return PasswordResetResponse(
            message="If an account with this email exists, a password reset link has been sent."
        )

    # Create new password reset token
    reset_token = create_password_reset_token_jwt(user.id)
    expires_at = datetime.utcnow() + timedelta(minutes=30)

    # Store token in database
    db_reset = PasswordReset(
        user_id=user.id,
        token=reset_token,
        expires_at=expires_at,
        used=False,
    )
    db.add(db_reset)
    await db.flush()  # Flush to get the ID

    # Mark old unused tokens as used (invalidate them)
    from sqlalchemy import update
    await db.execute(
        update(PasswordReset)
        .where(
            PasswordReset.user_id == user.id,
            PasswordReset.used == False,
            PasswordReset.id != db_reset.id,
        )
        .values(used=True)
    )
    await db.commit()

    # Send email
    await email_service.send_password_reset_email(
        email=user.email,
        reset_token=reset_token,
        user_name=user.full_name,
    )

    return PasswordResetResponse(
        message="If an account with this email exists, a password reset link has been sent."
    )


@router.post("/password-reset/confirm", response_model=PasswordResetResponse)
async def confirm_password_reset(reset_data: PasswordResetConfirm, db: DBSession):
    """
    Confirm password reset with token and new password.

    - **token**: Password reset token from email
    - **new_password**: New password (min 8 characters, uppercase, lowercase, digit)
    """
    # Verify token
    user_id = verify_password_reset_token(reset_data.token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user",
        )

    # Find the reset token record
    token_result = await db.execute(
        select(PasswordReset).where(
            PasswordReset.token == reset_data.token,
            PasswordReset.user_id == user.id,
            PasswordReset.used == False,
        )
    )
    reset_record = token_result.scalar_one_or_none()

    if not reset_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Check if token has expired
    if reset_record.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired",
        )

    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    user.updated_at = datetime.utcnow()

    # Mark token as used
    reset_record.used = True

    await db.commit()

    # Send confirmation email
    await email_service.send_password_changed_confirmation(
        email=user.email,
        user_name=user.full_name,
    )

    return PasswordResetResponse(
        message="Password has been reset successfully. You can now login with your new password."
    )


@router.post("/change-password", response_model=PasswordResetResponse)
async def change_password(
    password_data: ChangePasswordRequest,
    db: DBSession,
    current_user: CurrentUser,
):
    """
    Change password for authenticated user.

    - **old_password**: Current password
    - **new_password**: New password (min 8 characters, uppercase, lowercase, digit)
    """
    # Get the actual User model from database
    result = await db.execute(select(User).where(User.id == current_user.id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Verify old password
    if not verify_password(password_data.old_password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    user.hashed_password = get_password_hash(password_data.new_password)
    user.updated_at = datetime.utcnow()

    await db.commit()

    # Send confirmation email
    await email_service.send_password_changed_confirmation(
        email=user.email,
        user_name=user.full_name,
    )

    return PasswordResetResponse(
        message="Password changed successfully. Please use your new password for future logins."
    )
