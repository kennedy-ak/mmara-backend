"""
Authentication API endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.core.security import create_tokens, get_password_hash, verify_password
from app.db.models import User
from app.dependencies import DBSession
from app.models.user import Token, User as UserResponse, UserCreate, UserLogin

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: DBSession):
    """
    Register a new user.

    - **email**: User email address
    - **password**: User password (min 8 characters, must contain uppercase, lowercase, and digit)
    - **full_name**: Optional full name
    - **phone**: Optional phone number
    """
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
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: DBSession):
    """
    Login with email and password (OAuth2 compatible).

    - **username**: Email address
    - **password**: User password
    """
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
async def login_json(credentials: UserLogin, db: DBSession):
    """
    Login with JSON body.

    - **email**: User email address
    - **password**: User password
    """
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


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str, db: DBSession):
    """
    Refresh access token using refresh token.

    - **refresh_token**: Valid refresh token
    """
    from app.core.security import verify_refresh_token

    # Verify refresh token
    user_id = verify_refresh_token(refresh_token)

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
async def get_current_user_info(current_user: Annotated[UserResponse, Depends()]):
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
