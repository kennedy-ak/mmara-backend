"""
User-related Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserBase(BaseModel):
    """Base user model with common fields."""

    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)


class UserCreate(UserBase):
    """Model for user registration."""

    password: str = Field(..., min_length=8, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Model for updating user profile."""

    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)


class UserInDB(UserBase):
    """Model for user stored in database."""

    id: int
    role: str = "user"
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class User(UserInDB):
    """Model for user response (excludes sensitive data)."""

    pass


class UserLogin(BaseModel):
    """Model for user login."""

    email: EmailStr
    password: str


class Token(BaseModel):
    """Model for JWT token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """Model for JWT token payload."""

    sub: str  # user_id
    exp: int
    iat: int
    role: str


class UserStats(BaseModel):
    """Model for user statistics."""

    total_queries: int
    queries_this_month: int
    avg_response_time: float
    satisfaction_score: Optional[float] = None
