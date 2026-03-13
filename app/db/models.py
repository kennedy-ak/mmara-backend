"""
SQLAlchemy database models.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.session import Base


class User(Base):
    """User model for authentication and profiles."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), onupdate=func.now()
    )

    # Relationships
    sessions: Mapped[list["ChatSession"]] = relationship(
        "ChatSession", back_populates="user", cascade="all, delete-orphan"
    )
    analytics: Mapped[list["Analytics"]] = relationship(
        "Analytics",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="[Analytics.user_id]",
    )


class ChatSession(Base):
    """Chat session for storing conversation history."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(50))
    messages: Mapped[dict] = mapped_column(JSON, default=list)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")


class Document(Base):
    """Model for uploaded legal documents."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    doc_metadata: Mapped[Optional[dict]] = mapped_column(JSON)
    uploaded_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Analytics(Base):
    """Analytics model for tracking usage and performance."""

    __tablename__ = "analytics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    session_id: Mapped[Optional[str]] = mapped_column(String(100))
    message_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    query_type: Mapped[str] = mapped_column(String(50))
    category: Mapped[Optional[str]] = mapped_column(String(50))
    urgency: Mapped[Optional[str]] = mapped_column(String(20))
    response_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    satisfaction: Mapped[Optional[int]] = mapped_column(Integer)  # 1-5 rating
    feedback: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Admin management fields
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    flagged_reason: Mapped[Optional[str]] = mapped_column(Text)
    admin_response: Mapped[Optional[str]] = mapped_column(Text)
    admin_responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    admin_responded_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))

    # Relationships
    user: Mapped["User"] = relationship(
        "User", back_populates="analytics", foreign_keys=[user_id]
    )
    responded_by_admin: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[admin_responded_by], backref="admin_responses"
    )


class RateLimit(Base):
    """Rate limit tracking."""

    __tablename__ = "rate_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identifier: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    request_count: Mapped[int] = mapped_column(Integer, default=0)
    window_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_request: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PasswordReset(Base):
    """Password reset token tracking."""

    __tablename__ = "password_resets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", backref="password_resets")


class BugReport(Base):
    """Model for user-submitted bug reports."""

    __tablename__ = "bug_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    session_id: Mapped[Optional[str]] = mapped_column(String(100))

    # Bug details
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    bug_type: Mapped[str] = mapped_column(String(50))  # ui, api, performance, accuracy, other
    severity: Mapped[str] = mapped_column(String(20))  # low, medium, high, critical

    # Additional context
    steps_to_reproduce: Mapped[Optional[str]] = mapped_column(Text)
    expected_behavior: Mapped[Optional[str]] = mapped_column(Text)
    actual_behavior: Mapped[Optional[str]] = mapped_column(Text)

    # Device info (auto-captured)
    device_info: Mapped[Optional[str]] = mapped_column(String(255))
    app_version: Mapped[Optional[str]] = mapped_column(String(50))

    # Admin management
    status: Mapped[str] = mapped_column(String(20), default="open")  # open, in_progress, resolved, closed
    assigned_to: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    admin_responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    admin_responded_by: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"))

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[user_id])
    assignee: Mapped["User"] = relationship("User", foreign_keys=[assigned_to])
    responder: Mapped["User"] = relationship("User", foreign_keys=[admin_responded_by])
