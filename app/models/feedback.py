"""
Feedback management models for admin dashboard.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FeedbackItem(BaseModel):
    """Single feedback item with user and context info."""

    id: int
    user_id: int
    user_email: str
    user_name: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    query_type: str
    category: Optional[str] = None
    satisfaction: Optional[int] = None  # 1-5 rating
    feedback: Optional[str] = None  # User comment
    message_content: Optional[str] = None  # Original user query
    response_content: Optional[str] = None  # AI response
    flagged: bool = False
    flagged_reason: Optional[str] = None
    admin_response: Optional[str] = None
    admin_responded_at: Optional[datetime] = None
    admin_responded_by: Optional[int] = None
    admin_responded_by_name: Optional[str] = None
    created_at: datetime


class FeedbackListResponse(BaseModel):
    """Paginated list of feedback items."""

    items: list[FeedbackItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class FeedbackDetailResponse(BaseModel):
    """Full feedback detail with conversation history."""

    id: int
    user_id: int
    user_email: str
    user_name: Optional[str] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    query_type: str
    category: Optional[str] = None
    urgency: Optional[str] = None
    satisfaction: Optional[int] = None
    feedback: Optional[str] = None
    message_content: Optional[str] = None
    response_content: Optional[str] = None
    conversation_history: Optional[list[dict]] = None
    response_time_ms: Optional[float] = None
    retrieval_count: Optional[int] = None
    is_emergency: Optional[bool] = None
    flagged: bool = False
    flagged_reason: Optional[str] = None
    admin_response: Optional[str] = None
    admin_responded_at: Optional[datetime] = None
    admin_responded_by: Optional[int] = None
    admin_responded_by_name: Optional[str] = None
    created_at: datetime


class FeedbackStats(BaseModel):
    """Feedback statistics for dashboard overview."""

    total_feedback: int
    average_rating: Optional[float] = None
    rating_distribution: dict[int, int] = Field(default_factory=dict)
    flagged_count: int
    by_category: dict[str, int] = Field(default_factory=dict)
    recent_count: int  # Last 7 days


class FlagFeedbackRequest(BaseModel):
    """Request to flag/unflag feedback."""

    flagged: bool = Field(..., description="Whether to flag the feedback")
    reason: Optional[str] = Field(None, max_length=1000, description="Reason for flagging")


class AdminResponseRequest(BaseModel):
    """Request to respond to user feedback."""

    message: str = Field(..., min_length=1, max_length=5000, description="Admin response message")


class FeedbackExportParams(BaseModel):
    """Parameters for feedback export."""

    format: str = Field("csv", description="Export format: csv or json")
    date_from: Optional[str] = Field(None, description="ISO date string (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="ISO date string (YYYY-MM-DD)")
    category: Optional[str] = Field(None, description="Filter by category")
    min_rating: Optional[int] = Field(None, ge=1, le=5, description="Minimum rating")
    max_rating: Optional[int] = Field(None, ge=1, le=5, description="Maximum rating")
    flagged_only: bool = Field(False, description="Only include flagged feedback")
