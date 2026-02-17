"""
Chat-related Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Single message in a conversation."""

    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Citation(BaseModel):
    """Legal citation reference."""

    act: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    text: str
    source_file: Optional[str] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    category: Optional[str] = Field(None, pattern="^(criminal|road_traffic|general)$")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    session_id: str
    message_id: str
    citations: List[Citation] = []
    confidence: float = Field(ge=0.0, le=1.0)
    category: str
    urgency: str
    is_emergency: bool = False
    disclaimer: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None


class ChatHistory(BaseModel):
    """Model for chat history."""

    session_id: str
    user_id: int
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


class ChatFeedback(BaseModel):
    """Model for user feedback on chat response."""

    message_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    helpful: bool


class StreamingChunk(BaseModel):
    """Model for streaming response chunk."""

    delta: str
    done: bool = False
    citation: Optional[Citation] = None
