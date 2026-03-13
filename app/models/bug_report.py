"""
Bug report models for user submissions and admin management.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BugReportCreate(BaseModel):
    """Request model for creating a bug report."""

    title: str = Field(..., min_length=5, max_length=255, description="Brief summary of the bug")
    description: str = Field(..., min_length=10, description="Detailed description of the bug")
    bug_type: str = Field(..., pattern="^(ui|api|performance|accuracy|other)$", description="Type of bug")
    severity: str = Field(..., pattern="^(low|medium|high|critical)$", description="Severity level")
    steps_to_reproduce: Optional[str] = Field(None, max_length=2000, description="Steps to reproduce the bug")
    expected_behavior: Optional[str] = Field(None, max_length=1000, description="Expected behavior")
    actual_behavior: Optional[str] = Field(None, max_length=1000, description="Actual behavior observed")
    device_info: Optional[str] = Field(None, max_length=255, description="Device information")
    app_version: Optional[str] = Field(None, max_length=50, description="App version")


class BugReportResponse(BaseModel):
    """Response model for bug report."""

    id: int
    title: str
    description: str
    bug_type: str
    severity: str
    steps_to_reproduce: Optional[str]
    expected_behavior: Optional[str]
    actual_behavior: Optional[str]
    device_info: Optional[str]
    app_version: Optional[str]
    status: str
    user_id: Optional[int]
    user_email: Optional[str]
    user_name: Optional[str]
    assigned_to: Optional[int]
    assignee_name: Optional[str]
    resolution_notes: Optional[str]
    admin_responded_at: Optional[datetime]
    admin_responded_by: Optional[int]
    admin_responded_by_name: Optional[str]
    created_at: datetime
    updated_at: datetime


class BugReportListResponse(BaseModel):
    """Response model for bug report list."""

    items: list[BugReportResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class BugReportUpdate(BaseModel):
    """Request model for updating bug report status."""

    status: str = Field(..., pattern="^(open|in_progress|resolved|closed)$", description="New status")
    resolution_notes: Optional[str] = Field(None, max_length=2000, description="Resolution notes")
    assigned_to: Optional[int] = Field(None, description="Admin user ID to assign to")


class BugStats(BaseModel):
    """Bug statistics for dashboard."""

    total_bugs: int
    open_bugs: int
    in_progress_bugs: int
    resolved_bugs: int
    closed_bugs: int
    critical_bugs: int
    high_bugs: int
    medium_bugs: int
    low_bugs: int
    by_severity: dict[str, int]
    by_type: dict[str, int]
    by_status: dict[str, int]
    recent_count: int  # Last 7 days
