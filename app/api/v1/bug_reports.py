"""
User bug report API endpoints.
Allows authenticated users to submit and view their bug reports.
"""

import logging
from datetime import datetime

import pytz
from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.db.models import BugReport, User
from app.dependencies import AdminUser, DBSession, CurrentUser
from app.models.bug_report import (
    BugReportCreate,
    BugReportListResponse,
    BugReportResponse,
)

logger = logging.getLogger("mmara")

router = APIRouter(prefix="/bug-reports", tags=["Bug Reports"])


@router.post("", response_model=BugReportResponse, status_code=status.HTTP_201_CREATED)
async def create_bug_report(
    report: BugReportCreate,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Submit a bug report.

    - **title**: Brief summary of the bug (min 5 chars)
    - **description**: Detailed description (min 10 chars)
    - **bug_type**: Type of bug (ui, api, performance, accuracy, other)
    - **severity**: Severity level (low, medium, high, critical)
    - **steps_to_reproduce**: Optional steps to reproduce
    - **expected_behavior**: Optional expected behavior
    - **actual_behavior**: Optional actual behavior
    - **device_info**: Optional device information
    - **app_version**: Optional app version

    Requires authentication.
    """
    logger.info(f"Bug report submitted by user {current_user.id}: {report.title}")

    bug_report = BugReport(
        user_id=current_user.id,
        title=report.title,
        description=report.description,
        bug_type=report.bug_type,
        severity=report.severity,
        steps_to_reproduce=report.steps_to_reproduce,
        expected_behavior=report.expected_behavior,
        actual_behavior=report.actual_behavior,
        device_info=report.device_info,
        app_version=report.app_version,
        status="open",
    )

    db.add(bug_report)
    await db.commit()
    await db.refresh(bug_report)

    # Eagerly load the user relationship to avoid lazy loading issues
    await db.refresh(bug_report, attribute_names=["user"])

    return _build_response(bug_report, current_user)


@router.get("/my-reports", response_model=BugReportListResponse)
async def get_my_bug_reports(
    current_user: CurrentUser,
    db: DBSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    status_filter: str = Query(None, alias="status"),
):
    """
    Get current user's bug reports with pagination.

    - **page**: Page number (1-indexed)
    - **page_size**: Number of items per page (max 50)
    - **status**: Optional filter by status

    Requires authentication.
    """
    # Build base query
    base_query = select(BugReport).where(BugReport.user_id == current_user.id)

    # Apply status filter if provided
    if status_filter:
        valid_statuses = ["open", "in_progress", "resolved", "closed"]
        if status_filter not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        base_query = base_query.where(BugReport.status == status_filter)

    # Get total count
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Calculate pagination
    total_pages = (total + page_size - 1) // page_size
    offset = (page - 1) * page_size

    # Execute paginated query with eager loading
    query = base_query.options(
        selectinload(BugReport.user),
        selectinload(BugReport.assignee),
        selectinload(BugReport.responder),
    ).order_by(BugReport.created_at.desc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    bug_reports = result.scalars().all()

    items = [_build_response(br, current_user) for br in bug_reports]

    return BugReportListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{bug_id}", response_model=BugReportResponse)
async def get_bug_report_detail(
    bug_id: int,
    current_user: CurrentUser,
    db: DBSession,
):
    """
    Get details of a specific bug report.

    Users can only view their own bug reports. Admins can view all.
    """
    result = await db.execute(
        select(BugReport)
        .options(
            selectinload(BugReport.user),
            selectinload(BugReport.assignee),
            selectinload(BugReport.responder),
        )
        .where(BugReport.id == bug_id)
    )
    bug_report = result.scalar_one_or_none()

    if not bug_report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bug report not found"
        )

    # Check if user owns this report or is admin
    if bug_report.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this bug report"
        )

    return _build_response(bug_report, current_user)


def _build_response(bug_report: BugReport, current_user: User | None = None) -> BugReportResponse:
    """Build a BugReportResponse from a BugReport model."""
    # Get user info - use current_user if available and matches, otherwise try relationship
    user_email = None
    user_name = None
    if current_user and bug_report.user_id == current_user.id:
        user_email = current_user.email
        user_name = current_user.full_name
    elif bug_report.user_id and hasattr(bug_report, 'user') and bug_report.user:
        user_email = bug_report.user.email
        user_name = bug_report.user.full_name

    # Get assignee info - use relationship if loaded
    assignee_name = None
    if hasattr(bug_report, 'assignee') and bug_report.assignee:
        assignee_name = bug_report.assignee.full_name

    # Get admin responder info - use relationship if loaded
    admin_responded_by_name = None
    if hasattr(bug_report, 'responder') and bug_report.responder:
        admin_responded_by_name = bug_report.responder.full_name

    return BugReportResponse(
        id=bug_report.id,
        title=bug_report.title,
        description=bug_report.description,
        bug_type=bug_report.bug_type,
        severity=bug_report.severity,
        steps_to_reproduce=bug_report.steps_to_reproduce,
        expected_behavior=bug_report.expected_behavior,
        actual_behavior=bug_report.actual_behavior,
        device_info=bug_report.device_info,
        app_version=bug_report.app_version,
        status=bug_report.status,
        user_id=bug_report.user_id,
        user_email=user_email,
        user_name=user_name,
        assigned_to=bug_report.assigned_to,
        assignee_name=assignee_name,
        resolution_notes=bug_report.resolution_notes,
        admin_responded_at=bug_report.admin_responded_at,
        admin_responded_by=bug_report.admin_responded_by,
        admin_responded_by_name=admin_responded_by_name,
        created_at=bug_report.created_at,
        updated_at=bug_report.updated_at,
    )
