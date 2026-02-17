"""
User management API endpoints.
"""

from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from app.db.models import Analytics, User
from app.dependencies import AdminUser, CurrentUser, DBSession
from app.models.user import User as UserResponse, UserStats, UserUpdate

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def get_profile(current_user: Annotated[UserResponse, Depends(CurrentUser)]):
    """
    Get current user profile.
    """
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_profile(
    updates: UserUpdate, current_user: Annotated[UserResponse, Depends(CurrentUser)], db: DBSession
):
    """
    Update current user profile.

    - **full_name**: New full name (optional)
    - **phone**: New phone number (optional)
    """
    # Get user from database
    result = await db.execute(select(User).where(User.id == current_user.id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Update fields
    if updates.full_name is not None:
        user.full_name = updates.full_name
    if updates.phone is not None:
        user.phone = updates.phone

    await db.commit()
    await db.refresh(user)

    return UserResponse.model_validate(user)


@router.get("/me/stats", response_model=UserStats)
async def get_user_stats(
    current_user: Annotated[UserResponse, Depends(CurrentUser)], db: DBSession
):
    """
    Get current user statistics.
    """
    from datetime import datetime

    from sqlalchemy import func

    # Total queries
    total_result = await db.execute(
        select(func.count(Analytics.id)).where(Analytics.user_id == current_user.id)
    )
    total_queries = total_result.scalar() or 0

    # Queries this month
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_result = await db.execute(
        select(func.count(Analytics.id)).where(
            Analytics.user_id == current_user.id, Analytics.created_at >= month_start
        )
    )
    queries_this_month = month_result.scalar() or 0

    # Average response time
    avg_time_result = await db.execute(
        select(func.avg(Analytics.response_time_ms)).where(Analytics.user_id == current_user.id)
    )
    avg_response_time = avg_time_result.scalar() or 0

    # Average satisfaction
    satisfaction_result = await db.execute(
        select(func.avg(Analytics.satisfaction)).where(
            Analytics.user_id == current_user.id, Analytics.satisfaction.isnot(None)
        )
    )
    satisfaction = satisfaction_result.scalar()

    return UserStats(
        total_queries=total_queries,
        queries_this_month=queries_this_month,
        avg_response_time=float(avg_response_time),
        satisfaction_score=float(satisfaction) if satisfaction else None,
    )


@router.get("/", response_model=List[UserResponse])
async def list_users(
    current_user: Annotated[UserResponse, Depends(AdminUser)],
    db: DBSession,
    skip: int = 0,
    limit: int = 50,
):
    """
    List all users (Admin only).

    - **skip**: Number of users to skip
    - **limit**: Maximum number of users to return
    """
    result = await db.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all()

    return [UserResponse.model_validate(u) for u in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int, current_user: Annotated[UserResponse, Depends(AdminUser)], db: DBSession
):
    """
    Get a specific user by ID (Admin only).
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse.model_validate(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: int, current_user: Annotated[UserResponse, Depends(AdminUser)], db: DBSession
):
    """
    Delete a user (Admin only).
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete your own account"
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    await db.delete(user)
    await db.commit()

    return {"message": "User deleted successfully"}
