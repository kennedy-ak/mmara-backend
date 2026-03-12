"""
Cleanup script for expired password reset tokens.
Run this periodically (e.g., via cron) to remove expired tokens.
Usage: python -m scripts.cleanup_expired_tokens
"""

import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select, delete
from app.db.session import async_session_maker
from app.db.models import PasswordReset


async def cleanup_expired_tokens():
    """Delete expired and used password reset tokens older than 7 days."""
    async with async_session_maker() as session:
        # Delete expired tokens older than 7 days
        cutoff_date = datetime.utcnow() - timedelta(days=7)

        # Delete expired tokens
        delete_expired = delete(PasswordReset).where(
            PasswordReset.expires_at < cutoff_date
        )
        result = await session.execute(delete_expired)

        # Also delete used tokens older than 1 day
        cutoff_used = datetime.utcnow() - timedelta(days=1)
        delete_used = delete(PasswordReset).where(
            PasswordReset.used == True,
            PasswordReset.created_at < cutoff_used
        )
        result_used = await session.execute(delete_used)

        await session.commit()

        # Get count of remaining tokens
        count_result = await session.execute(
            select(PasswordReset).where(PasswordReset.used == False)
        )
        remaining = len(count_result.scalars().all())

        print(f"✅ Cleanup complete. {result.rowcount} expired tokens removed.")
        print(f"✅ {result_used.rowcount} used tokens removed.")
        print(f"📊 {remaining} active tokens remaining.")


if __name__ == "__main__":
    asyncio.run(cleanup_expired_tokens())
