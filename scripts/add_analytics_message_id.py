"""
Migration script to add message_id column to analytics table.
Run with: python scripts/add_analytics_message_id.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.db.session import engine


async def upgrade():
    """Add message_id column to analytics table."""
    async with engine.begin() as conn:
        # Add column if it doesn't exist
        await conn.execute(text(
            "ALTER TABLE analytics ADD COLUMN IF NOT EXISTS message_id VARCHAR(100)"
        ))
        # Create index if it doesn't exist
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_analytics_message_id ON analytics(message_id)"
        ))
        print("✅ Migration complete: Added message_id column and index to analytics table")


async def downgrade():
    """Remove message_id column from analytics table."""
    async with engine.begin() as conn:
        await conn.execute(text("DROP INDEX IF EXISTS ix_analytics_message_id"))
        await conn.execute(text("ALTER TABLE analytics DROP COLUMN IF EXISTS message_id"))
        print("✅ Rollback complete: Removed message_id column from analytics table")


if __name__ == "__main__":
    import sys
    asyncio.run(upgrade() if len(sys.argv) <= 1 or sys.argv[1] != "downgrade" else downgrade())
