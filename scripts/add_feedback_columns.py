"""
Add admin feedback management columns to analytics table.
Run this once to migrate the database schema.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.db.session import async_session_maker


async def migrate():
    """Add new columns to analytics table."""

    async with async_session_maker() as db:
        # Check if columns already exist
        check_sql = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'analytics'
            AND column_name IN ('flagged', 'flagged_reason', 'admin_response',
                               'admin_responded_at', 'admin_responded_by')
        """

        result = await db.execute(text(check_sql))
        existing_columns = {row[0] for row in result}

        columns_to_add = {
            'flagged': 'BOOLEAN DEFAULT FALSE NOT NULL',
            'flagged_reason': 'TEXT',
            'admin_response': 'TEXT',
            'admin_responded_at': 'TIMESTAMPTZ',
            'admin_responded_by': 'INTEGER REFERENCES users(id)',
        }

        for col_name, col_type in columns_to_add.items():
            if col_name in existing_columns:
                print(f"Column '{col_name}' already exists, skipping...")
                continue

            sql = f"ALTER TABLE analytics ADD COLUMN {col_name} {col_type}"
            print(f"Adding column: {col_name}")
            try:
                await db.execute(text(sql))
                print(f"  ✓ Added {col_name}")
            except Exception as e:
                print(f"  ✗ Error adding {col_name}: {e}")

        await db.commit()
        print("\nMigration complete!")


if __name__ == "__main__":
    asyncio.run(migrate())
