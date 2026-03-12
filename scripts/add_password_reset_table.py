"""
Migration script to add the password_resets table.
Run with: python -m scripts.add_password_reset_table
"""

import asyncio
from datetime import datetime
from sqlalchemy import text
from app.db.session import async_session_maker
from app.config import settings


async def upgrade():
    """Create the password_resets table."""
    async with async_session_maker() as session:
        # Create password_resets table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS password_resets (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token VARCHAR(255) UNIQUE NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
            used BOOLEAN DEFAULT FALSE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
        );
        """

        # Create indexes (must be run separately for asyncpg)
        create_index_1 = "CREATE INDEX IF NOT EXISTS idx_password_resets_user_id ON password_resets(user_id);"
        create_index_2 = "CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(token);"

        try:
            await session.execute(text(create_table_sql))
            await session.commit()  # Commit before indexes
            await session.execute(text(create_index_1))
            await session.commit()  # Commit each index separately
            await session.execute(text(create_index_2))
            await session.commit()
            print("✅ Successfully created password_resets table and indexes")
        except Exception as e:
            await session.rollback()
            print(f"❌ Error creating password_resets table: {e}")
            raise


async def downgrade():
    """Drop the password_resets table."""
    async with async_session_maker() as session:
        drop_table_sql = "DROP TABLE IF EXISTS password_resets CASCADE;"

        try:
            await session.execute(text(drop_table_sql))
            await session.commit()
            print("✅ Successfully dropped password_resets table")
        except Exception as e:
            await session.rollback()
            print(f"❌ Error dropping password_resets table: {e}")
            raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "down":
        asyncio.run(downgrade())
    else:
        asyncio.run(upgrade())
