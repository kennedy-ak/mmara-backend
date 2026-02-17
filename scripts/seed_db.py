"""
Database Seeding Script
Initialize the database with default data.
"""

import asyncio

from app.core.security import get_password_hash
from app.db.models import User
from app.db.session import async_session_maker, init_db


async def create_admin_user(email: str = "admin@mmara.gh", password: str = "Admin123!"):
    """Create the default admin user."""
    async with async_session_maker() as session:
        # Check if admin exists
        from sqlalchemy import select

        result = await session.execute(select(User).where(User.email == email))
        existing_admin = result.scalar_one_or_none()

        if existing_admin:
            print(f"Admin user already exists: {email}")
            return existing_admin

        # Create admin user
        admin = User(
            email=email,
            hashed_password=get_password_hash(password),
            full_name="MMara Administrator",
            role="admin",
            is_active=True,
            is_premium=True,
        )

        session.add(admin)
        await session.commit()
        await session.refresh(admin)

        print(f"Created admin user: {email} / {password}")
        return admin


async def create_test_user(email: str = "test@mmara.gh", password: str = "Test123!"):
    """Create a test user."""
    async with async_session_maker() as session:
        from sqlalchemy import select

        result = await session.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            print(f"Test user already exists: {email}")
            return existing_user

        # Create test user
        test_user = User(
            email=email,
            hashed_password=get_password_hash(password),
            full_name="Test User",
            role="user",
            is_active=True,
        )

        session.add(test_user)
        await session.commit()
        await session.refresh(test_user)

        print(f"Created test user: {email} / {password}")
        return test_user


async def main():
    """Main seeding function."""
    print("Seeding MMara database...")

    # Initialize database
    await init_db()
    print("Database tables created.")

    # Create users
    await create_admin_user()
    await create_test_user()

    print("\nSeeding complete!")
    print("\nDefault credentials:")
    print("  Admin: admin@mmara.gh / Admin123!")
    print("  Test:  test@mmara.gh / Test123!")
    print("\nPlease change these passwords in production!")


if __name__ == "__main__":
    asyncio.run(main())
