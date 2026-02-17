"""
Pytest configuration and fixtures.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest

from app.db.session import Base, async_session_maker, engine


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator:
    """Create a test database session."""
    # Create test tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async with async_session_maker() as session:
        yield session

    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    from app.core.security import get_password_hash
    from app.db.models import User

    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("Test123!"),
        full_name="Test User",
        role="user",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def admin_user(db_session):
    """Create an admin user."""
    from app.core.security import get_password_hash
    from app.db.models import User

    user = User(
        email="admin@example.com",
        hashed_password=get_password_hash("Admin123!"),
        full_name="Admin User",
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Get authentication headers for a test user."""
    from app.core.security import create_access_token

    token = create_access_token(test_user.id, {"role": test_user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(admin_user):
    """Get authentication headers for an admin user."""
    from app.core.security import create_access_token

    token = create_access_token(admin_user.id, {"role": admin_user.role})
    return {"Authorization": f"Bearer {token}"}
