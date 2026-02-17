"""
Dependency injection for FastAPI.
Provides database sessions, authentication, and service instances.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import AgentOrchestrator
from app.config import settings
from app.core.security import decode_token
from app.db.session import async_session_maker
from app.models.user import UserInDB
from app.services.embeddings import EmbeddingService
from app.services.groq_client import GroqClient
from app.services.redis_client import RedisService
from app.services.retrieval import RetrievalService

# HTTP Bearer token security
security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session.

    Yields:
        AsyncSession: SQLAlchemy async session
    """
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: AsyncSession = Depends(get_db),
) -> UserInDB:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials
        db: Database session

    Returns:
        UserInDB: Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Decode token
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    from sqlalchemy import select

    from app.db.models import User

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserInDB.model_validate(user)


async def get_current_active_user(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
) -> UserInDB:
    """
    Get current active user.

    Args:
        current_user: Current user

    Returns:
        UserInDB: Current active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return current_user


async def require_admin(
    current_user: Annotated[UserInDB, Depends(get_current_active_user)],
) -> UserInDB:
    """
    Require admin role.

    Args:
        current_user: Current user

    Returns:
        UserInDB: Current admin user

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user


# Service singletons (could be improved with proper DI container)
_embedding_service: EmbeddingService | None = None
_retrieval_service: RetrievalService | None = None
_groq_client: GroqClient | None = None
_redis_service: RedisService | None = None
_orchestrator: AgentOrchestrator | None = None


async def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(
            persist_directory=settings.chroma_persist_directory_resolved,
            embedding_model=settings.embedding_model,
        )
    return _embedding_service


async def get_retrieval_service(
    embedding_service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> RetrievalService:
    """Get or create retrieval service singleton."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService(embedding_service=embedding_service)
    return _retrieval_service


async def get_groq_client() -> GroqClient:
    """Get or create Groq client singleton."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)
    return _groq_client


async def get_redis_service() -> RedisService:
    """Get or create Redis service singleton."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService(url=settings.redis_url)
    return _redis_service


async def get_orchestrator(
    retrieval_service: Annotated[RetrievalService, Depends(get_retrieval_service)],
    groq_client: Annotated[GroqClient, Depends(get_groq_client)],
    redis_service: Annotated[RedisService, Depends(get_redis_service)],
) -> AgentOrchestrator:
    """Get or create agent orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(
            retrieval_service=retrieval_service,
            groq_client=groq_client,
            redis_service=redis_service,
        )
    return _orchestrator


# Type aliases for cleaner dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[UserInDB, Depends(get_current_active_user)]
AdminUser = Annotated[UserInDB, Depends(require_admin)]
EmbeddingSvc = Annotated[EmbeddingService, Depends(get_embedding_service)]
RetrievalSvc = Annotated[RetrievalService, Depends(get_retrieval_service)]
GroqSvc = Annotated[GroqClient, Depends(get_groq_client)]
RedisSvc = Annotated[RedisService, Depends(get_redis_service)]
OrchestratorSvc = Annotated[AgentOrchestrator, Depends(get_orchestrator)]
