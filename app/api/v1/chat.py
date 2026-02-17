"""
Chat API endpoints.
Handles user queries and conversation management.
"""

import time
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limit import check_rate_limit
from app.db.models import ChatSession
from app.db.session import async_session_maker
from app.dependencies import DBSession, OrchestratorSvc, RedisSvc, get_current_user
from app.models.chat import ChatFeedback, ChatRequest, ChatResponse
from app.models.user import UserInDB

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatHistoryResponse(BaseModel):
    """Response for chat history."""

    session_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    messages: List[dict]
    message_count: int
    created_at: str
    updated_at: str


class SessionsListResponse(BaseModel):
    """Response for listing user sessions."""

    sessions: List[ChatHistoryResponse]
    total: int


@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    orchestrator: OrchestratorSvc,
    redis: RedisSvc,
    db: DBSession,
):
    """
    Send a message and get a legal AI response.

    - **message**: User's question or query (1-2000 characters)
    - **session_id**: Optional session ID to continue a conversation
    - **category**: Optional category hint (criminal, road_traffic, general)

    The response will include:
    - AI-generated legal information based on Ghanaian law
    - Relevant legal citations
    - Appropriate disclaimers
    - Confidence score
    """
    # Check rate limit
    await check_rate_limit(
        None,  # Request object not available here
        redis,
        current_user.id if current_user else None,
        current_user.is_premium if current_user else False,
    )

    # Get or create session
    session_id = request.session_id or str(int(time.time() * 1000))

    # Process query through orchestrator
    result = await orchestrator.process_query(
        query=request.message, session_id=session_id, user_id=current_user.id
    )

    # Update or create chat session in database
    existing_session = await db.execute(
        select(ChatSession).where(
            ChatSession.session_id == session_id, ChatSession.user_id == current_user.id
        )
    )
    session = existing_session.scalar_one_or_none()

    if session:
        # Update existing session
        session.message_count += 1
        session.updated_at = result["timestamp"]
    else:
        # Create new session
        session = ChatSession(
            session_id=session_id,
            user_id=current_user.id,
            category=result.get("category"),
            message_count=1,
        )
        db.add(session)

    await db.commit()

    return ChatResponse(**result)


@router.get("/history", response_model=SessionsListResponse)
async def get_chat_history(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    db: DBSession,
    redis: RedisSvc,
    limit: int = 10,
    offset: int = 0,
):
    """
    Get list of user's chat sessions.

    - **limit**: Number of sessions to return (max 50)
    - **offset**: Number of sessions to skip
    """
    limit = min(limit, 50)

    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sessions = result.scalars().all()

    # Get total count
    count_result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == current_user.id)
    )
    total = len(count_result.all())

    formatted_sessions = []
    for session in sessions:
        # Get messages from Redis
        messages = await redis.get_session(session.session_id)
        if not messages:
            messages = session.messages or []

        formatted_sessions.append(
            ChatHistoryResponse(
                session_id=session.session_id,
                title=session.title,
                category=session.category,
                messages=messages,
                message_count=session.message_count,
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
            )
        )

    return SessionsListResponse(sessions=formatted_sessions, total=total)


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_session_history(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    db: DBSession,
    redis: RedisSvc,
):
    """
    Get full conversation history for a specific session.

    - **session_id**: Session identifier
    """
    # Get session from database
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.session_id == session_id, ChatSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Get messages from Redis
    messages = await redis.get_session(session_id)
    if not messages:
        messages = session.messages or []

    return ChatHistoryResponse(
        session_id=session.session_id,
        title=session.title,
        category=session.category,
        messages=messages,
        message_count=session.message_count,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.delete("/history/{session_id}")
async def delete_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    db: DBSession,
    redis: RedisSvc,
):
    """
    Delete a chat session.

    - **session_id**: Session identifier to delete
    """
    # Get session from database
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.session_id == session_id, ChatSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    # Delete from database
    await db.delete(session)
    await db.commit()

    # Delete from Redis
    await redis.delete_session(session_id)

    return {"message": "Session deleted successfully"}


@router.delete("/history")
async def delete_all_sessions(
    current_user: Annotated[UserInDB, Depends(get_current_user)], db: DBSession, redis_svc: RedisSvc
):
    """
    Delete all chat sessions for the current user.
    """
    # Get all sessions
    result = await db.execute(select(ChatSession).where(ChatSession.user_id == current_user.id))
    sessions = result.scalars().all()

    # Delete from Redis
    for session in sessions:
        await redis_svc.delete_session(session.session_id)

    # Delete from database
    for session in sessions:
        await db.delete(session)

    await db.commit()

    return {"message": f"Deleted {len(sessions)} sessions"}


@router.post("/feedback")
async def submit_feedback(
    feedback: ChatFeedback,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    db: DBSession,
):
    """
    Submit feedback on a chat response.

    - **message_id**: ID of the message to rate
    - **rating**: Rating from 1-5
    - **comment**: Optional comment
    - **helpful**: Whether the response was helpful
    """
    # Could store feedback in analytics table
    # For now, just acknowledge
    return {"message": "Feedback received. Thank you for helping us improve!"}


@router.websocket("/chat/stream")
async def websocket_chat(
    websocket: WebSocket,
    db: AsyncSession = Depends(async_session_maker),
    redis_svc: RedisSvc = Depends(),
    orchestrator: OrchestratorSvc = Depends(),
):
    """
    WebSocket endpoint for streaming chat responses.
    """
    await websocket.accept()

    try:
        # Receive initial message with authentication
        data = await websocket.receive_json()

        token = data.get("token")
        if not token:
            await websocket.close(code=1008, reason="No token provided")
            return

        # Verify token and get user
        from app.core.security import decode_token

        payload = decode_token(token)
        if not payload:
            await websocket.close(code=1008, reason="Invalid token")
            return

        user_id = int(payload.get("sub"))

        # Get user from database
        from app.db.models import User

        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user or not user.is_active:
            await websocket.close(code=1008, reason="Invalid user")
            return

        # Process message
        message = data.get("message", "")
        session_id = data.get("session_id")

        if not message:
            await websocket.close(code=1008, reason="No message provided")
            return

        # Stream response
        async for chunk in orchestrator.stream_query_with_chunks(
            query=message, session_id=session_id, user_id=user.id
        ):
            await websocket.send_json(chunk)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "data": {"message": str(e)}})
