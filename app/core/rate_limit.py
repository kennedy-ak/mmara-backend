"""
Rate limiting using Redis.
Implements sliding window rate limiting for API endpoints.
"""

import json
import time
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status

from app.config import settings
from app.services.redis_client import RedisService


class RateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    """

    def __init__(self, redis_service: RedisService):
        self.redis = redis_service
        self.free_limit = settings.rate_limit_free
        self.auth_limit = settings.rate_limit_auth
        self.premium_limit = settings.rate_limit_premium

    async def get_limit(self, user_id: Optional[int], is_premium: bool = False) -> int:
        """Get the rate limit for a user."""
        if user_id is None:
            return self.free_limit
        if is_premium:
            return self.premium_limit
        return self.auth_limit

    async def check_rate_limit(
        self, identifier: str, limit: int, window: int = 86400  # 24 hours in seconds
    ) -> Tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Args:
            identifier: Unique identifier (IP or user_id)
            limit: Maximum requests allowed
            window: Time window in seconds

        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        key = f"ratelimit:{identifier}"
        current_time = int(time.time())

        # Get current data
        data = await self.redis.get(key)
        if data:
            try:
                requests = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                requests = []
        else:
            requests = []

        # Filter out old requests outside the window
        window_start = current_time - window
        requests = [r for r in requests if r > window_start]

        # Check limit
        if limit >= 0 and len(requests) >= limit:
            return False, 0, requests[0] + window

        # Add current request
        requests.append(current_time)

        # Store back in Redis
        await self.redis.set(key, json.dumps(requests), expire=window)

        remaining = max(0, limit - len(requests)) if limit >= 0 else -1
        reset_time = requests[0] + window if requests else current_time + window

        return True, remaining, reset_time

    async def check_and_raise(
        self, identifier: str, user_id: Optional[int] = None, is_premium: bool = False
    ):
        """
        Check rate limit and raise HTTPException if exceeded.

        Args:
            identifier: Unique identifier
            user_id: Optional user ID
            is_premium: Whether user has premium tier

        Raises:
            HTTPException: If rate limit exceeded
        """
        limit = await self.get_limit(user_id, is_premium)

        # Unlimited for premium
        if limit < 0:
            return

        allowed, remaining, reset_time = await self.check_rate_limit(identifier, limit)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "Rate limit exceeded", "limit": limit, "reset_time": reset_time},
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time - int(time.time())),
                },
            )


async def check_rate_limit(
    request: Request,
    redis_service: RedisService,
    user_id: Optional[int] = None,
    is_premium: bool = False,
):
    """
    Dependency for rate limiting in FastAPI endpoints.

    Args:
        request: FastAPI request
        redis_service: Redis service
        user_id: Optional authenticated user ID
        is_premium: Whether user is premium

    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get identifier - use user_id if authenticated, otherwise IP
    if user_id:
        identifier = f"user:{user_id}"
    else:
        # Get IP from request
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            identifier = f"ip:{forwarded.split(',')[0].strip()}"
        else:
            identifier = f"ip:{request.client.host}"

    limiter = RateLimiter(redis_service)
    await limiter.check_and_raise(identifier, user_id, is_premium)


def get_identifier_from_request(request: Request) -> str:
    """
    Extract identifier from request for rate limiting.

    Args:
        request: FastAPI request

    Returns:
        str: Identifier string
    """
    # Check for API key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"apikey:{api_key}"

    # Use IP address
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"

    return f"ip:{request.client.host}"
