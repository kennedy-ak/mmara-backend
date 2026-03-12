"""
Rate limiting utilities for protecting sensitive endpoints.
"""

import time
from typing import Callable, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.services.redis_client import RedisService


class RateLimiter:
    """
    Rate limiter using Redis for distributed rate limiting.
    """

    def __init__(
        self,
        redis_service: RedisService,
        max_requests: int = 5,
        window_seconds: int = 60,
        identifier: Optional[str] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            redis_service: Redis service instance
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds
            identifier: Optional custom identifier (defaults to IP address)
        """
        self.redis = redis_service
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.identifier = identifier

    def get_identifier(self, request: Request) -> str:
        """
        Get identifier for rate limiting.
        Uses forwarded IP if available, otherwise direct IP.
        """
        if self.identifier:
            return f"{self.identifier}:{self.identifier}"

        # Check for forwarded IP (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ratelimit:{request.url.path}:{ip}"

    async def check(self, request: Request) -> None:
        """
        Check if the request should be rate limited.

        Raises:
            HTTPException: If rate limit is exceeded
        """
        identifier = self.get_identifier(request)

        # Get current count
        current = await self.redis.get_rate_limit_count(identifier)

        if current >= self.max_requests:
            # Get TTL to return retry-after header
            ttl = await self.redis.ttl(f"ratelimit:{request.url.path}:{request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded. Please try again later.",
                    "retry_after": ttl if ttl > 0 else self.window_seconds,
                },
                headers={"Retry-After": str(ttl if ttl > 0 else self.window_seconds)},
            )

        # Increment counter
        await self.redis.increment_rate_limit(identifier, self.window_seconds)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for applying rate limiting to specific paths.
    """

    def __init__(
        self,
        app,
        redis_service: RedisService,
        default_limit: int = 100,
        default_window: int = 60,
        protected_paths: Optional[dict] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI app
            redis_service: Redis service instance
            default_limit: Default max requests per window
            default_window: Default time window in seconds
            protected_paths: Dict mapping path patterns to (limit, window) tuples
        """
        super().__init__(app)
        self.redis = redis_service
        self.default_limit = default_limit
        self.default_window = default_window
        self.protected_paths = protected_paths or {
            "/api/v1/auth/login": (5, 300),  # 5 login attempts per 5 minutes
            "/api/v1/auth/register": (3, 3600),  # 3 registrations per hour
            "/api/v1/auth/password-reset": (3, 3600),  # 3 password resets per hour
        }

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Only apply to POST requests on protected paths
        if request.method == "POST":
            for path, (limit, window) in self.protected_paths.items():
                if request.url.path.startswith(path):
                    limiter = RateLimiter(self.redis, limit, window)
                    await limiter.check(request)
                    break

        response = await call_next(request)
        return response


# Predefined rate limiters for use as dependencies
async def auth_rate_limit(
    request: Request,
    redis_service: RedisService,
) -> None:
    """
    Rate limiter for authentication endpoints.
    5 attempts per 5 minutes per IP.
    """
    limiter = RateLimiter(redis_service, max_requests=5, window_seconds=300)
    await limiter.check(request)


async def strict_rate_limit(
    request: Request,
    redis_service: RedisService,
) -> None:
    """
    Strict rate limiter for sensitive operations.
    3 attempts per hour per IP.
    """
    limiter = RateLimiter(redis_service, max_requests=3, window_seconds=3600)
    await limiter.check(request)
