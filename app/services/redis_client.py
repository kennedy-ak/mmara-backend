"""
Redis Client Service.
Handles caching, session storage, and rate limiting.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger("mmara")


class RedisService:
    """
    Async Redis client for caching and session management.
    """

    def __init__(
        self, url: Optional[str] = None, encoding: str = "utf-8", decode_responses: bool = True
    ):
        self.url = url or settings.redis_url
        self.encoding = encoding
        self.decode_responses = decode_responses
        self._client: Optional[aioredis.Redis] = None
        self._pool: Optional[aioredis.ConnectionPool] = None
        self.max_connections = getattr(settings, "redis_max_connections", 50)

    async def connect(self):
        """Establish Redis connection with connection pooling."""
        if self._client is None:
            # Create connection pool with health checking
            self._pool = aioredis.ConnectionPool.from_url(
                self.url,
                encoding=self.encoding,
                decode_responses=self.decode_responses,
                max_connections=self.max_connections,
                socket_keepalive=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,  # Check connection health every 30s
            )
            self._client = aioredis.Redis(connection_pool=self._pool)

            # Test connection
            try:
                await self._client.ping()
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                await self.disconnect()
                raise

    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None

    @property
    def client(self) -> aioredis.Redis:
        """Get Redis client, creating connection if needed."""
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    async def get(self, key: str) -> Optional[str]:
        """Get a value from Redis."""
        return await self.client.get(key)

    async def set(self, key: str, value: str, expire: Optional[int] = None) -> bool:
        """Set a value in Redis."""
        return await self.client.set(key, value, ex=expire)

    async def delete(self, key: str) -> int:
        """Delete a key from Redis."""
        return await self.client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return await self.client.exists(key) > 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for a key."""
        return await self.client.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        """Get time to live for a key."""
        return await self.client.ttl(key)

    # JSON operations
    async def get_json(self, key: str) -> Optional[Any]:
        """Get and parse JSON value."""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None

    async def set_json(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Serialize and set JSON value."""
        serialized = json.dumps(value)
        return await self.set(key, serialized, expire)

    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get field from hash."""
        return await self.client.hget(name, key)

    async def hset(self, name: str, key: str, value: str) -> int:
        """Set field in hash."""
        return await self.client.hset(name, key, value)

    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all fields from hash."""
        return await self.client.hgetall(name)

    async def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from hash."""
        return await self.client.hdel(name, *keys)

    # List operations
    async def lpush(self, name: str, *values: str) -> int:
        """Push values to left of list."""
        return await self.client.lpush(name, *values)

    async def rpush(self, name: str, *values: str) -> int:
        """Push values to right of list."""
        return await self.client.rpush(name, *values)

    async def lrange(self, name: str, start: int = 0, end: int = -1) -> List[str]:
        """Get range of list elements."""
        return await self.client.lrange(name, start, end)

    async def ltrim(self, name: str, start: int, end: int) -> bool:
        """Trim list to range."""
        await self.client.ltrim(name, start, end)
        return True

    # Session management
    async def save_session(
        self, session_id: str, messages: List[Dict[str, str]], ttl: int = 86400  # 24 hours
    ) -> bool:
        """Save chat session to Redis."""
        key = f"session:{session_id}"
        return await self.set_json(key, messages, expire=ttl)

    async def get_session(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Get chat session from Redis."""
        key = f"session:{session_id}"
        return await self.get_json(key)

    async def add_message_to_session(
        self, session_id: str, message: Dict[str, str], max_messages: int = 50, ttl: int = 86400
    ) -> bool:
        """Add a message to a session."""
        key = f"session:{session_id}"
        messages = await self.get_session(session_id) or []
        messages.append(message)

        # Trim to max messages (keep recent)
        if len(messages) > max_messages:
            messages = messages[-max_messages:]

        return await self.set_json(key, messages, expire=ttl)

    async def delete_session(self, session_id: str) -> int:
        """Delete chat session."""
        key = f"session:{session_id}"
        return await self.delete(key)

    # Cache management
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return await self.get_json(f"cache:{key}")

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ttl = ttl or settings.redis_cache_ttl
        return await self.set_json(f"cache:{key}", value, expire=ttl)

    async def cache_delete(self, key: str) -> int:
        """Delete value from cache."""
        return await self.delete(f"cache:{key}")

    async def cache_clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        keys = []
        async for key in self.client.scan_iter(match=f"cache:{pattern}"):
            keys.append(key)
        if keys:
            return await self.client.delete(*keys)
        return 0

    # Query caching for RAG
    async def get_cached_query_result(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached retrieval result for a query."""
        key = f"query:{hash(query)}"
        return await self.get_json(key)

    async def cache_query_result(
        self, query: str, results: List[Dict[str, Any]], ttl: int = 3600  # 1 hour
    ) -> bool:
        """Cache retrieval results for a query."""
        key = f"query:{hash(query)}"
        return await self.set_json(key, results, expire=ttl)

    # Rate limiting
    async def increment_rate_limit(self, identifier: str, window: int = 86400) -> int:
        """Increment rate limit counter."""
        key = f"ratelimit:{identifier}"
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        return results[0]

    async def get_rate_limit_count(self, identifier: str) -> int:
        """Get current rate limit count."""
        key = f"ratelimit:{identifier}"
        value = await self.get(key)
        return int(value) if value else 0

    async def reset_rate_limit(self, identifier: str) -> int:
        """Reset rate limit counter."""
        key = f"ratelimit:{identifier}"
        return await self.delete(key)


# Singleton instance
_redis_service: Optional[RedisService] = None


async def get_redis_service() -> RedisService:
    """Get or create Redis service singleton."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
        await _redis_service.connect()
    return _redis_service
