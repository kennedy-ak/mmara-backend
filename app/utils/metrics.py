"""
Metrics and monitoring utilities.
"""

import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, Optional


class MetricsCollector:
    """Collect and track application metrics."""

    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._timings: Dict[str, list] = defaultdict(list)

    def increment(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        key = self._make_key(name, tags)
        self._counters[key] += value

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        key = self._make_key(name, tags)
        self._gauges[key] = value

    def record_timing(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a timing value."""
        key = self._make_key(name, tags)
        self._timings[key].append(value)

        # Keep only last 1000 values
        if len(self._timings[key]) > 1000:
            self._timings[key] = self._timings[key][-1000:]

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        key = self._make_key(name, tags)
        self._histograms[key].append(value)

    def get_counter(self, name: str, tags: Dict[str, str] = None) -> int:
        """Get counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key)

    def get_timing_stats(
        self, name: str, tags: Dict[str, str] = None
    ) -> Optional[Dict[str, float]]:
        """Get timing statistics (avg, min, max, p50, p95, p99)."""
        key = self._make_key(name, tags)
        timings = self._timings.get(key)

        if not timings:
            return None

        sorted_timings = sorted(timings)
        n = len(sorted_timings)

        return {
            "count": n,
            "avg": sum(sorted_timings) / n,
            "min": sorted_timings[0],
            "max": sorted_timings[-1],
            "p50": sorted_timings[int(n * 0.5)],
            "p95": sorted_timings[int(n * 0.95)],
            "p99": sorted_timings[int(n * 0.99)],
        }

    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timings.clear()

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "timings": {
                key: self.get_timing_stats(key.split("|")[0]) for key in self._timings.keys()
            },
        }

    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create a metric key from name and optional tags."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"


# Global metrics instance
metrics = MetricsCollector()


def track_time(metric_name: str, tags: Dict[str, str] = None):
    """
    Decorator to track function execution time.

    Args:
        metric_name: Name of the metric
        tags: Optional tags for the metric
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_timing(metric_name, duration, tags)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics.record_timing(metric_name, duration, tags)

        # Return appropriate wrapper based on whether function is async
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_counter(metric_name: str, tags: Dict[str, str] = None, value: int = 1):
    """
    Decorator to increment a counter when function is called.

    Args:
        metric_name: Name of the metric
        tags: Optional tags for the metric
        value: Value to increment by
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics.increment(metric_name, value, tags)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class RequestContext:
    """Context for tracking request-level metrics."""

    def __init__(self, request_id: str = None):
        self.request_id = request_id or str(int(time.time() * 1000))
        self.start_time = time.time()
        self.tags: Dict[str, str] = {}
        self.custom_metrics: Dict[str, Any] = {}

    def set_tag(self, key: str, value: str):
        """Set a tag for this request."""
        self.tags[key] = value

    def set_metric(self, key: str, value: Any):
        """Set a custom metric."""
        self.custom_metrics[key] = value

    def record_timing(self, name: str, duration: float = None):
        """Record a timing for this request."""
        if duration is None:
            duration = time.time() - self.start_time
        metrics.record_timing(name, duration, self.tags)

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter for this request."""
        metrics.increment(name, value, self.tags)

    def get_duration(self) -> float:
        """Get elapsed time since request start."""
        return time.time() - self.start_time

    def finish(self):
        """Finish request and record final metrics."""
        duration = self.get_duration()
        self.record_timing("request.duration", duration)
        self.increment_counter("request.count")
        return {
            "request_id": self.request_id,
            "duration_ms": duration * 1000,
            "tags": self.tags,
            "metrics": self.custom_metrics,
        }


# Predefined metric names
class MetricNames:
    """Standard metric names."""

    # Request metrics
    REQUEST_COUNT = "request.count"
    REQUEST_DURATION = "request.duration"
    REQUEST_ERRORS = "request.errors"

    # Agent metrics
    AGENT_EXECUTION = "agent.execution"
    AGENT_FAILURE = "agent.failure"

    # Retrieval metrics
    RETRIEVAL_COUNT = "retrieval.count"
    RETRIEVAL_DURATION = "retrieval.duration"
    RETRIEVAL_RESULTS = "retrieval.results"

    # LLM metrics
    LLM_REQUEST = "llm.request"
    LLM_DURATION = "llm.duration"
    LLM_TOKENS = "llm.tokens"

    # Database metrics
    DB_QUERY = "db.query"
    DB_DURATION = "db.duration"

    # Cache metrics
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
