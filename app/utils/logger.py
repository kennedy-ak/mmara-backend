"""
Structured logging configuration.
"""

import logging
import sys
from datetime import datetime

from pythonjsonlogger import jsonlogger

from app.config import settings

# Observo remote logging (optional — only attached if API key is configured)
try:
    from observo_handler import ObservoHandler as _ObservoHandler
    _OBSERVO_AVAILABLE = True
except ImportError:
    _OBSERVO_AVAILABLE = False


class JsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict):
        super().add_fields(log_record, record, message_dict)

        # Add custom fields
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)


def setup_logging(name: str = "mmara", level: str = None, log_format: str = None) -> logging.Logger:
    """
    Set up structured logging.

    Args:
        name: Logger name
        level: Log level (defaults to settings)
        log_format: Log format (json or text)

    Returns:
        Configured logger
    """
    level = level or settings.log_level
    log_format = log_format or settings.log_format

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Set formatter
    if log_format == "json":
        formatter = JsonFormatter("%(timestamp)s %(level)s %(logger)s %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Attach Observo remote handler if configured
    if _OBSERVO_AVAILABLE and settings.observo_api_key and settings.observo_project_id:
        try:
            observo_handler = _ObservoHandler(
                project_id=settings.observo_project_id,
                api_key=settings.observo_api_key,
                observo_url=settings.observo_url,
            )
            observo_handler.setLevel(logging.INFO)
            logger.addHandler(observo_handler)
        except Exception as exc:
            logger.warning(f"Observo handler could not be initialised: {exc}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"mmara.{name}")


# Create default logger
logger = setup_logging("mmara")


class LogContext:
    """Context manager for adding context to logs."""

    def __init__(self, **kwargs):
        self.extra = kwargs

    def __enter__(self):
        for key, value in self.extra.items():
            logger = logging.getLogger("mmara")
            logger.extra = getattr(logger, "extra", {})
            logger.extra[key] = value
        return self

    def __exit__(self, *args):
        logger = logging.getLogger("mmara")
        if hasattr(logger, "extra"):
            for key in self.extra:
                logger.extra.pop(key, None)


def log_request(
    method: str, path: str, status_code: int, response_time: float, user_id: int = None, **kwargs
):
    """Log an API request."""
    logger.info(
        "api_request",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time_ms": response_time * 1000,
            "user_id": user_id,
            **kwargs,
        },
    )


def log_agent_execution(agent_name: str, status: str, execution_time: float, **kwargs):
    """Log agent execution."""
    logger.info(
        "agent_execution",
        extra={
            "agent": agent_name,
            "status": status,
            "execution_time_ms": execution_time * 1000,
            **kwargs,
        },
    )


def log_retrieval(
    query: str, results_count: int, category: str = None, execution_time: float = None, **kwargs
):
    """Log retrieval operation."""
    logger.info(
        "retrieval",
        extra={
            "query": query[:100],  # Truncate long queries
            "results_count": results_count,
            "category": category,
            "execution_time_ms": execution_time * 1000 if execution_time else None,
            **kwargs,
        },
    )
