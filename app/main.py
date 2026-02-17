"""
MMara Backend - FastAPI Application Entry Point

An AI-powered legal first-aid assistant for Ghanaians.
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import admin, auth, chat, users
from app.config import settings
from app.db.session import close_db, init_db
from app.utils.logger import log_request, logger
from app.utils.metrics import metrics

# API metadata
API_TITLE = "MMara Legal AI API"
API_DESCRIPTION = """
AI-powered legal first-aid assistant for Ghanaians.

## Features
- Multi-agent RAG system for legal queries
- Ghanaian Criminal Law and Road Traffic Acts
- Real-time chat with legal citations
- Session history and context awareness

## Authentication
Most endpoints require JWT authentication. Use `/api/v1/auth/login` to get a token.

## Rate Limits
- Free tier: 50 requests/day
- Authenticated: 500 requests/day
- Premium: Unlimited
"""
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting MMara Backend...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down MMara Backend...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request ID middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable):
    """Add unique request ID to each request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.time()

    response = await call_next(request)

    # Add request ID to response
    response.headers["X-Request-ID"] = request_id

    # Log request
    duration = time.time() - start_time
    log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        response_time=duration,
        request_id=request_id,
    )

    # Track metrics
    metrics.record_timing("request.duration", duration)
    metrics.increment("request.count")

    # Add timing header
    response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

    return response


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "details": exc.errors(),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


# Include routers
app.include_router(auth.router, prefix=settings.api_v1_prefix, tags=["Authentication"])

app.include_router(chat.router, prefix=settings.api_v1_prefix, tags=["Chat"])

app.include_router(users.router, prefix=settings.api_v1_prefix, tags=["Users"])

app.include_router(admin.router, prefix=settings.api_v1_prefix, tags=["Admin"])


# Root endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": API_VERSION,
        "status": "operational",
        "docs": "/docs" if settings.debug else None,
        "health": "/health",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics."""
    return metrics.get_all_metrics()


# Startup event summary
@app.get("/info", tags=["Root"])
async def info():
    """Get application information."""
    return {
        "app": settings.app_name,
        "version": API_VERSION,
        "environment": settings.environment,
        "debug": settings.debug,
        "features": {"rag": True, "multi_agent": True, "streaming": True, "rate_limiting": True},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
