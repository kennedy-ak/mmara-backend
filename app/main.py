"""
MMara Backend - FastAPI Application Entry Point

An AI-powered legal first-aid assistant for Ghanaians.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.v1 import admin, auth, bug_reports, chat, users
from app.config import settings
from app.db.session import close_db, init_db
from app.utils.logger import log_error, log_request, logger
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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and add security headers to response."""
        response: Response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS filter (legacy but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (basic, can be customized)
        if settings.environment == "production":
            # Strict CSP for production
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            )
        else:
            # More lenient CSP for development (allow Swagger UI CDN)
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' 'unsafe-dynamic' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https: https://fastapi.tiangolo.com; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "connect-src 'self' http://127.0.0.1:* http://localhost:* https://cdn.jsdelivr.net ws://127.0.0.1:* ws://localhost:*; "
                "frame-ancestors 'self'; "
                "object-src 'none'; "
                "base-uri 'self';"
            )

        response.headers["Content-Security-Policy"] = csp

        # HSTS (HTTP Strict Transport Security) - only in production with HTTPS
        if settings.environment == "production":
            # Strict HSTS for 1 year
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Permissions policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )

        return response


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

    # Pre-build BM25 index in the background (non-blocking)
    try:
        from app.dependencies import get_embedding_service, get_retrieval_service

        emb = await get_embedding_service()
        ret = await get_retrieval_service(emb)
        asyncio.create_task(ret.build_bm25_index())
        logger.info("BM25 index build started in background")
    except Exception as e:
        logger.warning(f"Could not start BM25 background build: {e}")

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

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)


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
    user_id = getattr(request.state, "user_id", None)
    log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        response_time=duration,
        request_id=request_id,
        user_id=user_id,
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
    request_id = getattr(request.state, "request_id", None)
    level = logging.WARNING if exc.status_code < 500 else logging.ERROR
    logger.log(
        level,
        f"http_error {request.method} {request.url.path} {exc.status_code}: {exc.detail}",
        extra={
            "extra_data": {
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "request_id": request_id,
            }
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        f"validation_error {request.method} {request.url.path}: {len(exc.errors())} field(s) invalid",
        extra={
            "extra_data": {
                "method": request.method,
                "path": request.url.path,
                "errors": exc.errors(),
                "request_id": request_id,
            }
        },
    )
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
    request_id = getattr(request.state, "request_id", None)
    log_error(
        f"unhandled_exception {request.method} {request.url.path}: {type(exc).__name__}: {exc}",
        exc=exc,
        method=request.method,
        path=request.url.path,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "request_id": request_id,
        },
    )


# Include routers
app.include_router(auth.router, prefix=settings.api_v1_prefix, tags=["Authentication"])

app.include_router(chat.router, prefix=settings.api_v1_prefix, tags=["Chat"])

app.include_router(users.router, prefix=settings.api_v1_prefix, tags=["Users"])

app.include_router(admin.router, prefix=settings.api_v1_prefix, tags=["Admin"])

app.include_router(bug_reports.router, prefix=settings.api_v1_prefix, tags=["Bug Reports"])


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
    """Get application metrics (debug mode only)."""
    if not settings.debug:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
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
