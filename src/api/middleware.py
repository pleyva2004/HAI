"""API Middleware - Rate limiting, logging, error handling"""

import logging
import time
from typing import Callable

import redis.asyncio as redis
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-based rate limiting middleware"""

    def __init__(self, app, redis_url: str, rate_limit: int = 100):
        super().__init__(app)
        self.redis_client = None
        self.redis_url = redis_url
        self.rate_limit = rate_limit  # requests per minute
        self.window = 60  # seconds

    async def setup(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(self.redis_url)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        # Skip rate limiting for health check
        if request.url.path == "/api/v1/health":
            return await call_next(request)

        # Get client identifier (IP address)
        client_id = request.client.host if request.client else "unknown"

        try:
            if not self.redis_client:
                await self.setup()

            # Rate limit key
            key = f"rate_limit:{client_id}"

            # Increment counter
            current = await self.redis_client.incr(key)

            if current == 1:
                # First request in window, set expiration
                await self.redis_client.expire(key, self.window)

            # Check if rate limit exceeded
            if current > self.rate_limit:
                logger.warning(f"Rate limit exceeded for {client_id}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded. Please try again later.",
                        "limit": self.rate_limit,
                        "window": self.window,
                    },
                )

            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.rate_limit - current)
            )

            return response

        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}")
            # On error, allow request through
            return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""
        start_time = time.time()

        # Log request
        logger.info(f"→ {request.method} {request.url.path}")

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"← {request.method} {request.url.path} "
                f"[{response.status_code}] {duration:.3f}s"
            )

            # Add timing header
            response.headers["X-Process-Time"] = f"{duration:.3f}"

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"← {request.method} {request.url.path} "
                f"[ERROR] {duration:.3f}s - {str(e)}"
            )
            raise


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle uncaught exceptions"""
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions (handled by FastAPI)
            raise
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "error": str(e) if logger.level == logging.DEBUG else None,
                },
            )

