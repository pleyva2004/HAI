"""FastAPI Application - Main entry point"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..utils.config import get_settings, setup_logging
from .middleware import ErrorHandlerMiddleware, LoggingMiddleware, RateLimitMiddleware
from .routes import router

# Setup logging
settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 SAT Question Generator API starting...")
    logger.info(f"   Environment: {settings.log_level}")
    logger.info(f"   Database: {settings.database_url.split('@')[1] if '@' in settings.database_url else 'configured'}")
    logger.info(f"   Redis: {settings.redis_url}")
    logger.info(f"   OCR: {'enabled' if settings.enable_ocr else 'disabled'}")

    yield

    # Shutdown
    logger.info("👋 SAT Question Generator API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SAT Question Generator API",
    description="AI-powered SAT question generation with style matching, difficulty calibration, and anti-duplication",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middleware (order matters!)
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    redis_url=settings.redis_url,
    rate_limit=settings.rate_limit_per_minute,
)

# Include routers
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SAT Question Generator API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=300,  # 5 minutes for long-running LLM operations
    )

