"""Configuration management using Pydantic Settings"""

import logging
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # LLM API Keys
    openai_api_key: str
    anthropic_api_key: str

    # Database
    database_url: str

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # S3 Storage
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_bucket_name: str = "sat-questions"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    log_level: str = "INFO"

    # JWT Authentication
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Rate Limiting
    rate_limit_per_minute: int = 100

    # Model Configuration
    difficulty_model_path: str = "models/difficulty_model.pkl"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Feature Flags
    enable_ocr: bool = True
    enable_multi_model_validation: bool = True
    style_match_threshold: float = 0.7
    difficulty_tolerance: float = 10.0
    duplication_threshold: float = 0.85

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def setup_logging(log_level: str = "INFO"):
    """Configure application logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

