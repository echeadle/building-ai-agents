"""
Environment configuration management.

Chapter 40: Deployment Patterns

This module provides a centralized configuration system using Pydantic Settings.
Configuration is loaded from environment variables and .env files.

Usage:
    from settings import get_settings
    
    settings = get_settings()
    print(settings.anthropic_api_key)
    print(settings.is_production)
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic Settings automatically:
    - Loads values from environment variables
    - Validates types and constraints
    - Supports .env files for local development
    - Provides sensible defaults
    
    Environment variables can be prefixed (e.g., APP_API_PORT=8000).
    """
    
    # ----- Environment -----
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment (development, staging, production, testing)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (more logging, API docs in prod)"
    )
    
    # ----- API Server -----
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to bind the API server"
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of uvicorn workers"
    )
    
    # ----- Anthropic -----
    anthropic_api_key: str = Field(
        ...,  # Required - no default
        description="Anthropic API key"
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Claude model to use"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens per response"
    )
    
    # ----- Rate Limiting -----
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per minute per client"
    )
    
    # ----- Redis -----
    redis_enabled: bool = Field(
        default=False,
        description="Enable Redis for caching and task queues"
    )
    redis_host: str = Field(
        default="localhost",
        description="Redis host"
    )
    redis_port: int = Field(
        default=6379,
        description="Redis port"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password"
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    
    # ----- Logging -----
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    
    # ----- CORS -----
    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # ----- Timeouts -----
    request_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Maximum request timeout in seconds"
    )
    
    # ----- Validation -----
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v
    
    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is valid."""
        valid_formats = ["json", "text"]
        v = v.lower()
        if v not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
        return v
    
    # ----- Computed Properties -----
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Check if running tests."""
        return self.environment == Environment.TESTING
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # ----- Helper Methods -----
    def get_uvicorn_config(self) -> dict:
        """Get configuration dict for uvicorn."""
        config = {
            "host": self.api_host,
            "port": self.api_port,
            "workers": self.api_workers,
            "log_level": self.log_level.lower(),
        }
        
        if self.is_development:
            config["reload"] = True
            config["workers"] = 1  # Reload doesn't work with multiple workers
        
        return config
    
    def get_logging_config(self) -> dict:
        """Get logging configuration."""
        if self.log_format == "json":
            format_string = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        else:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        return {
            "level": self.log_level,
            "format": format_string
        }
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow both ANTHROPIC_API_KEY and anthropic_api_key
        populate_by_name = True
        # Environment variable prefix (optional)
        # env_prefix = "AGENT_"


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings.
    
    Uses lru_cache to load settings only once per process.
    Call get_settings.cache_clear() if you need to reload.
    """
    return Settings()


# ----- Example .env Files -----

DEVELOPMENT_ENV = """
# .env.development
ENVIRONMENT=development
DEBUG=true
ANTHROPIC_API_KEY=sk-ant-your-key-here
LOG_LEVEL=DEBUG
LOG_FORMAT=text
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
RATE_LIMIT_REQUESTS=1000
"""

PRODUCTION_ENV = """
# .env.production
ENVIRONMENT=production
DEBUG=false
ANTHROPIC_API_KEY=sk-ant-your-production-key
API_WORKERS=4
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=["https://myapp.com", "https://www.myapp.com"]
RATE_LIMIT_REQUESTS=60
REDIS_ENABLED=true
REDIS_HOST=redis.internal
"""


# ----- Usage Example -----

if __name__ == "__main__":
    import json
    
    # Get settings
    settings = get_settings()
    
    print("=" * 60)
    print("APPLICATION SETTINGS")
    print("=" * 60)
    print()
    
    # Display all settings (mask the API key)
    settings_dict = settings.model_dump()
    if "anthropic_api_key" in settings_dict:
        key = settings_dict["anthropic_api_key"]
        settings_dict["anthropic_api_key"] = f"{key[:10]}...{key[-4:]}"
    
    print(json.dumps(settings_dict, indent=2, default=str))
    
    print()
    print("-" * 60)
    print("COMPUTED PROPERTIES")
    print("-" * 60)
    print(f"is_production: {settings.is_production}")
    print(f"is_development: {settings.is_development}")
    print(f"redis_url: {settings.redis_url}")
    
    print()
    print("-" * 60)
    print("UVICORN CONFIG")
    print("-" * 60)
    print(json.dumps(settings.get_uvicorn_config(), indent=2))
    
    print()
    print("-" * 60)
    print("EXAMPLE .env FILES")
    print("-" * 60)
    print("\nDevelopment:")
    print(DEVELOPMENT_ENV)
    print("\nProduction:")
    print(PRODUCTION_ENV)
