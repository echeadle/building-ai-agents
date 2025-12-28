"""
Secure API key management.

Chapter 41: Security Considerations
"""

import os
from typing import Optional
from dotenv import load_dotenv


class SecureConfig:
    """
    Secure configuration management.
    
    Loads sensitive values from environment variables only,
    with validation and helpful error messages.
    """
    
    # Keys that should never be logged or displayed
    SENSITIVE_KEYS = {
        "anthropic_api_key",
        "database_password",
        "redis_password",
        "jwt_secret",
        "encryption_key"
    }
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file (for development only)
        """
        if env_file:
            load_dotenv(env_file)
        
        self._validate_required_keys()
    
    def _validate_required_keys(self) -> None:
        """Validate that all required keys are present."""
        required = ["ANTHROPIC_API_KEY"]
        missing = [key for key in required if not os.getenv(key)]
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {missing}. "
                "Set them in your environment or .env file."
            )
    
    @property
    def anthropic_api_key(self) -> str:
        """Get the Anthropic API key."""
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return key
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration value.
        
        Args:
            key: Environment variable name
            default: Default value if not set
        
        Returns:
            The configuration value
        """
        return os.getenv(key, default)
    
    def get_required(self, key: str) -> str:
        """
        Get a required configuration value.
        
        Args:
            key: Environment variable name
        
        Returns:
            The configuration value
        
        Raises:
            ValueError: If the key is not set
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
    def is_sensitive(self, key: str) -> bool:
        """Check if a key contains sensitive data."""
        return key.lower() in self.SENSITIVE_KEYS
    
    def safe_repr(self) -> dict[str, str]:
        """
        Get a safe representation of configuration for logging.
        
        Masks sensitive values.
        """
        result = {}
        for key in os.environ:
            if self.is_sensitive(key.lower()):
                result[key] = "***MASKED***"
            else:
                result[key] = os.getenv(key, "")[:50]  # Truncate long values
        return result


# Example usage
if __name__ == "__main__":
    config = SecureConfig(".env")
    
    # Safe to use
    api_key = config.anthropic_api_key
    print(f"API key loaded: {api_key[:10]}...{api_key[-4:]}")
    
    # Safe to log
    print("Configuration (safe repr):")
    for key, value in config.safe_repr().items():
        if "ANTHROPIC" in key or "API" in key:
            print(f"  {key}: {value}")