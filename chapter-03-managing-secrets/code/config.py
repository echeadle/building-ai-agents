"""
Configuration and secrets loading for the agents project.

This module provides a centralized way to load and validate
environment variables needed by the application.

Chapter 3: Managing Secrets with python-dotenv
"""

import os
from dotenv import load_dotenv


def load_config() -> dict[str, str]:
    """
    Load environment variables from .env file and return as a dictionary.

    Returns:
        Dictionary containing configuration values.

    Raises:
        ValueError: If required environment variables are missing.
    """
    # Load variables from .env file into environment
    # This looks for .env in the current directory and parent directories
    load_dotenv()

    # Get the API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Validate that required variables are present
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Please create a .env file with your API key. "
            "See .env.example for the required format."
        )

    return {
        "anthropic_api_key": api_key,
    }


def get_api_key() -> str:
    """
    Convenience function to get just the Anthropic API key.

    Returns:
        The Anthropic API key string.

    Raises:
        ValueError: If the API key is not configured.
    """
    config = load_config()
    return config["anthropic_api_key"]


if __name__ == "__main__":
    # Test that configuration loads correctly
    try:
        config = load_config()
        # Don't print the full key! Just confirm it exists
        key = config["anthropic_api_key"]
        print(f"✓ API key loaded successfully")
        print(f"  Key starts with: {key[:20]}...")
        print(f"  Key length: {len(key)} characters")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
