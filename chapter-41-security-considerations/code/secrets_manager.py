"""
Secrets management for production.

Chapter 41: Security Considerations
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional
from functools import lru_cache


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""
    
    @abstractmethod
    def get_secret(self, name: str) -> str:
        """Get a secret by name."""
        pass


class EnvironmentSecretsProvider(SecretsProvider):
    """
    Get secrets from environment variables.
    
    Simple, works everywhere, good for development.
    """
    
    def get_secret(self, name: str) -> str:
        value = os.getenv(name)
        if value is None:
            raise ValueError(f"Secret {name} not found in environment")
        return value


class FileSecretsProvider(SecretsProvider):
    """
    Get secrets from a JSON file.
    
    Useful for local development with multiple secrets.
    The file should be in .gitignore!
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._secrets: Optional[dict] = None
    
    def _load_secrets(self) -> dict:
        if self._secrets is None:
            with open(self.filepath) as f:
                self._secrets = json.load(f)
        return self._secrets
    
    def get_secret(self, name: str) -> str:
        secrets = self._load_secrets()
        if name not in secrets:
            raise ValueError(f"Secret {name} not found in {self.filepath}")
        return secrets[name]


class AWSSecretsProvider(SecretsProvider):
    """
    Get secrets from AWS Secrets Manager.
    
    Production-grade secrets management with:
    - Automatic rotation
    - Access logging
    - Fine-grained IAM permissions
    """
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client(
                    "secretsmanager",
                    region_name=self.region
                )
            except ImportError:
                raise ImportError("boto3 required for AWS secrets. pip install boto3")
        return self._client
    
    def get_secret(self, name: str) -> str:
        try:
            response = self.client.get_secret_value(SecretId=name)
            
            if "SecretString" in response:
                secret = response["SecretString"]
                # Handle JSON secrets
                try:
                    return json.loads(secret)
                except json.JSONDecodeError:
                    return secret
            else:
                raise ValueError(f"Binary secrets not supported: {name}")
                
        except Exception as e:
            raise ValueError(f"Failed to get secret {name}: {e}")


class SecretsManager:
    """
    Unified secrets management with caching.
    
    Usage:
        secrets = SecretsManager()
        api_key = secrets.get("ANTHROPIC_API_KEY")
    """
    
    def __init__(self, provider: Optional[SecretsProvider] = None):
        """
        Initialize the secrets manager.
        
        If no provider specified, auto-detects:
        1. AWS Secrets Manager (if AWS_REGION set)
        2. File provider (if secrets.json exists)
        3. Environment variables (fallback)
        """
        if provider:
            self.provider = provider
        else:
            self.provider = self._auto_detect_provider()
        
        self._cache: dict[str, str] = {}
    
    def _auto_detect_provider(self) -> SecretsProvider:
        """Auto-detect the best available secrets provider."""
        # Check for AWS
        if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
            try:
                import boto3
                return AWSSecretsProvider()
            except ImportError:
                pass
        
        # Check for secrets file
        if os.path.exists("secrets.json"):
            return FileSecretsProvider("secrets.json")
        
        # Fallback to environment
        return EnvironmentSecretsProvider()
    
    @lru_cache(maxsize=100)
    def get(self, name: str) -> str:
        """
        Get a secret, with caching.
        
        Args:
            name: Secret name
        
        Returns:
            Secret value
        """
        return self.provider.get_secret(name)
    
    def clear_cache(self) -> None:
        """Clear the secrets cache (for rotation)."""
        self.get.cache_clear()


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    secrets = SecretsManager()
    
    try:
        api_key = secrets.get("ANTHROPIC_API_KEY")
        print(f"Successfully loaded API key: {api_key[:10]}...")
    except ValueError as e:
        print(f"Error: {e}")