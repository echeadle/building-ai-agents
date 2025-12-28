"""
Fallback strategies for maintaining agent functionality.

Chapter 30: Error Handling and Recovery

This module demonstrates patterns for graceful degradation when
primary operations fail.
"""

import os
import json
from typing import Optional, Any, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

T = TypeVar('T')


# ============================================================
# Caching Infrastructure
# ============================================================

@dataclass
class CacheEntry:
    """
    A cached response with metadata.
    
    Tracks when data was cached and for how long it's valid.
    """
    data: Any
    timestamp: datetime
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_valid(self) -> bool:
        """Check if the cache entry is still within its TTL."""
        age = datetime.now() - self.timestamp
        return age < timedelta(seconds=self.ttl_seconds)
    
    def is_stale_but_usable(self, max_stale_seconds: int = 86400) -> bool:
        """
        Check if entry is stale but could be used as fallback.
        
        Stale data is better than no data in many cases.
        """
        age = datetime.now() - self.timestamp
        return age < timedelta(seconds=max_stale_seconds)
    
    @property
    def age_seconds(self) -> float:
        """Get the age of this cache entry in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class FallbackCache:
    """
    Simple in-memory cache for fallback responses.
    
    This cache supports both fresh and stale data retrieval,
    allowing for graceful degradation when live data is unavailable.
    """
    
    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry if it exists."""
        return self._cache.get(key)
    
    def set(self, key: str, data: Any, ttl_seconds: int = 3600):
        """
        Store data in the cache.
        
        Args:
            key: Cache key
            data: Data to store
            ttl_seconds: Time-to-live in seconds
        """
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )
    
    def get_valid(self, key: str) -> Optional[Any]:
        """Get data only if the cache entry is still valid (not expired)."""
        entry = self.get(key)
        if entry and entry.is_valid():
            return entry.data
        return None
    
    def get_stale(self, key: str, max_stale_seconds: int = 86400) -> Optional[Any]:
        """
        Get data even if stale (for fallback scenarios).
        
        This is useful when the primary source is unavailable and
        slightly outdated data is better than no data.
        """
        entry = self.get(key)
        if entry and entry.is_stale_but_usable(max_stale_seconds):
            return entry.data
        return None
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
    
    def stats(self) -> dict:
        """Get cache statistics."""
        valid_count = sum(1 for e in self._cache.values() if e.is_valid())
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "stale_entries": len(self._cache) - valid_count,
        }


# ============================================================
# Fallback Chain Pattern
# ============================================================

@dataclass
class FallbackResult(Generic[T]):
    """
    Result from a fallback chain execution.
    
    Contains both the value and metadata about how it was obtained.
    """
    value: T
    source: str  # Which strategy succeeded
    fallback_used: bool  # True if not from primary strategy
    attempts: list[str] = field(default_factory=list)  # What was tried


class FallbackChain(Generic[T]):
    """
    A chain of fallback strategies to try in order.
    
    Each strategy is a callable that returns a result or None.
    The chain tries each strategy until one succeeds.
    
    Example:
        chain = FallbackChain()
        chain.add("live_api", fetch_from_api)
        chain.add("cache", get_from_cache)
        chain.add("default", lambda: default_value)
        
        result = chain.execute()
    """
    
    def __init__(self):
        self._strategies: list[tuple[str, Callable[[], Optional[T]]]] = []
    
    def add(self, name: str, strategy: Callable[[], Optional[T]]) -> "FallbackChain[T]":
        """Add a named strategy to the chain."""
        self._strategies.append((name, strategy))
        return self
    
    def execute(self, default: Optional[T] = None) -> FallbackResult[T]:
        """
        Execute strategies in order until one succeeds.
        
        A strategy "succeeds" if it returns a non-None value without
        raising an exception.
        
        Args:
            default: Value to return if all strategies fail.
                    If None and all fail, raises an exception.
            
        Returns:
            FallbackResult containing the value and metadata
            
        Raises:
            RuntimeError: If all strategies fail and no default provided
        """
        attempts = []
        
        for i, (name, strategy) in enumerate(self._strategies):
            try:
                result = strategy()
                if result is not None:
                    return FallbackResult(
                        value=result,
                        source=name,
                        fallback_used=(i > 0),  # Not the first strategy
                        attempts=attempts
                    )
                attempts.append(f"{name}: returned None")
            except Exception as e:
                attempts.append(f"{name}: {type(e).__name__}: {e}")
        
        # All strategies failed
        if default is not None:
            return FallbackResult(
                value=default,
                source="default",
                fallback_used=True,
                attempts=attempts
            )
        
        # No default, must raise
        raise RuntimeError(
            f"All fallback strategies failed: {attempts}"
        )


# ============================================================
# Example: Weather Service with Fallback
# ============================================================

class WeatherServiceWithFallback:
    """
    Example service demonstrating fallback patterns.
    
    This simulates a weather service that tries multiple strategies
    when the primary source fails.
    """
    
    def __init__(self):
        self.cache = FallbackCache()
        self.default_weather = {
            "temperature": "Unknown",
            "conditions": "Data unavailable",
            "source": "default"
        }
    
    def _fetch_primary_api(self, location: str) -> Optional[dict]:
        """
        Simulate primary API call.
        
        In a real implementation, this would call the actual weather API.
        For demonstration, this always fails.
        """
        # Simulating API failure
        raise ConnectionError("Primary weather API unavailable")
    
    def _fetch_backup_api(self, location: str) -> Optional[dict]:
        """
        Simulate backup API call.
        
        For demonstration, this works for some locations.
        """
        # Simulating a backup that works for major cities
        known_cities = {
            "new york": {"temperature": "72°F", "conditions": "Partly cloudy"},
            "london": {"temperature": "58°F", "conditions": "Rainy"},
            "tokyo": {"temperature": "68°F", "conditions": "Clear"},
        }
        
        weather = known_cities.get(location.lower())
        if weather:
            return {**weather, "source": "backup_api"}
        return None
    
    def _get_cached(self, location: str) -> Optional[dict]:
        """Get cached weather data."""
        cached = self.cache.get_stale(f"weather:{location.lower()}")
        if cached:
            return {**cached, "source": "cache", "note": "May be outdated"}
        return None
    
    def _get_default(self, location: str) -> dict:
        """Return default response when all else fails."""
        return {
            **self.default_weather,
            "location": location,
            "note": "Weather data temporarily unavailable"
        }
    
    def get_weather(self, location: str) -> FallbackResult[dict]:
        """
        Get weather using fallback chain.
        
        Tries: primary API -> backup API -> cache -> default
        
        Args:
            location: City name
            
        Returns:
            FallbackResult containing weather data and source info
        """
        chain: FallbackChain[dict] = FallbackChain()
        chain.add("primary_api", lambda: self._fetch_primary_api(location))
        chain.add("backup_api", lambda: self._fetch_backup_api(location))
        chain.add("cache", lambda: self._get_cached(location))
        chain.add("default", lambda: self._get_default(location))
        
        result = chain.execute()
        
        # Cache successful API results for future fallback
        if result.source in ["primary_api", "backup_api"]:
            self.cache.set(f"weather:{location.lower()}", result.value)
        
        return result


# ============================================================
# Model Fallback Pattern
# ============================================================

@dataclass
class ModelFallbackConfig:
    """Configuration for LLM model fallback."""
    primary_model: str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = field(default_factory=lambda: [
        "claude-sonnet-4-20250514",  # Retry same model
        "claude-haiku-4-5-20250929",  # Faster, cheaper fallback
    ])
    reduce_tokens_on_fallback: bool = True
    token_reduction_factor: float = 0.75


def call_with_model_fallback(
    client: anthropic.Anthropic,
    messages: list[dict],
    config: Optional[ModelFallbackConfig] = None,
    **kwargs
) -> tuple[anthropic.types.Message, str]:
    """
    Make an API call with model fallback on failure.
    
    Tries the primary model first, then falls back to alternatives.
    
    Args:
        client: Anthropic client
        messages: Messages to send
        config: Fallback configuration
        **kwargs: Additional arguments for API call
        
    Returns:
        Tuple of (response, model_used)
    """
    config = config or ModelFallbackConfig()
    all_models = [config.primary_model] + config.fallback_models
    
    original_max_tokens = kwargs.get("max_tokens", 1024)
    last_error = None
    
    for i, model in enumerate(all_models):
        try:
            # Optionally reduce tokens for fallback models
            max_tokens = original_max_tokens
            if i > 0 and config.reduce_tokens_on_fallback:
                max_tokens = int(original_max_tokens * config.token_reduction_factor)
            
            response = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            
            return response, model
            
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            last_error = e
            print(f"  Model {model} failed: {type(e).__name__}, trying next...")
            continue
        except anthropic.APIError:
            # Non-retryable errors should not fallback
            raise
    
    # All models failed
    raise last_error or RuntimeError("All fallback models failed")


# ============================================================
# Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FALLBACK STRATEGIES DEMONSTRATION")
    print("=" * 60)
    
    # 1. Cache demonstration
    print("\n### 1. Fallback Cache ###\n")
    
    cache = FallbackCache()
    cache.set("data:1", {"value": "fresh"}, ttl_seconds=5)
    cache.set("data:2", {"value": "old"}, ttl_seconds=0)  # Immediately stale
    
    print(f"Fresh data: {cache.get_valid('data:1')}")
    print(f"Stale data (get_valid): {cache.get_valid('data:2')}")
    print(f"Stale data (get_stale): {cache.get_stale('data:2')}")
    print(f"Missing data: {cache.get_valid('data:3')}")
    print(f"Cache stats: {cache.stats()}")
    
    # 2. Fallback chain demonstration
    print("\n### 2. Fallback Chain ###\n")
    
    chain: FallbackChain[str] = FallbackChain()
    chain.add("primary", lambda: None)  # Returns None (fails)
    chain.add("secondary", lambda: (_ for _ in ()).throw(ValueError("Oops!")))  # Raises
    chain.add("tertiary", lambda: "Success from tertiary!")
    
    result = chain.execute()
    print(f"Value: {result.value}")
    print(f"Source: {result.source}")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Attempts: {result.attempts}")
    
    # 3. Weather service demonstration
    print("\n### 3. Weather Service with Fallback ###\n")
    
    weather_service = WeatherServiceWithFallback()
    
    # Pre-populate cache for demonstration
    weather_service.cache.set("weather:san francisco", {
        "temperature": "65°F",
        "conditions": "Foggy"
    })
    
    locations = ["New York", "San Francisco", "Unknown City"]
    
    for location in locations:
        print(f"Getting weather for {location}:")
        result = weather_service.get_weather(location)
        print(f"  Data: {result.value}")
        print(f"  Source: {result.source}")
        print(f"  Fallback used: {result.fallback_used}")
        print()
    
    # 4. Model fallback demonstration
    print("### 4. Model Fallback ###\n")
    
    client = anthropic.Anthropic()
    
    response, model_used = call_with_model_fallback(
        client,
        messages=[{"role": "user", "content": "Say 'Model fallback test!'"}],
        max_tokens=50
    )
    
    print(f"Response: {response.content[0].text}")
    print(f"Model used: {model_used}")
    
    # 5. Custom fallback chain for a tool
    print("\n### 5. Tool Fallback Pattern ###\n")
    
    def tool_with_fallback(query: str) -> dict:
        """Example tool that uses fallback chain."""
        
        def try_live_search():
            # Simulate API call
            raise ConnectionError("Search API down")
        
        def try_cached_results():
            # Simulate cache lookup
            if "python" in query.lower():
                return {"results": ["Python docs", "Python tutorial"], "source": "cache"}
            return None
        
        def return_default():
            return {"results": [], "source": "default", "message": "Search unavailable"}
        
        chain: FallbackChain[dict] = FallbackChain()
        chain.add("live_search", try_live_search)
        chain.add("cache", try_cached_results)
        chain.add("default", return_default)
        
        return chain.execute().value
    
    queries = ["Python tutorials", "Obscure topic"]
    for query in queries:
        print(f"Search: '{query}'")
        result = tool_with_fallback(query)
        print(f"  Result: {result}")
    
    print("\n" + "=" * 60)
    print("KEY PATTERNS:")
    print("- Cache with stale data support")
    print("- Fallback chains with multiple strategies")
    print("- Model fallback for resilience")
    print("- Transparent communication of degraded state")
    print("=" * 60)
