"""
Speed-optimized caching for agent responses.

Chapter 39: Latency Optimization

Caching reduces latency by avoiding redundant work. The best API
call is the one you don't make.
"""

import hashlib
import time
import json
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import OrderedDict
import threading


@dataclass
class CacheEntry:
    """A cached response with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    size: int = 0
    evictions: int = 0
    total_saved_ms: float = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "evictions": self.evictions,
            "hit_rate": f"{self.hit_rate:.1%}",
            "total_saved_ms": round(self.total_saved_ms, 2)
        }


class LRUCache:
    """
    Thread-safe LRU cache optimized for low latency.
    
    Features:
    - O(1) get and put operations
    - Automatic expiration
    - Size limits with LRU eviction
    - Hit/miss statistics
    
    Usage:
        cache = LRUCache(max_size=1000, default_ttl=3600)
        
        # Check cache first
        result = cache.get(key)
        if result is None:
            result = expensive_operation()
            cache.put(key, result)
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None
    ):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def _make_key(self, key: Any) -> str:
        """Convert any key to a string hash."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Returns None if not found or expired.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[str_key]
            
            # Check expiration
            if entry.is_expired:
                del self._cache[str_key]
                self._stats.misses += 1
                return None
            
            # Update access info and move to end (most recently used)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._cache.move_to_end(str_key)
            
            self._stats.hits += 1
            return entry.value
    
    def put(
        self,
        key: Any,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Put a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        str_key = self._make_key(key)
        now = time.time()
        
        # Estimate size
        try:
            size_bytes = len(json.dumps(value).encode())
        except:
            size_bytes = 0
        
        with self._lock:
            # Remove if exists (to update position)
            if str_key in self._cache:
                del self._cache[str_key]
            
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
                self._stats.evictions += 1
            
            # Add new entry
            entry = CacheEntry(
                key=str_key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            self._cache[str_key] = entry
            self._stats.size = len(self._cache)
    
    def delete(self, key: Any) -> bool:
        """
        Delete an entry from the cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False if not found
        """
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key in self._cache:
                del self._cache[str_key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=self._stats.size,
                evictions=self._stats.evictions,
                total_saved_ms=self._stats.total_saved_ms
            )
    
    def record_time_saved(self, ms: float) -> None:
        """Record time saved by a cache hit."""
        with self._lock:
            self._stats.total_saved_ms += ms
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                removed += 1
            self._stats.size = len(self._cache)
        return removed
    
    def get_entry_info(self, key: Any) -> Optional[dict[str, Any]]:
        """Get metadata about a cache entry."""
        str_key = self._make_key(key)
        
        with self._lock:
            if str_key not in self._cache:
                return None
            
            entry = self._cache[str_key]
            return {
                "age_seconds": round(entry.age_seconds, 2),
                "access_count": entry.access_count,
                "size_bytes": entry.size_bytes,
                "is_expired": entry.is_expired,
                "ttl_remaining": round(
                    entry.ttl_seconds - entry.age_seconds, 2
                ) if entry.ttl_seconds else None
            }


class SemanticCache:
    """
    Cache that matches semantically similar queries.
    
    Uses word overlap similarity to find cached responses
    for queries that are similar but not identical.
    
    Note: This is a simplified version. In production,
    you'd use embeddings and a vector database.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 1000
    ):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity to return a match
            max_size: Maximum entries to store
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self._entries: list[tuple[str, Any, float]] = []
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def _simple_similarity(self, query1: str, query2: str) -> float:
        """
        Simple word-overlap similarity (Jaccard index).
        
        In production, use proper embeddings.
        """
        # Normalize and tokenize
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "in", "to", "for"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def get(self, query: str) -> Optional[Any]:
        """
        Get a cached response for a similar query.
        
        Args:
            query: Query to match
        
        Returns:
            Cached response or None
        """
        with self._lock:
            best_match = None
            best_similarity = 0.0
            
            for cached_query, response, _ in self._entries:
                similarity = self._simple_similarity(query, cached_query)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = response
            
            if best_similarity >= self.similarity_threshold:
                self._stats.hits += 1
                return best_match
            
            self._stats.misses += 1
            return None
    
    def put(self, query: str, response: Any) -> None:
        """
        Cache a response for a query.
        
        Args:
            query: The query
            response: The response to cache
        """
        with self._lock:
            # Remove oldest if at capacity
            if len(self._entries) >= self.max_size:
                self._entries.pop(0)
            
            self._entries.append((query, response, time.time()))
            self._stats.size = len(self._entries)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                size=len(self._entries)
            )


class CachedLLMClient:
    """
    LLM client wrapper with automatic caching.
    
    Caches responses to reduce both latency and costs.
    """
    
    def __init__(
        self,
        client: Any,
        cache: Optional[LRUCache] = None,
        cache_ttl: float = 3600
    ):
        """
        Initialize cached client.
        
        Args:
            client: Anthropic client instance
            cache: Cache to use (creates new if None)
            cache_ttl: Default cache TTL in seconds
        """
        self.client = client
        self.cache = cache or LRUCache(max_size=1000, default_ttl=cache_ttl)
        self.cache_ttl = cache_ttl
        self._avg_response_time_ms = 500.0  # Initial estimate
    
    def _make_cache_key(
        self,
        model: str,
        messages: list[dict],
        **kwargs: Any
    ) -> str:
        """Create a cache key from request parameters."""
        key_data = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k not in ["stream"]}
        }
        return json.dumps(key_data, sort_keys=True)
    
    def create_message(
        self,
        model: str,
        messages: list[dict],
        use_cache: bool = True,
        **kwargs: Any
    ) -> Any:
        """
        Create a message with caching.
        
        Args:
            model: Model to use
            messages: Message history
            use_cache: Whether to use cache
            **kwargs: Additional API parameters
        
        Returns:
            API response (cached or fresh)
        """
        if not use_cache:
            return self.client.messages.create(
                model=model,
                messages=messages,
                **kwargs
            )
        
        cache_key = self._make_cache_key(model, messages, **kwargs)
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            # Record estimated time saved
            self.cache.record_time_saved(self._avg_response_time_ms)
            return cached
        
        # Make API call
        start_time = time.perf_counter()
        response = self.client.messages.create(
            model=model,
            messages=messages,
            **kwargs
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Update average response time (exponential moving average)
        self._avg_response_time_ms = 0.9 * self._avg_response_time_ms + 0.1 * duration_ms
        
        # Cache the response
        self.cache.put(cache_key, response)
        
        return response
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        stats = self.cache.get_stats()
        return {
            **stats.to_dict(),
            "avg_response_time_ms": round(self._avg_response_time_ms, 2)
        }
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self.cache.clear()


class TieredCache:
    """
    Two-tier cache with fast in-memory and slower persistent storage.
    
    L1: In-memory LRU cache (fast, limited size)
    L2: Could be Redis, disk, etc. (slower, larger)
    
    This example uses two in-memory caches to demonstrate the pattern.
    """
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l1_ttl: float = 300,
        l2_ttl: float = 3600
    ):
        """
        Initialize tiered cache.
        
        Args:
            l1_size: L1 cache size
            l2_size: L2 cache size
            l1_ttl: L1 TTL in seconds
            l2_ttl: L2 TTL in seconds
        """
        self.l1 = LRUCache(max_size=l1_size, default_ttl=l1_ttl)
        self.l2 = LRUCache(max_size=l2_size, default_ttl=l2_ttl)
    
    def get(self, key: Any) -> Optional[Any]:
        """Get from cache, checking L1 first then L2."""
        # Check L1
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Check L2
        value = self.l2.get(key)
        if value is not None:
            # Promote to L1
            self.l1.put(key, value)
            return value
        
        return None
    
    def put(self, key: Any, value: Any) -> None:
        """Put in both L1 and L2."""
        self.l1.put(key, value)
        self.l2.put(key, value)
    
    def get_stats(self) -> dict[str, Any]:
        """Get stats for both tiers."""
        return {
            "l1": self.l1.get_stats().to_dict(),
            "l2": self.l2.get_stats().to_dict()
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("SPEED-OPTIMIZED CACHE DEMO")
    print("=" * 60)
    
    cache = LRUCache(max_size=100, default_ttl=3600)
    
    # Simulate caching scenario
    print("\nBasic Cache Operations:")
    print("-" * 40)
    
    # First request - cache miss
    result = cache.get("query_1")
    print(f"Get 'query_1': {result} (miss)")
    
    # Simulate expensive operation
    time.sleep(0.1)  # 100ms operation
    cache.put("query_1", "expensive result")
    cache.record_time_saved(100)
    
    # Second request - cache hit
    start = time.perf_counter()
    result = cache.get("query_1")
    duration = (time.perf_counter() - start) * 1000
    print(f"Get 'query_1': '{result}' (hit, {duration:.3f}ms)")
    
    # Add more entries
    for i in range(5):
        cache.put(f"query_{i+2}", f"result_{i+2}")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(json.dumps(stats.to_dict(), indent=2))
    
    # Get entry info
    print(f"\nEntry Info for 'query_1':")
    print(json.dumps(cache.get_entry_info("query_1"), indent=2))
    
    # Demonstrate semantic cache
    print("\n" + "-" * 40)
    print("Semantic Cache Demo:")
    print("-" * 40)
    
    semantic = SemanticCache(similarity_threshold=0.5)
    
    # Cache a response
    semantic.put("What is the weather in New York?", "Sunny, 72°F")
    
    # Try similar queries
    similar_queries = [
        "What's the weather in New York?",
        "Tell me New York weather",
        "Weather New York today",
        "How's the weather in Boston?",  # Different city
    ]
    
    for query in similar_queries:
        result = semantic.get(query)
        match_status = "✓ Match" if result else "✗ No match"
        print(f"{match_status}: '{query}'")
        if result:
            print(f"    Result: {result}")
    
    # Tiered cache demo
    print("\n" + "-" * 40)
    print("Tiered Cache Demo:")
    print("-" * 40)
    
    tiered = TieredCache(l1_size=10, l2_size=100)
    
    # Put some values
    for i in range(20):
        tiered.put(f"key_{i}", f"value_{i}")
    
    # Access some values (should be in L1 or promoted from L2)
    for i in [0, 5, 15, 19]:
        result = tiered.get(f"key_{i}")
        print(f"Get 'key_{i}': {result}")
    
    print(f"\nTiered Cache Stats:")
    print(json.dumps(tiered.get_stats(), indent=2))
