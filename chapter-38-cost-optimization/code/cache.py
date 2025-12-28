"""
Caching strategies for AI agents.

Chapter 38: Cost Optimization
"""

import hashlib
import json
import time
from typing import Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    """A single cache entry."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ResponseCache:
    """In-memory cache for LLM responses with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, prompt: str, **kwargs) -> str:
        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        key = self._make_key(prompt, **kwargs)
        entry = self._cache.get(key)
        
        if entry is None:
            self.misses += 1
            return None
        
        if entry.is_expired():
            del self._cache[key]
            self._access_order.remove(key)
            self.misses += 1
            return None
        
        self._access_order.remove(key)
        self._access_order.append(key)
        entry.hit_count += 1
        self.hits += 1
        return entry.value
    
    def set(self, prompt: str, value: Any, **kwargs) -> None:
        key = self._make_key(prompt, **kwargs)
        
        while len(self._cache) >= self.max_size:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
        
        expires_at = time.time() + self.ttl_seconds if self.ttl_seconds else None
        
        entry = CacheEntry(
            key=key, value=value,
            created_at=time.time(), expires_at=expires_at
        )
        
        self._cache[key] = entry
        self._access_order.append(key)
    
    def get_stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


class SemanticCache:
    """Cache with semantic normalization for similar queries."""
    
    def __init__(self, base_cache: Optional[ResponseCache] = None):
        self.cache = base_cache or ResponseCache()
        self._stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be",
            "to", "of", "in", "for", "on", "with", "at", "by",
            "please", "thanks", "you", "i", "me", "my",
        }
    
    def _normalize(self, text: str) -> str:
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        words = [w for w in text.split() if w not in self._stop_words]
        words.sort()
        return ' '.join(words)
    
    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        normalized = self._normalize(prompt)
        return self.cache.get(normalized, **kwargs)
    
    def set(self, prompt: str, value: Any, **kwargs) -> None:
        normalized = self._normalize(prompt)
        self.cache.set(normalized, value, **kwargs)
    
    def get_stats(self) -> dict:
        return self.cache.get_stats()


if __name__ == "__main__":
    print("Caching Demo")
    print("=" * 50)
    
    cache = ResponseCache(max_size=100, ttl_seconds=3600)
    
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of France?",  # Duplicate
        "What is the capital of France?",  # Duplicate
    ]
    
    for prompt in prompts:
        cached = cache.get(prompt)
        if cached:
            print(f"HIT: {prompt[:30]}...")
        else:
            print(f"MISS: {prompt[:30]}...")
            cache.set(prompt, f"Response for: {prompt}")
    
    print(f"\nStats: {cache.get_stats()}")
