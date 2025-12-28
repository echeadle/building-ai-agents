"""
Rate limiting middleware for AI agent APIs.

Chapter 40: Deployment Patterns

Rate limiting protects your API from:
- Abuse and denial of service
- Accidental overuse
- Cost overruns from API calls

This module provides both in-memory and Redis-based rate limiters.
"""

import time
from collections import defaultdict
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter.
    
    Uses a sliding window algorithm to track requests.
    
    Limitations:
    - Not suitable for distributed systems (use Redis version)
    - State lost on restart
    - Memory grows with unique clients
    
    Usage:
        limiter = InMemoryRateLimiter(requests_per_minute=60)
        
        if limiter.is_allowed("client_123"):
            # Process request
        else:
            # Return 429 Too Many Requests
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        window_seconds: int = 60
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Max requests allowed per window
            window_seconds: Size of the sliding window in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a client is allowed to make a request.
        
        Args:
            client_id: Unique identifier for the client (IP, API key, etc.)
        
        Returns:
            True if request is allowed, False if rate limited
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Remove expired requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if t > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Record this request
        self.requests[client_id].append(now)
        return True
    
    def remaining(self, client_id: str) -> int:
        """Get remaining requests for a client in the current window."""
        now = time.time()
        window_start = now - self.window_seconds
        
        current = len([
            t for t in self.requests.get(client_id, [])
            if t > window_start
        ])
        
        return max(0, self.requests_per_minute - current)
    
    def reset_time(self, client_id: str) -> int:
        """Get seconds until the client's rate limit resets."""
        if client_id not in self.requests or not self.requests[client_id]:
            return 0
        
        oldest = min(self.requests[client_id])
        reset_at = oldest + self.window_seconds
        return max(0, int(reset_at - time.time()))
    
    def cleanup(self):
        """Remove expired entries to free memory."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean each client's requests
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if t > window_start
            ]
            # Remove empty entries
            if not self.requests[client_id]:
                del self.requests[client_id]


class RedisRateLimiter:
    """
    Redis-based rate limiter for distributed systems.
    
    Uses Redis sorted sets for efficient sliding window implementation.
    Works across multiple server instances.
    
    Requires: pip install redis
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        requests_per_minute: int = 60,
        window_seconds: int = 60,
        key_prefix: str = "ratelimit:"
    ):
        """
        Initialize Redis rate limiter.
        
        Args:
            redis_url: Redis connection URL
            requests_per_minute: Max requests per window
            window_seconds: Window size in seconds
            key_prefix: Prefix for Redis keys
        """
        import redis
        
        self.redis = redis.from_url(redis_url)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds
        key = f"{self.key_prefix}{client_id}"
        
        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        pipe.zcard(key)
        
        # Execute
        _, count = pipe.execute()
        
        if count >= self.requests_per_minute:
            return False
        
        # Add new entry
        pipe = self.redis.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self.window_seconds + 1)
        pipe.execute()
        
        return True
    
    def remaining(self, client_id: str) -> int:
        """Get remaining requests."""
        now = time.time()
        window_start = now - self.window_seconds
        key = f"{self.key_prefix}{client_id}"
        
        # Clean and count
        self.redis.zremrangebyscore(key, 0, window_start)
        count = self.redis.zcard(key)
        
        return max(0, self.requests_per_minute - count)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    
    Adds rate limiting to all routes (except excluded paths).
    
    Usage:
        app = FastAPI()
        limiter = InMemoryRateLimiter(requests_per_minute=60)
        app.add_middleware(
            RateLimitMiddleware,
            limiter=limiter,
            exclude_paths=["/health", "/metrics"]
        )
    """
    
    def __init__(
        self,
        app: FastAPI,
        limiter: InMemoryRateLimiter,
        exclude_paths: Optional[list[str]] = None,
        client_id_header: str = "X-API-Key"
    ):
        """
        Initialize middleware.
        
        Args:
            app: FastAPI application
            limiter: Rate limiter instance
            exclude_paths: Paths to exclude from rate limiting
            client_id_header: Header to use for client identification
        """
        super().__init__(app)
        self.limiter = limiter
        self.exclude_paths = exclude_paths or ["/health", "/health/live", "/health/ready"]
        self.client_id_header = client_id_header
    
    def get_client_id(self, request: Request) -> str:
        """
        Get client identifier from request.
        
        Priority:
        1. X-API-Key header (for authenticated clients)
        2. Client IP address (fallback)
        """
        # Try API key header first
        api_key = request.headers.get(self.client_id_header)
        if api_key:
            return f"key:{api_key}"
        
        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Get first IP in chain (original client)
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        if request.client:
            return f"ip:{request.client.host}"
        
        return "ip:unknown"
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with rate limiting."""
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)
        
        client_id = self.get_client_id(request)
        
        # Check rate limit
        if not self.limiter.is_allowed(client_id):
            return Response(
                content='{"detail": "Rate limit exceeded. Please try again later."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={
                    "Retry-After": str(self.limiter.reset_time(client_id)),
                    "X-RateLimit-Limit": str(self.limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(self.limiter.reset_time(client_id))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.limiter.remaining(client_id))
        response.headers["X-RateLimit-Reset"] = str(self.limiter.reset_time(client_id))
        
        return response


class TieredRateLimiter:
    """
    Rate limiter with different limits for different tiers.
    
    Useful for freemium APIs with different rate limits
    for free vs paid users.
    """
    
    def __init__(self, tiers: dict[str, int]):
        """
        Initialize tiered limiter.
        
        Args:
            tiers: Dict mapping tier name to requests per minute
                   e.g., {"free": 10, "pro": 100, "enterprise": 1000}
        """
        self.tiers = tiers
        self.limiters = {
            tier: InMemoryRateLimiter(requests_per_minute=rpm)
            for tier, rpm in tiers.items()
        }
        self.client_tiers: dict[str, str] = {}  # client_id -> tier
    
    def set_tier(self, client_id: str, tier: str):
        """Set a client's tier."""
        if tier not in self.tiers:
            raise ValueError(f"Unknown tier: {tier}")
        self.client_tiers[client_id] = tier
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed based on client's tier."""
        tier = self.client_tiers.get(client_id, "free")
        return self.limiters[tier].is_allowed(client_id)
    
    def remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        tier = self.client_tiers.get(client_id, "free")
        return self.limiters[tier].remaining(client_id)


# ----- Example Application -----

if __name__ == "__main__":
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI(
        title="Rate Limiting Demo",
        version="1.0.0"
    )
    
    # Create rate limiter
    limiter = InMemoryRateLimiter(requests_per_minute=10)
    
    # Add middleware
    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        exclude_paths=["/health", "/"]
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "Rate Limiting Demo",
            "note": "Try hitting /test more than 10 times per minute"
        }
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "Request successful!"}
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @app.get("/stats")
    async def stats():
        """Get rate limiter statistics."""
        return {
            "total_tracked_clients": len(limiter.requests),
            "limit_per_minute": limiter.requests_per_minute
        }
    
    print("=" * 60)
    print("RATE LIMITING DEMO")
    print("=" * 60)
    print()
    print("Limit: 10 requests per minute")
    print()
    print("Test it:")
    print('  for i in {1..15}; do curl http://localhost:8000/test; echo; done')
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
