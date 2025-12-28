"""
Rate limiting and abuse prevention.

Chapter 41: Security Considerations
"""

import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class RateLimitResult(Enum):
    """Result of a rate limit check."""
    ALLOWED = "allowed"
    LIMITED = "limited"
    BLOCKED = "blocked"


@dataclass
class ClientInfo:
    """Information about a client for rate limiting."""
    client_id: str
    requests: list[float] = field(default_factory=list)
    violations: int = 0
    blocked_until: Optional[float] = None
    total_requests: int = 0
    
    def is_blocked(self) -> bool:
        """Check if client is currently blocked."""
        if self.blocked_until is None:
            return False
        return time.time() < self.blocked_until


class RateLimiter:
    """
    Comprehensive rate limiter with multiple strategies.
    
    Features:
    - Sliding window rate limiting
    - Per-client tracking
    - Automatic blocking for repeat violators
    - Configurable limits
    
    Usage:
        limiter = RateLimiter(requests_per_minute=60)
        
        result = limiter.check("user_123")
        if result == RateLimitResult.ALLOWED:
            # Process request
        else:
            # Reject with 429
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_limit: int = 10,
        block_duration_seconds: int = 300,
        violation_threshold: int = 5
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour
            burst_limit: Maximum requests per second (burst)
            block_duration_seconds: How long to block repeat violators
            violation_threshold: Violations before blocking
        """
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.burst = burst_limit
        self.block_duration = block_duration_seconds
        self.violation_threshold = violation_threshold
        
        self.clients: dict[str, ClientInfo] = {}
    
    def _get_client(self, client_id: str) -> ClientInfo:
        """Get or create client info."""
        if client_id not in self.clients:
            self.clients[client_id] = ClientInfo(client_id=client_id)
        return self.clients[client_id]
    
    def _clean_old_requests(self, client: ClientInfo, now: float) -> None:
        """Remove requests older than 1 hour."""
        cutoff = now - 3600  # 1 hour
        client.requests = [t for t in client.requests if t > cutoff]
    
    def check(self, client_id: str) -> RateLimitResult:
        """
        Check if a request should be allowed.
        
        Args:
            client_id: Unique client identifier
        
        Returns:
            RateLimitResult indicating if request is allowed
        """
        now = time.time()
        client = self._get_client(client_id)
        
        # Clean old requests
        self._clean_old_requests(client, now)
        
        # Check if blocked
        if client.is_blocked():
            return RateLimitResult.BLOCKED
        
        # Check burst limit (requests in last second)
        recent_second = len([t for t in client.requests if t > now - 1])
        if recent_second >= self.burst:
            self._record_violation(client, now)
            return RateLimitResult.LIMITED
        
        # Check per-minute limit
        recent_minute = len([t for t in client.requests if t > now - 60])
        if recent_minute >= self.rpm:
            self._record_violation(client, now)
            return RateLimitResult.LIMITED
        
        # Check per-hour limit
        if len(client.requests) >= self.rph:
            self._record_violation(client, now)
            return RateLimitResult.LIMITED
        
        # Allow and record
        client.requests.append(now)
        client.total_requests += 1
        
        return RateLimitResult.ALLOWED
    
    def _record_violation(self, client: ClientInfo, now: float) -> None:
        """Record a rate limit violation."""
        client.violations += 1
        
        # Block repeat violators
        if client.violations >= self.violation_threshold:
            client.blocked_until = now + self.block_duration
    
    def get_client_status(self, client_id: str) -> dict:
        """Get the current status for a client."""
        client = self._get_client(client_id)
        now = time.time()
        
        self._clean_old_requests(client, now)
        
        last_minute = len([t for t in client.requests if t > now - 60])
        
        return {
            "client_id": client_id,
            "requests_last_minute": last_minute,
            "requests_last_hour": len(client.requests),
            "total_requests": client.total_requests,
            "violations": client.violations,
            "is_blocked": client.is_blocked(),
            "blocked_until": client.blocked_until,
            "remaining_minute": max(0, self.rpm - last_minute),
            "remaining_hour": max(0, self.rph - len(client.requests))
        }
    
    def unblock(self, client_id: str) -> bool:
        """Manually unblock a client."""
        if client_id in self.clients:
            self.clients[client_id].blocked_until = None
            self.clients[client_id].violations = 0
            return True
        return False
    
    def reset_client(self, client_id: str) -> bool:
        """Completely reset a client's state."""
        if client_id in self.clients:
            del self.clients[client_id]
            return True
        return False


class AbuseDetector:
    """
    Detects patterns of abuse beyond simple rate limiting.
    
    Looks for:
    - Repeated identical requests (automated attacks)
    - Sequential scanning patterns
    - Credential stuffing attempts
    """
    
    def __init__(
        self,
        duplicate_threshold: int = 5,
        window_seconds: int = 60
    ):
        """
        Initialize the abuse detector.
        
        Args:
            duplicate_threshold: How many duplicates before flagging
            window_seconds: Time window for duplicate detection
        """
        self.duplicate_threshold = duplicate_threshold
        self.window = window_seconds
        
        # Track request hashes per client
        self.request_history: dict[str, list[tuple[float, str]]] = defaultdict(list)
    
    def _hash_request(self, content: str) -> str:
        """Create a hash of request content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def check(self, client_id: str, request_content: str) -> tuple[bool, str]:
        """
        Check for abuse patterns.
        
        Args:
            client_id: Client identifier
            request_content: The request content to analyze
        
        Returns:
            Tuple of (is_suspicious, reason)
        """
        now = time.time()
        request_hash = self._hash_request(request_content)
        
        # Clean old entries
        cutoff = now - self.window
        self.request_history[client_id] = [
            (t, h) for t, h in self.request_history[client_id]
            if t > cutoff
        ]
        
        # Check for duplicates
        duplicates = sum(
            1 for _, h in self.request_history[client_id]
            if h == request_hash
        )
        
        # Record this request
        self.request_history[client_id].append((now, request_hash))
        
        if duplicates >= self.duplicate_threshold:
            return True, f"Duplicate request pattern detected ({duplicates} identical requests)"
        
        # Check for high-frequency unique requests (scanning)
        if len(self.request_history[client_id]) > 100:
            unique_hashes = set(h for _, h in self.request_history[client_id])
            if len(unique_hashes) > 90:  # >90% unique in 100 requests
                return True, "Scanning pattern detected"
        
        return False, ""
    
    def reset_client(self, client_id: str) -> None:
        """Reset tracking for a client."""
        if client_id in self.request_history:
            del self.request_history[client_id]


# Example usage
if __name__ == "__main__":
    print("Rate Limiting Demo")
    print("=" * 60)
    
    limiter = RateLimiter(
        requests_per_minute=10,
        burst_limit=3,
        violation_threshold=3
    )
    
    # Simulate requests
    client = "user_123"
    
    print(f"\nSimulating requests for {client}...")
    print(f"Limits: {limiter.rpm}/min, {limiter.burst}/sec burst")
    print()
    
    for i in range(15):
        result = limiter.check(client)
        status = limiter.get_client_status(client)
        
        print(f"Request {i+1}: {result.value}")
        print(f"  Remaining: {status['remaining_minute']}/min")
        
        if result == RateLimitResult.BLOCKED:
            print(f"  BLOCKED until: {status['blocked_until']}")
            break
        
        time.sleep(0.1)  # Small delay
    
    # Abuse detection demo
    print("\n" + "=" * 60)
    print("Abuse Detection Demo")
    print("=" * 60)
    
    detector = AbuseDetector(duplicate_threshold=3)
    
    # Normal requests
    requests = [
        "What is Python?",
        "Tell me about JavaScript",
        "What is Python?",  # Duplicate
        "Explain machine learning",
        "What is Python?",  # Duplicate
        "What is Python?",  # Duplicate (should trigger)
    ]
    
    for i, req in enumerate(requests):
        is_suspicious, reason = detector.check("user_456", req)
        print(f"\nRequest {i+1}: {req[:30]}...")
        if is_suspicious:
            print(f"  ⚠️ SUSPICIOUS: {reason}")
        else:
            print("  ✓ OK")