"""
Token bucket rate limiter for API calls.

Appendix B: API Reference Quick Guide
"""

import time
import os
from dotenv import load_dotenv
import anthropic
from threading import Lock

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows burst traffic while enforcing average rate limits.
    Each API call consumes one token. Tokens refill at a constant rate.
    
    Example:
        limiter = TokenBucket(rate=10, capacity=20)
        # Allows 10 requests/sec average, with bursts up to 20
        
        if limiter.acquire():
            response = client.messages.create(...)
        else:
            print("Rate limit reached, try again later")
    
    How it works:
        - Bucket starts with 'capacity' tokens
        - Tokens refill at 'rate' per second
        - Each request consumes 1 token
        - If no tokens available:
          * Non-blocking: Returns False
          * Blocking: Waits until token available
    """
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize the token bucket.
        
        Args:
            rate: Tokens added per second (requests per second)
            capacity: Maximum tokens (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            block: If True, wait until tokens are available.
                   If False, return immediately.
            
        Returns:
            True if tokens acquired, False if not available (non-blocking only)
        """
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Add tokens based on elapsed time
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now
                
                # Try to acquire tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # If not blocking, return False immediately
                if not block:
                    return False
                
                # Calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
            
            # Wait outside the lock to allow other threads
            time.sleep(wait_time)
    
    def available_tokens(self) -> float:
        """
        Get current number of available tokens.
        
        Returns:
            Number of tokens currently available
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            return min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )


def demonstrate_token_bucket():
    """Demonstrate token bucket rate limiting."""
    client = anthropic.Anthropic()
    
    print("=== Token Bucket Demo ===\n")
    
    # Allow 5 requests per minute = 5/60 per second
    # Burst capacity of 3 requests
    limiter = TokenBucket(rate=5/60, capacity=3)
    
    print(f"Rate: 5 requests/minute (burst of 3)")
    print(f"Making 10 requests...\n")
    
    for i in range(10):
        start = time.time()
        
        # Check available tokens before acquiring
        available = limiter.available_tokens()
        print(f"Request {i + 1} - Available tokens: {available:.2f}")
        
        # Acquire token (blocks if necessary)
        limiter.acquire()
        
        elapsed = time.time() - start
        if elapsed > 0.1:
            print(f"  ⏱️  Waited {elapsed:.1f}s for rate limit")
        
        # Make the API call
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": f"Count to {i + 1}"}]
        )
        
        print(f"  ✓ {response.content[0].text[:40]}...\n")


def demonstrate_non_blocking():
    """Demonstrate non-blocking token acquisition."""
    client = anthropic.Anthropic()
    
    print("=== Non-Blocking Token Bucket Demo ===\n")
    
    # Very restrictive: 1 request per 5 seconds, no burst
    limiter = TokenBucket(rate=0.2, capacity=1)
    
    print("Rate: 1 request per 5 seconds")
    print("Attempting rapid requests...\n")
    
    successful = 0
    rate_limited = 0
    
    for i in range(5):
        # Try to acquire without blocking
        if limiter.acquire(block=False):
            print(f"Request {i + 1}: ✓ Allowed")
            successful += 1
            
            # Make actual API call
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=30,
                messages=[{"role": "user", "content": "Hi"}]
            )
            print(f"  Response: {response.content[0].text}\n")
        else:
            print(f"Request {i + 1}: ✗ Rate limited\n")
            rate_limited += 1
        
        time.sleep(0.5)  # Small delay between attempts
    
    print(f"Results: {successful} successful, {rate_limited} rate limited")


def demonstrate_burst_capacity():
    """Demonstrate burst capacity handling."""
    print("\n=== Burst Capacity Demo ===\n")
    
    # Allow 2 requests/second, burst of 10
    limiter = TokenBucket(rate=2.0, capacity=10)
    
    print("Rate: 2 requests/second, burst capacity: 10")
    print("Bucket starts full (10 tokens)\n")
    
    # Burst: Use all 10 tokens immediately
    print("Burst phase: Using 10 tokens rapidly...")
    for i in range(10):
        acquired = limiter.acquire(block=False)
        print(f"  Token {i + 1}: {'✓' if acquired else '✗'}")
    
    print(f"\nTokens remaining: {limiter.available_tokens():.2f}")
    
    # Now we're rate limited
    print("\nRate-limited phase: Waiting for tokens to refill...")
    for i in range(3):
        print(f"  Request {i + 1}...")
        start = time.time()
        limiter.acquire(block=True)
        elapsed = time.time() - start
        print(f"    Waited {elapsed:.2f}s")


def demonstrate_multiple_tokens():
    """Show acquiring multiple tokens at once."""
    print("\n=== Multiple Token Acquisition Demo ===\n")
    
    limiter = TokenBucket(rate=5.0, capacity=10)
    
    print("Rate: 5 tokens/second, capacity: 10")
    print("Some operations cost more tokens\n")
    
    # Simulate different operation costs
    operations = [
        ("Small query", 1),
        ("Medium query", 3),
        ("Large query", 5),
        ("Small query", 1),
    ]
    
    for op_name, cost in operations:
        available = limiter.available_tokens()
        print(f"{op_name} (costs {cost} tokens)")
        print(f"  Available: {available:.2f}")
        
        if limiter.acquire(tokens=cost, block=False):
            print(f"  ✓ Allowed\n")
        else:
            print(f"  ✗ Not enough tokens, would need to wait\n")


if __name__ == "__main__":
    # Run basic demo
    demonstrate_token_bucket()
    
    print("\n" + "="*60 + "\n")
    
    # Show non-blocking mode
    demonstrate_non_blocking()
    
    print("\n" + "="*60 + "\n")
    
    # Show burst capacity
    demonstrate_burst_capacity()
    
    print("\n" + "="*60 + "\n")
    
    # Show multiple token acquisition
    demonstrate_multiple_tokens()
