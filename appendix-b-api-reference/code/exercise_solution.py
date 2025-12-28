"""
Exercise Solution: Production-Ready API Client

Build a robust API client that combines error handling, rate limiting,
and token tracking.

Appendix B: API Reference Quick Guide
"""

import os
from dotenv import load_dotenv
import anthropic
import time
from dataclasses import dataclass
from typing import Optional, Callable, Any
from threading import Lock

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class TokenUsage:
    """Track token usage."""
    input_tokens: int = 0
    output_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def estimate_cost(
        self,
        input_cost_per_million: float = 3.00,
        output_cost_per_million: float = 15.00
    ) -> float:
        """Estimate cost in dollars."""
        input_cost = (self.input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (self.output_tokens / 1_000_000) * output_cost_per_million
        return input_cost + output_cost
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()
    
    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """Acquire tokens from the bucket."""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Refill tokens
                self.tokens = min(
                    self.capacity,
                    self.tokens + elapsed * self.rate
                )
                self.last_update = now
                
                # Try to acquire
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                if not block:
                    return False
                
                # Calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
            
            time.sleep(wait_time)


class RobustAPIClient:
    """
    Production-ready Anthropic API client.
    
    Features:
    - Comprehensive error handling
    - Automatic retry with exponential backoff
    - Rate limiting with token bucket
    - Token usage tracking
    - Cost estimation
    - Request validation
    
    Usage:
        client = RobustAPIClient(
            requests_per_minute=10,
            max_retries=3
        )
        
        response = client.create_message(
            "What is Python?",
            max_tokens=200
        )
        
        print(response)
        print(client.get_usage_summary())
    """
    
    def __init__(
        self,
        requests_per_minute: int = 10,
        burst_capacity: Optional[int] = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0
    ):
        """
        Initialize the robust API client.
        
        Args:
            requests_per_minute: Rate limit (requests per minute)
            burst_capacity: Burst capacity (defaults to 2x rate)
            max_retries: Maximum retry attempts
            initial_retry_delay: Initial delay for exponential backoff
        """
        self.client = anthropic.Anthropic()
        
        # Rate limiting
        rate_per_second = requests_per_minute / 60
        capacity = burst_capacity or (requests_per_minute * 2)
        self.rate_limiter = TokenBucket(rate=rate_per_second, capacity=capacity)
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Usage tracking
        self.total_usage = TokenUsage()
        self.call_count = 0
        self.error_count = 0
        self.retry_count = 0
    
    def validate_request(
        self,
        model: str,
        max_tokens: int,
        messages: list
    ) -> bool:
        """
        Validate request parameters.
        
        Raises:
            ValueError: If parameters are invalid
        """
        if not model:
            raise ValueError("model is required")
        
        if not max_tokens or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        
        if max_tokens > 200000:
            raise ValueError("max_tokens cannot exceed 200,000")
        
        if not messages or len(messages) == 0:
            raise ValueError("messages cannot be empty")
        
        return True
    
    def create_message(
        self,
        content: str,
        max_tokens: int = 1024,
        model: str = "claude-sonnet-4-20250514",
        system: Optional[str] = None,
        temperature: float = 1.0
    ) -> str:
        """
        Create a message with robust error handling.
        
        Args:
            content: User message content
            max_tokens: Maximum tokens to generate
            model: Model to use
            system: Optional system prompt
            temperature: Temperature (0.0-1.0)
            
        Returns:
            Response text
            
        Raises:
            Various exceptions after retry attempts exhausted
        """
        messages = [{"role": "user", "content": content}]
        
        # Validate request
        self.validate_request(model, max_tokens, messages)
        
        # Acquire rate limit token (blocks if necessary)
        self.rate_limiter.acquire()
        
        # Retry loop
        delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries):
            try:
                # Build request parameters
                params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "temperature": temperature
                }
                
                if system:
                    params["system"] = system
                
                # Make API call
                response = self.client.messages.create(**params)
                
                # Track usage
                usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens
                )
                self.total_usage = self.total_usage + usage
                self.call_count += 1
                
                return response.content[0].text
                
            except anthropic.RateLimitError as e:
                self.retry_count += 1
                
                if attempt == self.max_retries - 1:
                    self.error_count += 1
                    raise
                
                wait_time = delay * (2 ** attempt)
                print(f"Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except anthropic.InternalServerError as e:
                self.retry_count += 1
                
                if attempt == self.max_retries - 1:
                    self.error_count += 1
                    raise
                
                print(f"Server error. Retrying in 5s...")
                time.sleep(5)
                
            except anthropic.APIConnectionError as e:
                self.error_count += 1
                raise Exception("Connection error. Check your internet connection.")
                
            except anthropic.AuthenticationError as e:
                self.error_count += 1
                raise Exception("Invalid API key. Check ANTHROPIC_API_KEY.")
                
            except anthropic.PermissionDeniedError as e:
                self.error_count += 1
                raise Exception("Request violates content policy.")
                
            except anthropic.NotFoundError as e:
                self.error_count += 1
                raise Exception(f"Invalid model or endpoint: {model}")
                
            except anthropic.BadRequestError as e:
                self.error_count += 1
                raise Exception(f"Invalid request parameters: {e}")
                
            except Exception as e:
                self.error_count += 1
                raise Exception(f"Unexpected error: {type(e).__name__}: {e}")
        
        self.error_count += 1
        raise Exception("Max retries exceeded")
    
    def get_usage_summary(self) -> str:
        """Get a summary of API usage."""
        cost = self.total_usage.estimate_cost()
        avg_tokens = (
            self.total_usage.total_tokens / self.call_count
            if self.call_count > 0 else 0
        )
        
        return (
            f"API Usage Summary\n"
            f"{'='*50}\n"
            f"Total Calls:       {self.call_count:,}\n"
            f"Successful:        {self.call_count - self.error_count:,}\n"
            f"Errors:            {self.error_count:,}\n"
            f"Retries:           {self.retry_count:,}\n"
            f"{'='*50}\n"
            f"Input Tokens:      {self.total_usage.input_tokens:,}\n"
            f"Output Tokens:     {self.total_usage.output_tokens:,}\n"
            f"Total Tokens:      {self.total_usage.total_tokens:,}\n"
            f"{'='*50}\n"
            f"Avg Tokens/Call:   {avg_tokens:.1f}\n"
            f"Total Cost:        ${cost:.4f}"
        )
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.total_usage = TokenUsage()
        self.call_count = 0
        self.error_count = 0
        self.retry_count = 0


def demonstrate_robust_client():
    """Demonstrate the robust API client."""
    print("=== Robust API Client Demo ===\n")
    
    # Create client with rate limiting
    client = RobustAPIClient(
        requests_per_minute=10,  # 10 requests per minute
        burst_capacity=5,        # Allow bursts of 5
        max_retries=3            # Retry up to 3 times
    )
    
    # Make several API calls
    queries = [
        "What is Python?",
        "Explain object-oriented programming briefly.",
        "What are decorators in Python?",
        "Describe list comprehensions.",
        "What is the GIL?"
    ]
    
    print("Making multiple API calls with rate limiting...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        
        try:
            response = client.create_message(
                content=query,
                max_tokens=150,
                temperature=0.7
            )
            
            print(f"✓ Response: {response[:100]}...")
            print()
            
        except Exception as e:
            print(f"✗ Error: {e}\n")
    
    # Print usage summary
    print("\n" + client.get_usage_summary())


def demonstrate_error_recovery():
    """Demonstrate error recovery."""
    print("\n\n=== Error Recovery Demo ===\n")
    
    client = RobustAPIClient(requests_per_minute=5)
    
    # Test with invalid parameters
    print("1. Testing with invalid parameters...")
    try:
        response = client.create_message(
            content="Hello",
            max_tokens=0  # Invalid
        )
    except Exception as e:
        print(f"   ✓ Caught error: {e}\n")
    
    # Test with valid parameters
    print("2. Testing with valid parameters...")
    try:
        response = client.create_message(
            content="Say hello in one sentence",
            max_tokens=50
        )
        print(f"   ✓ Success: {response}\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Show stats
    print(client.get_usage_summary())


def demonstrate_budget_aware():
    """Demonstrate budget-aware usage."""
    print("\n\n=== Budget-Aware Usage Demo ===\n")
    
    client = RobustAPIClient(requests_per_minute=10)
    budget = 0.05  # $0.05 budget
    
    print(f"Budget: ${budget:.2f}\n")
    
    queries = [
        "Explain neural networks briefly",
        "What is deep learning?",
        "Describe transformers",
        "What are attention mechanisms?"
    ]
    
    for i, query in enumerate(queries, 1):
        # Check budget before making call
        current_cost = client.total_usage.estimate_cost()
        
        if current_cost >= budget:
            print(f"⚠️  Budget limit reached after {i-1} queries")
            print(f"Spent: ${current_cost:.4f}")
            break
        
        remaining = budget - current_cost
        print(f"Query {i}: {query}")
        print(f"  Budget remaining: ${remaining:.4f}")
        
        try:
            response = client.create_message(
                content=query,
                max_tokens=100
            )
            
            cost = client.total_usage.estimate_cost()
            print(f"  ✓ Response received")
            print(f"  Total spent: ${cost:.4f}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print(client.get_usage_summary())


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_robust_client()
    demonstrate_error_recovery()
    demonstrate_budget_aware()
