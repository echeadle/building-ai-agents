"""
Basic rate limit handling with exponential backoff.

Appendix B: API Reference Quick Guide
"""

import time
import os
from dotenv import load_dotenv
import anthropic
from typing import Optional, Callable, Any

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class RateLimitHandler:
    """
    Handle rate limits with exponential backoff.
    
    This handler automatically retries API calls when rate limited,
    using exponential backoff to avoid overwhelming the API.
    
    Usage:
        handler = RateLimitHandler()
        response = handler.call_with_retry(
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello!"}]
            )
        )
    """
    
    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        max_delay: float = 60.0
    ):
        """
        Initialize the rate limit handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (doubles each retry)
            max_delay: Maximum delay between retries in seconds
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
    
    def call_with_retry(
        self,
        api_call: Callable[[], Any]
    ) -> Any:
        """
        Execute an API call with retry logic.
        
        If the call is rate limited, automatically retries with
        exponential backoff: 1s, 2s, 4s, 8s, etc.
        
        Args:
            api_call: Function that makes the API call
            
        Returns:
            API response
            
        Raises:
            RateLimitError: If max retries exceeded
            Other exceptions: Passes through other errors
        """
        delay = self.initial_delay
        
        for attempt in range(self.max_retries):
            try:
                # Try the API call
                return api_call()
                
            except anthropic.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    # Last attempt, give up
                    print(f"Max retries ({self.max_retries}) exceeded. Giving up.")
                    raise
                
                # Calculate delay with exponential backoff
                wait_time = min(delay * (2 ** attempt), self.max_delay)
                
                print(
                    f"Rate limited (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                time.sleep(wait_time)
            
            except Exception as e:
                # Don't retry other types of errors
                print(f"Non-rate-limit error: {type(e).__name__}: {e}")
                raise
        
        raise anthropic.RateLimitError("Max retries exceeded")


def demonstrate_rate_limit_handling():
    """Demonstrate the rate limit handler in action."""
    client = anthropic.Anthropic()
    handler = RateLimitHandler(max_retries=3, initial_delay=1.0)
    
    print("=== Rate Limit Handler Demo ===\n")
    
    # Make several quick requests to potentially trigger rate limiting
    for i in range(5):
        print(f"Request {i + 1}...")
        
        try:
            response = handler.call_with_retry(
                lambda: client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=100,
                    messages=[{"role": "user", "content": f"Say hello {i + 1}"}]
                )
            )
            
            print(f"✓ Success: {response.content[0].text[:50]}...")
            print(f"  Tokens: {response.usage.input_tokens + response.usage.output_tokens}\n")
            
        except anthropic.RateLimitError as e:
            print(f"✗ Failed after all retries: {e}\n")
            break
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {e}\n")
            break


def demonstrate_manual_backoff():
    """Show manual exponential backoff implementation."""
    client = anthropic.Anthropic()
    
    print("=== Manual Backoff Demo ===\n")
    
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=50,
                messages=[{"role": "user", "content": "Hello!"}]
            )
            
            print(f"✓ Success: {response.content[0].text}\n")
            break
            
        except anthropic.RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"✗ Max retries exceeded\n")
                raise
            
            # Exponential backoff: 1s, 2s, 4s
            delay = base_delay * (2 ** attempt)
            print(f"  Rate limited. Waiting {delay}s...\n")
            time.sleep(delay)


if __name__ == "__main__":
    # Run the handler demo
    demonstrate_rate_limit_handling()
    
    # Show manual implementation
    print("\n" + "="*50 + "\n")
    demonstrate_manual_backoff()
