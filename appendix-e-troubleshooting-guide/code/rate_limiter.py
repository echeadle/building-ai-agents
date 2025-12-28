"""
Rate Limiting Utility for AI Agents

Appendix E: Troubleshooting Guide
"""

import os
from dotenv import load_dotenv
import anthropic
import time
from typing import Optional
from dataclasses import dataclass, field

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class RateLimiter:
    """
    Simple rate limiter to prevent hitting API limits.
    
    Tracks request times and enforces a maximum requests per minute limit.
    """
    
    requests_per_minute: int = 50
    request_times: list[float] = field(default_factory=list)
    
    def wait_if_needed(self) -> Optional[float]:
        """
        Check if we need to wait before making another request.
        
        Returns:
            Number of seconds waited, or None if no wait was needed
        """
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times 
            if now - t < 60
        ]
        
        # If we're at the limit, calculate wait time
        if len(self.request_times) >= self.requests_per_minute:
            oldest_request = self.request_times[0]
            wait_time = 60 - (now - oldest_request) + 1  # +1 second for safety
            
            print(f"⏳ Rate limit: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
            
            return wait_time
        
        # Record this request
        self.request_times.append(now)
        return None
    
    def get_current_rate(self) -> float:
        """Get current requests per minute."""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]
        return len(recent_requests)
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without waiting."""
        now = time.time()
        recent_requests = [t for t in self.request_times if now - t < 60]
        return len(recent_requests) < self.requests_per_minute


class RateLimitedAgent:
    """
    Agent with built-in rate limiting.
    
    Automatically manages request rate to prevent hitting API limits.
    """
    
    def __init__(
        self, 
        tools: Optional[list[dict]] = None,
        requests_per_minute: int = 50
    ):
        self.tools = tools or []
        self.rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self.conversation_history: list[dict] = []
    
    def query(self, user_message: str, system_prompt: str = "") -> str:
        """
        Send a query to the agent with rate limiting.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            
        Returns:
            The agent's response
        """
        # Wait if necessary
        self.rate_limiter.wait_if_needed()
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Make API call
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt if system_prompt else [],
                messages=self.conversation_history,
                tools=self.tools if self.tools else []
            )
            
            # Add response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Extract text
            text_blocks = [block.text for block in response.content if hasattr(block, "text")]
            return "\n".join(text_blocks)
            
        except anthropic.RateLimitError as e:
            print(f"❌ Rate limit error: {e}")
            print("   Even with rate limiting, we hit the limit.")
            print("   This might mean:")
            print("   - Other processes are using the same API key")
            print("   - The configured limit is higher than your actual limit")
            return "Rate limit exceeded. Please try again later."
    
    def get_rate_info(self) -> dict:
        """Get information about current rate usage."""
        return {
            "requests_per_minute_limit": self.rate_limiter.requests_per_minute,
            "current_rate": self.rate_limiter.get_current_rate(),
            "can_make_request": self.rate_limiter.can_make_request(),
        }


class RetryingAgent:
    """
    Agent that automatically retries on rate limit errors.
    
    Uses exponential backoff to handle rate limits gracefully.
    """
    
    def __init__(self, tools: Optional[list[dict]] = None):
        self.tools = tools or []
        self.conversation_history: list[dict] = []
    
    def query(
        self, 
        user_message: str, 
        system_prompt: str = "",
        max_retries: int = 5
    ) -> Optional[str]:
        """
        Send a query with automatic retry on rate limits.
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt
            max_retries: Maximum number of retry attempts
            
        Returns:
            The agent's response, or None if all retries failed
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Try with exponential backoff
        base_wait = 2
        
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=system_prompt if system_prompt else [],
                    messages=self.conversation_history,
                    tools=self.tools if self.tools else []
                )
                
                # Success! Add to history and return
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                text_blocks = [block.text for block in response.content if hasattr(block, "text")]
                return "\n".join(text_blocks)
                
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    print(f"❌ Rate limit error after {max_retries} attempts")
                    return None
                
                # Calculate wait time with exponential backoff
                wait_time = base_wait * (2 ** attempt)
                print(f"⏳ Rate limited (attempt {attempt + 1}/{max_retries})")
                print(f"   Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
            except anthropic.APIConnectionError:
                print("❌ Failed to connect to API")
                return None
            
            except anthropic.APIStatusError as e:
                print(f"❌ API error: {e.status_code}")
                return None
        
        return None


def demonstrate_rate_limiting():
    """Demonstrate rate limiting in action."""
    print("=== Demonstrating Rate Limiting ===\n")
    
    # Create agent with low limit for demonstration
    agent = RateLimitedAgent(requests_per_minute=3)
    
    print("Making 5 requests with a limit of 3 per minute...\n")
    
    for i in range(5):
        start = time.time()
        
        # Show rate info before request
        rate_info = agent.get_rate_info()
        print(f"Request {i + 1}:")
        print(f"  Current rate: {rate_info['current_rate']}/min")
        print(f"  Can make request: {rate_info['can_make_request']}")
        
        # Make request
        response = agent.query(f"Hello #{i + 1}")
        
        elapsed = time.time() - start
        print(f"  Completed in {elapsed:.1f}s")
        print()


def demonstrate_retrying():
    """Demonstrate retry logic."""
    print("=== Demonstrating Retry Logic ===\n")
    
    agent = RetryingAgent()
    
    # Make a request that might hit rate limits
    print("Making request with automatic retry...")
    response = agent.query(
        "What is the capital of France?",
        max_retries=3
    )
    
    if response:
        print(f"✅ Success: {response}")
    else:
        print("❌ Failed after all retries")


# Example usage
if __name__ == "__main__":
    print("Rate Limiting Examples\n")
    print("=" * 50)
    print()
    
    # Example 1: Basic rate limiting
    print("Example 1: Rate-limited agent")
    print("-" * 50)
    
    agent = RateLimitedAgent(requests_per_minute=50)
    
    # Make a few requests
    for i in range(3):
        response = agent.query(f"Question {i + 1}: What is {i + 1} + {i + 1}?")
        print(f"Response {i + 1}: {response}")
        print()
    
    # Show rate info
    rate_info = agent.get_rate_info()
    print(f"Rate info: {rate_info}")
    print()
    
    # Example 2: Retrying agent
    print("\nExample 2: Retrying agent")
    print("-" * 50)
    
    retrying_agent = RetryingAgent()
    response = retrying_agent.query(
        "What is the capital of Japan?",
        max_retries=5
    )
    print(f"Response: {response}")
    
    # Uncomment to see rate limiting in action
    # demonstrate_rate_limiting()
    # demonstrate_retrying()
