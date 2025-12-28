"""
Comprehensive error handling for Anthropic API.

Appendix B: API Reference Quick Guide
"""

import os
from dotenv import load_dotenv
import anthropic
import time

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid format
        
    Raises:
        ValueError: If key is invalid
    """
    if not api_key:
        raise ValueError(
            "API key is required. "
            "Set ANTHROPIC_API_KEY in your .env file."
        )
    
    if not api_key.startswith("sk-ant-"):
        raise ValueError(
            "Invalid API key format. "
            "Anthropic API keys should start with 'sk-ant-'"
        )
    
    return True


def validate_request(model: str, max_tokens: int, messages: list) -> bool:
    """
    Validate request parameters before making API call.
    
    Args:
        model: Model name
        max_tokens: Max tokens to generate
        messages: Message history
        
    Returns:
        True if valid
        
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
    
    # Validate message format
    for i, msg in enumerate(messages):
        if "role" not in msg:
            raise ValueError(f"Message {i} missing 'role' field")
        if "content" not in msg:
            raise ValueError(f"Message {i} missing 'content' field")
        if msg["role"] not in ["user", "assistant"]:
            raise ValueError(f"Invalid role in message {i}: {msg['role']}")
    
    return True


def handle_api_errors_basic():
    """Demonstrate basic error handling."""
    print("=== Basic Error Handling Demo ===\n")
    
    client = anthropic.Anthropic()
    
    # Test with invalid model
    print("1. Testing with invalid model name...")
    try:
        response = client.messages.create(
            model="invalid-model-name",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.NotFoundError as e:
        print(f"✓ Caught NotFoundError: Invalid model name")
        print(f"  Error: {e}\n")
    
    # Test with invalid max_tokens
    print("2. Testing with invalid max_tokens...")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=999999999,  # Way too high
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.BadRequestError as e:
        print(f"✓ Caught BadRequestError: max_tokens too high")
        print(f"  Error: {e}\n")
    
    # Test with empty messages
    print("3. Testing with empty messages...")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[]  # Empty
        )
    except anthropic.BadRequestError as e:
        print(f"✓ Caught BadRequestError: Empty messages")
        print(f"  Error: {e}\n")


def handle_api_errors_comprehensive():
    """Demonstrate comprehensive error handling."""
    print("=== Comprehensive Error Handling Demo ===\n")
    
    client = anthropic.Anthropic()
    
    def make_safe_api_call(
        model: str,
        max_tokens: int,
        messages: list
    ) -> str:
        """
        Make an API call with comprehensive error handling.
        
        Returns:
            Response text or error message
        """
        try:
            # Validate inputs
            validate_request(model, max_tokens, messages)
            
            # Make API call
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages
            )
            
            return response.content[0].text
            
        except anthropic.APIConnectionError as e:
            return f"Connection Error: Failed to connect to API. Check your internet connection."
        
        except anthropic.RateLimitError as e:
            return f"Rate Limit Error: You've made too many requests. Wait and try again."
        
        except anthropic.AuthenticationError as e:
            return f"Authentication Error: Invalid API key. Check your ANTHROPIC_API_KEY."
        
        except anthropic.PermissionDeniedError as e:
            return f"Permission Denied: This request violates content policy."
        
        except anthropic.NotFoundError as e:
            return f"Not Found: Invalid model name or endpoint."
        
        except anthropic.BadRequestError as e:
            return f"Bad Request: Invalid parameters. {e}"
        
        except anthropic.InternalServerError as e:
            return f"Server Error: Problem on Anthropic's side. Try again later."
        
        except anthropic.APIError as e:
            # Catch-all for other API errors
            if e.status_code == 529:
                return "API Overloaded: Service is temporarily overloaded. Try again later."
            return f"API Error {e.status_code}: {e}"
        
        except ValueError as e:
            # Validation errors
            return f"Validation Error: {e}"
        
        except Exception as e:
            # Unexpected errors
            return f"Unexpected Error: {type(e).__name__}: {e}"
    
    # Test various scenarios
    print("1. Valid request:")
    result = make_safe_api_call(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print(f"   {result[:50]}...\n")
    
    print("2. Invalid model:")
    result = make_safe_api_call(
        model="invalid-model",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"   {result}\n")
    
    print("3. Invalid max_tokens:")
    result = make_safe_api_call(
        model="claude-sonnet-4-20250514",
        max_tokens=-1,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(f"   {result}\n")


def handle_rate_limits_with_retry():
    """Demonstrate rate limit handling with retry."""
    print("=== Rate Limit Handling with Retry ===\n")
    
    client = anthropic.Anthropic()
    
    def call_with_retry(
        model: str,
        max_tokens: int,
        messages: list,
        max_retries: int = 3
    ) -> str:
        """
        Make API call with automatic retry on rate limits.
        
        Returns:
            Response text or error message
        """
        delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages
                )
                return response.content[0].text
                
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    return f"Rate limit exceeded after {max_retries} attempts"
                
                wait_time = delay * (2 ** attempt)
                print(f"   Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
            except anthropic.InternalServerError as e:
                if attempt == max_retries - 1:
                    return "Server error persisted after retries"
                
                print(f"   Server error. Retrying in 5s...")
                time.sleep(5)
                
            except anthropic.APIError as e:
                # Don't retry other errors
                return f"API Error: {e}"
        
        return "Max retries exceeded"
    
    # Make a call with retry logic
    print("Making API call with automatic retry...")
    result = call_with_retry(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(f"Result: {result}\n")


def handle_validation_errors():
    """Demonstrate input validation."""
    print("=== Input Validation Demo ===\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Valid request",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        },
        {
            "name": "Missing model",
            "model": "",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        },
        {
            "name": "Invalid max_tokens",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 0,
            "messages": [{"role": "user", "content": "Hello"}]
        },
        {
            "name": "Empty messages",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": []
        },
        {
            "name": "Invalid role",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 100,
            "messages": [{"role": "invalid", "content": "Hello"}]
        }
    ]
    
    for test in test_cases:
        print(f"Testing: {test['name']}")
        try:
            validate_request(
                test["model"],
                test["max_tokens"],
                test["messages"]
            )
            print("   ✓ Validation passed\n")
        except ValueError as e:
            print(f"   ✗ Validation failed: {e}\n")


if __name__ == "__main__":
    # Validate API key first
    print("=== API Key Validation ===\n")
    try:
        validate_api_key(api_key)
        print("✓ API key is valid\n")
    except ValueError as e:
        print(f"✗ {e}\n")
    
    print("="*60 + "\n")
    
    # Basic error handling
    handle_api_errors_basic()
    
    print("="*60 + "\n")
    
    # Comprehensive error handling
    handle_api_errors_comprehensive()
    
    print("="*60 + "\n")
    
    # Rate limit handling
    handle_rate_limits_with_retry()
    
    print("="*60 + "\n")
    
    # Input validation
    handle_validation_errors()
