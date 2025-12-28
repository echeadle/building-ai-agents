---
appendix: B
title: "API Reference Quick Guide"
date: 2025-12-09
draft: false
---

# Appendix B: API Reference Quick Guide

This appendix provides a quick reference for the Anthropic API's most commonly used parameters, error codes, and operational considerations. Think of it as your desk reference while building agents—the information you need most often, organized for quick lookup.

## Anthropic API Parameters Reference

### Core Message Parameters

#### `model` (required)
**Type:** string  
**Description:** The Claude model to use.

**Available Models (as of this writing):**
- `claude-sonnet-4-20250514` — Our default throughout this book. Best balance of intelligence and speed.
- `claude-opus-4-20250514` — Most capable model. Use when quality matters most.
- `claude-haiku-4-20250514` — Fastest and most economical. Use for simple, high-volume tasks.

**Example:**
```python
model="claude-sonnet-4-20250514"
```

> **💡 Tip:** Model names include dates to indicate version. Always use the full string, including the date suffix.

---

#### `max_tokens` (required)
**Type:** integer  
**Description:** Maximum number of tokens to generate in the response.

**Common Values:**
- `1024` — Short responses (a few paragraphs)
- `4096` — Medium responses (a page or two)
- `8192` — Long responses (detailed explanations)
- `200000` — Maximum allowed (as of this writing)

**Example:**
```python
max_tokens=4096
```

> **⚠️ Warning:** Setting `max_tokens` too low will truncate responses mid-sentence. Set it higher than you think you need, then optimize if cost becomes an issue.

---

#### `messages` (required)
**Type:** array of message objects  
**Description:** The conversation history and current prompt.

**Message Object Structure:**
```python
{
    "role": "user" | "assistant",
    "content": "string or array of content blocks"
}
```

**Simple Example:**
```python
messages=[
    {"role": "user", "content": "What is the capital of France?"}
]
```

**Multi-turn Example:**
```python
messages=[
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What's its population?"}
]
```

**Content Blocks Example (with tool results):**
```python
messages=[
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather for you."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco, CA"}
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_123",
                "content": "72°F, partly cloudy"
            }
        ]
    }
]
```

> **💡 Tip:** The `messages` array represents the entire conversation. Each API call is stateless—you must include all relevant history.

---

#### `system` (optional)
**Type:** string or array of text blocks  
**Description:** System instructions that guide Claude's behavior.

**Simple Example:**
```python
system="You are a helpful assistant that speaks like a pirate."
```

**Multi-block Example:**
```python
system=[
    {
        "type": "text",
        "text": "You are a customer service agent."
    },
    {
        "type": "text",
        "text": "Company policy: Always be polite and never promise refunds without manager approval.",
        "cache_control": {"type": "ephemeral"}
    }
]
```

> **💡 Tip:** System prompts are powerful. Use them to set tone, provide context, define personality, and give instructions that apply to the entire conversation.

---

#### `temperature` (optional)
**Type:** number (0.0 to 1.0)  
**Default:** 1.0  
**Description:** Controls randomness in responses.

**Guidelines:**
- `0.0` — Deterministic (same input → same output)
- `0.3-0.5` — Low creativity, high consistency (good for agents)
- `0.7-0.9` — Balanced (default)
- `1.0` — Maximum creativity

**Example:**
```python
temperature=0.3  # More predictable for agent tasks
```

> **💡 Tip:** Lower temperature for agents that need consistent tool use. Higher temperature for creative tasks like writing or brainstorming.

---

#### `top_p` (optional)
**Type:** number (0.0 to 1.0)  
**Default:** Not set  
**Description:** Alternative to temperature. Uses nucleus sampling.

**Example:**
```python
top_p=0.9
```

> **⚠️ Warning:** Don't use both `temperature` and `top_p` together. Pick one based on your use case.

---

#### `top_k` (optional)
**Type:** integer  
**Default:** Not set  
**Description:** Limits sampling to top K tokens.

**Example:**
```python
top_k=50
```

> **💡 Tip:** Rarely needed. Temperature or top_p usually suffice.

---

#### `tools` (optional)
**Type:** array of tool definitions  
**Description:** Functions that Claude can call.

**Structure:**
```python
tools=[
    {
        "name": "tool_name",
        "description": "Clear description of what the tool does and when to use it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "What this parameter is for"
                }
            },
            "required": ["param_name"]
        }
    }
]
```

**Full Example:**
```python
tools=[
    {
        "name": "get_weather",
        "description": "Get current weather for a location. Use this when users ask about weather conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA'"
                },
                "units": {
                    "type": "string",
                    "enum": ["fahrenheit", "celsius"],
                    "description": "Temperature units to use"
                }
            },
            "required": ["location"]
        }
    }
]
```

> **📚 See Also:** Appendix C: Tool Design Patterns for comprehensive tool design guidance.

---

#### `tool_choice` (optional)
**Type:** object or "auto" or "any"  
**Default:** "auto"  
**Description:** Controls how Claude uses tools.

**Options:**

1. **"auto"** (default) — Claude decides when to use tools
```python
tool_choice="auto"
```

2. **"any"** — Claude must use at least one tool
```python
tool_choice="any"
```

3. **Specific tool** — Force Claude to use a specific tool
```python
tool_choice={
    "type": "tool",
    "name": "get_weather"
}
```

> **💡 Tip:** Use `"any"` when you want to ensure Claude uses tools instead of answering directly. Use specific tool choice for structured output patterns.

---

#### `metadata` (optional)
**Type:** object  
**Description:** Pass-through metadata for tracking and filtering.

**Example:**
```python
metadata={
    "user_id": "user_123",
    "session_id": "session_456"
}
```

---

#### `stop_sequences` (optional)
**Type:** array of strings  
**Description:** Sequences that will stop generation.

**Example:**
```python
stop_sequences=["Human:", "\n\n---"]
```

---

#### `stream` (optional)
**Type:** boolean  
**Default:** False  
**Description:** Stream the response incrementally.

**Example:**
```python
stream=True

with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a haiku"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

> **💡 Tip:** Streaming improves perceived latency for long responses. Users see output immediately rather than waiting for the entire response.

---

## Common Error Codes and Solutions

### HTTP Status Codes

#### 400 Bad Request
**Meaning:** Invalid request format or parameters.

**Common Causes:**
- Missing required parameters (`model`, `max_tokens`, `messages`)
- Invalid parameter values (e.g., `max_tokens` too high)
- Malformed JSON
- Invalid tool schema

**Example Error:**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "max_tokens: Field required"
  }
}
```

**Solution:**
```python
# Validate parameters before sending
def validate_request(model, max_tokens, messages):
    """Validate request parameters before making API call."""
    if not model:
        raise ValueError("model is required")
    if not max_tokens or max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if not messages or len(messages) == 0:
        raise ValueError("messages cannot be empty")
    
    return True

validate_request(model, max_tokens, messages)
response = client.messages.create(...)
```

---

#### 401 Unauthorized
**Meaning:** Invalid or missing API key.

**Common Causes:**
- API key not set in environment
- Typo in API key
- API key revoked or expired

**Solution:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY not found. "
        "Add it to your .env file or set it as an environment variable."
    )

# Verify key format (should start with 'sk-ant-')
if not api_key.startswith("sk-ant-"):
    raise ValueError(
        "Invalid API key format. "
        "Anthropic API keys should start with 'sk-ant-'"
    )
```

---

#### 403 Forbidden
**Meaning:** Request rejected due to safety filters or policy violation.

**Common Causes:**
- Content triggered safety filters
- Request violates usage policy
- Account suspended

**Solution:**
```python
try:
    response = client.messages.create(...)
except anthropic.PermissionDeniedError as e:
    # Log the error for review
    print(f"Request blocked: {e}")
    # Return a safe fallback response
    return "I cannot process this request due to content policy restrictions."
```

---

#### 404 Not Found
**Meaning:** Invalid endpoint or resource not found.

**Common Causes:**
- Wrong API endpoint URL
- Invalid model name
- Using deprecated API version

**Solution:**
```python
# Use the correct endpoint (handled by SDK)
client = anthropic.Anthropic()  # SDK handles endpoint

# Verify model name is correct
VALID_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-4-20250514"
]

if model not in VALID_MODELS:
    raise ValueError(f"Invalid model: {model}. Valid models: {VALID_MODELS}")
```

---

#### 429 Too Many Requests (Rate Limit)
**Meaning:** You've exceeded rate limits.

**Rate Limit Types:**
- **Requests per minute (RPM):** Number of API calls
- **Tokens per minute (TPM):** Total tokens processed
- **Tokens per day (TPD):** Daily token budget

**Example Error:**
```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded"
  }
}
```

**Solution:**
```python
import time
from anthropic import RateLimitError

def call_with_retry(client, max_retries=3, initial_delay=1.0):
    """Make API call with exponential backoff retry."""
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}]
            )
            return response
            
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, re-raise
            
            print(f"Rate limited. Waiting {delay}s before retry...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    raise Exception("Max retries exceeded")
```

**Better Solution Using Headers:**
```python
def call_with_rate_limit_awareness(client):
    """Make API call and check rate limit headers."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # Check rate limit headers (available in response)
        # Note: Exact header names may vary, check API docs
        remaining = response.headers.get("anthropic-ratelimit-requests-remaining")
        reset = response.headers.get("anthropic-ratelimit-requests-reset")
        
        if remaining and int(remaining) < 10:
            print(f"Warning: Only {remaining} requests remaining until {reset}")
        
        return response
        
    except RateLimitError:
        # Handle rate limit
        print("Rate limited. Implement backoff or queue.")
        raise
```

> **📚 See Also:** Chapter 33: Rate Limiting and Throttling for production-grade rate limit handling.

---

#### 500 Internal Server Error
**Meaning:** Server-side error at Anthropic.

**Solution:**
```python
from anthropic import InternalServerError

try:
    response = client.messages.create(...)
except InternalServerError as e:
    print(f"Server error: {e}")
    # Retry after a delay
    time.sleep(5)
    response = client.messages.create(...)  # Retry
```

---

#### 529 Overloaded
**Meaning:** API is temporarily overloaded.

**Solution:**
```python
from anthropic import APIError

def call_with_overload_retry(client, max_retries=5):
    """Retry on overload errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(...)
        except APIError as e:
            if e.status_code == 529 and attempt < max_retries - 1:
                delay = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"API overloaded. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
```

---

## Rate Limit Handling

### Understanding Rate Limits

Rate limits prevent abuse and ensure fair access. They're enforced on three dimensions:

1. **Requests Per Minute (RPM):** How many API calls you can make
2. **Tokens Per Minute (TPM):** How many tokens you can process
3. **Tokens Per Day (TPD):** Daily token budget

**Tier Examples (check current docs for your tier):**
- **Free Tier:** 5 RPM, 20,000 TPM, 300,000 TPD
- **Build Tier 1:** 50 RPM, 100,000 TPM, 5M TPD
- **Build Tier 2:** 100 RPM, 200,000 TPM, 10M TPD
- **Scale Tier:** 1,000+ RPM, 2M+ TPM, custom TPD

### Basic Rate Limit Handler

```python
"""
Basic rate limit handling with exponential backoff.

Appendix B: API Reference Quick Guide
"""

import time
import anthropic
from typing import Optional, Callable, Any


class RateLimitHandler:
    """
    Handle rate limits with exponential backoff.
    
    Usage:
        handler = RateLimitHandler()
        response = handler.call_with_retry(
            lambda: client.messages.create(...)
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
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
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
        
        Args:
            api_call: Function that makes the API call
            
        Returns:
            API response
            
        Raises:
            RateLimitError: If max retries exceeded
        """
        delay = self.initial_delay
        
        for attempt in range(self.max_retries):
            try:
                return api_call()
                
            except anthropic.RateLimitError as e:
                if attempt == self.max_retries - 1:
                    # Last attempt, give up
                    raise
                
                # Calculate delay with exponential backoff
                wait_time = min(delay * (2 ** attempt), self.max_delay)
                
                print(
                    f"Rate limited (attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {wait_time:.1f}s..."
                )
                
                time.sleep(wait_time)
        
        raise anthropic.RateLimitError("Max retries exceeded")


# Usage example
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    client = anthropic.Anthropic()
    handler = RateLimitHandler()
    
    # Make a call with automatic retry
    response = handler.call_with_retry(
        lambda: client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello!"}]
        )
    )
    
    print(response.content[0].text)
```

### Token Bucket Rate Limiter

For more sophisticated rate limiting, implement a token bucket:

```python
"""
Token bucket rate limiter for API calls.

Appendix B: API Reference Quick Guide
"""

import time
from threading import Lock
from typing import Optional


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows burst traffic while enforcing average rate limits.
    
    Usage:
        limiter = TokenBucket(rate=10, capacity=20)  # 10 requests/sec, burst of 20
        
        if limiter.acquire():
            response = client.messages.create(...)
        else:
            print("Rate limit reached, try again later")
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
            block: If True, wait until tokens are available
            
        Returns:
            True if tokens acquired, False otherwise
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
                
                # If not blocking, return False
                if not block:
                    return False
                
                # Calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
            
            # Wait outside the lock
            time.sleep(wait_time)


# Usage example
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import anthropic
    
    load_dotenv()
    
    client = anthropic.Anthropic()
    
    # 10 requests per minute = 10/60 per second
    limiter = TokenBucket(rate=10/60, capacity=5)
    
    # Make multiple requests
    for i in range(15):
        print(f"\nRequest {i + 1}...")
        
        # Acquire token (blocks if necessary)
        limiter.acquire()
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": f"Count to {i + 1}"}]
        )
        
        print(f"Response: {response.content[0].text[:50]}...")
```

---

## Token Counting

Understanding token usage is essential for cost management and staying within limits.

### What Are Tokens?

**Tokens** are the basic units that models process. Roughly:
- 1 token ≈ 4 characters of English text
- 1 token ≈ ¾ of an English word
- 100 tokens ≈ 75 words

**Examples:**
- "Hello, world!" ≈ 4 tokens
- "The quick brown fox jumps over the lazy dog" ≈ 10 tokens
- A typical paragraph ≈ 100 tokens
- A page of text ≈ 500 tokens

### Token Counting in Responses

The API returns token usage in every response:

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Explain Python in 100 words."}]
)

# Access token usage
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total tokens: {response.usage.input_tokens + response.usage.output_tokens}")

# Calculate cost (example rates, check current pricing)
INPUT_COST_PER_MILLION = 3.00  # $3 per million input tokens
OUTPUT_COST_PER_MILLION = 15.00  # $15 per million output tokens

input_cost = (response.usage.input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
output_cost = (response.usage.output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
total_cost = input_cost + output_cost

print(f"\nEstimated cost: ${total_cost:.6f}")
```

### Token Counter Utility

```python
"""
Token usage tracking and estimation.

Appendix B: API Reference Quick Guide
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
    
    def estimate_cost(
        self,
        input_cost_per_million: float = 3.00,
        output_cost_per_million: float = 15.00,
        cached_cost_per_million: float = 0.30
    ) -> float:
        """
        Estimate cost based on current token usage.
        
        Args:
            input_cost_per_million: Cost per million input tokens
            output_cost_per_million: Cost per million output tokens
            cached_cost_per_million: Cost per million cached tokens
            
        Returns:
            Estimated cost in dollars
        """
        input_cost = (self.input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (self.output_tokens / 1_000_000) * output_cost_per_million
        cached_cost = (self.cached_tokens / 1_000_000) * cached_cost_per_million
        
        return input_cost + output_cost + cached_cost
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens
        )
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"TokenUsage(input={self.input_tokens}, "
            f"output={self.output_tokens}, "
            f"total={self.total_tokens})"
        )


class TokenCounter:
    """
    Track token usage across multiple API calls.
    
    Usage:
        counter = TokenCounter()
        
        response = client.messages.create(...)
        counter.add_response(response)
        
        print(f"Total usage: {counter.total}")
        print(f"Total cost: ${counter.total_cost():.4f}")
    """
    
    def __init__(
        self,
        input_cost_per_million: float = 3.00,
        output_cost_per_million: float = 15.00
    ):
        """
        Initialize the token counter.
        
        Args:
            input_cost_per_million: Cost per million input tokens
            output_cost_per_million: Cost per million output tokens
        """
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.total = TokenUsage()
        self.call_count = 0
    
    def add_response(self, response) -> TokenUsage:
        """
        Add token usage from an API response.
        
        Args:
            response: API response object with usage attribute
            
        Returns:
            TokenUsage for this response
        """
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        
        self.total = self.total + usage
        self.call_count += 1
        
        return usage
    
    def total_cost(self) -> float:
        """Calculate total cost so far."""
        return self.total.estimate_cost(
            self.input_cost_per_million,
            self.output_cost_per_million
        )
    
    def average_tokens_per_call(self) -> float:
        """Calculate average tokens per API call."""
        if self.call_count == 0:
            return 0.0
        return self.total.total_tokens / self.call_count
    
    def summary(self) -> str:
        """Get a summary of token usage."""
        avg = self.average_tokens_per_call()
        cost = self.total_cost()
        
        return (
            f"Token Usage Summary\n"
            f"===================\n"
            f"API Calls: {self.call_count}\n"
            f"Input Tokens: {self.total.input_tokens:,}\n"
            f"Output Tokens: {self.total.output_tokens:,}\n"
            f"Total Tokens: {self.total.total_tokens:,}\n"
            f"Average per Call: {avg:.1f} tokens\n"
            f"Estimated Cost: ${cost:.4f}"
        )


# Usage example
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import anthropic
    
    load_dotenv()
    
    client = anthropic.Anthropic()
    counter = TokenCounter()
    
    # Make several API calls
    queries = [
        "What is Python?",
        "Explain object-oriented programming",
        "Write a haiku about coding"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": query}]
        )
        
        usage = counter.add_response(response)
        print(f"Usage: {usage}")
    
    # Print summary
    print("\n" + counter.summary())
```

### Estimating Tokens Before API Calls

For rough estimation without calling the API:

```python
def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count.
    
    This is approximate. For exact counts, use the API.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Rough heuristic: 1 token ≈ 4 characters
    return len(text) // 4


def will_fit_in_context(
    messages: list[dict],
    tools: list[dict],
    max_context: int = 200000
) -> tuple[bool, int]:
    """
    Check if a request will fit in the model's context window.
    
    Args:
        messages: Message history
        tools: Tool definitions
        max_context: Maximum context window size
        
    Returns:
        (will_fit, estimated_tokens)
    """
    # Estimate message tokens
    message_tokens = 0
    for msg in messages:
        if isinstance(msg["content"], str):
            message_tokens += estimate_tokens(msg["content"])
        elif isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.get("type") == "text":
                    message_tokens += estimate_tokens(block["text"])
    
    # Estimate tool definition tokens
    tool_tokens = 0
    for tool in tools:
        # Tools descriptions are roughly 50-200 tokens each
        tool_tokens += 100  # Conservative estimate
    
    total = message_tokens + tool_tokens
    
    return total < max_context, total


# Example usage
messages = [{"role": "user", "content": "Write a long essay about AI" * 100}]
tools = [{"name": "tool1", "description": "..." * 50}]

will_fit, estimated = will_fit_in_context(messages, tools)
print(f"Estimated tokens: {estimated}")
print(f"Will fit: {will_fit}")
```

---

## Quick Reference Tables

### Model Comparison

| Model | Best For | Speed | Cost | Context Window |
|-------|----------|-------|------|----------------|
| Claude Sonnet 4 | Balanced tasks | Fast | Medium | 200K tokens |
| Claude Opus 4 | Complex reasoning | Slower | Higher | 200K tokens |
| Claude Haiku 4 | Simple, high-volume | Fastest | Lowest | 200K tokens |

### Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Fix request parameters |
| 401 | Unauthorized | Check API key |
| 403 | Forbidden | Check content/policy |
| 429 | Rate limited | Implement backoff |
| 500 | Server error | Retry with delay |
| 529 | Overloaded | Retry with backoff |

### Rate Limit Strategies

| Strategy | Use Case | Complexity |
|----------|----------|------------|
| Simple retry | Low volume | Low |
| Exponential backoff | Medium volume | Medium |
| Token bucket | High volume | High |
| Queue + worker pool | Production | High |

### Token Cost Estimation (Example Rates)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Haiku 4 | $0.80 | $4.00 |

> **⚠️ Important:** These are example rates. Always check [anthropic.com/pricing](https://anthropic.com/pricing) for current pricing.

---

## Quick Debugging Checklist

When something goes wrong, check these in order:

1. **✓ API Key**
   - Is `ANTHROPIC_API_KEY` set?
   - Does it start with `sk-ant-`?
   - Is it loaded from `.env`?

2. **✓ Request Format**
   - Are `model`, `max_tokens`, and `messages` provided?
   - Is the JSON valid?
   - Are tool schemas valid?

3. **✓ Rate Limits**
   - Are you within RPM/TPM/TPD limits?
   - Do you have retry logic with backoff?

4. **✓ Token Limits**
   - Is `max_tokens` reasonable?
   - Does your input fit in the context window?

5. **✓ Error Handling**
   - Are you catching specific exceptions?
   - Do you log errors for debugging?
   - Do you retry appropriately?

---

## Best Practices Summary

### Security
- ✓ Always use `.env` files for API keys
- ✓ Never commit secrets to version control
- ✓ Validate API keys before making calls

### Reliability
- ✓ Implement exponential backoff for rate limits
- ✓ Handle all error types explicitly
- ✓ Set appropriate timeouts

### Performance
- ✓ Use streaming for long responses
- ✓ Implement caching where appropriate
- ✓ Choose the right model for the task

### Cost Management
- ✓ Track token usage
- ✓ Set `max_tokens` appropriately
- ✓ Use Haiku for simple tasks

### Monitoring
- ✓ Log all API calls
- ✓ Track error rates
- ✓ Monitor token usage trends
- ✓ Set up alerts for anomalies

---

## What's Next

This appendix provided a quick reference for the Anthropic API. For more detailed information:

- **Appendix C: Tool Design Patterns** — Learn to design effective tools
- **Appendix D: Prompt Engineering** — Master system prompts and few-shot examples
- **Appendix E: Troubleshooting Guide** — Solve common agent problems
- **Chapter 33: Cost Management** — Production-grade cost tracking
- **Chapter 35: Testing Strategies** — Ensure agent reliability

For the most up-to-date API documentation, visit [docs.anthropic.com](https://docs.anthropic.com).
