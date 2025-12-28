# Appendix B: API Reference Quick Guide - Code Examples

This directory contains code examples for Appendix B: API Reference Quick Guide.

## Files

### `rate_limit_handler.py`
Demonstrates handling rate limits with exponential backoff.

**Key Features:**
- `RateLimitHandler` class with automatic retry
- Exponential backoff strategy (1s, 2s, 4s, 8s, etc.)
- Manual backoff implementation example

**Run it:**
```bash
python rate_limit_handler.py
```

---

### `token_bucket.py`
Implements a token bucket rate limiter for fine-grained rate control.

**Key Features:**
- `TokenBucket` class for rate limiting
- Supports burst capacity
- Blocking and non-blocking modes
- Thread-safe implementation

**Run it:**
```bash
python token_bucket.py
```

**Use cases:**
- High-volume API usage
- Preventing rate limit errors proactively
- Allowing burst traffic while maintaining average rate

---

### `token_counter.py`
Tracks token usage and estimates costs across multiple API calls.

**Key Features:**
- `TokenCounter` class for tracking usage
- `TokenUsage` dataclass for aggregating tokens
- Cost estimation for different models
- Budget tracking
- Token estimation without API calls

**Run it:**
```bash
python token_counter.py
```

**Use cases:**
- Cost monitoring
- Budget enforcement
- Comparing costs across models
- Usage analytics

---

### `error_handling.py`
Demonstrates comprehensive error handling for the Anthropic API.

**Key Features:**
- Handling all common error types
- Input validation before API calls
- Retry logic for transient errors
- User-friendly error messages

**Run it:**
```bash
python error_handling.py
```

**Error types covered:**
- `APIConnectionError` - Network issues
- `RateLimitError` - Too many requests
- `AuthenticationError` - Invalid API key
- `PermissionDeniedError` - Content policy violations
- `NotFoundError` - Invalid model/endpoint
- `BadRequestError` - Invalid parameters
- `InternalServerError` - Server-side issues

---

### `exercise_solution.py`
Complete solution for the practical exercise.

Combines all the concepts from the appendix:
- Error handling
- Rate limiting
- Token tracking
- Cost estimation

**Run it:**
```bash
python exercise_solution.py
```

---

## Setup

All examples require:

1. **Python 3.10+**
2. **Dependencies:**
   ```bash
   uv add anthropic python-dotenv
   ```

3. **API Key in `.env`:**
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Common Patterns

### Basic API Call with Error Handling
```python
try:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.content[0].text)
except anthropic.RateLimitError:
    print("Rate limited - retry with backoff")
except anthropic.APIError as e:
    print(f"API error: {e}")
```

### Using the Rate Limit Handler
```python
handler = RateLimitHandler(max_retries=3)
response = handler.call_with_retry(
    lambda: client.messages.create(...)
)
```

### Tracking Token Usage
```python
counter = TokenCounter()

response = client.messages.create(...)
counter.add_response(response)

print(counter.summary())
```

### Rate Limiting with Token Bucket
```python
limiter = TokenBucket(rate=10/60, capacity=5)  # 10 req/min

if limiter.acquire():
    response = client.messages.create(...)
```

## Tips

1. **Always validate inputs** before making API calls
2. **Implement retry logic** for rate limits and server errors
3. **Track token usage** to monitor costs
4. **Use token buckets** for high-volume production use
5. **Handle all error types** explicitly

## Further Reading

- Chapter 33: Cost Management and Optimization
- Chapter 35: Testing Strategies for Agents
- Appendix E: Troubleshooting Guide
- Official Anthropic docs: https://docs.anthropic.com
