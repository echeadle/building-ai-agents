---
chapter: 4
title: "Your First API Call to Claude"
part: 1
date: 2025-01-15
draft: false
---

# Chapter 4: Your First API Call to Claude

## Introduction

You've set up your environment, installed your tools, and secured your API key. Now it's time for the moment you've been building toward: having a real conversation with an AI.

Making your first API call to Claude might feel like a big step, but here's a secret—it's remarkably simple. At its core, you're just sending a message and getting a response back. That's it. All the sophisticated agent behaviors we'll build later? They're just clever arrangements of this basic interaction.

In this chapter, you'll write a working Python script that talks to Claude. We'll dissect every part of the request so you understand exactly what's happening. By the end, you'll have demystified the "magic" of LLM APIs and established the foundation for everything else in this book.

## Learning Objectives

By the end of this chapter, you will be able to:

-   Install and import the Anthropic Python SDK
-   Construct a properly formatted API request with all required parameters
-   Send a message to Claude and receive a response
-   Parse and extract text from the API response structure
-   Handle common errors like authentication failures and rate limits
-   Understand token-based pricing and estimate API costs

## Installing the Anthropic SDK

Before we can talk to Claude, we need the official Anthropic Python library. This SDK handles all the low-level details of HTTP requests, authentication, and response parsing for us.

If you did not add the anthropic library in Chapter 1, open your terminal now and navigate to your project directory (the one we created in Chapter 2), and install the SDK:

```bash
uv add anthropic
```

You should see output indicating the package was installed and added to your `pyproject.toml`. The Anthropic SDK brings in a few dependencies automatically, including `httpx` for making HTTP requests. You can verify that `httpx` was installed by checking the uv.lock file.

Let's verify the installation worked:

```bash
uv run python -c "import anthropic; print(anthropic.__version__)"
```

This should print a version number like `0.70.0` or higher. If you see an error, double-check that you ran `uv add anthropic` in the correct directory.

> **Note:** Throughout this book, we use `uv run python` to execute Python scripts. This ensures your script runs with the correct virtual environment and dependencies. You can also activate the virtual environment first with `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows), then just use `python`.

## The Anatomy of an API Request

Every API call to Claude requires three essential pieces of information:

1. **model** — Which version of Claude to use
2. **max_tokens** — The maximum length of Claude's response
3. **messages** — The conversation history (what you're saying to Claude)

Let's understand each one.

### Choosing a Model

Anthropic offers several Claude models with different capabilities and price points. For this book, we'll primarily use:

```python
model = "claude-sonnet-4-20250514"
```

Claude Sonnet is an excellent balance of intelligence, speed, and cost—perfect for learning and building agents. Other options include:

-   **claude-opus-4-20250514** — Most capable, best for complex reasoning (higher cost)
-   **claude-haiku-3-5-20241022** — Fastest and cheapest, good for simple tasks

For now, stick with Sonnet. You can always experiment with other models later.

### Setting max_tokens

The `max_tokens` parameter sets the maximum length of Claude's response, measured in tokens. A **token** is roughly 3-4 characters of English text, or about ¾ of a word.

```python
max_tokens = 1024  # Allows responses up to ~750 words
```

A few guidelines:

-   **Too low** (like 50) and Claude's responses get cut off mid-sentence
-   **Too high** (like 100,000) and you might pay for tokens you don't need
-   **1024** is a sensible default for most conversational interactions

Claude won't use all the tokens you allocate—it stops when the response is complete. You're only charged for tokens actually generated.

### Structuring Messages

The `messages` parameter is a list of message objects, each with a `role` and `content`:

```python
messages = [
    {"role": "user", "content": "Hello, Claude! What's your favorite color?"}
]
```

The `role` can be:

-   **"user"** — Messages from the human (that's you!)
-   **"assistant"** — Previous responses from Claude (for multi-turn conversations)

We'll explore the **"system"** role in Chapter 6, but for now, `user` is all we need.

## Your First API Call

Let's put it all together. Create a new file called `first_call.py` in your project's code directory:

```python
"""
Your first API call to Claude.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Create the Anthropic client
# The client automatically uses ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Make the API call
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude! Please introduce yourself in one paragraph."}
    ]
)

# Print the response
print(message.content[0].text)
```

Save the file and run it:

```bash
uv run python first_call.py
```

If everything is set up correctly, you'll see Claude introduce itself! The response will be different each time you run it—LLMs are inherently variable in their outputs.

Let's break down what just happened:

1. **load_dotenv()** — Loaded your API key from the `.env` file
2. **anthropic.Anthropic()** — Created a client that handles authentication automatically
3. **client.messages.create()** — Sent our request to Anthropic's servers
4. **message.content[0].text** — Extracted the text from Claude's response

Congratulations! You've just had your first programmatic conversation with an AI.

## Understanding the Response Structure

The `message` object returned by the API contains more than just text. The response_structure.py file explores the response object:

**File:** response_structure.py

```python
"""
Exploring the API response structure.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)

# Explore the response structure
print("=== Full Response Object ===")
print(f"ID: {message.id}")
print(f"Model: {message.model}")
print(f"Role: {message.role}")
print(f"Stop Reason: {message.stop_reason}")
print()

print("=== Token Usage ===")
print(f"Input tokens: {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")
print()

print("=== Content ===")
print(f"Number of content blocks: {len(message.content)}")
print(f"Content type: {message.content[0].type}")
print(f"Text: {message.content[0].text}")
```

Running this gives you insight into what the API returns:

```
=== Full Response Object ===
ID: msg_01XFDUDYJgAACzvnptvVoYEL
Model: claude-sonnet-4-20250514
Role: assistant
Stop Reason: end_turn

=== Token Usage ===
Input tokens: 14
Output tokens: 15

=== Content ===
Number of content blocks: 1
Content type: text
Text: 2 + 2 equals 4.
```

Key fields to understand:

| Field                 | Description                                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `id`                  | Unique identifier for this message (useful for logging)                                                               |
| `model`               | Confirms which model processed your request                                                                           |
| `role`                | Always "assistant" for responses                                                                                      |
| `stop_reason`         | Why Claude stopped: "end_turn" (finished naturally), "max_tokens" (hit limit), or "stop_sequence" (hit a custom stop) |
| `usage.input_tokens`  | Tokens in your prompt (you pay for these)                                                                             |
| `usage.output_tokens` | Tokens in the response (you pay for these too)                                                                        |
| `content`             | List of content blocks (usually just one text block)                                                                  |

The `stop_reason` is particularly important. If you see `"max_tokens"`, it means Claude was cut off and you might want to increase your `max_tokens` value.

## Handling Errors Gracefully

API calls can fail for various reasons: network issues, invalid keys, rate limits, or server problems. Let's write code that handles these gracefully:

**File:** error_handling.py

```python
"""
Handling API errors gracefully.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


def ask_claude(prompt: str) -> str:
    """
    Send a prompt to Claude and return the response text.

    Args:
        prompt: The question or instruction to send to Claude

    Returns:
        Claude's response as a string

    Raises:
        Various anthropic exceptions on API errors
    """
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text

    except anthropic.AuthenticationError:
        # Invalid API key
        print("Error: Invalid API key. Please check your ANTHROPIC_API_KEY.")
        raise

    except anthropic.RateLimitError:
        # Too many requests
        print("Error: Rate limit exceeded. Please wait a moment and try again.")
        raise

    except anthropic.APIConnectionError:
        # Network issues
        print("Error: Could not connect to Anthropic API. Check your internet connection.")
        raise

    except anthropic.BadRequestError as e:
        # Invalid request (e.g., bad parameters)
        print(f"Error: Bad request - {e.message}")
        raise

    except anthropic.APIStatusError as e:
        # Other API errors
        print(f"Error: API returned status {e.status_code}")
        raise


# Test the function
if __name__ == "__main__":
    response = ask_claude("What's the capital of France?")
    print(response)
```

The Anthropic SDK provides specific exception types for different error conditions:

| Exception             | Cause                         | What to Do                                  |
| --------------------- | ----------------------------- | ------------------------------------------- |
| `AuthenticationError` | Invalid or missing API key    | Check your `.env` file and key validity     |
| `RateLimitError`      | Too many requests too quickly | Wait and retry with exponential backoff     |
| `APIConnectionError`  | Network failure               | Check internet connection, retry            |
| `BadRequestError`     | Invalid parameters            | Check your code for typos or invalid values |
| `APIStatusError`      | Server error or other issues  | Usually temporary; retry after a delay      |

> **💡 Tip:** In production code, you'll want to implement retry logic with exponential backoff for transient errors. We'll cover this in detail in Chapter 30: Error Handling and Recovery.

## Cost Awareness: Understanding Tokens and Pricing

API calls cost money, and understanding pricing helps you build cost-effective agents. Anthropic charges based on **tokens**—both the ones you send (input) and the ones you receive (output).

### What Are Tokens?

Tokens are the fundamental units that language models process. They're roughly:

-   **1 token ≈ 4 characters** of English text
-   **1 token ≈ ¾ of a word**
-   **100 tokens ≈ 75 words**

Some examples:

-   "Hello" = 1 token
-   "Hello, world!" = 4 tokens
-   A typical paragraph = 50-100 tokens
-   A full page of text = ~500 tokens

### Estimating Costs

Let's build a simple cost estimator.

**File:** cost_estimation.py

```python
"""
Token counting and cost estimation.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()

# Pricing per million tokens (as of early 2025 - check Anthropic's website for current rates)
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an API call in dollars.

    Args:
        model: The model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in dollars
    """
    if model not in PRICING:
        # Default to Sonnet pricing for unknown models
        model = "claude-sonnet-4-20250514"

    input_cost = (input_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (output_tokens / 1_000_000) * PRICING[model]["output"]

    return input_cost + output_cost


def ask_claude_with_cost(prompt: str) -> tuple[str, dict]:
    """
    Send a prompt to Claude and return the response with cost information.

    Args:
        prompt: The question or instruction to send to Claude

    Returns:
        Tuple of (response_text, cost_info_dict)
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    cost = estimate_cost(
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens
    )

    cost_info = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        "estimated_cost_usd": cost
    }

    return message.content[0].text, cost_info


# Test with cost tracking
if __name__ == "__main__":
    prompt = "Explain photosynthesis in two sentences."
    response, cost_info = ask_claude_with_cost(prompt)

    print("=== Response ===")
    print(response)
    print()
    print("=== Cost Information ===")
    print(f"Input tokens: {cost_info['input_tokens']}")
    print(f"Output tokens: {cost_info['output_tokens']}")
    print(f"Total tokens: {cost_info['total_tokens']}")
    print(f"Estimated cost: ${cost_info['estimated_cost_usd']:.6f}")
```

Running this shows you exactly what each call costs:

```
=== Response ===
Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This occurs primarily in the chloroplasts of plant cells, where chlorophyll captures light energy to drive the chemical reactions.

=== Cost Information ===
Input tokens: 16
Output tokens: 47
Total tokens: 63
Estimated cost: $0.000753
```

At under a penny for most simple calls, Claude is quite affordable for learning. But costs add up quickly in production systems making thousands of calls per day—we'll cover optimization strategies in Chapter 38.

> **⚠️ Warning:** Pricing changes over time. Always check [Anthropic's pricing page](https://www.anthropic.com/pricing) for current rates before building production systems.

## Putting It All Together

Let's create a polished, reusable function that incorporates everything we've learned:

```python
"""
A complete, reusable function for making API calls to Claude.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ClaudeResponse:
    """Container for Claude's response and metadata."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def ask_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024
) -> ClaudeResponse:
    """
    Send a prompt to Claude and return a structured response.

    Args:
        prompt: The question or instruction to send to Claude
        model: The Claude model to use (default: claude-sonnet-4-20250514)
        max_tokens: Maximum tokens in the response (default: 1024)

    Returns:
        ClaudeResponse object containing the response and metadata

    Raises:
        anthropic.AuthenticationError: If the API key is invalid
        anthropic.RateLimitError: If rate limits are exceeded
        anthropic.APIConnectionError: If connection fails
        anthropic.APIStatusError: For other API errors
    """
    client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return ClaudeResponse(
        text=message.content[0].text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        stop_reason=message.stop_reason
    )


if __name__ == "__main__":
    # Example usage
    response = ask_claude("What are three interesting facts about octopuses?")

    print("Claude says:")
    print(response.text)
    print()
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Stop reason: {response.stop_reason}")
```

This function:

-   ✅ Loads API keys securely from the environment
-   ✅ Has sensible defaults that can be overridden
-   ✅ Returns structured data, not just raw text
-   ✅ Includes type hints for IDE support
-   ✅ Has a clear docstring explaining usage

## Common Pitfalls

### 1. Forgetting to Load Environment Variables

**Symptom:** `AuthenticationError` even though your API key is in `.env`

**Cause:** You forgot to call `load_dotenv()` before creating the client.

**Fix:** Always call `load_dotenv()` at the start of your script, before accessing any environment variables.

### 2. Setting max_tokens Too Low

**Symptom:** Claude's responses end abruptly mid-sentence, and `stop_reason` is `"max_tokens"`

**Cause:** Your `max_tokens` value is too small for the expected response.

**Fix:** Increase `max_tokens`. Start with 1024 and adjust based on your use case. For long-form content, you might need 4096 or more.

### 3. Not Handling Rate Limits

**Symptom:** `RateLimitError` crashes your application during heavy use

**Cause:** You're making too many requests too quickly without any retry logic.

**Fix:** Implement exponential backoff. For now, just add a `time.sleep(1)` between calls if you're making many requests in a loop. We'll cover proper retry logic in Chapter 30.

## Practical Exercise

**Task:** Build a "Magic 8-Ball" that uses Claude to answer yes/no questions in the style of the classic toy.

**Requirements:**

1. Accept a yes/no question from the user via `input()`
2. Send the question to Claude with a prompt that instructs it to respond like a Magic 8-Ball
3. Display the mystical response
4. Show the token usage for the call
5. Allow the user to ask multiple questions (loop until they type "quit")

**Hints:**

-   You'll need to craft a good prompt that tells Claude to act like a Magic 8-Ball
-   Classic Magic 8-Ball responses include things like "It is certain," "Reply hazy, try again," "Don't count on it," etc.
-   Use a `while` loop to keep asking questions

**Solution:** See `code/magic_8_ball.py`

## Key Takeaways

-   **The Anthropic SDK makes API calls simple** — Install with `uv add anthropic`, create a client, call `messages.create()`

-   **Every request needs three things** — A model name, max_tokens limit, and messages list

-   **Responses contain more than text** — Token counts, stop reasons, and IDs are all useful for monitoring and debugging

-   **Handle errors gracefully** — The SDK provides specific exception types for different failure modes

-   **Understand token costs** — Input and output tokens are priced separately; track usage to avoid surprises

-   **At its core, it's just text in, text out** — Don't be intimidated by AI APIs; they're simpler than they seem

## What's Next

You've made your first API call—fantastic! But a single exchange isn't a conversation. In Chapter 5, we'll explore how to build multi-turn conversations by managing message history. You'll learn that the API is stateless (it remembers nothing between calls), and that means _you_ are responsible for maintaining context. This is a crucial concept that underpins everything agents do.
