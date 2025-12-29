---
chapter: 38
title: "Cost Optimization"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 38: Cost Optimization

## Introduction

Your agent works beautifully. It passes all tests, handles errors gracefully, and users love it. Then you check your API bill at the end of the month: $2,847.53. For a side project.

This isn't hypothetical. Unmonitored agents are notorious for generating surprising bills. Every LLM call costs money, and agents make *many* LLM calls. A single user request might trigger 5, 10, or even 20 API calls as the agent reasons through a complex task. Multiply that by thousands of users, and costs escalate quickly.

In Chapter 37, we learned to debug agent behavior. Now we'll debug agent *economics*. This chapter teaches you to understand, track, and optimize the costs of running AI agents in production. You'll learn where the money goes, how to spend less of it without sacrificing quality, and how to set up monitoring so you're never surprised by a bill again.

The goal isn't to minimize costs at all costs—it's to maximize *value per dollar*. Sometimes spending more on a better model is worth it. Sometimes a cached response is just as good as a fresh one. The key is making these tradeoffs deliberately, with data.

## Learning Objectives

By the end of this chapter, you will be able to:

- Calculate the true cost of agent operations using token-based pricing
- Implement prompt optimization techniques that reduce input tokens without losing effectiveness
- Control response length to manage output token costs
- Build caching systems that eliminate redundant API calls
- Select appropriate models based on task complexity and cost requirements
- Set up cost monitoring with alerts for budget protection

## Understanding Token Costs

Before we can optimize costs, we need to understand how they're calculated. Claude's API charges based on **tokens**—the fundamental units that LLMs use to process text.

### What Are Tokens?

Tokens are pieces of words. A rough rule of thumb:
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words

But tokenization isn't perfectly predictable. Common words might be single tokens, while unusual words get split into multiple tokens. Code and non-English text often use more tokens per character.

### Token Estimation

Here's a simple token estimator:

```python
"""
Token estimation utilities.

Chapter 38: Cost Optimization
"""

import re

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    This is an approximation - actual tokenization varies by model.
    For precise counts, use the Anthropic tokenizer API.
    """
    if not text:
        return 0

    # Count words and adjust for punctuation and numbers
    words = text.split()
    word_count = len(words)
    punctuation = len(re.findall(r'[^\w\s]', text))
    numbers = len(re.findall(r'\d+', text))

    # Base estimate: words + half of punctuation + numbers
    estimate = word_count + (punctuation // 2) + numbers

    # Adjust for long words (technical terms, code)
    if word_count > 0:
        avg_word_len = sum(len(w) for w in words) / word_count
        if avg_word_len > 6:  # Long words → more tokens
            estimate = int(estimate * 1.2)

    return max(1, estimate)


# Example usage
system_prompt = """You are a helpful AI assistant that analyzes code.
When reviewing code, identify bugs, suggest improvements, and explain
your reasoning clearly and concisely."""

print(f"System prompt tokens: {estimate_tokens(system_prompt)}")
# Output: ~35 tokens
```

### Claude's Pricing Model

Claude charges separately for **input tokens** (what you send) and **output tokens** (what Claude generates). As of early 2025, here are the approximate prices:

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku 3.5 | $0.80 | $4.00 |

> **Note:** Prices change over time. Always check Anthropic's current pricing at https://www.anthropic.com/pricing

**The key insight**: Output tokens cost 5x more than input tokens. This means controlling response length has a bigger impact on costs than reducing prompt length.

### Calculating Costs

Here's how to calculate the cost of an API call:

```python
"""
Cost calculation utilities.

Chapter 38: Cost Optimization
"""

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing information for a Claude model."""
    input_cost: float   # Cost per 1M input tokens
    output_cost: float  # Cost per 1M output tokens


# Current model pricing (as of early 2025)
MODEL_PRICING = {
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00),
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
}


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-20250514"
) -> float:
    """
    Calculate the cost of an API call.

    Args:
        input_tokens: Number of tokens sent to the model
        output_tokens: Number of tokens generated by the model
        model: Model identifier

    Returns:
        Cost in USD
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-20250514"])

    input_cost = (input_tokens / 1_000_000) * pricing.input_cost
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost

    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


# Example: Cost of a typical agent interaction
system_prompt_tokens = 500
user_message_tokens = 100
tool_definitions_tokens = 200
assistant_response_tokens = 150

input_tokens = system_prompt_tokens + user_message_tokens + tool_definitions_tokens
output_tokens = assistant_response_tokens

cost = calculate_cost(input_tokens, output_tokens, "claude-sonnet-4-20250514")
print(f"Single API call cost: {format_cost(cost)}")
# Output: $0.0039

# For an agent that makes 8 API calls per request:
total_cost = cost * 8
print(f"Full agent request cost: {format_cost(total_cost)}")
# Output: $0.0312

# Monthly cost for 10,000 daily users:
monthly_cost = total_cost * 10_000 * 30
print(f"Monthly cost (10k daily users): {format_cost(monthly_cost)}")
# Output: $9,360.00
```

### Where Agent Costs Come From

Agents are expensive because they make multiple API calls per request. Let's trace the costs:

```
User Request: "Research the top 3 electric vehicles and compare them"

Call 1: Planning (analyze request)
  - Input: 500 tokens (system prompt + user request)
  - Output: 200 tokens (plan)
  - Cost: $0.0045

Call 2: Search for EV #1
  - Input: 800 tokens (context + tool definitions)
  - Output: 100 tokens (tool call)
  - Cost: $0.0039

Call 3: Process search results
  - Input: 2000 tokens (context + search results)
  - Output: 300 tokens (analysis)
  - Cost: $0.0105

... (repeat for EVs #2 and #3)

Call 8: Final synthesis
  - Input: 4000 tokens (all gathered information)
  - Output: 1000 tokens (comprehensive comparison)
  - Cost: $0.027

Total: 8 API calls, ~$0.15 per user request
```

At $0.15 per request, 10,000 daily users would cost $1,500/day or $45,000/month. This is why cost optimization matters.

## Prompt Optimization Techniques

The first line of defense against high costs is writing efficient prompts. Here's how to reduce input tokens without sacrificing effectiveness.

### 1. Eliminate Redundancy

Many prompts repeat information or include unnecessary context:

```python
"""
Prompt optimization techniques.

Chapter 38: Cost Optimization
"""

import re


def analyze_system_prompt(prompt: str) -> dict:
    """Analyze a system prompt for optimization opportunities."""
    lines = prompt.strip().split('\n')
    words = prompt.split()
    estimated_tokens = len(words) + len(lines)

    analysis = {
        "original_length": len(prompt),
        "word_count": len(words),
        "estimated_tokens": estimated_tokens,
        "issues": [],
        "suggestions": [],
    }

    # Check for excessive whitespace
    if "  " in prompt or "\n\n\n" in prompt:
        analysis["issues"].append("Contains excessive whitespace")
        analysis["suggestions"].append("Remove extra spaces and newlines")

    # Check for long prompts
    if estimated_tokens > 1000:
        analysis["issues"].append(f"Prompt is very long ({estimated_tokens} tokens)")
        analysis["suggestions"].append("Consider splitting into focused sub-prompts")

    # Check for verbose phrasing
    verbose_phrases = [
        "you should always", "please make sure to",
        "it is important that you", "in order to"
    ]
    for phrase in verbose_phrases:
        if phrase in prompt.lower():
            analysis["issues"].append(f"Verbose phrasing: '{phrase}'")
            analysis["suggestions"].append("Use concise imperative statements")
            break

    return analysis


def optimize_system_prompt(prompt: str) -> str:
    """
    Optimize a system prompt by removing redundancy and verbosity.

    This reduces token count while maintaining clarity.
    """
    # Remove excessive whitespace
    optimized = re.sub(r' +', ' ', prompt)
    optimized = re.sub(r'\n{3,}', '\n\n', optimized)

    # Replace verbose phrases with concise alternatives
    replacements = [
        ("you should always", "Always"),
        ("please make sure to", ""),
        ("it is important that you", ""),
        ("in order to", "to"),
        ("due to the fact that", "because"),
        ("at this point in time", "now"),
    ]

    for verbose, concise in replacements:
        optimized = re.sub(
            re.escape(verbose),
            concise,
            optimized,
            flags=re.IGNORECASE
        )

    return optimized.strip()


# Example
verbose_prompt = """
You are a helpful assistant. It is important that you always help users.
You should always be polite and professional. Please make sure to think
carefully about each request before responding. In order to provide the
best possible assistance, you should analyze the user's needs thoroughly.
"""

print("Original prompt:")
print(f"  Tokens: {analyze_system_prompt(verbose_prompt)['estimated_tokens']}")
print()

optimized = optimize_system_prompt(verbose_prompt)
print("Optimized prompt:")
print(f"  Tokens: {analyze_system_prompt(optimized)['estimated_tokens']}")
print()
print(optimized)

# Output:
# Original: 68 tokens
# Optimized: 35 tokens (48% reduction)
# "You are a helpful assistant. Always help users.
#  Be polite and professional. Think carefully about
#  each request before responding. To provide the best
#  assistance, analyze the user's needs thoroughly."
```

### 2. Optimize Tool Definitions

Tool definitions are sent with every request. Bloated descriptions waste tokens on every single call:

```python
# ❌ Bad - verbose tool description (150 tokens)
calculator_tool = {
    "name": "calculator",
    "description": """This is a calculator tool that you can use whenever
you need to perform mathematical calculations. It supports addition,
subtraction, multiplication, and division operations. You should use this
tool whenever the user asks you to calculate something or when you need to
do math as part of solving a problem. The tool is very reliable and
accurate for all kinds of calculations.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The mathematical operation that you want to perform"
            },
            "a": {
                "type": "number",
                "description": "This is the first number in the calculation"
            },
            "b": {
                "type": "number",
                "description": "This is the second number in the calculation"
            }
        },
        "required": ["operation", "a", "b"]
    }
}

# ✅ Good - concise tool description (40 tokens)
calculator_tool = {
    "name": "calculator",
    "description": "Performs arithmetic: add, subtract, multiply, divide. Use for any calculations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "Operation to perform"
            },
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"}
        },
        "required": ["operation", "a", "b"]
    }
}
```

**Token savings**: 110 tokens per API call. With 1,000 calls/day, that's 110,000 tokens or $0.33/day saved just from one tool definition.

### 3. Context Window Management

As conversations grow, so do costs. Implement smart truncation:

```python
"""
Conversation history management with token budgets.

Chapter 38: Cost Optimization
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Message:
    """A conversation message."""
    role: str
    content: str
    tokens: int


class ConversationManager:
    """Manages conversation history with token budget constraints."""

    def __init__(self, max_tokens: int = 4000):
        """
        Initialize conversation manager.

        Args:
            max_tokens: Maximum tokens to keep in history
        """
        self.max_tokens = max_tokens
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, tokens: int) -> None:
        """Add a message to conversation history."""
        message = Message(role=role, content=content, tokens=tokens)
        self.messages.append(message)
        self._enforce_budget()

    def _enforce_budget(self) -> None:
        """Trim old messages to stay within token budget."""
        total_tokens = sum(m.tokens for m in self.messages)

        # Keep removing oldest messages until we're under budget
        # Always preserve the most recent message
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= removed.tokens

    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages in API format."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]

    def get_token_count(self) -> int:
        """Get current token count."""
        return sum(m.tokens for m in self.messages)


# Example usage
manager = ConversationManager(max_tokens=2000)

# Add several messages
manager.add_message("user", "Hello!", 10)
manager.add_message("assistant", "Hi! How can I help?", 20)
manager.add_message("user", "Tell me about Python.", 30)
manager.add_message("assistant", "Python is a programming language...", 500)
# ... many more messages ...

# Old messages are automatically removed to stay under budget
print(f"Messages kept: {len(manager.messages)}")
print(f"Total tokens: {manager.get_token_count()}")
```

## Response Length Management

Since output tokens cost 5x more than input tokens, controlling response length has the biggest impact on costs.

### Setting Appropriate max_tokens

```python
# ❌ Bad - wasteful max_tokens setting
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,  # Allows very long responses
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
# Might generate 50+ tokens for a simple answer

# ✅ Good - appropriate max_tokens for task
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,  # Sufficient for simple answers
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
# Generates ~10 tokens: "2+2 equals 4."
```

### Explicit Length Instructions

Include length requirements in your prompts:

```python
# Without length instruction
prompt = "Explain machine learning."
# Might generate 500+ tokens

# With length instruction
prompt = "Explain machine learning in 2-3 sentences."
# Generates 50-75 tokens

# For structured outputs
prompt = """Analyze this code and return JSON with:
{
  "issues": ["brief issue 1", "brief issue 2"],  // max 3 issues
  "score": 85  // 0-100
}
Keep each issue under 10 words."""
# Generates ~50 tokens vs 200+ without constraints
```

## Caching Strategies

The best API call is the one you don't make. Caching can dramatically reduce costs for repeated or similar queries.

### Basic Response Cache

Here's a complete caching implementation:

```python
"""
Caching system for AI agent responses.

Chapter 38: Cost Optimization
"""

import hashlib
import json
import time
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class ResponseCache:
    """
    In-memory cache for LLM responses.

    Features:
    - TTL-based expiration
    - LRU eviction when full
    - Hit/miss tracking
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries (None = no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self.hits = 0
        self.misses = 0

    def _make_key(self, prompt: str, **kwargs) -> str:
        """Create a cache key from prompt and parameters."""
        key_data = {"prompt": prompt, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        """
        Retrieve cached response.

        Returns None if not found or expired.
        """
        key = self._make_key(prompt, **kwargs)
        entry = self._cache.get(key)

        if entry is None:
            self.misses += 1
            return None

        # Check expiration
        if entry.is_expired():
            del self._cache[key]
            self._access_order.remove(key)
            self.misses += 1
            return None

        # Update access order (move to end for LRU)
        self._access_order.remove(key)
        self._access_order.append(key)
        entry.hit_count += 1
        self.hits += 1

        return entry.value

    def set(self, prompt: str, value: Any, **kwargs) -> None:
        """Store a response in cache."""
        key = self._make_key(prompt, **kwargs)

        # Evict least recently used entries if full
        while len(self._cache) >= self.max_size:
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

        # Calculate expiration time
        expires_at = None
        if self.ttl_seconds:
            expires_at = time.time() + self.ttl_seconds

        # Create and store entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=expires_at
        )

        self._cache[key] = entry
        self._access_order.append(key)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1%}",
            "total_requests": total,
        }

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0


# Example usage
cache = ResponseCache(max_size=100, ttl_seconds=3600)

# First request - cache miss
response = cache.get("What is the capital of France?")
if response is None:
    print("Cache miss - calling API")
    response = "Paris"  # Simulated API call
    cache.set("What is the capital of France?", response)

# Subsequent requests - cache hit
for _ in range(3):
    response = cache.get("What is the capital of France?")
    print(f"Cache hit! Response: {response}")

# Check statistics
stats = cache.get_stats()
print(f"\nCache stats: {stats}")
# Output: {'entries': 1, 'hits': 3, 'misses': 1, 'hit_rate': '75.0%', ...}
```

### Cache-Aware Agent Wrapper

Integrate caching into your agent:

```python
"""
Cache-aware agent wrapper.

Chapter 38: Cost Optimization
"""

from typing import Callable, Any


class CachedAgent:
    """Wraps an agent with response caching."""

    def __init__(self, agent_function: Callable, cache: ResponseCache):
        """
        Initialize cached agent.

        Args:
            agent_function: The agent function to wrap
            cache: ResponseCache instance
        """
        self.agent = agent_function
        self.cache = cache

    def run(self, prompt: str, **kwargs) -> str:
        """
        Run agent with caching.

        Checks cache first, only calls agent on cache miss.
        """
        # Try cache first
        cached_response = self.cache.get(prompt, **kwargs)
        if cached_response is not None:
            print("[CACHE HIT] Using cached response")
            return cached_response

        # Cache miss - call agent
        print("[CACHE MISS] Calling agent")
        response = self.agent(prompt, **kwargs)

        # Store in cache
        self.cache.set(prompt, response, **kwargs)

        return response


# Example usage
def my_agent(prompt: str, **kwargs) -> str:
    """Simulated agent that makes API calls."""
    # In real code, this would call Claude
    return f"Agent response to: {prompt}"

# Wrap agent with caching
cache = ResponseCache(max_size=1000, ttl_seconds=3600)
cached_agent = CachedAgent(my_agent, cache)

# First call - cache miss
result1 = cached_agent.run("Analyze this code")
# Output: [CACHE MISS] Calling agent

# Second call with same prompt - cache hit
result2 = cached_agent.run("Analyze this code")
# Output: [CACHE HIT] Using cached response

# Different prompt - cache miss
result3 = cached_agent.run("Different query")
# Output: [CACHE MISS] Calling agent
```

**Cost impact**: If 30% of requests hit the cache, you save 30% of your API costs immediately.

## Model Selection Strategy

Not every task needs the most powerful (and expensive) model. Choose the right model for each task:

```python
"""
Intelligent model selection for cost optimization.

Chapter 38: Cost Optimization
"""

from enum import Enum
from typing import Optional


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"          # Yes/no, classification, simple lookups
    MODERATE = "moderate"      # Analysis, coding, structured output
    COMPLEX = "complex"        # Creative work, complex reasoning


class ModelSelector:
    """Selects appropriate model based on task complexity and constraints."""

    # Model recommendations by complexity
    MODELS = {
        TaskComplexity.SIMPLE: "claude-3-5-haiku-20241022",
        TaskComplexity.MODERATE: "claude-sonnet-4-20250514",
        TaskComplexity.COMPLEX: "claude-opus-4-20250514",
    }

    @staticmethod
    def select_model(
        complexity: TaskComplexity,
        max_cost_per_request: Optional[float] = None
    ) -> str:
        """
        Select the most appropriate model.

        Args:
            complexity: Task complexity level
            max_cost_per_request: Optional cost constraint

        Returns:
            Model identifier
        """
        recommended = ModelSelector.MODELS[complexity]

        # If there's a cost constraint, might need to downgrade
        if max_cost_per_request is not None:
            # Calculate approximate cost for recommended model
            # (This is simplified - real code would estimate tokens)
            model_costs = {
                "claude-opus-4-20250514": 0.05,
                "claude-sonnet-4-20250514": 0.01,
                "claude-3-5-haiku-20241022": 0.003,
            }

            if model_costs[recommended] > max_cost_per_request:
                # Downgrade to cheaper model
                if complexity == TaskComplexity.COMPLEX:
                    recommended = "claude-sonnet-4-20250514"
                elif complexity == TaskComplexity.MODERATE:
                    recommended = "claude-3-5-haiku-20241022"

        return recommended

    @staticmethod
    def classify_task(prompt: str) -> TaskComplexity:
        """
        Classify task complexity from prompt.

        This is a simplified heuristic - real classification
        might use more sophisticated analysis.
        """
        prompt_lower = prompt.lower()

        # Simple tasks
        simple_keywords = ["yes or no", "true or false", "classify", "is this"]
        if any(kw in prompt_lower for kw in simple_keywords):
            return TaskComplexity.SIMPLE

        # Complex tasks
        complex_keywords = ["create", "design", "write a story", "be creative"]
        if any(kw in prompt_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX

        # Default to moderate
        return TaskComplexity.MODERATE


# Example usage
selector = ModelSelector()

# Simple task - use cheap model
task1 = "Is this email spam? Reply yes or no."
complexity1 = selector.classify_task(task1)
model1 = selector.select_model(complexity1)
print(f"Task: {task1[:40]}...")
print(f"Complexity: {complexity1.value}")
print(f"Model: {model1}")
print(f"→ Using Haiku (cheapest)")
print()

# Moderate task - use Sonnet
task2 = "Analyze this code for bugs and suggest improvements."
complexity2 = selector.classify_task(task2)
model2 = selector.select_model(complexity2)
print(f"Task: {task2[:40]}...")
print(f"Complexity: {complexity2.value}")
print(f"Model: {model2}")
print(f"→ Using Sonnet (balanced)")
print()

# Complex task - use Opus
task3 = "Write a creative short story about AI agents."
complexity3 = selector.classify_task(task3)
model3 = selector.select_model(complexity3)
print(f"Task: {task3[:40]}...")
print(f"Complexity: {complexity3.value}")
print(f"Model: {model3}")
print(f"→ Using Opus (most capable)")
```

**Cost impact**: Using Haiku instead of Opus for simple tasks saves 94% on costs ($0.80 vs $15.00 per million input tokens).

## Cost Monitoring and Alerts

The final piece: tracking costs in real-time and alerting when budgets are exceeded.

### Cost Tracker Implementation

```python
"""
Cost monitoring system with budgets and alerts.

Chapter 38: Cost Optimization
"""

import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CostRecord:
    """Record of a single API call's cost."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    request_type: str = "unknown"


@dataclass
class Budget:
    """Budget configuration with period and limits."""
    period: str  # "daily", "weekly", "monthly"
    limit: float  # Dollar amount
    alert_threshold: float = 0.8  # Alert at 80% of budget


class CostTracker:
    """
    Tracks API costs and enforces budgets.

    Features:
    - Per-request cost tracking
    - Daily/weekly/monthly aggregation
    - Budget enforcement with alerts
    - Cost summaries and reports
    """

    def __init__(self, budget: Optional[Budget] = None):
        """
        Initialize cost tracker.

        Args:
            budget: Optional budget to enforce
        """
        self.records: List[CostRecord] = []
        self.budget = budget
        self.total_cost = 0.0
        self.alerts: List[str] = []

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str = "unknown"
    ) -> float:
        """
        Track a single API call.

        Returns the cost of this call.
        """
        # Calculate cost
        cost = calculate_cost(input_tokens, output_tokens, model)

        # Create record
        record = CostRecord(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            request_type=request_type
        )

        self.records.append(record)
        self.total_cost += cost

        # Check budget
        if self.budget:
            self._check_budget()

        return cost

    def _check_budget(self) -> None:
        """Check if budget is exceeded and generate alerts."""
        if not self.budget:
            return

        # Get spending for budget period
        period_spending = self._get_period_spending(self.budget.period)
        utilization = period_spending / self.budget.limit

        # Generate alerts
        if utilization >= 1.0:
            alert = f"BUDGET EXCEEDED: {format_cost(period_spending)} / {format_cost(self.budget.limit)} ({self.budget.period})"
            if alert not in self.alerts:
                self.alerts.append(alert)
                print(f"⚠️  {alert}")

        elif utilization >= self.budget.alert_threshold:
            alert = f"Budget warning: {utilization:.0%} of {self.budget.period} budget used"
            if alert not in self.alerts:
                self.alerts.append(alert)
                print(f"⚠️  {alert}")

    def _get_period_spending(self, period: str) -> float:
        """Calculate spending for a time period."""
        now = time.time()

        # Determine period start time
        if period == "daily":
            period_start = now - 86400  # 24 hours
        elif period == "weekly":
            period_start = now - (86400 * 7)  # 7 days
        elif period == "monthly":
            period_start = now - (86400 * 30)  # 30 days
        else:
            return self.total_cost

        # Sum costs in period
        return sum(
            record.cost
            for record in self.records
            if record.timestamp >= period_start
        )

    def get_summary(self) -> Dict:
        """Get cost summary with breakdowns."""
        if not self.records:
            return {"total_cost": 0, "total_requests": 0}

        # Calculate aggregates
        total_input_tokens = sum(r.input_tokens for r in self.records)
        total_output_tokens = sum(r.output_tokens for r in self.records)

        # Group by model
        by_model = {}
        for record in self.records:
            if record.model not in by_model:
                by_model[record.model] = {"count": 0, "cost": 0.0}
            by_model[record.model]["count"] += 1
            by_model[record.model]["cost"] += record.cost

        return {
            "total_cost": self.total_cost,
            "total_requests": len(self.records),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "by_model": by_model,
            "daily_spending": self._get_period_spending("daily"),
            "weekly_spending": self._get_period_spending("weekly"),
            "monthly_spending": self._get_period_spending("monthly"),
            "alerts": self.alerts,
        }


# Example usage
budget = Budget(period="daily", limit=10.00, alert_threshold=0.8)
tracker = CostTracker(budget=budget)

# Track several API calls
tracker.track("claude-sonnet-4-20250514", 1000, 500, "code_review")
tracker.track("claude-sonnet-4-20250514", 1200, 600, "documentation")
tracker.track("claude-3-5-haiku-20241022", 800, 200, "classification")

# Get summary
summary = tracker.get_summary()
print(f"Total cost: {format_cost(summary['total_cost'])}")
print(f"Total requests: {summary['total_requests']}")
print(f"Daily spending: {format_cost(summary['daily_spending'])}")
print(f"Alerts: {summary['alerts']}")
```

## Common Pitfalls

### 1. Not Tracking Costs from Day One

**Problem**: Many developers add cost tracking after getting an unexpected bill. By then, you've lost valuable usage data and may have already overspent.

**Solution**: Add tracking before your first production API call:

```python
# ✅ Good - tracking from the start
tracker = CostTracker(budget=Budget(period="monthly", limit=1000.00))

def call_claude(prompt: str) -> str:
    response = client.messages.create(...)

    # Track immediately
    tracker.track(
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens
    )

    return response.content[0].text
```

### 2. Over-Aggressive Caching

**Problem**: Caching can cause issues when:
- Responses should vary (creative tasks, personalized content)
- Information changes frequently (real-time data)
- Cache keys are too broad (different contexts, same prompt)

**Solution**: Be selective about what to cache:

```python
# ✅ Good caching decisions
should_cache = {
    "What is the capital of France?": True,           # Static fact
    "What's the weather right now?": False,           # Real-time data
    "Write a creative story about...": False,         # Should be unique
    "Is this code valid Python?": True,               # Deterministic
    "Summarize this article: ...": True,              # Same input → same output
    "Chat with me about...": False,                   # Conversational
}
```

### 3. Wrong Model Selection Logic

**Problem**: Don't assume the cheapest model is always best.

**Solution**: Consider total cost of ownership:

```python
# Sometimes expensive models are cheaper overall
def calculate_total_cost(model: str, task: str) -> float:
    """Calculate total cost including retries."""
    base_cost = model_costs[model]

    # Cheap models might need retries due to failures
    if model == "haiku" and is_complex_task(task):
        # 30% chance of needing retry
        expected_cost = base_cost * 1.3
    else:
        expected_cost = base_cost

    # Factor in user experience costs
    if model == "haiku":
        # Slower responses might lose users
        churn_cost = 0.001  # $0.001 per slow response
        expected_cost += churn_cost

    return expected_cost
```

### 4. Ignoring Context Growth

**Problem**: Agents accumulate context over time. A conversation that starts at 500 tokens can grow to 50,000 tokens.

**Solution**: Always implement context management before you need it:

```python
# ✅ Good - context management from the start
class Agent:
    def __init__(self):
        self.conversation = ConversationManager(max_tokens=4000)
        self.cache = ResponseCache()
        self.tracker = CostTracker()

    def chat(self, message: str) -> str:
        # Add user message
        self.conversation.add_message("user", message, estimate_tokens(message))

        # ... make API call ...

        # Track cost
        self.tracker.track(...)

        return response
```

## Practical Exercise

**Task**: Build a cost dashboard that tracks agent spending

**Requirements**:

1. Create a simple web dashboard (HTML + JavaScript) that displays:
   - Current daily, weekly, monthly spending
   - Budget utilization as progress bars
   - Recent requests with costs
   - Alerts and warnings

2. The dashboard should read from a JSON file updated by `CostTracker`

3. Include a "Cost Projection" section that estimates end-of-month costs based on current usage

**Hints**:
- Use `CostTracker.get_summary()` for aggregate data
- Store usage records with timestamps for trend analysis
- Calculate projections by averaging daily costs and multiplying by remaining days
- Use chart libraries like Chart.js for visualizations

**Solution**: See `code/exercise_solution.py`

## Key Takeaways

- **Output tokens cost 5x more than input tokens**—controlling response length has the biggest impact on costs
- **Track costs from day one**—use the `CostTracker` class to monitor every API call
- **Cache aggressively but wisely**—caching can eliminate most redundant API calls, but not all requests should be cached
- **Use the right model for the task**—Haiku for simple tasks, Sonnet for most work, Opus for complex reasoning
- **Optimize prompts systematically**—remove redundancy, compress instructions, manage context window
- **Set budgets and alerts**—never be surprised by a bill again
- **Monitor token accumulation**—conversations grow over time, implement context management early
- **Calculate total cost of ownership**—sometimes expensive models are cheaper when you factor in retries and user experience

## What's Next

Cost optimization keeps your agents affordable. But users also care about speed. In the next chapter, **Latency Optimization**, you'll learn to make your agents faster through streaming, parallel execution, caching, and smart architecture choices. Because in production, every millisecond matters.
