---
appendix: A
title: "Python Refresher for Agent Development"
date: 2024-12-09
draft: false
---

# Appendix A: Python Refresher for Agent Development

This appendix covers Python features that are essential for building AI agents. If you're an intermediate Python programmer, some of this will be review—but even experienced developers often benefit from seeing these concepts in the context of agent development.

## Who This Is For

This appendix is for you if:
- You know Python basics but want to solidify advanced concepts
- You've seen `async/await` but aren't sure when to use it
- Type hints feel optional, and you want to understand why they matter
- You want to write cleaner, more maintainable agent code

## What We'll Cover

1. **Async/Await Essentials** — Why agents need asynchronous code
2. **Type Hints and Pydantic** — Making your agent code reliable
3. **Context Managers** — Resource management in agent workflows
4. **Decorators for Agents** — Retry logic, caching, and observability
5. **Dataclasses for Configuration** — Clean agent configuration

---

## 1. Async/Await Essentials

### Why Agents Need Async

AI agents make many I/O-bound operations:
- API calls to Claude
- Database queries
- HTTP requests to tools
- File operations

Without async, each operation blocks your program. With async, your agent can handle multiple operations concurrently—critical for responsive agents.

### The Basics

```python
"""
Synchronous vs asynchronous API calls.

Appendix A: Python Refresher
"""

import asyncio
import time
import anthropic


def sync_call(prompt: str) -> str:
    """Synchronous API call - blocks until complete."""
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text


async def async_call(prompt: str) -> str:
    """Asynchronous API call - doesn't block."""
    client = anthropic.AsyncAnthropic()
    
    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text


# Compare performance
def compare_sync_vs_async():
    """Demonstrate the power of async."""
    prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "What is the speed of light?",
    ]
    
    # Synchronous - one at a time
    print("Synchronous execution:")
    start = time.time()
    for prompt in prompts:
        result = sync_call(prompt)
        print(f"  {prompt[:30]}... -> {result[:50]}...")
    sync_time = time.time() - start
    print(f"Took {sync_time:.2f} seconds\n")
    
    # Asynchronous - all at once
    print("Asynchronous execution:")
    start = time.time()
    
    async def run_all():
        tasks = [async_call(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results
    
    results = asyncio.run(run_all())
    for prompt, result in zip(prompts, results):
        print(f"  {prompt[:30]}... -> {result[:50]}...")
    
    async_time = time.time() - start
    print(f"Took {async_time:.2f} seconds")
    print(f"Speedup: {sync_time / async_time:.1f}x faster")


if __name__ == "__main__":
    compare_sync_vs_async()
```

### Key Concepts

**`async def`** — Declares an asynchronous function (coroutine)
```python
async def fetch_data():
    # This is a coroutine
    return await some_async_operation()
```

**`await`** — Waits for an async operation to complete
```python
result = await async_call("Hello")  # Waits here, but doesn't block other tasks
```

**`asyncio.gather()`** — Runs multiple coroutines concurrently
```python
results = await asyncio.gather(
    async_call("First"),
    async_call("Second"),
    async_call("Third"),
)
```

**`asyncio.run()`** — Runs the main async function
```python
asyncio.run(main())  # Entry point for async programs
```

### Async Patterns for Agents

**Pattern 1: Parallel Tool Calls**

When an agent needs multiple tools at once:

```python
async def agent_with_parallel_tools(query: str):
    """Agent that calls multiple tools in parallel."""
    async with anthropic.AsyncAnthropic() as client:
        # Get initial tool calls
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[weather_tool, calculator_tool, search_tool],
            messages=[{"role": "user", "content": query}]
        )
        
        # Execute all tool calls in parallel
        tool_tasks = []
        for block in response.content:
            if block.type == "tool_use":
                task = execute_tool_async(block.name, block.input)
                tool_tasks.append(task)
        
        # Wait for all tools to complete
        tool_results = await asyncio.gather(*tool_tasks)
        
        return tool_results
```

**Pattern 2: Timeout Protection**

Prevent tools from hanging forever:

```python
async def safe_tool_call(tool_func, timeout_seconds: float = 30.0):
    """Call a tool with a timeout."""
    try:
        result = await asyncio.wait_for(
            tool_func(),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        return {"error": f"Tool timed out after {timeout_seconds}s"}
```

**Pattern 3: Streaming Responses**

Stream Claude's responses for better UX:

```python
async def stream_agent_response(prompt: str):
    """Stream the agent's response token by token."""
    async with anthropic.AsyncAnthropic() as client:
        async with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
            print()  # Newline at end
```

> **💡 Tip:** Use async when your agent makes multiple API calls, database queries, or HTTP requests. Use sync for simple scripts or when you only make one call at a time.

---

## 2. Type Hints and Pydantic

### Why Type Hints Matter

Type hints make your agent code:
- **Self-documenting** — Function signatures explain themselves
- **Easier to debug** — Catch errors before runtime
- **IDE-friendly** — Better autocomplete and refactoring
- **More maintainable** — Changes are safer

### Basic Type Hints

```python
"""
Type hints for agent development.

Appendix A: Python Refresher
"""

from typing import Optional, Union, List, Dict, Callable, Any


def simple_types(
    name: str,
    age: int,
    score: float,
    active: bool
) -> str:
    """Basic type hints for primitives."""
    return f"{name} is {age} years old"


def collection_types(
    items: list[str],                    # List of strings
    scores: dict[str, int],              # Dict with string keys, int values
    data: tuple[str, int, bool],         # Tuple with specific types
    unique: set[str],                    # Set of strings
) -> list[dict[str, Any]]:              # Return complex nested type
    """Type hints for collections."""
    return [{"item": item, "score": scores.get(item, 0)} for item in items]


def optional_and_union(
    required: str,
    optional: Optional[str] = None,      # Can be str or None
    multiple: Union[str, int, None] = None,  # Can be str, int, or None
) -> str | None:                         # Python 3.10+ union syntax
    """Optional and union types."""
    if optional:
        return optional
    return None


def function_types(
    callback: Callable[[str], int],      # Function that takes str, returns int
    handler: Callable[..., None],        # Function with any args, returns None
) -> None:
    """Type hints for functions."""
    result = callback("hello")
    handler(result)
```

### Type Hints for Agent Components

```python
from typing import Protocol, Literal
from dataclasses import dataclass


# Protocol: Define interface without inheritance
class Tool(Protocol):
    """A tool that an agent can use."""
    
    name: str
    description: str
    
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool."""
        ...


# Literal: Restrict to specific values
MessageRole = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """A message in a conversation."""
    role: MessageRole
    content: str


# Generic types for agent state
from typing import TypeVar, Generic

T = TypeVar('T')


class AgentState(Generic[T]):
    """Generic agent state container."""
    
    def __init__(self, initial_value: T):
        self.value: T = initial_value
    
    def update(self, new_value: T) -> None:
        self.value = new_value
    
    def get(self) -> T:
        return self.value


# Usage
state_int = AgentState[int](42)
state_str = AgentState[str]("hello")
```

### Pydantic for Validation

Pydantic combines type hints with runtime validation—essential for agent configuration and tool inputs:

```python
"""
Pydantic for agent configuration and validation.

Appendix A: Python Refresher
"""

from pydantic import BaseModel, Field, validator, field_validator
from typing import Optional, Literal


class ToolInput(BaseModel):
    """
    Base class for tool inputs.
    
    Pydantic validates types and constraints automatically.
    """
    
    query: str = Field(
        ...,  # Required
        min_length=1,
        max_length=1000,
        description="The query to process"
    )
    
    max_results: int = Field(
        default=10,
        ge=1,  # Greater than or equal to 1
        le=100,  # Less than or equal to 100
        description="Maximum number of results"
    )
    
    include_metadata: bool = Field(
        default=False,
        description="Whether to include metadata"
    )
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Custom validation for query."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class AgentConfig(BaseModel):
    """Configuration for an AI agent."""
    
    model: Literal["claude-sonnet-4-20250514", "claude-opus-4-20250514"]
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    system_prompt: str
    
    # Pydantic v2 configuration
    model_config = {
        "frozen": True,  # Immutable after creation
        "extra": "forbid",  # Don't allow extra fields
    }


class ToolResult(BaseModel):
    """Result from a tool execution."""
    
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    
    @field_validator('error')
    @classmethod
    def error_only_if_not_success(cls, v: Optional[str], info) -> Optional[str]:
        """Validate error field consistency."""
        # Access other field values through info.data
        if not info.data.get('success') and v is None:
            raise ValueError("Error message required when success is False")
        return v


# Usage examples
if __name__ == "__main__":
    # Valid tool input
    tool_input = ToolInput(query="search for python tutorials", max_results=5)
    print(f"Tool input: {tool_input.query}, max: {tool_input.max_results}")
    
    # Validation catches errors
    try:
        bad_input = ToolInput(query="   ", max_results=500)
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Immutable config
    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant"
    )
    
    # Can't modify frozen config
    try:
        config.temperature = 0.5
    except Exception as e:
        print(f"Can't modify frozen config: {e}")
```

### Why Pydantic Matters for Agents

1. **Automatic validation** — Catch bad tool inputs before they reach your functions
2. **Type coercion** — `"42"` becomes `42` automatically if the field is an `int`
3. **Clear error messages** — Know exactly what went wrong and where
4. **JSON serialization** — Convert models to/from JSON effortlessly
5. **Documentation** — Field descriptions become API documentation

> **💡 Tip:** Use Pydantic for all tool input schemas, agent configurations, and API request/response models. The validation is free insurance against bad data.

---

## 3. Context Managers

Context managers handle resource setup and cleanup automatically—critical when managing API connections, file handles, and database sessions in agents.

### The Basics

```python
"""
Context managers for resource management.

Appendix A: Python Refresher
"""

import os
from typing import Optional
from contextlib import contextmanager
import anthropic


# The with statement ensures cleanup happens
def basic_example():
    """File handles are automatically closed."""
    with open("agent.log", "w") as f:
        f.write("Agent started\n")
        # File automatically closes when exiting the block
        # Even if an exception occurs!


# Creating your own context manager
@contextmanager
def timer(name: str):
    """Time a block of code."""
    import time
    start = time.perf_counter()
    print(f"{name}: Starting...")
    
    try:
        yield  # Code inside 'with' block runs here
    finally:
        # This always runs, even if there's an exception
        elapsed = time.perf_counter() - start
        print(f"{name}: Completed in {elapsed:.2f}s")


# Usage
with timer("API Call"):
    # Simulated work
    import time
    time.sleep(0.5)
```

### Context Managers for Agents

**Pattern 1: API Client Management**

```python
class ManagedAnthropicClient:
    """
    Context manager for Anthropic API client.
    
    Ensures the client is properly initialized and cleaned up.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client: Optional[anthropic.Anthropic] = None
    
    def __enter__(self) -> anthropic.Anthropic:
        """Initialize the client when entering the context."""
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        return self.client
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context."""
        # The Anthropic client doesn't need explicit cleanup,
        # but this pattern is useful for database connections,
        # file handles, etc.
        self.client = None
        
        # Return False to propagate exceptions
        return False


# Usage
with ManagedAnthropicClient() as client:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(message.content[0].text)
```

**Pattern 2: Agent Conversation Context**

```python
from contextlib import contextmanager
from typing import Generator


@contextmanager
def conversation_context(agent: "Agent", user_id: str) -> Generator[list, None, None]:
    """
    Manage a conversation context for an agent.
    
    Loads conversation history on enter, saves on exit.
    """
    # Setup: Load conversation history
    conversation = agent.load_conversation(user_id)
    print(f"Loaded conversation with {len(conversation)} messages")
    
    try:
        yield conversation  # Give conversation to the with block
    finally:
        # Cleanup: Save conversation history
        agent.save_conversation(user_id, conversation)
        print(f"Saved conversation with {len(conversation)} messages")


# Usage in an agent
class Agent:
    def __init__(self):
        self.conversations: dict[str, list] = {}
    
    def load_conversation(self, user_id: str) -> list:
        return self.conversations.get(user_id, [])
    
    def save_conversation(self, user_id: str, conversation: list):
        self.conversations[user_id] = conversation
    
    def chat(self, user_id: str, message: str) -> str:
        with conversation_context(self, user_id) as conversation:
            conversation.append({"role": "user", "content": message})
            
            # Make API call with conversation history
            # ...
            
            response_text = "Agent response here"
            conversation.append({"role": "assistant", "content": response_text})
            
            return response_text
```

**Pattern 3: Temporary State**

```python
@contextmanager
def temporary_agent_mode(agent: "Agent", mode: str):
    """
    Temporarily change an agent's mode.
    
    Restores original mode on exit.
    """
    original_mode = agent.mode
    agent.mode = mode
    print(f"Agent mode changed: {original_mode} → {mode}")
    
    try:
        yield agent
    finally:
        agent.mode = original_mode
        print(f"Agent mode restored: {mode} → {original_mode}")


# Usage
agent = Agent()
agent.mode = "normal"

# Temporarily switch to verbose mode
with temporary_agent_mode(agent, "verbose"):
    # Agent is in verbose mode here
    agent.process("Analyze this data...")

# Agent is back to normal mode here
```

> **💡 Tip:** Use context managers whenever you need paired setup/cleanup operations: opening/closing connections, acquiring/releasing locks, changing/restoring state, or starting/stopping timers.

---

## 4. Decorators for Agents

Decorators add functionality to functions without modifying their code—perfect for cross-cutting concerns like retry logic, caching, logging, and timing.

### The Basics

```python
"""
Decorators for agent development.

Appendix A: Python Refresher
"""

from functools import wraps
from typing import Callable, Any
import time


def simple_decorator(func: Callable) -> Callable:
    """A simple decorator that prints before/after."""
    
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    
    return wrapper


@simple_decorator
def greet(name: str) -> str:
    return f"Hello, {name}!"


# When you call greet(), you're actually calling wrapper()
print(greet("Alice"))
# Output:
# Calling greet
# Finished greet
# Hello, Alice!
```

### Decorators for Agent Operations

**Pattern 1: Retry with Exponential Backoff**

```python
import time
from functools import wraps
from typing import TypeVar, Callable

T = TypeVar('T')


def retry(max_attempts: int = 3, backoff_factor: float = 2.0):
    """
    Retry a function with exponential backoff.
    
    Essential for handling transient API failures.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        sleep_time = backoff_factor ** (attempt - 1)
                        print(f"Attempt {attempt} failed: {e}")
                        print(f"Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        print(f"All {max_attempts} attempts failed")
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


# Usage
@retry(max_attempts=3, backoff_factor=2.0)
def make_api_call(prompt: str) -> str:
    """Make an API call with automatic retry."""
    # This will retry up to 3 times with exponential backoff
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

**Pattern 2: Function Caching**

```python
from functools import lru_cache, wraps
import hashlib
import json


def cache_with_ttl(max_size: int = 128, ttl_seconds: float = 300):
    """
    Cache function results with time-to-live.
    
    Useful for expensive tool calls that don't change frequently.
    """
    def decorator(func):
        # Simple time-based cache
        cache: dict[str, tuple[Any, float]] = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
            key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            now = time.time()
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    print(f"Cache hit for {func.__name__}")
                    return value
            
            # Cache miss - call function
            print(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = (result, now)
            
            # Limit cache size
            if len(cache) > max_size:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator


@cache_with_ttl(max_size=100, ttl_seconds=60)
def fetch_weather(city: str) -> dict:
    """Fetch weather data (simulated)."""
    print(f"Fetching weather for {city}...")
    time.sleep(0.5)  # Simulate API call
    return {"city": city, "temp": 72, "condition": "sunny"}


# First call - cache miss
weather1 = fetch_weather("San Francisco")

# Second call within 60s - cache hit
weather2 = fetch_weather("San Francisco")
```

**Pattern 3: Timing and Performance Monitoring**

```python
def measure_time(func: Callable) -> Callable:
    """Measure and log function execution time."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result
    
    return wrapper


@measure_time
def process_with_agent(text: str) -> str:
    """Process text with timing."""
    # Agent processing here
    time.sleep(0.2)  # Simulate work
    return f"Processed: {text}"
```

**Pattern 4: Logging Decorator**

```python
import logging

logger = logging.getLogger(__name__)


def log_calls(func: Callable) -> Callable:
    """Log all calls to a function with arguments and results."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


@log_calls
def execute_tool(tool_name: str, **params) -> dict:
    """Execute a tool with full logging."""
    # Tool execution logic
    return {"status": "success", "tool": tool_name}
```

> **💡 Tip:** Chain decorators by stacking them. Order matters—they're applied bottom-to-top:

```python
@retry(max_attempts=3)
@measure_time
@log_calls
def critical_operation():
    # This will be: retry(measure_time(log_calls(critical_operation)))
    pass
```

---

## 5. Dataclasses for Configuration

Dataclasses provide a clean way to define configuration objects with less boilerplate than regular classes.

### The Basics

```python
"""
Dataclasses for agent configuration.

Appendix A: Python Refresher
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for an AI agent using dataclasses."""
    
    # Required fields
    model: str
    max_tokens: int
    
    # Optional fields with defaults
    temperature: float = 1.0
    system_prompt: str = "You are a helpful assistant."
    
    # Field with factory function (for mutable defaults)
    tools: list[str] = field(default_factory=list)
    
    # Computed field (not in __init__)
    version: str = field(default="1.0.0", init=False)
    
    def __post_init__(self):
        """Validate after initialization."""
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")


# Usage
config = AgentConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.7,
    tools=["calculator", "search"]
)

print(config.model)  # claude-sonnet-4-20250514
print(config.version)  # 1.0.0

# Convert to dict for serialization
config_dict = asdict(config)
print(config_dict)
```

### Dataclasses for Agent State

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    ERROR = "error"


@dataclass
class AgentState:
    """Complete state of an agent during execution."""
    
    # Identity
    agent_id: str
    user_id: str
    
    # Status
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    
    # Conversation
    messages: list[dict] = field(default_factory=list)
    conversation_id: str = field(default_factory=lambda: f"conv_{int(time.time())}")
    
    # Metrics
    total_tokens: int = 0
    tool_calls_made: int = 0
    errors_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})
        self.last_activity = datetime.now()
    
    def record_tool_call(self) -> None:
        """Record a tool call."""
        self.tool_calls_made += 1
        self.last_activity = datetime.now()
    
    def record_error(self) -> None:
        """Record an error."""
        self.errors_count += 1
        self.status = AgentStatus.ERROR
        self.last_activity = datetime.now()


# Usage
state = AgentState(
    agent_id="agent_123",
    user_id="user_456"
)

state.add_message("user", "What's the weather?")
state.status = AgentStatus.CALLING_TOOL
state.record_tool_call()
```

### Frozen Dataclasses (Immutable Configuration)

```python
@dataclass(frozen=True)
class ImmutableConfig:
    """Configuration that can't be changed after creation."""
    
    api_key: str
    model: str
    max_tokens: int = 1024


config = ImmutableConfig(
    api_key="sk-ant-...",
    model="claude-sonnet-4-20250514"
)

# This raises an error:
# config.max_tokens = 2048  # FrozenInstanceError
```

### Dataclasses with Slots (Memory Optimization)

```python
@dataclass(slots=True)
class OptimizedState:
    """Use __slots__ for better memory efficiency."""
    
    message_count: int
    token_count: int
    status: str
    
    # With slots=True, Python doesn't create a __dict__ for each instance
    # This saves memory when you have many instances
```

> **💡 Tip:** Use dataclasses for configuration objects, state containers, and simple data structures. Use Pydantic (which builds on dataclasses) when you need validation, serialization, and API integration.

---

## Putting It All Together

Here's a realistic agent component that uses all five concepts:

```python
"""
Complete example combining all Python features.

Appendix A: Python Refresher
"""

import os
import asyncio
import time
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps
from typing import AsyncGenerator, Callable, Optional
from pydantic import BaseModel, Field
import anthropic


# --- Configuration with Dataclasses ---

@dataclass
class AgentConfig:
    """Agent configuration."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 1.0
    max_retries: int = 3


# --- Request/Response with Pydantic ---

class AgentRequest(BaseModel):
    """Agent API request."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent API response."""
    response: str
    tokens_used: int
    duration_ms: float


# --- Decorators for Resilience ---

def retry_async(max_attempts: int = 3):
    """Retry async functions with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts:
                        wait_time = 2 ** (attempt - 1)
                        await asyncio.sleep(wait_time)
                    else:
                        raise
        return wrapper
    return decorator


def measure_async_time(func: Callable):
    """Measure async function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{func.__name__} took {elapsed:.2f}ms")
        return result
    return wrapper


# --- Context Manager for Client ---

@asynccontextmanager
async def anthropic_client() -> AsyncGenerator[anthropic.AsyncAnthropic, None]:
    """Managed Anthropic client with proper cleanup."""
    client = anthropic.AsyncAnthropic()
    try:
        yield client
    finally:
        await client.close()


# --- Complete Agent ---

class Agent:
    """
    Production-ready agent using all Python features.
    
    - Type hints for safety
    - Async/await for concurrency
    - Decorators for retry and timing
    - Context managers for resources
    - Pydantic for validation
    - Dataclasses for config
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
    
    @retry_async(max_attempts=3)
    @measure_async_time
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a request with full type safety and resilience."""
        async with anthropic_client() as client:
            start = time.perf_counter()
            
            message = await client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": request.prompt}]
            )
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            return AgentResponse(
                response=message.content[0].text,
                tokens_used=message.usage.input_tokens + message.usage.output_tokens,
                duration_ms=duration_ms
            )


# --- Usage Example ---

async def main():
    """Demonstrate the complete agent."""
    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        max_retries=3
    )
    
    agent = Agent(config)
    
    request = AgentRequest(
        prompt="What are the three most important Python features for AI agents?"
    )
    
    response = await agent.process(request)
    
    print(f"Response: {response.response[:100]}...")
    print(f"Tokens: {response.tokens_used}")
    print(f"Duration: {response.duration_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Common Pitfalls

**1. Forgetting `await` in async functions**

```python
# Wrong
result = async_function()  # Returns coroutine, not result

# Right
result = await async_function()
```

**2. Using mutable defaults in dataclasses**

```python
# Wrong - all instances share the same list!
@dataclass
class Bad:
    items: list = []

# Right - use field with default_factory
@dataclass
class Good:
    items: list = field(default_factory=list)
```

**3. Not using `@wraps` in decorators**

```python
# Wrong - loses function metadata
def bad_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

# Right - preserves function name, docstring, etc.
from functools import wraps

def good_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

**4. Mixing sync and async code incorrectly**

```python
# Wrong - can't await sync function
result = await sync_function()  # Error

# Right - run sync function in executor
import asyncio
result = await asyncio.get_event_loop().run_in_executor(None, sync_function)
```

---

## Practical Exercise

**Task:** Build a resilient API client for tool execution

**Requirements:**

1. Create a `ToolExecutor` class using all five concepts covered
2. Use dataclasses for configuration
3. Use Pydantic for tool input validation
4. Use async/await for concurrent tool execution
5. Use decorators for retry logic and timing
6. Use context managers for resource management

**Hints:**

- Define a `ToolConfig` dataclass with `timeout`, `max_retries`, and `cache_ttl`
- Define a `ToolInput` Pydantic model that validates tool parameters
- Implement `async def execute_tool()` with retry decorator
- Use an async context manager to manage the tool execution lifecycle

**Solution:** See `code/exercise_solution.py`

---

## Key Takeaways

- **Async/await** enables concurrent operations—essential for responsive agents that make multiple API calls
- **Type hints + Pydantic** catch errors early and make code self-documenting
- **Context managers** ensure proper resource cleanup, even when errors occur
- **Decorators** add cross-cutting concerns (retry, caching, logging) without cluttering business logic
- **Dataclasses** reduce boilerplate for configuration and state objects

These five features aren't just nice to have—they're the foundation of production-quality agent code. Master them, and you'll write agents that are reliable, maintainable, and performant.

---

## What's Next

This appendix covered the Python essentials for building agents. The other appendices provide quick references for common tasks:

- **Appendix B**: API Reference Quick Guide
- **Appendix C**: Tool Design Patterns
- **Appendix D**: Prompt Engineering for Agents
- **Appendix E**: Troubleshooting Guide
- **Appendix F**: Glossary
- **Appendix G**: Resources and Further Reading

Now that you have these Python fundamentals solid, you're ready to build agents that are not just functional, but professional-grade.