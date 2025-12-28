# Code Developer Skill

This skill provides standards and guidelines for creating code examples in "Building AI Agents from Scratch with Python."

## Purpose

Create code examples that:
1. **Teach effectively** - Illuminate concepts clearly
2. **Work perfectly** - Are tested and production-quality
3. **Follow standards** - Match book style guidelines
4. **Build progressively** - Advance logically through topics

## When to Use This Skill

Trigger this skill when:
- Writing new code examples for chapters
- Creating exercise solutions
- Updating existing code to fix bugs or improve clarity
- Implementing new patterns or architectures
- Building on previous chapter code

## Code Structure Standards

### File Organization

Each chapter's code folder follows this structure:

```
chapter-XX-title/code/
├── README.md                   # Explains all code files
├── example_01_description.py   # First example (simple)
├── example_02_description.py   # Second example (builds complexity)
├── example_03_description.py   # Third example (advanced)
├── exercise_task_name.py       # Exercise solution
├── .env.example               # Template for environment variables
└── pyproject.toml             # uv dependencies (if needed)
```

### Naming Conventions

**Example files:**
- `example_01_simple_loop.py` - Descriptive, numbered sequentially
- `example_02_with_errors.py` - Shows progression
- `example_03_complete.py` - Final, polished version

**Exercise files:**
- `exercise_memory_limit.py` - Matches exercise in chapter
- `exercise_token_counter.py` - One file per exercise
- `exercise_solution.py` - If only one exercise

## Required Elements in Every Python File

### 1. Module Docstring
Every file starts with a comprehensive docstring:

```python
"""
[One-line summary of what this demonstrates]

Chapter XX: [Chapter Title]

This example demonstrates [core concept]. It shows how to [key learning goal].
[Any important notes about what makes this example special or different from others].

Requirements:
- Python 3.10+
- anthropic SDK
- python-dotenv

Setup:
1. Copy .env.example to .env
2. Add your ANTHROPIC_API_KEY to .env
3. Run: uv run example_01.py
   Or: python example_01.py

Concepts Covered:
- [Concept 1]
- [Concept 2]
- [Concept 3]
"""
```

### 2. Complete Imports
Show ALL imports - never assume anything:

```python
import os
import sys
from typing import Any, Dict, List
from dotenv import load_dotenv
import anthropic

# Only for chapters 14+
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from chapter_14.code.augmented_llm import AugmentedLLM
```

### 3. Environment Setup with Security
ALWAYS use the dotenv pattern:

```python
# Load environment variables from .env file
load_dotenv()

# Get API key with error handling
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY not found in environment variables. "
        "Copy .env.example to .env and add your API key."
    )

# Initialize client
client = anthropic.Anthropic(api_key=api_key)
```

### 4. Type Hints on All Functions
Use modern Python 3.10+ type hints:

```python
def process_message(
    message: str,
    history: list[dict[str, str]],
    system_prompt: str = ""
) -> dict[str, Any]:
    """Process a user message and return the agent response.

    Args:
        message: The user's input message
        history: Previous conversation messages
        system_prompt: Optional system prompt for behavior guidance

    Returns:
        Dict containing the response text and metadata

    Raises:
        ValueError: If message is empty
        APIError: If the API call fails
    """
    # Implementation here
    pass
```

### 5. Docstrings (Google Style)
All public functions and classes need docstrings:

```python
def create_tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any]
) -> dict[str, Any]:
    """Create a tool definition for Claude's function calling.

    Args:
        name: The tool name (must be lowercase with underscores)
        description: Clear description of what the tool does
        parameters: JSON Schema defining the tool's input parameters

    Returns:
        A properly formatted tool definition dict

    Example:
        >>> tool = create_tool_definition(
        ...     name="get_weather",
        ...     description="Get current weather for a location",
        ...     parameters={
        ...         "type": "object",
        ...         "properties": {
        ...             "location": {"type": "string"}
        ...         }
        ...     }
        ... )
    """
    return {
        "name": name,
        "description": description,
        "input_schema": parameters
    }
```

### 6. Error Handling
Show proper error handling appropriate to the chapter:

**Basic (early chapters):**
```python
try:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages
    )
    return response.content[0].text
except anthropic.APIError as e:
    print(f"API call failed: {e}")
    raise
```

**Advanced (later chapters):**
```python
from anthropic import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APIStatusError
)

def call_with_retry(
    messages: list[dict[str, str]],
    max_retries: int = 3
) -> str:
    """Call API with exponential backoff retry logic.

    Args:
        messages: The messages to send
        max_retries: Maximum number of retry attempts

    Returns:
        The model's response text

    Raises:
        APIError: If all retries are exhausted
    """
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages
            )
            return response.content[0].text

        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIConnectionError as e:
            print(f"Connection error: {e}")
            if attempt == max_retries - 1:
                raise

        except APIStatusError as e:
            print(f"API status error: {e.status_code}")
            raise  # Don't retry on status errors
```

### 7. Configuration Constants
Define constants at module level:

```python
# Model configuration
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
TEMPERATURE = 1.0

# Agent configuration
MAX_ITERATIONS = 10
MAX_CONVERSATION_LENGTH = 100
```

### 8. Main Function and Entry Point
Every example should be runnable:

```python
def main() -> None:
    """Main example demonstrating [concept]."""
    print("=== Chapter XX: [Title] ===\n")

    # Your example code here
    result = demonstrate_concept()

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
```

## Code Quality Standards

### Clarity Over Cleverness
**This is a teaching tool, not a code golf competition.**

❌ **Too clever:**
```python
def process(msgs): return [client.messages.create(m="gpt",t=1024,msg=[m])
    for m in msgs if m]
```

✅ **Clear and teachable:**
```python
def process_messages(messages: list[str]) -> list[str]:
    """Process a list of messages through the LLM.

    Args:
        messages: List of user messages to process

    Returns:
        List of LLM responses
    """
    responses = []

    for message in messages:
        if not message:  # Skip empty messages
            continue

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": message}]
        )

        responses.append(response.content[0].text)

    return responses
```

### Meaningful Variable Names

❌ **Unclear:**
```python
def f(x, y, z):
    r = c.m.c(m=x, t=y, msg=z)
    return r.c[0].t
```

✅ **Self-documenting:**
```python
def call_llm(
    model: str,
    max_tokens: int,
    messages: list[dict[str, str]]
) -> str:
    """Call the LLM with the given parameters."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages
    )
    return response.content[0].text
```

### Comments for Learning Moments
Don't over-comment, but DO explain non-obvious decisions:

```python
def run_agentic_loop(task: str, max_iterations: int = 10) -> str:
    """Run the agent loop with iteration limit."""
    messages = [{"role": "user", "content": task}]
    iterations = 0

    while iterations < max_iterations:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages,
            tools=tools
        )

        # Check stop reason - this determines if we continue looping
        # "end_turn" means the agent is done
        # "tool_use" means we need to execute tools and continue
        if response.stop_reason == "end_turn":
            break

        # Process tool uses (if any)
        for content_block in response.content:
            if content_block.type == "tool_use":
                result = execute_tool(content_block)
                # Important: Add BOTH the assistant's message AND tool result
                # The assistant message includes the tool_use block
                # The tool result provides the output
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    }]
                })

        iterations += 1

    return messages[-1]["content"]
```

### Progressive Complexity
Show evolution from simple to complex:

**example_01_simple.py - Minimal viable example:**
```python
def simple_chat(user_message: str) -> str:
    """Most basic chat implementation."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text
```

**example_02_with_history.py - Add conversation memory:**
```python
def chat_with_history(
    user_message: str,
    history: list[dict[str, str]]
) -> tuple[str, list[dict[str, str]]]:
    """Chat with conversation history."""
    # Copy history to avoid mutation
    messages = history.copy()
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=messages
    )

    assistant_message = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message, messages
```

**example_03_complete.py - Production-ready:**
```python
class ChatAgent:
    """Production-ready chat agent with history and error handling."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        system_prompt: str = ""
    ):
        """Initialize the chat agent."""
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.history: list[dict[str, str]] = []

    def chat(self, user_message: str) -> str:
        """Send a message and get response."""
        try:
            self.history.append({"role": "user", "content": user_message})

            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=self.history
            )

            assistant_message = response.content[0].text
            self.history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except anthropic.APIError as e:
            print(f"API error: {e}")
            # Remove the user message we added since the call failed
            self.history.pop()
            raise
```

## Security Best Practices

### 1. Never Hardcode Secrets
❌ **Dangerous:**
```python
client = anthropic.Anthropic(api_key="sk-ant-api03-...")
```

✅ **Secure:**
```python
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found")
client = anthropic.Anthropic(api_key=api_key)
```

### 2. Provide .env.example
Every chapter with code needs:

```bash
# .env.example
# Copy this to .env and add your actual API key

ANTHROPIC_API_KEY=your-api-key-here
```

### 3. Input Validation
Validate user input appropriately:

```python
def validate_message(message: str) -> None:
    """Validate user message input.

    Args:
        message: The message to validate

    Raises:
        ValueError: If message is invalid
    """
    if not message:
        raise ValueError("Message cannot be empty")

    if not message.strip():
        raise ValueError("Message cannot be only whitespace")

    if len(message) > 10000:
        raise ValueError("Message too long (max 10000 characters)")
```

## Common Code Patterns

### Pattern 1: Simple API Call
```python
def call_llm(prompt: str, system: str = "") -> str:
    """Make a simple LLM call.

    Args:
        prompt: The user's prompt
        system: Optional system prompt

    Returns:
        The model's response
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### Pattern 2: Tool Use Loop
```python
def call_with_tools(
    user_message: str,
    tools: list[dict],
    tool_implementations: dict[str, callable]
) -> str:
    """Call LLM with tools and execute tool use loop.

    Args:
        user_message: User's message
        tools: List of tool definitions
        tool_implementations: Dict mapping tool names to functions

    Returns:
        Final response text
    """
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages,
            tools=tools
        )

        # Check if we're done
        if response.stop_reason == "end_turn":
            # Find the text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Process tool uses
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "tool_use":
                # Execute the tool
                tool_name = block.name
                tool_input = block.input
                tool_function = tool_implementations[tool_name]
                result = tool_function(**tool_input)

                # Add tool result
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    }]
                })
```

### Pattern 3: Using AugmentedLLM (Chapters 14+)
```python
from chapter_14.code.augmented_llm import AugmentedLLM, ToolRegistry

def example_with_augmented_llm() -> None:
    """Example using the AugmentedLLM class."""

    # Define tools
    registry = ToolRegistry()
    registry.register_tool(
        name="calculator",
        description="Perform basic arithmetic",
        parameters={
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "sub"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        },
        implementation=calculator_function
    )

    # Create augmented LLM
    llm = AugmentedLLM(
        model="claude-sonnet-4-20250514",
        tools=registry,
        system_prompt="You are a helpful assistant."
    )

    # Use it
    response = llm.process("What is 25 + 17?")
    print(response)
```

## Testing Your Code

### Before Submitting Code:

1. **Run it multiple times**
   ```bash
   python example_01.py
   python example_01.py
   python example_01.py
   ```

2. **Test edge cases**
   - Empty inputs
   - Very long inputs
   - API errors (disconnect wifi)
   - Rate limiting (rapid calls)

3. **Verify imports**
   - Start fresh terminal
   - Clear any cached modules
   - Make sure it works standalone

4. **Check security**
   - No hardcoded keys
   - .env.example provided
   - .env in .gitignore

5. **Verify formatting**
   ```bash
   black example_01.py
   ```

## Code Review Checklist

Before marking code as complete:

### Functionality
- [ ] Code runs without errors
- [ ] Produces expected output
- [ ] Handles edge cases
- [ ] Error messages are helpful

### Teaching
- [ ] Demonstrates the chapter concept clearly
- [ ] Appropriate complexity for chapter position
- [ ] Shows progression from previous examples
- [ ] Includes helpful comments for complex parts
- [ ] Doesn't over-comment obvious code

### Code Quality
- [ ] Type hints on all functions
- [ ] Docstrings on public functions/classes
- [ ] Meaningful variable names
- [ ] Functions are focused and single-purpose
- [ ] Proper error handling (no bare excepts)
- [ ] No unnecessary code duplication

### Standards
- [ ] Uses uv for package management
- [ ] Uses dotenv for API key loading
- [ ] No frameworks (langchain, etc.)
- [ ] Builds from primitives
- [ ] Follows Black formatting (88 chars)
- [ ] Import order correct (stdlib → third-party → local)

### Security
- [ ] No hardcoded credentials
- [ ] API keys loaded from environment
- [ ] .env.example provided
- [ ] Input validation where needed
- [ ] Error messages don't leak sensitive info

### Documentation
- [ ] Module docstring complete
- [ ] All functions documented
- [ ] README.md updated
- [ ] Example numbers sequential
- [ ] File naming follows conventions

## README.md for Each Chapter

Every code folder needs a README.md:

```markdown
# Chapter XX: [Title] - Code Examples

This folder contains code examples for Chapter XX.

## Files

- `example_01_simple.py` - Basic [concept] implementation
- `example_02_with_history.py` - Adds conversation memory
- `example_03_complete.py` - Production-ready implementation
- `exercise_solution.py` - Solution to the chapter exercise

## Setup

1. **Create environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Add your API key** to `.env`:
   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

3. **Install dependencies** (if using uv):
   ```bash
   uv sync
   ```

## Running Examples

```bash
# Run any example
python example_01_simple.py

# Or with uv
uv run example_01_simple.py
```

## What Each Example Demonstrates

### example_01_simple.py
Shows the most basic implementation of [concept]. This is the minimal
viable example to understand the core idea.

### example_02_with_history.py
Builds on example 01 by adding [feature]. Demonstrates how to [specific
capability].

### example_03_complete.py
Production-ready implementation with error handling, proper structure,
and all features from the chapter.

## Exercise

The exercise asks you to [task description]. The solution is in
`exercise_solution.py` but try implementing it yourself first!

## Key Concepts

- [Concept 1 from the chapter]
- [Concept 2 from the chapter]
- [Concept 3 from the chapter]
```

## Reference Materials

**Check before writing code:**
- `ai_agents/skills/PROJECT_INSTRUCTIONS.md` - Complete coding standards
- `ai_agents/CLAUDE.md` - Technical architecture

**For architecture patterns:**
- `chapter-14-building-the-complete-augmented-llm/code/augmented_llm.py`
- `chapter-33-the-complete-agent-class/code/agent.py`

**Previous chapter code:**
Always check how previous chapters implemented similar patterns to maintain consistency.

---

**Remember**: Code examples are teaching tools. They must be clear, complete, secure, and runnable. Prioritize clarity over cleverness, and always show the progression from simple to complex.
