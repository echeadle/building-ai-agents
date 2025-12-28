---
chapter: 27
title: "The Agentic Loop"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 27: The Agentic Loop

## Introduction

In Chapter 26, we crossed a significant threshold: from workflows where *we* orchestrate the steps to agents where the *LLM* decides what happens next. But how does an agent actually make those decisions? How does it know when to use a tool, when to think harder, and when to stop?

The answer lies in a deceptively simple pattern called the **agentic loop**. It's the heartbeat of every autonomous agent—a cycle of perceiving, thinking, and acting that repeats until the task is complete. If workflows are like following a recipe, the agentic loop is like being dropped in a kitchen and figuring out what to cook based on what's in the fridge.

In this chapter, we'll implement the core execution cycle that powers all agents. You'll see that the loop itself is remarkably simple—just a few lines of code. The complexity, as we'll discover, lives in the details: knowing when to stop, handling edge cases, and keeping the agent on track.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement the basic perceive-think-act loop that powers all agents
- Understand how each phase of the loop contributes to agent behavior
- Define clear termination conditions to prevent infinite loops
- Build a minimal but functional agentic loop from scratch
- Identify where complexity emerges in real agent implementations

## The Perceive-Think-Act Cycle

Every agent, from simple chatbots to complex autonomous systems, operates on the same fundamental cycle:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│    │          │    │          │    │          │    │
│ ──▶│ PERCEIVE │───▶│  THINK   │───▶│   ACT    │────┤
│    │          │    │          │    │          │    │
│    └──────────┘    └──────────┘    └──────────┘    │
│         ▲                                │         │
│         │                                │         │
│         └────────────────────────────────┘         │
│                                                     │
│                   REPEAT UNTIL DONE                │
└─────────────────────────────────────────────────────┘
```

Let's understand each phase:

### Perceive: Gathering Information

In the **perceive** phase, the agent gathers all available information about its current situation. This includes:

- The original user request
- The conversation history so far
- Results from previous tool calls
- Any new information that has arrived
- The current state of the task

For an LLM-based agent, "perceiving" means assembling the messages array that will be sent to the API. The agent doesn't literally see the world—it sees text representations of the world.

### Think: Reasoning and Planning

In the **think** phase, the LLM processes all perceived information and decides what to do next. This is where the magic happens:

- The LLM considers the user's goal
- It evaluates what has been accomplished so far
- It determines what actions (if any) are needed
- It formulates a response or selects a tool to use

The thinking happens inside the LLM during the API call. We don't control it directly, but we influence it through our system prompts, tool definitions, and the information we provide.

### Act: Taking Action

In the **act** phase, the agent executes whatever the LLM decided to do:

- If the LLM requested a tool call, we execute that tool
- If the LLM provided a final answer, we deliver it to the user
- If the LLM needs more information, we might prompt for it

After acting, the cycle repeats. The results of the action become new input for the next perception phase.

## The Simplest Agentic Loop

Let's implement the most basic version of this loop. We'll build up from here:

```python
"""
The simplest possible agentic loop.

Chapter 27: The Agentic Loop
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

def simple_agent_loop(user_message: str, tools: list, max_iterations: int = 10) -> str:
    """
    The simplest agentic loop: perceive, think, act, repeat.
    
    Args:
        user_message: The user's initial request
        tools: List of tool definitions the agent can use
        max_iterations: Safety limit to prevent infinite loops
        
    Returns:
        The agent's final response
    """
    # Initialize conversation with user's message
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        # THINK: Send everything to the LLM and let it decide
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        # Check if we're done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract and return the final text response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        
        # ACT: Process any tool calls
        if response.stop_reason == "tool_use":
            # Add assistant's response (with tool calls) to history
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Execute the tool (you'd implement this)
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # PERCEIVE: Add tool results for next iteration
            messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached without completing the task."
```

That's it. The entire agentic loop in about 40 lines of code. Let's break down what's happening:

1. **Initialize**: We start with the user's message in our conversation history
2. **Think**: We send everything to Claude and get a response
3. **Check termination**: If `stop_reason` is `"end_turn"`, Claude is done—return the answer
4. **Act**: If `stop_reason` is `"tool_use"`, execute the requested tools
5. **Perceive**: Add the tool results to the conversation and loop back

## Building a Complete Example

Let's build a working agent with actual tools to see the loop in action. We'll create a simple agent that can do math and tell time:

```python
"""
A complete agentic loop with working tools.

Chapter 27: The Agentic Loop
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# Define our tools
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations. Use this for any math calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g., '2 + 2' or '10 * 5'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Gets the current date and time. Use this when the user asks about the current time or date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (default: local time)"
                }
            },
            "required": []
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool and return its result as a string.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Dictionary of input parameters
        
    Returns:
        String result of the tool execution
    """
    if tool_name == "calculator":
        try:
            # Safety: only allow basic math operations
            expression = tool_input["expression"]
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)  # Safe because we validated input
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif tool_name == "get_current_time":
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    else:
        return f"Error: Unknown tool '{tool_name}'"


def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """
    Run the agentic loop with the user's message.
    
    Args:
        user_message: What the user wants to accomplish
        max_iterations: Safety limit for the loop
        
    Returns:
        The agent's final response
    """
    messages = [{"role": "user", "content": user_message}]
    
    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print(f"{'='*60}")
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # THINK: Let Claude process and decide
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        print(f"Stop reason: {response.stop_reason}")
        
        # Check for completion
        if response.stop_reason == "end_turn":
            # Extract final response
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent: {block.text}")
                    return block.text
            return ""
        
        # ACT: Handle tool calls
        if response.stop_reason == "tool_use":
            # Add assistant message to history
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool use
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool: {block.name}")
                    print(f"  Input: {block.input}")
                    
                    # Execute the tool
                    result = execute_tool(block.name, block.input)
                    print(f"  Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # PERCEIVE: Add results for next iteration
            messages.append({"role": "user", "content": tool_results})
    
    return "Max iterations reached."


if __name__ == "__main__":
    # Test with a simple calculation
    run_agent("What is 25 * 4 + 10?")
    
    # Test with time
    run_agent("What time is it right now?")
    
    # Test with multiple steps
    run_agent("What's 100 divided by 4, and also what time is it?")
```

When you run this, you'll see the loop in action:

```
============================================================
User: What is 25 * 4 + 10?
============================================================

--- Iteration 1 ---
Stop reason: tool_use
  Tool: calculator
  Input: {'expression': '25 * 4 + 10'}
  Result: 110

--- Iteration 2 ---
Stop reason: end_turn

Agent: 25 × 4 + 10 = 110
```

Notice how the agent:
1. Received the question
2. Decided to use the calculator tool
3. Got the result
4. Formulated a response
5. Stopped because the task was complete

## Termination Conditions

One of the most critical aspects of the agentic loop is knowing when to stop. An agent that never stops is worse than useless—it burns through API costs and delivers nothing.

### The Primary Termination: Task Completion

The main way an agent should stop is when the task is complete. In Claude's API, this is signaled by `stop_reason == "end_turn"`. This means Claude has decided it has nothing more to do and is providing a final response.

### Safety Terminations

We also need safety mechanisms to catch runaway agents:

```python
"""
Agentic loop with multiple termination conditions.

Chapter 27: The Agentic Loop
"""

import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


class AgentTerminated(Exception):
    """Raised when the agent terminates for any reason."""
    pass


def run_agent_with_termination(
    user_message: str,
    tools: list,
    execute_tool_fn,
    max_iterations: int = 10,
    max_tool_calls: int = 20,
    timeout_seconds: float = 120.0
) -> str:
    """
    Run an agentic loop with comprehensive termination conditions.
    
    Args:
        user_message: The user's request
        tools: Available tool definitions
        execute_tool_fn: Function to execute tools
        max_iterations: Maximum loop iterations
        max_tool_calls: Maximum total tool calls allowed
        timeout_seconds: Maximum time before forced termination
        
    Returns:
        The agent's final response
        
    Raises:
        AgentTerminated: If safety limits are hit
    """
    messages = [{"role": "user", "content": user_message}]
    
    start_time = time.time()
    total_tool_calls = 0
    
    for iteration in range(max_iterations):
        # CHECK: Time limit
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise AgentTerminated(
                f"Timeout: Agent ran for {elapsed:.1f}s (limit: {timeout_seconds}s)"
            )
        
        # CHECK: Tool call limit
        if total_tool_calls >= max_tool_calls:
            raise AgentTerminated(
                f"Tool limit: Agent made {total_tool_calls} tool calls (limit: {max_tool_calls})"
            )
        
        # THINK
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        # CHECK: Normal completion
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        
        # CHECK: Unexpected stop reason
        if response.stop_reason not in ("tool_use", "end_turn"):
            raise AgentTerminated(
                f"Unexpected stop reason: {response.stop_reason}"
            )
        
        # ACT: Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                total_tool_calls += 1
                result = execute_tool_fn(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        # PERCEIVE
        messages.append({"role": "user", "content": tool_results})
    
    # CHECK: Iteration limit
    raise AgentTerminated(
        f"Iteration limit: Agent ran for {max_iterations} iterations"
    )
```

### What Each Limit Protects Against

| Limit | Protects Against |
|-------|------------------|
| `max_iterations` | Infinite reasoning loops where the agent keeps thinking but never finishes |
| `max_tool_calls` | Runaway tool usage (e.g., agent calling search 100 times) |
| `timeout_seconds` | Slow tools or accumulated delays causing indefinite hangs |

> **Warning:** Always include at least `max_iterations`. Without it, a confused agent could loop forever, consuming API credits until you notice.

## The Anatomy of a Loop Iteration

Let's trace through exactly what happens in one iteration:

```python
"""
Detailed breakdown of a single loop iteration.

Chapter 27: The Agentic Loop
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


def detailed_iteration(messages: list, tools: list) -> tuple[list, bool]:
    """
    Perform one detailed iteration of the agentic loop.
    
    Args:
        messages: Current conversation history
        tools: Available tools
        
    Returns:
        Tuple of (updated_messages, is_complete)
    """
    print("\n" + "="*60)
    print("PHASE 1: PERCEIVE")
    print("="*60)
    print(f"Messages in context: {len(messages)}")
    print(f"Last message role: {messages[-1]['role']}")
    
    # Show what the agent "sees"
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:100] + "..." if len(content) > 100 else content
        else:
            preview = f"[{len(content)} content blocks]"
        print(f"  [{i}] {role}: {preview}")
    
    print("\n" + "="*60)
    print("PHASE 2: THINK")
    print("="*60)
    print("Sending to Claude for processing...")
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )
    
    print(f"Response received!")
    print(f"  Stop reason: {response.stop_reason}")
    print(f"  Content blocks: {len(response.content)}")
    print(f"  Input tokens: {response.usage.input_tokens}")
    print(f"  Output tokens: {response.usage.output_tokens}")
    
    # Analyze what Claude decided
    has_text = False
    has_tool_use = False
    tool_calls = []
    
    for block in response.content:
        if hasattr(block, "text"):
            has_text = True
            print(f"  Text response: {block.text[:100]}...")
        elif block.type == "tool_use":
            has_tool_use = True
            tool_calls.append(block)
            print(f"  Tool call: {block.name}({block.input})")
    
    print("\n" + "="*60)
    print("PHASE 3: ACT")
    print("="*60)
    
    # Check if we're done
    if response.stop_reason == "end_turn":
        print("Decision: COMPLETE - Agent finished its task")
        return messages, True
    
    if response.stop_reason == "tool_use":
        print(f"Decision: CONTINUE - Agent wants to use {len(tool_calls)} tool(s)")
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})
        
        # Execute tools (placeholder - you'd implement real execution)
        tool_results = []
        for tool_call in tool_calls:
            print(f"  Executing: {tool_call.name}")
            # Placeholder result
            result = f"[Result of {tool_call.name}]"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            })
            print(f"  Got result: {result}")
        
        # Add tool results to history
        messages.append({"role": "user", "content": tool_results})
        
        return messages, False
    
    print(f"Decision: UNEXPECTED - stop_reason was {response.stop_reason}")
    return messages, True
```

## Why the Loop is "Simple but Complex"

The code for the agentic loop is short. A working implementation is maybe 50 lines. So why does this chapter's key takeaway say "the complexity is in the details"?

Because the loop itself isn't where agents fail. They fail in:

**1. Tool Design**: If tools don't do what Claude expects, the agent spins its wheels:

```python
# Bad tool description - Claude won't know when to use it
{
    "name": "process_data",
    "description": "Processes data",  # Too vague!
    ...
}

# Good tool description
{
    "name": "calculate_average",
    "description": "Calculates the arithmetic mean of a list of numbers. Use this when you need to find the average of multiple values.",
    ...
}
```

**2. Error Handling**: Tools fail. Networks time out. APIs return errors. Each failure mode needs handling:

```python
def execute_tool_safely(name: str, input: dict) -> str:
    """Execute a tool with comprehensive error handling."""
    try:
        result = execute_tool(name, input)
        return result
    except TimeoutError:
        return f"Error: {name} timed out. Try again or use a different approach."
    except ValueError as e:
        return f"Error: Invalid input to {name}: {e}"
    except Exception as e:
        return f"Error: {name} failed unexpectedly: {e}"
```

**3. Context Management**: Long conversations exhaust the context window:

```python
def manage_context(messages: list, max_tokens: int = 100000) -> list:
    """Ensure messages fit within context limits."""
    # This is a simplified version - real implementation is complex
    total_tokens = estimate_tokens(messages)
    
    while total_tokens > max_tokens and len(messages) > 2:
        # Remove oldest messages (keep system prompt and latest)
        messages.pop(1)
        total_tokens = estimate_tokens(messages)
    
    return messages
```

**4. Stuck Detection**: Sometimes agents get stuck in unproductive patterns:

```python
def detect_stuck_pattern(messages: list, last_n: int = 6) -> bool:
    """Detect if the agent is repeating itself."""
    if len(messages) < last_n:
        return False
    
    recent_tool_calls = []
    for msg in messages[-last_n:]:
        if msg["role"] == "assistant":
            for block in msg.get("content", []):
                if hasattr(block, "type") and block.type == "tool_use":
                    recent_tool_calls.append(f"{block.name}:{block.input}")
    
    # If same tool call appears multiple times, agent might be stuck
    if len(recent_tool_calls) != len(set(recent_tool_calls)):
        return True
    
    return False
```

These complexities—and many more—are why Part 4 has eight chapters, not just this one.

## Common Pitfalls

### 1. Forgetting to Add Assistant Messages

A subtle but devastating bug:

```python
# WRONG - Loses the assistant's message
if response.stop_reason == "tool_use":
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({...})
    # Missing: messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})

# RIGHT - Preserves conversation flow
if response.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": response.content})  # Don't forget!
    tool_results = []
    # ... rest of handling
```

Without the assistant message, Claude loses track of what it asked for, leading to confusion and repeated tool calls.

### 2. Not Handling Empty Responses

Sometimes Claude responds with no text content:

```python
# WRONG - Crashes on empty response
if response.stop_reason == "end_turn":
    return response.content[0].text  # May crash!

# RIGHT - Handle gracefully
if response.stop_reason == "end_turn":
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""  # Graceful fallback
```

### 3. Infinite Loops from Tool Errors

If a tool always returns an error, Claude might keep trying:

```python
# The agent will loop forever trying to fix unfixable errors
def bad_tool(input):
    return "Error: Service unavailable"

# Better: Track consecutive errors
def run_agent_with_error_tracking(messages, tools, execute_fn, max_consecutive_errors=3):
    consecutive_errors = 0
    
    for iteration in range(max_iterations):
        # ... normal loop ...
        
        for tool_call in tool_calls:
            result = execute_fn(tool_call.name, tool_call.input)
            
            if result.startswith("Error:"):
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    return "I'm having trouble completing this task due to repeated errors."
            else:
                consecutive_errors = 0  # Reset on success
```

## Practical Exercise

**Task:** Build a multi-tool agent that can answer questions about a fictional inventory system.

**Requirements:**

1. Implement three tools:
   - `check_stock(item_name)`: Returns the quantity in stock (make up numbers)
   - `get_price(item_name)`: Returns the price per unit
   - `calculate_total(quantity, unit_price)`: Returns the total cost

2. The agent should be able to answer questions like:
   - "How many widgets do we have?"
   - "What's the total value of our gadget inventory?"
   - "Do we have enough gizmos to fulfill an order of 50?"

3. Include proper termination conditions (max iterations, max tool calls)

4. Log each iteration showing the perceive-think-act phases

**Hints:**
- Start by defining your tool schemas carefully
- The LLM will chain tool calls automatically when needed
- "Total value of inventory" requires checking stock AND price, then calculating

**Solution:** See `code/exercise.py`

## Key Takeaways

- **The agentic loop is perceive → think → act → repeat**: This cycle powers all autonomous agents, from simple to complex.

- **Termination is critical**: Always include max iterations at minimum. Consider time limits and tool call limits for production.

- **The loop code is simple; the complexity lives elsewhere**: Tool design, error handling, context management, and stuck detection are where real challenges emerge.

- **Messages are everything**: The conversation history is your agent's entire world. Handle it carefully, especially when adding assistant and tool result messages.

- **Start minimal, add complexity as needed**: The 40-line version works. Add sophistication only when you encounter real problems.

## What's Next

We've built the engine, but our agent has amnesia—it forgets everything between runs. In Chapter 28, we'll implement **State Management** so our agents can remember what they're doing, maintain context across interactions, and pick up where they left off.
