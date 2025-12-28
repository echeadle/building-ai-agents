---
chapter: 12
title: "Sequential Tool Calls"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 12: Sequential Tool Calls

## Introduction

In the previous chapter, you built an agent with multiple tools—a calculator, a weather service, and a datetime utility. That agent could choose the right tool for a given question, but what happens when a question requires *multiple* tools working together?

Consider this request: "What's the weather in Tokyo, and if the temperature is above 25°C, calculate how much a 20% discount on a ¥5000 umbrella would save me."

This isn't a single-tool question. Your agent needs to:

1. Get the weather in Tokyo (weather tool)
2. Check if the temperature exceeds 25°C (reasoning)
3. If yes, calculate the discount (calculator tool)

This is **sequential tool calling**—the agent makes one tool call, processes the result, decides what to do next, and potentially makes another tool call. This back-and-forth continues until the agent has enough information to answer the question.

Sequential tool calling is where agents start to feel truly intelligent. They're not just matching questions to tools; they're reasoning through multi-step problems and orchestrating their tools to find answers.

In this chapter, you'll build the **agentic loop**—the core mechanism that powers all autonomous AI agents.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand how multi-turn tool use works with the Claude API
- Implement the agentic loop pattern for sequential tool calls
- Design proper termination conditions to prevent infinite loops
- Track and log tool call history for debugging and observability
- Build an agent that chains multiple tool calls to answer complex questions

## Understanding Multi-Turn Tool Use

Let's first understand what happens in a single tool call versus multiple sequential calls.

### Single Tool Call Flow

In Chapter 9, you learned the basic tool use flow:

```
User Question → Claude → Tool Call → Your Code Executes Tool → Result to Claude → Final Answer
```

This works great for simple questions like "What's 15 × 7?" or "What's the weather in Paris?"

### Multi-Turn Tool Call Flow

But for complex questions, the flow looks different:

```
User Question → Claude → Tool Call #1 → Execute → Result → Claude → Tool Call #2 → Execute → Result → Claude → ... → Final Answer
```

Claude receives the result of each tool call and decides whether to:
1. **Make another tool call** — More information is needed
2. **Provide a final answer** — Enough information has been gathered

This decision-making is the heart of agentic behavior.

### The API Mechanics

Here's how multi-turn tool use works with the Claude API:

1. You send a message with tools available
2. Claude responds with a `tool_use` content block
3. You execute the tool and send back a `tool_result`
4. Claude processes the result and either:
   - Returns another `tool_use` (needs more information)
   - Returns a `text` response (ready to answer)
5. Repeat until Claude provides a final text response

The key insight: **you must keep sending messages back to Claude** until it stops requesting tool calls.

## The Agentic Loop

The **agentic loop** is a simple but powerful pattern:

```python
while True:
    response = call_claude(messages)
    
    if response.has_tool_calls():
        results = execute_tools(response.tool_calls)
        messages.append(response)
        messages.append(tool_results)
    else:
        return response.text  # Final answer
```

Let's build this step by step.

### Basic Agentic Loop Implementation

Here's a complete implementation of the agentic loop:

```python
"""
Basic agentic loop implementation.

Chapter 12: Sequential Tool Calls
"""

import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

def run_agent(
    user_message: str,
    tools: list[dict],
    tool_executor: callable,
    system_prompt: str = "You are a helpful assistant.",
    max_iterations: int = 10
) -> str:
    """
    Run an agent that can make sequential tool calls.
    
    Args:
        user_message: The user's question or request
        tools: List of tool definitions
        tool_executor: Function that executes tools and returns results
        system_prompt: System prompt for the agent
        max_iterations: Maximum number of tool call iterations (safety limit)
    
    Returns:
        The agent's final text response
    """
    # Initialize conversation with user message
    messages = [{"role": "user", "content": user_message}]
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        
        # Call Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages
        )
        
        print(f"Stop reason: {response.stop_reason}")
        
        # Check if Claude is done (no more tool calls)
        if response.stop_reason == "end_turn":
            # Extract the final text response
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Add Claude's response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Tool call: {block.name}({block.input})")
                    
                    # Execute the tool
                    result = tool_executor(block.name, block.input)
                    print(f"Tool result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            
            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})
        
        else:
            # Unexpected stop reason
            print(f"Unexpected stop reason: {response.stop_reason}")
            break
    
    return "Maximum iterations reached without a final response."
```

Let's break down the key parts:

### Message Structure for Tool Results

When Claude makes a tool call, you need to:

1. **Add Claude's response** to the messages (including the tool_use blocks)
2. **Add the tool results** as a new user message with `tool_result` blocks

The structure looks like this:

```python
messages = [
    {"role": "user", "content": "What's 15 * 7?"},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_123", "name": "calculator", "input": {"expression": "15 * 7"}}
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_123", "content": "105"}
    ]},
    {"role": "assistant", "content": [
        {"type": "text", "text": "15 × 7 = 105"}
    ]}
]
```

> **Note:** Tool results are sent as `role: "user"` messages. This might seem counterintuitive, but it makes sense—you (the developer) are providing information back to Claude, just like a user would.

### The stop_reason Field

Claude's response includes a `stop_reason` that tells you why it stopped generating:

- **`end_turn`**: Claude is done and has provided a final response
- **`tool_use`**: Claude wants to use one or more tools
- **`max_tokens`**: Response was cut off due to token limit
- **`stop_sequence`**: A custom stop sequence was encountered

For the agentic loop, you primarily care about `end_turn` (we're done) and `tool_use` (keep going).

## Building a Multi-Tool Agent

Let's put this together with our tools from Chapter 11. First, let's define our tools and executor:

```python
"""
Multi-tool agent with sequential tool calling.

Chapter 12: Sequential Tool Calls
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# Tool definitions
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use this for any arithmetic, percentages, or mathematical expressions. Input should be a valid mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '15 * 7', '100 / 4', '2 ** 10')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Gets the current date and time. Use this when you need to know the current time or date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional timezone (e.g., 'UTC', 'US/Eastern'). Defaults to local time."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Gets the current weather for a specified city. Returns temperature in Celsius, conditions, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name (e.g., 'Tokyo', 'New York', 'London')"
                }
            },
            "required": ["city"]
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """
    Execute a tool and return the result.
    
    Args:
        name: The tool name
        arguments: The tool arguments
    
    Returns:
        The tool result as a string
    """
    if name == "calculator":
        try:
            expression = arguments.get("expression", "")
            # Safety: only allow safe mathematical operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif name == "get_current_time":
        timezone = arguments.get("timezone", "local")
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S") + f" ({timezone})"
    
    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        # Simulated weather data for demonstration
        weather_data = {
            "Tokyo": {"temp": 28, "conditions": "Sunny", "humidity": 65},
            "New York": {"temp": 22, "conditions": "Partly Cloudy", "humidity": 55},
            "London": {"temp": 18, "conditions": "Rainy", "humidity": 80},
            "Paris": {"temp": 24, "conditions": "Clear", "humidity": 45},
            "Sydney": {"temp": 15, "conditions": "Windy", "humidity": 60},
        }
        if city in weather_data:
            data = weather_data[city]
            return json.dumps({
                "city": city,
                "temperature_celsius": data["temp"],
                "conditions": data["conditions"],
                "humidity_percent": data["humidity"]
            })
        else:
            return json.dumps({
                "city": city,
                "temperature_celsius": 20,
                "conditions": "Unknown",
                "humidity_percent": 50,
                "note": "Data not available, using defaults"
            })
    
    else:
        return f"Error: Unknown tool '{name}'"
```

### Running the Agent

Now let's create a complete runnable example:

```python
def run_agent(
    user_message: str,
    max_iterations: int = 10
) -> str:
    """Run the multi-tool agent."""
    
    messages = [{"role": "user", "content": user_message}]
    
    system_prompt = """You are a helpful assistant with access to tools.
    
When answering questions:
1. Think about what information you need
2. Use the appropriate tools to gather that information
3. Combine the results to give a complete answer

Always explain your reasoning and show your work."""

    for iteration in range(max_iterations):
        print(f"\n{'='*50}")
        print(f"Iteration {iteration + 1}")
        print('='*50)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        print(f"Stop reason: {response.stop_reason}")
        
        # Check if we're done
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "text":
                    print(f"Claude's thinking: {block.text[:200]}...")
                elif block.type == "tool_use":
                    print(f"\nTool: {block.name}")
                    print(f"Input: {json.dumps(block.input, indent=2)}")
                    
                    result = execute_tool(block.name, block.input)
                    print(f"Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Maximum iterations reached."


if __name__ == "__main__":
    # Test with a question that requires multiple tools
    question = """
    What's the weather like in Tokyo and Paris right now? 
    If Tokyo is warmer than Paris, calculate how much warmer it is 
    and what percentage difference that represents.
    """
    
    print("User question:", question)
    print("\n" + "="*60)
    print("AGENT EXECUTION")
    print("="*60)
    
    answer = run_agent(question)
    
    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(answer)
```

When you run this, you'll see the agent:

1. Call `get_weather` for Tokyo
2. Call `get_weather` for Paris  
3. Compare the temperatures
4. Call `calculator` to find the difference
5. Call `calculator` to compute the percentage
6. Synthesize everything into a final answer

## Preventing Infinite Loops

One of the most important aspects of the agentic loop is knowing when to stop. Without proper safeguards, an agent could loop forever.

### The max_iterations Guard

The simplest protection is a hard limit on iterations:

```python
max_iterations = 10  # Reasonable default for most tasks

for iteration in range(max_iterations):
    # ... agent loop ...
    pass
else:
    # Loop completed without breaking
    return "Maximum iterations reached without completing the task."
```

> **Warning:** Always set a `max_iterations` limit. An agent without this safeguard could run indefinitely, consuming API credits and never returning a response.

### Detecting Stuck States

Sometimes an agent gets stuck in a pattern—repeatedly calling the same tool with the same arguments. Here's how to detect and handle this:

```python
def detect_loop(tool_history: list[tuple[str, dict]], window: int = 3) -> bool:
    """
    Detect if the agent is stuck in a loop.
    
    Args:
        tool_history: List of (tool_name, arguments) tuples
        window: Number of recent calls to check for repetition
    
    Returns:
        True if a loop is detected
    """
    if len(tool_history) < window * 2:
        return False
    
    recent = tool_history[-window:]
    previous = tool_history[-window*2:-window]
    
    # Check if the last N calls are identical to the N before that
    return recent == previous


def run_agent_with_loop_detection(user_message: str, max_iterations: int = 10) -> str:
    """Run agent with loop detection."""
    
    messages = [{"role": "user", "content": user_message}]
    tool_history = []
    
    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "No response generated."
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Track this call
                    tool_history.append((block.name, block.input))
                    
                    # Check for loops
                    if detect_loop(tool_history):
                        return "Agent appears to be stuck in a loop. Stopping."
                    
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    return "Maximum iterations reached."
```

### Timeout Protection

For production systems, you also want a time-based limit:

```python
import time

def run_agent_with_timeout(
    user_message: str,
    max_iterations: int = 10,
    timeout_seconds: float = 60.0
) -> str:
    """Run agent with timeout protection."""
    
    start_time = time.time()
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            return f"Agent timed out after {elapsed:.1f} seconds."
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        # ... rest of the loop ...
```

## Tracking Tool Call History

For debugging and observability, you need to track what your agent is doing. Here's a pattern for comprehensive history tracking:

```python
from dataclasses import dataclass, field
from typing import Any
from datetime import datetime


@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    arguments: dict
    result: str
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    duration_ms: float = 0.0


@dataclass
class AgentRun:
    """Complete record of an agent run."""
    user_message: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_response: str = ""
    total_iterations: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime = None
    status: str = "running"  # running, completed, error, timeout, loop_detected
    
    def add_tool_call(self, call: ToolCall) -> None:
        """Add a tool call to the history."""
        self.tool_calls.append(call)
    
    def complete(self, response: str, status: str = "completed") -> None:
        """Mark the run as complete."""
        self.final_response = response
        self.status = status
        self.completed_at = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "user_message": self.user_message,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result": tc.result,
                    "iteration": tc.iteration,
                    "duration_ms": tc.duration_ms
                }
                for tc in self.tool_calls
            ],
            "final_response": self.final_response,
            "total_iterations": self.total_iterations,
            "status": self.status,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            )
        }


def run_agent_with_tracking(user_message: str, max_iterations: int = 10) -> AgentRun:
    """Run agent with comprehensive tracking."""
    
    run = AgentRun(user_message=user_message)
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        run.total_iterations = iteration + 1
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    run.complete(block.text)
                    return run
            run.complete("No response generated.", status="error")
            return run
        
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    start = time.time()
                    result = execute_tool(block.name, block.input)
                    duration = (time.time() - start) * 1000
                    
                    # Track the call
                    run.add_tool_call(ToolCall(
                        tool_name=block.name,
                        arguments=block.input,
                        result=result,
                        iteration=iteration + 1,
                        duration_ms=duration
                    ))
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    run.complete("Maximum iterations reached.", status="timeout")
    return run
```

Using this tracking, you can:

- Log every tool call for debugging
- Measure performance of individual tools
- Identify patterns in agent behavior
- Store run history for analysis

```python
# Example usage with tracking
run = run_agent_with_tracking(
    "What's 2+2, and what time is it in Tokyo right now?"
)

print(f"Status: {run.status}")
print(f"Iterations: {run.total_iterations}")
print(f"Tool calls: {len(run.tool_calls)}")
for tc in run.tool_calls:
    print(f"  - {tc.tool_name}: {tc.duration_ms:.1f}ms")
print(f"\nFinal answer: {run.final_response}")
```

## Handling Multiple Simultaneous Tool Calls

Claude can request multiple tools in a single response. Your loop needs to handle this:

```python
for block in response.content:
    if block.type == "tool_use":
        # This might execute multiple times in one iteration!
        result = execute_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result
        })
```

When Claude requests multiple tools at once, it's because they're independent—they don't depend on each other's results. For example:

- "What's the weather in Tokyo AND Paris?" → Two parallel weather calls
- "What's 2+2 and what time is it?" → Calculator and time calls together

> **💡 Tip:** In Chapter 21, you'll learn how to execute these parallel tool calls concurrently using `asyncio` for better performance. For now, sequential execution works fine.

## When to Stop the Loop

The agentic loop should stop when:

1. **Claude returns `stop_reason: "end_turn"`** — The primary signal that Claude has finished
2. **Maximum iterations reached** — Safety limit to prevent runaway agents
3. **Timeout exceeded** — Time-based safety limit
4. **Loop detected** — Agent is stuck repeating the same calls
5. **Critical error** — An unrecoverable error occurred

Here's a comprehensive stopping logic:

```python
class StopReason:
    """Reasons for stopping the agent loop."""
    COMPLETED = "completed"           # Claude finished naturally
    MAX_ITERATIONS = "max_iterations" # Hit iteration limit
    TIMEOUT = "timeout"               # Exceeded time limit
    LOOP_DETECTED = "loop_detected"   # Stuck in a loop
    ERROR = "error"                   # Unrecoverable error
    USER_CANCELLED = "cancelled"      # User requested stop
```

## The Complete Sequential Agent

Let's put everything together into a production-ready sequential agent:

```python
"""
Complete sequential tool-calling agent.

Chapter 12: Sequential Tool Calls
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Any
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    arguments: dict
    result: str
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Result from running the agent."""
    response: str
    tool_calls: list[ToolCall]
    iterations: int
    status: str
    duration_seconds: float


class SequentialAgent:
    """
    An agent that can make sequential tool calls to answer complex questions.
    
    This agent implements the agentic loop pattern:
    1. Receive user input
    2. Call Claude with available tools
    3. If Claude requests tools, execute them and continue
    4. If Claude provides a final answer, return it
    5. Repeat until done or limits reached
    """
    
    def __init__(
        self,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        system_prompt: str = "You are a helpful assistant with access to tools.",
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
        timeout_seconds: float = 120.0
    ):
        """
        Initialize the sequential agent.
        
        Args:
            tools: List of tool definitions
            tool_executor: Function that executes tools (name, args) -> result
            system_prompt: System prompt for the agent
            model: Claude model to use
            max_iterations: Maximum number of tool-calling iterations
            timeout_seconds: Maximum time for the entire run
        """
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.model = model
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
    
    def run(self, user_message: str) -> AgentResult:
        """
        Run the agent on a user message.
        
        Args:
            user_message: The user's question or request
        
        Returns:
            AgentResult with the response and execution details
        """
        start_time = time.time()
        messages = [{"role": "user", "content": user_message}]
        tool_calls = []
        tool_history = []  # For loop detection
        
        for iteration in range(self.max_iterations):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                return AgentResult(
                    response=f"Agent timed out after {elapsed:.1f} seconds.",
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="timeout",
                    duration_seconds=elapsed
                )
            
            # Call Claude
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=self.tools,
                    messages=messages
                )
            except Exception as e:
                return AgentResult(
                    response=f"API error: {str(e)}",
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="error",
                    duration_seconds=time.time() - start_time
                )
            
            # Check if Claude is done
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text = block.text
                        break
                
                return AgentResult(
                    response=final_text or "No response generated.",
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="completed",
                    duration_seconds=time.time() - start_time
                )
            
            # Process tool calls
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # Track for loop detection
                        call_signature = (block.name, json.dumps(block.input, sort_keys=True))
                        tool_history.append(call_signature)
                        
                        # Check for loops
                        if self._detect_loop(tool_history):
                            return AgentResult(
                                response="Agent detected a loop and stopped.",
                                tool_calls=tool_calls,
                                iterations=iteration + 1,
                                status="loop_detected",
                                duration_seconds=time.time() - start_time
                            )
                        
                        # Execute the tool
                        call_start = time.time()
                        try:
                            result = self.tool_executor(block.name, block.input)
                        except Exception as e:
                            result = f"Tool error: {str(e)}"
                        call_duration = (time.time() - call_start) * 1000
                        
                        # Record the call
                        tool_calls.append(ToolCall(
                            tool_name=block.name,
                            arguments=block.input,
                            result=result,
                            iteration=iteration + 1,
                            duration_ms=call_duration
                        ))
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                messages.append({"role": "user", "content": tool_results})
        
        # Max iterations reached
        return AgentResult(
            response="Maximum iterations reached without completing the task.",
            tool_calls=tool_calls,
            iterations=self.max_iterations,
            status="max_iterations",
            duration_seconds=time.time() - start_time
        )
    
    def _detect_loop(self, history: list, window: int = 3) -> bool:
        """Detect if the agent is stuck in a loop."""
        if len(history) < window * 2:
            return False
        
        recent = history[-window:]
        previous = history[-window*2:-window]
        return recent == previous
```

## Common Pitfalls

### 1. Forgetting to Append Claude's Response

A common mistake is only appending the tool results, not Claude's original response:

```python
# WRONG - Missing Claude's response
tool_results = [...]
messages.append({"role": "user", "content": tool_results})

# CORRECT - Include both
messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": tool_results})
```

Without Claude's original response in the conversation, the context is broken and Claude can't continue coherently.

### 2. No Safety Limits

Never run an agentic loop without limits:

```python
# DANGEROUS - Can run forever
while True:
    response = call_claude(messages)
    if is_done(response):
        break
    # ...

# SAFE - Has limits
for iteration in range(max_iterations):
    if time.time() - start > timeout:
        break
    # ...
```

### 3. Ignoring Tool Errors

Tools fail. Handle it gracefully:

```python
# WRONG - Crash on error
result = execute_tool(name, args)

# CORRECT - Handle errors
try:
    result = execute_tool(name, args)
except Exception as e:
    result = f"Error: {str(e)}"
    # Don't crash - Claude can often work around tool failures
```

## Practical Exercise

**Task:** Build a trip planning agent that uses sequential tool calls to answer travel-related questions.

**Requirements:**
1. Create a `get_flight_price` tool that returns mock prices between cities
2. Create a `get_hotel_price` tool that returns mock hotel prices for a city
3. Create a `currency_converter` tool that converts between currencies
4. Implement the agentic loop to handle questions like:
   - "What would a 3-night trip from New York to Paris cost in total (flights + hotel), and how much is that in Japanese Yen?"

**Hints:**
- The agent will need to call multiple tools and combine their results
- Use the loop detection pattern to prevent infinite loops
- Track all tool calls for debugging

**Solution:** See `code/exercise_trip_planner.py`

## Key Takeaways

- **The agentic loop** is the core pattern for sequential tool calls: call → execute → respond → repeat
- **Stop conditions** are critical: always use max_iterations, timeout, and loop detection
- **Tool results** must be sent back as user messages with `tool_result` content blocks
- **Track everything** for debugging: tool names, arguments, results, timing
- **Handle errors gracefully**: tools fail, networks timeout, APIs rate limit—plan for it
- **Claude decides when it's done** by returning `stop_reason: "end_turn"` instead of `tool_use`

## What's Next

In Chapter 13, you'll learn about **Structured Outputs and Response Parsing**—how to get Claude to return predictable, parseable responses like JSON objects. This is essential for building agents that integrate with other systems, where you need reliable data formats rather than free-form text.

The combination of sequential tool calls (this chapter) and structured outputs (next chapter) will give you powerful building blocks for the complete Augmented LLM class we'll build in Chapter 14.
