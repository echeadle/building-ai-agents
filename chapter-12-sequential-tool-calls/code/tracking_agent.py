"""
Agent with comprehensive tool call tracking.

This script demonstrates how to track and log all tool calls
made by an agent. This is essential for debugging, observability,
and analyzing agent behavior.

Chapter 12: Sequential Tool Calls
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the client
client = anthropic.Anthropic()


@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    arguments: dict
    result: str
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "iteration": self.iteration,
            "duration_ms": self.duration_ms
        }


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
    
    def duration_seconds(self) -> float:
        """Get the total duration in seconds."""
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_message": self.user_message,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "final_response": self.final_response,
            "total_iterations": self.total_iterations,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "duration_seconds": self.duration_seconds()
        }
    
    def summary(self) -> str:
        """Get a human-readable summary of the run."""
        lines = [
            f"Agent Run Summary",
            f"-" * 40,
            f"Status: {self.status}",
            f"Duration: {self.duration_seconds():.2f}s",
            f"Iterations: {self.total_iterations}",
            f"Tool calls: {len(self.tool_calls)}",
        ]
        
        if self.tool_calls:
            lines.append(f"\nTool Call Breakdown:")
            tool_counts = {}
            tool_times = {}
            for tc in self.tool_calls:
                tool_counts[tc.tool_name] = tool_counts.get(tc.tool_name, 0) + 1
                tool_times[tc.tool_name] = tool_times.get(tc.tool_name, 0) + tc.duration_ms
            
            for tool, count in sorted(tool_counts.items()):
                avg_time = tool_times[tool] / count
                lines.append(f"  - {tool}: {count} calls, avg {avg_time:.1f}ms")
        
        return "\n".join(lines)


# Tool definitions
TOOLS = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Gets the current weather for a specified city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Gets the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if name == "calculator":
        try:
            expression = arguments.get("expression", "")
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        weather_data = {
            "Tokyo": {"temp": 28, "conditions": "Sunny", "humidity": 65},
            "New York": {"temp": 22, "conditions": "Cloudy", "humidity": 55},
            "London": {"temp": 18, "conditions": "Rainy", "humidity": 80},
            "Paris": {"temp": 24, "conditions": "Clear", "humidity": 45},
        }
        if city in weather_data:
            return json.dumps({"city": city, **weather_data[city]})
        return json.dumps({"city": city, "temp": 20, "conditions": "Unknown"})
    
    elif name == "get_current_time":
        now = datetime.now()
        return json.dumps({
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day": now.strftime("%A")
        })
    
    return f"Error: Unknown tool '{name}'"


def run_agent_with_tracking(
    user_message: str,
    max_iterations: int = 10
) -> AgentRun:
    """
    Run agent with comprehensive tracking.
    
    Args:
        user_message: The user's question or request
        max_iterations: Maximum number of iterations
    
    Returns:
        AgentRun object with complete execution history
    """
    run = AgentRun(user_message=user_message)
    messages = [{"role": "user", "content": user_message}]
    
    system_prompt = """You are a helpful assistant with access to calculator, weather, and time tools.
Use the appropriate tools to answer questions thoroughly."""
    
    for iteration in range(max_iterations):
        run.total_iterations = iteration + 1
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages
            )
        except Exception as e:
            run.complete(f"API error: {str(e)}", status="error")
            return run
        
        # Check if Claude is done
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    run.complete(block.text)
                    return run
            run.complete("No response generated.", status="error")
            return run
        
        # Process tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Time the tool execution
                    start_time = time.time()
                    result = execute_tool(block.name, block.input)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Track the call
                    tool_call = ToolCall(
                        tool_name=block.name,
                        arguments=block.input,
                        result=result,
                        iteration=iteration + 1,
                        duration_ms=duration_ms
                    )
                    run.add_tool_call(tool_call)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    run.complete("Maximum iterations reached.", status="max_iterations")
    return run


def main():
    """Demonstrate the tracking agent."""
    
    print("="*60)
    print("TRACKING AGENT DEMONSTRATION")
    print("="*60)
    
    # Run a complex query
    query = """
    What time is it, and what's the weather in Tokyo and Paris?
    Also calculate 15% of 250.
    """
    
    print(f"\nUser: {query}")
    print("\n" + "-"*60)
    print("Executing agent...")
    print("-"*60)
    
    run = run_agent_with_tracking(query)
    
    # Print the summary
    print("\n" + "="*60)
    print(run.summary())
    print("="*60)
    
    # Print detailed tool call history
    print("\nDetailed Tool Call History:")
    print("-"*40)
    for i, tc in enumerate(run.tool_calls, 1):
        print(f"\n{i}. {tc.tool_name} (iteration {tc.iteration})")
        print(f"   Arguments: {tc.arguments}")
        print(f"   Result: {tc.result}")
        print(f"   Duration: {tc.duration_ms:.2f}ms")
    
    # Print final response
    print("\n" + "="*60)
    print("FINAL RESPONSE:")
    print("="*60)
    print(run.final_response)
    
    # Export to JSON (useful for logging/analysis)
    print("\n" + "="*60)
    print("JSON EXPORT (for logging):")
    print("="*60)
    print(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    main()
