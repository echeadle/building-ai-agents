"""
Complete sequential tool-calling agent.

This is a production-ready implementation of the SequentialAgent class
that combines all the patterns from this chapter:
- The agentic loop
- Multiple tool support
- Loop detection
- Timeout protection
- Comprehensive tracking

Chapter 12: Sequential Tool Calls
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


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
    status: str  # completed, timeout, max_iterations, loop_detected, error
    duration_seconds: float
    
    def __str__(self) -> str:
        return (
            f"AgentResult(status={self.status}, "
            f"iterations={self.iterations}, "
            f"tool_calls={len(self.tool_calls)}, "
            f"duration={self.duration_seconds:.2f}s)"
        )


class SequentialAgent:
    """
    An agent that can make sequential tool calls to answer complex questions.
    
    This agent implements the agentic loop pattern:
    1. Receive user input
    2. Call Claude with available tools
    3. If Claude requests tools, execute them and continue
    4. If Claude provides a final answer, return it
    5. Repeat until done or limits reached
    
    Features:
    - Multiple tool support
    - Loop detection to prevent infinite loops
    - Timeout protection
    - Comprehensive tracking of all tool calls
    - Configurable system prompts
    
    Example:
        ```python
        agent = SequentialAgent(
            tools=my_tools,
            tool_executor=my_executor,
            system_prompt="You are a helpful assistant."
        )
        result = agent.run("What's the weather in Tokyo?")
        print(result.response)
        ```
    """
    
    def __init__(
        self,
        tools: list[dict],
        tool_executor: Callable[[str, dict], str],
        system_prompt: str = "You are a helpful assistant with access to tools.",
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
        timeout_seconds: float = 120.0,
        loop_detection_window: int = 3,
        verbose: bool = False
    ):
        """
        Initialize the sequential agent.
        
        Args:
            tools: List of tool definitions (Anthropic format)
            tool_executor: Function that executes tools: (name, args) -> result
            system_prompt: System prompt for the agent
            model: Claude model to use
            max_iterations: Maximum number of tool-calling iterations
            timeout_seconds: Maximum time for the entire run
            loop_detection_window: Number of calls to check for loops
            verbose: Whether to print debug information
        """
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.tool_executor = tool_executor
        self.system_prompt = system_prompt
        self.model = model
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.loop_detection_window = loop_detection_window
        self.verbose = verbose
    
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
        tool_calls: list[ToolCall] = []
        tool_history: list[tuple] = []  # For loop detection
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Starting agent run")
            print(f"User message: {user_message[:100]}...")
            print(f"{'='*50}")
        
        for iteration in range(self.max_iterations):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                if self.verbose:
                    print(f"Timeout after {elapsed:.1f}s")
                return AgentResult(
                    response=f"Agent timed out after {elapsed:.1f} seconds.",
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="timeout",
                    duration_seconds=elapsed
                )
            
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Call Claude
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=self.tools,
                    messages=messages
                )
            except anthropic.APIError as e:
                return AgentResult(
                    response=f"API error: {str(e)}",
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="error",
                    duration_seconds=time.time() - start_time
                )
            
            if self.verbose:
                print(f"Stop reason: {response.stop_reason}")
            
            # Check if Claude is done
            if response.stop_reason == "end_turn":
                final_text = self._extract_text(response)
                return AgentResult(
                    response=final_text,
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="completed",
                    duration_seconds=time.time() - start_time
                )
            
            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add Claude's response to conversation
                messages.append({"role": "assistant", "content": response.content})
                
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # Track for loop detection
                        call_signature = (
                            block.name,
                            json.dumps(block.input, sort_keys=True)
                        )
                        tool_history.append(call_signature)
                        
                        # Check for loops
                        if self._detect_loop(tool_history):
                            if self.verbose:
                                print("Loop detected!")
                            return AgentResult(
                                response="Agent detected a loop and stopped.",
                                tool_calls=tool_calls,
                                iterations=iteration + 1,
                                status="loop_detected",
                                duration_seconds=time.time() - start_time
                            )
                        
                        # Execute the tool
                        if self.verbose:
                            print(f"Tool: {block.name}")
                            print(f"Args: {block.input}")
                        
                        call_start = time.time()
                        try:
                            result = self.tool_executor(block.name, block.input)
                        except Exception as e:
                            result = f"Tool error: {str(e)}"
                        call_duration = (time.time() - call_start) * 1000
                        
                        if self.verbose:
                            print(f"Result: {result[:100]}...")
                        
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
                
                # Add tool results to conversation
                messages.append({"role": "user", "content": tool_results})
            
            elif response.stop_reason == "max_tokens":
                return AgentResult(
                    response="Response was cut off due to token limits.",
                    tool_calls=tool_calls,
                    iterations=iteration + 1,
                    status="error",
                    duration_seconds=time.time() - start_time
                )
        
        # Max iterations reached
        return AgentResult(
            response="Maximum iterations reached without completing the task.",
            tool_calls=tool_calls,
            iterations=self.max_iterations,
            status="max_iterations",
            duration_seconds=time.time() - start_time
        )
    
    def _extract_text(self, response) -> str:
        """Extract text content from a response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return "No response generated."
    
    def _detect_loop(self, history: list[tuple]) -> bool:
        """Detect if the agent is stuck in a loop."""
        window = self.loop_detection_window
        if len(history) < window * 2:
            return False
        
        recent = history[-window:]
        previous = history[-window*2:-window]
        return recent == previous


# Example tools for demonstration
EXAMPLE_TOOLS = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g., '15 * 7')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_weather",
        "description": "Gets current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., 'Tokyo')"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "get_time",
        "description": "Gets the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def example_tool_executor(name: str, arguments: dict) -> str:
    """Example tool executor for demonstration."""
    if name == "calculator":
        try:
            expression = arguments.get("expression", "")
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters"
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"
    
    elif name == "get_weather":
        city = arguments.get("city", "Unknown")
        weather = {
            "Tokyo": {"temp": 28, "conditions": "Sunny"},
            "New York": {"temp": 22, "conditions": "Cloudy"},
            "London": {"temp": 18, "conditions": "Rainy"},
            "Paris": {"temp": 24, "conditions": "Clear"},
        }
        data = weather.get(city, {"temp": 20, "conditions": "Unknown"})
        return json.dumps({"city": city, **data})
    
    elif name == "get_time":
        now = datetime.now()
        return json.dumps({
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "day": now.strftime("%A")
        })
    
    return f"Error: Unknown tool '{name}'"


def main():
    """Demonstrate the SequentialAgent class."""
    
    print("="*60)
    print("SEQUENTIAL AGENT DEMONSTRATION")
    print("="*60)
    
    # Create the agent
    agent = SequentialAgent(
        tools=EXAMPLE_TOOLS,
        tool_executor=example_tool_executor,
        system_prompt="""You are a helpful assistant with access to:
- calculator: For math calculations
- get_weather: For weather information
- get_time: For current time

Use these tools to answer questions thoroughly. Explain your reasoning.""",
        verbose=True
    )
    
    # Test queries
    queries = [
        "What's 42 * 17?",
        "What's the weather in Tokyo and what time is it?",
        "Compare the weather in London and Paris. Which is warmer, and by how much?",
    ]
    
    for query in queries:
        print("\n" + "="*60)
        print(f"QUERY: {query}")
        print("="*60)
        
        result = agent.run(query)
        
        print(f"\n{result}")
        print(f"\nResponse: {result.response}")
        
        if result.tool_calls:
            print(f"\nTool calls made:")
            for tc in result.tool_calls:
                print(f"  - {tc.tool_name}: {tc.duration_ms:.1f}ms")


if __name__ == "__main__":
    main()
