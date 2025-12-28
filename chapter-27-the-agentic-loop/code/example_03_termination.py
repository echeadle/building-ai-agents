"""
Agentic loop with comprehensive termination conditions.

This example demonstrates how to build safety limits into your agents
including max iterations, max tool calls, and timeouts.

Chapter 27: The Agentic Loop
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


class AgentTerminated(Exception):
    """Exception raised when the agent terminates due to safety limits."""
    
    def __init__(self, reason: str, details: dict = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(reason)


class AgentStats:
    """Track statistics about agent execution."""
    
    def __init__(self):
        self.iterations = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.start_time = time.time()
        self.tool_call_history = []
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    def record_tool_call(self, tool_name: str, tool_input: dict) -> None:
        """Record a tool call for analysis."""
        self.tool_calls += 1
        self.tool_call_history.append({
            "name": tool_name,
            "input": tool_input,
            "timestamp": time.time()
        })
    
    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# Define tools for demonstration
TOOLS = [
    {
        "name": "slow_operation",
        "description": "A deliberately slow operation for testing timeouts. Simulates a slow API call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "number",
                    "description": "How long to wait (max 5 seconds)"
                }
            },
            "required": ["delay_seconds"]
        }
    },
    {
        "name": "counter",
        "description": "Increments a counter and returns the new value. Use this to count things.",
        "input_schema": {
            "type": "object",
            "properties": {
                "increment": {
                    "type": "integer",
                    "description": "Amount to increment (default: 1)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_time",
        "description": "Returns the current time.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# Global counter for demonstration
_counter = 0


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result."""
    global _counter
    
    if tool_name == "slow_operation":
        delay = min(tool_input.get("delay_seconds", 1), 5)  # Cap at 5 seconds
        time.sleep(delay)
        return f"Operation completed after {delay} seconds"
    
    elif tool_name == "counter":
        increment = tool_input.get("increment", 1)
        _counter += increment
        return f"Counter is now: {_counter}"
    
    elif tool_name == "get_time":
        return datetime.now().strftime("%H:%M:%S")
    
    return f"Unknown tool: {tool_name}"


def detect_stuck_pattern(stats: AgentStats, window: int = 4) -> bool:
    """
    Detect if the agent is stuck in a repetitive pattern.
    
    Args:
        stats: The agent stats object
        window: Number of recent calls to check
        
    Returns:
        True if a stuck pattern is detected
    """
    if len(stats.tool_call_history) < window:
        return False
    
    recent = stats.tool_call_history[-window:]
    
    # Check for exact repetition (same tool with same input)
    signatures = [f"{call['name']}:{call['input']}" for call in recent]
    if len(set(signatures)) == 1:
        return True
    
    # Check for alternating pattern (A, B, A, B)
    if window >= 4:
        if (signatures[0] == signatures[2] and 
            signatures[1] == signatures[3] and 
            signatures[0] != signatures[1]):
            return True
    
    return False


def run_agent_with_limits(
    user_message: str,
    max_iterations: int = 10,
    max_tool_calls: int = 15,
    timeout_seconds: float = 60.0,
    detect_stuck: bool = True
) -> tuple[str, AgentStats]:
    """
    Run an agentic loop with comprehensive termination conditions.
    
    Args:
        user_message: The user's request
        max_iterations: Maximum number of loop iterations
        max_tool_calls: Maximum total tool calls allowed
        timeout_seconds: Maximum runtime before termination
        detect_stuck: Whether to detect and stop stuck patterns
        
    Returns:
        Tuple of (response, stats)
        
    Raises:
        AgentTerminated: If any safety limit is hit
    """
    global _counter
    _counter = 0  # Reset counter for each run
    
    messages = [{"role": "user", "content": user_message}]
    stats = AgentStats()
    
    print(f"\n{'═' * 60}")
    print(f"Starting agent with limits:")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Max tool calls: {max_tool_calls}")
    print(f"  Timeout: {timeout_seconds}s")
    print(f"  Stuck detection: {detect_stuck}")
    print(f"{'═' * 60}")
    
    for iteration in range(max_iterations):
        stats.iterations = iteration + 1
        
        # CHECK: Timeout
        if stats.elapsed_seconds > timeout_seconds:
            raise AgentTerminated(
                f"Timeout after {stats.elapsed_seconds:.1f}s",
                {"limit": timeout_seconds, "actual": stats.elapsed_seconds}
            )
        
        # CHECK: Tool call limit
        if stats.tool_calls >= max_tool_calls:
            raise AgentTerminated(
                f"Tool call limit reached ({max_tool_calls})",
                {"limit": max_tool_calls, "actual": stats.tool_calls}
            )
        
        # CHECK: Stuck pattern
        if detect_stuck and detect_stuck_pattern(stats):
            raise AgentTerminated(
                "Stuck pattern detected",
                {"recent_calls": stats.tool_call_history[-4:]}
            )
        
        print(f"\n--- Iteration {iteration + 1} (elapsed: {stats.elapsed_seconds:.1f}s) ---")
        
        # THINK
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        )
        
        stats.input_tokens += response.usage.input_tokens
        stats.output_tokens += response.usage.output_tokens
        
        print(f"Stop reason: {response.stop_reason}")
        
        # CHECK: Normal completion
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"Complete! Response: {block.text[:100]}...")
                    return block.text, stats
            return "", stats
        
        # CHECK: Unexpected stop reason
        if response.stop_reason not in ("tool_use", "end_turn"):
            raise AgentTerminated(
                f"Unexpected stop reason: {response.stop_reason}",
                {"stop_reason": response.stop_reason}
            )
        
        # ACT: Process tool calls
        messages.append({"role": "assistant", "content": response.content})
        
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                # Check limit before executing
                if stats.tool_calls >= max_tool_calls:
                    raise AgentTerminated(
                        f"Tool call limit reached ({max_tool_calls})",
                        {"limit": max_tool_calls, "actual": stats.tool_calls}
                    )
                
                print(f"  Tool: {block.name}({block.input})")
                stats.record_tool_call(block.name, block.input)
                
                result = execute_tool(block.name, block.input)
                print(f"  Result: {result}")
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        messages.append({"role": "user", "content": tool_results})
    
    # Iteration limit reached
    raise AgentTerminated(
        f"Iteration limit reached ({max_iterations})",
        {"limit": max_iterations, "actual": stats.iterations}
    )


if __name__ == "__main__":
    print("\n" + "▓" * 60)
    print("  TERMINATION CONDITIONS DEMONSTRATION")
    print("▓" * 60)
    
    # Test 1: Normal completion within limits
    print("\n\n" + "▒" * 60)
    print("  TEST 1: Normal Completion")
    print("▒" * 60)
    try:
        response, stats = run_agent_with_limits(
            "What time is it right now?",
            max_iterations=5,
            max_tool_calls=10,
            timeout_seconds=30.0
        )
        print(f"\n✓ Success!")
        print(f"  Response: {response}")
        print(f"  Stats: {stats.to_dict()}")
    except AgentTerminated as e:
        print(f"\n✗ Terminated: {e.reason}")
    
    # Test 2: Hitting tool call limit
    print("\n\n" + "▒" * 60)
    print("  TEST 2: Tool Call Limit")
    print("▒" * 60)
    try:
        response, stats = run_agent_with_limits(
            "Increment the counter 20 times, one increment at a time.",
            max_iterations=25,
            max_tool_calls=5,  # Low limit
            timeout_seconds=60.0
        )
        print(f"\n✓ Success!")
        print(f"  Response: {response}")
    except AgentTerminated as e:
        print(f"\n✗ Terminated: {e.reason}")
        print(f"  Details: {e.details}")
    
    # Test 3: Timeout (with slow operation)
    print("\n\n" + "▒" * 60)
    print("  TEST 3: Timeout")
    print("▒" * 60)
    try:
        response, stats = run_agent_with_limits(
            "Run a slow operation for 3 seconds, three times in a row.",
            max_iterations=10,
            max_tool_calls=20,
            timeout_seconds=5.0  # Short timeout
        )
        print(f"\n✓ Success!")
        print(f"  Response: {response}")
    except AgentTerminated as e:
        print(f"\n✗ Terminated: {e.reason}")
        print(f"  Details: {e.details}")
    
    # Test 4: Iteration limit
    print("\n\n" + "▒" * 60)
    print("  TEST 4: Iteration Limit")
    print("▒" * 60)
    try:
        response, stats = run_agent_with_limits(
            "Increment the counter 100 times.",
            max_iterations=3,  # Very low
            max_tool_calls=100,
            timeout_seconds=120.0
        )
        print(f"\n✓ Success!")
        print(f"  Response: {response}")
    except AgentTerminated as e:
        print(f"\n✗ Terminated: {e.reason}")
        print(f"  Details: {e.details}")
    
    print("\n" + "▓" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("▓" * 60)
