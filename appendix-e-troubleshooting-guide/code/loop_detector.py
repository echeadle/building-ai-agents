"""
Loop Detection Utility for AI Agents

Appendix E: Troubleshooting Guide
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Any
from dataclasses import dataclass, field

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@dataclass
class LoopDetector:
    """
    Detects infinite loops in agent tool calls.
    
    Tracks tool call history and identifies repeated patterns
    that indicate the agent is stuck in a loop.
    """
    
    tool_call_history: list[str] = field(default_factory=list)
    max_repeated_calls: int = 2  # How many times before it's a loop
    
    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """
        Add a tool call and check if we're in a loop.
        
        Args:
            tool_name: Name of the tool being called
            tool_input: Input parameters for the tool
            
        Returns:
            True if a loop is detected, False otherwise
        """
        # Create a signature for this tool call
        signature = self._create_signature(tool_name, tool_input)
        
        # Check if we've seen this exact call recently
        recent_calls = self.tool_call_history[-5:]  # Look at last 5 calls
        repetitions = recent_calls.count(signature)
        
        # Add to history
        self.tool_call_history.append(signature)
        
        # Detect loop
        if repetitions >= self.max_repeated_calls:
            print(f"⚠️  Loop detected: {tool_name} called {repetitions + 1} times with same input")
            return True
        
        return False
    
    def _create_signature(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Create a unique signature for a tool call."""
        # Sort input to ensure consistent ordering
        sorted_input = str(sorted(tool_input.items()))
        return f"{tool_name}({sorted_input})"
    
    def detect_cycle(self) -> bool:
        """
        Detect if the agent is in a cycle of different tools.
        
        Example: A → B → C → A → B → C (cycle of 3)
        """
        if len(self.tool_call_history) < 6:
            return False
        
        # Check for cycles of length 2, 3, 4
        for cycle_length in [2, 3, 4]:
            recent = self.tool_call_history[-cycle_length * 2:]
            first_half = recent[:cycle_length]
            second_half = recent[cycle_length:]
            
            if first_half == second_half:
                print(f"⚠️  Cycle detected: Pattern of {cycle_length} tools repeating")
                print(f"   Pattern: {' → '.join(first_half)}")
                return True
        
        return False
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about tool calls."""
        if not self.tool_call_history:
            return {"total_calls": 0}
        
        # Count each unique tool
        tool_counts: dict[str, int] = {}
        for signature in self.tool_call_history:
            tool_name = signature.split("(")[0]
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        return {
            "total_calls": len(self.tool_call_history),
            "unique_tools": len(set(sig.split("(")[0] for sig in self.tool_call_history)),
            "tool_counts": tool_counts,
            "most_called": max(tool_counts, key=tool_counts.get) if tool_counts else None,
        }
    
    def reset(self):
        """Reset the detector for a new conversation."""
        self.tool_call_history.clear()


class LoopDetectingAgent:
    """
    Agent with built-in loop detection.
    
    Automatically stops execution if it detects the agent
    is stuck in an infinite loop.
    """
    
    def __init__(self, tools: list[dict], max_iterations: int = 10):
        self.tools = tools
        self.max_iterations = max_iterations
        self.loop_detector = LoopDetector(max_repeated_calls=2)
    
    def run(self, user_message: str, system_prompt: str = "") -> str:
        """
        Run the agent with loop detection.
        
        Args:
            user_message: The user's input
            system_prompt: Optional system prompt
            
        Returns:
            The agent's final response
        """
        conversation = [{"role": "user", "content": user_message}]
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Make API call
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt if system_prompt else [],
                messages=conversation,
                tools=self.tools
            )
            
            # Check if agent is done
            if response.stop_reason == "end_turn":
                final_text = self._extract_text(response)
                print(f"\n✅ Completed in {iteration + 1} iterations")
                return final_text
            
            # Process tool calls
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            
            if not tool_uses:
                return self._extract_text(response)
            
            # Check each tool call for loops
            for tool_use in tool_uses:
                print(f"Tool: {tool_use.name}")
                
                # Check for loop
                if self.loop_detector.add_tool_call(tool_use.name, tool_use.input):
                    stats = self.loop_detector.get_statistics()
                    return f"⚠️  Loop detected and stopped. Stats: {stats}"
                
                # Check for cycles
                if self.loop_detector.detect_cycle():
                    stats = self.loop_detector.get_statistics()
                    return f"⚠️  Cycle detected and stopped. Stats: {stats}"
            
            # Execute tools (simplified for example)
            tool_results = []
            for tool_use in tool_uses:
                result = self._execute_tool(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result)
                })
            
            # Add to conversation
            conversation.append({"role": "assistant", "content": response.content})
            conversation.append({"role": "user", "content": tool_results})
        
        return f"⚠️  Max iterations ({self.max_iterations}) reached"
    
    def _extract_text(self, response) -> str:
        """Extract text from response."""
        text_blocks = [block.text for block in response.content if hasattr(block, "text")]
        return "\n".join(text_blocks)
    
    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool (stub for example)."""
        # In real implementation, this would call actual tool functions
        return f"Tool {tool_name} executed with {tool_input}"


# Example usage
if __name__ == "__main__":
    # Define some tools
    tools = [
        {
            "name": "search",
            "description": "Search for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    ]
    
    # Create agent with loop detection
    agent = LoopDetectingAgent(tools=tools, max_iterations=15)
    
    # Test with a query
    result = agent.run(
        user_message="What is 2 + 2?",
        system_prompt="You are a helpful assistant. Use tools when needed."
    )
    
    print(f"\nFinal result: {result}")
    
    # Show statistics
    stats = agent.loop_detector.get_statistics()
    print(f"\nTool call statistics: {stats}")
