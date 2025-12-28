"""
Detailed tracing for agent decisions and tool calls.

Chapter 36: Observability and Logging

This script demonstrates how to trace not just what an agent does,
but why it makes each decision. This level of detail is essential
for debugging and understanding agent behavior.

Note: This example makes actual API calls to Claude.
"""

import os
import time
import json
import logging
import sys
from typing import Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

import anthropic


@dataclass
class DecisionTrace:
    """
    Captures the reasoning behind an agent decision.
    
    This helps understand not just what the agent did,
    but why it made that choice.
    """
    decision_type: str  # "tool_selection", "response_generation", "planning"
    options_considered: list[str] = field(default_factory=list)
    chosen_option: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: Optional[str] = None


class DecisionTracer:
    """
    A tracer that captures detailed decision information.
    
    This class wraps agent operations and records:
    - What options were available
    - What option was chosen
    - Why (based on available information)
    - When the decision was made
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.decisions: list[DecisionTrace] = []
    
    def log_tool_selection(
        self,
        available_tools: list[str],
        selected_tool: str,
        tool_input: dict[str, Any],
        context: str = ""
    ) -> DecisionTrace:
        """
        Log a tool selection decision.
        
        Args:
            available_tools: List of tools that were available
            selected_tool: The tool that was selected
            tool_input: The input provided to the tool
            context: Additional context about why this tool was selected
        """
        from datetime import datetime, timezone
        
        decision = DecisionTrace(
            decision_type="tool_selection",
            options_considered=available_tools,
            chosen_option=selected_tool,
            reasoning=f"Selected for input: {json.dumps(tool_input)[:100]}. {context}",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.decisions.append(decision)
        
        self.logger.info(json.dumps({
            "event": "decision_made",
            "decision_type": "tool_selection",
            "available_options": available_tools,
            "selected": selected_tool,
            "input_summary": str(tool_input)[:100]
        }))
        
        return decision
    
    def log_response_decision(
        self,
        stop_reason: str,
        has_tool_calls: bool,
        content_blocks: int
    ) -> DecisionTrace:
        """Log the model's response decision."""
        from datetime import datetime, timezone
        
        decision = DecisionTrace(
            decision_type="response_generation",
            chosen_option=stop_reason,
            reasoning=f"Generated {content_blocks} content blocks, tool_calls={has_tool_calls}",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        self.decisions.append(decision)
        
        self.logger.info(json.dumps({
            "event": "response_decision",
            "stop_reason": stop_reason,
            "has_tool_calls": has_tool_calls,
            "content_blocks": content_blocks
        }))
        
        return decision
    
    def get_decision_summary(self) -> dict[str, Any]:
        """Get a summary of all decisions made."""
        tool_selections = [
            d for d in self.decisions 
            if d.decision_type == "tool_selection"
        ]
        
        return {
            "total_decisions": len(self.decisions),
            "tool_selections": len(tool_selections),
            "tools_selected": [d.chosen_option for d in tool_selections],
            "decision_types": list(set(d.decision_type for d in self.decisions))
        }


class TracingAgent:
    """
    An agent with detailed decision tracing.
    
    This demonstrates how to capture not just what the agent does,
    but why it makes each decision.
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"
        
        # Set up logging
        self.logger = logging.getLogger("tracing_agent")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        
        # Decision tracer
        self.tracer = DecisionTracer(self.logger)
        
        # Define available tools
        self.tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location. Use this when the user asks about weather conditions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and state, e.g., San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations. Use this for any math operations.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate, e.g., '2 + 2' or '15 * 3'"
                        }
                    },
                    "required": ["expression"]
                }
            },
            {
                "name": "get_time",
                "description": "Get the current time. Use this when the user asks about the current time.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        self.tool_names = [t["name"] for t in self.tools]
    
    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name == "get_weather":
            # Simulated weather response
            location = tool_input.get("location", "Unknown")
            return f"Weather in {location}: 72°F, sunny with light clouds"
        
        elif tool_name == "calculate":
            try:
                expression = tool_input.get("expression", "")
                # WARNING: eval is dangerous! Use a proper math parser in production
                result = eval(expression, {"__builtins__": {}}, {})
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        elif tool_name == "get_time":
            from datetime import datetime
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        else:
            return f"Unknown tool: {tool_name}"
    
    def process_request(self, user_input: str) -> str:
        """
        Process a user request with full decision tracing.
        
        Args:
            user_input: The user's message
        
        Returns:
            The agent's final response
        """
        self.logger.info(json.dumps({
            "event": "request_started",
            "user_input": user_input
        }))
        
        messages = [{"role": "user", "content": user_input}]
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            
            # Make LLM call
            start_time = time.perf_counter()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                tools=self.tools,
                messages=messages
            )
            
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log LLM call details
            self.logger.info(json.dumps({
                "event": "llm_call_completed",
                "iteration": iteration,
                "duration_ms": round(duration_ms, 2),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "stop_reason": response.stop_reason,
                "content_blocks": len(response.content)
            }))
            
            # Trace the response decision
            has_tool_calls = any(
                block.type == "tool_use" for block in response.content
            )
            self.tracer.log_response_decision(
                stop_reason=response.stop_reason,
                has_tool_calls=has_tool_calls,
                content_blocks=len(response.content)
            )
            
            # Check if we're done
            if response.stop_reason == "end_turn":
                # Extract text response
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text
                
                self.logger.info(json.dumps({
                    "event": "request_completed",
                    "iterations": iteration,
                    "final_response_length": len(final_text)
                }))
                
                return final_text
            
            # Handle tool use
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        # Trace the tool selection
                        self.tracer.log_tool_selection(
                            available_tools=self.tool_names,
                            selected_tool=block.name,
                            tool_input=block.input
                        )
                        
                        # Execute the tool
                        self.logger.info(json.dumps({
                            "event": "tool_execution_started",
                            "tool_name": block.name,
                            "tool_input": block.input
                        }))
                        
                        tool_start = time.perf_counter()
                        result = self._execute_tool(block.name, block.input)
                        tool_duration = (time.perf_counter() - tool_start) * 1000
                        
                        self.logger.info(json.dumps({
                            "event": "tool_execution_completed",
                            "tool_name": block.name,
                            "duration_ms": round(tool_duration, 2),
                            "result_length": len(result)
                        }))
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                # Continue the conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            
            else:
                # Unexpected stop reason
                self.logger.warning(json.dumps({
                    "event": "unexpected_stop_reason",
                    "stop_reason": response.stop_reason
                }))
                break
        
        return "Max iterations reached"
    
    def get_decision_summary(self) -> dict[str, Any]:
        """Get a summary of all decisions made during processing."""
        return self.tracer.get_decision_summary()


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Agent Decision Tracing Demo")
    print("=" * 70)
    print()
    print("This demo shows detailed tracing of agent decisions.")
    print("Each log line shows what the agent is doing and why.")
    print()
    print("-" * 70)
    print()
    
    agent = TracingAgent()
    
    # Test with a query that requires tool use
    test_query = "What's the weather in San Francisco and what is 15 * 7?"
    
    print(f"User: {test_query}")
    print()
    print("Agent processing (with decision tracing):")
    print("-" * 40)
    
    response = agent.process_request(test_query)
    
    print("-" * 40)
    print()
    print(f"Agent response: {response}")
    print()
    print("-" * 70)
    print("Decision Summary:")
    print(json.dumps(agent.get_decision_summary(), indent=2))
