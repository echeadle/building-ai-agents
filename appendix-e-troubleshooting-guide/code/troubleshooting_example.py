"""
Comprehensive Troubleshooting Example

Appendix E: Troubleshooting Guide

This example demonstrates how to diagnose and fix a broken agent.
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Optional

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ====================================================================
# BROKEN AGENT - This agent has multiple problems!
# ====================================================================

class BrokenAgent:
    """
    An agent with multiple common problems.
    Can you identify them all?
    """
    
    def __init__(self):
        # Problem 1: Tools not configured properly
        self.tools = [
            {
                "name": "search",
                # Missing description!
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}  # Missing description!
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calculate",
                "description": "calc",  # Too vague!
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        ]
        
        self.conversation = []
        # Problem 2: No token management
        # Problem 3: No rate limiting
        # Problem 4: No loop detection
    
    def run(self, user_message: str) -> str:
        """Run the agent - but it has problems!"""
        
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        # Problem 5: No error handling
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,  # Problem 6: Too low for tool calls!
            messages=self.conversation
            # Problem 7: Not passing tools!
        )
        
        # Problem 8: No loop detection - could run forever
        text = response.content[0].text if response.content else "No response"
        
        self.conversation.append({
            "role": "assistant",
            "content": response.content
        })
        
        return text


# ====================================================================
# STEP 1: DIAGNOSE THE PROBLEMS
# ====================================================================

def diagnose_broken_agent():
    """Identify all problems with the broken agent."""
    
    print("=" * 60)
    print("STEP 1: DIAGNOSING BROKEN AGENT")
    print("=" * 60)
    print()
    
    agent = BrokenAgent()
    
    # Import diagnostic tools
    from diagnostics import AgentDiagnostics
    
    diagnostics = AgentDiagnostics()
    
    # Check 1: API connectivity
    print("Check 1: API Connectivity")
    print("-" * 60)
    api_check = diagnostics.check_api_connectivity()
    print(f"Connected: {api_check['connected']}")
    if not api_check['connected']:
        print(f"Issue: {api_check['message']}")
    print()
    
    # Check 2: Tool configuration
    print("Check 2: Tool Configuration")
    print("-" * 60)
    tool_check = diagnostics.check_tools_configuration(agent.tools)
    print(f"Valid: {tool_check['valid']}")
    print(f"Issues found: {len(tool_check['issues'])}")
    for issue in tool_check['issues']:
        print(f"  ❌ {issue}")
    print(f"Warnings: {len(tool_check['warnings'])}")
    for warning in tool_check['warnings']:
        print(f"  ⚠️  {warning}")
    print()
    
    # Check 3: Try to run the agent
    print("Check 3: Test Run")
    print("-" * 60)
    try:
        result = agent.run("What is 2 + 2?")
        print(f"Result: {result}")
        print("⚠️  Agent ran but tools were not used (tools not passed to API)")
    except Exception as e:
        print(f"❌ Error: {e}")
    print()
    
    print("Summary of Problems Found:")
    print("-" * 60)
    print("1. Tool descriptions missing or too vague")
    print("2. Tool parameters missing descriptions")
    print("3. max_tokens too low (50 - not enough for tool calls)")
    print("4. Tools not passed to API")
    print("5. No error handling")
    print("6. No token management")
    print("7. No rate limiting")
    print("8. No loop detection")
    print()


# ====================================================================
# STEP 2: FIX THE AGENT
# ====================================================================

class FixedAgent:
    """
    The agent with all problems fixed.
    """
    
    def __init__(self, max_iterations: int = 10):
        # Fix 1: Proper tool definitions
        self.tools = [
            {
                "name": "search",
                "description": "Search for information on the web. Use this when you need current information or facts you don't know.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find information"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform mathematical calculations. Use this for any arithmetic, algebra, or mathematical expressions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate (e.g., '2 + 2', '15 * 8')"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
        
        self.conversation = []
        self.max_iterations = max_iterations
        
        # Fix 2: Add utilities
        from loop_detector import LoopDetector
        from rate_limiter import RateLimiter
        from token_manager import TokenEstimator
        
        self.loop_detector = LoopDetector()
        self.rate_limiter = RateLimiter(requests_per_minute=50)
        self.token_estimator = TokenEstimator()
        self.max_conversation_tokens = 150000
    
    def run(self, user_message: str, system_prompt: str = "") -> str:
        """Run the agent - now properly!"""
        
        self.conversation.append({
            "role": "user",
            "content": user_message
        })
        
        # Fix 3: Check token usage before each call
        self._manage_tokens()
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Fix 4: Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Fix 5: Error handling
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,  # Fix 6: Adequate tokens
                    system=system_prompt if system_prompt else [],
                    messages=self.conversation,
                    tools=self.tools  # Fix 7: Pass tools!
                )
            except anthropic.RateLimitError:
                print("⚠️  Rate limit hit - waiting...")
                import time
                time.sleep(10)
                continue
            except anthropic.APIConnectionError:
                print("❌ Connection error")
                return "Could not connect to API"
            except Exception as e:
                print(f"❌ Error: {e}")
                return f"Error occurred: {str(e)}"
            
            # Check if done
            if response.stop_reason == "end_turn":
                text = self._extract_text(response)
                self.conversation.append({
                    "role": "assistant",
                    "content": response.content
                })
                print(f"✅ Completed in {iteration + 1} iterations")
                return text
            
            # Process tool calls
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            
            if not tool_uses:
                text = self._extract_text(response)
                self.conversation.append({
                    "role": "assistant",
                    "content": response.content
                })
                return text
            
            # Fix 8: Loop detection
            for tool_use in tool_uses:
                print(f"Tool: {tool_use.name}")
                
                if self.loop_detector.add_tool_call(tool_use.name, tool_use.input):
                    return "⚠️  Loop detected - stopping"
                
                if self.loop_detector.detect_cycle():
                    return "⚠️  Cycle detected - stopping"
            
            # Execute tools
            tool_results = self._execute_tools(tool_uses)
            
            # Continue conversation
            self.conversation.append({
                "role": "assistant",
                "content": response.content
            })
            self.conversation.append({
                "role": "user",
                "content": tool_results
            })
        
        return f"⚠️  Max iterations reached ({self.max_iterations})"
    
    def _manage_tokens(self):
        """Trim conversation if needed."""
        current_tokens = self.token_estimator.estimate_conversation_tokens(
            self.conversation
        )
        
        if current_tokens > self.max_conversation_tokens:
            print(f"⚠️  Trimming conversation ({current_tokens} tokens)")
            # Keep last 8 messages
            self.conversation = self.conversation[-8:]
    
    def _extract_text(self, response) -> str:
        """Extract text from response."""
        text_blocks = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(text_blocks)
    
    def _execute_tools(self, tool_uses) -> list:
        """Execute tools and return results."""
        results = []
        
        for tool_use in tool_uses:
            # In real implementation, call actual tools
            # For this example, we'll simulate
            if tool_use.name == "calculate":
                try:
                    expression = tool_use.input.get("expression", "")
                    result = eval(expression)  # Dangerous in production!
                    content = str(result)
                except Exception as e:
                    content = f"Error: {str(e)}"
            else:
                content = f"Tool {tool_use.name} executed"
            
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": content
            })
        
        return results
    
    def get_statistics(self) -> dict:
        """Get agent statistics."""
        return {
            "loop_detector": self.loop_detector.get_statistics(),
            "current_rate": self.rate_limiter.get_current_rate(),
            "token_estimate": self.token_estimator.estimate_conversation_tokens(
                self.conversation
            ),
        }


# ====================================================================
# STEP 3: COMPARE BROKEN VS FIXED
# ====================================================================

def compare_agents():
    """Compare the broken and fixed agents."""
    
    print("\n" + "=" * 60)
    print("STEP 2: TESTING FIXED AGENT")
    print("=" * 60)
    print()
    
    agent = FixedAgent(max_iterations=5)
    
    test_query = "What is 15 * 23 + 47?"
    
    print(f"Query: {test_query}")
    print("-" * 60)
    
    result = agent.run(
        test_query,
        system_prompt="You are a helpful assistant. Use tools when appropriate."
    )
    
    print()
    print("Result:", result)
    print()
    
    # Show statistics
    stats = agent.get_statistics()
    print("Statistics:")
    print("-" * 60)
    print(f"Tool calls: {stats['loop_detector']['total_calls']}")
    print(f"Current rate: {stats['current_rate']}/min")
    print(f"Token estimate: {stats['token_estimate']}")
    print()


# ====================================================================
# MAIN: RUN THE DEMONSTRATION
# ====================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "TROUBLESHOOTING DEMONSTRATION" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Step 1: Diagnose the broken agent
    diagnose_broken_agent()
    
    input("Press Enter to see the fixed agent in action...")
    
    # Step 2: Show the fixed agent
    compare_agents()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()
    print("Key Lessons:")
    print("1. Always validate tool definitions first")
    print("2. Use adequate max_tokens for tool calls (1024+)")
    print("3. Pass tools parameter to API calls")
    print("4. Add loop detection for safety")
    print("5. Implement rate limiting and error handling")
    print("6. Monitor token usage in long conversations")
    print()
    print("The diagnostic utilities in this appendix help you")
    print("identify and fix these problems systematically.")
    print()
