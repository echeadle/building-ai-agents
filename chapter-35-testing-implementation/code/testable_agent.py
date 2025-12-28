"""
A minimal agent class designed for testing.

Chapter 35: Testing AI Agents - Implementation

This module provides a simplified agent implementation that is easy
to test in isolation. It demonstrates the core agent loop without
external dependencies.
"""

from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """
    Configuration for the test agent.
    
    Attributes:
        max_iterations: Maximum number of LLM calls before stopping
        system_prompt: The system prompt for the agent
        verbose: Whether to print debug information
    """
    max_iterations: int = 10
    system_prompt: str = "You are a helpful assistant."
    verbose: bool = False


class TestableAgent:
    """
    A minimal agent implementation designed for testing.
    
    This agent demonstrates the core agentic loop:
    1. Receive user message
    2. Call LLM
    3. If LLM requests tool use, execute tool and loop back to step 2
    4. If LLM returns text, return it to the user
    
    The agent is designed to work with any LLM client that has a
    create_message method, making it easy to test with MockLLM.
    
    Example:
        >>> from mock_llm import MockLLM
        >>> 
        >>> mock = MockLLM()
        >>> mock.add_response(text="Hello!")
        >>> 
        >>> agent = TestableAgent(llm_client=mock)
        >>> response = agent.run("Hi there!")
        >>> print(response)  # "Hello!"
    """
    
    def __init__(
        self,
        llm_client: Any,
        tools: dict[str, Callable] = None,
        tool_definitions: list[dict] = None,
        config: AgentConfig = None
    ):
        """
        Initialize the agent.
        
        Args:
            llm_client: An object with a create_message method (real or mock)
            tools: Dict mapping tool names to callable functions
            tool_definitions: List of tool definitions for the LLM
            config: Agent configuration
        """
        self.llm = llm_client
        self.tools = tools or {}
        self.tool_definitions = tool_definitions or []
        self.config = config or AgentConfig()
        self.conversation_history: list[dict] = []
        self.tool_call_log: list[dict] = []
        self._iteration_count = 0
    
    def run(self, user_message: str) -> str:
        """
        Run the agent with a user message.
        
        This is the main entry point for interacting with the agent.
        It manages the agentic loop, handling tool calls until the
        LLM returns a final text response.
        
        Args:
            user_message: The user's input message
            
        Returns:
            The agent's final response text
            
        Raises:
            RuntimeError: If max iterations is exceeded
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        self._iteration_count = 0
        
        # Agentic loop
        for iteration in range(self.config.max_iterations):
            self._iteration_count = iteration + 1
            
            if self.config.verbose:
                print(f"[Agent] Iteration {iteration + 1}")
            
            # Call the LLM
            response = self._call_llm()
            
            # Check if the LLM wants to use a tool
            if response.stop_reason == "tool_use":
                if self.config.verbose:
                    print(f"[Agent] Tool use requested")
                self._handle_tool_calls(response)
            else:
                # Extract and return the text response
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "text":
                        # Add assistant response to history
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": block.text
                        })
                        return block.text
                
                # No text content found
                return ""
        
        raise RuntimeError(
            f"Agent exceeded maximum iterations ({self.config.max_iterations})"
        )
    
    def _call_llm(self) -> Any:
        """
        Make a call to the LLM.
        
        Returns:
            The LLM response object
        """
        return self.llm.create_message(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=self.conversation_history,
            system=self.config.system_prompt,
            tools=self.tool_definitions if self.tool_definitions else None
        )
    
    def _handle_tool_calls(self, response: Any) -> None:
        """
        Process tool calls from the LLM response.
        
        Executes each requested tool and adds results to the
        conversation history for the next LLM call.
        
        Args:
            response: The LLM response containing tool use blocks
        """
        # Add assistant message to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        # Process each tool call
        tool_results = []
        for block in response.content:
            if hasattr(block, 'type') and block.type == "tool_use":
                if self.config.verbose:
                    print(f"[Agent] Executing tool: {block.name}")
                    print(f"[Agent] With input: {block.input}")
                
                result = self._execute_tool(block.name, block.input, block.id)
                tool_results.append(result)
                
                if self.config.verbose:
                    print(f"[Agent] Tool result: {result['content'][:100]}...")
        
        # Add tool results to history
        if tool_results:
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
    
    def _execute_tool(self, name: str, inputs: dict, tool_id: str) -> dict:
        """
        Execute a tool and return the result.
        
        Args:
            name: The name of the tool to execute
            inputs: The input parameters for the tool
            tool_id: The ID of the tool use block
            
        Returns:
            A tool_result dict for the API
        """
        # Log the tool call
        self.tool_call_log.append({
            "name": name,
            "inputs": inputs,
            "id": tool_id
        })
        
        # Check if tool exists
        if name not in self.tools:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Error: Unknown tool '{name}'. Available tools: {list(self.tools.keys())}",
                "is_error": True
            }
        
        # Execute the tool
        try:
            result = self.tools[name](**inputs)
            
            # Convert result to string if needed
            if isinstance(result, dict):
                import json
                result_str = json.dumps(result)
            else:
                result_str = str(result)
            
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result_str
            }
        except TypeError as e:
            # Wrong arguments
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Error: Invalid arguments for tool '{name}': {str(e)}",
                "is_error": True
            }
        except Exception as e:
            # Other errors
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Error executing tool '{name}': {str(e)}",
                "is_error": True
            }
    
    def clear_history(self) -> None:
        """Clear the conversation history and tool call log."""
        self.conversation_history.clear()
        self.tool_call_log.clear()
        self._iteration_count = 0
    
    def get_iteration_count(self) -> int:
        """Get the number of iterations from the last run."""
        return self._iteration_count
    
    def get_tool_calls(self) -> list[dict]:
        """Get a copy of the tool call log."""
        return self.tool_call_log.copy()


if __name__ == "__main__":
    # Demonstration with MockLLM
    from mock_llm import MockLLM
    from calculator import calculator, CALCULATOR_TOOL_DEFINITION
    
    print("TestableAgent Demonstration")
    print("=" * 40)
    
    # Create mock with conversation
    mock = MockLLM()
    mock.add_response(text="Hello! I'm ready to help you with calculations.")
    mock.add_response(
        tool_call={
            "name": "calculator",
            "id": "toolu_1",
            "input": {"operation": "add", "a": 15, "b": 27}
        }
    )
    mock.add_response(text="15 + 27 = 42. The answer is 42!")
    
    # Create agent
    agent = TestableAgent(
        llm_client=mock,
        tools={"calculator": calculator},
        tool_definitions=[CALCULATOR_TOOL_DEFINITION],
        config=AgentConfig(verbose=True)
    )
    
    # Run conversations
    print("\n1. Simple greeting:")
    response = agent.run("Hello!")
    print(f"   Agent: {response}")
    
    print("\n2. Math question:")
    response = agent.run("What is 15 + 27?")
    print(f"   Agent: {response}")
    
    print(f"\n3. Tool calls made: {len(agent.tool_call_log)}")
    for call in agent.tool_call_log:
        print(f"   - {call['name']}({call['inputs']})")
