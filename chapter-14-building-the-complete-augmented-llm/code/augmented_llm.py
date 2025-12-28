"""
The AugmentedLLM class - a complete building block for agent development.

This module provides the core building block for all agent development in this book.
The AugmentedLLM class integrates:
- System prompts for behavior configuration
- Tool registration and automatic execution
- Multi-turn tool use loops
- Optional structured output with validation

Chapter 14: Building the Complete Augmented LLM
"""

import os
import json
from typing import Any, Callable
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass(frozen=True)
class AugmentedLLMConfig:
    """
    Configuration for an AugmentedLLM instance.
    
    This dataclass is frozen (immutable) to prevent accidental configuration
    changes during operation that could cause subtle bugs.
    
    Attributes:
        model: The Claude model to use for completions
        max_tokens: Maximum tokens in the response
        system_prompt: Instructions that define the LLM's behavior
        max_tool_iterations: Safety limit for tool use loops (prevents infinite loops)
        temperature: Randomness in responses (0.0 = deterministic, 1.0 = creative)
        response_schema: Optional JSON Schema for structured output validation
    """
    
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    system_prompt: str = "You are a helpful assistant."
    max_tool_iterations: int = 10
    temperature: float = 1.0
    response_schema: dict[str, Any] | None = None


class ToolRegistry:
    """
    A registry that maps tool names to their definitions and implementations.
    
    This class ensures that every defined tool has an implementation and every
    implementation has a definition—no mismatches possible. It serves as the
    single source of truth for tool management.
    
    Example:
        registry = ToolRegistry()
        registry.register(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {}},
            function=lambda: datetime.now().isoformat()
        )
        
        # Get definitions for Claude
        tools = registry.get_definitions()
        
        # Execute a tool
        result = registry.execute("get_time", {})
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, dict[str, Any]] = {}
        self._functions: dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable
    ) -> None:
        """
        Register a tool with its definition and implementation.
        
        Args:
            name: Unique identifier for the tool. Should be lowercase with
                  underscores (e.g., 'get_weather', 'calculate_sum').
            description: What the tool does. This is read by Claude, so write
                        it clearly! Include when to use the tool and what it returns.
            parameters: JSON Schema defining the tool's parameters. Must be a
                       valid JSON Schema object.
            function: The Python function that implements the tool. Its signature
                     should match the parameters schema.
        
        Raises:
            ValueError: If a tool with this name is already registered.
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        # Store the definition in Claude's expected format
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": parameters
        }
        
        # Store the implementation
        self._functions[name] = function
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """
        Return all tool definitions in the format Claude expects.
        
        Returns:
            A list of tool definition dictionaries, each containing:
            - name: The tool's name
            - description: What the tool does
            - input_schema: JSON Schema for parameters
        """
        return list(self._tools.values())
    
    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool by name with the given arguments.
        
        Args:
            name: The name of the tool to execute
            arguments: A dictionary of arguments to pass to the tool function.
                      Keys should match the parameter names in the schema.
        
        Returns:
            Whatever the tool function returns
        
        Raises:
            KeyError: If no tool with this name exists
            Exception: Any exception raised by the tool function
        """
        if name not in self._functions:
            raise KeyError(f"Unknown tool: {name}")
        
        return self._functions[name](**arguments)
    
    def has_tool(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: The tool name to check
            
        Returns:
            True if the tool exists, False otherwise
        """
        return name in self._tools
    
    def get_tool_names(self) -> list[str]:
        """
        Get a list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
    
    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        tools = ", ".join(self._tools.keys())
        return f"ToolRegistry([{tools}])"


class AugmentedLLM:
    """
    An LLM enhanced with tools, system prompts, and structured output.
    
    This class is the fundamental building block for all agent development.
    It wraps the Anthropic API and provides:
    
    - System prompts for behavior configuration
    - Tool registration and automatic execution
    - Multi-turn tool use loops (handles sequential tool calls)
    - Optional structured output with JSON Schema validation
    - Conversation history management
    
    The class handles the complexity of the tool-use loop internally,
    allowing you to focus on defining tools and processing results.
    
    Example:
        # Basic usage
        llm = AugmentedLLM()
        response = llm.run("Hello, how are you?")
        
        # With tools
        llm = AugmentedLLM()
        llm.register_tool(
            name="get_time",
            description="Get the current time",
            parameters={"type": "object", "properties": {}},
            function=lambda: datetime.now().isoformat()
        )
        response = llm.run("What time is it?")
        
        # With structured output
        config = AugmentedLLMConfig(
            system_prompt="Respond with JSON containing 'answer' and 'confidence'",
            response_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"}
                },
                "required": ["answer", "confidence"]
            }
        )
        llm = AugmentedLLM(config=config)
        response = llm.run("What is the capital of France?")
    """
    
    def __init__(
        self,
        config: AugmentedLLMConfig | None = None,
        tools: ToolRegistry | None = None
    ):
        """
        Initialize an AugmentedLLM instance.
        
        Args:
            config: Configuration options. Uses sensible defaults if not provided.
            tools: Pre-configured tool registry. Creates an empty registry if
                   not provided (you can add tools later with register_tool).
        """
        self.config = config or AugmentedLLMConfig()
        self.tools = tools or ToolRegistry()
        self._client = anthropic.Anthropic()
        self._conversation_history: list[dict[str, Any]] = []
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable
    ) -> None:
        """
        Register a tool that the LLM can use.
        
        This is a convenience method that delegates to the internal ToolRegistry.
        
        Args:
            name: Unique identifier for the tool
            description: What the tool does. Write this for Claude—explain when
                        to use the tool and what it returns.
            parameters: JSON Schema for the tool's parameters
            function: The Python function that implements the tool
        """
        self.tools.register(name, description, parameters, function)
    
    def run(
        self,
        message: str,
        conversation_history: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Send a message and get a response, handling any tool calls automatically.
        
        This is the main entry point for interacting with the LLM. It:
        1. Adds your message to the conversation
        2. Sends the request to Claude
        3. If Claude wants to use tools, executes them automatically
        4. Continues until Claude provides a final text response
        5. Optionally validates the response against a schema
        
        The method maintains conversation history internally, so you can have
        multi-turn conversations by calling run() multiple times.
        
        Args:
            message: The user's message to send
            conversation_history: Optional external history. If provided, this
                                 is used instead of the internal history. Useful
                                 for advanced scenarios where you manage history
                                 yourself.
        
        Returns:
            The LLM's text response. If a response_schema was configured, this
            will be a validated JSON string.
        
        Raises:
            RuntimeError: If max_tool_iterations is exceeded (likely an infinite loop)
            ValueError: If response doesn't match the required schema
            anthropic.APIError: If the API request fails
        """
        # Initialize conversation history
        if conversation_history is not None:
            messages = conversation_history.copy()
        else:
            messages = self._conversation_history.copy()
        
        # Add the new user message
        messages.append({"role": "user", "content": message})
        
        # Build the base API request parameters
        request_params = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": self.config.system_prompt,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        
        # Add tools if any are registered
        if len(self.tools) > 0:
            request_params["tools"] = self.tools.get_definitions()
        
        # The main loop: call API, handle tools, repeat until done
        iterations = 0
        
        while iterations < self.config.max_tool_iterations:
            iterations += 1
            
            # Make the API call
            response = self._client.messages.create(**request_params)
            
            # Check if we're done (no more tool use)
            if response.stop_reason == "end_turn":
                # Extract the text response
                text_response = self._extract_text(response)
                
                # Update internal history with the complete exchange
                messages.append({"role": "assistant", "content": response.content})
                self._conversation_history = messages
                
                # Validate against schema if provided
                if self.config.response_schema is not None:
                    return self._validate_response(text_response)
                
                return text_response
            
            # Handle tool use
            if response.stop_reason == "tool_use":
                # Add assistant's response to history (includes tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})
                
                # Process each tool call in the response
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        result = self._execute_tool(
                            content_block.name,
                            content_block.input,
                            content_block.id
                        )
                        tool_results.append(result)
                
                # Add tool results to messages
                messages.append({"role": "user", "content": tool_results})
                
                # Update request for next iteration
                request_params["messages"] = messages
            else:
                # Unexpected stop reason (max_tokens, etc.)
                text_response = self._extract_text(response)
                messages.append({"role": "assistant", "content": response.content})
                self._conversation_history = messages
                return text_response
        
        # If we get here, we've exceeded max iterations
        raise RuntimeError(
            f"Exceeded maximum tool iterations ({self.config.max_tool_iterations}). "
            "The LLM may be stuck in a loop. Check your tool implementations and prompts."
        )
    
    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str
    ) -> dict[str, Any]:
        """
        Execute a tool and format the result for Claude.
        
        This method handles tool execution including error handling. If a tool
        raises an exception, the error is returned to Claude so it can decide
        how to proceed.
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Arguments for the tool (parsed from Claude's request)
            tool_use_id: Unique ID for this tool use (required by the API)
        
        Returns:
            A tool_result block suitable for adding to the messages array
        """
        try:
            result = self.tools.execute(tool_name, tool_input)
            
            # Convert result to string if needed (Claude expects string content)
            if not isinstance(result, str):
                result = json.dumps(result)
            
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result
            }
        except Exception as e:
            # Return error as tool result so Claude can handle it gracefully
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error executing tool: {str(e)}",
                "is_error": True
            }
    
    def _extract_text(self, response: anthropic.types.Message) -> str:
        """
        Extract text content from an API response.
        
        Claude's responses can contain multiple content blocks (text and tool_use).
        This method finds and returns the text content.
        
        Args:
            response: The API response message
        
        Returns:
            The text content, or empty string if no text found
        """
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
    
    def _validate_response(self, response: str) -> str:
        """
        Validate and parse a response against the configured schema.
        
        This performs basic JSON Schema validation including:
        - JSON parsing
        - Required field checking
        - Type validation for basic types
        - Enum validation
        
        Args:
            response: The raw text response from Claude
        
        Returns:
            The validated JSON string (re-serialized for consistency)
        
        Raises:
            ValueError: If response doesn't match the schema
        """
        try:
            # Parse the JSON
            data = json.loads(response)
            
            schema = self.config.response_schema
            
            # Check required fields
            if "required" in schema:
                for field in schema["required"]:
                    if field not in data:
                        raise ValueError(f"Missing required field: {field}")
            
            # Validate property types
            if "properties" in schema:
                for field, field_schema in schema["properties"].items():
                    if field in data:
                        self._validate_field(field, data[field], field_schema)
            
            # Return the validated JSON (re-serialized)
            return json.dumps(data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Response is not valid JSON: {e}")
    
    def _validate_field(self, name: str, value: Any, schema: dict) -> None:
        """
        Validate a single field against its schema.
        
        Args:
            name: Field name (for error messages)
            value: The value to validate
            schema: The JSON Schema for this field
        
        Raises:
            ValueError: If validation fails
        """
        expected_type = schema.get("type")
        
        # Map JSON Schema types to Python types
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        
        # Check type
        if expected_type in type_mapping:
            if not isinstance(value, type_mapping[expected_type]):
                raise ValueError(
                    f"Field '{name}' should be {expected_type}, "
                    f"got {type(value).__name__}"
                )
        
        # Check enum values
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(
                f"Field '{name}' must be one of {schema['enum']}, got '{value}'"
            )
    
    def clear_history(self) -> None:
        """
        Clear the internal conversation history.
        
        Call this between unrelated queries to prevent context bleed.
        """
        self._conversation_history = []
    
    def get_history(self) -> list[dict[str, Any]]:
        """
        Return a copy of the conversation history.
        
        Returns:
            A copy of the messages list. Modifying this won't affect
            the internal history.
        """
        return self._conversation_history.copy()


# Convenience function for simple one-shot queries
def quick_query(
    message: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "claude-sonnet-4-20250514"
) -> str:
    """
    Send a quick one-shot query without configuring a full AugmentedLLM.
    
    This is a convenience function for simple queries that don't need
    tools or conversation history.
    
    Args:
        message: The message to send
        system_prompt: Optional system prompt
        model: The model to use
    
    Returns:
        The response text
    """
    config = AugmentedLLMConfig(
        model=model,
        system_prompt=system_prompt
    )
    llm = AugmentedLLM(config=config)
    return llm.run(message)


if __name__ == "__main__":
    # Quick demonstration
    print("AugmentedLLM - Quick Demo")
    print("=" * 50)
    
    # Create a simple instance
    llm = AugmentedLLM(
        config=AugmentedLLMConfig(
            system_prompt="You are a helpful assistant. Be concise."
        )
    )
    
    # Test basic query
    response = llm.run("What are the three primary colors?")
    print(f"\nQ: What are the three primary colors?")
    print(f"A: {response}")
