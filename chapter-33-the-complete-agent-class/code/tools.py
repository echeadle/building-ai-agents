"""
Tool registry for managing agent tools.

Chapter 33: The Complete Agent Class

This module provides a clean interface for registering, managing,
and executing tools that agents can use.
"""

import json
from typing import Any, Callable
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ToolDefinition:
    """
    A registered tool with its metadata and handler.
    
    Attributes:
        name: Unique identifier for the tool
        description: What the tool does (LLMs read this!)
        input_schema: JSON Schema defining expected parameters
        handler: Function to call when tool is invoked
        requires_approval: Whether human approval is needed
    """
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]
    requires_approval: bool = False


class ToolRegistry:
    """
    Registry for managing agent tools.
    
    Handles tool registration, validation, and execution.
    Provides both programmatic and decorator-based registration.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(
        ...     name="add",
        ...     description="Add two numbers",
        ...     input_schema={"type": "object", "properties": {...}},
        ...     handler=lambda a, b: a + b
        ... )
        >>> result = registry.execute("add", {"a": 5, "b": 3})
        >>> print(result)  # 8
    """
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
    
    def register(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[..., Any],
        requires_approval: bool = False
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Unique tool name (lowercase, underscores allowed)
            description: What the tool does - BE DESCRIPTIVE! LLMs read this.
            input_schema: JSON Schema for parameters
            handler: Function to call when tool is invoked
            requires_approval: Whether this tool needs human approval
            
        Raises:
            ValueError: If tool name is already registered
            
        Example:
            >>> registry.register(
            ...     name="get_weather",
            ...     description="Get current weather for a location",
            ...     input_schema={
            ...         "type": "object",
            ...         "properties": {
            ...             "city": {"type": "string", "description": "City name"}
            ...         },
            ...         "required": ["city"]
            ...     },
            ...     handler=get_weather_func
            ... )
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        
        # Validate schema has required structure
        if "type" not in input_schema:
            input_schema["type"] = "object"
        if "properties" not in input_schema:
            input_schema["properties"] = {}
        
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            requires_approval=requires_approval
        )
    
    def register_decorator(
        self,
        name: str,
        description: str,
        input_schema: dict,
        requires_approval: bool = False
    ) -> Callable:
        """
        Decorator for registering tools.
        
        A convenient way to register functions as tools.
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON Schema for parameters
            requires_approval: Whether approval is needed
            
        Returns:
            Decorator function
            
        Example:
            >>> @registry.register_decorator(
            ...     "multiply",
            ...     "Multiply two numbers together",
            ...     {
            ...         "type": "object",
            ...         "properties": {
            ...             "a": {"type": "number"},
            ...             "b": {"type": "number"}
            ...         },
            ...         "required": ["a", "b"]
            ...     }
            ... )
            ... def multiply(a: float, b: float) -> float:
            ...     return a * b
        """
        def decorator(func: Callable) -> Callable:
            self.register(name, description, input_schema, func, requires_approval)
            return func
        return decorator
    
    def unregister(self, name: str) -> None:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove
            
        Raises:
            KeyError: If tool is not registered
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        del self._tools[name]
    
    def get(self, name: str) -> ToolDefinition:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolDefinition for the requested tool
            
        Raises:
            KeyError: If tool is not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {self.list_names()}")
        return self._tools[name]
    
    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            name: Tool name to check
            
        Returns:
            True if tool exists, False otherwise
        """
        return name in self._tools
    
    def list_names(self) -> list[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_definitions_for_api(
        self,
        allowed_tools: list[str] | None = None
    ) -> list[dict]:
        """
        Get tool definitions in the format expected by the Claude API.
        
        Args:
            allowed_tools: Optional list to filter tools. None means all.
            
        Returns:
            List of tool definitions formatted for the API
            
        Example:
            >>> tools = registry.get_definitions_for_api()
            >>> # Use in API call:
            >>> response = client.messages.create(..., tools=tools)
        """
        definitions = []
        
        for name, tool in self._tools.items():
            if allowed_tools is not None and name not in allowed_tools:
                continue
            
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })
        
        return definitions
    
    def execute(self, name: str, arguments: dict) -> Any:
        """
        Execute a tool with the given arguments.
        
        Args:
            name: Tool name
            arguments: Tool arguments as a dictionary
            
        Returns:
            Tool execution result
            
        Raises:
            KeyError: If tool is not found
            Exception: Any exception raised by the tool handler
        """
        tool = self.get(name)
        return tool.handler(**arguments)
    
    def requires_approval(self, name: str) -> bool:
        """
        Check if a tool requires human approval.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool requires approval
        """
        return self.get(name).requires_approval
    
    def get_tool_info(self, name: str) -> dict:
        """
        Get detailed information about a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Dictionary with tool details
        """
        tool = self.get(name)
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "requires_approval": tool.requires_approval
        }
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator."""
        return name in self._tools
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ToolRegistry(tools={self.list_names()})"


# Common tool builders
def create_calculator_tool() -> tuple[str, str, dict, Callable]:
    """
    Create a calculator tool definition.
    
    Returns:
        Tuple of (name, description, schema, handler)
    """
    def safe_calculate(expression: str) -> str:
        """Safely evaluate a math expression."""
        # Only allow safe characters
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters"
        
        try:
            # Evaluate with restricted globals
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return (
        "calculator",
        "Perform mathematical calculations. Supports +, -, *, /, and parentheses.",
        {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '(3 + 4) * 2')"
                }
            },
            "required": ["expression"]
        },
        safe_calculate
    )


def create_datetime_tool() -> tuple[str, str, dict, Callable]:
    """
    Create a datetime tool definition.
    
    Returns:
        Tuple of (name, description, schema, handler)
    """
    from datetime import datetime
    
    def get_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Get current date/time."""
        try:
            return datetime.now().strftime(format)
        except Exception as e:
            return f"Error: {str(e)}"
    
    return (
        "get_datetime",
        "Get the current date and time. Can specify format string.",
        {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Python strftime format string (default: '%Y-%m-%d %H:%M:%S')"
                }
            }
        },
        get_datetime
    )


if __name__ == "__main__":
    # Demonstrate tool registry usage
    print("=== ToolRegistry Demonstration ===\n")
    
    # Create registry
    registry = ToolRegistry()
    
    # Register tools using the builder functions
    calc_name, calc_desc, calc_schema, calc_handler = create_calculator_tool()
    registry.register(calc_name, calc_desc, calc_schema, calc_handler)
    
    dt_name, dt_desc, dt_schema, dt_handler = create_datetime_tool()
    registry.register(dt_name, dt_desc, dt_schema, dt_handler)
    
    # Register using decorator
    @registry.register_decorator(
        "greet",
        "Generate a greeting message",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"}
            },
            "required": ["name"]
        }
    )
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    # Show registered tools
    print(f"Registry: {registry}")
    print(f"Number of tools: {len(registry)}")
    
    # Execute tools
    print("\n=== Tool Execution ===")
    print(f"Calculator: {registry.execute('calculator', {'expression': '(10 + 5) * 2'})}")
    print(f"DateTime: {registry.execute('get_datetime', {})}")
    print(f"Greet: {registry.execute('greet', {'name': 'World'})}")
    
    # Get API format
    print("\n=== API Format ===")
    api_tools = registry.get_definitions_for_api()
    for tool in api_tools:
        print(f"  {tool['name']}: {tool['description'][:50]}...")
