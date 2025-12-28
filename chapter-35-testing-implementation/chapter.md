---
chapter: 35
title: "Testing AI Agents - Implementation"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 35: Testing AI Agents - Implementation

## Introduction

In Chapter 34, we explored the philosophy behind testing AI agents—why it's challenging, what makes agent testing different from traditional software testing, and the types of tests we need. Now it's time to roll up our sleeves and write actual tests.

This chapter is intensely practical. We'll build a complete test suite for an AI agent, starting with simple unit tests for individual tools and working up to integration tests for the full agentic loop. You'll learn how to mock LLM responses to create deterministic tests, use property-based testing to catch edge cases, and set up continuous evaluation in your CI/CD pipeline.

By the end of this chapter, you'll have a reusable testing infrastructure that you can apply to any agent you build.

## Learning Objectives

By the end of this chapter, you will be able to:

- Write unit tests for agent tools that run without API calls
- Create mock LLM responses for deterministic agent testing
- Test the agentic loop with controlled inputs and outputs
- Use property-based testing to discover edge cases in agent behavior
- Set up a complete test suite with pytest fixtures and helpers
- Integrate agent tests into a CI/CD pipeline

## Prerequisites

Before diving in, make sure you have:

- Completed Chapters 33-34 (the Agent class and testing philosophy)
- pytest installed (`uv add pytest pytest-asyncio hypothesis`)
- A working Agent class from Chapter 33

Let's also install the testing dependencies we'll need:

```bash
uv add pytest pytest-asyncio hypothesis pytest-cov
```

## Setting Up the Test Environment

Before writing tests, let's establish a solid foundation. We'll create a `tests/` directory with proper configuration and shared fixtures.

### Project Structure

```
your-agent-project/
├── src/
│   └── agents/
│       ├── core/
│       │   ├── agent.py
│       │   └── augmented_llm.py
│       └── tools/
│           ├── calculator.py
│           └── weather.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures
│   ├── test_tools.py        # Tool unit tests
│   ├── test_agent_loop.py   # Agentic loop tests
│   └── test_integration.py  # End-to-end tests
├── pyproject.toml
└── .env
```

### The conftest.py File

The `conftest.py` file contains fixtures that are shared across all test files. Let's create a comprehensive one:

```python
"""
Shared pytest fixtures for agent testing.

Chapter 35: Testing AI Agents - Implementation
"""

import os
import pytest
from unittest.mock import Mock, MagicMock
from typing import Generator, Any

# Ensure we don't accidentally make real API calls during tests
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"


@pytest.fixture
def mock_anthropic_client() -> Mock:
    """
    Creates a mock Anthropic client that returns controlled responses.
    
    This fixture prevents any real API calls during testing.
    """
    client = Mock()
    client.messages = Mock()
    client.messages.create = Mock()
    return client


@pytest.fixture
def simple_text_response() -> dict:
    """
    Returns a mock API response containing only text (no tool calls).
    
    Use this for testing basic conversation flow.
    """
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a test response from Claude."
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 15
        }
    }


@pytest.fixture
def tool_call_response() -> dict:
    """
    Returns a mock API response containing a tool call.
    
    Use this for testing tool execution flow.
    """
    return {
        "id": "msg_test456",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01ABC123",
                "name": "calculator",
                "input": {
                    "operation": "add",
                    "a": 5,
                    "b": 3
                }
            }
        ],
        "model": "claude-sonnet-4-20250514",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 25,
            "output_tokens": 30
        }
    }


@pytest.fixture
def calculator_tool() -> dict:
    """
    Returns a simple calculator tool definition for testing.
    """
    return {
        "name": "calculator",
        "description": "Performs basic arithmetic operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "The first operand"
                },
                "b": {
                    "type": "number",
                    "description": "The second operand"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }


@pytest.fixture
def sample_conversation_history() -> list[dict]:
    """
    Returns a sample conversation history for testing multi-turn interactions.
    """
    return [
        {"role": "user", "content": "What is 5 plus 3?"},
        {"role": "assistant", "content": "5 plus 3 equals 8."},
        {"role": "user", "content": "Now multiply that by 2."},
    ]


class MockResponse:
    """
    A class that mimics the Anthropic API response structure.
    
    This allows us to create responses that behave like real API responses
    but with controlled content.
    """
    
    def __init__(self, content: list, stop_reason: str = "end_turn"):
        self.id = "msg_mock"
        self.type = "message"
        self.role = "assistant"
        self.content = [self._make_content_block(c) for c in content]
        self.model = "claude-sonnet-4-20250514"
        self.stop_reason = stop_reason
        self.usage = Mock(input_tokens=10, output_tokens=20)
    
    def _make_content_block(self, content: dict) -> Any:
        """Convert a dict to a mock content block with attribute access."""
        block = Mock()
        for key, value in content.items():
            setattr(block, key, value)
        return block


@pytest.fixture
def mock_response_factory():
    """
    Factory fixture for creating mock responses with custom content.
    
    Usage:
        response = mock_response_factory([
            {"type": "text", "text": "Hello!"}
        ])
    """
    def _create_response(content: list, stop_reason: str = "end_turn"):
        return MockResponse(content, stop_reason)
    
    return _create_response
```

This `conftest.py` provides the foundation for all our tests. The fixtures are automatically available to any test file in the `tests/` directory.

## Testing Tools in Isolation

Tools are the easiest part of an agent to test because they're typically pure functions—given the same input, they produce the same output. Let's start here.

### The Calculator Tool

First, let's define a simple calculator tool that we'll test:

```python
"""
Calculator tool implementation for testing examples.

Chapter 35: Testing AI Agents - Implementation
"""

from typing import Union


def calculator(operation: str, a: float, b: float) -> dict:
    """
    Performs basic arithmetic operations.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First operand
        b: Second operand
        
    Returns:
        A dict containing the result or an error message
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }
    
    if operation not in operations:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Valid operations: {list(operations.keys())}"
        }
    
    if operation == "divide" and b == 0:
        return {
            "success": False,
            "error": "Cannot divide by zero"
        }
    
    result = operations[operation](a, b)
    
    return {
        "success": True,
        "result": result,
        "expression": f"{a} {operation} {b} = {result}"
    }
```

### Writing Tool Tests

Now let's write comprehensive tests for this tool:

```python
"""
Unit tests for the calculator tool.

Chapter 35: Testing AI Agents - Implementation

Run with: pytest tests/test_tools.py -v
"""

import pytest
from tools.calculator import calculator


class TestCalculatorBasicOperations:
    """Tests for basic arithmetic operations."""
    
    def test_addition(self):
        """Test that addition works correctly."""
        result = calculator("add", 5, 3)
        
        assert result["success"] is True
        assert result["result"] == 8
        assert "5 add 3 = 8" in result["expression"]
    
    def test_subtraction(self):
        """Test that subtraction works correctly."""
        result = calculator("subtract", 10, 4)
        
        assert result["success"] is True
        assert result["result"] == 6
    
    def test_multiplication(self):
        """Test that multiplication works correctly."""
        result = calculator("multiply", 7, 6)
        
        assert result["success"] is True
        assert result["result"] == 42
    
    def test_division(self):
        """Test that division works correctly."""
        result = calculator("divide", 20, 4)
        
        assert result["success"] is True
        assert result["result"] == 5.0


class TestCalculatorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_divide_by_zero(self):
        """Test that division by zero returns an error."""
        result = calculator("divide", 10, 0)
        
        assert result["success"] is False
        assert "divide by zero" in result["error"].lower()
    
    def test_unknown_operation(self):
        """Test that unknown operations return an error."""
        result = calculator("power", 2, 3)
        
        assert result["success"] is False
        assert "Unknown operation" in result["error"]
    
    def test_negative_numbers(self):
        """Test that negative numbers work correctly."""
        result = calculator("add", -5, -3)
        
        assert result["success"] is True
        assert result["result"] == -8
    
    def test_floating_point_numbers(self):
        """Test that floating point numbers work correctly."""
        result = calculator("multiply", 2.5, 4.0)
        
        assert result["success"] is True
        assert result["result"] == 10.0
    
    def test_very_large_numbers(self):
        """Test that very large numbers are handled."""
        result = calculator("multiply", 1e100, 1e100)
        
        assert result["success"] is True
        assert result["result"] == 1e200


class TestCalculatorWithParameterization:
    """
    Demonstrates pytest parameterization for testing multiple inputs.
    
    This is a powerful technique for testing many cases without writing
    repetitive test methods.
    """
    
    @pytest.mark.parametrize("operation,a,b,expected", [
        ("add", 1, 1, 2),
        ("add", 0, 0, 0),
        ("add", -1, 1, 0),
        ("subtract", 5, 3, 2),
        ("subtract", 3, 5, -2),
        ("multiply", 3, 4, 12),
        ("multiply", 0, 100, 0),
        ("divide", 10, 2, 5),
        ("divide", 7, 2, 3.5),
    ])
    def test_operations(self, operation: str, a: float, b: float, expected: float):
        """Test various operation combinations."""
        result = calculator(operation, a, b)
        
        assert result["success"] is True
        assert result["result"] == expected
```

> **💡 Tip:** Use `pytest.mark.parametrize` to test many input combinations without writing separate test methods. This makes your tests more maintainable and easier to extend.

### Testing Tool Error Handling

Tools should fail gracefully. Let's add tests specifically for error conditions:

```python
class TestCalculatorErrorMessages:
    """Tests that verify error messages are helpful."""
    
    def test_error_includes_valid_operations(self):
        """Verify error messages tell users what operations ARE valid."""
        result = calculator("modulo", 10, 3)
        
        assert result["success"] is False
        # Error should list valid operations
        assert "add" in result["error"]
        assert "subtract" in result["error"]
        assert "multiply" in result["error"]
        assert "divide" in result["error"]
    
    def test_error_messages_are_clear(self):
        """Verify error messages are human-readable."""
        result = calculator("divide", 5, 0)
        
        # Message should be a complete sentence, not a code
        assert result["success"] is False
        assert len(result["error"]) > 10  # Not just an error code
        assert result["error"][0].isupper()  # Starts with capital
```

## Mock LLM Responses for Deterministic Tests

Real LLM calls are slow, expensive, and non-deterministic. For unit and integration tests, we mock the LLM to return controlled responses.

### Creating a Mock LLM

Here's a flexible mock LLM class that can simulate various response patterns:

```python
"""
Mock LLM for deterministic agent testing.

Chapter 35: Testing AI Agents - Implementation
"""

from typing import Any, Callable
from unittest.mock import Mock
from dataclasses import dataclass, field


@dataclass
class MockContentBlock:
    """Represents a content block in an API response."""
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass  
class MockMessage:
    """Represents a complete API response message."""
    id: str = "msg_mock"
    type: str = "message"
    role: str = "assistant"
    content: list = field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    
    @property
    def usage(self):
        mock_usage = Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        return mock_usage


class MockLLM:
    """
    A mock LLM that returns predetermined responses.
    
    This class simulates the Anthropic API for testing purposes,
    allowing you to control exactly what the "LLM" returns.
    
    Usage:
        mock = MockLLM()
        mock.add_response("What is 2+2?", "2 plus 2 equals 4.")
        
        # In your test:
        response = mock.create_message(messages=[...])
    """
    
    def __init__(self):
        self.responses: list[MockMessage] = []
        self.response_index: int = 0
        self.call_history: list[dict] = []
        self.pattern_responses: list[tuple[Callable, MockMessage]] = []
    
    def add_response(
        self,
        text: str | None = None,
        tool_call: dict | None = None,
        stop_reason: str = "end_turn"
    ) -> "MockLLM":
        """
        Add a response to the queue of responses.
        
        Args:
            text: Text content to return
            tool_call: Tool call to include (name, id, input)
            stop_reason: The stop reason (end_turn, tool_use, etc.)
            
        Returns:
            self, for method chaining
        """
        content = []
        
        if text:
            content.append(MockContentBlock(type="text", text=text))
        
        if tool_call:
            content.append(MockContentBlock(
                type="tool_use",
                id=tool_call.get("id", "toolu_mock"),
                name=tool_call["name"],
                input=tool_call["input"]
            ))
            stop_reason = "tool_use"
        
        self.responses.append(MockMessage(
            content=content,
            stop_reason=stop_reason
        ))
        
        return self
    
    def add_pattern_response(
        self,
        pattern_matcher: Callable[[list[dict]], bool],
        response: MockMessage
    ) -> "MockLLM":
        """
        Add a response that triggers when a pattern matches.
        
        Args:
            pattern_matcher: A function that takes messages and returns True if matched
            response: The response to return when matched
            
        This is useful for testing branching behavior.
        """
        self.pattern_responses.append((pattern_matcher, response))
        return self
    
    def create_message(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        messages: list[dict] = None,
        system: str = None,
        tools: list[dict] = None,
        **kwargs
    ) -> MockMessage:
        """
        Simulates the Anthropic messages.create() API call.
        
        Records the call and returns the next queued response.
        """
        # Record this call for later inspection
        self.call_history.append({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages or [],
            "system": system,
            "tools": tools,
            "kwargs": kwargs
        })
        
        # Check pattern-based responses first
        for matcher, response in self.pattern_responses:
            if matcher(messages or []):
                return response
        
        # Return next queued response
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        
        # Default response if queue is empty
        return MockMessage(
            content=[MockContentBlock(type="text", text="No more mock responses available.")]
        )
    
    def reset(self):
        """Reset the mock to its initial state."""
        self.response_index = 0
        self.call_history.clear()
    
    def assert_called_with_tool(self, tool_name: str) -> bool:
        """Verify that the LLM was called with a specific tool available."""
        for call in self.call_history:
            tools = call.get("tools", [])
            for tool in tools or []:
                if tool.get("name") == tool_name:
                    return True
        return False
    
    def get_last_messages(self) -> list[dict]:
        """Get the messages from the most recent call."""
        if self.call_history:
            return self.call_history[-1].get("messages", [])
        return []


# Helper function for creating tool use responses
def create_tool_response(
    tool_name: str,
    tool_input: dict,
    tool_id: str = "toolu_test"
) -> MockMessage:
    """
    Helper function to create a mock tool use response.
    
    Args:
        tool_name: Name of the tool being called
        tool_input: Arguments for the tool
        tool_id: ID for the tool use block
        
    Returns:
        A MockMessage configured for tool use
    """
    return MockMessage(
        content=[
            MockContentBlock(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input
            )
        ],
        stop_reason="tool_use"
    )


# Helper function for creating text responses
def create_text_response(text: str) -> MockMessage:
    """
    Helper function to create a mock text response.
    
    Args:
        text: The text content to return
        
    Returns:
        A MockMessage with text content
    """
    return MockMessage(
        content=[MockContentBlock(type="text", text=text)],
        stop_reason="end_turn"
    )
```

### Using the Mock LLM in Tests

Now let's write tests that use our mock:

```python
"""
Tests demonstrating mock LLM usage.

Chapter 35: Testing AI Agents - Implementation
"""

import pytest
from mock_llm import MockLLM, create_tool_response, create_text_response


class TestMockLLMBasics:
    """Basic tests for the MockLLM class."""
    
    def test_returns_queued_responses_in_order(self):
        """Verify responses are returned in the order they were added."""
        mock = MockLLM()
        mock.add_response(text="First response")
        mock.add_response(text="Second response")
        mock.add_response(text="Third response")
        
        # Each call should return the next response
        r1 = mock.create_message(messages=[])
        assert r1.content[0].text == "First response"
        
        r2 = mock.create_message(messages=[])
        assert r2.content[0].text == "Second response"
        
        r3 = mock.create_message(messages=[])
        assert r3.content[0].text == "Third response"
    
    def test_records_call_history(self):
        """Verify the mock records all calls made to it."""
        mock = MockLLM()
        mock.add_response(text="Response")
        
        mock.create_message(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful."
        )
        
        assert len(mock.call_history) == 1
        assert mock.call_history[0]["messages"][0]["content"] == "Hello"
        assert mock.call_history[0]["system"] == "You are helpful."
    
    def test_tool_call_response(self):
        """Verify tool call responses are structured correctly."""
        mock = MockLLM()
        mock.add_response(
            tool_call={
                "name": "calculator",
                "input": {"operation": "add", "a": 5, "b": 3}
            }
        )
        
        response = mock.create_message(messages=[])
        
        assert response.stop_reason == "tool_use"
        assert response.content[0].type == "tool_use"
        assert response.content[0].name == "calculator"
        assert response.content[0].input["operation"] == "add"


class TestMockLLMWithAgent:
    """Tests showing how to use MockLLM with an agent."""
    
    def test_agent_uses_tool_when_appropriate(self):
        """
        Test that an agent correctly invokes a tool when the LLM requests it.
        
        This simulates a multi-turn interaction:
        1. User asks a question
        2. LLM requests a tool call
        3. Tool is executed
        4. LLM provides final answer
        """
        mock = MockLLM()
        
        # First response: LLM decides to use the calculator
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_123",
                "input": {"operation": "add", "a": 5, "b": 3}
            }
        )
        
        # Second response: LLM provides the answer
        mock.add_response(text="5 plus 3 equals 8.")
        
        # Simulate the agent loop
        messages = [{"role": "user", "content": "What is 5 + 3?"}]
        
        # First call - should get tool use
        response = mock.create_message(messages=messages)
        assert response.stop_reason == "tool_use"
        
        # Agent would execute the tool and add result to messages
        # Then make second call
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "8"}]
        })
        
        # Second call - should get final answer
        response = mock.create_message(messages=messages)
        assert response.stop_reason == "end_turn"
        assert "8" in response.content[0].text
```

## Testing the Agentic Loop

The agentic loop is the heart of your agent. Testing it requires simulating multi-turn conversations with tool calls.

### A Minimal Agent for Testing

Let's create a simplified agent class that we can test:

```python
"""
A minimal agent class designed for testing.

Chapter 35: Testing AI Agents - Implementation
"""

from typing import Any, Callable
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for the test agent."""
    max_iterations: int = 10
    system_prompt: str = "You are a helpful assistant."


class TestableAgent:
    """
    A minimal agent implementation designed for testing.
    
    This agent demonstrates the core loop without external dependencies,
    making it easy to test in isolation.
    """
    
    def __init__(
        self,
        llm_client: Any,
        tools: dict[str, Callable] = None,
        tool_definitions: list[dict] = None,
        config: AgentConfig = None
    ):
        self.llm = llm_client
        self.tools = tools or {}
        self.tool_definitions = tool_definitions or []
        self.config = config or AgentConfig()
        self.conversation_history: list[dict] = []
        self.tool_call_log: list[dict] = []
    
    def run(self, user_message: str) -> str:
        """
        Run the agent with a user message.
        
        Args:
            user_message: The user's input
            
        Returns:
            The agent's final response text
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        for iteration in range(self.config.max_iterations):
            response = self._call_llm()
            
            # Check if the LLM wants to use a tool
            if response.stop_reason == "tool_use":
                self._handle_tool_calls(response)
            else:
                # Extract and return the text response
                for block in response.content:
                    if hasattr(block, 'type') and block.type == "text":
                        return block.text
                return ""
        
        raise RuntimeError(f"Agent exceeded maximum iterations ({self.config.max_iterations})")
    
    def _call_llm(self) -> Any:
        """Make a call to the LLM."""
        return self.llm.create_message(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=self.conversation_history,
            system=self.config.system_prompt,
            tools=self.tool_definitions if self.tool_definitions else None
        )
    
    def _handle_tool_calls(self, response: Any) -> None:
        """Process tool calls from the LLM response."""
        # Add assistant message to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        # Process each tool call
        tool_results = []
        for block in response.content:
            if hasattr(block, 'type') and block.type == "tool_use":
                result = self._execute_tool(block.name, block.input, block.id)
                tool_results.append(result)
        
        # Add tool results to history
        self.conversation_history.append({
            "role": "user",
            "content": tool_results
        })
    
    def _execute_tool(self, name: str, inputs: dict, tool_id: str) -> dict:
        """Execute a tool and return the result."""
        self.tool_call_log.append({
            "name": name,
            "inputs": inputs,
            "id": tool_id
        })
        
        if name not in self.tools:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Error: Unknown tool '{name}'",
                "is_error": True
            }
        
        try:
            result = self.tools[name](**inputs)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result)
            }
        except Exception as e:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Error executing tool: {str(e)}",
                "is_error": True
            }
```

### Testing the Agent Loop

Now let's write tests for the agent:

```python
"""
Tests for the agentic loop.

Chapter 35: Testing AI Agents - Implementation
"""

import pytest
from testable_agent import TestableAgent, AgentConfig
from mock_llm import MockLLM, MockContentBlock, MockMessage


def simple_calculator(operation: str, a: float, b: float) -> float:
    """Simple calculator for testing."""
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y
    }
    return ops[operation](a, b)


class TestAgentBasicResponses:
    """Tests for basic agent responses without tool use."""
    
    def test_agent_returns_text_response(self):
        """Verify agent returns text when LLM doesn't use tools."""
        mock = MockLLM()
        mock.add_response(text="Hello! How can I help you today?")
        
        agent = TestableAgent(llm_client=mock)
        response = agent.run("Hello!")
        
        assert response == "Hello! How can I help you today?"
    
    def test_agent_maintains_conversation_history(self):
        """Verify agent properly maintains conversation history."""
        mock = MockLLM()
        mock.add_response(text="I'm doing well!")
        mock.add_response(text="The weather is nice.")
        
        agent = TestableAgent(llm_client=mock)
        
        agent.run("How are you?")
        agent.run("What's the weather like?")
        
        # Check history includes both exchanges
        assert len(agent.conversation_history) == 4  # 2 user + 2 assistant
        assert agent.conversation_history[0]["content"] == "How are you?"
        assert agent.conversation_history[2]["content"] == "What's the weather like?"


class TestAgentToolUse:
    """Tests for agent tool use."""
    
    @pytest.fixture
    def agent_with_calculator(self):
        """Create an agent with a calculator tool."""
        mock = MockLLM()
        
        tools = {"calculator": simple_calculator}
        tool_definitions = [{
            "name": "calculator",
            "description": "Performs arithmetic",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }]
        
        return TestableAgent(
            llm_client=mock,
            tools=tools,
            tool_definitions=tool_definitions
        ), mock
    
    def test_agent_executes_tool_call(self, agent_with_calculator):
        """Verify agent executes tool when LLM requests it."""
        agent, mock = agent_with_calculator
        
        # First response: tool call
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_1",
                "input": {"operation": "add", "a": 5, "b": 3}
            }
        )
        # Second response: final answer
        mock.add_response(text="The answer is 8.")
        
        response = agent.run("What is 5 + 3?")
        
        assert response == "The answer is 8."
        assert len(agent.tool_call_log) == 1
        assert agent.tool_call_log[0]["name"] == "calculator"
    
    def test_agent_handles_multiple_tool_calls(self, agent_with_calculator):
        """Verify agent handles multiple sequential tool calls."""
        agent, mock = agent_with_calculator
        
        # First tool call
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_1",
                "input": {"operation": "add", "a": 5, "b": 3}
            }
        )
        # Second tool call
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_2",
                "input": {"operation": "multiply", "a": 8, "b": 2}
            }
        )
        # Final answer
        mock.add_response(text="5 + 3 = 8, and 8 × 2 = 16.")
        
        response = agent.run("What is (5 + 3) × 2?")
        
        assert "16" in response
        assert len(agent.tool_call_log) == 2
    
    def test_agent_handles_unknown_tool(self):
        """Verify agent handles gracefully when LLM requests unknown tool."""
        mock = MockLLM()
        mock.add_response(
            tool_call={
                "name": "nonexistent_tool",
                "id": "toolu_1",
                "input": {}
            }
        )
        mock.add_response(text="I encountered an error with that tool.")
        
        agent = TestableAgent(llm_client=mock, tools={})
        response = agent.run("Use the magic tool")
        
        # Agent should continue despite the error
        assert "error" in response.lower()


class TestAgentGuardrails:
    """Tests for agent safety and guardrails."""
    
    def test_agent_respects_max_iterations(self):
        """Verify agent stops after max iterations."""
        mock = MockLLM()
        
        # Add responses that would cause infinite tool calls
        for _ in range(20):
            mock.add_response(
                tool_call={
                    "name": "calculator",
                    "id": "toolu_loop",
                    "input": {"operation": "add", "a": 1, "b": 1}
                }
            )
        
        agent = TestableAgent(
            llm_client=mock,
            tools={"calculator": simple_calculator},
            config=AgentConfig(max_iterations=5)
        )
        
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            agent.run("Keep calculating forever")
    
    def test_agent_handles_tool_exceptions(self):
        """Verify agent handles exceptions from tools gracefully."""
        mock = MockLLM()
        mock.add_response(
            tool_call={
                "name": "failing_tool",
                "id": "toolu_1",
                "input": {}
            }
        )
        mock.add_response(text="The tool failed, but I can help anyway.")
        
        def failing_tool():
            raise ValueError("This tool always fails!")
        
        agent = TestableAgent(
            llm_client=mock,
            tools={"failing_tool": failing_tool}
        )
        
        # Should not raise, should handle gracefully
        response = agent.run("Use the failing tool")
        assert response is not None
```

## Property-Based Testing for Agents

Property-based testing uses the Hypothesis library to generate random inputs and verify that certain properties always hold. This is powerful for finding edge cases.

### Introduction to Hypothesis

```python
"""
Property-based testing for agent tools using Hypothesis.

Chapter 35: Testing AI Agents - Implementation

Property-based testing generates random inputs to verify that
certain properties ALWAYS hold, regardless of the specific input.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from tools.calculator import calculator


class TestCalculatorProperties:
    """Property-based tests for the calculator."""
    
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    def test_addition_is_commutative(self, a: float, b: float):
        """
        Property: Addition is commutative (a + b == b + a)
        
        Hypothesis will generate many random float pairs to verify this.
        """
        result_ab = calculator("add", a, b)
        result_ba = calculator("add", b, a)
        
        assert result_ab["success"] == result_ba["success"]
        if result_ab["success"]:
            assert abs(result_ab["result"] - result_ba["result"]) < 1e-10
    
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    def test_multiplication_is_commutative(self, a: float, b: float):
        """Property: Multiplication is commutative (a × b == b × a)"""
        result_ab = calculator("multiply", a, b)
        result_ba = calculator("multiply", b, a)
        
        assert result_ab["success"] == result_ba["success"]
        if result_ab["success"]:
            assert abs(result_ab["result"] - result_ba["result"]) < 1e-10
    
    @given(a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_adding_zero_returns_same_number(self, a: float):
        """Property: Adding zero gives the same number (a + 0 == a)"""
        result = calculator("add", a, 0)
        
        assert result["success"] is True
        assert abs(result["result"] - a) < 1e-10
    
    @given(a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_multiplying_by_one_returns_same_number(self, a: float):
        """Property: Multiplying by one gives the same number (a × 1 == a)"""
        result = calculator("multiply", a, 1)
        
        assert result["success"] is True
        assert abs(result["result"] - a) < 1e-10
    
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    def test_subtraction_inverse_of_addition(self, a: float, b: float):
        """Property: (a + b) - b == a"""
        add_result = calculator("add", a, b)
        assume(add_result["success"])  # Skip if addition fails
        
        sub_result = calculator("subtract", add_result["result"], b)
        
        assert sub_result["success"] is True
        assert abs(sub_result["result"] - a) < 1e-9
    
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=1, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=1, max_value=1e10)
    )
    def test_division_inverse_of_multiplication(self, a: float, b: float):
        """Property: (a × b) / b == a"""
        mul_result = calculator("multiply", a, b)
        assume(mul_result["success"])
        
        div_result = calculator("divide", mul_result["result"], b)
        
        assert div_result["success"] is True
        assert abs(div_result["result"] - a) < 1e-6  # Allow for floating point error
    
    @given(operation=st.text(min_size=1, max_size=20))
    def test_unknown_operation_always_fails(self, operation: str):
        """Property: Unknown operations always return failure."""
        # Skip valid operations
        assume(operation not in ["add", "subtract", "multiply", "divide"])
        
        result = calculator(operation, 1, 1)
        
        assert result["success"] is False
        assert "error" in result
    
    @given(
        operation=st.sampled_from(["add", "subtract", "multiply", "divide"]),
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    @settings(max_examples=200)  # Run more examples for better coverage
    def test_result_always_has_expected_structure(
        self, operation: str, a: float, b: float
    ):
        """Property: Result always has success key and either result or error."""
        result = calculator(operation, a, b)
        
        # Always has success key
        assert "success" in result
        assert isinstance(result["success"], bool)
        
        # Has result if success, error if not
        if result["success"]:
            assert "result" in result
        else:
            assert "error" in result


class TestAgentPropertyTests:
    """Property-based tests for agent behavior."""
    
    @given(message=st.text(min_size=1, max_size=1000))
    def test_agent_always_returns_string(self, message: str):
        """Property: Agent always returns a string response."""
        from mock_llm import MockLLM
        from testable_agent import TestableAgent
        
        mock = MockLLM()
        mock.add_response(text="Response to: " + message[:50])
        
        agent = TestableAgent(llm_client=mock)
        response = agent.run(message)
        
        assert isinstance(response, str)
    
    @given(messages=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5))
    def test_conversation_history_grows_predictably(self, messages: list[str]):
        """Property: Conversation history length is predictable."""
        from mock_llm import MockLLM
        from testable_agent import TestableAgent
        
        mock = MockLLM()
        for _ in messages:
            mock.add_response(text="Response")
        
        agent = TestableAgent(llm_client=mock)
        
        for msg in messages:
            agent.run(msg)
        
        # Each exchange adds 2 messages (user + assistant)
        # But we only count user messages in this simple test
        user_messages = [m for m in agent.conversation_history if m["role"] == "user"]
        assert len(user_messages) == len(messages)
```

> **💡 Tip:** Property-based testing is especially valuable for finding edge cases like:
> - Empty strings
> - Very long inputs  
> - Special characters
> - Unicode edge cases
> - Numeric edge cases (very large, very small, negative, zero)

## Continuous Evaluation in CI/CD

Now let's set up tests to run automatically in a CI/CD pipeline.

### pytest Configuration

Create a `pytest.ini` or add to `pyproject.toml`:

```toml
# pyproject.toml

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "api: marks tests that require real API calls",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:

```yaml
# .github/workflows/tests.yml
name: Agent Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run unit tests
        run: uv run pytest tests/ -m "not integration and not api" -v
      
      - name: Run integration tests
        run: uv run pytest tests/ -m "integration" -v
      
      - name: Generate coverage report
        run: uv run pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml

  api-tests:
    runs-on: ubuntu-latest
    # Only run on main branch, not PRs (to protect API key)
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run API tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: uv run pytest tests/ -m "api" -v --tb=long
```

### Test Markers and Organization

Use markers to categorize tests:

```python
"""
Example of using test markers for organization.

Chapter 35: Testing AI Agents - Implementation
"""

import pytest


class TestToolUnit:
    """Unit tests that run without any external dependencies."""
    
    @pytest.mark.unit
    def test_calculator_addition(self):
        """Unit test for calculator."""
        from tools.calculator import calculator
        result = calculator("add", 1, 2)
        assert result["result"] == 3


class TestAgentIntegration:
    """Integration tests using mocks."""
    
    @pytest.mark.integration
    def test_agent_with_multiple_tools(self):
        """Integration test for agent with tools."""
        # ... test implementation
        pass


class TestRealAPI:
    """Tests that make real API calls."""
    
    @pytest.mark.api
    @pytest.mark.slow
    def test_real_claude_response(self):
        """
        Test with real API.
        
        Only runs in CI on main branch or locally with API key.
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key or api_key == "test-key-not-real":
            pytest.skip("No real API key available")
        
        # ... actual API test
        pass
```

Run specific test categories:

```bash
# Run only unit tests (fast, no external dependencies)
pytest -m unit

# Run integration tests (uses mocks)
pytest -m integration

# Run everything except API tests
pytest -m "not api"

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Creating a Complete Test Suite

Let's put it all together in a complete, organized test suite:

```python
"""
Complete test suite for the AI agent.

Chapter 35: Testing AI Agents - Implementation

This file demonstrates how to organize a comprehensive test suite
with proper setup, teardown, and test organization.
"""

import pytest
import os
from unittest.mock import patch
from typing import Generator

# Import test utilities
from mock_llm import MockLLM, create_tool_response, create_text_response
from testable_agent import TestableAgent, AgentConfig
from tools.calculator import calculator


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def shared_tool_definitions() -> list[dict]:
    """Tool definitions shared across all tests in the session."""
    return [
        {
            "name": "calculator",
            "description": "Performs arithmetic operations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    ]


@pytest.fixture
def mock_llm() -> MockLLM:
    """Fresh MockLLM for each test."""
    return MockLLM()


@pytest.fixture
def agent_factory(shared_tool_definitions):
    """Factory for creating agents with common configuration."""
    def _create_agent(
        mock: MockLLM,
        include_tools: bool = True,
        max_iterations: int = 10
    ) -> TestableAgent:
        tools = {"calculator": calculator} if include_tools else {}
        tool_defs = shared_tool_definitions if include_tools else []
        
        return TestableAgent(
            llm_client=mock,
            tools=tools,
            tool_definitions=tool_defs,
            config=AgentConfig(max_iterations=max_iterations)
        )
    
    return _create_agent


# =============================================================================
# Unit Tests: Tools
# =============================================================================

class TestCalculatorTool:
    """Unit tests for the calculator tool."""
    
    @pytest.mark.unit
    class TestBasicOperations:
        """Test basic arithmetic operations."""
        
        def test_add(self):
            assert calculator("add", 2, 3)["result"] == 5
        
        def test_subtract(self):
            assert calculator("subtract", 5, 3)["result"] == 2
        
        def test_multiply(self):
            assert calculator("multiply", 4, 3)["result"] == 12
        
        def test_divide(self):
            assert calculator("divide", 10, 2)["result"] == 5.0
    
    @pytest.mark.unit
    class TestEdgeCases:
        """Test edge cases and error conditions."""
        
        def test_divide_by_zero_returns_error(self):
            result = calculator("divide", 5, 0)
            assert result["success"] is False
            assert "zero" in result["error"].lower()
        
        def test_invalid_operation_returns_error(self):
            result = calculator("power", 2, 3)
            assert result["success"] is False
        
        def test_negative_numbers(self):
            assert calculator("add", -5, 3)["result"] == -2
        
        def test_decimal_numbers(self):
            result = calculator("multiply", 0.5, 4)
            assert result["result"] == pytest.approx(2.0)


# =============================================================================
# Unit Tests: Mock LLM
# =============================================================================

class TestMockLLM:
    """Unit tests for the MockLLM class."""
    
    @pytest.mark.unit
    def test_returns_responses_in_order(self, mock_llm):
        mock_llm.add_response(text="First")
        mock_llm.add_response(text="Second")
        
        r1 = mock_llm.create_message(messages=[])
        r2 = mock_llm.create_message(messages=[])
        
        assert r1.content[0].text == "First"
        assert r2.content[0].text == "Second"
    
    @pytest.mark.unit
    def test_records_call_history(self, mock_llm):
        mock_llm.add_response(text="Response")
        
        mock_llm.create_message(
            messages=[{"role": "user", "content": "Test"}],
            system="System prompt"
        )
        
        assert len(mock_llm.call_history) == 1
        assert mock_llm.call_history[0]["system"] == "System prompt"
    
    @pytest.mark.unit
    def test_reset_clears_state(self, mock_llm):
        mock_llm.add_response(text="Response")
        mock_llm.create_message(messages=[])
        
        mock_llm.reset()
        
        assert mock_llm.response_index == 0
        assert len(mock_llm.call_history) == 0


# =============================================================================
# Integration Tests: Agent
# =============================================================================

class TestAgentIntegration:
    """Integration tests for the agent."""
    
    @pytest.mark.integration
    def test_simple_conversation(self, mock_llm, agent_factory):
        mock_llm.add_response(text="Hello! I'm here to help.")
        agent = agent_factory(mock_llm, include_tools=False)
        
        response = agent.run("Hello!")
        
        assert "help" in response.lower()
    
    @pytest.mark.integration
    def test_tool_execution(self, mock_llm, agent_factory):
        mock_llm.add_response(
            tool_call={"name": "calculator", "id": "t1", "input": {"operation": "add", "a": 2, "b": 3}}
        )
        mock_llm.add_response(text="The sum is 5.")
        
        agent = agent_factory(mock_llm)
        response = agent.run("What is 2 + 3?")
        
        assert "5" in response
        assert len(agent.tool_call_log) == 1
    
    @pytest.mark.integration
    def test_multi_step_reasoning(self, mock_llm, agent_factory):
        # First tool call
        mock_llm.add_response(
            tool_call={"name": "calculator", "id": "t1", "input": {"operation": "add", "a": 10, "b": 5}}
        )
        # Second tool call
        mock_llm.add_response(
            tool_call={"name": "calculator", "id": "t2", "input": {"operation": "multiply", "a": 15, "b": 2}}
        )
        # Final answer
        mock_llm.add_response(text="10 + 5 = 15, then 15 × 2 = 30")
        
        agent = agent_factory(mock_llm)
        response = agent.run("What is (10 + 5) × 2?")
        
        assert "30" in response
        assert len(agent.tool_call_log) == 2
    
    @pytest.mark.integration
    def test_respects_max_iterations(self, mock_llm, agent_factory):
        # Create infinite loop of tool calls
        for _ in range(20):
            mock_llm.add_response(
                tool_call={"name": "calculator", "id": "t1", "input": {"operation": "add", "a": 1, "b": 1}}
            )
        
        agent = agent_factory(mock_llm, max_iterations=3)
        
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            agent.run("Loop forever")


# =============================================================================
# End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end tests simulating real user scenarios."""
    
    @pytest.mark.integration
    def test_math_homework_scenario(self, mock_llm, agent_factory):
        """Simulate a student asking for help with math."""
        # Setup responses
        mock_llm.add_response(
            tool_call={"name": "calculator", "id": "t1", "input": {"operation": "multiply", "a": 7, "b": 8}}
        )
        mock_llm.add_response(text="7 × 8 = 56. The answer is 56!")
        
        agent = agent_factory(mock_llm)
        response = agent.run("I need help with my homework. What is 7 times 8?")
        
        assert "56" in response
    
    @pytest.mark.integration  
    def test_graceful_error_handling(self, mock_llm, agent_factory):
        """Test that errors don't crash the agent."""
        # Tool that will fail
        mock_llm.add_response(
            tool_call={"name": "nonexistent", "id": "t1", "input": {}}
        )
        mock_llm.add_response(text="Sorry, I encountered an error.")
        
        agent = agent_factory(mock_llm)
        response = agent.run("Use the broken tool")
        
        # Should complete without raising
        assert response is not None


# =============================================================================
# Property-Based Tests
# =============================================================================

class TestPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("operation", ["add", "subtract", "multiply"])
    def test_valid_operations_always_succeed(self, operation):
        """All valid operations with valid numbers should succeed."""
        from hypothesis import given, strategies as st
        
        @given(
            a=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            b=st.floats(min_value=-1000, max_value=1000, allow_nan=False)
        )
        def check_operation(a, b):
            result = calculator(operation, a, b)
            assert result["success"] is True
        
        check_operation()


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not api",  # Skip API tests when running directly
    ])
```

## Common Pitfalls

### 1. Not Mocking External Dependencies

**Problem:** Tests make real API calls, causing flaky tests and unexpected costs.

```python
# ❌ Bad: Real API call in test
def test_agent():
    import anthropic
    client = anthropic.Anthropic()  # Real client!
    # This will fail without API key and cost money

# ✅ Good: Use mock
def test_agent(mock_llm):
    agent = TestableAgent(llm_client=mock_llm)
    # No real API calls
```

### 2. Testing Implementation Instead of Behavior

**Problem:** Tests break when you refactor internal code.

```python
# ❌ Bad: Testing internal implementation
def test_agent_internals():
    agent = Agent()
    assert agent._internal_counter == 0  # Testing private state
    agent.run("hi")
    assert agent._internal_counter == 1

# ✅ Good: Testing behavior
def test_agent_responds():
    agent = Agent()
    response = agent.run("hi")
    assert isinstance(response, str)
    assert len(response) > 0
```

### 3. Insufficient Test Isolation

**Problem:** Tests affect each other through shared state.

```python
# ❌ Bad: Shared state between tests
mock = MockLLM()  # Module-level shared state

def test_first():
    mock.add_response(text="First")
    # ...

def test_second():
    # This test is affected by test_first!
    # ...

# ✅ Good: Fresh fixtures for each test
@pytest.fixture
def mock_llm():
    return MockLLM()  # Fresh for each test

def test_first(mock_llm):
    mock_llm.add_response(text="First")

def test_second(mock_llm):  # Gets a fresh mock
    # Isolated from test_first
```

## Practical Exercise

**Task:** Create a test suite for a text classification tool.

Build and test a tool that classifies text into categories (positive, negative, neutral). Your tests should cover:

1. Unit tests for the classification function itself
2. Mock LLM tests simulating the agent calling the tool
3. Property-based tests for edge cases
4. Integration tests for the full agent workflow

**Requirements:**

- Use pytest fixtures effectively
- Include at least 10 unit tests
- Include at least 3 property-based tests using Hypothesis
- Use parameterized tests where appropriate
- Achieve at least 80% code coverage

**Hints:**

- Start with the tool function before testing it with the agent
- Use `@pytest.mark.parametrize` for testing multiple inputs
- Think about edge cases: empty strings, very long text, special characters
- Your property tests should verify invariants like "result is always one of the valid categories"

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

- **Test tools in isolation first**—they're pure functions and easy to test
- **Mock LLM responses** for deterministic, fast tests that don't cost money
- **Test behavior, not implementation**—focus on what the agent does, not how
- **Use pytest fixtures** to share setup code and ensure test isolation
- **Property-based testing** finds edge cases you wouldn't think of manually
- **Organize tests with markers** to run different test types in different contexts
- **Set up CI/CD** to run tests automatically on every commit
- **Test infrastructure is as important as the agent itself**—invest in it early

## What's Next

In Chapter 36, we'll explore **Observability and Logging**—how to see what your agent is doing in production. You can't debug what you can't see, and good logging is essential for understanding agent behavior in the wild. We'll build a structured logging system specifically designed for agentic systems.
