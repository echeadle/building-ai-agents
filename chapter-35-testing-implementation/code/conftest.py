"""
Shared pytest fixtures for agent testing.

Chapter 35: Testing AI Agents - Implementation

This conftest.py file is automatically loaded by pytest and makes
all fixtures defined here available to every test file in the
tests/ directory.
"""

import os
import pytest
from typing import Generator, Any
from unittest.mock import Mock

# Import our testing utilities
from mock_llm import MockLLM, MockMessage, MockContentBlock
from testable_agent import TestableAgent, AgentConfig
from calculator import calculator, CALCULATOR_TOOL_DEFINITION

# =============================================================================
# Environment Setup
# =============================================================================

# Ensure we don't accidentally make real API calls during tests
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"


# =============================================================================
# LLM Mocks
# =============================================================================

@pytest.fixture
def mock_llm() -> MockLLM:
    """
    Creates a fresh MockLLM for each test.
    
    This fixture provides a clean mock that can be configured
    with responses specific to each test.
    
    Example:
        def test_something(mock_llm):
            mock_llm.add_response(text="Hello!")
            # Use mock_llm in test...
    """
    return MockLLM()


@pytest.fixture
def mock_anthropic_client() -> Mock:
    """
    Creates a mock Anthropic client for testing code that
    uses the real Anthropic SDK structure.
    
    This is useful when testing code that creates an
    anthropic.Anthropic() client directly.
    """
    client = Mock()
    client.messages = Mock()
    client.messages.create = Mock()
    return client


# =============================================================================
# Response Templates
# =============================================================================

@pytest.fixture
def simple_text_response() -> dict:
    """
    Returns a mock API response containing only text.
    
    Use this for testing basic conversation flow without tools.
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


# =============================================================================
# Tool Fixtures
# =============================================================================

@pytest.fixture
def calculator_tool() -> dict:
    """
    Returns the calculator tool definition.
    """
    return CALCULATOR_TOOL_DEFINITION.copy()


@pytest.fixture
def calculator_func():
    """
    Returns the calculator function for testing.
    """
    return calculator


@pytest.fixture
def all_tools() -> dict[str, Any]:
    """
    Returns a dict of all available tool functions.
    """
    return {
        "calculator": calculator
    }


@pytest.fixture
def all_tool_definitions() -> list[dict]:
    """
    Returns a list of all tool definitions.
    """
    return [CALCULATOR_TOOL_DEFINITION]


# =============================================================================
# Agent Fixtures
# =============================================================================

@pytest.fixture
def agent_config() -> AgentConfig:
    """
    Returns a default agent configuration.
    """
    return AgentConfig(
        max_iterations=10,
        system_prompt="You are a helpful assistant.",
        verbose=False
    )


@pytest.fixture
def agent_with_tools(mock_llm, all_tools, all_tool_definitions, agent_config):
    """
    Creates an agent with tools configured.
    
    Returns:
        A tuple of (agent, mock_llm) for configuring responses
    """
    agent = TestableAgent(
        llm_client=mock_llm,
        tools=all_tools,
        tool_definitions=all_tool_definitions,
        config=agent_config
    )
    return agent, mock_llm


@pytest.fixture
def simple_agent(mock_llm, agent_config):
    """
    Creates a simple agent without tools.
    
    Returns:
        A tuple of (agent, mock_llm) for configuring responses
    """
    agent = TestableAgent(
        llm_client=mock_llm,
        config=agent_config
    )
    return agent, mock_llm


# =============================================================================
# Conversation Fixtures
# =============================================================================

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


@pytest.fixture
def empty_conversation() -> list[dict]:
    """
    Returns an empty conversation history.
    """
    return []


# =============================================================================
# Mock Response Factory
# =============================================================================

@pytest.fixture
def mock_response_factory():
    """
    Factory fixture for creating mock responses with custom content.
    
    Usage:
        def test_something(mock_response_factory):
            response = mock_response_factory([
                {"type": "text", "text": "Hello!"}
            ])
    """
    def _create_response(
        content: list[dict],
        stop_reason: str = "end_turn"
    ) -> MockMessage:
        blocks = []
        for item in content:
            if item["type"] == "text":
                blocks.append(MockContentBlock(
                    type="text",
                    text=item["text"]
                ))
            elif item["type"] == "tool_use":
                blocks.append(MockContentBlock(
                    type="tool_use",
                    id=item.get("id", "toolu_test"),
                    name=item["name"],
                    input=item.get("input", {})
                ))
        
        return MockMessage(content=blocks, stop_reason=stop_reason)
    
    return _create_response


# =============================================================================
# Assertion Helpers
# =============================================================================

@pytest.fixture
def assert_tool_called():
    """
    Fixture that provides a helper to assert a tool was called.
    
    Usage:
        def test_something(agent_with_tools, assert_tool_called):
            agent, mock = agent_with_tools
            # ... run agent ...
            assert_tool_called(agent, "calculator", {"operation": "add"})
    """
    def _assert_tool_called(
        agent: TestableAgent,
        tool_name: str,
        expected_inputs: dict = None
    ) -> None:
        tool_calls = agent.get_tool_calls()
        
        matching_calls = [c for c in tool_calls if c["name"] == tool_name]
        assert len(matching_calls) > 0, f"Tool '{tool_name}' was not called"
        
        if expected_inputs:
            for call in matching_calls:
                for key, value in expected_inputs.items():
                    if key in call["inputs"] and call["inputs"][key] == value:
                        return  # Found matching call
            
            raise AssertionError(
                f"Tool '{tool_name}' was called but not with expected inputs. "
                f"Expected: {expected_inputs}, Got: {[c['inputs'] for c in matching_calls]}"
            )
    
    return _assert_tool_called


# =============================================================================
# Test Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require real API calls"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
