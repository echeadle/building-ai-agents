"""
Tests for the AugmentedLLM class.

This module contains both unit tests (no API calls) and integration tests
(require ANTHROPIC_API_KEY). Integration tests are marked with @pytest.mark.integration
and can be skipped in CI environments.

Run unit tests only:
    pytest test_augmented_llm.py -v -m "not integration"

Run all tests:
    pytest test_augmented_llm.py -v

Chapter 14: Building the Complete Augmented LLM
"""

import os
import json
import pytest

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from augmented_llm import AugmentedLLM, AugmentedLLMConfig, ToolRegistry


# =============================================================================
# Unit Tests - No API calls required
# =============================================================================

class TestToolRegistry:
    """Unit tests for the ToolRegistry class."""
    
    def test_empty_registry(self):
        """Test that a new registry is empty."""
        registry = ToolRegistry()
        
        assert len(registry) == 0
        assert registry.get_definitions() == []
        assert registry.get_tool_names() == []
    
    def test_register_tool(self):
        """Test basic tool registration."""
        registry = ToolRegistry()
        
        registry.register(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            function=lambda: "test result"
        )
        
        assert len(registry) == 1
        assert registry.has_tool("test_tool")
        assert not registry.has_tool("nonexistent")
    
    def test_get_definitions_format(self):
        """Test that definitions match Claude's expected format."""
        registry = ToolRegistry()
        
        registry.register(
            name="my_tool",
            description="Does something useful",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                },
                "required": ["param1"]
            },
            function=lambda param1: param1
        )
        
        definitions = registry.get_definitions()
        
        assert len(definitions) == 1
        tool = definitions[0]
        
        # Check required fields for Claude API
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        
        # Check values
        assert tool["name"] == "my_tool"
        assert tool["description"] == "Does something useful"
        assert tool["input_schema"]["type"] == "object"
    
    def test_execute_tool(self):
        """Test tool execution with arguments."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            },
            function=add
        )
        
        result = registry.execute("add", {"a": 2, "b": 3})
        assert result == 5
    
    def test_execute_unknown_tool_raises_error(self):
        """Test that executing an unknown tool raises KeyError."""
        registry = ToolRegistry()
        
        with pytest.raises(KeyError, match="Unknown tool"):
            registry.execute("nonexistent", {})
    
    def test_register_duplicate_raises_error(self):
        """Test that registering a duplicate tool raises ValueError."""
        registry = ToolRegistry()
        
        registry.register(
            name="my_tool",
            description="First version",
            parameters={"type": "object", "properties": {}},
            function=lambda: "first"
        )
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                name="my_tool",
                description="Second version",
                parameters={"type": "object", "properties": {}},
                function=lambda: "second"
            )
    
    def test_get_tool_names(self):
        """Test getting list of tool names."""
        registry = ToolRegistry()
        
        registry.register("tool_a", "A", {"type": "object", "properties": {}}, lambda: "a")
        registry.register("tool_b", "B", {"type": "object", "properties": {}}, lambda: "b")
        registry.register("tool_c", "C", {"type": "object", "properties": {}}, lambda: "c")
        
        names = registry.get_tool_names()
        
        assert len(names) == 3
        assert "tool_a" in names
        assert "tool_b" in names
        assert "tool_c" in names
    
    def test_repr(self):
        """Test string representation of registry."""
        registry = ToolRegistry()
        registry.register("tool1", "T1", {"type": "object", "properties": {}}, lambda: None)
        registry.register("tool2", "T2", {"type": "object", "properties": {}}, lambda: None)
        
        repr_str = repr(registry)
        
        assert "ToolRegistry" in repr_str
        assert "tool1" in repr_str
        assert "tool2" in repr_str


class TestAugmentedLLMConfig:
    """Unit tests for the configuration class."""
    
    def test_default_values(self):
        """Test that defaults are sensible."""
        config = AugmentedLLMConfig()
        
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096
        assert config.max_tool_iterations == 10
        assert config.temperature == 1.0
        assert config.system_prompt == "You are a helpful assistant."
        assert config.response_schema is None
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = AugmentedLLMConfig(
            model="claude-opus-4-20250514",
            max_tokens=1000,
            system_prompt="Custom prompt",
            max_tool_iterations=5,
            temperature=0.5
        )
        
        assert config.model == "claude-opus-4-20250514"
        assert config.max_tokens == 1000
        assert config.system_prompt == "Custom prompt"
        assert config.max_tool_iterations == 5
        assert config.temperature == 0.5
    
    def test_immutability(self):
        """Test that config is immutable (frozen)."""
        config = AugmentedLLMConfig()
        
        with pytest.raises(AttributeError):
            config.model = "different-model"
        
        with pytest.raises(AttributeError):
            config.max_tokens = 500
    
    def test_with_schema(self):
        """Test config with response schema."""
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"}
            },
            "required": ["answer"]
        }
        
        config = AugmentedLLMConfig(response_schema=schema)
        
        assert config.response_schema == schema


class TestAugmentedLLMUnit:
    """Unit tests for AugmentedLLM that don't require API calls."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        llm = AugmentedLLM()
        
        assert llm.config.model == "claude-sonnet-4-20250514"
        assert len(llm.tools) == 0
    
    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = AugmentedLLMConfig(
            system_prompt="Custom prompt",
            max_tokens=1000
        )
        
        llm = AugmentedLLM(config=config)
        
        assert llm.config.system_prompt == "Custom prompt"
        assert llm.config.max_tokens == 1000
    
    def test_initialization_with_tools(self):
        """Test initialization with pre-configured tools."""
        registry = ToolRegistry()
        registry.register("test", "A test", {"type": "object", "properties": {}}, lambda: "result")
        
        llm = AugmentedLLM(tools=registry)
        
        assert len(llm.tools) == 1
        assert llm.tools.has_tool("test")
    
    def test_register_tool_method(self):
        """Test registering tools through the LLM instance."""
        llm = AugmentedLLM()
        
        llm.register_tool(
            name="my_tool",
            description="Does something",
            parameters={"type": "object", "properties": {}},
            function=lambda: "result"
        )
        
        assert len(llm.tools) == 1
        assert llm.tools.has_tool("my_tool")
    
    def test_history_management(self):
        """Test conversation history methods."""
        llm = AugmentedLLM()
        
        # History starts empty
        assert len(llm.get_history()) == 0
        
        # Clear on empty is safe
        llm.clear_history()
        assert len(llm.get_history()) == 0
    
    def test_get_history_returns_copy(self):
        """Test that get_history returns a copy, not the original."""
        llm = AugmentedLLM()
        
        history1 = llm.get_history()
        history1.append({"test": "data"})
        
        history2 = llm.get_history()
        
        assert len(history2) == 0  # Original not modified


class TestResponseValidation:
    """Unit tests for response validation logic."""
    
    def test_valid_json_passes(self):
        """Test that valid JSON matching schema passes."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {
                    "value": {"type": "string"}
                },
                "required": ["value"]
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        result = llm._validate_response('{"value": "test"}')
        assert json.loads(result) == {"value": "test"}
    
    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {}
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        with pytest.raises(ValueError, match="not valid JSON"):
            llm._validate_response("not json at all")
    
    def test_missing_required_field_raises_error(self):
        """Test that missing required fields raise ValueError."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"}
                },
                "required": ["required_field"]
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        with pytest.raises(ValueError, match="Missing required field"):
            llm._validate_response('{"other_field": "value"}')
    
    def test_wrong_type_raises_error(self):
        """Test that wrong types raise ValueError."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"}
                }
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        with pytest.raises(ValueError, match="should be integer"):
            llm._validate_response('{"count": "not a number"}')
    
    def test_invalid_enum_raises_error(self):
        """Test that invalid enum values raise ValueError."""
        config = AugmentedLLMConfig(
            response_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "inactive"]
                    }
                }
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        with pytest.raises(ValueError, match="must be one of"):
            llm._validate_response('{"status": "unknown"}')


class TestToolExecution:
    """Unit tests for the tool execution helper method."""
    
    def test_successful_execution(self):
        """Test successful tool execution returns proper format."""
        llm = AugmentedLLM()
        llm.register_tool(
            name="greet",
            description="Returns a greeting",
            parameters={"type": "object", "properties": {}},
            function=lambda: "Hello!"
        )
        
        result = llm._execute_tool("greet", {}, "test-id-123")
        
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "test-id-123"
        assert result["content"] == "Hello!"
        assert "is_error" not in result
    
    def test_dict_result_serialized(self):
        """Test that dict results are JSON serialized."""
        llm = AugmentedLLM()
        llm.register_tool(
            name="get_data",
            description="Returns data",
            parameters={"type": "object", "properties": {}},
            function=lambda: {"key": "value", "count": 42}
        )
        
        result = llm._execute_tool("get_data", {}, "test-id")
        
        # Should be JSON string
        assert result["content"] == '{"key": "value", "count": 42}'
    
    def test_error_handling(self):
        """Test that tool errors are captured properly."""
        llm = AugmentedLLM()
        llm.register_tool(
            name="failing_tool",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            function=lambda: 1/0  # Will raise ZeroDivisionError
        )
        
        result = llm._execute_tool("failing_tool", {}, "test-id")
        
        assert result["type"] == "tool_result"
        assert result["is_error"] is True
        assert "Error" in result["content"]
        assert "division by zero" in result["content"]


# =============================================================================
# Integration Tests - Require API key
# =============================================================================

@pytest.mark.integration
class TestAugmentedLLMIntegration:
    """Integration tests that make actual API calls."""
    
    @pytest.fixture(autouse=True)
    def check_api_key(self):
        """Skip tests if API key is not available."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
    
    def test_simple_query(self):
        """Test a simple query without tools."""
        llm = AugmentedLLM()
        
        response = llm.run("Reply with exactly the word 'HELLO' and nothing else.")
        
        assert "hello" in response.lower()
    
    def test_query_with_system_prompt(self):
        """Test that system prompts affect responses."""
        config = AugmentedLLMConfig(
            system_prompt="You are a pirate. Always respond in pirate speak."
        )
        llm = AugmentedLLM(config=config)
        
        response = llm.run("Say hello to me.")
        
        # Should contain pirate-like language
        pirate_words = ["ahoy", "matey", "arr", "ye", "me hearty", "aye"]
        assert any(word in response.lower() for word in pirate_words)
    
    def test_tool_usage(self):
        """Test that tools are called when appropriate."""
        llm = AugmentedLLM(
            config=AugmentedLLMConfig(
                system_prompt="Use the add tool to answer math questions. Only respond with the numeric result."
            )
        )
        
        llm.register_tool(
            name="add",
            description="Add two numbers together. Always use this for addition.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            function=lambda a, b: a + b
        )
        
        response = llm.run("What is 7 + 5?")
        
        assert "12" in response
    
    def test_conversation_history_preserved(self):
        """Test that conversation history is maintained across calls."""
        config = AugmentedLLMConfig(
            system_prompt="You have a perfect memory. Remember everything the user tells you."
        )
        llm = AugmentedLLM(config=config)
        
        # First message: tell it something
        llm.run("My favorite color is blue.")
        
        # Second message: ask about it
        response = llm.run("What is my favorite color?")
        
        assert "blue" in response.lower()
    
    def test_history_cleared_properly(self):
        """Test that clearing history removes context."""
        llm = AugmentedLLM()
        
        llm.run("My name is TestUser12345.")
        llm.clear_history()
        
        response = llm.run("What is my name?")
        
        # Should not know the name after clearing
        assert "TestUser12345" not in response
    
    def test_structured_output(self):
        """Test structured output validation."""
        config = AugmentedLLMConfig(
            system_prompt="""Always respond with valid JSON in this exact format:
{"answer": "your answer here", "confident": true}
No other text, just the JSON.""",
            response_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confident": {"type": "boolean"}
                },
                "required": ["answer", "confident"]
            }
        )
        
        llm = AugmentedLLM(config=config)
        
        response = llm.run("What is 2+2?")
        
        # Should be valid JSON
        data = json.loads(response)
        assert "answer" in data
        assert "confident" in data
        assert isinstance(data["confident"], bool)


if __name__ == "__main__":
    # Run with: python -m pytest test_augmented_llm.py -v
    pytest.main([__file__, "-v"])
