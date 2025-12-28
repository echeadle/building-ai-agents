"""
Complete test suite for the AI agent.

Chapter 35: Testing AI Agents - Implementation

This file demonstrates how to organize a comprehensive test suite
with proper setup, teardown, and test organization. Use this as
a template for your own agent test suites.

Run with: pytest test_suite.py -v
Run with coverage: pytest test_suite.py --cov=. --cov-report=html
"""

import pytest
import os
from typing import Generator

# Import test utilities
from mock_llm import MockLLM, create_tool_response, create_text_response
from testable_agent import TestableAgent, AgentConfig
from calculator import calculator, CALCULATOR_TOOL_DEFINITION


# =============================================================================
# Session-Scoped Fixtures (shared across all tests)
# =============================================================================

@pytest.fixture(scope="session")
def shared_tool_definitions() -> list[dict]:
    """Tool definitions shared across all tests in the session."""
    return [CALCULATOR_TOOL_DEFINITION]


# =============================================================================
# Function-Scoped Fixtures (fresh for each test)
# =============================================================================

@pytest.fixture
def mock_llm() -> MockLLM:
    """Fresh MockLLM for each test."""
    return MockLLM()


@pytest.fixture
def agent_factory(shared_tool_definitions):
    """
    Factory for creating agents with common configuration.
    
    Usage:
        def test_something(agent_factory):
            agent, mock = agent_factory(include_tools=True)
            mock.add_response(text="Hello")
            response = agent.run("Hi")
    """
    def _create_agent(
        mock: MockLLM = None,
        include_tools: bool = True,
        max_iterations: int = 10,
        system_prompt: str = "You are a helpful assistant."
    ) -> tuple[TestableAgent, MockLLM]:
        if mock is None:
            mock = MockLLM()
        
        tools = {"calculator": calculator} if include_tools else {}
        tool_defs = shared_tool_definitions if include_tools else []
        
        agent = TestableAgent(
            llm_client=mock,
            tools=tools,
            tool_definitions=tool_defs,
            config=AgentConfig(
                max_iterations=max_iterations,
                system_prompt=system_prompt
            )
        )
        
        return agent, mock
    
    return _create_agent


# =============================================================================
# UNIT TESTS: Tools
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
    
    @pytest.mark.unit
    class TestOutputStructure:
        """Test that output has correct structure."""
        
        def test_success_has_result_key(self):
            result = calculator("add", 1, 1)
            assert "success" in result
            assert "result" in result
            assert result["success"] is True
        
        def test_failure_has_error_key(self):
            result = calculator("divide", 1, 0)
            assert "success" in result
            assert "error" in result
            assert result["success"] is False


# =============================================================================
# UNIT TESTS: Mock LLM
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
    
    @pytest.mark.unit
    def test_tool_call_response_structure(self, mock_llm):
        mock_llm.add_response(
            tool_call={"name": "test_tool", "input": {"key": "value"}}
        )
        
        response = mock_llm.create_message(messages=[])
        
        assert response.stop_reason == "tool_use"
        assert response.content[0].type == "tool_use"
        assert response.content[0].name == "test_tool"


# =============================================================================
# INTEGRATION TESTS: Agent Basic Behavior
# =============================================================================

class TestAgentBasicBehavior:
    """Integration tests for basic agent behavior."""
    
    @pytest.mark.integration
    def test_simple_conversation(self, agent_factory):
        agent, mock = agent_factory(include_tools=False)
        mock.add_response(text="Hello! I'm here to help.")
        
        response = agent.run("Hello!")
        
        assert "help" in response.lower()
    
    @pytest.mark.integration
    def test_maintains_conversation_history(self, agent_factory):
        agent, mock = agent_factory(include_tools=False)
        mock.add_response(text="Response 1")
        mock.add_response(text="Response 2")
        
        agent.run("Message 1")
        agent.run("Message 2")
        
        user_messages = [
            m for m in agent.conversation_history
            if m["role"] == "user" and isinstance(m["content"], str)
        ]
        assert len(user_messages) == 2
    
    @pytest.mark.integration
    def test_uses_system_prompt(self, agent_factory):
        agent, mock = agent_factory(
            include_tools=False,
            system_prompt="You are a pirate."
        )
        mock.add_response(text="Arrr!")
        
        agent.run("Hello!")
        
        assert mock.assert_called_with_system("pirate")


# =============================================================================
# INTEGRATION TESTS: Agent Tool Use
# =============================================================================

class TestAgentToolUse:
    """Integration tests for agent tool usage."""
    
    @pytest.mark.integration
    def test_single_tool_execution(self, agent_factory):
        agent, mock = agent_factory()
        
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t1",
                "input": {"operation": "add", "a": 5, "b": 3}
            }
        )
        mock.add_response(text="The answer is 8.")
        
        response = agent.run("What is 5 + 3?")
        
        assert "8" in response
        assert len(agent.tool_call_log) == 1
    
    @pytest.mark.integration
    def test_multiple_sequential_tools(self, agent_factory):
        agent, mock = agent_factory()
        
        # First tool call
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t1",
                "input": {"operation": "add", "a": 10, "b": 5}
            }
        )
        # Second tool call
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t2",
                "input": {"operation": "multiply", "a": 15, "b": 2}
            }
        )
        # Final answer
        mock.add_response(text="10 + 5 = 15, then 15 × 2 = 30")
        
        response = agent.run("What is (10 + 5) × 2?")
        
        assert "30" in response
        assert len(agent.tool_call_log) == 2
    
    @pytest.mark.integration
    def test_handles_unknown_tool(self, agent_factory):
        agent, mock = agent_factory(include_tools=False)  # No tools registered
        
        mock.add_response(
            tool_call={
                "name": "unknown_tool",
                "id": "t1",
                "input": {}
            }
        )
        mock.add_response(text="Tool not found.")
        
        response = agent.run("Use unknown tool")
        
        # Should complete without crashing
        assert response is not None
    
    @pytest.mark.integration
    def test_handles_tool_exception(self, agent_factory):
        agent, mock = agent_factory()
        
        # Replace calculator with failing function
        def failing_tool(**kwargs):
            raise ValueError("Tool failed!")
        agent.tools["calculator"] = failing_tool
        
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t1",
                "input": {"operation": "add", "a": 1, "b": 1}
            }
        )
        mock.add_response(text="Tool error handled.")
        
        response = agent.run("Calculate")
        
        assert response is not None


# =============================================================================
# INTEGRATION TESTS: Agent Guardrails
# =============================================================================

class TestAgentGuardrails:
    """Integration tests for agent safety guardrails."""
    
    @pytest.mark.integration
    def test_respects_max_iterations(self, agent_factory):
        agent, mock = agent_factory(max_iterations=3)
        
        # Create infinite loop of tool calls
        for i in range(20):
            mock.add_response(
                tool_call={
                    "name": "calculator",
                    "id": f"t{i}",
                    "input": {"operation": "add", "a": 1, "b": 1}
                }
            )
        
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            agent.run("Loop forever")
        
        assert agent.get_iteration_count() == 3
    
    @pytest.mark.integration
    def test_completes_before_max_iterations(self, agent_factory):
        agent, mock = agent_factory(max_iterations=10)
        
        # Add 3 tool calls then final response
        for i in range(3):
            mock.add_response(
                tool_call={
                    "name": "calculator",
                    "id": f"t{i}",
                    "input": {"operation": "add", "a": i, "b": 1}
                }
            )
        mock.add_response(text="Done!")
        
        response = agent.run("Calculate")
        
        assert response == "Done!"
        assert agent.get_iteration_count() == 4  # 3 tool calls + 1 final


# =============================================================================
# END-TO-END TESTS: Real Scenarios
# =============================================================================

class TestEndToEndScenarios:
    """End-to-end tests simulating real user scenarios."""
    
    @pytest.mark.integration
    def test_math_homework_scenario(self, agent_factory):
        """Simulate a student asking for help with math."""
        agent, mock = agent_factory()
        
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t1",
                "input": {"operation": "multiply", "a": 7, "b": 8}
            }
        )
        mock.add_response(text="7 × 8 = 56. The answer is 56!")
        
        response = agent.run("I need help with my homework. What is 7 times 8?")
        
        assert "56" in response
    
    @pytest.mark.integration
    def test_multi_step_calculation(self, agent_factory):
        """Test a calculation requiring multiple steps."""
        agent, mock = agent_factory()
        
        # Step 1: Add 100 + 50
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t1",
                "input": {"operation": "add", "a": 100, "b": 50}
            }
        )
        # Step 2: Multiply by 2
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t2",
                "input": {"operation": "multiply", "a": 150, "b": 2}
            }
        )
        # Step 3: Divide by 3
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t3",
                "input": {"operation": "divide", "a": 300, "b": 3}
            }
        )
        mock.add_response(text="The final answer is 100.")
        
        response = agent.run("What is ((100 + 50) × 2) ÷ 3?")
        
        assert "100" in response
        assert len(agent.tool_call_log) == 3
    
    @pytest.mark.integration
    def test_conversation_with_context(self, agent_factory):
        """Test that agent maintains context across turns."""
        agent, mock = agent_factory()
        
        # First turn
        mock.add_response(text="I'll help you with math. What would you like to calculate?")
        agent.run("I need help with math")
        
        # Second turn with calculation
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "t1",
                "input": {"operation": "add", "a": 25, "b": 17}
            }
        )
        mock.add_response(text="25 + 17 = 42")
        response = agent.run("What is 25 plus 17?")
        
        assert "42" in response
        
        # Verify history contains both turns
        assert len(agent.conversation_history) >= 4  # At least 2 user + 2 assistant


# =============================================================================
# STATE MANAGEMENT TESTS
# =============================================================================

class TestAgentStateManagement:
    """Tests for agent state management."""
    
    @pytest.mark.integration
    def test_clear_history(self, agent_factory):
        agent, mock = agent_factory()
        
        mock.add_response(text="First response")
        agent.run("First message")
        
        agent.clear_history()
        
        assert len(agent.conversation_history) == 0
        assert len(agent.tool_call_log) == 0
    
    @pytest.mark.integration
    def test_tool_log_accumulates(self, agent_factory):
        agent, mock = agent_factory()
        
        # First run
        mock.add_response(
            tool_call={"name": "calculator", "id": "t1", "input": {"operation": "add", "a": 1, "b": 1}}
        )
        mock.add_response(text="2")
        agent.run("1+1")
        
        # Second run
        mock.add_response(
            tool_call={"name": "calculator", "id": "t2", "input": {"operation": "multiply", "a": 2, "b": 2}}
        )
        mock.add_response(text="4")
        agent.run("2*2")
        
        assert len(agent.tool_call_log) == 2


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

class TestParametrizedOperations:
    """Parametrized tests for multiple scenarios."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("operation,a,b,expected", [
        ("add", 1, 1, 2),
        ("add", -1, 1, 0),
        ("subtract", 10, 3, 7),
        ("multiply", 6, 7, 42),
        ("divide", 100, 4, 25),
    ])
    def test_calculator_operations(self, operation, a, b, expected):
        result = calculator(operation, a, b)
        assert result["success"] is True
        assert result["result"] == expected
    
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_op", ["power", "mod", "sqrt", ""])
    def test_invalid_operations_fail(self, invalid_op):
        result = calculator(invalid_op, 1, 1)
        assert result["success"] is False


# =============================================================================
# TEST RUNNER CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not api",  # Skip API tests when running directly
        "--durations=10",  # Show 10 slowest tests
    ])
