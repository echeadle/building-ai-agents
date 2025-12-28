"""
Tests for the agentic loop.

Chapter 35: Testing AI Agents - Implementation

This file demonstrates how to test the core agent loop, including:
- Basic conversation flow
- Tool execution
- Multi-step reasoning
- Error handling
- Guardrails (max iterations)

Run with: pytest test_agent_loop.py -v
"""

import pytest
from mock_llm import MockLLM
from testable_agent import TestableAgent, AgentConfig
from calculator import calculator, CALCULATOR_TOOL_DEFINITION


# =============================================================================
# Fixtures specific to this test file
# =============================================================================

@pytest.fixture
def agent_with_calculator():
    """Create an agent with just the calculator tool."""
    mock = MockLLM()
    
    tools = {"calculator": calculator}
    tool_definitions = [CALCULATOR_TOOL_DEFINITION]
    
    agent = TestableAgent(
        llm_client=mock,
        tools=tools,
        tool_definitions=tool_definitions,
        config=AgentConfig(max_iterations=10)
    )
    
    return agent, mock


@pytest.fixture
def simple_agent():
    """Create an agent without any tools."""
    mock = MockLLM()
    agent = TestableAgent(llm_client=mock)
    return agent, mock


# =============================================================================
# Basic Conversation Tests
# =============================================================================

class TestAgentBasicResponses:
    """Tests for basic agent responses without tool use."""
    
    @pytest.mark.integration
    def test_agent_returns_text_response(self, simple_agent):
        """Verify agent returns text when LLM doesn't use tools."""
        agent, mock = simple_agent
        mock.add_response(text="Hello! How can I help you today?")
        
        response = agent.run("Hello!")
        
        assert response == "Hello! How can I help you today?"
    
    @pytest.mark.integration
    def test_agent_handles_empty_response(self, simple_agent):
        """Verify agent handles when LLM returns no text."""
        agent, mock = simple_agent
        # Add response with no text content
        mock.add_response(text="")
        
        response = agent.run("Hello!")
        
        assert response == ""
    
    @pytest.mark.integration
    def test_agent_maintains_conversation_history(self, simple_agent):
        """Verify agent properly maintains conversation history."""
        agent, mock = simple_agent
        mock.add_response(text="I'm doing well!")
        mock.add_response(text="The weather is nice.")
        
        agent.run("How are you?")
        agent.run("What's the weather like?")
        
        # History should include both user messages
        user_messages = [
            m for m in agent.conversation_history 
            if m["role"] == "user" and isinstance(m["content"], str)
        ]
        
        assert len(user_messages) == 2
        assert user_messages[0]["content"] == "How are you?"
        assert user_messages[1]["content"] == "What's the weather like?"
    
    @pytest.mark.integration
    def test_agent_passes_system_prompt(self, simple_agent):
        """Verify agent passes system prompt to LLM."""
        agent, mock = simple_agent
        agent.config.system_prompt = "You are a pirate."
        mock.add_response(text="Arrr!")
        
        agent.run("Hello!")
        
        # Check the mock was called with system prompt
        assert mock.assert_called_with_system("pirate")


# =============================================================================
# Tool Execution Tests
# =============================================================================

class TestAgentToolUse:
    """Tests for agent tool use."""
    
    @pytest.mark.integration
    def test_agent_executes_single_tool_call(self, agent_with_calculator):
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
        assert agent.tool_call_log[0]["inputs"]["operation"] == "add"
    
    @pytest.mark.integration
    def test_agent_handles_multiple_sequential_tool_calls(self, agent_with_calculator):
        """Verify agent handles multiple sequential tool calls."""
        agent, mock = agent_with_calculator
        
        # First tool call: addition
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_1",
                "input": {"operation": "add", "a": 5, "b": 3}
            }
        )
        # Second tool call: multiplication
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
        assert agent.tool_call_log[0]["inputs"]["operation"] == "add"
        assert agent.tool_call_log[1]["inputs"]["operation"] == "multiply"
    
    @pytest.mark.integration
    def test_agent_tool_result_added_to_history(self, agent_with_calculator):
        """Verify tool results are added to conversation history."""
        agent, mock = agent_with_calculator
        
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_1",
                "input": {"operation": "add", "a": 2, "b": 2}
            }
        )
        mock.add_response(text="2 + 2 = 4")
        
        agent.run("What is 2 + 2?")
        
        # History should include tool result
        tool_result_messages = [
            m for m in agent.conversation_history
            if m["role"] == "user" and isinstance(m.get("content"), list)
        ]
        
        assert len(tool_result_messages) >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestAgentErrorHandling:
    """Tests for agent error handling."""
    
    @pytest.mark.integration
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
        assert response is not None
        # Tool call should be logged even if it failed
        assert len(agent.tool_call_log) == 1
    
    @pytest.mark.integration
    def test_agent_handles_tool_exception(self, agent_with_calculator):
        """Verify agent handles exceptions from tools gracefully."""
        agent, mock = agent_with_calculator
        
        # Replace calculator with a failing function
        def failing_calculator(**kwargs):
            raise ValueError("Simulated tool failure!")
        
        agent.tools["calculator"] = failing_calculator
        
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_1",
                "input": {"operation": "add", "a": 1, "b": 1}
            }
        )
        mock.add_response(text="The tool failed, but I can help anyway.")
        
        # Should not raise, should handle gracefully
        response = agent.run("Calculate something")
        
        assert response == "The tool failed, but I can help anyway."
    
    @pytest.mark.integration
    def test_agent_handles_wrong_tool_arguments(self, agent_with_calculator):
        """Verify agent handles when LLM passes wrong arguments to tool."""
        agent, mock = agent_with_calculator
        
        # LLM passes wrong argument names
        mock.add_response(
            tool_call={
                "name": "calculator",
                "id": "toolu_1",
                "input": {"wrong_arg": "value"}  # Wrong arguments
            }
        )
        mock.add_response(text="I made a mistake with the arguments.")
        
        response = agent.run("Calculate something")
        
        # Should complete without crashing
        assert response is not None


# =============================================================================
# Guardrail Tests
# =============================================================================

class TestAgentGuardrails:
    """Tests for agent safety and guardrails."""
    
    @pytest.mark.integration
    def test_agent_respects_max_iterations(self, agent_with_calculator):
        """Verify agent stops after max iterations."""
        agent, mock = agent_with_calculator
        agent.config.max_iterations = 3
        
        # Add responses that would cause infinite tool calls
        for i in range(20):
            mock.add_response(
                tool_call={
                    "name": "calculator",
                    "id": f"toolu_{i}",
                    "input": {"operation": "add", "a": 1, "b": 1}
                }
            )
        
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            agent.run("Keep calculating forever")
        
        # Should have stopped at max_iterations
        assert agent.get_iteration_count() == 3
    
    @pytest.mark.integration
    def test_agent_with_high_max_iterations(self, agent_with_calculator):
        """Verify agent works with higher iteration limits."""
        agent, mock = agent_with_calculator
        agent.config.max_iterations = 100
        
        # Add 5 tool calls then a text response
        for i in range(5):
            mock.add_response(
                tool_call={
                    "name": "calculator",
                    "id": f"toolu_{i}",
                    "input": {"operation": "add", "a": i, "b": 1}
                }
            )
        mock.add_response(text="Done with all calculations!")
        
        response = agent.run("Calculate multiple things")
        
        assert response == "Done with all calculations!"
        assert len(agent.tool_call_log) == 5


# =============================================================================
# State Management Tests
# =============================================================================

class TestAgentState:
    """Tests for agent state management."""
    
    @pytest.mark.integration
    def test_clear_history_resets_state(self, agent_with_calculator):
        """Verify clear_history properly resets agent state."""
        agent, mock = agent_with_calculator
        
        mock.add_response(text="First response")
        agent.run("First message")
        
        # Clear and check
        agent.clear_history()
        
        assert len(agent.conversation_history) == 0
        assert len(agent.tool_call_log) == 0
    
    @pytest.mark.integration
    def test_tool_call_log_persists_across_runs(self, agent_with_calculator):
        """Verify tool calls are accumulated across multiple runs."""
        agent, mock = agent_with_calculator
        
        # First run with tool
        mock.add_response(
            tool_call={"name": "calculator", "id": "t1", "input": {"operation": "add", "a": 1, "b": 1}}
        )
        mock.add_response(text="1 + 1 = 2")
        agent.run("What is 1+1?")
        
        # Second run with tool
        mock.add_response(
            tool_call={"name": "calculator", "id": "t2", "input": {"operation": "multiply", "a": 2, "b": 3}}
        )
        mock.add_response(text="2 × 3 = 6")
        agent.run("What is 2*3?")
        
        assert len(agent.tool_call_log) == 2
        assert agent.tool_call_log[0]["inputs"]["operation"] == "add"
        assert agent.tool_call_log[1]["inputs"]["operation"] == "multiply"


# =============================================================================
# LLM Call Verification Tests
# =============================================================================

class TestAgentLLMCalls:
    """Tests that verify how the agent calls the LLM."""
    
    @pytest.mark.integration
    def test_agent_passes_tools_to_llm(self, agent_with_calculator):
        """Verify agent passes tool definitions to the LLM."""
        agent, mock = agent_with_calculator
        mock.add_response(text="I have access to calculator")
        
        agent.run("What tools do you have?")
        
        assert mock.assert_called_with_tool("calculator")
    
    @pytest.mark.integration
    def test_agent_builds_correct_message_history(self, agent_with_calculator):
        """Verify agent builds correct message history for LLM."""
        agent, mock = agent_with_calculator
        
        mock.add_response(
            tool_call={"name": "calculator", "id": "t1", "input": {"operation": "add", "a": 1, "b": 2}}
        )
        mock.add_response(text="The result is 3")
        
        agent.run("What is 1+2?")
        
        # Check the second call had proper history
        last_call = mock.call_history[-1]
        messages = last_call["messages"]
        
        # Should have: user message, assistant (tool call), user (tool result)
        assert len(messages) >= 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is 1+2?"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
