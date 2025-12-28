"""
Tests demonstrating mock LLM usage patterns.

Chapter 35: Testing AI Agents - Implementation

This file shows various patterns for using MockLLM effectively
in your tests. Use this as a reference for common testing scenarios.

Run with: pytest test_with_mocks.py -v
"""

import pytest
from mock_llm import (
    MockLLM, 
    MockMessage, 
    MockContentBlock,
    create_tool_response, 
    create_text_response,
    create_multi_content_response
)


class TestMockLLMBasics:
    """Basic tests for the MockLLM class."""
    
    @pytest.mark.unit
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
    
    @pytest.mark.unit
    def test_method_chaining(self):
        """Verify add_response supports method chaining."""
        mock = MockLLM()
        
        # Should be able to chain multiple add_response calls
        result = (mock
            .add_response(text="First")
            .add_response(text="Second")
            .add_response(text="Third"))
        
        assert result is mock  # Returns self
        assert len(mock.responses) == 3
    
    @pytest.mark.unit
    def test_records_call_history(self):
        """Verify the mock records all calls made to it."""
        mock = MockLLM()
        mock.add_response(text="Response")
        
        mock.create_message(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are helpful.",
            max_tokens=500
        )
        
        assert len(mock.call_history) == 1
        call = mock.call_history[0]
        assert call["messages"][0]["content"] == "Hello"
        assert call["system"] == "You are helpful."
        assert call["max_tokens"] == 500
    
    @pytest.mark.unit
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
    
    @pytest.mark.unit
    def test_reset_functionality(self):
        """Verify reset clears call history and resets index."""
        mock = MockLLM()
        mock.add_response(text="Response 1")
        mock.add_response(text="Response 2")
        
        mock.create_message(messages=[])
        mock.create_message(messages=[])
        
        assert len(mock.call_history) == 2
        
        mock.reset()
        
        assert len(mock.call_history) == 0
        assert mock.response_index == 0
        
        # Should return Response 1 again
        r = mock.create_message(messages=[])
        assert r.content[0].text == "Response 1"


class TestMockLLMAssertions:
    """Tests for MockLLM assertion helpers."""
    
    @pytest.mark.unit
    def test_assert_called(self):
        """Test assert_called helper."""
        mock = MockLLM()
        mock.add_response(text="Response")
        
        assert mock.assert_called() is False
        
        mock.create_message(messages=[])
        
        assert mock.assert_called() is True
    
    @pytest.mark.unit
    def test_assert_called_times(self):
        """Test assert_called_times helper."""
        mock = MockLLM()
        mock.add_response(text="R1")
        mock.add_response(text="R2")
        mock.add_response(text="R3")
        
        assert mock.assert_called_times(0) is True
        
        mock.create_message(messages=[])
        assert mock.assert_called_times(1) is True
        
        mock.create_message(messages=[])
        mock.create_message(messages=[])
        assert mock.assert_called_times(3) is True
    
    @pytest.mark.unit
    def test_assert_called_with_tool(self):
        """Test assert_called_with_tool helper."""
        mock = MockLLM()
        mock.add_response(text="Response")
        
        tools = [
            {"name": "calculator", "description": "Math"},
            {"name": "weather", "description": "Weather"}
        ]
        
        mock.create_message(messages=[], tools=tools)
        
        assert mock.assert_called_with_tool("calculator") is True
        assert mock.assert_called_with_tool("weather") is True
        assert mock.assert_called_with_tool("unknown") is False
    
    @pytest.mark.unit
    def test_assert_called_with_system(self):
        """Test assert_called_with_system helper."""
        mock = MockLLM()
        mock.add_response(text="Response")
        
        mock.create_message(
            messages=[],
            system="You are a helpful pirate assistant."
        )
        
        assert mock.assert_called_with_system("pirate") is True
        assert mock.assert_called_with_system("helpful") is True
        assert mock.assert_called_with_system("robot") is False


class TestMockLLMPatternMatching:
    """Tests for pattern-based response matching."""
    
    @pytest.mark.unit
    def test_pattern_response_basic(self):
        """Test basic pattern matching."""
        mock = MockLLM()
        
        # Add pattern: if message contains "hello", respond with greeting
        mock.add_pattern_response(
            lambda msgs: any("hello" in m.get("content", "").lower() 
                           for m in msgs if isinstance(m.get("content"), str)),
            create_text_response("Hello to you too!")
        )
        
        # Add default response
        mock.add_response(text="Default response")
        
        # Test pattern match
        r1 = mock.create_message(messages=[
            {"role": "user", "content": "Hello there!"}
        ])
        assert r1.content[0].text == "Hello to you too!"
        
        # Test non-match (falls through to queue)
        r2 = mock.create_message(messages=[
            {"role": "user", "content": "Goodbye!"}
        ])
        assert r2.content[0].text == "Default response"
    
    @pytest.mark.unit
    def test_pattern_for_math_questions(self):
        """Test pattern matching for math questions."""
        mock = MockLLM()
        
        # Pattern: if message asks about math, use calculator
        def is_math_question(msgs):
            last_msg = msgs[-1] if msgs else {}
            content = last_msg.get("content", "")
            if isinstance(content, str):
                math_words = ["plus", "minus", "times", "divided", "+", "-", "*", "/"]
                return any(word in content.lower() for word in math_words)
            return False
        
        mock.add_pattern_response(
            is_math_question,
            create_tool_response("calculator", {"operation": "add", "a": 1, "b": 1})
        )
        mock.add_response(text="I don't know")
        
        r1 = mock.create_message(messages=[
            {"role": "user", "content": "What is 2 plus 2?"}
        ])
        assert r1.stop_reason == "tool_use"
        
        r2 = mock.create_message(messages=[
            {"role": "user", "content": "Tell me about cats"}
        ])
        assert r2.content[0].text == "I don't know"


class TestHelperFunctions:
    """Tests for helper functions."""
    
    @pytest.mark.unit
    def test_create_text_response(self):
        """Test create_text_response helper."""
        response = create_text_response("Hello, world!")
        
        assert isinstance(response, MockMessage)
        assert response.stop_reason == "end_turn"
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello, world!"
    
    @pytest.mark.unit
    def test_create_text_response_custom_stop_reason(self):
        """Test create_text_response with custom stop reason."""
        response = create_text_response("Partial response", stop_reason="max_tokens")
        
        assert response.stop_reason == "max_tokens"
    
    @pytest.mark.unit
    def test_create_tool_response(self):
        """Test create_tool_response helper."""
        response = create_tool_response(
            "calculator",
            {"operation": "add", "a": 5, "b": 3},
            tool_id="custom_id"
        )
        
        assert isinstance(response, MockMessage)
        assert response.stop_reason == "tool_use"
        assert response.content[0].type == "tool_use"
        assert response.content[0].name == "calculator"
        assert response.content[0].id == "custom_id"
        assert response.content[0].input["operation"] == "add"
    
    @pytest.mark.unit
    def test_create_multi_content_response(self):
        """Test create_multi_content_response helper."""
        response = create_multi_content_response(
            text="Let me calculate that for you.",
            tool_name="calculator",
            tool_input={"operation": "multiply", "a": 6, "b": 7}
        )
        
        assert len(response.content) == 2
        assert response.content[0].type == "text"
        assert response.content[0].text == "Let me calculate that for you."
        assert response.content[1].type == "tool_use"
        assert response.content[1].name == "calculator"


class TestMockLLMWithAgent:
    """Tests showing how to use MockLLM with an agent."""
    
    @pytest.mark.integration
    def test_simple_conversation_flow(self):
        """Test a simple conversation without tools."""
        from testable_agent import TestableAgent
        
        mock = MockLLM()
        mock.add_response(text="Hello! How can I help you today?")
        mock.add_response(text="I'm doing well, thank you for asking!")
        
        agent = TestableAgent(llm_client=mock)
        
        r1 = agent.run("Hi there!")
        assert "help" in r1.lower()
        
        r2 = agent.run("How are you?")
        assert "well" in r2.lower()
    
    @pytest.mark.integration
    def test_tool_usage_flow(self):
        """
        Test that an agent correctly invokes a tool when the LLM requests it.
        
        This simulates a multi-turn interaction:
        1. User asks a question
        2. LLM requests a tool call
        3. Tool is executed
        4. LLM provides final answer
        """
        from testable_agent import TestableAgent
        from calculator import calculator
        
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
        
        agent = TestableAgent(
            llm_client=mock,
            tools={"calculator": calculator},
            tool_definitions=[{
                "name": "calculator",
                "description": "Math operations",
                "input_schema": {"type": "object", "properties": {}}
            }]
        )
        
        response = agent.run("What is 5 + 3?")
        
        assert "8" in response
        assert len(agent.tool_call_log) == 1
        assert agent.tool_call_log[0]["name"] == "calculator"
    
    @pytest.mark.integration
    def test_verify_llm_received_correct_messages(self):
        """Test that we can verify what the LLM received."""
        from testable_agent import TestableAgent
        
        mock = MockLLM()
        mock.add_response(text="I understand you want to know about Python.")
        
        agent = TestableAgent(
            llm_client=mock,
            config=type('Config', (), {
                'max_iterations': 10,
                'system_prompt': 'You are a Python expert.',
                'verbose': False
            })()
        )
        
        agent.run("Tell me about Python")
        
        # Verify the LLM was called correctly
        assert mock.assert_called()
        assert mock.assert_called_with_system("Python expert")
        
        last_messages = mock.get_last_messages()
        assert any("Python" in str(m.get("content", "")) for m in last_messages)


class TestMockingEdgeCases:
    """Tests for edge cases in mocking."""
    
    @pytest.mark.unit
    def test_exhausted_response_queue(self):
        """Test behavior when response queue is exhausted."""
        mock = MockLLM()
        mock.add_response(text="Only response")
        
        r1 = mock.create_message(messages=[])
        assert r1.content[0].text == "Only response"
        
        # Queue is now empty
        r2 = mock.create_message(messages=[])
        assert "No more mock responses" in r2.content[0].text
    
    @pytest.mark.unit
    def test_clear_all_completely_resets(self):
        """Test that clear_all removes everything."""
        mock = MockLLM()
        mock.add_response(text="Response")
        mock.add_pattern_response(lambda x: True, create_text_response("Pattern"))
        mock.create_message(messages=[])
        
        mock.clear_all()
        
        assert len(mock.responses) == 0
        assert len(mock.pattern_responses) == 0
        assert len(mock.call_history) == 0
        assert mock.response_index == 0
    
    @pytest.mark.unit
    def test_auto_generated_tool_ids(self):
        """Test that tool IDs are auto-generated when not provided."""
        mock = MockLLM()
        mock.add_response(tool_call={"name": "tool1", "input": {}})
        mock.add_response(tool_call={"name": "tool2", "input": {}})
        
        r1 = mock.create_message(messages=[])
        r2 = mock.create_message(messages=[])
        
        # Should have different auto-generated IDs
        assert r1.content[0].id != r2.content[0].id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
