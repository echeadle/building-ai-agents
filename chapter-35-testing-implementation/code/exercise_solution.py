"""
Exercise Solution: Test Suite for a Text Classification Tool

Chapter 35: Testing AI Agents - Implementation

This file contains the complete solution to the chapter exercise:
Building a test suite for a text classification tool.

The exercise requirements were:
1. Unit tests for the classification function itself
2. Mock LLM tests simulating the agent calling the tool
3. Property-based tests for edge cases
4. Integration tests for the full agent workflow
5. At least 10 unit tests
6. At least 3 property-based tests using Hypothesis
7. Use parameterized tests where appropriate
8. Achieve at least 80% code coverage

Run with: pytest exercise_solution.py -v --cov=. --cov-report=term-missing
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from typing import Literal
from dataclasses import dataclass
from enum import Enum

# Import our testing utilities
from mock_llm import MockLLM, create_text_response, create_tool_response
from testable_agent import TestableAgent, AgentConfig


# =============================================================================
# THE TEXT CLASSIFICATION TOOL
# =============================================================================

class Sentiment(Enum):
    """Possible sentiment classifications."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


def classify_text(text: str) -> dict:
    """
    Classifies text into positive, negative, or neutral sentiment.
    
    This is a simple rule-based classifier for demonstration purposes.
    In production, you might use a more sophisticated model.
    
    Args:
        text: The text to classify
        
    Returns:
        A dict containing:
        - success (bool): Whether classification succeeded
        - sentiment (str): The detected sentiment (positive/negative/neutral)
        - confidence (float): Confidence score between 0 and 1
        - reason (str): Explanation for the classification
        - error (str): Error message if success is False
    """
    # Validate input
    if not isinstance(text, str):
        return {
            "success": False,
            "error": f"Expected string input, got {type(text).__name__}"
        }
    
    if not text or not text.strip():
        return {
            "success": False,
            "error": "Cannot classify empty text"
        }
    
    # Normalize text
    text_lower = text.lower().strip()
    
    # Define sentiment keywords
    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "happy", "joy", "best", "awesome", "perfect", "beautiful",
        "brilliant", "outstanding", "superb", "delighted", "pleased",
        "excited", "grateful", "thankful", "impressive", "incredible"
    }
    
    negative_words = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "sad",
        "angry", "disappointed", "frustrating", "annoying", "poor",
        "disgusting", "dreadful", "miserable", "pathetic", "useless",
        "failed", "broken", "ugly", "stupid", "boring", "painful"
    }
    
    # Count sentiment words
    words = set(text_lower.split())
    positive_count = len(words & positive_words)
    negative_count = len(words & negative_words)
    total_sentiment_words = positive_count + negative_count
    
    # Determine sentiment
    if total_sentiment_words == 0:
        sentiment = Sentiment.NEUTRAL
        confidence = 0.5
        reason = "No strong sentiment indicators found"
    elif positive_count > negative_count:
        sentiment = Sentiment.POSITIVE
        confidence = min(0.95, 0.5 + (positive_count - negative_count) * 0.15)
        reason = f"Found {positive_count} positive indicator(s)"
    elif negative_count > positive_count:
        sentiment = Sentiment.NEGATIVE
        confidence = min(0.95, 0.5 + (negative_count - positive_count) * 0.15)
        reason = f"Found {negative_count} negative indicator(s)"
    else:
        sentiment = Sentiment.NEUTRAL
        confidence = 0.4
        reason = "Equal positive and negative indicators found"
    
    return {
        "success": True,
        "sentiment": sentiment.value,
        "confidence": round(confidence, 2),
        "reason": reason,
        "word_count": len(text.split()),
        "analyzed_text_preview": text[:50] + "..." if len(text) > 50 else text
    }


# Tool definition for agent use
CLASSIFY_TEXT_TOOL_DEFINITION = {
    "name": "classify_text",
    "description": "Analyzes text and classifies its sentiment as positive, negative, or neutral. Use this tool when you need to determine the emotional tone of text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to classify"
            }
        },
        "required": ["text"]
    }
}


# =============================================================================
# UNIT TESTS: Classification Function
# =============================================================================

class TestClassifyTextBasicFunctionality:
    """Basic unit tests for the classify_text function."""
    
    @pytest.mark.unit
    def test_classifies_positive_text(self):
        """Test that clearly positive text is classified as positive."""
        result = classify_text("This is great and amazing!")
        
        assert result["success"] is True
        assert result["sentiment"] == "positive"
        assert result["confidence"] > 0.5
    
    @pytest.mark.unit
    def test_classifies_negative_text(self):
        """Test that clearly negative text is classified as negative."""
        result = classify_text("This is terrible and awful!")
        
        assert result["success"] is True
        assert result["sentiment"] == "negative"
        assert result["confidence"] > 0.5
    
    @pytest.mark.unit
    def test_classifies_neutral_text(self):
        """Test that neutral text is classified as neutral."""
        result = classify_text("The sky is blue and water is wet.")
        
        assert result["success"] is True
        assert result["sentiment"] == "neutral"
    
    @pytest.mark.unit
    def test_handles_mixed_sentiment(self):
        """Test text with both positive and negative words."""
        result = classify_text("Good but also bad")
        
        assert result["success"] is True
        assert result["sentiment"] == "neutral"
        assert "equal" in result["reason"].lower() or "neutral" in result["sentiment"]
    
    @pytest.mark.unit
    def test_confidence_increases_with_more_indicators(self):
        """Test that confidence increases with more sentiment words."""
        result_weak = classify_text("This is good")
        result_strong = classify_text("This is good, great, amazing, and wonderful!")
        
        assert result_strong["confidence"] > result_weak["confidence"]


class TestClassifyTextEdgeCases:
    """Edge case tests for the classify_text function."""
    
    @pytest.mark.unit
    def test_empty_string_returns_error(self):
        """Test that empty string returns an error."""
        result = classify_text("")
        
        assert result["success"] is False
        assert "empty" in result["error"].lower()
    
    @pytest.mark.unit
    def test_whitespace_only_returns_error(self):
        """Test that whitespace-only string returns an error."""
        result = classify_text("   \t\n   ")
        
        assert result["success"] is False
        assert "empty" in result["error"].lower()
    
    @pytest.mark.unit
    def test_handles_very_long_text(self):
        """Test that very long text is handled."""
        long_text = "This is great! " * 1000
        result = classify_text(long_text)
        
        assert result["success"] is True
        assert result["sentiment"] == "positive"
    
    @pytest.mark.unit
    def test_handles_special_characters(self):
        """Test that text with special characters is handled."""
        result = classify_text("This is great!!! 🎉🎉🎉 @#$%^&*()")
        
        assert result["success"] is True
        assert result["sentiment"] == "positive"
    
    @pytest.mark.unit
    def test_case_insensitive(self):
        """Test that classification is case-insensitive."""
        result_lower = classify_text("great")
        result_upper = classify_text("GREAT")
        result_mixed = classify_text("GrEaT")
        
        assert result_lower["sentiment"] == result_upper["sentiment"] == result_mixed["sentiment"]
    
    @pytest.mark.unit
    def test_non_string_input_returns_error(self):
        """Test that non-string input returns an error."""
        result = classify_text(12345)  # type: ignore
        
        assert result["success"] is False
        assert "string" in result["error"].lower()


class TestClassifyTextOutputStructure:
    """Tests for the output structure of classify_text."""
    
    @pytest.mark.unit
    def test_success_response_has_required_keys(self):
        """Test that successful response has all required keys."""
        result = classify_text("This is a test")
        
        assert "success" in result
        assert "sentiment" in result
        assert "confidence" in result
        assert "reason" in result
        assert "word_count" in result
    
    @pytest.mark.unit
    def test_sentiment_is_valid_value(self):
        """Test that sentiment is one of the valid values."""
        result = classify_text("Test text here")
        
        assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    @pytest.mark.unit
    def test_confidence_is_in_valid_range(self):
        """Test that confidence is between 0 and 1."""
        result = classify_text("Amazing wonderful fantastic!")
        
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.unit
    def test_error_response_has_error_key(self):
        """Test that error response has error key."""
        result = classify_text("")
        
        assert result["success"] is False
        assert "error" in result


class TestClassifyTextParametrized:
    """Parameterized tests for classify_text."""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("text,expected_sentiment", [
        ("I love this!", "positive"),
        ("This is wonderful", "positive"),
        ("Best thing ever", "positive"),
        ("I hate this!", "negative"),
        ("This is terrible", "negative"),
        ("Worst experience", "negative"),
        ("The meeting is at 3pm", "neutral"),
        ("Water boils at 100 degrees", "neutral"),
    ])
    def test_various_sentiments(self, text: str, expected_sentiment: str):
        """Test various texts are classified correctly."""
        result = classify_text(text)
        
        assert result["success"] is True
        assert result["sentiment"] == expected_sentiment
    
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_input", [
        None,
        123,
        [],
        {},
        True,
    ])
    def test_invalid_inputs_return_error(self, invalid_input):
        """Test that invalid inputs return errors."""
        result = classify_text(invalid_input)  # type: ignore
        
        assert result["success"] is False


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

class TestClassifyTextProperties:
    """Property-based tests using Hypothesis."""
    
    @pytest.mark.unit
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_always_returns_valid_structure(self, text: str):
        """Property: Always returns a dict with success key."""
        assume(text.strip())  # Skip empty/whitespace
        
        result = classify_text(text)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert isinstance(result["success"], bool)
    
    @pytest.mark.unit
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_success_implies_valid_sentiment(self, text: str):
        """Property: If success, sentiment is one of three valid values."""
        assume(text.strip())
        
        result = classify_text(text)
        
        if result["success"]:
            assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    @pytest.mark.unit
    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=100)
    def test_confidence_always_in_range(self, text: str):
        """Property: Confidence is always between 0 and 1."""
        assume(text.strip())
        
        result = classify_text(text)
        
        if result["success"]:
            assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.unit
    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    def test_result_is_json_serializable(self, text: str):
        """Property: Result is always JSON serializable."""
        import json
        
        assume(text.strip())
        
        result = classify_text(text)
        
        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["success"] == result["success"]
    
    @pytest.mark.unit
    @given(
        positive_word=st.sampled_from(["good", "great", "amazing", "wonderful"]),
        count=st.integers(min_value=1, max_value=10)
    )
    def test_more_positive_words_increase_confidence(self, positive_word: str, count: int):
        """Property: More positive words should not decrease confidence."""
        text_few = f"This is {positive_word}"
        text_many = f"This is {' '.join([positive_word] * count)}"
        
        result_few = classify_text(text_few)
        result_many = classify_text(text_many)
        
        assert result_many["confidence"] >= result_few["confidence"] - 0.01  # Small tolerance


# =============================================================================
# MOCK LLM TESTS
# =============================================================================

class TestClassifyTextWithMockLLM:
    """Tests simulating the agent calling the classification tool."""
    
    @pytest.fixture
    def agent_with_classifier(self):
        """Create an agent with the classify_text tool."""
        mock = MockLLM()
        
        agent = TestableAgent(
            llm_client=mock,
            tools={"classify_text": classify_text},
            tool_definitions=[CLASSIFY_TEXT_TOOL_DEFINITION],
            config=AgentConfig(max_iterations=10)
        )
        
        return agent, mock
    
    @pytest.mark.integration
    def test_agent_uses_classifier_tool(self, agent_with_classifier):
        """Test that agent correctly uses the classification tool."""
        agent, mock = agent_with_classifier
        
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t1",
                "input": {"text": "This product is amazing!"}
            }
        )
        mock.add_response(text="The text has a positive sentiment.")
        
        response = agent.run("What is the sentiment of 'This product is amazing!'?")
        
        assert len(agent.tool_call_log) == 1
        assert agent.tool_call_log[0]["name"] == "classify_text"
    
    @pytest.mark.integration
    def test_agent_handles_classification_result(self, agent_with_classifier):
        """Test that agent processes classification results correctly."""
        agent, mock = agent_with_classifier
        
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t1",
                "input": {"text": "I hate waiting in long lines"}
            }
        )
        mock.add_response(text="The sentiment is negative with high confidence.")
        
        response = agent.run("Analyze the sentiment: 'I hate waiting in long lines'")
        
        # Verify the tool was called with correct input
        tool_input = agent.tool_call_log[0]["inputs"]
        assert "hate" in tool_input["text"]
    
    @pytest.mark.integration
    def test_agent_multiple_classifications(self, agent_with_classifier):
        """Test agent performing multiple classifications."""
        agent, mock = agent_with_classifier
        
        # First classification
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t1",
                "input": {"text": "Great service!"}
            }
        )
        # Second classification
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t2",
                "input": {"text": "Terrible food!"}
            }
        )
        mock.add_response(text="First is positive, second is negative.")
        
        response = agent.run("Compare sentiment of 'Great service!' and 'Terrible food!'")
        
        assert len(agent.tool_call_log) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestClassifyTextIntegration:
    """Full integration tests for the classification tool with agent."""
    
    @pytest.fixture
    def full_agent(self):
        """Create a fully configured agent."""
        mock = MockLLM()
        
        agent = TestableAgent(
            llm_client=mock,
            tools={"classify_text": classify_text},
            tool_definitions=[CLASSIFY_TEXT_TOOL_DEFINITION],
            config=AgentConfig(
                max_iterations=10,
                system_prompt="You are a sentiment analysis assistant."
            )
        )
        
        return agent, mock
    
    @pytest.mark.integration
    def test_customer_review_analysis_workflow(self, full_agent):
        """Test a realistic customer review analysis workflow."""
        agent, mock = full_agent
        
        # Simulate analyzing a customer review
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t1",
                "input": {"text": "The product exceeded my expectations! Absolutely love it!"}
            }
        )
        mock.add_response(
            text="Based on my analysis, this customer review is very positive. "
                 "The customer expresses enthusiasm and satisfaction."
        )
        
        response = agent.run(
            "Analyze this customer review: 'The product exceeded my expectations! "
            "Absolutely love it!'"
        )
        
        assert "positive" in response.lower() or len(response) > 0
        assert len(agent.tool_call_log) == 1
    
    @pytest.mark.integration
    def test_handles_empty_text_gracefully(self, full_agent):
        """Test that agent handles tool errors gracefully."""
        agent, mock = full_agent
        
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t1",
                "input": {"text": ""}  # Empty text will cause error
            }
        )
        mock.add_response(text="I couldn't classify that text as it appears to be empty.")
        
        response = agent.run("Classify this: ''")
        
        # Should complete without crashing
        assert response is not None
    
    @pytest.mark.integration
    def test_conversation_maintains_context(self, full_agent):
        """Test that conversation history is maintained."""
        agent, mock = full_agent
        
        # First turn
        mock.add_response(text="I can help you analyze sentiment. What text would you like me to classify?")
        agent.run("Can you help me with sentiment analysis?")
        
        # Second turn with actual classification
        mock.add_response(
            tool_call={
                "name": "classify_text",
                "id": "t1",
                "input": {"text": "wonderful experience"}
            }
        )
        mock.add_response(text="That text is positive!")
        
        response = agent.run("Classify 'wonderful experience'")
        
        # Conversation should have history from both turns
        assert len(agent.conversation_history) >= 4


# =============================================================================
# TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-fail-under=80",  # Require 80% coverage
    ])
