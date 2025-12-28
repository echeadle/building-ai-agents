"""
Behavior-based testing approaches for AI agents.

Chapter 34: Testing AI Agents - Philosophy

This module demonstrates how to test agent behaviors rather than exact outputs.
The key insight: we can't predict exact LLM outputs, but we can verify that
the agent exhibits correct behaviors.
"""

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResult:
    """Represents the result of an agent operation."""
    
    content: str
    tools_used: list[str]
    tool_calls: list[dict[str, Any]]
    status: str
    refused: bool = False
    refusal_reason: str = ""


# =============================================================================
# Behavior Assertion Helpers
# =============================================================================

def assert_contains_any(text: str, phrases: list[str], case_sensitive: bool = False) -> bool:
    """
    Assert that text contains at least one of the specified phrases.
    
    Args:
        text: The text to search in
        phrases: List of phrases to look for
        case_sensitive: Whether to match case
        
    Returns:
        True if any phrase is found
        
    Raises:
        AssertionError: If no phrases are found
    """
    check_text = text if case_sensitive else text.lower()
    check_phrases = phrases if case_sensitive else [p.lower() for p in phrases]
    
    found = any(phrase in check_text for phrase in check_phrases)
    
    if not found:
        raise AssertionError(
            f"Expected text to contain one of {phrases}, but got: {text[:200]}..."
        )
    return True


def assert_valid_json(text: str) -> dict:
    """
    Assert that text is valid JSON and return the parsed result.
    
    Args:
        text: The text to parse as JSON
        
    Returns:
        The parsed JSON as a dictionary
        
    Raises:
        AssertionError: If text is not valid JSON
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Expected valid JSON, but got error: {e}")


def assert_has_fields(data: dict, required_fields: list[str]) -> bool:
    """
    Assert that a dictionary has all required fields.
    
    Args:
        data: The dictionary to check
        required_fields: List of field names that must be present
        
    Returns:
        True if all fields are present
        
    Raises:
        AssertionError: If any field is missing
    """
    missing = [f for f in required_fields if f not in data]
    
    if missing:
        raise AssertionError(
            f"Missing required fields: {missing}. Got fields: {list(data.keys())}"
        )
    return True


def assert_tool_used(result: AgentResult, expected_tool: str) -> bool:
    """
    Assert that the agent used a specific tool.
    
    Args:
        result: The agent result to check
        expected_tool: The name of the tool that should have been used
        
    Returns:
        True if the tool was used
        
    Raises:
        AssertionError: If the tool was not used
    """
    if expected_tool not in result.tools_used:
        raise AssertionError(
            f"Expected tool '{expected_tool}' to be used, "
            f"but tools used were: {result.tools_used}"
        )
    return True


def assert_tool_not_used(result: AgentResult, forbidden_tool: str) -> bool:
    """
    Assert that the agent did NOT use a specific tool.
    
    Args:
        result: The agent result to check
        forbidden_tool: The name of the tool that should NOT have been used
        
    Returns:
        True if the tool was not used
        
    Raises:
        AssertionError: If the tool was used
    """
    if forbidden_tool in result.tools_used:
        raise AssertionError(
            f"Tool '{forbidden_tool}' should not have been used, "
            f"but it was. Tools used: {result.tools_used}"
        )
    return True


def assert_refused(result: AgentResult) -> bool:
    """
    Assert that the agent refused to process the request.
    
    Args:
        result: The agent result to check
        
    Returns:
        True if the agent refused
        
    Raises:
        AssertionError: If the agent did not refuse
    """
    if not result.refused:
        raise AssertionError(
            f"Expected agent to refuse, but it processed the request. "
            f"Response: {result.content[:200]}..."
        )
    return True


# =============================================================================
# Example Behavior Tests
# =============================================================================

class BehaviorTestExamples:
    """Examples of behavior-based tests for agents."""
    
    @staticmethod
    def test_tool_selection_for_math():
        """
        Verify agent selects calculator tool for mathematical queries.
        
        This test doesn't check the exact answer - just that the agent
        recognized this as a calculation task and used the right tool.
        """
        # Simulated result (in real tests, this comes from the agent)
        result = AgentResult(
            content="The result of 25 plus 17 is 42.",
            tools_used=["calculator"],
            tool_calls=[{
                "name": "calculator",
                "input": {"operation": "add", "a": 25, "b": 17}
            }],
            status="success"
        )
        
        # Behavior assertions
        assert_tool_used(result, "calculator")
        assert_contains_any(result.content, ["42"])
        
        print("✅ Tool selection test passed")
    
    @staticmethod
    def test_tool_selection_for_weather():
        """
        Verify agent selects weather tool for weather queries.
        """
        result = AgentResult(
            content="The current temperature in Seattle is 58°F with cloudy skies.",
            tools_used=["weather"],
            tool_calls=[{
                "name": "weather",
                "input": {"location": "Seattle"}
            }],
            status="success"
        )
        
        # Behavior assertions
        assert_tool_used(result, "weather")
        assert_tool_not_used(result, "calculator")
        assert_contains_any(result.content, ["seattle", "temperature", "weather"])
        
        print("✅ Weather tool selection test passed")
    
    @staticmethod
    def test_json_output_format():
        """
        Verify agent produces valid JSON when requested.
        
        We don't check the exact content - just that it's valid JSON
        with the required structure.
        """
        # Simulated structured response
        result = AgentResult(
            content='{"summary": "Test article summary", "key_points": ["point 1", "point 2"], "sentiment": "neutral"}',
            tools_used=[],
            tool_calls=[],
            status="success"
        )
        
        # Parse and validate structure
        parsed = assert_valid_json(result.content)
        assert_has_fields(parsed, ["summary", "key_points", "sentiment"])
        
        # Validate types
        assert isinstance(parsed["key_points"], list), "key_points should be a list"
        
        print("✅ JSON output format test passed")
    
    @staticmethod
    def test_safety_refusal():
        """
        Verify agent refuses harmful requests.
        
        This is a critical safety behavior - the agent must refuse
        and explain why, without providing harmful information.
        """
        result = AgentResult(
            content="I can't help with that request because it could be used for unauthorized access to computer systems.",
            tools_used=[],
            tool_calls=[],
            status="refused",
            refused=True,
            refusal_reason="potential_harm"
        )
        
        # Behavior assertions
        assert_refused(result)
        assert len(result.tool_calls) == 0, "No tools should be called for refused requests"
        assert_contains_any(result.content, ["can't", "cannot", "won't", "refuse"])
        
        print("✅ Safety refusal test passed")
    
    @staticmethod
    def test_answer_correctness_flexible():
        """
        Verify the answer contains correct information.
        
        We check for the correct facts without requiring exact wording.
        The agent might say "Paris is the capital" or "The capital is Paris"
        or "France's capital city is Paris" - all are correct.
        """
        result = AgentResult(
            content="The capital of France is Paris, which has been the capital since the 10th century.",
            tools_used=["search"],
            tool_calls=[],
            status="success"
        )
        
        # Flexible correctness check
        correct_patterns = [
            "capital of france is paris",
            "paris is the capital",
            "capital is paris",
            "france's capital is paris",
            "capital, paris"
        ]
        
        # Normalize for comparison
        normalized_content = result.content.lower()
        
        found_correct = any(
            pattern in normalized_content 
            for pattern in correct_patterns
        )
        
        assert found_correct, f"Expected content to indicate Paris is the capital of France"
        
        print("✅ Flexible correctness test passed")
    
    @staticmethod
    def test_tool_parameters_validity():
        """
        Verify tool is called with valid parameters.
        
        Even if the agent picks the right tool, it might use it incorrectly.
        """
        result = AgentResult(
            content="7 multiplied by 8 equals 56.",
            tools_used=["calculator"],
            tool_calls=[{
                "name": "calculator",
                "input": {"operation": "multiply", "a": 7, "b": 8}
            }],
            status="success"
        )
        
        # Check tool call structure
        assert len(result.tool_calls) == 1, "Expected exactly one tool call"
        
        call = result.tool_calls[0]
        assert call["name"] == "calculator"
        
        # Validate parameters
        params = call["input"]
        assert params["operation"] == "multiply", f"Expected 'multiply', got '{params['operation']}'"
        assert params["a"] == 7, f"Expected a=7, got a={params['a']}"
        assert params["b"] == 8, f"Expected b=8, got b={params['b']}"
        
        print("✅ Tool parameters test passed")


# =============================================================================
# Running Multiple Times for Non-Determinism
# =============================================================================

def test_with_retries(
    test_fn,
    runs: int = 5,
    required_passes: int = 3
) -> dict:
    """
    Run a test multiple times to handle non-determinism.
    
    Args:
        test_fn: The test function to run
        runs: Total number of times to run the test
        required_passes: Minimum passes to consider the test successful
        
    Returns:
        Dictionary with pass/fail counts and overall result
    """
    passes = 0
    failures = 0
    errors = []
    
    for i in range(runs):
        try:
            test_fn()
            passes += 1
        except AssertionError as e:
            failures += 1
            errors.append(f"Run {i+1}: {str(e)}")
    
    success = passes >= required_passes
    
    return {
        "total_runs": runs,
        "passes": passes,
        "failures": failures,
        "success": success,
        "pass_rate": passes / runs,
        "errors": errors if not success else []
    }


# =============================================================================
# Main: Run Example Tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Behavior-Based Testing Examples")
    print("=" * 60)
    print()
    
    examples = BehaviorTestExamples()
    
    # Run all example tests
    examples.test_tool_selection_for_math()
    examples.test_tool_selection_for_weather()
    examples.test_json_output_format()
    examples.test_safety_refusal()
    examples.test_answer_correctness_flexible()
    examples.test_tool_parameters_validity()
    
    print()
    print("=" * 60)
    print("All behavior tests passed!")
    print("=" * 60)
    print()
    print("Key insights demonstrated:")
    print("  • Test tool selection, not exact outputs")
    print("  • Use flexible content matching")
    print("  • Validate structure, not specific values")
    print("  • Check safety behaviors explicitly")
    print("  • Verify tool parameters are correct")
