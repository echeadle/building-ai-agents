"""
Property-based testing for agent tools using Hypothesis.

Chapter 35: Testing AI Agents - Implementation

Property-based testing generates random inputs to verify that
certain properties ALWAYS hold, regardless of the specific input.
This is powerful for finding edge cases you wouldn't think of.

Run with: pytest property_tests.py -v
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, example
from calculator import calculator


class TestCalculatorMathematicalProperties:
    """
    Property-based tests verifying mathematical properties.
    
    These tests verify that our calculator obeys the laws of arithmetic.
    """
    
    @pytest.mark.unit
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
    
    @pytest.mark.unit
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
    
    @pytest.mark.unit
    @given(a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_adding_zero_is_identity(self, a: float):
        """Property: Adding zero gives the same number (a + 0 == a)"""
        result = calculator("add", a, 0)
        
        assert result["success"] is True
        assert abs(result["result"] - a) < 1e-10
    
    @pytest.mark.unit
    @given(a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_multiplying_by_one_is_identity(self, a: float):
        """Property: Multiplying by one gives the same number (a × 1 == a)"""
        result = calculator("multiply", a, 1)
        
        assert result["success"] is True
        assert abs(result["result"] - a) < 1e-10
    
    @pytest.mark.unit
    @given(a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_multiplying_by_zero_gives_zero(self, a: float):
        """Property: Multiplying by zero gives zero (a × 0 == 0)"""
        result = calculator("multiply", a, 0)
        
        assert result["success"] is True
        assert result["result"] == 0
    
    @pytest.mark.unit
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    def test_subtraction_is_inverse_of_addition(self, a: float, b: float):
        """Property: (a + b) - b == a"""
        add_result = calculator("add", a, b)
        assume(add_result["success"])  # Skip if addition somehow fails
        
        sub_result = calculator("subtract", add_result["result"], b)
        
        assert sub_result["success"] is True
        # Allow for floating point error
        assert abs(sub_result["result"] - a) < 1e-9
    
    @pytest.mark.unit
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=0.001, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=0.001, max_value=1e10)
    )
    def test_division_is_inverse_of_multiplication(self, a: float, b: float):
        """Property: (a × b) / b == a"""
        mul_result = calculator("multiply", a, b)
        assume(mul_result["success"])
        
        div_result = calculator("divide", mul_result["result"], b)
        
        assert div_result["success"] is True
        # Allow for floating point error
        assert abs(div_result["result"] - a) < 1e-6
    
    @pytest.mark.unit
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        c=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    @settings(max_examples=50)  # Limit for performance
    def test_addition_is_associative(self, a: float, b: float, c: float):
        """Property: (a + b) + c == a + (b + c)"""
        # (a + b) + c
        ab = calculator("add", a, b)
        assume(ab["success"])
        abc_left = calculator("add", ab["result"], c)
        assume(abc_left["success"])
        
        # a + (b + c)
        bc = calculator("add", b, c)
        assume(bc["success"])
        abc_right = calculator("add", a, bc["result"])
        assume(abc_right["success"])
        
        # Should be equal (within floating point tolerance)
        assert abs(abc_left["result"] - abc_right["result"]) < 1e-6


class TestCalculatorOutputStructureProperties:
    """
    Property-based tests for output structure invariants.
    
    These tests verify that the calculator always returns properly
    structured output, regardless of input.
    """
    
    @pytest.mark.unit
    @given(
        operation=st.sampled_from(["add", "subtract", "multiply", "divide"]),
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    @settings(max_examples=200)
    def test_result_always_has_success_key(self, operation: str, a: float, b: float):
        """Property: Result always has a 'success' key."""
        result = calculator(operation, a, b)
        
        assert "success" in result
        assert isinstance(result["success"], bool)
    
    @pytest.mark.unit
    @given(
        operation=st.sampled_from(["add", "subtract", "multiply", "divide"]),
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    @settings(max_examples=200)
    def test_success_implies_result(self, operation: str, a: float, b: float):
        """Property: If success is True, result key must exist."""
        result = calculator(operation, a, b)
        
        if result["success"]:
            assert "result" in result
            assert isinstance(result["result"], (int, float))
    
    @pytest.mark.unit
    @given(
        operation=st.sampled_from(["add", "subtract", "multiply", "divide"]),
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
    )
    @settings(max_examples=200)
    def test_failure_implies_error(self, operation: str, a: float, b: float):
        """Property: If success is False, error key must exist."""
        result = calculator(operation, a, b)
        
        if not result["success"]:
            assert "error" in result
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0
    
    @pytest.mark.unit
    @given(operation=st.text(min_size=1, max_size=50))
    def test_unknown_operation_always_fails(self, operation: str):
        """Property: Unknown operations always return failure."""
        # Skip valid operations
        assume(operation not in ["add", "subtract", "multiply", "divide"])
        
        result = calculator(operation, 1, 1)
        
        assert result["success"] is False
        assert "error" in result


class TestCalculatorDivisionByZeroProperty:
    """
    Property-based tests specifically for division by zero.
    """
    
    @pytest.mark.unit
    @given(a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
    def test_division_by_zero_always_fails(self, a: float):
        """Property: Division by zero always fails, for any dividend."""
        result = calculator("divide", a, 0)
        
        assert result["success"] is False
        assert "zero" in result["error"].lower()
    
    @pytest.mark.unit
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=0.001, max_value=1e10)
    )
    def test_division_by_nonzero_succeeds(self, a: float, b: float):
        """Property: Division by non-zero always succeeds."""
        # Test positive divisor
        result = calculator("divide", a, b)
        assert result["success"] is True
        
        # Test negative divisor
        result = calculator("divide", a, -b)
        assert result["success"] is True


class TestAgentPropertyTests:
    """
    Property-based tests for agent behavior.
    
    These tests verify invariants of the agent's behavior.
    """
    
    @pytest.mark.integration
    @given(message=st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    def test_agent_always_returns_string(self, message: str):
        """Property: Agent always returns a string response."""
        from mock_llm import MockLLM
        from testable_agent import TestableAgent
        
        # Skip empty or whitespace-only messages
        assume(message.strip())
        
        mock = MockLLM()
        mock.add_response(text="Response to user message")
        
        agent = TestableAgent(llm_client=mock)
        response = agent.run(message)
        
        assert isinstance(response, str)
    
    @pytest.mark.integration
    @given(num_messages=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_conversation_history_grows_predictably(self, num_messages: int):
        """Property: Conversation history length is predictable."""
        from mock_llm import MockLLM
        from testable_agent import TestableAgent
        
        mock = MockLLM()
        for _ in range(num_messages):
            mock.add_response(text="Response")
        
        agent = TestableAgent(llm_client=mock)
        
        for i in range(num_messages):
            agent.run(f"Message {i}")
        
        # Each exchange adds at least 1 user message
        user_messages = [
            m for m in agent.conversation_history 
            if m["role"] == "user" and isinstance(m["content"], str)
        ]
        assert len(user_messages) == num_messages
    
    @pytest.mark.integration
    @given(max_iters=st.integers(min_value=1, max_value=10))
    @settings(max_examples=10)
    def test_agent_respects_max_iterations(self, max_iters: int):
        """Property: Agent never exceeds max_iterations."""
        from mock_llm import MockLLM
        from testable_agent import TestableAgent, AgentConfig
        from calculator import calculator
        
        mock = MockLLM()
        
        # Add more tool calls than max_iterations
        for i in range(max_iters + 10):
            mock.add_response(
                tool_call={
                    "name": "calculator",
                    "id": f"t{i}",
                    "input": {"operation": "add", "a": 1, "b": 1}
                }
            )
        
        agent = TestableAgent(
            llm_client=mock,
            tools={"calculator": calculator},
            config=AgentConfig(max_iterations=max_iters)
        )
        
        with pytest.raises(RuntimeError, match="exceeded maximum iterations"):
            agent.run("Loop forever")
        
        # Should have stopped at exactly max_iterations
        assert agent.get_iteration_count() == max_iters


class TestStringGenerationProperties:
    """
    Property tests for string handling edge cases.
    
    These tests use Hypothesis to find edge cases in string handling.
    """
    
    @pytest.mark.unit
    @given(
        operation=st.sampled_from(["add", "subtract", "multiply", "divide"]),
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=100)
    )
    def test_expression_contains_operands(self, operation: str, a: float, b: float):
        """Property: Expression string contains both operands."""
        result = calculator(operation, a, b)
        
        if result["success"]:
            expression = result["expression"]
            # Both numbers should appear in the expression (as strings)
            assert str(a) in expression or f"{a}" in expression
            assert str(b) in expression or f"{b}" in expression
    
    @pytest.mark.unit
    @given(
        operation=st.sampled_from(["add", "subtract", "multiply", "divide"]),
        a=st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
        b=st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=100)
    )
    def test_result_is_json_serializable(self, operation: str, a: float, b: float):
        """Property: Result is always JSON serializable."""
        import json
        
        result = calculator(operation, a, b)
        
        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        
        assert parsed["success"] == result["success"]


# Example-based tests to document specific edge cases found by Hypothesis
class TestDiscoveredEdgeCases:
    """
    Tests for specific edge cases discovered by property-based testing.
    
    When Hypothesis finds a failing case, add it here as a regression test.
    """
    
    @pytest.mark.unit
    @example(a=0.0, b=0.0)
    @given(
        a=st.floats(allow_nan=False, allow_infinity=False),
        b=st.floats(allow_nan=False, allow_infinity=False)
    )
    def test_zero_plus_zero(self, a: float, b: float):
        """Ensure 0 + 0 = 0 (Hypothesis might find this edge case)."""
        if a == 0.0 and b == 0.0:
            result = calculator("add", a, b)
            assert result["success"] is True
            assert result["result"] == 0.0
    
    @pytest.mark.unit
    def test_very_small_number_division(self):
        """Test division with very small numbers (potential precision issues)."""
        result = calculator("divide", 1e-300, 1e-300)
        assert result["success"] is True
        assert abs(result["result"] - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--hypothesis-show-statistics"])
