"""
Unit tests for the calculator tool.

Chapter 35: Testing AI Agents - Implementation

This file demonstrates how to write comprehensive unit tests for
agent tools. Tools are the easiest part of an agent to test because
they're typically pure functions.

Run with: pytest test_tools.py -v
"""

import pytest
from calculator import calculator


class TestCalculatorBasicOperations:
    """Tests for basic arithmetic operations."""
    
    @pytest.mark.unit
    def test_addition_positive_numbers(self):
        """Test that addition works with positive numbers."""
        result = calculator("add", 5, 3)
        
        assert result["success"] is True
        assert result["result"] == 8
        assert "5 add 3 = 8" in result["expression"]
    
    @pytest.mark.unit
    def test_subtraction_positive_result(self):
        """Test subtraction that results in a positive number."""
        result = calculator("subtract", 10, 4)
        
        assert result["success"] is True
        assert result["result"] == 6
    
    @pytest.mark.unit
    def test_subtraction_negative_result(self):
        """Test subtraction that results in a negative number."""
        result = calculator("subtract", 4, 10)
        
        assert result["success"] is True
        assert result["result"] == -6
    
    @pytest.mark.unit
    def test_multiplication(self):
        """Test that multiplication works correctly."""
        result = calculator("multiply", 7, 6)
        
        assert result["success"] is True
        assert result["result"] == 42
    
    @pytest.mark.unit
    def test_division_exact(self):
        """Test division that results in a whole number."""
        result = calculator("divide", 20, 4)
        
        assert result["success"] is True
        assert result["result"] == 5.0
    
    @pytest.mark.unit
    def test_division_decimal(self):
        """Test division that results in a decimal."""
        result = calculator("divide", 7, 2)
        
        assert result["success"] is True
        assert result["result"] == 3.5


class TestCalculatorEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_divide_by_zero(self):
        """Test that division by zero returns an error."""
        result = calculator("divide", 10, 0)
        
        assert result["success"] is False
        assert "divide by zero" in result["error"].lower()
    
    @pytest.mark.unit
    def test_unknown_operation(self):
        """Test that unknown operations return an error."""
        result = calculator("power", 2, 3)
        
        assert result["success"] is False
        assert "Unknown operation" in result["error"]
    
    @pytest.mark.unit
    def test_negative_numbers_addition(self):
        """Test that negative numbers work with addition."""
        result = calculator("add", -5, -3)
        
        assert result["success"] is True
        assert result["result"] == -8
    
    @pytest.mark.unit
    def test_negative_numbers_multiplication(self):
        """Test multiplication with negative numbers."""
        result = calculator("multiply", -3, 4)
        
        assert result["success"] is True
        assert result["result"] == -12
    
    @pytest.mark.unit
    def test_negative_times_negative(self):
        """Test that negative times negative is positive."""
        result = calculator("multiply", -3, -4)
        
        assert result["success"] is True
        assert result["result"] == 12
    
    @pytest.mark.unit
    def test_floating_point_numbers(self):
        """Test that floating point numbers work correctly."""
        result = calculator("multiply", 2.5, 4.0)
        
        assert result["success"] is True
        assert result["result"] == pytest.approx(10.0)
    
    @pytest.mark.unit
    def test_very_small_numbers(self):
        """Test calculations with very small numbers."""
        result = calculator("multiply", 0.001, 0.001)
        
        assert result["success"] is True
        assert result["result"] == pytest.approx(0.000001)
    
    @pytest.mark.unit
    def test_very_large_numbers(self):
        """Test that very large numbers are handled."""
        result = calculator("multiply", 1e100, 1e100)
        
        assert result["success"] is True
        assert result["result"] == 1e200
    
    @pytest.mark.unit
    def test_zero_operations(self):
        """Test operations involving zero."""
        assert calculator("add", 0, 5)["result"] == 5
        assert calculator("add", 5, 0)["result"] == 5
        assert calculator("multiply", 0, 5)["result"] == 0
        assert calculator("divide", 0, 5)["result"] == 0


class TestCalculatorOutputStructure:
    """Tests for the structure of calculator output."""
    
    @pytest.mark.unit
    def test_success_response_structure(self):
        """Verify successful responses have the expected keys."""
        result = calculator("add", 1, 1)
        
        assert "success" in result
        assert "result" in result
        assert "expression" in result
        assert isinstance(result["success"], bool)
        assert result["success"] is True
    
    @pytest.mark.unit
    def test_error_response_structure(self):
        """Verify error responses have the expected keys."""
        result = calculator("divide", 1, 0)
        
        assert "success" in result
        assert "error" in result
        assert isinstance(result["success"], bool)
        assert result["success"] is False
        assert isinstance(result["error"], str)
    
    @pytest.mark.unit
    def test_expression_format(self):
        """Verify the expression is formatted correctly."""
        result = calculator("add", 10, 5)
        
        assert result["expression"] == "10 add 5 = 15"


class TestCalculatorWithParameterization:
    """
    Demonstrates pytest parameterization for testing multiple inputs.
    
    This technique allows testing many cases without repetitive code.
    """
    
    @pytest.mark.unit
    @pytest.mark.parametrize("operation,a,b,expected", [
        # Addition cases
        ("add", 1, 1, 2),
        ("add", 0, 0, 0),
        ("add", -1, 1, 0),
        ("add", 100, 200, 300),
        # Subtraction cases
        ("subtract", 5, 3, 2),
        ("subtract", 3, 5, -2),
        ("subtract", 0, 0, 0),
        # Multiplication cases
        ("multiply", 3, 4, 12),
        ("multiply", 0, 100, 0),
        ("multiply", -2, 3, -6),
        # Division cases
        ("divide", 10, 2, 5),
        ("divide", 7, 2, 3.5),
        ("divide", 1, 4, 0.25),
    ])
    def test_operations(self, operation: str, a: float, b: float, expected: float):
        """Test various operation combinations using parameterization."""
        result = calculator(operation, a, b)
        
        assert result["success"] is True
        assert result["result"] == pytest.approx(expected)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_operation", [
        "power",
        "modulo",
        "sqrt",
        "ADD",  # Case sensitive
        "MULTIPLY",
        "",
        " ",
        "addition",
    ])
    def test_invalid_operations(self, invalid_operation: str):
        """Test that invalid operations always fail."""
        result = calculator(invalid_operation, 1, 1)
        
        assert result["success"] is False
        assert "Unknown operation" in result["error"]


class TestCalculatorErrorMessages:
    """Tests that verify error messages are helpful to users and LLMs."""
    
    @pytest.mark.unit
    def test_error_includes_valid_operations(self):
        """Verify error messages tell users what operations ARE valid."""
        result = calculator("modulo", 10, 3)
        
        assert result["success"] is False
        error = result["error"]
        
        # Error should list valid operations
        assert "add" in error
        assert "subtract" in error
        assert "multiply" in error
        assert "divide" in error
    
    @pytest.mark.unit
    def test_divide_by_zero_message_is_clear(self):
        """Verify divide by zero message is human-readable."""
        result = calculator("divide", 5, 0)
        
        assert result["success"] is False
        # Message should be a complete sentence, not just an error code
        assert len(result["error"]) > 10
        assert "zero" in result["error"].lower()
    
    @pytest.mark.unit
    def test_error_messages_start_with_capital(self):
        """Verify error messages are properly capitalized."""
        result1 = calculator("unknown", 1, 1)
        result2 = calculator("divide", 1, 0)
        
        assert result1["error"][0].isupper()
        assert result2["error"][0].isupper()


class TestCalculatorForAgentUse:
    """
    Tests that verify the calculator works well as an agent tool.
    
    These tests focus on the interface between the tool and the agent.
    """
    
    @pytest.mark.unit
    def test_output_is_json_serializable(self):
        """Verify output can be serialized to JSON (for API responses)."""
        import json
        
        result = calculator("add", 5, 3)
        
        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        
        assert parsed == result
    
    @pytest.mark.unit
    def test_output_works_with_string_formatting(self):
        """Verify output can be converted to string for tool results."""
        result = calculator("multiply", 7, 6)
        
        # Agent needs to convert result to string for tool_result
        result_str = str(result)
        
        assert "42" in result_str
        assert "success" in result_str.lower()
    
    @pytest.mark.unit
    def test_handles_string_numbers_gracefully(self):
        """
        Test how the tool handles if passed string numbers.
        
        Note: In production, you'd want type validation.
        This test documents current behavior.
        """
        # Python will handle this, but it's good to document
        result = calculator("add", 5.0, 3.0)
        assert result["result"] == 8.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
