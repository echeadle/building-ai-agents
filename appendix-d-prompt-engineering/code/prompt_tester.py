"""
Prompt testing framework for agents.

Appendix D: Prompt Engineering for Agents

This tool helps you test prompts against diverse scenarios to find weaknesses
before deploying your agent.
"""

import os
from typing import Any
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class TestCategory(Enum):
    """Categories for test cases."""
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    AMBIGUOUS = "ambiguous"
    MISSING_INFO = "missing_info"
    CONTRADICTORY = "contradictory"
    OFF_TOPIC = "off_topic"
    MALFORMED = "malformed"
    COMPLEX = "complex"
    FOLLOW_UP = "follow_up"
    STRESS_TEST = "stress_test"


@dataclass
class TestCase:
    """A single test case for prompt evaluation."""
    input: str
    category: TestCategory
    expected_behavior: str
    should_use_tool: bool | None = None
    expected_tool: str | None = None


@dataclass
class TestResult:
    """Result of running a test case."""
    test_case: TestCase
    actual_response: str
    passed: bool
    notes: str = ""
    tokens_used: int = 0


class PromptTester:
    """
    Test prompts against diverse scenarios.
    
    Usage:
        tester = PromptTester(system_prompt, tools)
        results = tester.run_test_suite(test_cases)
        tester.print_report(results)
    """
    
    def __init__(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the tester.
        
        Args:
            system_prompt: The system prompt to test
            tools: Optional list of tool definitions
            model: Model to use for testing
        """
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def run_test_case(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: The test to run
            
        Returns:
            TestResult with outcome
        """
        try:
            # Make API call
            params = {
                "model": self.model,
                "max_tokens": 1024,
                "system": self.system_prompt,
                "messages": [{"role": "user", "content": test_case.input}]
            }
            
            if self.tools:
                params["tools"] = self.tools
            
            response = self.client.messages.create(**params)
            
            # Extract response text and check for tool use
            response_text = ""
            tool_used = None
            
            for block in response.content:
                if block.type == "text":
                    response_text += block.text
                elif block.type == "tool_use":
                    tool_used = block.name
            
            # Evaluate the response
            passed = self._evaluate_response(
                test_case, 
                response_text, 
                tool_used
            )
            
            return TestResult(
                test_case=test_case,
                actual_response=response_text,
                passed=passed,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens
            )
        
        except Exception as e:
            return TestResult(
                test_case=test_case,
                actual_response=f"Error: {str(e)}",
                passed=False,
                notes=f"Exception occurred: {str(e)}"
            )
    
    def _evaluate_response(
        self, 
        test_case: TestCase, 
        response: str, 
        tool_used: str | None
    ) -> bool:
        """
        Evaluate if the response meets expectations.
        
        This is a simple heuristic check. In production, you'd want
        more sophisticated evaluation.
        """
        # Check tool usage if specified
        if test_case.should_use_tool is not None:
            if test_case.should_use_tool and tool_used is None:
                return False
            if not test_case.should_use_tool and tool_used is not None:
                return False
        
        if test_case.expected_tool and tool_used != test_case.expected_tool:
            return False
        
        # Check if key phrases from expected behavior appear in response
        # This is a basic check - real evaluation would be more sophisticated
        expected_lower = test_case.expected_behavior.lower()
        response_lower = response.lower()
        
        # For off-topic cases, expect refusal or redirection
        if test_case.category == TestCategory.OFF_TOPIC:
            refusal_phrases = [
                "can't help with that",
                "outside my",
                "not able to",
                "cannot assist",
                "not appropriate"
            ]
            return any(phrase in response_lower for phrase in refusal_phrases)
        
        # For missing info cases, expect clarification request
        if test_case.category == TestCategory.MISSING_INFO:
            clarification_phrases = [
                "could you",
                "can you provide",
                "need more",
                "which",
                "what"
            ]
            return any(phrase in response_lower for phrase in clarification_phrases)
        
        # For other cases, this is a placeholder
        # Real evaluation would check semantic similarity
        return True
    
    def run_test_suite(self, test_cases: list[TestCase]) -> list[TestResult]:
        """
        Run all test cases.
        
        Args:
            test_cases: List of tests to run
            
        Returns:
            List of test results
        """
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Running test {i}/{len(test_cases)}: {test_case.category.value}")
            result = self.run_test_case(test_case)
            results.append(result)
        
        return results
    
    def print_report(self, results: list[TestResult]) -> None:
        """
        Print a summary report of test results.
        
        Args:
            results: List of test results
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        total_tokens = sum(r.tokens_used for r in results)
        
        print("\n" + "=" * 70)
        print("PROMPT TEST REPORT")
        print("=" * 70)
        print(f"\nTotal tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Total tokens used: {total_tokens:,}")
        
        # Category breakdown
        print("\n" + "-" * 70)
        print("RESULTS BY CATEGORY")
        print("-" * 70)
        
        categories = {}
        for result in results:
            cat = result.test_case.category.value
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if result.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        for category, counts in sorted(categories.items()):
            total_cat = counts["passed"] + counts["failed"]
            pass_rate = counts["passed"] / total_cat * 100
            status = "✓" if pass_rate == 100 else "✗"
            print(f"{status} {category:20s}: {counts['passed']}/{total_cat} passed ({pass_rate:.0f}%)")
        
        # Show failures
        failures = [r for r in results if not r.passed]
        if failures:
            print("\n" + "-" * 70)
            print("FAILED TESTS")
            print("-" * 70)
            
            for i, result in enumerate(failures, 1):
                print(f"\n{i}. {result.test_case.category.value}")
                print(f"   Input: {result.test_case.input}")
                print(f"   Expected: {result.test_case.expected_behavior}")
                print(f"   Got: {result.actual_response[:100]}...")
                if result.notes:
                    print(f"   Notes: {result.notes}")


def create_sample_test_suite() -> list[TestCase]:
    """
    Create a sample test suite for demonstration.
    
    Returns:
        List of test cases covering the 10 test categories
    """
    return [
        # 1. Happy path
        TestCase(
            input="What's the weather in San Francisco?",
            category=TestCategory.HAPPY_PATH,
            expected_behavior="Use weather tool with location San Francisco",
            should_use_tool=True,
            expected_tool="get_weather"
        ),
        
        # 2. Edge case
        TestCase(
            input="What's the weather at the North Pole?",
            category=TestCategory.EDGE_CASE,
            expected_behavior="Handle unusual location appropriately",
            should_use_tool=True,
            expected_tool="get_weather"
        ),
        
        # 3. Ambiguous
        TestCase(
            input="What's it like in Paris?",
            category=TestCategory.AMBIGUOUS,
            expected_behavior="Clarify if asking about weather or something else",
            should_use_tool=None
        ),
        
        # 4. Missing information
        TestCase(
            input="What's the weather?",
            category=TestCategory.MISSING_INFO,
            expected_behavior="Ask which location",
            should_use_tool=False
        ),
        
        # 5. Contradictory
        TestCase(
            input="What's the weather in London, but don't check any weather sources?",
            category=TestCategory.CONTRADICTORY,
            expected_behavior="Clarify the contradiction",
            should_use_tool=None
        ),
        
        # 6. Off topic
        TestCase(
            input="Write me a poem about clouds",
            category=TestCategory.OFF_TOPIC,
            expected_behavior="Decline politely or redirect",
            should_use_tool=False
        ),
        
        # 7. Malformed
        TestCase(
            input="weahter in sanfransciso???",
            category=TestCategory.MALFORMED,
            expected_behavior="Understand intent despite typos",
            should_use_tool=True,
            expected_tool="get_weather"
        ),
        
        # 8. Complex
        TestCase(
            input="Compare the weather in New York, London, and Tokyo, and tell me which has the best conditions for outdoor activities today",
            category=TestCategory.COMPLEX,
            expected_behavior="Check weather for all three cities and compare",
            should_use_tool=True,
            expected_tool="get_weather"
        ),
        
        # 9. Follow-up (would need conversation context)
        TestCase(
            input="What about tomorrow?",
            category=TestCategory.FOLLOW_UP,
            expected_behavior="Ask for clarification since no prior context",
            should_use_tool=False
        ),
        
        # 10. Stress test
        TestCase(
            input="I need detailed weather forecasts for every major city in California including temperature, humidity, wind speed, precipitation, and UV index for the next 7 days, formatted as a table",
            category=TestCategory.STRESS_TEST,
            expected_behavior="Handle gracefully, possibly breaking into manageable steps",
            should_use_tool=True
        ),
    ]


# Example usage
if __name__ == "__main__":
    # Define a sample system prompt
    system_prompt = """You are a weather assistant that helps users check weather conditions.

Your capabilities:
- Check current weather for any location
- Provide temperature, conditions, and forecasts

Important rules:
- Always ask for location if not provided
- Only provide weather information, nothing else
- Use the get_weather tool for all weather queries

When responding:
- Be concise and factual
- Include relevant details (temperature, conditions)
- Suggest appropriate clothing or activities if relevant
"""
    
    # Define a sample tool
    tools = [
        {
            "name": "get_weather",
            "description": """Gets current weather for a specified location.
            
Use this tool when:
- User asks about weather conditions
- User asks about temperature or forecast
- User mentions a specific location with weather-related query

DO NOT use for:
- General questions about weather (explain instead)
- Historical weather data (not available)
""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location. Can include country for disambiguation."
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    # Create tester and run tests
    print("Creating prompt tester...")
    tester = PromptTester(system_prompt, tools)
    
    print("\nGenerating test cases...")
    test_cases = create_sample_test_suite()
    
    print(f"\nRunning {len(test_cases)} tests...")
    print("=" * 70)
    results = tester.run_test_suite(test_cases)
    
    # Print report
    tester.print_report(results)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on the test results, consider:

1. If ambiguous cases failed:
   → Add clarifying examples to system prompt
   
2. If edge cases failed:
   → Add explicit handling for unusual inputs
   
3. If off-topic tests failed:
   → Strengthen role definition and constraints
   
4. If complex cases failed:
   → Add multi-step reasoning guidance
   
5. If missing info cases failed:
   → Add explicit instructions to ask for clarification

Remember: Iterate on your prompt based on real failures!
""")
