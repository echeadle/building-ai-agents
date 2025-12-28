"""
Exercise Solution: Prompt Testing Framework

Build a framework that systematically tests and compares different prompts.

Appendix D: Prompt Engineering for Agents
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import json
import time

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class TestCase:
    """Represents a test case for prompt evaluation."""
    name: str
    query: str
    expected_tool: str  # Which tool should be used
    expected_behavior: str  # Description of expected behavior
    should_terminate: bool  # Should agent stop after first response


@dataclass
class PromptVariant:
    """Represents a prompt to test."""
    name: str
    system_prompt: str
    description: str


@dataclass
class TestResult:
    """Results from testing a prompt variant."""
    prompt_name: str
    test_name: str
    passed: bool
    tool_used: str | None
    stop_reason: str
    response_text: str
    error: str | None = None


class PromptTestingFramework:
    """Framework for systematically testing and comparing prompts."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.results: List[TestResult] = []
        
        # Define test tools
        self.tools = [
            {
                "name": "web_search",
                "description": "Searches the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "calculator",
                "description": "Performs arithmetic calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"]
                }
            },
            {
                "name": "get_weather",
                "description": "Gets weather forecast",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        ]
    
    def create_test_suite(self) -> List[TestCase]:
        """Define test cases for evaluation."""
        return [
            TestCase(
                name="simple_calculation",
                query="What's 2 + 2?",
                expected_tool="none",  # Should answer directly
                expected_behavior="Answer without using calculator",
                should_terminate=True
            ),
            TestCase(
                name="complex_calculation",
                query="What's 17% of 892?",
                expected_tool="calculator",
                expected_behavior="Use calculator for non-trivial math",
                should_terminate=True
            ),
            TestCase(
                name="current_information",
                query="What's the current weather in Seattle?",
                expected_tool="get_weather",
                expected_behavior="Use weather tool for current data",
                should_terminate=True
            ),
            TestCase(
                name="research_question",
                query="What are recent developments in quantum computing?",
                expected_tool="web_search",
                expected_behavior="Search for recent information",
                should_terminate=True
            ),
            TestCase(
                name="general_knowledge",
                query="What is photosynthesis?",
                expected_tool="none",
                expected_behavior="Answer from knowledge, no tools needed",
                should_terminate=True
            )
        ]
    
    def create_prompt_variants(self) -> List[PromptVariant]:
        """Define prompt variants to test."""
        return [
            PromptVariant(
                name="minimal",
                system_prompt="You are a helpful assistant with access to tools.",
                description="Minimal prompt with no guidance"
            ),
            PromptVariant(
                name="tool_aware",
                system_prompt="""You are a helpful assistant with access to tools.

Guidelines:
1. Use tools when you need information you don't have
2. Answer directly for simple questions
3. Use calculator for non-trivial arithmetic
4. Use web_search for current events

Important: Don't use tools for trivial queries.""",
                description="Prompt with basic tool usage guidelines"
            ),
            PromptVariant(
                name="with_examples",
                system_prompt="""You are a helpful assistant with access to tools.

Examples of good tool usage:

Example 1:
User: "What's 2 + 2?"
Response: "4" (no tool needed)

Example 2:
User: "What's 17% of 892?"
Action: calculator(multiply, 892, 0.17)
Response: "17% of 892 is 151.64"

Example 3:
User: "What is photosynthesis?"
Response: [Answer from knowledge] (no tool needed)

Example 4:
User: "Recent news about AI?"
Action: web_search("AI news")
Response: [Based on search results]

Use tools wisely - only when truly needed.""",
                description="Prompt with few-shot examples"
            ),
            PromptVariant(
                name="explicit_reasoning",
                system_prompt="""You are a helpful assistant with access to tools.

For each query, think through:
[THINKING]
Does this need a tool? Why/why not?
[/THINKING]

Then respond.

Tool usage rules:
- Simple math (2+2): Answer directly
- Complex math: Use calculator
- Current info: Use web_search or get_weather
- General knowledge: Answer from training

Be smart about tool usage.""",
                description="Prompt encouraging explicit reasoning"
            )
        ]
    
    def run_test(
        self,
        prompt_variant: PromptVariant,
        test_case: TestCase
    ) -> TestResult:
        """Run a single test case with a prompt variant."""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=prompt_variant.system_prompt,
                tools=self.tools,
                messages=[
                    {"role": "user", "content": test_case.query}
                ]
            )
            
            # Extract what tool was used (if any)
            tool_used = None
            response_text = ""
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_used = block.name
                elif block.type == "text":
                    response_text = block.text
            
            # Determine if test passed
            if test_case.expected_tool == "none":
                passed = (tool_used is None)
            else:
                passed = (tool_used == test_case.expected_tool)
            
            return TestResult(
                prompt_name=prompt_variant.name,
                test_name=test_case.name,
                passed=passed,
                tool_used=tool_used,
                stop_reason=response.stop_reason,
                response_text=response_text[:200]  # Truncate for display
            )
            
        except Exception as e:
            return TestResult(
                prompt_name=prompt_variant.name,
                test_name=test_case.name,
                passed=False,
                tool_used=None,
                stop_reason="error",
                response_text="",
                error=str(e)
            )
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run all test cases against all prompt variants."""
        test_cases = self.create_test_suite()
        prompt_variants = self.create_prompt_variants()
        
        print("="*70)
        print("PROMPT TESTING FRAMEWORK")
        print("="*70)
        print(f"\nTesting {len(prompt_variants)} prompt variants")
        print(f"Against {len(test_cases)} test cases")
        print(f"Total tests: {len(prompt_variants) * len(test_cases)}\n")
        
        results = []
        
        for prompt in prompt_variants:
            print(f"\n{'='*70}")
            print(f"Testing: {prompt.name}")
            print(f"Description: {prompt.description}")
            print(f"{'='*70}")
            
            prompt_results = []
            
            for test in test_cases:
                print(f"\n  Test: {test.name}")
                print(f"  Query: {test.query}")
                print(f"  Expected: {test.expected_behavior}")
                
                result = self.run_test(prompt, test)
                prompt_results.append(result)
                results.append(result)
                
                # Display result
                status = "✓ PASS" if result.passed else "✗ FAIL"
                print(f"  Result: {status}")
                print(f"  Tool used: {result.tool_used or 'none'}")
                
                if result.error:
                    print(f"  Error: {result.error}")
                
                # Rate limit consideration
                time.sleep(0.5)
            
            # Summary for this prompt
            passed = sum(1 for r in prompt_results if r.passed)
            total = len(prompt_results)
            print(f"\n  Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        return self.generate_report(results)
    
    def generate_report(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate a comprehensive report of test results."""
        report = {
            "summary": {},
            "by_prompt": {},
            "by_test": {}
        }
        
        # Overall summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        report["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "pass_rate": f"{passed_tests/total_tests*100:.1f}%"
        }
        
        # Group by prompt
        for result in results:
            if result.prompt_name not in report["by_prompt"]:
                report["by_prompt"][result.prompt_name] = {
                    "passed": 0,
                    "failed": 0,
                    "tests": []
                }
            
            if result.passed:
                report["by_prompt"][result.prompt_name]["passed"] += 1
            else:
                report["by_prompt"][result.prompt_name]["failed"] += 1
            
            report["by_prompt"][result.prompt_name]["tests"].append({
                "test": result.test_name,
                "passed": result.passed,
                "tool_used": result.tool_used
            })
        
        # Group by test
        for result in results:
            if result.test_name not in report["by_test"]:
                report["by_test"][result.test_name] = {
                    "prompts_passed": [],
                    "prompts_failed": []
                }
            
            if result.passed:
                report["by_test"][result.test_name]["prompts_passed"].append(result.prompt_name)
            else:
                report["by_test"][result.test_name]["prompts_failed"].append(result.prompt_name)
        
        return report
    
    def print_final_report(self, report: Dict[str, Any]) -> None:
        """Print a formatted final report."""
        print("\n\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        
        # Overall summary
        print("\nOVERALL RESULTS:")
        print(f"  Total tests: {report['summary']['total_tests']}")
        print(f"  Passed: {report['summary']['passed']}")
        print(f"  Failed: {report['summary']['failed']}")
        print(f"  Pass rate: {report['summary']['pass_rate']}")
        
        # By prompt
        print("\n\nRESULTS BY PROMPT:")
        for prompt_name, data in report['by_prompt'].items():
            total = data['passed'] + data['failed']
            rate = data['passed'] / total * 100
            print(f"\n  {prompt_name}:")
            print(f"    Passed: {data['passed']}/{total} ({rate:.1f}%)")
            
            # Show which tests failed
            failed_tests = [t['test'] for t in data['tests'] if not t['passed']]
            if failed_tests:
                print(f"    Failed tests: {', '.join(failed_tests)}")
        
        # Best performing prompt
        best_prompt = max(
            report['by_prompt'].items(),
            key=lambda x: x[1]['passed'] / (x[1]['passed'] + x[1]['failed'])
        )
        print(f"\n\nBEST PERFORMING PROMPT: {best_prompt[0]}")
        
        # Hardest test
        hardest_test = min(
            report['by_test'].items(),
            key=lambda x: len(x[1]['prompts_passed'])
        )
        print(f"HARDEST TEST: {hardest_test[0]}")
        print(f"  Only {len(hardest_test[1]['prompts_passed'])} prompts passed")
        
        # Save report to file
        with open("/home/claude/appendix-d-prompt-engineering/test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("\n\nFull report saved to: test_report.json")


def main():
    """Run the prompt testing framework."""
    framework = PromptTestingFramework()
    
    print("Prompt Testing Framework")
    print("=" * 70)
    print("This framework systematically tests different prompt variants")
    print("to determine which performs best across various test cases.")
    print()
    
    # Run evaluation
    report = framework.run_full_evaluation()
    
    # Print final report
    framework.print_final_report(report)
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
From this testing, you can see:

1. PROMPT QUALITY MATTERS
   - Different prompts show dramatically different pass rates
   - Even small changes can have big impacts

2. EXAMPLES HELP
   - Prompts with few-shot examples typically perform better
   - Examples clarify intent better than rules alone

3. EXPLICIT REASONING AIDS DEBUGGING
   - When tests fail, reasoning shows WHY
   - Easier to fix prompts with visible reasoning

4. START SIMPLE, TEST THOROUGHLY
   - Begin with minimal prompt
   - Add complexity based on test failures
   - Validate each improvement

5. SYSTEMATIC TESTING IS ESSENTIAL
   - Manual testing misses edge cases
   - Automated testing catches regressions
   - Quantitative results guide improvements

Use this framework pattern for YOUR agents!
""")


if __name__ == "__main__":
    main()
