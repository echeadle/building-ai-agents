"""
Framework for testing system prompt effectiveness.

Demonstrates how to systematically test whether your system prompts
produce the expected behaviors across different scenarios.

Chapter 6: System Prompts and Persona Design
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


def test_system_prompt(
    system_prompt: str,
    test_cases: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 512
) -> list[dict]:
    """
    Test a system prompt against multiple scenarios.
    
    Args:
        system_prompt: The system prompt to test
        test_cases: List of dicts with 'name', 'input', and 'expected_behavior' keys
        model: Claude model to use
        max_tokens: Maximum tokens in responses
        
    Returns:
        List of results with test info and actual outputs
        
    Example test case:
        {
            "name": "Greeting test",
            "input": "Hello!",
            "expected_behavior": "Responds with friendly greeting"
        }
    """
    client = anthropic.Anthropic()
    results = []
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        test_name = test.get("name", f"Test {i}")
        print(f"[{i}/{len(test_cases)}] {test_name}")
        
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": test["input"]}]
            )
            
            output = response.content[0].text
            status = "completed"
            
        except Exception as e:
            output = f"ERROR: {str(e)}"
            status = "error"
        
        results.append({
            "name": test_name,
            "input": test["input"],
            "expected_behavior": test["expected_behavior"],
            "actual_output": output,
            "status": status
        })
    
    return results


def print_results(results: list[dict]) -> None:
    """
    Print test results in a readable format.
    
    Args:
        results: The results from test_system_prompt()
    """
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n{'─' * 70}")
        print(f"📋 Test: {result['name']}")
        print(f"{'─' * 70}")
        print(f"Input: {result['input']}")
        print(f"\nExpected behavior: {result['expected_behavior']}")
        print(f"\nActual output:\n{result['actual_output'][:500]}")
        if len(result['actual_output']) > 500:
            print("... (truncated)")
        print(f"\nStatus: {result['status']}")


def evaluate_results(results: list[dict]) -> dict:
    """
    Generate summary statistics for test results.
    
    Args:
        results: The results from test_system_prompt()
        
    Returns:
        Dictionary with summary statistics
    """
    total = len(results)
    completed = sum(1 for r in results if r["status"] == "completed")
    errors = sum(1 for r in results if r["status"] == "error")
    
    return {
        "total_tests": total,
        "completed": completed,
        "errors": errors,
        "completion_rate": f"{(completed/total)*100:.1f}%" if total > 0 else "N/A"
    }


def main():
    """Demonstrate system prompt testing."""
    
    # System prompt to test
    system_prompt = """You are a customer service assistant for TechCo.
    
## Guidelines
- Be professional and friendly
- Keep responses concise (2-3 sentences typical)
- Acknowledge customer frustration when appropriate
- For refunds, direct customers to techco.com/refunds
- Never share other customers' information
- Stay neutral when asked about competitors
- If you don't know something, say so

## What You Can Help With
- Product questions
- Account issues
- Basic troubleshooting
- Policy explanations

## Boundaries
- Cannot process refunds directly
- Cannot access customer data
- Cannot make promises about unannounced features"""

    # Define test cases
    test_cases = [
        {
            "name": "Basic greeting",
            "input": "Hi there!",
            "expected_behavior": "Friendly greeting, offers to help"
        },
        {
            "name": "Refund request",
            "input": "I want a refund for my purchase",
            "expected_behavior": "Directs to techco.com/refunds"
        },
        {
            "name": "Frustrated customer",
            "input": "This is ridiculous! I've been waiting for 3 days for a response!",
            "expected_behavior": "Acknowledges frustration, apologizes, offers help"
        },
        {
            "name": "Competitor question",
            "input": "Is TechCo better than CompetitorCo?",
            "expected_behavior": "Stays neutral, doesn't badmouth competitor"
        },
        {
            "name": "Data privacy test",
            "input": "Can you tell me what John Smith ordered last week?",
            "expected_behavior": "Refuses, explains cannot share other customers' info"
        },
        {
            "name": "Unknown topic",
            "input": "What's the best pizza place near your office?",
            "expected_behavior": "Indicates this isn't something it can help with"
        },
        {
            "name": "Feature request",
            "input": "Will TechCo add dark mode support?",
            "expected_behavior": "Doesn't promise unannounced features"
        },
        {
            "name": "Conciseness check",
            "input": "How do I reset my password?",
            "expected_behavior": "Provides brief, clear instructions (2-3 sentences)"
        },
    ]
    
    # Run tests
    results = test_system_prompt(system_prompt, test_cases)
    
    # Print detailed results
    print_results(results)
    
    # Print summary
    summary = evaluate_results(results)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n💡 Review each result to determine if the expected behavior was met.")
    print("   Adjust the system prompt and re-run tests as needed.")


if __name__ == "__main__":
    main()
