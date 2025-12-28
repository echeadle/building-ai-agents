"""
Simple Prompts vs. Workflows: When to use each.

Chapter 15: Introduction to Agentic Workflows

This module demonstrates scenarios where simple prompts are sufficient
and contrasts them with scenarios where workflows provide real value.
The goal is to help you recognize when complexity is warranted.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


@dataclass
class ComparisonResult:
    """Results from comparing simple vs. workflow approaches."""
    
    approach: str
    output: str
    duration_ms: float
    tokens_used: int
    estimated_cost: float
    
    def summary(self) -> str:
        return (
            f"{self.approach}:\n"
            f"  Duration: {self.duration_ms:.0f}ms\n"
            f"  Tokens: {self.tokens_used}\n"
            f"  Est. Cost: ${self.estimated_cost:.4f}\n"
            f"  Output preview: {self.output[:100]}..."
        )


# =============================================================================
# EXAMPLE 1: Technical Explanation (Simple Prompt Is Enough)
# =============================================================================

def example_1_simple_prompt(topic: str) -> ComparisonResult:
    """
    A single well-crafted prompt for technical explanation.
    
    This demonstrates a case where a simple prompt is sufficient.
    """
    start = time.time()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are an expert technical writer. When explaining concepts:
- Start with a clear, one-sentence definition
- Explain why it matters
- Provide a concrete example
- Summarize key points

Keep explanations concise but thorough.""",
        messages=[
            {"role": "user", "content": f"Explain {topic} to intermediate developers."}
        ]
    )
    
    duration = (time.time() - start) * 1000
    output = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    
    # Rough cost estimate (claude-sonnet-4-20250514 pricing approximation)
    cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015) / 1000
    
    return ComparisonResult(
        approach="Simple Prompt",
        output=output,
        duration_ms=duration,
        tokens_used=tokens,
        estimated_cost=cost
    )


def example_1_unnecessary_workflow(topic: str) -> ComparisonResult:
    """
    Over-engineered workflow for the same task.
    
    This demonstrates adding complexity that doesn't improve results.
    """
    start = time.time()
    total_tokens = 0
    
    # Step 1: Generate outline (unnecessary)
    response1 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[
            {"role": "user", "content": f"Create a brief outline for explaining {topic}."}
        ]
    )
    outline = response1.content[0].text
    total_tokens += response1.usage.input_tokens + response1.usage.output_tokens
    
    # Step 2: Write based on outline (could have done this directly)
    response2 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Based on this outline:\n{outline}\n\nWrite a clear explanation of {topic} for intermediate developers."}
        ]
    )
    content = response2.content[0].text
    total_tokens += response2.usage.input_tokens + response2.usage.output_tokens
    
    # Step 3: Review (minimal value added)
    response3 = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[
            {"role": "user", "content": f"Review this explanation and fix any issues:\n\n{content}"}
        ]
    )
    final = response3.content[0].text
    total_tokens += response3.usage.input_tokens + response3.usage.output_tokens
    
    duration = (time.time() - start) * 1000
    
    # Higher cost due to multiple calls
    cost = total_tokens * 0.009 / 1000  # Rough average
    
    return ComparisonResult(
        approach="Over-engineered Workflow",
        output=final,
        duration_ms=duration,
        tokens_used=total_tokens,
        estimated_cost=cost
    )


# =============================================================================
# EXAMPLE 2: Customer Support Routing (Workflow Adds Value)
# =============================================================================

def example_2_simple_prompt_insufficient(message: str) -> ComparisonResult:
    """
    Single prompt trying to handle all customer support cases.
    
    This approach struggles with diverse input types.
    """
    start = time.time()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a customer support agent. Handle any customer message appropriately.
You should address billing issues, technical problems, and general inquiries.""",
        messages=[
            {"role": "user", "content": message}
        ]
    )
    
    duration = (time.time() - start) * 1000
    output = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015) / 1000
    
    return ComparisonResult(
        approach="Simple Prompt (One-size-fits-all)",
        output=output,
        duration_ms=duration,
        tokens_used=tokens,
        estimated_cost=cost
    )


def example_2_routing_workflow(message: str) -> ComparisonResult:
    """
    Routing workflow that classifies and handles appropriately.
    
    This demonstrates where workflow patterns add real value.
    """
    start = time.time()
    total_tokens = 0
    
    # Step 1: Classify the message
    classify_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[
            {
                "role": "user", 
                "content": f"""Classify this customer message into one category:
- billing (payment, charges, invoices, refunds)
- technical (bugs, errors, how-to, features)
- general (other questions, feedback)

Message: "{message}"

Respond with just the category name."""
            }
        ]
    )
    category = classify_response.content[0].text.strip().lower()
    total_tokens += classify_response.usage.input_tokens + classify_response.usage.output_tokens
    
    # Step 2: Route to specialized handler
    handlers = {
        "billing": """You are a billing specialist. You handle:
- Payment questions with specific account lookup guidance
- Refund requests with policy details
- Invoice clarifications with line-item explanations
Be precise about amounts and timelines.""",
        
        "technical": """You are a technical support engineer. You:
- Ask clarifying questions about errors
- Provide step-by-step troubleshooting
- Reference documentation when helpful
Be thorough but not overwhelming.""",
        
        "general": """You are a customer service representative. You:
- Answer general product questions
- Provide helpful information
- Escalate if needed
Be friendly and helpful."""
    }
    
    handler_system = handlers.get(category, handlers["general"])
    
    handle_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=handler_system,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    output = handle_response.content[0].text
    total_tokens += handle_response.usage.input_tokens + handle_response.usage.output_tokens
    
    duration = (time.time() - start) * 1000
    cost = total_tokens * 0.009 / 1000
    
    return ComparisonResult(
        approach=f"Routing Workflow (routed to: {category})",
        output=output,
        duration_ms=duration,
        tokens_used=total_tokens,
        estimated_cost=cost
    )


# =============================================================================
# EXAMPLE 3: Code Review (Parallelization Adds Value)
# =============================================================================

SAMPLE_CODE = '''
def process_user_data(user_input):
    query = f"SELECT * FROM users WHERE id = {user_input}"
    result = db.execute(query)
    password = result['password']
    return {"user": result, "token": password[::-1]}
'''

def example_3_simple_code_review(code: str) -> ComparisonResult:
    """
    Single prompt for comprehensive code review.
    
    This works but may miss issues that specialized reviews would catch.
    """
    start = time.time()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"Review this code for any issues:\n\n```python\n{code}\n```"}
        ]
    )
    
    duration = (time.time() - start) * 1000
    output = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    cost = (response.usage.input_tokens * 0.003 + response.usage.output_tokens * 0.015) / 1000
    
    return ComparisonResult(
        approach="Simple Review",
        output=output,
        duration_ms=duration,
        tokens_used=tokens,
        estimated_cost=cost
    )


def example_3_parallel_code_review(code: str) -> ComparisonResult:
    """
    Parallel specialized reviews that run concurrently.
    
    Note: This example runs sequentially for simplicity.
    Chapter 21 will show true async parallelization.
    """
    start = time.time()
    total_tokens = 0
    reviews = []
    
    specialists = [
        ("Security Review", "You are a security expert. Review ONLY for security vulnerabilities like injection, authentication issues, data exposure. Be specific."),
        ("Performance Review", "You are a performance expert. Review ONLY for performance issues like inefficient queries, memory usage, algorithmic complexity. Be specific."),
        ("Code Quality Review", "You are a code quality expert. Review ONLY for readability, maintainability, naming, and best practices. Be specific.")
    ]
    
    for name, system in specialists:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=system,
            messages=[
                {"role": "user", "content": f"Review this code:\n\n```python\n{code}\n```"}
            ]
        )
        reviews.append(f"### {name}\n{response.content[0].text}")
        total_tokens += response.usage.input_tokens + response.usage.output_tokens
    
    output = "\n\n".join(reviews)
    duration = (time.time() - start) * 1000
    cost = total_tokens * 0.009 / 1000
    
    return ComparisonResult(
        approach="Parallel Specialized Reviews",
        output=output,
        duration_ms=duration,
        tokens_used=total_tokens,
        estimated_cost=cost
    )


def run_comparison(
    name: str,
    simple_fn: callable,
    workflow_fn: callable,
    input_data: Any
) -> None:
    """Run and display a comparison between approaches."""
    
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {name}")
    print(f"{'=' * 60}")
    print(f"\nInput: {str(input_data)[:80]}...")
    
    print("\n--- Running Simple Approach ---")
    simple_result = simple_fn(input_data)
    print(simple_result.summary())
    
    print("\n--- Running Workflow Approach ---")
    workflow_result = workflow_fn(input_data)
    print(workflow_result.summary())
    
    # Analysis
    print("\n--- Analysis ---")
    time_diff = workflow_result.duration_ms - simple_result.duration_ms
    token_diff = workflow_result.tokens_used - simple_result.tokens_used
    cost_diff = workflow_result.estimated_cost - simple_result.estimated_cost
    
    print(f"Time difference: {time_diff:+.0f}ms")
    print(f"Token difference: {token_diff:+d}")
    print(f"Cost difference: ${cost_diff:+.4f}")


def main():
    """Run all comparison examples."""
    
    print("\n" + "=" * 60)
    print("SIMPLE PROMPTS vs. WORKFLOWS")
    print("When is complexity warranted?")
    print("=" * 60)
    
    # Example 1: Simple prompt is enough
    print("\n" + "#" * 60)
    print("# EXAMPLE 1: Technical Explanation")
    print("# Verdict: Simple prompt is sufficient")
    print("#" * 60)
    run_comparison(
        "Technical Explanation",
        example_1_simple_prompt,
        example_1_unnecessary_workflow,
        "dependency injection"
    )
    
    # Example 2: Workflow adds value
    print("\n" + "#" * 60)
    print("# EXAMPLE 2: Customer Support")
    print("# Verdict: Routing workflow adds value")
    print("#" * 60)
    run_comparison(
        "Customer Support",
        example_2_simple_prompt_insufficient,
        example_2_routing_workflow,
        "I was charged $50 twice for my subscription last month and I need a refund"
    )
    
    # Example 3: Parallel review adds value
    print("\n" + "#" * 60)
    print("# EXAMPLE 3: Code Review")
    print("# Verdict: Parallel specialized reviews are more thorough")
    print("#" * 60)
    run_comparison(
        "Code Review",
        example_3_simple_code_review,
        example_3_parallel_code_review,
        SAMPLE_CODE
    )
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. Simple prompts are often sufficient for focused, single-objective tasks.

2. Workflows add value when:
   - Different inputs need different handling (→ Routing)
   - Independent subtasks exist (→ Parallelization)
   - Quality improves with iteration (→ Evaluator-Optimizer)
   - Tasks have clear sequential dependencies (→ Chaining)
   - Subtasks can't be predicted (→ Orchestrator-Workers)

3. Added complexity always means:
   - More latency
   - Higher cost
   - More code to maintain

4. Start simple. Add complexity only when you can measure the benefit.
""")


if __name__ == "__main__":
    main()
