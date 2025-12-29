"""
Token counting and cost estimation.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()

# Pricing per million tokens (as of early 2025)
# IMPORTANT: Pricing changes over time!
# Always check https://www.anthropic.com/pricing for current rates before using in production
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5-20241022": {"input": 0.80, "output": 4.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an API call in dollars.
    
    Args:
        model: The model name used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in dollars
    """
    if model not in PRICING:
        # Default to Sonnet pricing for unknown models
        model = "claude-sonnet-4-20250514"
    
    input_cost = (input_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (output_tokens / 1_000_000) * PRICING[model]["output"]
    
    return input_cost + output_cost


def ask_claude_with_cost(prompt: str) -> tuple[str, dict]:
    """
    Send a prompt to Claude and return the response with cost information.
    
    Args:
        prompt: The question or instruction to send to Claude
        
    Returns:
        Tuple of (response_text, cost_info_dict)
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    cost = estimate_cost(
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens
    )
    
    cost_info = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
        "estimated_cost_usd": cost
    }
    
    return message.content[0].text, cost_info


# Test with cost tracking
if __name__ == "__main__":
    prompt = "Explain photosynthesis in two sentences."
    response, cost_info = ask_claude_with_cost(prompt)
    
    print("=== Response ===")
    print(response)
    print()
    print("=== Cost Information ===")
    print(f"Input tokens: {cost_info['input_tokens']}")
    print(f"Output tokens: {cost_info['output_tokens']}")
    print(f"Total tokens: {cost_info['total_tokens']}")
    print(f"Estimated cost: ${cost_info['estimated_cost_usd']:.6f}")
