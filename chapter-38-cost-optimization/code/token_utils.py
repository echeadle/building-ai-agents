"""
Token estimation and cost calculation utilities.

Chapter 38: Cost Optimization
"""

import re
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string."""
    if not text:
        return 0
    
    words = text.split()
    word_count = len(words)
    punctuation = len(re.findall(r'[^\w\s]', text))
    numbers = len(re.findall(r'\d+', text))
    
    estimate = word_count + (punctuation // 2) + numbers
    
    if word_count > 0:
        avg_word_len = sum(len(w) for w in words) / word_count
        if avg_word_len > 6:
            estimate = int(estimate * 1.2)
    
    return max(1, estimate)


@dataclass
class ModelPricing:
    input_cost: float
    output_cost: float


MODEL_PRICING = {
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00),
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00),
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str = "claude-sonnet-4-20250514") -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-sonnet-4-20250514"])
    return (input_tokens / 1_000_000) * pricing.input_cost + (output_tokens / 1_000_000) * pricing.output_cost


def format_cost(cost: float) -> str:
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.00:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


if __name__ == "__main__":
    print("Token Estimation Demo")
    print("=" * 40)
    texts = ["Hello, world!", "What is machine learning?"]
    for text in texts:
        print(f"{text}: {estimate_tokens(text)} tokens")
