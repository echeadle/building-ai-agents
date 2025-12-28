"""
Token usage tracking and cost estimation.

Appendix B: API Reference Quick Guide
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class TokenUsage:
    """
    Track token usage and costs.
    
    This class makes it easy to track and aggregate token usage
    across multiple API calls.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens
    
    def estimate_cost(
        self,
        input_cost_per_million: float = 3.00,
        output_cost_per_million: float = 15.00,
        cached_cost_per_million: float = 0.30
    ) -> float:
        """
        Estimate cost based on current token usage.
        
        Args:
            input_cost_per_million: Cost per million input tokens
            output_cost_per_million: Cost per million output tokens
            cached_cost_per_million: Cost per million cached tokens
            
        Returns:
            Estimated cost in dollars
        """
        input_cost = (self.input_tokens / 1_000_000) * input_cost_per_million
        output_cost = (self.output_tokens / 1_000_000) * output_cost_per_million
        cached_cost = (self.cached_tokens / 1_000_000) * cached_cost_per_million
        
        return input_cost + output_cost + cached_cost
    
    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects together."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens
        )
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"TokenUsage(input={self.input_tokens}, "
            f"output={self.output_tokens}, "
            f"total={self.total_tokens})"
        )


class TokenCounter:
    """
    Track token usage across multiple API calls.
    
    Usage:
        counter = TokenCounter()
        
        # Make API calls
        response = client.messages.create(...)
        counter.add_response(response)
        
        # Check totals
        print(f"Total usage: {counter.total}")
        print(f"Total cost: ${counter.total_cost():.4f}")
        print(counter.summary())
    """
    
    def __init__(
        self,
        input_cost_per_million: float = 3.00,
        output_cost_per_million: float = 15.00
    ):
        """
        Initialize the token counter.
        
        Args:
            input_cost_per_million: Cost per million input tokens
            output_cost_per_million: Cost per million output tokens
        """
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.total = TokenUsage()
        self.call_count = 0
        self.history: list[TokenUsage] = []
    
    def add_response(self, response) -> TokenUsage:
        """
        Add token usage from an API response.
        
        Args:
            response: API response object with usage attribute
            
        Returns:
            TokenUsage for this response
        """
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        
        self.total = self.total + usage
        self.call_count += 1
        self.history.append(usage)
        
        return usage
    
    def total_cost(self) -> float:
        """Calculate total cost so far."""
        return self.total.estimate_cost(
            self.input_cost_per_million,
            self.output_cost_per_million
        )
    
    def average_tokens_per_call(self) -> float:
        """Calculate average tokens per API call."""
        if self.call_count == 0:
            return 0.0
        return self.total.total_tokens / self.call_count
    
    def average_cost_per_call(self) -> float:
        """Calculate average cost per API call."""
        if self.call_count == 0:
            return 0.0
        return self.total_cost() / self.call_count
    
    def summary(self) -> str:
        """Get a summary of token usage."""
        avg_tokens = self.average_tokens_per_call()
        avg_cost = self.average_cost_per_call()
        total = self.total_cost()
        
        return (
            f"Token Usage Summary\n"
            f"{'='*50}\n"
            f"API Calls:         {self.call_count:,}\n"
            f"Input Tokens:      {self.total.input_tokens:,}\n"
            f"Output Tokens:     {self.total.output_tokens:,}\n"
            f"Total Tokens:      {self.total.total_tokens:,}\n"
            f"{'='*50}\n"
            f"Avg per Call:      {avg_tokens:.1f} tokens\n"
            f"Avg Cost per Call: ${avg_cost:.6f}\n"
            f"Total Cost:        ${total:.4f}"
        )
    
    def reset(self):
        """Reset all counters."""
        self.total = TokenUsage()
        self.call_count = 0
        self.history = []


def demonstrate_basic_tracking():
    """Demonstrate basic token tracking."""
    client = anthropic.Anthropic()
    counter = TokenCounter()
    
    print("=== Basic Token Tracking Demo ===\n")
    
    # Make several API calls
    queries = [
        "What is Python?",
        "Explain object-oriented programming in detail.",
        "Write a haiku about coding"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": query}]
        )
        
        # Track usage
        usage = counter.add_response(response)
        cost = usage.estimate_cost()
        
        print(f"  Tokens: {usage}")
        print(f"  Cost: ${cost:.6f}\n")
    
    # Print summary
    print("\n" + counter.summary())


def demonstrate_cost_comparison():
    """Compare costs across different models."""
    client = anthropic.Anthropic()
    
    print("\n=== Cost Comparison Demo ===\n")
    
    query = "Explain machine learning in 100 words"
    
    # Model configurations with their pricing
    models = [
        {
            "name": "Claude Sonnet 4",
            "model": "claude-sonnet-4-20250514",
            "input_cost": 3.00,
            "output_cost": 15.00
        },
        {
            "name": "Claude Haiku 4",
            "model": "claude-haiku-4-20250514",
            "input_cost": 0.80,
            "output_cost": 4.00
        }
    ]
    
    print(f"Query: {query}\n")
    
    for model_config in models:
        print(f"{model_config['name']}:")
        
        response = client.messages.create(
            model=model_config["model"],
            max_tokens=150,
            messages=[{"role": "user", "content": query}]
        )
        
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens
        )
        
        cost = usage.estimate_cost(
            input_cost_per_million=model_config["input_cost"],
            output_cost_per_million=model_config["output_cost"]
        )
        
        print(f"  Input:  {usage.input_tokens:,} tokens")
        print(f"  Output: {usage.output_tokens:,} tokens")
        print(f"  Cost:   ${cost:.6f}\n")


def demonstrate_budget_tracking():
    """Track usage against a budget."""
    client = anthropic.Anthropic()
    counter = TokenCounter()
    
    print("=== Budget Tracking Demo ===\n")
    
    # Set a budget
    budget = 0.10  # $0.10
    print(f"Budget: ${budget:.2f}\n")
    
    queries = [
        "Explain neural networks",
        "What is deep learning?",
        "Describe transformers architecture",
        "What are attention mechanisms?",
        "Explain backpropagation"
    ]
    
    for i, query in enumerate(queries, 1):
        current_cost = counter.total_cost()
        
        if current_cost >= budget:
            print(f"⚠️  Budget exceeded after {i-1} queries")
            print(f"Spent: ${current_cost:.4f}")
            break
        
        print(f"Query {i}: {query}")
        print(f"  Budget remaining: ${budget - current_cost:.4f}")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": query}]
        )
        
        usage = counter.add_response(response)
        print(f"  Used: {usage}")
        print(f"  Cost: ${usage.estimate_cost():.6f}\n")
    
    print(counter.summary())


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count without API call.
    
    This is approximate. For exact counts, use the API.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Rough heuristic: 1 token ≈ 4 characters
    return len(text) // 4


def demonstrate_estimation():
    """Demonstrate token estimation."""
    print("\n=== Token Estimation Demo ===\n")
    
    texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
        "This is a longer text that spans multiple sentences. "
        "It contains more words and should use more tokens. "
        "Token estimation helps predict costs before making API calls."
    ]
    
    for text in texts:
        estimated = estimate_tokens(text)
        print(f"Text: {text[:50]}...")
        print(f"  Estimated tokens: {estimated}")
        print(f"  Character length: {len(text)}\n")


if __name__ == "__main__":
    # Basic tracking
    demonstrate_basic_tracking()
    
    print("\n" + "="*60 + "\n")
    
    # Cost comparison
    demonstrate_cost_comparison()
    
    print("\n" + "="*60 + "\n")
    
    # Budget tracking
    demonstrate_budget_tracking()
    
    print("\n" + "="*60 + "\n")
    
    # Estimation
    demonstrate_estimation()
