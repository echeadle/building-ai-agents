"""
Exercise Solution: Cost-Aware Research Agent

Chapter 38: Cost Optimization

A research agent that:
1. Uses model tiering (Haiku for classification, Sonnet for research)
2. Caches responses to avoid redundant calls
3. Tracks costs against a daily budget
4. Reports detailed cost breakdowns
"""

import os
import hashlib
import time
from typing import Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

import anthropic


@dataclass
class CostTracker:
    daily_budget: float = 1.00
    pricing: dict = field(default_factory=lambda: {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    })
    total_cost: float = 0.0
    total_calls: int = 0
    
    def record_call(self, model: str, input_tokens: int, output_tokens: int) -> float:
        prices = self.pricing.get(model, self.pricing["claude-sonnet-4-20250514"])
        cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
        self.total_cost += cost
        self.total_calls += 1
        return cost
    
    def budget_remaining(self) -> float:
        return max(0, self.daily_budget - self.total_cost)
    
    def budget_exhausted(self) -> bool:
        return self.total_cost >= self.daily_budget


class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[str, float]] = {}
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, prompt: str) -> str:
        return hashlib.sha256(prompt.lower().strip().encode()).hexdigest()[:16]
    
    def get(self, prompt: str) -> Optional[str]:
        key = self._make_key(prompt)
        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                return response
        self.misses += 1
        return None
    
    def set(self, prompt: str, response: str) -> None:
        key = self._make_key(prompt)
        self._cache[key] = (response, time.time())


class CostAwareResearchAgent:
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-sonnet-4-20250514"
    
    def __init__(self, daily_budget: float = 1.00):
        self.client = anthropic.Anthropic()
        self.cost_tracker = CostTracker(daily_budget=daily_budget)
        self.cache = ResponseCache()
    
    def _classify_query(self, query: str) -> str:
        prompt = f"""Classify this query: simple, moderate, or complex.
Query: {query}
Respond with only one word."""
        
        response = self.client.messages.create(
            model=self.HAIKU,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        self.cost_tracker.record_call(self.HAIKU, response.usage.input_tokens, response.usage.output_tokens)
        
        result = response.content[0].text.strip().lower()
        if result in ["simple", "moderate", "complex"]:
            return result
        return "moderate"
    
    def research(self, query: str) -> dict[str, Any]:
        result = {
            "query": query, "response": None, "cached": False,
            "complexity": None, "model_used": None,
            "total_cost": 0.0, "budget_remaining": self.cost_tracker.budget_remaining(),
        }
        
        if self.cost_tracker.budget_exhausted():
            result["error"] = "⚠️ Budget exhausted!"
            return result
        
        # Check cache
        cached = self.cache.get(query)
        if cached:
            result["response"] = cached
            result["cached"] = True
            result["model_used"] = "cached"
            return result
        
        # Classify query
        cost_before = self.cost_tracker.total_cost
        complexity = self._classify_query(query)
        classification_cost = self.cost_tracker.total_cost - cost_before
        result["complexity"] = complexity
        
        # Select model and tokens
        if complexity == "simple":
            model, max_tokens = self.HAIKU, 150
        else:
            model, max_tokens = self.SONNET, 500 if complexity == "moderate" else 1000
        
        result["model_used"] = "Haiku" if "haiku" in model else "Sonnet"
        
        # Generate response
        cost_before = self.cost_tracker.total_cost
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": query}]
        )
        self.cost_tracker.record_call(model, response.usage.input_tokens, response.usage.output_tokens)
        
        result["response"] = response.content[0].text
        result["total_cost"] = classification_cost + (self.cost_tracker.total_cost - cost_before)
        result["budget_remaining"] = self.cost_tracker.budget_remaining()
        
        # Cache response
        self.cache.set(query, result["response"])
        
        return result
    
    def print_result(self, result: dict) -> None:
        print("\n" + "=" * 50)
        print(f"Query: {result['query']}")
        print("=" * 50)
        
        if result.get("error"):
            print(result["error"])
            return
        
        print(f"Complexity: {result['complexity']}")
        print(f"Model: {result['model_used']}")
        print(f"Cached: {'Yes' if result['cached'] else 'No'}")
        print(f"\nResponse:\n{result['response'][:200]}...")
        print(f"\nCost: ${result['total_cost']:.4f}")
        print(f"Budget remaining: ${result['budget_remaining']:.4f}")


def main():
    print("Cost-Aware Research Agent Demo")
    print("=" * 50)
    
    agent = CostAwareResearchAgent(daily_budget=1.00)
    
    queries = [
        "What is the capital of France?",
        "How does machine learning work?",
        "What is the capital of France?",  # Should hit cache
    ]
    
    for query in queries:
        result = agent.research(query)
        agent.print_result(result)
        
        if agent.cost_tracker.budget_exhausted():
            print("\n🛑 Budget exhausted!")
            break
        
        time.sleep(0.5)
    
    print(f"\n\nFinal Stats:")
    print(f"  Total calls: {agent.cost_tracker.total_calls}")
    print(f"  Total cost: ${agent.cost_tracker.total_cost:.4f}")
    print(f"  Cache hits: {agent.cache.hits}")
    print(f"  Cache misses: {agent.cache.misses}")


if __name__ == "__main__":
    main()
