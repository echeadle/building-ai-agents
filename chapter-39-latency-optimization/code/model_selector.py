"""
Model selection for optimal latency.

Chapter 39: Latency Optimization

Different models have different latency characteristics. Choosing
the right model can significantly impact response time.
"""

import os
import time
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


class TaskComplexity(Enum):
    """Classification of task complexity."""
    SIMPLE = "simple"       # Yes/no, classification, extraction
    MODERATE = "moderate"   # Summarization, simple analysis
    COMPLEX = "complex"     # Multi-step reasoning, creative work


@dataclass
class ModelProfile:
    """Performance profile for a model."""
    name: str
    avg_latency_ms: float      # Average time to first token
    tokens_per_second: float   # Generation speed
    quality_score: float       # Relative quality (0-1)
    cost_per_1k_tokens: float  # Input cost
    best_for: list[str]        # Task types this model excels at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "avg_latency_ms": self.avg_latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "quality_score": self.quality_score,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "best_for": self.best_for
        }


class LatencyAwareModelSelector:
    """
    Selects models based on latency requirements and task complexity.
    
    Balances:
    - Response time requirements
    - Task complexity needs
    - Quality requirements
    - Cost constraints
    
    Usage:
        selector = LatencyAwareModelSelector()
        
        model = selector.select(
            task_complexity=TaskComplexity.SIMPLE,
            max_latency_ms=1000,
            quality_threshold=0.7
        )
    """
    
    # Model profiles (approximate values - actual performance varies)
    MODEL_PROFILES = {
        "claude-3-5-haiku-20241022": ModelProfile(
            name="claude-3-5-haiku-20241022",
            avg_latency_ms=300,
            tokens_per_second=150,
            quality_score=0.75,
            cost_per_1k_tokens=0.0008,
            best_for=["classification", "extraction", "simple_qa", "routing"]
        ),
        "claude-sonnet-4-20250514": ModelProfile(
            name="claude-sonnet-4-20250514",
            avg_latency_ms=500,
            tokens_per_second=100,
            quality_score=0.90,
            cost_per_1k_tokens=0.003,
            best_for=["analysis", "coding", "summarization", "general"]
        ),
        "claude-opus-4-20250514": ModelProfile(
            name="claude-opus-4-20250514",
            avg_latency_ms=800,
            tokens_per_second=60,
            quality_score=0.98,
            cost_per_1k_tokens=0.015,
            best_for=["complex_reasoning", "creative", "research", "difficult_coding"]
        ),
    }
    
    # Task complexity to minimum quality mapping
    COMPLEXITY_QUALITY_MAP = {
        TaskComplexity.SIMPLE: 0.7,
        TaskComplexity.MODERATE: 0.85,
        TaskComplexity.COMPLEX: 0.95,
    }
    
    def __init__(self):
        """Initialize the model selector."""
        self.profiles = {k: ModelProfile(**v.__dict__) for k, v in self.MODEL_PROFILES.items()}
        self._latency_history: dict[str, list[float]] = {}
    
    def select(
        self,
        task_complexity: TaskComplexity = TaskComplexity.MODERATE,
        max_latency_ms: Optional[float] = None,
        quality_threshold: Optional[float] = None,
        max_cost_per_1k: Optional[float] = None,
        task_type: Optional[str] = None
    ) -> str:
        """
        Select the best model given constraints.
        
        Args:
            task_complexity: How complex is the task
            max_latency_ms: Maximum acceptable latency
            quality_threshold: Minimum quality score required
            max_cost_per_1k: Maximum cost per 1000 tokens
            task_type: Specific task type for optimization
        
        Returns:
            Model name to use
        """
        # Set defaults based on complexity
        if quality_threshold is None:
            quality_threshold = self.COMPLEXITY_QUALITY_MAP[task_complexity]
        
        candidates = []
        
        for name, profile in self.profiles.items():
            # Filter by latency
            if max_latency_ms and profile.avg_latency_ms > max_latency_ms:
                continue
            
            # Filter by quality
            if profile.quality_score < quality_threshold:
                continue
            
            # Filter by cost
            if max_cost_per_1k and profile.cost_per_1k_tokens > max_cost_per_1k:
                continue
            
            # Calculate score
            score = self._calculate_score(
                profile,
                task_type,
                max_latency_ms
            )
            
            candidates.append((name, score, profile))
        
        if not candidates:
            # No model meets all constraints, return best available
            return "claude-sonnet-4-20250514"
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[0][0]
    
    def _calculate_score(
        self,
        profile: ModelProfile,
        task_type: Optional[str],
        max_latency_ms: Optional[float]
    ) -> float:
        """Calculate a selection score for a model."""
        score = 0.0
        
        # Quality contributes most
        score += profile.quality_score * 50
        
        # Latency bonus (lower is better)
        if max_latency_ms:
            latency_ratio = profile.avg_latency_ms / max_latency_ms
            score += (1 - latency_ratio) * 30
        else:
            score += (1 - profile.avg_latency_ms / 1000) * 30
        
        # Task type match bonus
        if task_type and task_type in profile.best_for:
            score += 20
        
        # Speed bonus
        score += min(profile.tokens_per_second / 150, 1) * 10
        
        return score
    
    def record_latency(self, model: str, latency_ms: float) -> None:
        """
        Record actual latency for a model to improve selection.
        
        Args:
            model: Model name
            latency_ms: Observed latency
        """
        if model not in self._latency_history:
            self._latency_history[model] = []
        
        self._latency_history[model].append(latency_ms)
        
        # Keep only recent measurements
        if len(self._latency_history[model]) > 100:
            self._latency_history[model] = self._latency_history[model][-100:]
        
        # Update profile with actual measurements
        if model in self.profiles:
            avg = sum(self._latency_history[model]) / len(self._latency_history[model])
            self.profiles[model].avg_latency_ms = avg
    
    def get_recommendation(
        self,
        task_description: str,
        time_budget_ms: float
    ) -> dict[str, Any]:
        """
        Get a detailed model recommendation with explanation.
        
        Args:
            task_description: Description of the task
            time_budget_ms: Time budget in milliseconds
        
        Returns:
            Recommendation with reasoning
        """
        # Classify task complexity (simplified)
        complexity = TaskComplexity.MODERATE
        simple_keywords = ["simple", "yes/no", "extract", "classify", "quick"]
        complex_keywords = ["complex", "analyze", "research", "creative", "detailed"]
        
        task_lower = task_description.lower()
        if any(word in task_lower for word in simple_keywords):
            complexity = TaskComplexity.SIMPLE
        elif any(word in task_lower for word in complex_keywords):
            complexity = TaskComplexity.COMPLEX
        
        # Determine task type
        task_type = None
        for profile in self.profiles.values():
            for t in profile.best_for:
                if t.replace("_", " ") in task_lower or t in task_lower:
                    task_type = t
                    break
            if task_type:
                break
        
        # Get recommendation
        model = self.select(
            task_complexity=complexity,
            max_latency_ms=time_budget_ms * 0.6,  # Allow 60% for LLM
            task_type=task_type
        )
        
        profile = self.profiles[model]
        
        return {
            "recommended_model": model,
            "task_complexity": complexity.value,
            "detected_task_type": task_type,
            "expected_latency_ms": profile.avg_latency_ms,
            "quality_score": profile.quality_score,
            "fits_budget": profile.avg_latency_ms <= time_budget_ms * 0.6,
            "reasoning": self._generate_reasoning(profile, complexity, time_budget_ms)
        }
    
    def _generate_reasoning(
        self,
        profile: ModelProfile,
        complexity: TaskComplexity,
        budget_ms: float
    ) -> str:
        """Generate explanation for the recommendation."""
        reasons = []
        
        if complexity == TaskComplexity.SIMPLE:
            reasons.append("Task is simple, prioritizing speed")
        elif complexity == TaskComplexity.COMPLEX:
            reasons.append("Task is complex, prioritizing quality")
        else:
            reasons.append("Task is moderate complexity, balancing speed and quality")
        
        if profile.avg_latency_ms <= budget_ms * 0.5:
            reasons.append("Model latency well within budget")
        elif profile.avg_latency_ms <= budget_ms * 0.8:
            reasons.append("Model latency fits budget with some margin")
        else:
            reasons.append("Model latency is tight for budget")
        
        return "; ".join(reasons)
    
    def get_all_profiles(self) -> dict[str, dict[str, Any]]:
        """Get all model profiles."""
        return {name: profile.to_dict() for name, profile in self.profiles.items()}


class ModelSpeedBenchmark:
    """
    Benchmarks actual model speeds for accurate selection.
    
    Run this periodically to update model profiles with
    real performance data.
    """
    
    TEST_PROMPTS = {
        "simple": "Is Python a programming language? Answer yes or no.",
        "moderate": "Summarize the benefits of cloud computing in 2 sentences.",
        "complex": "Explain the tradeoffs between consistency and availability in distributed systems."
    }
    
    def __init__(self):
        """Initialize the benchmark."""
        self.client = anthropic.Anthropic()
    
    def benchmark_model(
        self,
        model: str,
        iterations: int = 3
    ) -> dict[str, Any]:
        """
        Benchmark a model's latency.
        
        Args:
            model: Model name to benchmark
            iterations: Number of test iterations
        
        Returns:
            Benchmark results
        """
        results = {
            "model": model,
            "tests": {}
        }
        
        for test_name, prompt in self.TEST_PROMPTS.items():
            latencies = []
            token_counts = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                
                response = self.client.messages.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                latencies.append(latency_ms)
                token_counts.append(response.usage.output_tokens)
            
            avg_latency = sum(latencies) / len(latencies)
            avg_tokens = sum(token_counts) / len(token_counts)
            
            results["tests"][test_name] = {
                "avg_latency_ms": round(avg_latency, 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
                "avg_tokens": round(avg_tokens, 1),
                "tokens_per_second": round(avg_tokens / (avg_latency / 1000), 1) if avg_latency > 0 else 0
            }
        
        # Calculate overall averages
        all_latencies = [r["avg_latency_ms"] for r in results["tests"].values()]
        results["overall"] = {
            "avg_latency_ms": round(sum(all_latencies) / len(all_latencies), 2)
        }
        
        return results
    
    def benchmark_all_models(self, iterations: int = 2) -> dict[str, Any]:
        """
        Benchmark all available models.
        
        Args:
            iterations: Number of iterations per test
        
        Returns:
            Results for all models
        """
        models = ["claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"]
        results = {}
        
        for model in models:
            print(f"Benchmarking {model}...")
            try:
                results[model] = self.benchmark_model(model, iterations)
            except Exception as e:
                results[model] = {"error": str(e)}
        
        return results


# Example usage
if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("LATENCY-AWARE MODEL SELECTION DEMO")
    print("=" * 60)
    
    selector = LatencyAwareModelSelector()
    
    # Test scenarios
    scenarios = [
        {
            "description": "Simple classification task",
            "task_complexity": TaskComplexity.SIMPLE,
            "max_latency_ms": 500,
        },
        {
            "description": "Code analysis task",
            "task_complexity": TaskComplexity.MODERATE,
            "max_latency_ms": 2000,
            "task_type": "coding"
        },
        {
            "description": "Complex research task",
            "task_complexity": TaskComplexity.COMPLEX,
            "max_latency_ms": 5000,
        },
        {
            "description": "Quick extraction with quality requirement",
            "task_complexity": TaskComplexity.SIMPLE,
            "max_latency_ms": 300,
            "quality_threshold": 0.9,
        },
    ]
    
    print("\nModel Selection Results:")
    print("-" * 60)
    
    for scenario in scenarios:
        model = selector.select(
            task_complexity=scenario["task_complexity"],
            max_latency_ms=scenario.get("max_latency_ms"),
            quality_threshold=scenario.get("quality_threshold"),
            task_type=scenario.get("task_type")
        )
        
        profile = selector.profiles[model]
        
        print(f"\n{scenario['description']}:")
        print(f"  Selected: {model}")
        print(f"  Expected latency: {profile.avg_latency_ms}ms")
        print(f"  Quality score: {profile.quality_score}")
    
    # Get detailed recommendation
    print("\n" + "-" * 60)
    print("Detailed Recommendation:")
    print("-" * 60)
    
    recommendation = selector.get_recommendation(
        task_description="Analyze this code for bugs and suggest improvements",
        time_budget_ms=3000
    )
    
    print(json.dumps(recommendation, indent=2))
    
    # Show all profiles
    print("\n" + "-" * 60)
    print("All Model Profiles:")
    print("-" * 60)
    
    for name, profile in selector.get_all_profiles().items():
        print(f"\n{name}:")
        print(f"  Latency: {profile['avg_latency_ms']}ms")
        print(f"  Speed: {profile['tokens_per_second']} tok/s")
        print(f"  Quality: {profile['quality_score']}")
        print(f"  Best for: {', '.join(profile['best_for'])}")
