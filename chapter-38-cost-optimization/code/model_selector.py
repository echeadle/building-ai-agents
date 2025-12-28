"""
Model selection strategies for cost optimization.

Chapter 38: Cost Optimization
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class TaskComplexity(Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


@dataclass
class ModelConfig:
    model_id: str
    name: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    strengths: list[str]


MODELS = {
    "opus": ModelConfig(
        model_id="claude-opus-4-20250514",
        name="Claude Opus 4",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        strengths=["complex reasoning", "expert analysis", "safety-critical"]
    ),
    "sonnet": ModelConfig(
        model_id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        strengths=["balanced performance", "code generation", "analysis"]
    ),
    "haiku": ModelConfig(
        model_id="claude-3-5-haiku-20241022",
        name="Claude Haiku 3.5",
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.00,
        strengths=["speed", "simple tasks", "classification", "high volume"]
    ),
}


class ModelSelector:
    """Select optimal model based on task requirements."""
    
    COMPLEXITY_MODEL_MAP = {
        TaskComplexity.TRIVIAL: "haiku",
        TaskComplexity.SIMPLE: "haiku",
        TaskComplexity.MODERATE: "sonnet",
        TaskComplexity.COMPLEX: "sonnet",
        TaskComplexity.CRITICAL: "opus",
    }
    
    TASK_COMPLEXITY = {
        "classification": TaskComplexity.TRIVIAL,
        "yes_no": TaskComplexity.TRIVIAL,
        "extraction": TaskComplexity.TRIVIAL,
        "summarization": TaskComplexity.SIMPLE,
        "short_qa": TaskComplexity.SIMPLE,
        "code_review": TaskComplexity.MODERATE,
        "analysis": TaskComplexity.MODERATE,
        "code_generation": TaskComplexity.COMPLEX,
        "creative_writing": TaskComplexity.COMPLEX,
        "legal_analysis": TaskComplexity.CRITICAL,
        "medical_advice": TaskComplexity.CRITICAL,
    }
    
    def __init__(self, default_model: str = "sonnet"):
        self.default_model = default_model
    
    def select_model(
        self,
        task_type: Optional[str] = None,
        complexity: Optional[TaskComplexity] = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
    ) -> ModelConfig:
        if complexity is None:
            if task_type and task_type in self.TASK_COMPLEXITY:
                complexity = self.TASK_COMPLEXITY[task_type]
            else:
                complexity = TaskComplexity.MODERATE
        
        model_key = self.COMPLEXITY_MODEL_MAP[complexity]
        
        if prefer_speed:
            model_key = "haiku"
        elif prefer_quality:
            if complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
                model_key = "opus"
            else:
                model_key = "sonnet"
        
        return MODELS[model_key]


def calculate_model_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    config = MODELS.get(model_key, MODELS["sonnet"])
    return (
        (input_tokens / 1_000_000) * config.input_cost_per_1m +
        (output_tokens / 1_000_000) * config.output_cost_per_1m
    )


if __name__ == "__main__":
    print("Model Selection Demo")
    print("=" * 50)
    
    selector = ModelSelector()
    
    tasks = [
        ("classification", "Classify email as spam"),
        ("short_qa", "What is the capital of France?"),
        ("code_generation", "Write a sort function"),
        ("legal_analysis", "Analyze contract risks"),
    ]
    
    for task_type, description in tasks:
        model = selector.select_model(task_type=task_type)
        print(f"\n{task_type}: {model.name}")
        print(f"  Example: {description}")
    
    print("\n" + "-" * 50)
    print("Cost Comparison (1000 calls, 500 in / 200 out):")
    
    for key, config in MODELS.items():
        cost = 1000 * calculate_model_cost(key, 500, 200)
        print(f"  {config.name}: ${cost:.2f}")
