# Chapter 25: Evaluator-Optimizer - Implementation

## Code Files

This directory contains complete, runnable code examples for the evaluator-optimizer pattern.

### Files Overview

| File | Description | Complexity |
|------|-------------|------------|
| `evaluator.py` | Core pattern implementation with Generator, Evaluator, and Loop Controller | Advanced |
| `writing_assistant.py` | Complete writing assistant with preset criteria | Advanced |
| `simple_example.py` | Minimal standalone example (slogan optimizer) | Beginner |
| `exercise.py` | Exercise solution: Code Review Assistant | Intermediate |

## Quick Start

1. **Set up your environment:**
   ```bash
   # Create .env file with your API key
   echo "ANTHROPIC_API_KEY=your-key-here" > .env
   ```

2. **Install dependencies:**
   ```bash
   pip install anthropic python-dotenv
   ```

3. **Run the simple example first:**
   ```bash
   uv run python simple_example.py
   ```

## File Details

### `evaluator.py`

The core implementation containing:
- `Generator` - Creates and revises content
- `Evaluator` - Assesses quality with structured feedback
- `EvaluatorOptimizer` - Manages the refinement loop
- Configuration dataclasses for customization

**Usage:**
```python
from evaluator import Generator, Evaluator, EvaluatorOptimizer

client = anthropic.Anthropic()
generator = Generator(client)
evaluator = Evaluator(client, ["Clarity", "Conciseness"])
optimizer = EvaluatorOptimizer(generator, evaluator)

result = optimizer.optimize("Write a welcome email")
print(result.final_content)
```

### `writing_assistant.py`

A high-level writing assistant with:
- Preset criteria for different content types (general, technical, marketing, academic, email)
- `write()` method for new content
- `improve()` method for existing content
- `evaluate_only()` for assessment without modification

**Run demos:**
```bash
uv run python writing_assistant.py product    # Product description
uv run python writing_assistant.py technical  # Technical documentation
uv run python writing_assistant.py email      # Professional email
uv run python writing_assistant.py improve    # Improve existing content
```

### `simple_example.py`

A minimal, self-contained example that's easy to understand:
- Creates marketing slogans
- Evaluates against 4 criteria
- Iteratively improves
- No external dependencies on other files

**Best for:** Understanding the pattern before diving into the full implementation.

```bash
uv run python simple_example.py
```

### `exercise.py`

The chapter exercise solution - a code review assistant that:
- Evaluates Python code against 7 quality criteria
- Suggests improvements
- Applies improvements and re-evaluates
- Implements convergence detection
- Handles already-good code gracefully

**Run demos:**
```bash
uv run python exercise.py poor    # Improve poor code
uv run python exercise.py good    # Review already-good code
uv run python exercise.py review  # Assessment only
```

## Key Concepts Demonstrated

### 1. The Generator-Evaluator Loop
```
┌─────────────┐    content    ┌─────────────┐
│  Generator  │──────────────▶│  Evaluator  │
│  (create/   │               │  (assess)   │
│   revise)   │◀──────────────│             │
└─────────────┘   feedback    └─────────────┘
       ▲                            │
       │         ┌──────────┐       │
       └─────────│  Loop    │◀──────┘
                 │ Controller│
                 └──────────┘
```

### 2. Convergence Detection
The loop stops when:
- Content passes quality threshold
- Improvement between iterations < threshold
- Maximum iterations reached

### 3. Structured Feedback
Evaluators return JSON with:
- `score`: 0.0 to 1.0
- `passed`: boolean
- `feedback`/`issues`: specific improvements needed
- `strengths`: what's working well

## Customization Tips

### Custom Evaluation Criteria
```python
my_criteria = [
    "Specific criterion 1: detailed description",
    "Specific criterion 2: detailed description",
]
evaluator = Evaluator(client, my_criteria)
```

### Adjusting Quality Threshold
```python
config = EvaluatorConfig(quality_threshold=0.9)  # Higher bar
evaluator = Evaluator(client, criteria, config)
```

### Changing Max Iterations
```python
config = LoopConfig(max_iterations=10)  # More refinement cycles
optimizer = EvaluatorOptimizer(generator, evaluator, config)
```

## Common Issues

### "ANTHROPIC_API_KEY not found"
Ensure your `.env` file exists and contains:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### JSON Parsing Errors
The evaluator sometimes returns malformed JSON. The code handles this gracefully, but if it happens frequently, try lowering the evaluator temperature:
```python
config = EvaluatorConfig(temperature=0.2)
```

### Low Scores Despite Good Content
Evaluation is somewhat subjective. Try:
1. Making criteria more specific
2. Lowering the quality threshold
3. Adding more context to the evaluation

## Next Steps

After understanding this chapter:
1. Try creating your own evaluator for a different use case
2. Experiment with different criteria formulations
3. Move on to Chapter 26: From Workflows to Agents
