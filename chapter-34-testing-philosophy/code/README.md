# Chapter 34: Testing AI Agents - Philosophy

Code examples demonstrating the philosophy of testing AI agents.

## Files

### behavior_tests.py
Examples of behavior-based testing approaches for agents. Shows how to test tool selection, output format, and safety behaviors rather than exact string outputs.

### test_dataset.py
A complete test dataset structure for agent evaluation. Demonstrates how to organize test cases by category and define expected behaviors.

### evaluation_metrics.py
Implementation of key evaluation metrics for agents including task success rate, tool accuracy, safety metrics, and efficiency measurements.

### exercise_solution.py
Solution to the chapter's practical exercise: designing a test dataset for a Q&A agent with a calculator tool.

## Running the Examples

These examples are conceptual and demonstrate testing philosophy. They require the `Agent` class from previous chapters to be fully functional.

```bash
# Install dependencies
uv add pytest anthropic python-dotenv

# Run the examples
python behavior_tests.py
python test_dataset.py
python evaluation_metrics.py
```

## Key Concepts

1. **Behavior over exact outputs**: Test what the agent does, not exactly what it says
2. **Comprehensive datasets**: Cover happy paths, edge cases, errors, and adversarial inputs
3. **Meaningful metrics**: Success rate, tool accuracy, safety scores, efficiency
4. **Multiple runs**: Run tests multiple times to account for non-determinism
