# Chapter 32: Guardrails and Safety - Code Examples

This directory contains complete, runnable code examples for Chapter 32.

## Files

### Core Guardrails Components

| File | Description |
|------|-------------|
| `input_validator.py` | Input validation and sanitization |
| `output_filter.py` | Output filtering and PII redaction |
| `action_constraints.py` | Action allowlists, blocklists, and constraints |
| `resource_manager.py` | Rate limiting and resource management |
| `sandbox.py` | Sandboxing for dangerous operations |

### Complete Implementation

| File | Description |
|------|-------------|
| `guardrails.py` | Complete unified guardrails module |
| `agent_with_guardrails.py` | Example agent using all guardrails |

### Exercise

| File | Description |
|------|-------------|
| `exercise_solution.py` | Solution to the chapter exercise |

## Setup

1. Ensure you have Python 3.10+ installed
2. Create a virtual environment (recommended)
3. Install dependencies:

```bash
pip install python-dotenv anthropic
```

4. Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=your-api-key-here
```

## Running Examples

Each file can be run independently:

```bash
# Test input validation
python input_validator.py

# Test output filtering
python output_filter.py

# Test action constraints
python action_constraints.py

# Test resource management
python resource_manager.py

# Test sandboxing
python sandbox.py

# Run complete guardrails demo
python guardrails.py

# Run agent with guardrails
python agent_with_guardrails.py

# Run exercise solution
python exercise_solution.py
```

## Key Concepts

1. **Input Validation**: Prevent prompt injection and validate data before processing
2. **Output Filtering**: Redact sensitive data and verify output safety
3. **Action Constraints**: Control what tools can be used and with what arguments
4. **Resource Limits**: Prevent runaway costs and infinite loops
5. **Sandboxing**: Isolate dangerous operations to limit potential damage

## Notes

- All examples use `python-dotenv` for secrets management
- The sandbox examples require a Unix-like environment for full functionality
- Resource limits are approximate and should be tuned for your use case
- Always test guardrails with adversarial inputs before production deployment
