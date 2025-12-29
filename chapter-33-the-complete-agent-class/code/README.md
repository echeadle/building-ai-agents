# Chapter 33: The Complete Agent Class - Code

This directory contains the complete, production-ready Agent class and all supporting modules.

## Files Overview

| File | Description |
|------|-------------|
| `config.py` | Agent configuration system with presets |
| `tools.py` | Tool registry for managing agent tools |
| `state.py` | State management (history, memory, persistence) |
| `guardrails.py` | Safety guardrails (input validation, output filtering) |
| `errors.py` | Error handling with retry logic |
| `agent.py` | The complete Agent class integrating all components |
| `example_usage.py` | Example demonstrating how to use the Agent |
| `exercise_customer_service_agent.py` | Exercise solution: CustomerServiceAgent |

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install anthropic python-dotenv
   ```

2. **Set up your API key:**
   ```bash
   echo "ANTHROPIC_API_KEY=your-key-here" > .env
   ```

3. **Run the example:**
   ```bash
   uv run python example_usage.py
   ```

## Module Dependencies

```
agent.py
├── config.py      (AgentConfig, PlanningMode, HumanApprovalMode)
├── tools.py       (ToolRegistry, ToolDefinition)
├── state.py       (AgentState, Message, ToolCall)
├── guardrails.py  (Guardrails, GuardrailResult)
└── errors.py      (ErrorHandler, AgentError, ErrorSeverity)
```

## Usage Examples

### Basic Agent

```python
from agent import Agent
from config import AgentConfig

# Create with defaults
agent = Agent()

# Or with custom config
config = AgentConfig(
    system_prompt="You are a helpful assistant.",
    max_iterations=10,
    verbose=True
)
agent = Agent(config)

# Run
response = agent.run("Hello, how are you?")
print(response)
```

### Agent with Tools

```python
from agent import Agent

agent = Agent()

# Register a tool
agent.register_tool(
    name="get_weather",
    description="Get weather for a city",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    },
    handler=lambda city: f"Weather in {city}: Sunny, 72°F"
)

response = agent.run("What's the weather in Paris?")
```

### Configuration Presets

```python
from config import AgentConfig

# For simple chat bots
config = AgentConfig.for_simple_chat()

# For autonomous agents
config = AgentConfig.for_autonomous_agent()

# For maximum safety
config = AgentConfig.for_safe_agent()

# For development/testing
config = AgentConfig.for_development()
```

### Extending the Agent

See `exercise_customer_service_agent.py` for a complete example of extending the Agent class for a specific domain.

## Testing Individual Modules

Each module can be run standalone to see a demonstration:

```bash
uv run python config.py      # Shows configuration presets
uv run python tools.py       # Demonstrates tool registry
uv run python state.py       # Shows state management
uv run python guardrails.py  # Demonstrates safety checks
uv run python errors.py      # Shows error handling
uv run python agent.py       # Quick agent demo
```

## Key Concepts

### Configuration
- Use `AgentConfig` for all settings
- Validate with `config.validate()` before creating agent
- Use presets for common use cases

### Tools
- Register tools before running
- Write clear descriptions (LLMs read them!)
- Use `requires_approval=True` for sensitive operations

### State
- Conversation history is managed automatically
- Use `state.set_memory()` for persistent data
- Enable `persist_state=True` for session persistence

### Guardrails
- Input validation blocks malicious inputs
- Output filtering redacts sensitive data
- Action constraints limit tool capabilities

### Error Handling
- Automatic retry with exponential backoff
- Error categorization by severity
- Fallback values for graceful degradation
