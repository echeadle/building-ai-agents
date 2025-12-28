# Appendix E: Troubleshooting Guide - Code Examples

This directory contains practical utilities for diagnosing and fixing common agent problems.

## Files Overview

### 1. `diagnostics.py`
Diagnostic tools for checking your agent configuration.

**Use when:**
- Setting up a new agent
- Agent isn't working as expected
- Need to validate tool definitions

**Key features:**
- Check tool configurations
- Test API connectivity
- Analyze conversation flow
- Validate tool definitions

**Example usage:**
```python
from diagnostics import AgentDiagnostics, run_full_diagnostics

# Run full diagnostics
report = run_full_diagnostics(tools=your_tools)

# Or use specific checks
diagnostics = AgentDiagnostics()
tool_check = diagnostics.check_tools_configuration(tools)
api_check = diagnostics.check_api_connectivity()
```

### 2. `loop_detector.py`
Detect and prevent infinite loops in agent execution.

**Use when:**
- Agent keeps calling the same tool repeatedly
- Agent gets stuck in cycles
- Need to prevent runaway execution

**Key features:**
- Detect repeated tool calls
- Identify cyclic patterns
- Automatic loop prevention
- Tool call statistics

**Example usage:**
```python
from loop_detector import LoopDetectingAgent

agent = LoopDetectingAgent(
    tools=your_tools,
    max_iterations=10
)

result = agent.run("What is 2 + 2?")
stats = agent.loop_detector.get_statistics()
```

### 3. `rate_limiter.py`
Manage API rate limits to prevent 429 errors.

**Use when:**
- Getting rate limit errors
- Making many requests quickly
- Need to control request rate

**Key features:**
- Automatic rate limiting
- Retry with exponential backoff
- Rate usage tracking
- Configurable limits

**Example usage:**
```python
from rate_limiter import RateLimitedAgent, RetryingAgent

# Option 1: Rate-limited agent
agent = RateLimitedAgent(
    tools=your_tools,
    requests_per_minute=50
)
response = agent.query("Your question")

# Option 2: Retrying agent
agent = RetryingAgent(tools=your_tools)
response = agent.query("Your question", max_retries=5)
```

### 4. `token_manager.py`
Manage conversation length to avoid token limit errors.

**Use when:**
- Getting "maximum context length exceeded" errors
- Long conversations losing context
- Need to optimize token usage

**Key features:**
- Automatic conversation trimming
- Token usage estimation
- Conversation summarization
- Token statistics tracking

**Example usage:**
```python
from token_manager import TokenAwareAgent, SummarizingAgent

# Option 1: Trim old messages
agent = TokenAwareAgent(max_conversation_tokens=150000)
response = agent.query("Your question")
stats = agent.get_token_stats()

# Option 2: Summarize old messages
agent = SummarizingAgent(max_conversation_tokens=150000)
response = agent.query("Your question")
```

## Quick Start

### Installation

```bash
# Install dependencies
uv add anthropic python-dotenv

# Set up environment
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

### Basic Troubleshooting Workflow

1. **Check API connectivity:**
```python
from diagnostics import AgentDiagnostics

diagnostics = AgentDiagnostics()
result = diagnostics.check_api_connectivity()
print(result)
```

2. **Validate tool definitions:**
```python
tool_check = diagnostics.check_tools_configuration(your_tools)
if not tool_check["valid"]:
    print("Issues:", tool_check["issues"])
```

3. **Add loop detection:**
```python
from loop_detector import LoopDetectingAgent

agent = LoopDetectingAgent(tools=your_tools)
result = agent.run(user_message)
```

4. **Add rate limiting:**
```python
from rate_limiter import RateLimitedAgent

agent = RateLimitedAgent(
    tools=your_tools,
    requests_per_minute=50
)
```

5. **Add token management:**
```python
from token_manager import TokenAwareAgent

agent = TokenAwareAgent(
    tools=your_tools,
    max_conversation_tokens=150000
)
```

## Common Problems and Solutions

### Problem: Agent won't use tools

**Solution:**
```python
from diagnostics import AgentDiagnostics

# 1. Check tool configuration
diagnostics = AgentDiagnostics()
tool_check = diagnostics.check_tools_configuration(tools)

# 2. Fix any issues found
for issue in tool_check["issues"]:
    print(f"Fix: {issue}")

# 3. Test with API
test_results = diagnostics.test_tool_definitions(tools)
```

### Problem: Agent loops forever

**Solution:**
```python
from loop_detector import LoopDetectingAgent

# Use loop-detecting agent
agent = LoopDetectingAgent(
    tools=tools,
    max_iterations=10  # Will stop after 10 iterations
)

result = agent.run(user_message)
```

### Problem: Rate limit errors

**Solution:**
```python
from rate_limiter import RetryingAgent

# Automatically retry on rate limits
agent = RetryingAgent(tools=tools)
result = agent.query(
    user_message,
    max_retries=5  # Will retry up to 5 times
)
```

### Problem: Token limit errors

**Solution:**
```python
from token_manager import TokenAwareAgent

# Automatically manage conversation length
agent = TokenAwareAgent(
    tools=tools,
    max_conversation_tokens=150000
)

# Check token usage
stats = agent.get_token_stats()
print(f"Using {stats['current_tokens']} / {stats['max_tokens']} tokens")
```

## Combining Utilities

For production agents, combine multiple utilities:

```python
from loop_detector import LoopDetector
from rate_limiter import RateLimiter
from token_manager import TokenEstimator

class ProductionAgent:
    def __init__(self, tools):
        self.tools = tools
        self.loop_detector = LoopDetector()
        self.rate_limiter = RateLimiter(requests_per_minute=50)
        self.token_estimator = TokenEstimator()
        self.conversation = []
    
    def run(self, user_message):
        # Check rate limit
        self.rate_limiter.wait_if_needed()
        
        # Check tokens
        tokens = self.token_estimator.estimate_conversation_tokens(
            self.conversation
        )
        if tokens > 150000:
            self._trim_conversation()
        
        # Add loop detection to your agentic loop
        # ... your agent logic here ...
```

## Testing Your Agent

Use the diagnostic tools to test your agent:

```python
from diagnostics import run_full_diagnostics

# Run complete diagnostic
report = run_full_diagnostics(
    tools=your_tools,
    conversation=your_conversation_history
)

# Check the report
if not report["api_connectivity"]["connected"]:
    print("Fix API connection first!")

if not report["tool_configuration"]["valid"]:
    print("Fix tool definitions!")

if report["conversation_flow"]["issues"]:
    print("Conversation flow issues:", report["conversation_flow"]["issues"])
```

## Tips

1. **Always start with diagnostics** - Run `diagnostics.py` first to identify the problem
2. **Add one fix at a time** - Don't combine all utilities at once; add them as needed
3. **Monitor in production** - Keep track of loop detections, rate limits, and token usage
4. **Test thoroughly** - Use the test functions to verify fixes work

## Need More Help?

Refer to Appendix E in the book for:
- Detailed explanations of each problem
- Additional troubleshooting strategies
- Edge cases and advanced scenarios
- Links to official documentation
