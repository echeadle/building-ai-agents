# Chapter 39: Latency Optimization - Code Examples

This directory contains all code examples for Chapter 39 of "Building AI Agents from Scratch with Python."

## Files

### Core Modules

| File | Description |
|------|-------------|
| `latency_profiler.py` | Latency profiling and measurement tools |
| `streaming_agent.py` | Streaming responses for improved perceived latency |
| `parallel_executor.py` | Parallel tool execution for reduced latency |
| `response_budget.py` | Response time budgets for predictable performance |
| `model_selector.py` | Model selection based on latency requirements |
| `speed_cache.py` | Speed-optimized caching for agent responses |

### Exercise

| File | Description |
|------|-------------|
| `exercise_solution.py` | Complete latency dashboard with web interface |

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install anthropic python-dotenv
   ```

2. **Set up your API key:**
   ```bash
   echo "ANTHROPIC_API_KEY=your-key-here" > .env
   ```

3. **Run individual modules:**
   ```bash
   python latency_profiler.py      # Profiling demo
   python streaming_agent.py       # Streaming demo
   python parallel_executor.py     # Parallel execution demo
   python response_budget.py       # Budget management demo
   python model_selector.py        # Model selection demo
   python speed_cache.py           # Caching demo
   ```

4. **Run the latency dashboard:**
   ```bash
   python exercise_solution.py
   # Open http://localhost:8080 in your browser
   ```

## Module Details

### latency_profiler.py

Provides tools for measuring and analyzing latency in agent operations:

```python
from latency_profiler import LatencyProfiler

profiler = LatencyProfiler()

with profiler.measure("llm_call", category="llm"):
    response = client.messages.create(...)

profiler.print_report()
```

### streaming_agent.py

Implements streaming responses for better perceived performance:

```python
from streaming_agent import StreamingAgent

agent = StreamingAgent()

for chunk in agent.stream("Tell me about Python"):
    print(chunk, end="", flush=True)
```

### parallel_executor.py

Executes multiple tool calls simultaneously:

```python
from parallel_executor import ParallelToolExecutor

executor = ParallelToolExecutor()
executor.register_tool("weather", get_weather)
executor.register_tool("stock", get_stock)

results = executor.execute_parallel(tool_calls)
```

### response_budget.py

Enforces response time limits:

```python
from response_budget import ResponseTimeBudget

budget = ResponseTimeBudget(total_ms=5000)
budget.allocate(llm_ms=3000, tool_ms=1500, buffer_ms=500)

with budget.track("operation", "llm"):
    # Your code here
    pass

if budget.can_continue(500):
    # More operations
    pass
```

### model_selector.py

Selects optimal models based on latency requirements:

```python
from model_selector import LatencyAwareModelSelector, TaskComplexity

selector = LatencyAwareModelSelector()

model = selector.select(
    task_complexity=TaskComplexity.SIMPLE,
    max_latency_ms=500
)
```

### speed_cache.py

Provides high-performance caching:

```python
from speed_cache import LRUCache, CachedLLMClient

cache = LRUCache(max_size=1000, default_ttl=3600)

result = cache.get(key)
if result is None:
    result = expensive_operation()
    cache.put(key, result)
```

## Key Concepts

1. **Measure before optimizing** - Use the profiler to identify actual bottlenecks

2. **Stream for perceived speed** - Users see progress immediately

3. **Parallelize independent operations** - Run tools simultaneously when possible

4. **Set and enforce budgets** - Guarantee response times

5. **Choose the right model** - Faster models for simple tasks

6. **Cache aggressively** - Avoid redundant work

## Exercise: Latency Dashboard

The exercise solution creates a real-time web dashboard showing:
- Response time distribution (histogram)
- Breakdown by category (LLM, tools, network)
- Cache hit rate
- Slowest operations
- Alerts for high latency

Run with:
```bash
python exercise_solution.py
```

Then open http://localhost:8080 in your browser.

## Dependencies

- Python 3.10+
- anthropic
- python-dotenv

Optional (for exercise):
- No additional dependencies (uses built-in http.server)
