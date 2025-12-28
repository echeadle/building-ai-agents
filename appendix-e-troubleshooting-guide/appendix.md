---
appendix: E
title: "Troubleshooting Guide"
date: 2024-12-09
draft: false
---

# Appendix E: Troubleshooting Guide

This appendix provides systematic solutions to the most common agent problems you'll encounter. Each section follows the same pattern: symptoms, diagnosis, and fixes.

> **💡 Pro Tip:** Before diving into specific problems, always check your logs first. Most agent issues become obvious when you can see what the agent is thinking and doing.

---

## How to Use This Guide

1. **Identify the symptom** - What is the agent doing wrong?
2. **Follow the diagnosis steps** - Gather information
3. **Apply the fix** - Try solutions in order
4. **Verify** - Test that the problem is resolved
5. **Document** - Note what worked for future reference

---

## Problem 1: Agent Won't Use Tools

### Symptoms

- Agent responds with text instead of calling available tools
- Agent says it "doesn't have access" to tools you've provided
- Agent apologizes for not being able to help

### Diagnosis

```python
# Check your setup
def diagnose_tool_availability():
    """Verify tools are properly configured."""
    
    # 1. Are tools being passed to the API?
    print("Tools in API call:", [t["name"] for t in tools])
    
    # 2. Is the response indicating tool use is possible?
    print("Stop reason:", response.stop_reason)
    # Should be "tool_use" or "end_turn", not "max_tokens"
    
    # 3. Are tool descriptions clear?
    for tool in tools:
        print(f"\n{tool['name']}:")
        print(f"  Description: {tool['description']}")
        print(f"  Parameters: {tool['input_schema']['properties'].keys()}")
```

### Common Causes and Fixes

#### Cause 1: Tools Not Included in API Call

**Fix:**
```python
# ❌ Wrong
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather?"}]
    # Missing: tools parameter!
)

# ✅ Correct
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=tools  # Don't forget this!
)
```

#### Cause 2: Vague Tool Descriptions

**Fix:**
```python
# ❌ Vague description
{
    "name": "get_weather",
    "description": "Gets weather",  # Too vague!
    "input_schema": {...}
}

# ✅ Clear description
{
    "name": "get_weather",
    "description": "Get current weather conditions for a specific city. Use this tool when the user asks about weather, temperature, or current conditions.",
    "input_schema": {...}
}
```

#### Cause 3: System Prompt Discourages Tool Use

**Fix:**
```python
# ❌ Confusing system prompt
system_prompt = """You are a helpful assistant.
If you don't know something, just say so.
Only use tools when absolutely necessary."""  # This discourages tool use!

# ✅ Clear system prompt
system_prompt = """You are a helpful assistant with access to tools.
When a user asks a question that requires real-time data or calculations,
use the appropriate tool. Always prefer using tools over making up information."""
```

#### Cause 4: max_tokens Too Low

**Fix:**
```python
# ❌ Too low - agent can't fit tool call
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=50,  # Not enough for tool calls!
    ...
)

# ✅ Adequate tokens
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,  # Plenty of room
    ...
)
```

### Verification

After applying fixes:

```python
def verify_tool_use():
    """Verify the agent is now using tools."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": "What's the weather in London?"}],
        tools=tools
    )
    
    # Check for tool use
    assert response.stop_reason == "tool_use", f"Expected tool_use, got {response.stop_reason}"
    
    # Check that a tool was called
    tool_uses = [block for block in response.content if block.type == "tool_use"]
    assert len(tool_uses) > 0, "No tools were called"
    
    print(f"✅ Tool '{tool_uses[0].name}' called successfully")
```

---

## Problem 2: Agent Uses Wrong Tool

### Symptoms

- Agent calls `calculate` when it should call `search`
- Agent uses tools in illogical sequences
- Agent tries to use non-existent tools

### Diagnosis

```python
def analyze_tool_selection(conversation_history):
    """Analyze why wrong tools were selected."""
    
    for turn in conversation_history:
        if turn["role"] == "assistant":
            for block in turn["content"]:
                if block.type == "tool_use":
                    print(f"Tool called: {block.name}")
                    print(f"Input: {block.input}")
                    print(f"Context: {turn['user_query']}\n")
```

### Common Causes and Fixes

#### Cause 1: Similar Tool Names

**Fix:**
```python
# ❌ Confusing names
tools = [
    {"name": "search", "description": "Search the database"},
    {"name": "search_web", "description": "Search the internet"},
    {"name": "search_files", "description": "Search files"}
]

# ✅ Distinct names
tools = [
    {"name": "database_query", "description": "Query the local database"},
    {"name": "web_search", "description": "Search the internet using a search engine"},
    {"name": "file_search", "description": "Search through uploaded files"}
]
```

#### Cause 2: Overlapping Tool Descriptions

**Fix:**
```python
# ❌ Overlapping descriptions
tools = [
    {
        "name": "get_data",
        "description": "Get data from various sources"  # Too broad!
    },
    {
        "name": "fetch_info",
        "description": "Fetch information from APIs"  # Also broad!
    }
]

# ✅ Clear boundaries
tools = [
    {
        "name": "get_user_data",
        "description": "Get user profile data from the authentication database. Use this for user-specific information like name, email, preferences."
    },
    {
        "name": "fetch_weather",
        "description": "Fetch current weather from the weather API. Use this only for weather-related queries."
    }
]
```

#### Cause 3: Missing Usage Examples in Descriptions

**Fix:**
```python
# ❌ No examples
{
    "name": "calculate",
    "description": "Performs calculations"
}

# ✅ With examples
{
    "name": "calculate",
    "description": """Performs mathematical calculations on numbers.
    
Use this tool for:
- Arithmetic operations (addition, subtraction, multiplication, division)
- Mathematical expressions like "2 + 2" or "15 * 8"
- Percentage calculations

Do NOT use this for:
- Date/time calculations (use date_calculator instead)
- Unit conversions (use unit_converter instead)"""
}
```

#### Cause 4: No Explicit When-to-Use Guidance

**Fix:**
```python
# Add usage guidance to system prompt
system_prompt = """You are a helpful assistant with access to tools.

Tool Usage Guidelines:
- Use 'web_search' for current events, news, or information not in your knowledge
- Use 'calculator' for any mathematical operations
- Use 'get_user_profile' only when user information is specifically requested
- Always prefer the most specific tool available

If you're unsure which tool to use, start with the most specific option."""
```

### Verification

```python
def verify_correct_tool_selection():
    """Test that the agent chooses correct tools."""
    
    test_cases = [
        {
            "query": "What's 25 * 37?",
            "expected_tool": "calculator",
        },
        {
            "query": "What's happening in the news today?",
            "expected_tool": "web_search",
        },
        {
            "query": "What's my email address?",
            "expected_tool": "get_user_profile",
        }
    ]
    
    for test in test_cases:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": test["query"]}],
            tools=tools
        )
        
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        actual_tool = tool_uses[0].name if tool_uses else None
        
        if actual_tool == test["expected_tool"]:
            print(f"✅ '{test['query']}' → {actual_tool}")
        else:
            print(f"❌ '{test['query']}' → {actual_tool} (expected {test['expected_tool']})")
```

---

## Problem 3: Agent Loops Forever

### Symptoms

- Agent makes the same tool call repeatedly
- Agent gets stuck in a cycle of tool calls
- Max iterations reached without completing task

### Diagnosis

```python
def detect_loops(conversation_history):
    """Detect infinite loops in tool calls."""
    
    tool_sequence = []
    for turn in conversation_history:
        if turn["role"] == "assistant":
            for block in turn["content"]:
                if block.type == "tool_use":
                    tool_sequence.append(f"{block.name}({block.input})")
    
    # Check for repeated patterns
    for i in range(len(tool_sequence) - 1):
        for j in range(i + 1, len(tool_sequence)):
            if tool_sequence[i] == tool_sequence[j]:
                print(f"⚠️ Repeated call detected: {tool_sequence[i]}")
                print(f"   At positions {i} and {j}")
    
    # Check for cycles
    if len(tool_sequence) > 3:
        last_three = tool_sequence[-3:]
        if last_three[0] == last_three[2]:
            print(f"⚠️ Possible cycle: {' → '.join(last_three)}")
```

### Common Causes and Fixes

#### Cause 1: Tool Returns Error But Agent Keeps Trying

**Fix:**
```python
# ❌ Tool that returns generic error
def bad_tool(param):
    try:
        return do_something(param)
    except Exception:
        return "Error occurred"  # Too vague!

# ✅ Tool that returns helpful error
def good_tool(param):
    try:
        return do_something(param)
    except ValueError as e:
        return {
            "error": "invalid_parameter",
            "message": f"The parameter '{param}' is invalid: {str(e)}",
            "suggestion": "Please provide a valid parameter format"
        }
    except Exception as e:
        return {
            "error": "execution_failed",
            "message": str(e),
            "suggestion": "This tool cannot complete the requested operation. Try a different approach."
        }
```

#### Cause 2: Missing Loop Detection

**Fix:**
```python
class LoopDetectingAgent:
    """Agent with built-in loop detection."""
    
    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.tool_call_history: list[str] = []
    
    def run(self, user_message: str) -> str:
        """Run agent with loop detection."""
        conversation = [{"role": "user", "content": user_message}]
        
        for iteration in range(self.max_iterations):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=conversation,
                tools=tools
            )
            
            # Check for loops
            if self._detect_loop(response):
                return "Loop detected. Stopping to prevent infinite iterations."
            
            # Process response...
            if response.stop_reason == "end_turn":
                return self._extract_text(response)
            
            # Continue conversation...
        
        return "Max iterations reached without completion."
    
    def _detect_loop(self, response) -> bool:
        """Detect if we're in a loop."""
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        
        for tool_use in tool_uses:
            call_signature = f"{tool_use.name}({tool_use.input})"
            
            # Check if we've made this exact call recently
            if call_signature in self.tool_call_history[-3:]:
                print(f"⚠️ Loop detected: {call_signature}")
                return True
            
            self.tool_call_history.append(call_signature)
        
        return False
```

#### Cause 3: Tool Results Don't Progress Toward Goal

**Fix:**
```python
# Add progress tracking to system prompt
system_prompt = """You are a helpful assistant with access to tools.

Important: After each tool call, evaluate if you're making progress toward answering the user's question.

If a tool call:
- Returns an error → Try a different approach, don't retry the same call
- Returns partial information → Use it and move forward
- Returns irrelevant information → Acknowledge this and try a different tool

If you've made 3 tool calls without progress, summarize what you've tried and ask the user for clarification or different information."""
```

#### Cause 4: No Stopping Condition

**Fix:**
```python
class ImprovedAgent:
    """Agent with multiple stopping conditions."""
    
    def run(self, user_message: str, max_iterations: int = 10) -> str:
        """Run agent with multiple exit conditions."""
        conversation = [{"role": "user", "content": user_message}]
        successful_tool_calls = 0
        failed_tool_calls = 0
        
        for iteration in range(max_iterations):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=conversation,
                tools=tools
            )
            
            # Exit condition 1: Agent is done
            if response.stop_reason == "end_turn":
                return self._extract_text(response)
            
            # Exit condition 2: Too many failures
            if failed_tool_calls >= 3:
                return "Unable to complete task after multiple failures."
            
            # Exit condition 3: No progress after several successful calls
            if successful_tool_calls >= 5 and iteration > 7:
                return "Task is taking too long. Providing best available answer."
            
            # Process tool calls and update counters...
            results = self._execute_tools(response)
            for result in results:
                if "error" in result:
                    failed_tool_calls += 1
                else:
                    successful_tool_calls += 1
            
            # Continue conversation...
        
        return "Max iterations reached."
```

### Verification

```python
def test_loop_prevention():
    """Verify loop detection works."""
    
    # Create a scenario that would cause loops
    def broken_tool(query: str) -> dict:
        """A tool that always returns an error."""
        return {"error": "not_found", "message": "Information not available"}
    
    agent = LoopDetectingAgent(max_iterations=5)
    
    # This should stop before max iterations due to loop detection
    result = agent.run("Find information using the broken tool")
    
    assert "Loop detected" in result or "Max iterations" in result
    assert len(agent.tool_call_history) < 5  # Should stop early
    print("✅ Loop prevention working")
```

---

## Problem 4: Agent Gives Inconsistent Results

### Symptoms

- Same input produces different outputs
- Agent's behavior changes unpredictably
- Results don't match testing behavior

### Diagnosis

```python
def test_consistency(prompt: str, n_runs: int = 5):
    """Test if agent gives consistent results."""
    results = []
    
    for i in range(n_runs):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            temperature=1.0,  # Note the temperature setting
            messages=[{"role": "user", "content": prompt}],
            tools=tools
        )
        
        result = extract_result(response)
        results.append(result)
        print(f"Run {i+1}: {result}")
    
    # Check for consistency
    unique_results = set(results)
    if len(unique_results) == 1:
        print("✅ Perfectly consistent")
    else:
        print(f"⚠️ Got {len(unique_results)} different results")
        print(f"Results: {unique_results}")
```

### Common Causes and Fixes

#### Cause 1: High Temperature Setting

**Fix:**
```python
# ❌ High temperature = more randomness
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=1.0,  # Maximum randomness
    messages=messages
)

# ✅ Lower temperature = more consistency
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    temperature=0.2,  # More deterministic
    messages=messages
)
```

> **Note:** Temperature affects randomness:
> - `0.0` = Most deterministic (but not 100% deterministic)
> - `1.0` = Most random
> - For production agents, use `0.2-0.4`

#### Cause 2: Non-Deterministic Tool Results

**Fix:**
```python
# ❌ Tool that returns random results
def get_recommendations(category: str) -> list:
    items = database.query(category)
    return random.sample(items, 5)  # Different every time!

# ✅ Tool with deterministic results
def get_recommendations(category: str, seed: int = 42) -> list:
    items = database.query(category)
    # Sort to ensure consistent ordering
    items = sorted(items, key=lambda x: x["popularity"], reverse=True)
    return items[:5]  # Top 5, always the same
```

#### Cause 3: Ambiguous Prompts

**Fix:**
```python
# ❌ Ambiguous prompt
prompt = "Analyze this data and tell me what's important"
# Agent might focus on different aspects each time

# ✅ Specific prompt
prompt = """Analyze this sales data and provide:
1. Total revenue for the quarter
2. Top 3 performing products
3. Comparison to last quarter

Use exact numbers from the data."""
```

#### Cause 4: Timing-Dependent Results

**Fix:**
```python
# ❌ Results depend on current time
def get_trending_topics() -> list:
    # Returns different results based on when it's called
    return api.get_trending_now()

# ✅ Consistent results with explicit time
def get_trending_topics(date: str = None) -> list:
    # Returns same results for same date
    if date is None:
        date = "2024-01-01"  # Use fixed date for testing
    return api.get_trending_for_date(date)
```

#### Cause 5: Incomplete Conversation History

**Fix:**
```python
# ❌ Not preserving context
def query_agent(question: str) -> str:
    # Each call starts fresh
    messages = [{"role": "user", "content": question}]
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=messages
    )
    return extract_text(response)

# ✅ Maintaining conversation history
class ConsistentAgent:
    def __init__(self):
        self.conversation_history = []
    
    def query(self, question: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=self.conversation_history
        )
        
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return extract_text(response)
```

### Verification

```python
def verify_consistency():
    """Verify agent gives consistent results."""
    
    test_query = "What is 15 * 23 + 47?"
    results = []
    
    # Run multiple times
    for _ in range(10):
        agent = ImprovedAgent(temperature=0.2)
        result = agent.query(test_query)
        results.append(result)
    
    # All results should be identical for math
    assert len(set(results)) == 1, f"Got inconsistent results: {set(results)}"
    print("✅ Agent is consistent")
```

---

## Problem 5: Token Limit Errors

### Symptoms

- Error: "maximum context length exceeded"
- Conversation cuts off mid-response
- Agent loses track of earlier conversation

### Diagnosis

```python
def diagnose_token_usage():
    """Analyze token usage in conversation."""
    import anthropic
    
    # Count tokens in conversation
    total_tokens = 0
    for message in conversation_history:
        # Rough estimate: 1 token ≈ 4 characters
        content_str = str(message["content"])
        estimated_tokens = len(content_str) / 4
        total_tokens += estimated_tokens
        print(f"{message['role']}: ~{int(estimated_tokens)} tokens")
    
    print(f"\nTotal: ~{int(total_tokens)} tokens")
    print(f"Model limit: 200,000 tokens")
    print(f"Remaining: ~{200000 - int(total_tokens)} tokens")
```

### Common Causes and Fixes

#### Cause 1: Conversation Too Long

**Fix:**
```python
class TokenAwareAgent:
    """Agent that manages conversation length."""
    
    def __init__(self, max_conversation_tokens: int = 150000):
        self.conversation_history = []
        self.max_tokens = max_conversation_tokens
    
    def add_message(self, role: str, content: any):
        """Add message with token management."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Check if we need to trim
        if self._estimate_tokens() > self.max_tokens:
            self._trim_conversation()
    
    def _estimate_tokens(self) -> int:
        """Estimate total tokens in conversation."""
        total = 0
        for msg in self.conversation_history:
            content_str = str(msg["content"])
            total += len(content_str) / 4
        return int(total)
    
    def _trim_conversation(self):
        """Remove old messages to stay under limit."""
        # Always keep system message and most recent messages
        if len(self.conversation_history) > 10:
            # Keep first message (system) and last 8 messages
            self.conversation_history = [
                self.conversation_history[0]
            ] + self.conversation_history[-8:]
            print("⚠️ Trimmed conversation to manage tokens")
```

#### Cause 2: Returning Large Tool Results

**Fix:**
```python
# ❌ Tool returns everything
def search_database(query: str) -> str:
    results = database.search(query)
    # Might return megabytes of data!
    return json.dumps(results)

# ✅ Tool returns summary
def search_database(query: str, max_results: int = 10) -> str:
    results = database.search(query)
    
    # Limit and summarize
    limited_results = results[:max_results]
    
    summary = {
        "total_found": len(results),
        "showing": len(limited_results),
        "results": [
            {
                "id": r["id"],
                "title": r["title"],
                "snippet": r["content"][:200]  # First 200 chars only
            }
            for r in limited_results
        ]
    }
    
    return json.dumps(summary)
```

#### Cause 3: Including Unnecessary Context

**Fix:**
```python
# ❌ Including everything
system_prompt = f"""You are a helpful assistant.

Here is the entire user manual: {10000_page_manual}
Here is all company data: {entire_database}
Here are all past conversations: {all_history}"""

# ✅ Including only relevant context
def build_system_prompt(user_query: str) -> str:
    # Retrieve only relevant sections
    relevant_manual_sections = retrieve_relevant(user_query, manual)
    relevant_data = retrieve_relevant(user_query, database)
    
    return f"""You are a helpful assistant.

Relevant manual sections:
{relevant_manual_sections}

Relevant data:
{relevant_data}

Answer the user's question using this information."""
```

#### Cause 4: Not Using max_tokens Effectively

**Fix:**
```python
# ❌ May generate very long responses
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,  # May return 4096 tokens!
    messages=messages
)

# ✅ Limit response length appropriately
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,  # Reasonable length for most responses
    messages=messages
)

# For longer responses, use streaming
stream = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=messages,
    stream=True
)
```

### Verification

```python
def test_token_management():
    """Test that agent handles long conversations."""
    
    agent = TokenAwareAgent(max_conversation_tokens=1000)
    
    # Simulate long conversation
    for i in range(50):
        agent.add_message("user", f"Question {i}")
        agent.add_message("assistant", f"Answer {i}")
    
    # Verify conversation was trimmed
    assert len(agent.conversation_history) < 50
    print(f"✅ Conversation trimmed to {len(agent.conversation_history)} messages")
    
    # Verify we can still make API calls
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=agent.conversation_history
    )
    assert response is not None
    print("✅ Can still make API calls")
```

---

## Problem 6: Rate Limit Errors

### Symptoms

- Error 429: "rate_limit_error"
- "Too many requests"
- Intermittent failures during high usage

### Diagnosis

```python
def check_rate_limits():
    """Check current rate limit status."""
    import time
    
    # Make several requests and time them
    times = []
    errors = []
    
    for i in range(5):
        start = time.time()
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hi"}]
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Request {i+1}: {elapsed:.2f}s")
        except anthropic.RateLimitError as e:
            errors.append(str(e))
            print(f"Request {i+1}: Rate limited!")
        time.sleep(1)
    
    if errors:
        print(f"\n⚠️ Hit rate limit {len(errors)} times")
    else:
        avg_time = sum(times) / len(times)
        print(f"\n✅ No rate limits. Avg response time: {avg_time:.2f}s")
```

### Common Causes and Fixes

#### Cause 1: No Rate Limiting in Agent

**Fix:**
```python
import time
from typing import Optional

class RateLimitedAgent:
    """Agent with built-in rate limiting."""
    
    def __init__(self, requests_per_minute: int = 50):
        self.requests_per_minute = requests_per_minute
        self.request_times: list[float] = []
    
    def _wait_if_needed(self):
        """Wait if we're approaching rate limit."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times
            if now - t < 60
        ]
        
        # If we're at the limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            oldest = self.request_times[0]
            wait_time = 60 - (now - oldest) + 1  # +1 for safety
            print(f"Rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        self.request_times.append(now)
    
    def call_api(self, messages: list) -> dict:
        """Make API call with rate limiting."""
        self._wait_if_needed()
        
        return client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages
        )
```

#### Cause 2: No Retry Logic

**Fix:**
```python
import anthropic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RetryingAgent:
    """Agent that retries on rate limits."""
    
    @retry(
        retry=retry_if_exception_type(anthropic.RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def call_api(self, messages: list) -> dict:
        """Make API call with automatic retries."""
        return client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages
        )
    
    def call_api_manual_retry(self, messages: list) -> dict:
        """Manual retry implementation."""
        max_retries = 5
        base_wait = 2
        
        for attempt in range(max_retries):
            try:
                return client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=messages
                )
            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise  # Final attempt, give up
                
                # Exponential backoff
                wait_time = base_wait * (2 ** attempt)
                print(f"Rate limited. Retrying in {wait_time}s...")
                time.sleep(wait_time)
```

#### Cause 3: Parallel Requests Without Coordination

**Fix:**
```python
import asyncio
from asyncio import Semaphore

class ConcurrentAgent:
    """Agent that limits concurrent requests."""
    
    def __init__(self, max_concurrent: int = 5):
        self.semaphore = Semaphore(max_concurrent)
    
    async def call_api_async(self, messages: list) -> dict:
        """Make API call with concurrency limit."""
        async with self.semaphore:
            # Only max_concurrent requests run at once
            return await client.messages.create_async(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages
            )
    
    async def process_batch(self, message_list: list[list]) -> list[dict]:
        """Process multiple requests with concurrency control."""
        tasks = [
            self.call_api_async(messages)
            for messages in message_list
        ]
        return await asyncio.gather(*tasks)
```

#### Cause 4: Not Handling 429 Errors

**Fix:**
```python
def robust_api_call(messages: list) -> Optional[dict]:
    """API call with proper error handling."""
    
    try:
        return client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=messages
        )
    
    except anthropic.RateLimitError as e:
        print("Rate limit exceeded.")
        # Check if headers provide retry-after
        if hasattr(e, 'response') and e.response:
            retry_after = e.response.headers.get('retry-after')
            if retry_after:
                wait_time = int(retry_after)
                print(f"Waiting {wait_time}s as suggested by API")
                time.sleep(wait_time)
                return robust_api_call(messages)  # Retry once
        
        print("Could not complete request due to rate limits")
        return None
    
    except anthropic.APIConnectionError:
        print("Could not connect to API")
        return None
    
    except anthropic.APIStatusError as e:
        print(f"API error: {e.status_code}")
        return None
```

### Verification

```python
def test_rate_limiting():
    """Test that rate limiting works."""
    
    agent = RateLimitedAgent(requests_per_minute=5)
    
    # Make 10 requests quickly
    start = time.time()
    for i in range(10):
        response = agent.call_api([
            {"role": "user", "content": f"Test {i}"}
        ])
    elapsed = time.time() - start
    
    # Should take at least 60 seconds (due to rate limiting)
    assert elapsed >= 60, "Rate limiting not working"
    print(f"✅ Rate limiting working. Took {elapsed:.1f}s for 10 requests")
```

---

## Quick Reference: Common Error Messages

| Error Message | Most Likely Cause | Quick Fix |
|--------------|-------------------|-----------|
| `Tool use not supported` | Tools not passed to API | Add `tools` parameter |
| `maximum context length exceeded` | Conversation too long | Trim conversation history |
| `rate_limit_error` | Too many requests | Add retry with backoff |
| `invalid_request_error` | Malformed tool schema | Check tool definitions |
| `Authentication failed` | Missing/invalid API key | Check `.env` file |
| `Model not found` | Wrong model name | Use `claude-sonnet-4-20250514` |

---

## Debugging Checklist

Before filing a bug report, verify:

- [ ] All API keys are loaded from `.env`
- [ ] Tools are included in API call if needed
- [ ] `max_tokens` is sufficient (at least 1024)
- [ ] Temperature is set appropriately (0.2-0.4 for production)
- [ ] Logging is enabled and you've reviewed logs
- [ ] You can reproduce the issue consistently
- [ ] You've tried the specific fixes in this guide
- [ ] Conversation history is being managed properly
- [ ] Error handling is in place

---

## Additional Resources

For problems not covered here:

- **Anthropic Documentation**: https://docs.anthropic.com
- **API Status Page**: https://status.anthropic.com
- **Support**: Check current rate limits and quotas in your dashboard

Remember: Most agent problems are configuration issues, not bugs. Start with the basics and work your way up!
