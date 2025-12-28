"""
Demonstrating different types of agent errors.

Chapter 30: Error Handling and Recovery

This module shows the various categories of errors you'll encounter
when building AI agents, helping you understand what to handle and how.
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


def demonstrate_api_errors():
    """
    Show different API error types.
    
    These are errors from the Anthropic API that you'll need to handle.
    """
    print("--- API Connection Error ---")
    # Connection error (simulated with bad base URL)
    try:
        bad_client = anthropic.Anthropic(
            base_url="https://nonexistent.anthropic.com"
        )
        bad_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.APIConnectionError as e:
        print(f"  Caught: APIConnectionError")
        print(f"  Message: {e}")
        print(f"  This is RETRYABLE - network issues are often transient")
    
    print("\n--- Authentication Error ---")
    # Authentication error (simulated with bad key)
    try:
        bad_client = anthropic.Anthropic(api_key="invalid-key-12345")
        bad_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.AuthenticationError as e:
        print(f"  Caught: AuthenticationError")
        print(f"  Message: {e}")
        print(f"  This is NOT RETRYABLE - fix your API key")
    
    print("\n--- Bad Request Error ---")
    # Bad request error (invalid parameters)
    try:
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=-1,  # Invalid: must be positive
            messages=[{"role": "user", "content": "Hello"}]
        )
    except anthropic.BadRequestError as e:
        print(f"  Caught: BadRequestError")
        print(f"  Message: {e}")
        print(f"  This is NOT RETRYABLE - fix your request parameters")


def demonstrate_output_errors():
    """
    Show LLM output parsing errors.
    
    These happen when the LLM produces output your code can't process.
    """
    print("--- JSON Parse Error ---")
    # Simulating malformed JSON response
    bad_json_examples = [
        '{"name": "test", "value": }',  # Missing value
        "{'single': 'quotes'}",  # Single quotes not valid JSON
        '{"trailing": "comma",}',  # Trailing comma
        'Sure! Here is the JSON: {"data": 1}',  # Extra text
    ]
    
    for i, bad_json in enumerate(bad_json_examples, 1):
        try:
            json.loads(bad_json)
        except json.JSONDecodeError as e:
            print(f"  Example {i}: {bad_json[:30]}...")
            print(f"    Error: {e.msg} at position {e.pos}")
    
    print("\n--- Schema Violation ---")
    # Valid JSON but wrong structure
    expected_schema = {
        "required_fields": ["name", "age", "email"],
        "types": {"name": str, "age": int, "email": str}
    }
    
    responses = [
        '{"wrong_field": "data"}',  # Missing required fields
        '{"name": "John", "age": "thirty", "email": "test@test.com"}',  # Wrong type
        '{"name": "John", "age": 30}',  # Missing email
    ]
    
    for response in responses:
        parsed = json.loads(response)
        missing = [f for f in expected_schema["required_fields"] if f not in parsed]
        
        if missing:
            print(f"  Response: {response}")
            print(f"    Missing fields: {missing}")
        else:
            # Check types
            for field, expected_type in expected_schema["types"].items():
                if not isinstance(parsed.get(field), expected_type):
                    print(f"  Response: {response}")
                    print(f"    Type error: '{field}' should be {expected_type.__name__}")
                    break


def demonstrate_tool_errors():
    """
    Show tool execution errors.
    
    These happen when tools fail during execution.
    """
    print("--- File Not Found ---")
    try:
        with open("/nonexistent/path/file.txt") as f:
            f.read()
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        print(f"  Action: Check file path, use fallback, or inform user")
    
    print("\n--- Permission Denied ---")
    try:
        with open("/etc/shadow", "r") as f:  # Requires root
            f.read()
    except PermissionError as e:
        print(f"  Error: {e}")
        print(f"  Action: Check permissions, run with appropriate access")
    
    print("\n--- Invalid Tool Input ---")
    def calculate_percentage(value: float, total: float) -> float:
        """Calculate what percentage 'value' is of 'total'."""
        if total == 0:
            raise ValueError("Cannot calculate percentage with zero total")
        if value < 0 or total < 0:
            raise ValueError("Values must be non-negative")
        return (value / total) * 100
    
    test_cases = [
        (50, 0),      # Division by zero
        (-10, 100),   # Negative value
        (50, -100),   # Negative total
    ]
    
    for value, total in test_cases:
        try:
            calculate_percentage(value, total)
        except ValueError as e:
            print(f"  Input: value={value}, total={total}")
            print(f"  Error: {e}")
    
    print("\n--- External API Error ---")
    # Simulating external API failure
    class ExternalAPIError(Exception):
        """Error from external service."""
        def __init__(self, status_code: int, message: str):
            self.status_code = status_code
            self.message = message
            super().__init__(f"API Error {status_code}: {message}")
    
    def simulate_external_api_call():
        # Simulating various API failures
        raise ExternalAPIError(503, "Service temporarily unavailable")
    
    try:
        simulate_external_api_call()
    except ExternalAPIError as e:
        print(f"  Error: {e}")
        print(f"  Status Code: {e.status_code}")
        if e.status_code in [502, 503, 504]:
            print(f"  Action: RETRY - this is likely transient")
        elif e.status_code == 401:
            print(f"  Action: Check API credentials")
        elif e.status_code == 429:
            print(f"  Action: Back off - rate limited")


def demonstrate_logic_errors():
    """
    Show agent logic errors.
    
    These are higher-level problems with agent behavior.
    """
    print("--- Infinite Loop Detection ---")
    
    class LoopDetector:
        """Detects when an agent is stuck in a loop."""
        
        def __init__(self, max_identical_actions: int = 3):
            self.action_history: list[str] = []
            self.max_identical = max_identical_actions
        
        def record_action(self, action: str) -> bool:
            """
            Record an action and check for loops.
            
            Returns True if a loop is detected.
            """
            self.action_history.append(action)
            
            # Check if last N actions are identical
            if len(self.action_history) >= self.max_identical:
                recent = self.action_history[-self.max_identical:]
                if len(set(recent)) == 1:  # All identical
                    return True
            
            return False
    
    detector = LoopDetector(max_identical_actions=3)
    actions = [
        "search: weather",
        "search: weather",  
        "search: weather",  # Loop detected!
    ]
    
    for action in actions:
        is_loop = detector.record_action(action)
        status = "LOOP DETECTED!" if is_loop else "ok"
        print(f"  Action: {action} -> {status}")
    
    print("\n--- Resource Exhaustion ---")
    
    class ResourceTracker:
        """Track resource usage to prevent exhaustion."""
        
        def __init__(self, max_tokens: int = 100000, max_api_calls: int = 50):
            self.max_tokens = max_tokens
            self.max_api_calls = max_api_calls
            self.tokens_used = 0
            self.api_calls = 0
        
        def record_usage(self, tokens: int, api_calls: int = 1) -> dict:
            """Record usage and return status."""
            self.tokens_used += tokens
            self.api_calls += api_calls
            
            return {
                "tokens_used": self.tokens_used,
                "tokens_remaining": self.max_tokens - self.tokens_used,
                "api_calls": self.api_calls,
                "calls_remaining": self.max_api_calls - self.api_calls,
                "tokens_exhausted": self.tokens_used >= self.max_tokens,
                "calls_exhausted": self.api_calls >= self.max_api_calls,
            }
    
    tracker = ResourceTracker(max_tokens=1000, max_api_calls=5)
    
    # Simulate several API calls
    for i in range(6):
        status = tracker.record_usage(tokens=200, api_calls=1)
        exhausted = status["tokens_exhausted"] or status["calls_exhausted"]
        print(f"  Call {i+1}: {status['tokens_used']} tokens, {status['api_calls']} calls", end="")
        if exhausted:
            print(" -> RESOURCE LIMIT REACHED!")
            break
        else:
            print()


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT ERROR TYPES DEMONSTRATION")
    print("=" * 60)
    
    print("\n### 1. API AND NETWORK ERRORS ###\n")
    demonstrate_api_errors()
    
    print("\n### 2. LLM OUTPUT ERRORS ###\n")
    demonstrate_output_errors()
    
    print("\n### 3. TOOL EXECUTION ERRORS ###\n")
    demonstrate_tool_errors()
    
    print("\n### 4. AGENT LOGIC ERRORS ###\n")
    demonstrate_logic_errors()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Different errors need different handling!")
    print("- Transient errors: RETRY with backoff")
    print("- Validation errors: FIX the input/request")
    print("- Auth errors: STOP and fix credentials")
    print("- Logic errors: DETECT and intervene")
    print("=" * 60)
