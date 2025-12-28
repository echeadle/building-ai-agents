"""
Self-correction patterns for agent error recovery.

Chapter 30: Error Handling and Recovery

This module demonstrates how agents can detect and fix their own
mistakes, particularly for LLM output errors like invalid JSON
or incorrect tool usage.
"""

import os
import json
import re
from typing import Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# ============================================================
# Correction Result Types
# ============================================================

@dataclass
class CorrectionAttempt:
    """
    Record of a self-correction attempt.
    
    Tracks what was wrong, what was tried, and whether it worked.
    """
    original_output: str
    error_message: str
    corrected_output: Optional[str]
    success: bool
    attempts: int
    method: str = ""  # How it was corrected


# ============================================================
# JSON Extraction and Correction
# ============================================================

def extract_json_from_response(text: str) -> Optional[str]:
    """
    Attempt to extract JSON from a response that may have extra text.
    
    LLMs often wrap JSON in markdown code blocks or add explanatory
    text before/after. This function tries to find the JSON.
    
    Args:
        text: Raw response text that may contain JSON
        
    Returns:
        Extracted JSON string if found, None otherwise
        
    Examples:
        >>> extract_json_from_response('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
        >>> extract_json_from_response('Here is the data: {"key": "value"}')
        '{"key": "value"}'
    """
    # Strategy 1: Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text)
    
    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Try to find raw JSON (object or array)
    json_patterns = [
        r'(\{[\s\S]*\})',  # JSON object
        r'(\[[\s\S]*\])',  # JSON array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    
    return None


def fix_common_json_errors(text: str) -> Optional[str]:
    """
    Attempt to fix common JSON syntax errors.
    
    Handles issues like:
    - Trailing commas
    - Single quotes instead of double
    - Unquoted keys
    - Comments
    
    Args:
        text: Potentially malformed JSON string
        
    Returns:
        Fixed JSON string if successful, None otherwise
    """
    original = text
    
    # Remove comments (// and /* */)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Replace single quotes with double quotes (simple cases)
    # This is imperfect but catches common cases
    text = re.sub(r"'([^']*)':", r'"\1":', text)
    text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
    
    # Remove trailing commas before ] or }
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Try to parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


def self_correct_json(
    original_response: str,
    error: json.JSONDecodeError,
    max_attempts: int = 2
) -> CorrectionAttempt:
    """
    Ask the LLM to fix its own invalid JSON output.
    
    First tries local fixes (extraction, common error fixes),
    then asks the LLM to correct it.
    
    Args:
        original_response: The original response containing invalid JSON
        error: The JSON decode error that occurred
        max_attempts: Maximum LLM correction attempts
        
    Returns:
        CorrectionAttempt with results
    """
    # Strategy 1: Try to extract JSON from the response
    extracted = extract_json_from_response(original_response)
    if extracted:
        try:
            json.loads(extracted)
            return CorrectionAttempt(
                original_output=original_response,
                error_message=str(error),
                corrected_output=extracted,
                success=True,
                attempts=0,
                method="extraction"
            )
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try common fixes
    fixed = fix_common_json_errors(original_response)
    if fixed:
        return CorrectionAttempt(
            original_output=original_response,
            error_message=str(error),
            corrected_output=fixed,
            success=True,
            attempts=0,
            method="common_fixes"
        )
    
    # Strategy 3: Ask the LLM to fix it
    for attempt in range(max_attempts):
        correction_prompt = f"""Your previous response contained invalid JSON that could not be parsed.

Original response:
{original_response}

Error: {error}

Please provide ONLY the corrected, valid JSON with no additional text, explanation, or markdown formatting. Just the raw JSON."""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": correction_prompt}]
            )
            
            corrected = response.content[0].text.strip()
            
            # Try to extract from response (LLM might still add wrapping)
            extracted = extract_json_from_response(corrected)
            if extracted:
                corrected = extracted
            
            # Validate
            json.loads(corrected)
            
            return CorrectionAttempt(
                original_output=original_response,
                error_message=str(error),
                corrected_output=corrected,
                success=True,
                attempts=attempt + 1,
                method="llm_correction"
            )
            
        except json.JSONDecodeError:
            continue
        except anthropic.APIError as e:
            # API error, stop trying
            break
    
    return CorrectionAttempt(
        original_output=original_response,
        error_message=str(error),
        corrected_output=None,
        success=False,
        attempts=max_attempts,
        method="failed"
    )


# ============================================================
# Tool Call Correction
# ============================================================

def self_correct_tool_call(
    tool_name: str,
    tool_input: dict,
    error_message: str,
    available_tools: list[dict],
    conversation_context: Optional[list[dict]] = None,
    max_attempts: int = 2
) -> Optional[dict]:
    """
    Ask the LLM to fix an invalid tool call.
    
    When a tool call fails due to invalid input, this function
    asks the LLM to correct its input based on the error message
    and tool definition.
    
    Args:
        tool_name: The tool that was called
        tool_input: The input that caused the error
        error_message: Description of what went wrong
        available_tools: List of available tool definitions
        conversation_context: Recent conversation for context
        max_attempts: Maximum correction attempts
        
    Returns:
        Corrected tool input dict, or None if correction failed
    """
    # Find the tool definition
    tool_def = next(
        (t for t in available_tools if t["name"] == tool_name),
        None
    )
    
    if not tool_def:
        return None  # Can't correct if we don't know the tool
    
    for attempt in range(max_attempts):
        correction_prompt = f"""Your previous tool call resulted in an error. Please provide a corrected input.

Tool: {tool_name}
Input you provided: {json.dumps(tool_input, indent=2)}
Error: {error_message}

Tool definition:
{json.dumps(tool_def, indent=2)}

Please provide the corrected tool input as a valid JSON object. Output ONLY the JSON object, no explanation or markdown."""

        messages = []
        
        # Include some conversation context if provided
        if conversation_context:
            messages.extend(conversation_context[-2:])
        
        messages.append({"role": "user", "content": correction_prompt})
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages
            )
            
            corrected_text = response.content[0].text.strip()
            
            # Try to extract JSON
            extracted = extract_json_from_response(corrected_text)
            if extracted:
                return json.loads(extracted)
            
            # Try parsing directly
            return json.loads(corrected_text)
            
        except (json.JSONDecodeError, anthropic.APIError):
            continue
    
    return None


# ============================================================
# Self-Correcting Agent
# ============================================================

class SelfCorrectingAgent:
    """
    An agent wrapper that attempts self-correction on errors.
    
    This wraps the basic agent loop and intercepts errors,
    attempting to fix them before giving up.
    """
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        tools: list[dict],
        system_prompt: str = "You are a helpful assistant.",
        max_correction_attempts: int = 2
    ):
        """
        Initialize the self-correcting agent.
        
        Args:
            client: Anthropic client instance
            tools: List of tool definitions
            system_prompt: System prompt for the agent
            max_correction_attempts: Max self-correction tries
        """
        self.client = client
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_correction_attempts = max_correction_attempts
        self.correction_history: list[CorrectionAttempt] = []
        
        # Tool execution registry
        self._tool_handlers: dict[str, callable] = {}
    
    def register_tool(self, name: str, handler: callable):
        """Register a tool execution handler."""
        self._tool_handlers[name] = handler
    
    def _execute_tool(self, name: str, input_data: dict) -> str:
        """
        Execute a tool and return the result as JSON string.
        
        Args:
            name: Tool name
            input_data: Tool input
            
        Returns:
            JSON string with result
            
        Raises:
            ValueError: If tool execution fails
        """
        if name not in self._tool_handlers:
            raise ValueError(f"Unknown tool: {name}")
        
        try:
            result = self._tool_handlers[name](input_data)
            return json.dumps({"result": result})
        except Exception as e:
            raise ValueError(f"Tool execution failed: {e}")
    
    def process_message(self, user_message: str) -> str:
        """
        Process a user message with self-correction capabilities.
        
        Returns the final response after any needed corrections.
        """
        messages = [{"role": "user", "content": user_message}]
        
        while True:
            # Make API call
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages
            )
            
            # Check for tool use
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        try:
                            result = self._execute_tool(tool_name, tool_input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                            
                        except ValueError as e:
                            # Attempt self-correction
                            corrected_input = self_correct_tool_call(
                                tool_name=tool_name,
                                tool_input=tool_input,
                                error_message=str(e),
                                available_tools=self.tools,
                                conversation_context=messages,
                                max_attempts=self.max_correction_attempts
                            )
                            
                            if corrected_input:
                                # Try with corrected input
                                try:
                                    result = self._execute_tool(
                                        tool_name, corrected_input
                                    )
                                    tool_results.append({
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": result
                                    })
                                    
                                    # Record successful correction
                                    self.correction_history.append(
                                        CorrectionAttempt(
                                            original_output=json.dumps(tool_input),
                                            error_message=str(e),
                                            corrected_output=json.dumps(corrected_input),
                                            success=True,
                                            attempts=1,
                                            method="tool_call_correction"
                                        )
                                    )
                                    print(f"  [Self-correction succeeded for {tool_name}]")
                                    continue
                                    
                                except ValueError:
                                    pass  # Correction didn't help
                            
                            # Correction failed, report error to LLM
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({
                                    "error": str(e),
                                    "note": "Tool execution failed, correction unsuccessful"
                                }),
                                "is_error": True
                            })
                            
                            # Record failed correction
                            self.correction_history.append(
                                CorrectionAttempt(
                                    original_output=json.dumps(tool_input),
                                    error_message=str(e),
                                    corrected_output=None,
                                    success=False,
                                    attempts=self.max_correction_attempts,
                                    method="tool_call_correction"
                                )
                            )
                
                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                
            else:
                # End turn - return text response
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                
                return ""


# ============================================================
# Demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SELF-CORRECTION PATTERNS DEMONSTRATION")
    print("=" * 60)
    
    # 1. JSON extraction
    print("\n### 1. JSON Extraction ###\n")
    
    test_cases = [
        '```json\n{"name": "test", "value": 42}\n```',
        'Here is the data you requested: {"name": "test"}',
        'The result is {"items": [1, 2, 3]} as expected.',
    ]
    
    for text in test_cases:
        extracted = extract_json_from_response(text)
        print(f"Input: {text[:40]}...")
        print(f"Extracted: {extracted}")
        print()
    
    # 2. Common JSON fixes
    print("### 2. Common JSON Fixes ###\n")
    
    broken_jsons = [
        '{"name": "test",}',  # Trailing comma
        "{'name': 'test'}",   # Single quotes
        '{"name": "test", // comment\n "value": 1}',  # Comment
    ]
    
    for broken in broken_jsons:
        fixed = fix_common_json_errors(broken)
        print(f"Broken: {broken}")
        print(f"Fixed:  {fixed}")
        print()
    
    # 3. Full JSON self-correction
    print("### 3. LLM JSON Self-Correction ###\n")
    
    invalid_json = """Here's the data you requested:

```json
{
    "name": "Test",
    "values": [1, 2, 3,],
    "nested": {
        "key": "value",
    }
}
```

Let me know if you need anything else!"""
    
    try:
        json.loads(invalid_json)
    except json.JSONDecodeError as e:
        print(f"Original error: {e}")
        print(f"\nAttempting self-correction...")
        
        result = self_correct_json(invalid_json, e, max_attempts=2)
        
        print(f"\nCorrection successful: {result.success}")
        print(f"Method used: {result.method}")
        print(f"Attempts: {result.attempts}")
        
        if result.corrected_output:
            print(f"\nCorrected JSON:")
            parsed = json.loads(result.corrected_output)
            print(json.dumps(parsed, indent=2))
    
    # 4. Self-correcting agent demo
    print("\n### 4. Self-Correcting Agent ###\n")
    
    # Define tools
    tools = [
        {
            "name": "calculator",
            "description": "Perform mathematical calculations. Supports +, -, *, /.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression like '2 + 2' or '10 * 5'"
                    }
                },
                "required": ["expression"]
            }
        },
        {
            "name": "get_length",
            "description": "Get the length of a string.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to measure"
                    }
                },
                "required": ["text"]
            }
        }
    ]
    
    # Create agent
    agent = SelfCorrectingAgent(
        client=client,
        tools=tools,
        system_prompt="You are a helpful assistant with access to tools."
    )
    
    # Register tool handlers
    def calculator_handler(input_data: dict) -> float:
        expr = input_data.get("expression", "")
        # Safe eval for basic math only
        allowed_chars = set("0123456789+-*/.(). ")
        if not all(c in allowed_chars for c in expr):
            raise ValueError(f"Invalid expression: {expr}")
        return eval(expr)
    
    def get_length_handler(input_data: dict) -> int:
        text = input_data.get("text")
        if text is None:
            raise ValueError("Missing 'text' parameter")
        return len(text)
    
    agent.register_tool("calculator", calculator_handler)
    agent.register_tool("get_length", get_length_handler)
    
    # Test the agent
    print("Testing self-correcting agent...")
    print("Query: What is 15 * 7?")
    
    response = agent.process_message("What is 15 * 7?")
    print(f"Response: {response}")
    
    print(f"\nCorrection history: {len(agent.correction_history)} attempts recorded")
    
    print("\n" + "=" * 60)
    print("SELF-CORRECTION STRATEGIES:")
    print("1. Extract JSON from wrapped responses")
    print("2. Fix common syntax errors locally")
    print("3. Ask LLM to correct its output")
    print("4. Retry tool calls with corrected input")
    print("=" * 60)
