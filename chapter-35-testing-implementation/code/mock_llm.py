"""
Mock LLM for deterministic agent testing.

Chapter 35: Testing AI Agents - Implementation

This module provides a MockLLM class that simulates the Anthropic API
for testing purposes, allowing deterministic and controllable tests.
"""

from typing import Any, Callable
from unittest.mock import Mock
from dataclasses import dataclass, field


@dataclass
class MockContentBlock:
    """
    Represents a content block in an API response.
    
    Mimics the structure of actual Anthropic API content blocks.
    """
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass  
class MockMessage:
    """
    Represents a complete API response message.
    
    Mimics the structure of actual Anthropic API responses.
    """
    id: str = "msg_mock"
    type: str = "message"
    role: str = "assistant"
    content: list = field(default_factory=list)
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    
    @property
    def usage(self):
        """Return mock usage statistics."""
        mock_usage = Mock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 20
        return mock_usage


class MockLLM:
    """
    A mock LLM that returns predetermined responses.
    
    This class simulates the Anthropic API for testing purposes,
    allowing you to control exactly what the "LLM" returns.
    
    Features:
    - Queue responses to be returned in order
    - Pattern-based response matching
    - Call history tracking for assertions
    - Method chaining for easy setup
    
    Example:
        >>> mock = MockLLM()
        >>> mock.add_response(text="Hello!")
        >>> mock.add_response(tool_call={"name": "calculator", "input": {"a": 1, "b": 2}})
        >>> 
        >>> # Simulate API calls
        >>> r1 = mock.create_message(messages=[{"role": "user", "content": "Hi"}])
        >>> print(r1.content[0].text)  # "Hello!"
        >>> 
        >>> # Check what was called
        >>> print(mock.call_history)
    """
    
    def __init__(self):
        """Initialize the mock with empty response queue and history."""
        self.responses: list[MockMessage] = []
        self.response_index: int = 0
        self.call_history: list[dict] = []
        self.pattern_responses: list[tuple[Callable, MockMessage]] = []
    
    def add_response(
        self,
        text: str | None = None,
        tool_call: dict | None = None,
        stop_reason: str = "end_turn"
    ) -> "MockLLM":
        """
        Add a response to the queue of responses.
        
        Responses are returned in the order they were added.
        
        Args:
            text: Text content to return
            tool_call: Tool call to include (requires name and input keys)
            stop_reason: The stop reason (end_turn, tool_use, etc.)
            
        Returns:
            self, for method chaining
            
        Example:
            >>> mock = MockLLM()
            >>> mock.add_response(text="First").add_response(text="Second")
        """
        content = []
        
        if text:
            content.append(MockContentBlock(type="text", text=text))
        
        if tool_call:
            content.append(MockContentBlock(
                type="tool_use",
                id=tool_call.get("id", f"toolu_mock_{len(self.responses)}"),
                name=tool_call["name"],
                input=tool_call.get("input", {})
            ))
            stop_reason = "tool_use"
        
        self.responses.append(MockMessage(
            content=content,
            stop_reason=stop_reason
        ))
        
        return self
    
    def add_pattern_response(
        self,
        pattern_matcher: Callable[[list[dict]], bool],
        response: MockMessage
    ) -> "MockLLM":
        """
        Add a response that triggers when a pattern matches.
        
        Pattern responses are checked before the queue and allow
        for more dynamic test behavior.
        
        Args:
            pattern_matcher: A function that takes messages and returns True if matched
            response: The response to return when matched
            
        Returns:
            self, for method chaining
            
        Example:
            >>> mock = MockLLM()
            >>> mock.add_pattern_response(
            ...     lambda msgs: "hello" in msgs[-1]["content"].lower(),
            ...     create_text_response("Hello to you too!")
            ... )
        """
        self.pattern_responses.append((pattern_matcher, response))
        return self
    
    def create_message(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        messages: list[dict] = None,
        system: str = None,
        tools: list[dict] = None,
        **kwargs
    ) -> MockMessage:
        """
        Simulates the Anthropic messages.create() API call.
        
        Records the call for later inspection and returns the next
        queued response or matches a pattern response.
        
        Args:
            model: The model to use (ignored, but recorded)
            max_tokens: Maximum tokens (ignored, but recorded)
            messages: The conversation messages
            system: The system prompt
            tools: Tool definitions
            **kwargs: Additional parameters
            
        Returns:
            The next MockMessage from the queue
        """
        # Record this call for later inspection
        self.call_history.append({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages or [],
            "system": system,
            "tools": tools,
            "kwargs": kwargs
        })
        
        # Check pattern-based responses first
        for matcher, response in self.pattern_responses:
            try:
                if matcher(messages or []):
                    return response
            except Exception:
                # If matcher fails, continue to next
                pass
        
        # Return next queued response
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1
            return response
        
        # Default response if queue is empty
        return MockMessage(
            content=[MockContentBlock(
                type="text", 
                text="[MockLLM] No more mock responses available."
            )]
        )
    
    def reset(self):
        """
        Reset the mock to its initial state.
        
        Clears call history and resets the response index.
        The response queue is preserved.
        """
        self.response_index = 0
        self.call_history.clear()
    
    def clear_all(self):
        """
        Completely reset the mock, including the response queue.
        """
        self.responses.clear()
        self.pattern_responses.clear()
        self.reset()
    
    def assert_called(self) -> bool:
        """Verify that create_message was called at least once."""
        return len(self.call_history) > 0
    
    def assert_called_times(self, n: int) -> bool:
        """Verify that create_message was called exactly n times."""
        return len(self.call_history) == n
    
    def assert_called_with_tool(self, tool_name: str) -> bool:
        """Verify that the LLM was called with a specific tool available."""
        for call in self.call_history:
            tools = call.get("tools", [])
            for tool in tools or []:
                if tool.get("name") == tool_name:
                    return True
        return False
    
    def assert_called_with_system(self, system_contains: str) -> bool:
        """Verify that the LLM was called with a system prompt containing text."""
        for call in self.call_history:
            system = call.get("system", "")
            if system and system_contains in system:
                return True
        return False
    
    def get_last_messages(self) -> list[dict]:
        """Get the messages from the most recent call."""
        if self.call_history:
            return self.call_history[-1].get("messages", [])
        return []
    
    def get_all_user_messages(self) -> list[str]:
        """Extract all user message content from call history."""
        user_messages = []
        for call in self.call_history:
            for msg in call.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_messages.append(content)
        return user_messages


# =============================================================================
# Helper Functions
# =============================================================================

def create_tool_response(
    tool_name: str,
    tool_input: dict,
    tool_id: str = None
) -> MockMessage:
    """
    Helper function to create a mock tool use response.
    
    Args:
        tool_name: Name of the tool being called
        tool_input: Arguments for the tool
        tool_id: ID for the tool use block (auto-generated if not provided)
        
    Returns:
        A MockMessage configured for tool use
        
    Example:
        >>> response = create_tool_response("calculator", {"a": 1, "b": 2})
        >>> print(response.stop_reason)  # "tool_use"
    """
    if tool_id is None:
        tool_id = f"toolu_{tool_name}_test"
    
    return MockMessage(
        content=[
            MockContentBlock(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input
            )
        ],
        stop_reason="tool_use"
    )


def create_text_response(text: str, stop_reason: str = "end_turn") -> MockMessage:
    """
    Helper function to create a mock text response.
    
    Args:
        text: The text content to return
        stop_reason: The stop reason (default: end_turn)
        
    Returns:
        A MockMessage with text content
        
    Example:
        >>> response = create_text_response("Hello, world!")
        >>> print(response.content[0].text)  # "Hello, world!"
    """
    return MockMessage(
        content=[MockContentBlock(type="text", text=text)],
        stop_reason=stop_reason
    )


def create_multi_content_response(
    text: str,
    tool_name: str,
    tool_input: dict,
    tool_id: str = None
) -> MockMessage:
    """
    Create a response with both text and a tool call.
    
    Some responses include both text (thinking/explanation) and
    a tool call. This helper creates such responses.
    
    Args:
        text: The text content
        tool_name: Name of the tool
        tool_input: Tool input parameters
        tool_id: Optional tool ID
        
    Returns:
        A MockMessage with both text and tool_use blocks
    """
    if tool_id is None:
        tool_id = f"toolu_{tool_name}_test"
    
    return MockMessage(
        content=[
            MockContentBlock(type="text", text=text),
            MockContentBlock(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input
            )
        ],
        stop_reason="tool_use"
    )


if __name__ == "__main__":
    # Demonstration
    print("MockLLM Demonstration")
    print("=" * 40)
    
    # Create mock with queued responses
    mock = MockLLM()
    mock.add_response(text="Hello! How can I help you?")
    mock.add_response(
        tool_call={
            "name": "calculator",
            "input": {"operation": "add", "a": 5, "b": 3}
        }
    )
    mock.add_response(text="The result is 8!")
    
    # Simulate conversation
    messages = [{"role": "user", "content": "Hello!"}]
    
    print("\n1. First call (should get greeting):")
    r1 = mock.create_message(messages=messages)
    print(f"   Response: {r1.content[0].text}")
    print(f"   Stop reason: {r1.stop_reason}")
    
    print("\n2. Second call (should get tool use):")
    r2 = mock.create_message(messages=messages)
    print(f"   Tool: {r2.content[0].name}")
    print(f"   Input: {r2.content[0].input}")
    print(f"   Stop reason: {r2.stop_reason}")
    
    print("\n3. Third call (should get result):")
    r3 = mock.create_message(messages=messages)
    print(f"   Response: {r3.content[0].text}")
    
    print(f"\n4. Call history length: {len(mock.call_history)}")
    print(f"   Was called: {mock.assert_called()}")
