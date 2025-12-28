"""
A complete, reusable function for making API calls to Claude.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class ClaudeResponse:
    """Container for Claude's response and metadata."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def ask_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024
) -> ClaudeResponse:
    """
    Send a prompt to Claude and return a structured response.
    
    Args:
        prompt: The question or instruction to send to Claude
        model: The Claude model to use (default: claude-sonnet-4-20250514)
        max_tokens: Maximum tokens in the response (default: 1024)
        
    Returns:
        ClaudeResponse object containing the response and metadata
        
    Raises:
        anthropic.AuthenticationError: If the API key is invalid
        anthropic.RateLimitError: If rate limits are exceeded
        anthropic.APIConnectionError: If connection fails
        anthropic.APIStatusError: For other API errors
    """
    client = anthropic.Anthropic()
    
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return ClaudeResponse(
        text=message.content[0].text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        stop_reason=message.stop_reason
    )


if __name__ == "__main__":
    # Example usage
    response = ask_claude("What are three interesting facts about octopuses?")
    
    print("Claude says:")
    print(response.text)
    print()
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Stop reason: {response.stop_reason}")
