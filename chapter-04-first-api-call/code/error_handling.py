"""
Handling API errors gracefully.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


def ask_claude(prompt: str) -> str:
    """
    Send a prompt to Claude and return the response text.
    
    Args:
        prompt: The question or instruction to send to Claude
        
    Returns:
        Claude's response as a string
        
    Raises:
        Various anthropic exceptions on API errors
    """
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
        
    except anthropic.AuthenticationError:
        # Invalid API key
        print("Error: Invalid API key. Please check your ANTHROPIC_API_KEY.")
        raise
        
    except anthropic.RateLimitError:
        # Too many requests
        print("Error: Rate limit exceeded. Please wait a moment and try again.")
        raise
        
    except anthropic.APIConnectionError:
        # Network issues
        print("Error: Could not connect to Anthropic API. Check your internet connection.")
        raise
        
    except anthropic.BadRequestError as e:
        # Invalid request (e.g., bad parameters)
        print(f"Error: Bad request - {e.message}")
        raise
        
    except anthropic.APIStatusError as e:
        # Other API errors
        print(f"Error: API returned status {e.status_code}")
        raise


# Test the function
if __name__ == "__main__":
    response = ask_claude("What's the capital of France?")
    print(response)
