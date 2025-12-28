"""
Your first API call to Claude.

Chapter 4: Your First API Call to Claude
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Create the Anthropic client
# The client automatically uses ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Make the API call
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude! Please introduce yourself in one paragraph."}
    ]
)

# Print the response
print(message.content[0].text)
