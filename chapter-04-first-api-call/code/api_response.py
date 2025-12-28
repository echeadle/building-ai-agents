"""
Exploring the API response structure.

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

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)

# Explore the response structure
print("=== Full Response Object ===")
print(f"ID: {message.id}")
print(f"Model: {message.model}")
print(f"Role: {message.role}")
print(f"Stop Reason: {message.stop_reason}")
print()

print("=== Token Usage ===")
print(f"Input tokens: {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")
print()

print("=== Content ===")
print(f"Number of content blocks: {len(message.content)}")
print(f"Content type: {message.content[0].type}")
print(f"Text: {message.content[0].text}")

# --- NEW CODE FOR DICTIONARY OUTPUT ---

print("\n=== Full Response Object as a Pretty-Printed Dictionary ===")
# Convert the Pydantic object to a dictionary
message_dict = message.model_dump()
# Pretty-print the dictionary for easy inspection
pprint.pprint(message_dict)