"""
Simple two-step prompt chain: Generate content, then translate it.

This example demonstrates the most basic prompt chain pattern where
the output of one step becomes the input of the next.

Chapter 17: Prompt Chaining - Implementation
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

# Initialize client and model
client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


def generate_content(topic: str, style: str = "professional") -> str:
    """
    Step 1: Generate marketing copy about a topic.
    
    Args:
        topic: The subject to write about
        style: The tone of the content (professional, casual, enthusiastic)
    
    Returns:
        Generated marketing copy as a string
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Write a short marketing paragraph (3-4 sentences) about: {topic}
                
Style: {style}

Focus on benefits and include a call to action. Keep it concise."""
            }
        ]
    )
    return response.content[0].text


def translate_content(content: str, target_language: str) -> str:
    """
    Step 2: Translate content to another language.
    
    Args:
        content: The text to translate
        target_language: The language to translate into
    
    Returns:
        Translated content as a string
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Translate the following marketing copy to {target_language}.

Maintain the same tone, style, and persuasive intent. Adapt idioms naturally.

Content to translate:
{content}"""
            }
        ]
    )
    return response.content[0].text


def content_chain(topic: str, target_language: str, style: str = "professional") -> dict:
    """
    Execute the full content generation and translation chain.
    
    This function orchestrates the two-step chain:
    1. Generate marketing content about the topic
    2. Translate the content to the target language
    
    Args:
        topic: The subject to write about
        target_language: The language to translate into
        style: The tone of the content
    
    Returns:
        Dictionary containing original and translated content
    """
    # Step 1: Generate content
    print(f"Step 1: Generating {style} content about '{topic}'...")
    original_content = generate_content(topic, style)
    print(f"Generated: {original_content[:100]}...")
    
    # Step 2: Translate content
    print(f"\nStep 2: Translating to {target_language}...")
    translated_content = translate_content(original_content, target_language)
    print(f"Translated: {translated_content[:100]}...")
    
    return {
        "topic": topic,
        "style": style,
        "target_language": target_language,
        "original": original_content,
        "translated": translated_content
    }


if __name__ == "__main__":
    # Example: Generate enthusiastic marketing copy and translate to Spanish
    result = content_chain(
        topic="cloud-based project management software",
        target_language="Spanish",
        style="enthusiastic"
    )
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"\nTopic: {result['topic']}")
    print(f"Style: {result['style']}")
    print(f"\nOriginal (English):\n{result['original']}")
    print(f"\nTranslated ({result['target_language']}):\n{result['translated']}")
