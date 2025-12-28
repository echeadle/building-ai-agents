"""
Basic JSON output requesting from Claude.

This example demonstrates the simplest approaches to getting
structured JSON responses from Claude.

Chapter 13: Structured Outputs and Response Parsing
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


def extract_book_info_basic(description: str) -> dict:
    """
    Extract structured book information from a text description.
    Uses a simple prompt-based approach.
    
    Args:
        description: A text description of a book
        
    Returns:
        A dictionary containing extracted book information
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Extract information from this book description and return it as JSON.

The JSON should have these fields:
- title: the book's title
- author: the author's name
- year: publication year (as a number)
- genres: a list of genres

Book description:
{description}

Return ONLY the JSON object, no other text."""
            }
        ]
    )
    
    response_text = message.content[0].text
    return json.loads(response_text)


def extract_book_info_with_system_prompt(description: str) -> dict:
    """
    Extract book information using a system prompt for consistency.
    This approach is more reliable for production use.
    
    Args:
        description: A text description of a book
        
    Returns:
        A dictionary containing extracted book information
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a data extraction assistant. You always respond with valid JSON and nothing else.
Never include explanations, markdown formatting, or additional text.
If you cannot extract a field, use null for that field.""",
        messages=[
            {
                "role": "user",
                "content": f"""Extract book information from this description.

Required JSON format:
{{
    "title": "string",
    "author": "string", 
    "year": number or null,
    "genres": ["string"]
}}

Description: {description}"""
            }
        ]
    )
    
    return json.loads(message.content[0].text)


def extract_multiple_items(descriptions: list[str]) -> list[dict]:
    """
    Extract information from multiple book descriptions.
    Demonstrates consistent JSON output across multiple requests.
    
    Args:
        descriptions: List of book descriptions
        
    Returns:
        List of dictionaries with extracted information
    """
    results = []
    
    for desc in descriptions:
        try:
            info = extract_book_info_with_system_prompt(desc)
            results.append(info)
        except json.JSONDecodeError as e:
            print(f"Failed to parse response for: {desc[:50]}...")
            print(f"Error: {e}")
            results.append(None)
    
    return results


if __name__ == "__main__":
    # Example 1: Basic extraction
    print("=" * 60)
    print("Example 1: Basic JSON Extraction")
    print("=" * 60)
    
    description1 = """
    "The Hitchhiker's Guide to the Galaxy" is a comedic science fiction novel 
    written by Douglas Adams, first published in 1979. It's a blend of science 
    fiction and comedy that has become a cult classic.
    """
    
    result1 = extract_book_info_basic(description1)
    print("Extracted information:")
    print(json.dumps(result1, indent=2))
    
    # Example 2: With system prompt
    print("\n" + "=" * 60)
    print("Example 2: With System Prompt")
    print("=" * 60)
    
    description2 = """
    "1984" by George Orwell, published in 1949, is a dystopian novel 
    that explores themes of totalitarianism and surveillance.
    """
    
    result2 = extract_book_info_with_system_prompt(description2)
    print("Extracted information:")
    print(json.dumps(result2, indent=2))
    
    # Example 3: Multiple items
    print("\n" + "=" * 60)
    print("Example 3: Multiple Items")
    print("=" * 60)
    
    descriptions = [
        "Pride and Prejudice by Jane Austen (1813) - a classic romance novel.",
        "Dune by Frank Herbert, 1965 - an epic science fiction saga.",
        "The Great Gatsby, F. Scott Fitzgerald, 1925 - American literary fiction."
    ]
    
    results = extract_multiple_items(descriptions)
    for i, result in enumerate(results):
        print(f"\nBook {i + 1}:")
        print(json.dumps(result, indent=2))
