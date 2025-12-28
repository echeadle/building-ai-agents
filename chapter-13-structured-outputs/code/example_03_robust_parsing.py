"""
Robust response parsing with multiple fallback strategies.

This example demonstrates how to handle malformed JSON responses
from Claude, including cleaning up common issues and retrying.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
import re
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import TypeVar, Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()

T = TypeVar("T", bound=BaseModel)


class ProductInfo(BaseModel):
    """Schema for product information."""
    
    name: str = Field(description="Product name")
    price: float = Field(ge=0, description="Price in dollars")
    category: str = Field(description="Product category")
    in_stock: bool = Field(default=True, description="Whether item is in stock")
    features: list[str] = Field(default_factory=list, description="Key features")


def clean_json_response(text: str) -> str:
    """
    Clean up common issues in JSON responses from LLMs.
    
    Handles:
    - Markdown code blocks (```json ... ```)
    - Text before/after the JSON
    - Extra whitespace
    
    Args:
        text: Raw response text that should contain JSON
        
    Returns:
        Cleaned text more likely to be valid JSON
    """
    original = text
    
    # Remove markdown code blocks if present
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
            print(f"  [Cleaned] Extracted from ```json code block")
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
            print(f"  [Cleaned] Extracted from ``` code block")
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Remove any text before the first { or [
    json_start_brace = text.find("{")
    json_start_bracket = text.find("[")
    
    # Find the first JSON character
    if json_start_brace == -1:
        json_start = json_start_bracket
    elif json_start_bracket == -1:
        json_start = json_start_brace
    else:
        json_start = min(json_start_brace, json_start_bracket)
    
    if json_start > 0:
        removed = text[:json_start].strip()
        if removed:
            print(f"  [Cleaned] Removed prefix: '{removed[:50]}...'")
        text = text[json_start:]
    
    # Remove any text after the last } or ]
    json_end_brace = text.rfind("}")
    json_end_bracket = text.rfind("]")
    json_end = max(json_end_brace, json_end_bracket)
    
    if json_end != -1 and json_end < len(text) - 1:
        removed = text[json_end + 1:].strip()
        if removed:
            print(f"  [Cleaned] Removed suffix: '{removed[:50]}...'")
        text = text[:json_end + 1]
    
    if text != original:
        print(f"  [Cleaned] Final length: {len(text)} (was {len(original)})")
    
    return text


def parse_json_safely(text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Attempt to parse JSON with detailed error reporting.
    
    Args:
        text: Text to parse as JSON
        
    Returns:
        Tuple of (parsed_data, error_message)
        If successful, error_message is None
        If failed, parsed_data is None
    """
    try:
        data = json.loads(text)
        return data, None
    except json.JSONDecodeError as e:
        # Provide helpful error context
        error_location = text[max(0, e.pos - 20):e.pos + 20]
        return None, f"JSON error at position {e.pos}: {e.msg}. Context: '...{error_location}...'"


def parse_with_retry(
    response_text: str,
    schema: type[T],
    original_prompt: str,
    max_retries: int = 2
) -> T:
    """
    Parse a response with automatic retry on failure.
    
    This function:
    1. Cleans the response text
    2. Attempts to parse as JSON
    3. Validates against the schema
    4. If any step fails, asks Claude to fix the response
    
    Args:
        response_text: The raw response from Claude
        schema: Pydantic model class to validate against
        original_prompt: The original user prompt (for context in retries)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Validated instance of the schema
        
    Raises:
        ValueError: If parsing fails after all retries
    """
    last_error = None
    current_response = response_text
    
    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt + 1}/{max_retries + 1} ---")
        
        try:
            # Step 1: Clean the response
            cleaned = clean_json_response(current_response)
            
            # Step 2: Parse JSON
            data, json_error = parse_json_safely(cleaned)
            if json_error:
                raise json.JSONDecodeError(json_error, cleaned, 0)
            
            # Step 3: Validate against schema
            result = schema.model_validate(data)
            print(f"  [Success] Parsed and validated!")
            return result
            
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            print(f"  [Error] {last_error}")
        except ValidationError as e:
            errors = "; ".join(f"{err['loc']}: {err['msg']}" for err in e.errors())
            last_error = f"Validation failed: {errors}"
            print(f"  [Error] {last_error}")
        
        # If we have retries left, ask Claude to fix the response
        if attempt < max_retries:
            print(f"  [Retry] Requesting corrected response...")
            
            correction_message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system="You are a data extraction assistant. Respond only with valid JSON.",
                messages=[
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": current_response},
                    {
                        "role": "user",
                        "content": f"""Your previous response had an error: {last_error}

Please provide a corrected JSON response. Return ONLY the JSON object, no explanations or markdown."""
                    }
                ]
            )
            
            current_response = correction_message.content[0].text
            print(f"  [Retry] Got new response ({len(current_response)} chars)")
    
    raise ValueError(f"Failed to parse response after {max_retries + 1} attempts: {last_error}")


def extract_product_info(description: str) -> ProductInfo:
    """
    Extract product information with robust parsing.
    
    Args:
        description: Product description text
        
    Returns:
        Validated ProductInfo object
    """
    prompt = f"""Extract product information from this description.

Return a JSON object with:
- name: string (product name)
- price: number (price in dollars)
- category: string (product category)
- in_stock: boolean (availability)
- features: array of strings (key features)

Description: {description}"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a data extraction assistant. Respond only with valid JSON.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return parse_with_retry(
        message.content[0].text,
        ProductInfo,
        prompt,
        max_retries=2
    )


def demonstrate_cleaning():
    """
    Demonstrate the JSON cleaning function on various messy inputs.
    """
    print("=" * 60)
    print("Demonstrating JSON Cleaning")
    print("=" * 60)
    
    messy_responses = [
        # Markdown code block
        '''Here's the extracted information:

```json
{"name": "Widget", "price": 29.99}
```

Let me know if you need anything else!''',

        # Just extra text
        '''Based on the description, here is the JSON:
{"name": "Gadget", "price": 49.99}
I hope this helps!''',

        # Clean JSON (should pass through)
        '{"name": "Doohickey", "price": 19.99}'
    ]
    
    for i, messy in enumerate(messy_responses):
        print(f"\n--- Test {i + 1} ---")
        print(f"Original ({len(messy)} chars):")
        print(f"  {messy[:80]}...")
        cleaned = clean_json_response(messy)
        print(f"Cleaned:")
        print(f"  {cleaned}")
        
        # Verify it's valid JSON
        try:
            data = json.loads(cleaned)
            print(f"  ✓ Valid JSON: {data}")
        except json.JSONDecodeError as e:
            print(f"  ✗ Still invalid: {e}")


if __name__ == "__main__":
    # First, demonstrate the cleaning function
    demonstrate_cleaning()
    
    # Then, do a real extraction with retry logic
    print("\n" + "=" * 60)
    print("Real Extraction with Retry Logic")
    print("=" * 60)
    
    description = """
    The UltraWidget Pro 3000 is our premium gadget, priced at $149.99.
    It falls under our Smart Home category. Currently in stock!
    
    Key features include:
    - Voice control compatibility
    - Energy efficient design
    - 2-year warranty
    - Works with all major smart home platforms
    """
    
    try:
        product = extract_product_info(description)
        print("\n" + "=" * 60)
        print("Final Result")
        print("=" * 60)
        print(f"Name: {product.name}")
        print(f"Price: ${product.price:.2f}")
        print(f"Category: {product.category}")
        print(f"In Stock: {product.in_stock}")
        print(f"Features:")
        for feature in product.features:
            print(f"  - {feature}")
    except ValueError as e:
        print(f"\nExtraction failed: {e}")
