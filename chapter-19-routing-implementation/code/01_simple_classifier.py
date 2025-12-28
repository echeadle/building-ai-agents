"""
Simple LLM-based classifier for routing customer queries.

This module demonstrates how to build a classifier that categorizes
customer messages into predefined categories using Claude.

Chapter 19: Routing - Implementation
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

# Initialize the Anthropic client
client = anthropic.Anthropic()

# The classification prompt - notice the careful structure:
# 1. Clear role definition
# 2. Explicit category definitions with examples
# 3. Rules for handling edge cases
# 4. Strict output format
CLASSIFICATION_PROMPT = """You are a customer service query classifier. Your job is to analyze customer messages and classify them into exactly ONE category.

Categories:
- BILLING: Questions about charges, invoices, payments, refunds, pricing, subscription costs
- TECHNICAL: Issues with product functionality, bugs, errors, how-to questions, setup problems
- ACCOUNT: Password resets, login issues, profile changes, account settings, subscription management
- GENERAL: General inquiries, feedback, compliments, complaints, anything that doesn't fit above

Rules:
1. Respond with ONLY the category name (BILLING, TECHNICAL, ACCOUNT, or GENERAL)
2. Choose the BEST fit even if the query could belong to multiple categories
3. When in doubt, choose GENERAL

Examples:
- "I was charged twice this month" → BILLING
- "The app keeps crashing" → TECHNICAL
- "How do I change my password?" → ACCOUNT
- "I love your product!" → GENERAL

Now classify this customer message:
{message}

Category:"""

# Valid categories for validation
VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}


def classify_query(message: str) -> str:
    """
    Classify a customer query into a category.
    
    This function sends the customer message to Claude along with
    a classification prompt, then parses and validates the response.
    
    Args:
        message: The customer's message to classify
        
    Returns:
        One of: BILLING, TECHNICAL, ACCOUNT, GENERAL
        
    Example:
        >>> category = classify_query("Why was I charged $50?")
        >>> print(category)
        BILLING
    """
    # Make the API call with low max_tokens since we only need one word
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,  # We only need one word
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    # Extract the response text
    category = response.content[0].text.strip().upper()
    
    # Handle potential variations in response format
    # The model might respond "BILLING" or "BILLING." or "The category is BILLING"
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            return valid_cat
    
    # Default fallback if we can't parse the response
    return "GENERAL"


def classify_with_details(message: str) -> dict:
    """
    Classify a query and return detailed information.
    
    Args:
        message: The customer's message to classify
        
    Returns:
        Dictionary with category and metadata
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    raw_response = response.content[0].text.strip()
    category = raw_response.upper()
    
    # Find matching category
    matched_category = "GENERAL"
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            matched_category = valid_cat
            break
    
    return {
        "category": matched_category,
        "raw_response": raw_response,
        "message": message,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


if __name__ == "__main__":
    # Test the classifier with various queries
    test_queries = [
        # Clear-cut cases
        "Why was I charged $50 last week?",
        "The login button doesn't work on mobile",
        "I need to update my email address",
        "What are your business hours?",
        
        # More nuanced cases
        "I paid for premium but still see ads",
        "My subscription renewed but I thought I cancelled",
        "The app is slow when loading large files",
        "Can I get a student discount?",
    ]
    
    print("=" * 60)
    print("Customer Query Classifier Demo")
    print("=" * 60)
    
    for query in test_queries:
        result = classify_with_details(query)
        print(f"\nQuery: {query}")
        print(f"Category: {result['category']}")
        print(f"Tokens used: {result['input_tokens']} in, {result['output_tokens']} out")
        print("-" * 40)
