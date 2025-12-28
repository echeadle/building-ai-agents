"""
Complete customer service router with classification and specialized handlers.

This module combines the classifier and handlers into a complete
routing workflow that can handle customer queries end-to-end.

Chapter 19: Routing - Implementation
"""

import os
from dataclasses import dataclass
from typing import Optional
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


# ============================================================
# Classification Component
# ============================================================

CLASSIFICATION_PROMPT = """You are a customer service query classifier. Analyze the message and classify it into exactly ONE category.

Categories:
- BILLING: Charges, invoices, payments, refunds, pricing, subscription costs
- TECHNICAL: Product issues, bugs, errors, how-to questions, setup problems
- ACCOUNT: Password resets, login issues, profile changes, account settings
- GENERAL: General inquiries, feedback, compliments, complaints, other

Rules:
1. Respond with ONLY the category name (BILLING, TECHNICAL, ACCOUNT, or GENERAL)
2. Choose the BEST fit if the query could belong to multiple categories
3. If truly ambiguous, choose GENERAL

Message: {message}

Category:"""

VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}


def classify_query(message: str) -> str:
    """
    Classify a customer query into a category.
    
    Args:
        message: The customer's message to classify
        
    Returns:
        One of: BILLING, TECHNICAL, ACCOUNT, GENERAL
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    category = response.content[0].text.strip().upper()
    
    # Find matching category in response
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            return valid_cat
    
    return "GENERAL"


# ============================================================
# Handler Component
# ============================================================

SYSTEM_PROMPTS = {
    "BILLING": """You are a billing support specialist. You help customers with charges, refunds, invoices, and payment issues.

Guidelines:
- Be empathetic about billing concerns
- Explain charges clearly
- For refund requests, acknowledge and explain the process
- Offer to escalate complex issues

Keep responses concise and helpful (2-3 paragraphs max).""",

    "TECHNICAL": """You are a technical support specialist. You help troubleshoot product issues and explain features.

Guidelines:
- Ask clarifying questions if needed
- Provide step-by-step solutions
- Offer to escalate unresolved issues
- Be patient with all skill levels

Keep responses clear and actionable.""",

    "ACCOUNT": """You are an account support specialist. You help with passwords, settings, and account security.

Guidelines:
- Prioritize security
- Never ask for full passwords
- Explain verification processes
- Recommend 2FA when appropriate

Keep responses helpful and security-conscious.""",

    "GENERAL": """You are a friendly customer service representative. You handle general inquiries, feedback, and anything else.

Guidelines:
- Be warm and personable
- Acknowledge complaints before addressing them
- Thank users for feedback
- Direct to specialists when appropriate

Keep responses friendly and helpful.""",
}


def handle_query(category: str, message: str) -> str:
    """
    Handle a query using the appropriate specialist prompt.
    
    Args:
        category: The classified category
        message: The customer's message
        
    Returns:
        The specialist's response
    """
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["GENERAL"])
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=system_prompt,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    
    return response.content[0].text


# ============================================================
# Router Component (combines classification and handling)
# ============================================================

@dataclass
class RoutingResult:
    """Result from routing a customer query."""
    category: str
    response: str
    original_message: str


def route_customer_query(
    message: str, 
    verbose: bool = False
) -> RoutingResult:
    """
    Route a customer query through classification to the appropriate handler.
    
    This is the main entry point for the routing system. It:
    1. Classifies the incoming message
    2. Routes it to the appropriate handler
    3. Returns the response with metadata
    
    Args:
        message: The customer's message
        verbose: If True, print routing information
        
    Returns:
        RoutingResult with category, response, and original message
        
    Example:
        >>> result = route_customer_query("Why was I charged twice?")
        >>> print(f"Routed to: {result.category}")
        >>> print(f"Response: {result.response}")
    """
    # Step 1: Classify the query
    category = classify_query(message)
    
    if verbose:
        print(f"📋 Classified as: {category}")
    
    # Step 2: Route to appropriate handler
    response = handle_query(category, message)
    
    return RoutingResult(
        category=category,
        response=response,
        original_message=message,
    )


def route_batch(
    messages: list[str], 
    verbose: bool = False
) -> list[RoutingResult]:
    """
    Route multiple customer queries.
    
    Args:
        messages: List of customer messages
        verbose: If True, print routing information
        
    Returns:
        List of RoutingResults
    """
    return [route_customer_query(msg, verbose) for msg in messages]


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Customer Service Router Demo")
    print("=" * 70)
    
    # Test queries covering various categories
    test_queries = [
        # Billing queries
        "I noticed a $99 charge on my account that I don't recognize. "
        "Can you explain what this is for?",
        
        # Technical queries
        "How do I export my data to CSV? I can't find the option anywhere.",
        
        # Account queries
        "I need to change the email address on my account. "
        "My old email is no longer accessible.",
        
        # General queries
        "I just wanted to say your team has been incredibly helpful! "
        "Keep up the great work.",
        
        # Edge case: Billing + Account overlap
        "My subscription renewed but I wanted to cancel it. "
        "Can I get a refund?",
        
        # Edge case: Technical + Billing overlap
        "The checkout page keeps failing when I try to upgrade my plan.",
    ]
    
    for query in test_queries:
        print(f"\n💬 Customer: {query}\n")
        result = route_customer_query(query, verbose=True)
        print(f"\n🤖 Response:\n{result.response}")
        print("-" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("Routing Summary")
    print("=" * 70)
    
    results = route_batch(test_queries)
    category_counts = {}
    for result in results:
        category_counts[result.category] = category_counts.get(result.category, 0) + 1
    
    print("\nQueries by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
