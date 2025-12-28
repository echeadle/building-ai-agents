"""
Specialized route handlers for customer service queries.

This module demonstrates how to create domain-specific handlers
with tailored system prompts for different types of queries.

Chapter 19: Routing - Implementation
"""

import os
from typing import Callable
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
# System Prompts for Each Handler
# ============================================================
# Each handler has a specialized system prompt that gives it
# domain expertise and appropriate guidelines.

BILLING_SYSTEM_PROMPT = """You are a billing support specialist at a software company. You help customers with:
- Understanding charges and invoices
- Processing refund requests
- Explaining pricing and subscription tiers
- Resolving payment issues

Guidelines:
- Always be empathetic about billing concerns - money matters are stressful
- If a refund might be warranted, acknowledge this and explain the process
- For complex billing issues, offer to escalate to a billing specialist
- Never share specific account balance information without verification
- Provide clear explanations of charges

Keep responses concise, empathetic, and helpful. Aim for 2-3 paragraphs maximum."""

TECHNICAL_SYSTEM_PROMPT = """You are a technical support specialist at a software company. You help customers with:
- Troubleshooting product issues and bugs
- Explaining how to use features
- Diagnosing error messages
- Providing workarounds for known issues

Guidelines:
- Ask clarifying questions if the issue is unclear
- Provide step-by-step solutions when possible
- If you can't resolve the issue, offer to escalate to engineering
- Suggest relevant documentation when appropriate
- Be patient with users of all technical levels

Keep responses clear, structured, and actionable. Use numbered steps for instructions."""

ACCOUNT_SYSTEM_PROMPT = """You are an account support specialist at a software company. You help customers with:
- Password and login issues
- Profile and settings changes
- Subscription management (upgrades, downgrades, cancellations)
- Security concerns and 2FA setup

Guidelines:
- Prioritize account security in all interactions
- Never ask for or confirm full passwords
- For sensitive changes (email, password), explain the verification process
- If there are security concerns, recommend enabling two-factor authentication
- Be clear about what actions the user needs to take

Keep responses helpful and security-conscious."""

GENERAL_SYSTEM_PROMPT = """You are a friendly customer service representative at a software company. You handle:
- General inquiries about the company and products
- Feedback and suggestions
- Complaints and concerns
- Questions that don't fit other categories

Guidelines:
- Be warm and personable in your responses
- For complaints, acknowledge the customer's frustration before addressing the issue
- For feedback, thank them and explain how it will be used
- If the query would be better handled by a specialist (billing, technical, account), suggest that
- Maintain a positive, helpful attitude throughout

Keep responses friendly, warm, and helpful."""


# ============================================================
# Handler Factory
# ============================================================

def create_handler(system_prompt: str, max_tokens: int = 500) -> Callable[[str], str]:
    """
    Create a handler function with the given system prompt.
    
    This factory function returns a new handler function that
    will use the specified system prompt when responding.
    
    Args:
        system_prompt: The system prompt for this handler
        max_tokens: Maximum tokens in the response
        
    Returns:
        A function that handles queries using this system prompt
        
    Example:
        >>> handler = create_handler("You are a helpful assistant.")
        >>> response = handler("Hello!")
    """
    def handler(message: str) -> str:
        """Handle a customer query with the specialized system prompt."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response.content[0].text
    
    return handler


# ============================================================
# Create Handlers for Each Category
# ============================================================

billing_handler = create_handler(BILLING_SYSTEM_PROMPT)
technical_handler = create_handler(TECHNICAL_SYSTEM_PROMPT)
account_handler = create_handler(ACCOUNT_SYSTEM_PROMPT)
general_handler = create_handler(GENERAL_SYSTEM_PROMPT)


# Handler registry - maps category names to handler functions
HANDLERS: dict[str, Callable[[str], str]] = {
    "BILLING": billing_handler,
    "TECHNICAL": technical_handler,
    "ACCOUNT": account_handler,
    "GENERAL": general_handler,
}


def route_to_handler(category: str, message: str) -> str:
    """
    Route a message to the appropriate handler based on category.
    
    Args:
        category: The classified category (BILLING, TECHNICAL, etc.)
        message: The customer's message
        
    Returns:
        The handler's response
        
    Raises:
        KeyError: If category is not found and no default is available
    """
    # Get the handler for this category, defaulting to general
    handler = HANDLERS.get(category.upper(), general_handler)
    return handler(message)


def get_available_categories() -> list[str]:
    """Return list of available routing categories."""
    return list(HANDLERS.keys())


if __name__ == "__main__":
    # Test each handler directly with representative queries
    test_cases = [
        ("BILLING", "I was charged twice for my subscription this month. "
                   "Can you help me understand why and get a refund?"),
        
        ("TECHNICAL", "The app crashes whenever I try to upload a photo "
                     "larger than 5MB. I'm using the latest version on iOS."),
        
        ("ACCOUNT", "I forgot my password and the reset email never arrived. "
                   "I've checked my spam folder too."),
        
        ("GENERAL", "I've been using your product for 2 years now and wanted "
                   "to say how much I love it! Any new features coming soon?"),
    ]
    
    print("=" * 70)
    print("Specialized Route Handlers Demo")
    print("=" * 70)
    
    for category, message in test_cases:
        print(f"\n📁 Category: {category}")
        print(f"💬 Query: {message}")
        print("\n🤖 Response:")
        print("-" * 50)
        response = route_to_handler(category, message)
        print(response)
        print("=" * 70)
