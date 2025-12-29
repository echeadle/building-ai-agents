---
chapter: 19
title: "Routing - Implementation"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 19: Routing - Implementation

## Introduction

In the previous chapter, we explored the concept of routing—directing inputs to specialized handlers based on their content. We examined classification strategies, discussed when to use LLM-based versus rule-based routing, and designed the architecture for a customer service router.

Now it's time to build it.

In this chapter, you'll implement a complete routing system from scratch. We'll start with a simple LLM-based classifier, build specialized route handlers, and assemble them into a fully functional customer service router. Along the way, you'll learn how to test classification accuracy—a critical skill, because as we'll see, **good routing depends on good classification**.

By the end of this chapter, you'll have a reusable `Router` class that you can adapt for any application that needs to direct queries to specialized handlers.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement an LLM-based classifier that categorizes inputs into predefined routes
- Build specialized route handlers that process different types of queries
- Assemble a complete routing workflow for customer service queries
- Test and measure classification accuracy using a test dataset
- Create a reusable `Router` class for future projects

## Building an LLM-Based Classifier

The heart of any routing system is the **classifier**—the component that decides which route to take. In Chapter 18, we discussed two approaches: rule-based (using keywords or patterns) and LLM-based (using the model's understanding). Let's implement the LLM-based approach first, since it's more flexible and handles nuance better.

### The Classification Prompt

For classification to work reliably, we need a clear, specific prompt that tells the LLM exactly what we want:

```python
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
```

Notice several important design choices:

1. **Clear categories with examples**: Each category has specific examples of what belongs there.
2. **Explicit rules**: We tell the model exactly how to respond (category name only) and how to handle edge cases.
3. **Examples**: Few-shot examples help the model understand the boundaries between categories.
4. **Single output format**: Requesting just the category name makes parsing trivial.

### Implementing the Classifier

Let's build a simple classifier function:

```python
"""
Simple LLM-based classifier for routing customer queries.

Chapter 19: Routing - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

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
        max_tokens=10,  # We only need one word
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    # Extract and validate the category
    category = response.content[0].text.strip().upper()
    
    # Handle potential variations in response
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            return valid_cat
    
    # Default fallback
    return "GENERAL"


if __name__ == "__main__":
    # Test the classifier
    test_queries = [
        "Why was I charged $50 last week?",
        "The login button doesn't work on mobile",
        "I need to update my email address",
        "What are your business hours?",
    ]
    
    for query in test_queries:
        category = classify_query(query)
        print(f"Query: {query}")
        print(f"Category: {category}\n")
```

**Key implementation details:**

- **Low `max_tokens`**: We set `max_tokens=10` because we only expect a single word response. This reduces cost and latency.
- **Validation**: We check that the response is one of our valid categories.
- **Robust parsing**: We search for valid categories within the response rather than requiring an exact match. This handles cases where the model might respond "BILLING." instead of "BILLING".
- **Default fallback**: If parsing fails completely, we default to "GENERAL" rather than crashing.

## Creating Specialized Route Handlers

Now that we can classify queries, we need handlers for each category. A **handler** is a function (or class) that processes a specific type of query. Each handler can have its own system prompt, tools, and behavior optimized for its domain.

### Handler Design Principles

Good handlers follow these principles:

1. **Domain expertise**: Each handler's system prompt should contain specialized knowledge for its domain.
2. **Appropriate tools**: Only include tools relevant to the handler's purpose.
3. **Consistent interface**: All handlers should accept the same input and return the same output type.
4. **Graceful degradation**: Handlers should provide helpful responses even when they can't fully resolve the issue.

### Implementing Route Handlers

Let's create handlers for each of our categories:

```python
"""
Specialized route handlers for customer service queries.

Chapter 19: Routing - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# System prompts for each handler
BILLING_SYSTEM_PROMPT = """You are a billing support specialist. You help customers with:
- Understanding charges and invoices
- Processing refund requests
- Explaining pricing and subscription tiers
- Resolving payment issues

Guidelines:
- Always be empathetic about billing concerns
- If a refund might be warranted, acknowledge this and explain the process
- For complex billing issues, offer to escalate to a billing specialist
- Never share specific account balance information in this demo

Keep responses concise and helpful."""

TECHNICAL_SYSTEM_PROMPT = """You are a technical support specialist. You help customers with:
- Troubleshooting product issues
- Explaining how to use features
- Diagnosing error messages
- Providing workarounds for known bugs

Guidelines:
- Ask clarifying questions if the issue is unclear
- Provide step-by-step solutions when possible
- If you can't resolve the issue, offer to escalate to engineering
- Suggest relevant documentation or resources

Keep responses clear and actionable."""

ACCOUNT_SYSTEM_PROMPT = """You are an account support specialist. You help customers with:
- Password and login issues
- Profile and settings changes
- Subscription management
- Security concerns

Guidelines:
- Prioritize account security in all interactions
- Never ask for or confirm full passwords
- For sensitive changes, explain the verification process
- If there are security concerns, recommend enabling 2FA

Keep responses helpful and security-conscious."""

GENERAL_SYSTEM_PROMPT = """You are a friendly customer service representative. You handle:
- General inquiries about the company and products
- Feedback and suggestions
- Complaints and concerns
- Anything that doesn't fit other categories

Guidelines:
- Be warm and personable
- For complaints, acknowledge the customer's frustration
- For feedback, thank them and explain how it will be used
- If the query would be better handled by a specialist, say so

Keep responses friendly and helpful."""


def create_handler(system_prompt: str):
    """
    Create a handler function with the given system prompt.
    
    Args:
        system_prompt: The system prompt for this handler
        
    Returns:
        A function that handles queries using this system prompt
    """
    def handler(message: str) -> str:
        """Handle a customer query with the specialized system prompt."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        return response.content[0].text
    
    return handler


# Create handlers for each category
billing_handler = create_handler(BILLING_SYSTEM_PROMPT)
technical_handler = create_handler(TECHNICAL_SYSTEM_PROMPT)
account_handler = create_handler(ACCOUNT_SYSTEM_PROMPT)
general_handler = create_handler(GENERAL_SYSTEM_PROMPT)

# Map categories to handlers
HANDLERS = {
    "BILLING": billing_handler,
    "TECHNICAL": technical_handler,
    "ACCOUNT": account_handler,
    "GENERAL": general_handler,
}


def route_to_handler(category: str, message: str) -> str:
    """
    Route a message to the appropriate handler based on category.
    
    Args:
        category: The classified category
        message: The customer's message
        
    Returns:
        The handler's response
    """
    handler = HANDLERS.get(category, general_handler)
    return handler(message)


if __name__ == "__main__":
    # Test each handler directly
    test_cases = [
        ("BILLING", "I was charged twice for my subscription this month."),
        ("TECHNICAL", "The app crashes whenever I try to upload a photo."),
        ("ACCOUNT", "I forgot my password and can't log in."),
        ("GENERAL", "What makes your product different from competitors?"),
    ]
    
    for category, message in test_cases:
        print(f"Category: {category}")
        print(f"Query: {message}")
        print(f"Response: {route_to_handler(category, message)}")
        print("-" * 50)
```

**Design notes:**

- **Factory function**: `create_handler()` is a factory that produces handler functions with embedded system prompts. This keeps the code DRY.
- **Handler registry**: The `HANDLERS` dictionary maps category names to handler functions, making routing a simple lookup.
- **Default fallback**: `route_to_handler()` falls back to the general handler if given an unknown category.

## Building the Complete Customer Service Router

Now let's combine the classifier and handlers into a complete routing workflow:

```python
"""
Complete customer service router with classification and specialized handlers.

Chapter 19: Routing - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


# ============ Classification ============

CLASSIFICATION_PROMPT = """You are a customer service query classifier. Analyze the message and classify it into exactly ONE category.

Categories:
- BILLING: Charges, invoices, payments, refunds, pricing, subscription costs
- TECHNICAL: Product issues, bugs, errors, how-to questions, setup problems
- ACCOUNT: Password resets, login issues, profile changes, account settings
- GENERAL: General inquiries, feedback, compliments, complaints, other

Respond with ONLY the category name.

Message: {message}

Category:"""

VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}


def classify_query(message: str) -> str:
    """Classify a customer query into a category."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    category = response.content[0].text.strip().upper()
    
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            return valid_cat
    
    return "GENERAL"


# ============ Handlers ============

SYSTEM_PROMPTS = {
    "BILLING": """You are a billing support specialist. Help with charges, refunds, and payment issues.
Be empathetic about billing concerns. Offer to escalate complex issues. Keep responses concise.""",
    
    "TECHNICAL": """You are a technical support specialist. Help troubleshoot product issues and explain features.
Provide step-by-step solutions. Ask clarifying questions if needed. Keep responses clear and actionable.""",
    
    "ACCOUNT": """You are an account support specialist. Help with passwords, settings, and security.
Prioritize security. Never ask for full passwords. Recommend 2FA for security concerns.""",
    
    "GENERAL": """You are a friendly customer service representative. Handle general inquiries and feedback.
Be warm and personable. Acknowledge complaints. Thank customers for feedback.""",
}


def handle_query(category: str, message: str) -> str:
    """Handle a query with the appropriate specialist."""
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


# ============ Router ============

def route_customer_query(message: str, verbose: bool = False) -> dict:
    """
    Route a customer query through classification to the appropriate handler.
    
    Args:
        message: The customer's message
        verbose: If True, print routing information
        
    Returns:
        Dictionary with category, response, and metadata
    """
    # Step 1: Classify the query
    category = classify_query(message)
    
    if verbose:
        print(f"📋 Classified as: {category}")
    
    # Step 2: Route to appropriate handler
    response = handle_query(category, message)
    
    return {
        "category": category,
        "response": response,
        "message": message,
    }


if __name__ == "__main__":
    # Interactive demo
    print("=" * 60)
    print("Customer Service Router Demo")
    print("=" * 60)
    
    test_queries = [
        "I noticed a $99 charge on my account that I don't recognize.",
        "How do I export my data to CSV?",
        "I need to change the email address on my account.",
        "I just wanted to say your team has been incredibly helpful!",
        "My subscription renewed but I wanted to cancel it.",
    ]
    
    for query in test_queries:
        print(f"\n💬 Customer: {query}\n")
        result = route_customer_query(query, verbose=True)
        print(f"\n🤖 Response:\n{result['response']}")
        print("-" * 60)
```

When you run this, you'll see each query get classified and routed to the appropriate handler. The verbose mode shows which category was selected, making it easy to debug routing decisions.

## Testing Classification Accuracy

Here's a crucial insight: **your routing system is only as good as your classifier**. If queries get routed to the wrong handler, customers get unhelpful responses. That's why testing classification accuracy is essential.

### Creating a Test Dataset

A good test dataset has:

1. **Representative samples**: Queries similar to what you'll see in production
2. **Edge cases**: Ambiguous queries that could belong to multiple categories
3. **Ground truth labels**: The correct category for each query

```python
"""
Testing classification accuracy for the customer service router.

Chapter 19: Routing - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Tuple

load_dotenv()

client = anthropic.Anthropic()


# ============ Classifier (same as before) ============

CLASSIFICATION_PROMPT = """You are a customer service query classifier. Analyze the message and classify it into exactly ONE category.

Categories:
- BILLING: Charges, invoices, payments, refunds, pricing, subscription costs
- TECHNICAL: Product issues, bugs, errors, how-to questions, setup problems
- ACCOUNT: Password resets, login issues, profile changes, account settings
- GENERAL: General inquiries, feedback, compliments, complaints, other

Respond with ONLY the category name.

Message: {message}

Category:"""

VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}


def classify_query(message: str) -> str:
    """Classify a customer query into a category."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    category = response.content[0].text.strip().upper()
    
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            return valid_cat
    
    return "GENERAL"


# ============ Test Dataset ============

# Format: (query, expected_category)
TEST_DATASET: list[Tuple[str, str]] = [
    # BILLING - Clear cases
    ("Why was I charged $50 yesterday?", "BILLING"),
    ("I need a refund for my last purchase", "BILLING"),
    ("What's the difference between the basic and pro plans?", "BILLING"),
    ("My credit card was declined", "BILLING"),
    ("Can I get an invoice for tax purposes?", "BILLING"),
    
    # TECHNICAL - Clear cases
    ("The app crashes when I click the submit button", "TECHNICAL"),
    ("How do I enable dark mode?", "TECHNICAL"),
    ("I'm getting an error message: 'Connection timeout'", "TECHNICAL"),
    ("The search feature isn't returning any results", "TECHNICAL"),
    ("How do I integrate with the API?", "TECHNICAL"),
    
    # ACCOUNT - Clear cases
    ("I forgot my password", "ACCOUNT"),
    ("How do I change my username?", "ACCOUNT"),
    ("I can't log in to my account", "ACCOUNT"),
    ("I want to delete my account", "ACCOUNT"),
    ("How do I enable two-factor authentication?", "ACCOUNT"),
    
    # GENERAL - Clear cases
    ("What are your office hours?", "GENERAL"),
    ("I love your product!", "GENERAL"),
    ("Do you have a mobile app?", "GENERAL"),
    ("I want to provide some feedback", "GENERAL"),
    ("Where is your company located?", "GENERAL"),
    
    # Edge cases - These are trickier
    ("I was charged but never received my order", "BILLING"),  # Billing + fulfillment
    ("My password reset email never arrived", "ACCOUNT"),  # Account + technical
    ("The pricing page shows different prices than my invoice", "BILLING"),  # Billing + technical
    ("I can't access my subscription benefits", "ACCOUNT"),  # Account + billing
    ("When will the new feature be available?", "GENERAL"),  # Could be technical
]


def run_accuracy_test() -> dict:
    """
    Run the classifier against the test dataset and report accuracy.
    
    Returns:
        Dictionary with accuracy metrics and detailed results
    """
    correct = 0
    total = len(TEST_DATASET)
    results = []
    
    # Track per-category performance
    category_stats = {cat: {"correct": 0, "total": 0} for cat in VALID_CATEGORIES}
    
    print("Running classification tests...\n")
    
    for query, expected in TEST_DATASET:
        predicted = classify_query(query)
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
            category_stats[expected]["correct"] += 1
        
        category_stats[expected]["total"] += 1
        
        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })
        
        # Show progress
        status = "✓" if is_correct else "✗"
        if not is_correct:
            print(f"{status} '{query[:50]}...'")
            print(f"   Expected: {expected}, Got: {predicted}\n")
    
    # Calculate overall accuracy
    accuracy = correct / total * 100
    
    # Calculate per-category accuracy
    category_accuracy = {}
    for cat, stats in category_stats.items():
        if stats["total"] > 0:
            category_accuracy[cat] = stats["correct"] / stats["total"] * 100
        else:
            category_accuracy[cat] = 0.0
    
    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
        "results": results,
    }


def print_report(metrics: dict) -> None:
    """Print a formatted accuracy report."""
    print("\n" + "=" * 50)
    print("CLASSIFICATION ACCURACY REPORT")
    print("=" * 50)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.1f}%")
    print(f"Correct: {metrics['correct']} / {metrics['total']}")
    
    print("\nPer-Category Accuracy:")
    for category, accuracy in metrics["category_accuracy"].items():
        print(f"  {category}: {accuracy:.1f}%")
    
    # Show misclassifications
    misclassified = [r for r in metrics["results"] if not r["correct"]]
    if misclassified:
        print(f"\nMisclassifications ({len(misclassified)}):")
        for r in misclassified:
            print(f"  • {r['query'][:60]}...")
            print(f"    Expected: {r['expected']}, Got: {r['predicted']}")
    else:
        print("\n✓ All queries classified correctly!")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    metrics = run_accuracy_test()
    print_report(metrics)
```

### Interpreting Results

When you run the test, pay attention to:

1. **Overall accuracy**: Aim for 90%+ for production use.
2. **Per-category accuracy**: Identify categories that are frequently confused.
3. **Misclassification patterns**: Look for systematic errors that suggest prompt improvements.

If accuracy is low, consider:

- **Adding more examples** to the classification prompt
- **Clarifying category boundaries** in the prompt
- **Creating sub-categories** for ambiguous areas
- **Using rule-based pre-routing** for clear-cut cases

> **💡 Tip:** Run accuracy tests whenever you modify the classification prompt. Small changes can have surprising effects on classification behavior.

## The Router Class Pattern

For production use, let's wrap everything in a clean, reusable `Router` class:

```python
"""
Reusable Router class for building routing workflows.

Chapter 19: Routing - Implementation
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class RouteResult:
    """Result from routing a query."""
    route: str
    response: str
    original_message: str
    confidence: Optional[str] = None


class Router:
    """
    A flexible routing system that classifies inputs and directs them
    to specialized handlers.
    
    Example usage:
        router = Router()
        router.add_route("BILLING", billing_prompt, billing_handler)
        router.add_route("TECHNICAL", technical_prompt, technical_handler)
        result = router.route("Why was I charged twice?")
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        default_route: str = "DEFAULT",
    ):
        """
        Initialize the router.
        
        Args:
            model: The Claude model to use
            default_route: Name of the default route for unclassified queries
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.default_route = default_route
        
        self.routes: dict[str, dict] = {}
        self.classification_prompt_template: Optional[str] = None
    
    def add_route(
        self,
        name: str,
        description: str,
        handler: Callable[[str], str],
        system_prompt: Optional[str] = None,
    ) -> "Router":
        """
        Add a route to the router.
        
        Args:
            name: Route name (e.g., "BILLING")
            description: What this route handles (used in classification)
            handler: Function that handles queries for this route
            system_prompt: Optional system prompt if handler uses default handling
            
        Returns:
            Self, for method chaining
        """
        self.routes[name] = {
            "description": description,
            "handler": handler,
            "system_prompt": system_prompt,
        }
        
        # Rebuild classification prompt when routes change
        self._build_classification_prompt()
        
        return self
    
    def _build_classification_prompt(self) -> None:
        """Build the classification prompt from registered routes."""
        route_descriptions = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.routes.items()
        )
        
        route_names = ", ".join(self.routes.keys())
        
        self.classification_prompt_template = f"""Classify the following message into exactly ONE category.

Categories:
{route_descriptions}

Rules:
1. Respond with ONLY the category name ({route_names})
2. Choose the BEST fit if the message could belong to multiple categories
3. If unsure, respond with {self.default_route}

Message: {{message}}

Category:"""
    
    def set_default_handler(
        self,
        handler: Callable[[str], str],
        system_prompt: Optional[str] = None,
    ) -> "Router":
        """
        Set the default handler for unclassified queries.
        
        Args:
            handler: Function to handle unclassified queries
            system_prompt: Optional system prompt
            
        Returns:
            Self, for method chaining
        """
        self.routes[self.default_route] = {
            "description": "Anything that doesn't fit other categories",
            "handler": handler,
            "system_prompt": system_prompt,
        }
        return self
    
    def classify(self, message: str) -> str:
        """
        Classify a message into a route.
        
        Args:
            message: The message to classify
            
        Returns:
            The route name
        """
        if not self.classification_prompt_template:
            raise ValueError("No routes configured. Add routes before classifying.")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": self.classification_prompt_template.format(message=message)
            }]
        )
        
        result = response.content[0].text.strip().upper()
        
        # Find matching route
        for route_name in self.routes:
            if route_name in result:
                return route_name
        
        return self.default_route
    
    def route(self, message: str, verbose: bool = False) -> RouteResult:
        """
        Route a message to the appropriate handler.
        
        Args:
            message: The message to route
            verbose: If True, print routing information
            
        Returns:
            RouteResult with route, response, and metadata
        """
        # Classify
        route_name = self.classify(message)
        
        if verbose:
            print(f"🔀 Routed to: {route_name}")
        
        # Get handler
        route_info = self.routes.get(route_name, self.routes.get(self.default_route))
        
        if route_info is None:
            raise ValueError(f"No handler for route '{route_name}' and no default configured")
        
        # Execute handler
        handler = route_info["handler"]
        response = handler(message)
        
        return RouteResult(
            route=route_name,
            response=response,
            original_message=message,
        )
    
    def create_llm_handler(self, system_prompt: str) -> Callable[[str], str]:
        """
        Create a handler function that uses Claude with the given system prompt.
        
        Args:
            system_prompt: The system prompt for this handler
            
        Returns:
            A handler function
        """
        def handler(message: str) -> str:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
        
        return handler


def create_customer_service_router() -> Router:
    """
    Create a pre-configured customer service router.
    
    Returns:
        A Router configured for customer service queries
    """
    router = Router()
    
    # Create handlers using the router's helper method
    billing_handler = router.create_llm_handler(
        "You are a billing specialist. Help with charges, refunds, and payments. "
        "Be empathetic and concise."
    )
    
    technical_handler = router.create_llm_handler(
        "You are a technical support specialist. Help troubleshoot issues. "
        "Provide clear, step-by-step solutions."
    )
    
    account_handler = router.create_llm_handler(
        "You are an account specialist. Help with passwords, settings, and security. "
        "Prioritize security in all responses."
    )
    
    general_handler = router.create_llm_handler(
        "You are a friendly customer service rep. Handle general inquiries. "
        "Be warm and helpful."
    )
    
    # Add routes
    router.add_route(
        "BILLING",
        "Charges, invoices, payments, refunds, pricing questions",
        billing_handler
    )
    router.add_route(
        "TECHNICAL", 
        "Product issues, bugs, errors, how-to questions",
        technical_handler
    )
    router.add_route(
        "ACCOUNT",
        "Password resets, login issues, profile changes, security",
        account_handler
    )
    router.set_default_handler(general_handler)
    
    return router


if __name__ == "__main__":
    # Demo the Router class
    router = create_customer_service_router()
    
    test_queries = [
        "I was charged twice this month",
        "The export button doesn't work",
        "I need to reset my password",
        "What's the best way to reach your team?",
    ]
    
    print("=" * 60)
    print("Router Class Demo")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n💬 Query: {query}")
        result = router.route(query, verbose=True)
        print(f"\n📝 Response:\n{result.response}")
        print("-" * 60)
```

### Router Class Design

The `Router` class offers several advantages:

1. **Method chaining**: `router.add_route(...).add_route(...)` makes configuration readable.
2. **Auto-generated prompts**: The classification prompt is built automatically from route descriptions.
3. **Handler factory**: `create_llm_handler()` makes it easy to create handlers with different system prompts.
4. **Structured results**: `RouteResult` dataclass provides a clean interface.
5. **Extensibility**: Easy to add custom classification or handler logic.

### Using the Router Class

Here's how you might use this in a real application:

```python
# Create a custom router for an e-commerce platform
router = Router()

# Add routes with custom handlers
router.add_route(
    "ORDER_STATUS",
    "Questions about order tracking, delivery, and shipping",
    order_status_handler  # Your custom function
)
router.add_route(
    "RETURNS",
    "Return requests, refund status, exchange policies",
    returns_handler
)
router.add_route(
    "PRODUCT_QUESTIONS",
    "Questions about product features, specifications, availability",
    product_handler
)

# Route a customer query
result = router.route("Where is my package?")
print(f"Route: {result.route}")  # ORDER_STATUS
print(f"Response: {result.response}")
```

## Common Pitfalls

### 1. Overlapping Category Definitions

**Problem:** Categories like "billing issues" and "subscription problems" have significant overlap, leading to inconsistent classification.

**Solution:** Make categories mutually exclusive or define clear priority rules in your prompt:

```python
# Bad: Overlapping
"BILLING: Payment and subscription issues"
"SUBSCRIPTIONS: Subscription management and billing"

# Good: Clear boundaries
"BILLING: One-time charges, refunds, payment methods"
"SUBSCRIPTIONS: Plan changes, renewals, cancellations"
```

### 2. Insufficient Classification Examples

**Problem:** The classifier fails on queries that are phrased differently from your examples.

**Solution:** Include diverse phrasing in your few-shot examples:

```python
# Include variations
"""
Examples:
- "I was charged twice" → BILLING
- "Why is there a double charge?" → BILLING  
- "You billed me two times" → BILLING
- "There's a duplicate transaction" → BILLING
"""
```

### 3. Not Testing Edge Cases

**Problem:** The router works great on clear-cut queries but fails on ambiguous ones.

**Solution:** Specifically include edge cases in your test dataset:

```python
# Edge cases to test
("I paid but my account is still locked", "ACCOUNT"),  # Billing + Account
("The app crashed after I made a payment", "TECHNICAL"),  # Technical + Billing
("I want to upgrade but the page won't load", "TECHNICAL"),  # Billing + Technical
```

### 4. Ignoring Classification Latency

**Problem:** Every query requires a classification API call, adding latency.

**Solution:** Consider rule-based pre-routing for obvious cases:

```python
def hybrid_classify(message: str) -> str:
    """Use rules for obvious cases, LLM for complex ones."""
    message_lower = message.lower()
    
    # Quick rule-based routing for clear cases
    if any(word in message_lower for word in ["password", "login", "log in"]):
        return "ACCOUNT"
    if any(word in message_lower for word in ["refund", "charged", "invoice"]):
        return "BILLING"
    
    # Fall back to LLM for ambiguous cases
    return llm_classify(message)
```

## Practical Exercise

**Task:** Build a routing system for a technical documentation chatbot that routes queries to specialized handlers.

**Requirements:**

1. Create a router with four categories:
   - `INSTALLATION`: Setup, installation, and configuration questions
   - `API_REFERENCE`: Questions about API endpoints, parameters, and responses
   - `TROUBLESHOOTING`: Error messages, bugs, and issues
   - `CONCEPTUAL`: High-level concepts, architecture, and design questions

2. Implement specialized handlers for each category with appropriate system prompts

3. Create a test dataset with at least 5 queries per category (20 total)

4. Run an accuracy test and achieve at least 80% accuracy

5. Handle at least 3 edge cases that could belong to multiple categories

**Hints:**
- Use the `Router` class as your foundation
- Think about what makes each category distinct
- Consider queries that mention multiple topics (e.g., "I'm getting an error during installation")

**Bonus:** Add a confidence threshold—if the classifier is uncertain, route to a human agent.

**Solution:** See `code/exercise_solution.py` for a complete implementation.

## Key Takeaways

- **Classification is the foundation** of routing. Invest time in crafting clear classification prompts with good examples.

- **Specialized handlers** should have domain-specific system prompts optimized for their query type.

- **Test classification accuracy** rigorously. A 90%+ accuracy rate is a good target for production systems.

- **The Router class pattern** provides a reusable, extensible foundation for routing workflows.

- **Edge cases matter**—include ambiguous queries in your test dataset and define clear rules for handling them.

- **Consider hybrid approaches** that combine rule-based pre-routing with LLM classification for better performance.

## What's Next

Now that you can route queries to specialized handlers, what if you need to handle multiple queries simultaneously? In Chapter 20, we'll explore **Parallelization**—running multiple LLM calls at the same time. You'll learn about two powerful patterns: **sectioning** (dividing work into independent subtasks) and **voting** (getting multiple perspectives on the same input). These patterns trade cost for speed and confidence, and they're essential for building responsive, reliable agents.
