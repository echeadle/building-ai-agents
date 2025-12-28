"""
Exercise Solution: Documentation Chatbot Router

This module implements a routing system for a technical documentation chatbot
with four specialized categories: INSTALLATION, API_REFERENCE, TROUBLESHOOTING,
and CONCEPTUAL.

Chapter 19: Routing - Implementation
"""

import os
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
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
# Router Class (from the chapter)
# ============================================================

@dataclass
class RouteResult:
    """Result from routing a query."""
    route: str
    response: str
    original_message: str


class Router:
    """Flexible routing system with classification and handlers."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.routes: dict[str, dict] = {}
        self._classification_prompt: Optional[str] = None
    
    def add_route(
        self,
        name: str,
        description: str,
        handler: Callable[[str], str],
    ) -> "Router":
        """Add a route to the router."""
        name = name.upper()
        self.routes[name] = {
            "description": description,
            "handler": handler,
        }
        self._classification_prompt = None
        return self
    
    def _build_classification_prompt(self) -> str:
        """Build classification prompt from routes."""
        route_descriptions = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.routes.items()
        )
        route_names = ", ".join(self.routes.keys())
        
        return f"""You are a documentation query classifier. Classify the user's question into exactly ONE category.

Categories:
{route_descriptions}

Rules:
1. Respond with ONLY the category name ({route_names})
2. Choose the BEST fit if the question touches multiple areas
3. For questions about fixing problems, choose TROUBLESHOOTING
4. For questions about understanding concepts, choose CONCEPTUAL

Question: {{message}}

Category:"""
    
    def classify(self, message: str) -> str:
        """Classify a message into a route."""
        if self._classification_prompt is None:
            self._classification_prompt = self._build_classification_prompt()
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": self._classification_prompt.format(message=message)
            }]
        )
        
        result = response.content[0].text.strip().upper()
        
        for route_name in self.routes:
            if route_name in result:
                return route_name
        
        return "CONCEPTUAL"  # Default fallback
    
    def route(self, message: str, verbose: bool = False) -> RouteResult:
        """Route a message to the appropriate handler."""
        route_name = self.classify(message)
        
        if verbose:
            print(f"🔀 Routed to: {route_name}")
        
        handler = self.routes[route_name]["handler"]
        response = handler(message)
        
        return RouteResult(
            route=route_name,
            response=response,
            original_message=message,
        )
    
    def create_llm_handler(self, system_prompt: str) -> Callable[[str], str]:
        """Create an LLM-based handler."""
        def handler(message: str) -> str:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=system_prompt,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
        return handler


# ============================================================
# Documentation Chatbot Router
# ============================================================

def create_docs_router() -> Router:
    """
    Create a router for a technical documentation chatbot.
    
    Categories:
    - INSTALLATION: Setup, installation, and configuration
    - API_REFERENCE: API endpoints, parameters, and responses
    - TROUBLESHOOTING: Error messages, bugs, and issues
    - CONCEPTUAL: High-level concepts, architecture, and design
    """
    router = Router()
    
    # Installation handler
    installation_handler = router.create_llm_handler("""You are a setup and installation specialist for a software documentation system.

You help users with:
- Initial installation and setup
- Configuration options
- Environment variables and settings
- Dependencies and prerequisites
- Upgrade and migration procedures

Guidelines:
- Provide step-by-step instructions
- Mention system requirements when relevant
- Warn about common installation pitfalls
- Include verification steps to confirm success

Always provide clear, numbered steps when giving instructions.""")
    
    # API Reference handler
    api_handler = router.create_llm_handler("""You are an API documentation specialist.

You help users understand:
- API endpoints and their purposes
- Request parameters and body formats
- Response structures and status codes
- Authentication and authorization
- Rate limits and best practices

Guidelines:
- Use clear, technical language
- Provide example requests and responses when helpful
- Mention related endpoints for context
- Note any deprecations or version differences

Format code examples clearly and include both request and response formats.""")
    
    # Troubleshooting handler
    troubleshooting_handler = router.create_llm_handler("""You are a technical troubleshooting specialist.

You help users resolve:
- Error messages and their causes
- Unexpected behavior and bugs
- Performance issues
- Integration problems
- Configuration mistakes

Guidelines:
- Ask clarifying questions if the problem isn't clear
- Provide systematic debugging steps
- Explain why the error occurs, not just how to fix it
- Offer multiple solutions when applicable
- Know when to escalate to support

Start by acknowledging the issue, then provide diagnostic steps.""")
    
    # Conceptual handler
    conceptual_handler = router.create_llm_handler("""You are a technical educator specializing in software architecture and concepts.

You help users understand:
- Core concepts and terminology
- Architecture and design patterns
- How components work together
- Best practices and recommendations
- Trade-offs and decision points

Guidelines:
- Start with simple explanations before diving into details
- Use analogies when they help clarify concepts
- Explain the "why" behind design decisions
- Reference related concepts for deeper learning
- Provide practical examples to illustrate abstract ideas

Make complex topics accessible without oversimplifying.""")
    
    # Add all routes
    router.add_route(
        "INSTALLATION",
        "Setup, installation, configuration, dependencies, upgrades",
        installation_handler
    ).add_route(
        "API_REFERENCE",
        "API endpoints, parameters, responses, authentication, rate limits",
        api_handler
    ).add_route(
        "TROUBLESHOOTING",
        "Error messages, bugs, issues, debugging, performance problems",
        troubleshooting_handler
    ).add_route(
        "CONCEPTUAL",
        "Core concepts, architecture, design patterns, how things work",
        conceptual_handler
    )
    
    return router


# ============================================================
# Test Dataset (5+ per category, 20+ total)
# ============================================================

TEST_DATASET: list[Tuple[str, str]] = [
    # INSTALLATION (5 queries)
    ("How do I install the SDK on Ubuntu?", "INSTALLATION"),
    ("What are the system requirements?", "INSTALLATION"),
    ("How do I configure the database connection?", "INSTALLATION"),
    ("What environment variables do I need to set?", "INSTALLATION"),
    ("How do I upgrade from version 2 to version 3?", "INSTALLATION"),
    
    # API_REFERENCE (5 queries)
    ("What parameters does the /users endpoint accept?", "API_REFERENCE"),
    ("How do I authenticate API requests?", "API_REFERENCE"),
    ("What's the rate limit for the search API?", "API_REFERENCE"),
    ("Can you show me the response format for GET /orders?", "API_REFERENCE"),
    ("What HTTP methods does the /products endpoint support?", "API_REFERENCE"),
    
    # TROUBLESHOOTING (5 queries)
    ("I'm getting a 403 Forbidden error", "TROUBLESHOOTING"),
    ("The application crashes when I submit the form", "TROUBLESHOOTING"),
    ("Why is my query returning empty results?", "TROUBLESHOOTING"),
    ("The API is really slow, taking 10+ seconds", "TROUBLESHOOTING"),
    ("I see 'Connection refused' in the logs", "TROUBLESHOOTING"),
    
    # CONCEPTUAL (5 queries)
    ("What is the difference between sync and async mode?", "CONCEPTUAL"),
    ("How does the caching system work?", "CONCEPTUAL"),
    ("Can you explain the event-driven architecture?", "CONCEPTUAL"),
    ("What's the best practice for handling errors?", "CONCEPTUAL"),
    ("How do microservices communicate in this system?", "CONCEPTUAL"),
    
    # EDGE CASES (5 queries)
    # Installation + Troubleshooting
    ("I'm getting an error during installation", "TROUBLESHOOTING"),
    
    # API + Troubleshooting
    ("The API returns 500 error for POST /users", "TROUBLESHOOTING"),
    
    # Conceptual + API
    ("Why was the API designed with pagination?", "CONCEPTUAL"),
    
    # Installation + Conceptual
    ("Should I use Docker or install directly?", "INSTALLATION"),
    
    # Multiple possible categories
    ("How do I set up authentication for the API?", "INSTALLATION"),
]


def run_accuracy_test(router: Router) -> dict:
    """Run accuracy test on the documentation router."""
    correct = 0
    total = len(TEST_DATASET)
    results = []
    
    print("Running classification tests...\n")
    
    for query, expected in TEST_DATASET:
        predicted = router.classify(query)
        is_correct = predicted == expected
        
        if is_correct:
            correct += 1
        
        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })
        
        status = "✓" if is_correct else "✗"
        if not is_correct:
            print(f"{status} '{query[:50]}...'")
            print(f"   Expected: {expected}, Got: {predicted}\n")
    
    accuracy = correct / total * 100
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def print_accuracy_report(metrics: dict) -> None:
    """Print accuracy report."""
    print("\n" + "=" * 50)
    print("ACCURACY REPORT")
    print("=" * 50)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.1f}%")
    print(f"Correct: {metrics['correct']} / {metrics['total']}")
    
    if metrics["accuracy"] >= 80:
        print("\n✅ Meets the 80% accuracy requirement!")
    else:
        print("\n❌ Below 80% accuracy. Consider improving the prompts.")
    
    # Show misclassifications
    misclassified = [r for r in metrics["results"] if not r["correct"]]
    if misclassified:
        print(f"\nMisclassifications ({len(misclassified)}):")
        for r in misclassified:
            print(f"  • {r['query'][:50]}...")
            print(f"    Expected: {r['expected']}, Got: {r['predicted']}")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Documentation Chatbot Router - Exercise Solution")
    print("=" * 60)
    
    # Create the router
    router = create_docs_router()
    print(f"\nCreated router with routes: {list(router.routes.keys())}")
    
    # Run accuracy test
    print("\n" + "-" * 60)
    print("STEP 1: Testing Classification Accuracy")
    print("-" * 60)
    
    metrics = run_accuracy_test(router)
    print_accuracy_report(metrics)
    
    # Demo the router with real queries
    print("\n" + "-" * 60)
    print("STEP 2: Demo Routing with Full Responses")
    print("-" * 60)
    
    demo_queries = [
        "How do I install the Python SDK?",
        "What parameters does the search endpoint accept?",
        "I'm getting a timeout error when connecting",
        "Can you explain how the queue system works?",
    ]
    
    for query in demo_queries:
        print(f"\n💬 Question: {query}")
        result = router.route(query, verbose=True)
        print(f"\n📝 Response:\n{result.response}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Exercise Complete!")
    print("=" * 60)
    print(f"\nFinal Accuracy: {metrics['accuracy']:.1f}%")
    
    if metrics["accuracy"] >= 80:
        print("✅ Successfully met all exercise requirements!")
    else:
        print("⚠️  Accuracy below 80%. Try improving classification prompts.")
