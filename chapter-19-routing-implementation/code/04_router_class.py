"""
Reusable Router class for building routing workflows.

This module provides a flexible, extensible Router class that can be
adapted for any application that needs to route queries to specialized
handlers based on classification.

Chapter 19: Routing - Implementation
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Optional
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class RouteResult:
    """
    Result from routing a query.
    
    Attributes:
        route: The name of the route that was selected
        response: The handler's response
        original_message: The original input message
        metadata: Optional additional metadata
    """
    route: str
    response: str
    original_message: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RouteConfig:
    """
    Configuration for a single route.
    
    Attributes:
        name: Route name (e.g., "BILLING")
        description: What this route handles (used in classification prompt)
        handler: Function that processes queries for this route
        system_prompt: Optional system prompt for LLM handlers
    """
    name: str
    description: str
    handler: Callable[[str], str]
    system_prompt: Optional[str] = None


class Router:
    """
    A flexible routing system that classifies inputs and directs them
    to specialized handlers.
    
    The Router class provides:
    - Dynamic route registration with method chaining
    - Auto-generated classification prompts
    - Handler factory for creating LLM-based handlers
    - Configurable default/fallback routes
    
    Example usage:
        router = Router()
        
        # Add routes with custom handlers
        router.add_route(
            "BILLING",
            "Charges, refunds, payment issues",
            billing_handler
        )
        
        # Or use the built-in handler factory
        router.add_route(
            "TECHNICAL",
            "Product issues and bugs", 
            router.create_llm_handler("You are a tech support specialist...")
        )
        
        # Set a default handler for unclassified queries
        router.set_default_handler(general_handler)
        
        # Route a query
        result = router.route("Why was I charged twice?")
        print(f"Routed to: {result.route}")
        print(f"Response: {result.response}")
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        default_route: str = "DEFAULT",
        classification_max_tokens: int = 20,
        handler_max_tokens: int = 500,
    ):
        """
        Initialize the router.
        
        Args:
            model: The Claude model to use for classification and handlers
            default_route: Name of the default route for unclassified queries
            classification_max_tokens: Max tokens for classification response
            handler_max_tokens: Max tokens for handler responses
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.default_route = default_route
        self.classification_max_tokens = classification_max_tokens
        self.handler_max_tokens = handler_max_tokens
        
        # Route storage
        self.routes: dict[str, RouteConfig] = {}
        
        # Classification prompt (built dynamically)
        self._classification_prompt: Optional[str] = None
    
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
            name: Route name (e.g., "BILLING"). Will be uppercased.
            description: What this route handles (used in classification prompt)
            handler: Function that handles queries for this route
            system_prompt: Optional system prompt (for documentation)
            
        Returns:
            Self, for method chaining
            
        Example:
            router.add_route(
                "BILLING",
                "Payment and refund questions",
                billing_handler
            ).add_route(
                "TECHNICAL",
                "Product bugs and issues",
                technical_handler
            )
        """
        name = name.upper()
        self.routes[name] = RouteConfig(
            name=name,
            description=description,
            handler=handler,
            system_prompt=system_prompt,
        )
        
        # Invalidate cached classification prompt
        self._classification_prompt = None
        
        return self
    
    def set_default_handler(
        self,
        handler: Callable[[str], str],
        description: str = "Anything that doesn't fit other categories",
        system_prompt: Optional[str] = None,
    ) -> "Router":
        """
        Set the default handler for unclassified queries.
        
        Args:
            handler: Function to handle unclassified queries
            description: Description for the default route
            system_prompt: Optional system prompt
            
        Returns:
            Self, for method chaining
        """
        return self.add_route(
            self.default_route,
            description,
            handler,
            system_prompt,
        )
    
    def remove_route(self, name: str) -> "Router":
        """
        Remove a route from the router.
        
        Args:
            name: Route name to remove
            
        Returns:
            Self, for method chaining
        """
        name = name.upper()
        if name in self.routes:
            del self.routes[name]
            self._classification_prompt = None
        return self
    
    def _build_classification_prompt(self) -> str:
        """Build the classification prompt from registered routes."""
        if not self.routes:
            raise ValueError("No routes configured. Add routes before classifying.")
        
        # Build category descriptions
        route_descriptions = "\n".join(
            f"- {config.name}: {config.description}"
            for config in self.routes.values()
        )
        
        # Build list of valid route names
        route_names = ", ".join(self.routes.keys())
        
        return f"""Classify the following message into exactly ONE category.

Categories:
{route_descriptions}

Rules:
1. Respond with ONLY the category name ({route_names})
2. Choose the BEST fit if the message could belong to multiple categories
3. If truly ambiguous, respond with {self.default_route}

Message: {{message}}

Category:"""
    
    @property
    def classification_prompt(self) -> str:
        """Get the classification prompt, building it if necessary."""
        if self._classification_prompt is None:
            self._classification_prompt = self._build_classification_prompt()
        return self._classification_prompt
    
    def classify(self, message: str) -> str:
        """
        Classify a message into a route.
        
        Args:
            message: The message to classify
            
        Returns:
            The route name
            
        Raises:
            ValueError: If no routes are configured
        """
        prompt = self.classification_prompt.format(message=message)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.classification_max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.content[0].text.strip().upper()
        
        # Find matching route
        for route_name in self.routes:
            if route_name in result:
                return route_name
        
        # Fall back to default
        if self.default_route in self.routes:
            return self.default_route
        
        # If no default, return first route
        return next(iter(self.routes))
    
    def route(
        self, 
        message: str, 
        verbose: bool = False
    ) -> RouteResult:
        """
        Route a message to the appropriate handler.
        
        Args:
            message: The message to route
            verbose: If True, print routing information
            
        Returns:
            RouteResult with route name, response, and metadata
            
        Raises:
            ValueError: If no routes are configured
        """
        # Step 1: Classify
        route_name = self.classify(message)
        
        if verbose:
            print(f"🔀 Routed to: {route_name}")
        
        # Step 2: Get handler
        route_config = self.routes.get(route_name)
        if route_config is None:
            raise ValueError(f"No handler configured for route: {route_name}")
        
        # Step 3: Execute handler
        response = route_config.handler(message)
        
        return RouteResult(
            route=route_name,
            response=response,
            original_message=message,
            metadata={"model": self.model},
        )
    
    def route_batch(
        self, 
        messages: list[str], 
        verbose: bool = False
    ) -> list[RouteResult]:
        """
        Route multiple messages.
        
        Args:
            messages: List of messages to route
            verbose: If True, print routing information
            
        Returns:
            List of RouteResults
        """
        return [self.route(msg, verbose) for msg in messages]
    
    def create_llm_handler(
        self, 
        system_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Callable[[str], str]:
        """
        Create a handler function that uses Claude with the given system prompt.
        
        This is a convenience method for creating handlers without defining
        separate functions.
        
        Args:
            system_prompt: The system prompt for this handler
            max_tokens: Max tokens for response (uses default if not specified)
            
        Returns:
            A handler function
            
        Example:
            handler = router.create_llm_handler(
                "You are a billing specialist. Help with payment issues."
            )
            router.add_route("BILLING", "Payment questions", handler)
        """
        tokens = max_tokens or self.handler_max_tokens
        
        def handler(message: str) -> str:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text
        
        return handler
    
    def get_routes(self) -> list[str]:
        """Get list of configured route names."""
        return list(self.routes.keys())
    
    def get_route_info(self, name: str) -> Optional[RouteConfig]:
        """Get configuration for a specific route."""
        return self.routes.get(name.upper())
    
    def __repr__(self) -> str:
        routes = ", ".join(self.routes.keys())
        return f"Router(routes=[{routes}], default={self.default_route})"


# ============================================================
# Factory Function for Common Use Case
# ============================================================

def create_customer_service_router() -> Router:
    """
    Create a pre-configured customer service router.
    
    This is a convenience function that creates a Router with
    standard customer service routes already configured.
    
    Returns:
        A Router configured for customer service queries
        
    Example:
        router = create_customer_service_router()
        result = router.route("Why was I charged twice?")
    """
    router = Router()
    
    # Create handlers using the router's helper method
    billing_handler = router.create_llm_handler(
        "You are a billing specialist. Help with charges, refunds, and payments. "
        "Be empathetic and concise. Offer to escalate complex issues."
    )
    
    technical_handler = router.create_llm_handler(
        "You are a technical support specialist. Help troubleshoot product issues. "
        "Provide clear, step-by-step solutions. Ask clarifying questions if needed."
    )
    
    account_handler = router.create_llm_handler(
        "You are an account specialist. Help with passwords, settings, and security. "
        "Prioritize security. Never ask for full passwords. Recommend 2FA."
    )
    
    general_handler = router.create_llm_handler(
        "You are a friendly customer service rep. Handle general inquiries and feedback. "
        "Be warm and helpful. Acknowledge complaints before addressing them."
    )
    
    # Add routes using method chaining
    router.add_route(
        "BILLING",
        "Charges, invoices, payments, refunds, pricing questions",
        billing_handler
    ).add_route(
        "TECHNICAL", 
        "Product issues, bugs, errors, how-to questions, setup",
        technical_handler
    ).add_route(
        "ACCOUNT",
        "Password resets, login issues, profile changes, security",
        account_handler
    ).set_default_handler(
        general_handler,
        "General inquiries, feedback, compliments, complaints"
    )
    
    return router


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Router Class Demo")
    print("=" * 70)
    
    # Create the router using the factory function
    router = create_customer_service_router()
    print(f"\nCreated: {router}")
    print(f"Available routes: {router.get_routes()}")
    
    # Test queries
    test_queries = [
        "I was charged twice this month for my subscription.",
        "The export button doesn't work on the dashboard.",
        "I need to reset my password but didn't receive the email.",
        "What's the best way to reach your support team?",
    ]
    
    print("\n" + "-" * 70)
    
    for query in test_queries:
        print(f"\n💬 Query: {query}")
        result = router.route(query, verbose=True)
        print(f"\n📝 Response:\n{result.response}")
        print("-" * 70)
    
    # Demonstrate custom router creation
    print("\n" + "=" * 70)
    print("Custom Router Demo (E-commerce)")
    print("=" * 70)
    
    # Create a custom e-commerce router
    ecommerce_router = Router()
    
    ecommerce_router.add_route(
        "ORDER_STATUS",
        "Order tracking, delivery updates, shipping questions",
        ecommerce_router.create_llm_handler(
            "You are an order specialist. Help track orders and shipping."
        )
    ).add_route(
        "RETURNS",
        "Return requests, refunds, exchanges, return policies",
        ecommerce_router.create_llm_handler(
            "You are a returns specialist. Help process returns and exchanges."
        )
    ).add_route(
        "PRODUCT",
        "Product questions, specifications, availability, recommendations",
        ecommerce_router.create_llm_handler(
            "You are a product specialist. Help with product information."
        )
    ).set_default_handler(
        ecommerce_router.create_llm_handler(
            "You are a helpful e-commerce assistant."
        )
    )
    
    print(f"\nCreated: {ecommerce_router}")
    
    ecommerce_query = "Where is my package? I ordered 3 days ago."
    print(f"\n💬 Query: {ecommerce_query}")
    result = ecommerce_router.route(ecommerce_query, verbose=True)
    print(f"\n📝 Response:\n{result.response}")
