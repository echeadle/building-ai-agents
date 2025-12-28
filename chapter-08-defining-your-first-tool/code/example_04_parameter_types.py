"""
Example 04: JSON Schema parameter types reference.

This is a reference showing all common parameter types:
- string (with enum constraints)
- number (with min/max)
- integer
- boolean
- array (of various types)
- object (nested structures)

Chapter 8: Defining Your First Tool
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

# Initialize the Anthropic client
client = anthropic.Anthropic()


# ============================================================================
# STRING PARAMETERS
# ============================================================================

string_examples = {
    # Basic string
    "basic_string": {
        "type": "string",
        "description": "A simple text input"
    },
    
    # String with enum (constrained choices)
    "enum_string": {
        "type": "string",
        "description": "The size of the order",
        "enum": ["small", "medium", "large", "extra-large"]
    },
    
    # String with pattern (regex validation - Claude may not enforce this)
    "email_string": {
        "type": "string",
        "description": "User's email address"
    }
}


# ============================================================================
# NUMBER PARAMETERS
# ============================================================================

number_examples = {
    # Basic number (float or integer)
    "basic_number": {
        "type": "number",
        "description": "Any numeric value"
    },
    
    # Number with constraints
    "constrained_number": {
        "type": "number",
        "description": "A price value in dollars",
        "minimum": 0,
        "maximum": 10000
    },
    
    # Integer only (whole numbers)
    "integer_only": {
        "type": "integer",
        "description": "The quantity to order (must be a whole number)"
    },
    
    # Integer with constraints
    "constrained_integer": {
        "type": "integer",
        "description": "Number of results to return (1-100)",
        "minimum": 1,
        "maximum": 100
    }
}


# ============================================================================
# BOOLEAN PARAMETERS
# ============================================================================

boolean_examples = {
    "basic_boolean": {
        "type": "boolean",
        "description": "Whether to include detailed results"
    },
    
    "flag_boolean": {
        "type": "boolean",
        "description": "If true, the search is case-sensitive. Defaults to false if not provided."
    }
}


# ============================================================================
# ARRAY PARAMETERS
# ============================================================================

array_examples = {
    # Array of strings
    "string_array": {
        "type": "array",
        "items": {"type": "string"},
        "description": "A list of tags to apply"
    },
    
    # Array of numbers
    "number_array": {
        "type": "array",
        "items": {"type": "number"},
        "description": "A list of values to calculate statistics for"
    },
    
    # Array with constraints
    "constrained_array": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of recipient email addresses (1-10)",
        "minItems": 1,
        "maxItems": 10
    },
    
    # Array of objects
    "object_array": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "quantity": {"type": "integer"}
            },
            "required": ["name", "quantity"]
        },
        "description": "List of items to order"
    }
}


# ============================================================================
# OBJECT PARAMETERS
# ============================================================================

object_examples = {
    # Simple nested object
    "simple_object": {
        "type": "object",
        "description": "User profile information",
        "properties": {
            "name": {
                "type": "string",
                "description": "User's full name"
            },
            "age": {
                "type": "integer",
                "description": "User's age"
            }
        },
        "required": ["name"]
    },
    
    # Address object (common pattern)
    "address_object": {
        "type": "object",
        "description": "A mailing address",
        "properties": {
            "street": {
                "type": "string",
                "description": "Street address including number"
            },
            "city": {
                "type": "string",
                "description": "City name"
            },
            "state": {
                "type": "string",
                "description": "State or province code"
            },
            "postal_code": {
                "type": "string",
                "description": "ZIP or postal code"
            },
            "country": {
                "type": "string",
                "description": "Country name or code",
                "enum": ["US", "CA", "UK", "AU"]
            }
        },
        "required": ["street", "city", "postal_code"]
    }
}


# ============================================================================
# COMPLETE EXAMPLE TOOL
# ============================================================================

# A comprehensive tool that demonstrates multiple parameter types
comprehensive_tool = {
    "name": "create_order",
    "description": """Creates a new customer order.
Use this tool when a user wants to place an order for products.
Requires at least a customer name and one item.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_name": {
                "type": "string",
                "description": "The customer's full name"
            },
            "email": {
                "type": "string",
                "description": "Customer's email for order confirmation"
            },
            "items": {
                "type": "array",
                "description": "List of items to order",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 1},
                        "size": {
                            "type": "string",
                            "enum": ["small", "medium", "large"]
                        }
                    },
                    "required": ["product_name", "quantity"]
                }
            },
            "shipping_address": {
                "type": "object",
                "description": "Where to ship the order",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "zip": {"type": "string"}
                },
                "required": ["street", "city", "zip"]
            },
            "express_shipping": {
                "type": "boolean",
                "description": "If true, use express shipping (additional cost)"
            },
            "discount_code": {
                "type": "string",
                "description": "Optional discount code to apply"
            },
            "priority": {
                "type": "integer",
                "description": "Order priority level (1-5, where 5 is highest)",
                "minimum": 1,
                "maximum": 5
            }
        },
        "required": ["customer_name", "items", "shipping_address"]
    }
}


def main():
    """Demonstrate the comprehensive tool with a complex order."""
    
    print("=" * 60)
    print("Example 4: Parameter Types Reference")
    print("=" * 60)
    print()
    
    print("This example demonstrates all common JSON Schema parameter types.")
    print()
    
    # Print the comprehensive tool
    print("Comprehensive Order Tool:")
    print("-" * 40)
    print(json.dumps(comprehensive_tool, indent=2))
    print()
    
    # Test with a complex order request
    test_message = """
    I'd like to place an order. My name is Jane Smith, email jane@example.com.
    I want to order:
    - 2 large t-shirts
    - 1 medium hoodie
    
    Ship to: 123 Main St, Portland, OR 97201
    
    Please use express shipping, and I have a discount code: SAVE20
    """
    
    print("Test message:")
    print(test_message)
    print("-" * 40)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[comprehensive_tool],
        messages=[
            {
                "role": "user",
                "content": test_message
            }
        ]
    )
    
    print("Claude's response:")
    for block in response.content:
        if block.type == "tool_use":
            print(f"Tool: {block.name}")
            print(f"Arguments:")
            print(json.dumps(block.input, indent=2))
        elif block.type == "text" and block.text.strip():
            print(f"Text: {block.text}")
    
    print()
    print("=" * 60)
    print("Notice how Claude extracts structured data from natural language!")
    print("=" * 60)


if __name__ == "__main__":
    main()
