"""
Exercise Solution: Inventory Management Agent

This agent can answer questions about a fictional inventory system using
three tools: check_stock, get_price, and calculate_total.

Chapter 27: The Agentic Loop
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


# Simulated inventory database
INVENTORY = {
    "widget": {"stock": 150, "price": 12.99},
    "gadget": {"stock": 75, "price": 49.99},
    "gizmo": {"stock": 200, "price": 7.50},
    "doohickey": {"stock": 30, "price": 125.00},
    "thingamajig": {"stock": 500, "price": 3.25},
    "whatchamacallit": {"stock": 25, "price": 299.99},
}


# Tool definitions
INVENTORY_TOOLS = [
    {
        "name": "check_stock",
        "description": "Check how many units of an item are currently in stock. Returns the quantity available in the inventory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item to check (e.g., 'widget', 'gadget')"
                }
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "get_price",
        "description": "Get the price per unit for an item. Returns the price in dollars.",
        "input_schema": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item to get the price for"
                }
            },
            "required": ["item_name"]
        }
    },
    {
        "name": "calculate_total",
        "description": "Calculate the total cost for a given quantity and unit price. Use this after getting the price to calculate order totals or inventory values.",
        "input_schema": {
            "type": "object",
            "properties": {
                "quantity": {
                    "type": "number",
                    "description": "The number of units"
                },
                "unit_price": {
                    "type": "number",
                    "description": "The price per unit in dollars"
                }
            },
            "required": ["quantity", "unit_price"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute an inventory tool and return the result.
    
    Args:
        tool_name: Name of the tool to execute
        tool_input: Input parameters for the tool
        
    Returns:
        String result of the tool execution
    """
    if tool_name == "check_stock":
        item_name = tool_input["item_name"].lower()
        if item_name in INVENTORY:
            quantity = INVENTORY[item_name]["stock"]
            return f"{quantity} units in stock"
        else:
            available_items = ", ".join(INVENTORY.keys())
            return f"Item '{item_name}' not found. Available items: {available_items}"
    
    elif tool_name == "get_price":
        item_name = tool_input["item_name"].lower()
        if item_name in INVENTORY:
            price = INVENTORY[item_name]["price"]
            return f"${price:.2f} per unit"
        else:
            available_items = ", ".join(INVENTORY.keys())
            return f"Item '{item_name}' not found. Available items: {available_items}"
    
    elif tool_name == "calculate_total":
        quantity = tool_input["quantity"]
        unit_price = tool_input["unit_price"]
        total = quantity * unit_price
        return f"${total:.2f}"
    
    else:
        return f"Error: Unknown tool '{tool_name}'"


def log_phase(phase: str, message: str) -> None:
    """Log a phase of the agentic loop."""
    print(f"  [{phase}] {message}")


def run_inventory_agent(
    user_message: str,
    max_iterations: int = 10,
    max_tool_calls: int = 15
) -> str:
    """
    Run the inventory management agent.
    
    Args:
        user_message: The user's question about inventory
        max_iterations: Maximum loop iterations
        max_tool_calls: Maximum tool calls allowed
        
    Returns:
        The agent's response
    """
    messages = [{"role": "user", "content": user_message}]
    tool_calls_made = 0
    
    print(f"\n{'═' * 60}")
    print(f"  QUESTION: {user_message}")
    print(f"{'═' * 60}")
    
    # System prompt to guide the agent
    system_prompt = """You are an inventory management assistant. You help users check stock levels, 
prices, and calculate inventory values. Use the available tools to answer questions accurately.

When asked about inventory value, you need to:
1. First check the stock quantity
2. Then get the price per unit  
3. Finally calculate the total value

Always provide clear, helpful responses with the actual numbers."""
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # PERCEIVE
        log_phase("PERCEIVE", f"{len(messages)} messages in context")
        
        # THINK
        log_phase("THINK", "Sending to Claude...")
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            tools=INVENTORY_TOOLS,
            messages=messages
        )
        
        log_phase("THINK", f"Stop reason: {response.stop_reason}")
        
        # Check for completion
        if response.stop_reason == "end_turn":
            log_phase("ACT", "Task complete!")
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""
        
        # ACT: Handle tool calls
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_made += 1
                    
                    if tool_calls_made > max_tool_calls:
                        return "I've made too many tool calls. Please try a simpler question."
                    
                    log_phase("ACT", f"Tool: {block.name}({json.dumps(block.input)})")
                    
                    result = execute_tool(block.name, block.input)
                    log_phase("ACT", f"Result: {result}")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add results for next iteration
            messages.append({"role": "user", "content": tool_results})
    
    return "I couldn't complete the task within the allowed iterations."


def main():
    """Run demonstration queries."""
    
    print("\n" + "▓" * 60)
    print("  INVENTORY MANAGEMENT AGENT")
    print("▓" * 60)
    
    # Available items for reference
    print("\nAvailable inventory items:")
    for item, data in INVENTORY.items():
        print(f"  • {item}: {data['stock']} units @ ${data['price']:.2f}")
    
    # Test queries
    test_queries = [
        "How many widgets do we have in stock?",
        "What's the total value of our gadget inventory?",
        "Do we have enough gizmos to fulfill an order of 50?",
        "What's more valuable: our widget inventory or our thingamajig inventory?",
        "If I need to buy 10 doohickeys, how much will it cost?",
    ]
    
    for query in test_queries:
        response = run_inventory_agent(query)
        print(f"\n{'─' * 60}")
        print(f"ANSWER: {response}")
        print(f"{'─' * 60}")
        
        # Pause between queries for readability
        input("\nPress Enter for next question...")


if __name__ == "__main__":
    main()
