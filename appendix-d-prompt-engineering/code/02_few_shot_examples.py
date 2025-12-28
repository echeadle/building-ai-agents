"""
Few-Shot Examples for Tools

Demonstrates how providing examples dramatically improves tool usage.

Appendix D: Prompt Engineering for Agents
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# Without examples - just tool definition
PROMPT_WITHOUT_EXAMPLES = """You are a helpful assistant with access to a calculator tool.

The calculator can perform: add, subtract, multiply, divide

Use it when needed."""


# With examples - showing good usage patterns
PROMPT_WITH_EXAMPLES = """You are a helpful assistant with access to a calculator tool.

Here are examples of when to use it:

Example 1:
User: "What's 15% of 240?"
Thought: This is a calculation, I should use the calculator.
Action: multiply(240, 0.15)
Result: 36
Response: "15% of 240 is 36."

Example 2:
User: "If I save $500/month for 3 years, how much will I have?"
Thought: I need to calculate total savings over time.
Action: multiply(500, 36)
Result: 18000
Response: "You'll save $18,000 over 3 years (500 × 36 months)."

Example 3:
User: "What's 2 + 2?"
Thought: This is very simple, I can answer directly.
Response: "4"

Example 4:
User: "I bought 3 items at $12.99 each and paid with a $50 bill. What's my change?"
Thought: This needs multiple calculations.
Step 1: multiply(12.99, 3) = 38.97
Step 2: subtract(50, 38.97) = 11.03
Response: "Your change is $11.03."

Guidelines:
- Use the calculator for non-trivial arithmetic
- For very simple math (2+2), answer directly
- For multi-step problems, break it down
- Always show your work in the response
"""


# With negative examples - showing what NOT to do
PROMPT_WITH_NEGATIVE_EXAMPLES = """You are a helpful assistant with access to a calculator tool.

Good examples (DO THIS):

User: "What's 23 × 47?"
✓ Action: multiply(23, 47)
✓ Why: Multi-digit multiplication is non-trivial

User: "Calculate 15% tip on $67.50"
✓ Action: multiply(67.50, 0.15)
✓ Why: Percentage calculations are clearer with calculator

Bad examples (DON'T DO THIS):

User: "What's 10 + 5?"
✗ Action: add(10, 5)
✗ Why: Too simple, just answer "15"

User: "Roughly how much is 200 × 3?"
✗ Action: multiply(200, 3)
✗ Why: User asked for "roughly", answer "about 600" directly

User: "Is 7 bigger than 5?"
✗ Action: [trying to use calculator for comparison]
✗ Why: This isn't a calculation, just answer "Yes"

Rule: Use the calculator when precision matters and the math isn't trivial.
"""


def test_few_shot_learning(system_prompt: str, test_queries: list[str], prompt_name: str) -> None:
    """Test how few-shot examples affect tool usage."""
    print(f"\n{'='*70}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*70}")
    
    # Define calculator tool
    tools = [
        {
            "name": "calculator",
            "description": "Performs arithmetic calculations: add, subtract, multiply, divide",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                tools=tools,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            
            # Check if tool was used
            tool_used = False
            for block in response.content:
                if block.type == "tool_use":
                    tool_used = True
                    print(f"✓ Used calculator: {block.name}({block.input})")
                elif block.type == "text":
                    print(f"Response: {block.text[:150]}")
            
            if not tool_used:
                print("✓ Answered directly without tool")
                
        except Exception as e:
            print(f"Error: {e}")


def demonstrate_few_shot_impact() -> None:
    """Show how few-shot examples improve tool selection."""
    
    # Test queries that require judgment about tool usage
    test_queries = [
        "What's 2 + 2?",  # Should NOT use tool
        "What's 17% of 892?",  # SHOULD use tool
        "Calculate 47 × 83",  # SHOULD use tool
        "Is 100 more than 50?",  # Should NOT use tool
        "What's the total cost of 7 items at $23.49 each?",  # SHOULD use tool
    ]
    
    print("\n" + "="*70)
    print("DEMONSTRATING FEW-SHOT LEARNING IMPACT")
    print("="*70)
    print("\nWe'll test the same queries with different prompts:")
    
    # Test without examples
    test_few_shot_learning(
        PROMPT_WITHOUT_EXAMPLES,
        test_queries,
        "Without Examples (Just Tool Definition)"
    )
    
    # Test with examples
    test_few_shot_learning(
        PROMPT_WITH_EXAMPLES,
        test_queries,
        "With Few-Shot Examples"
    )
    
    # Test with negative examples
    test_few_shot_learning(
        PROMPT_WITH_NEGATIVE_EXAMPLES,
        test_queries,
        "With Positive and Negative Examples"
    )


def demonstrate_complex_tool_examples() -> None:
    """Show how examples help with tools that have complex parameters."""
    
    print("\n" + "="*70)
    print("COMPLEX TOOL WITH STRUCTURED PARAMETERS")
    print("="*70)
    
    # Tool with multiple optional parameters
    search_tool = {
        "name": "web_search",
        "description": """Searches the web with optional filters.
        
Examples of effective use:

Example 1 - Recent news:
User: "What happened with tech layoffs this week?"
Tool: web_search(query="tech layoffs", date_range="week")

Example 2 - Academic research:
User: "Find research papers on neural networks"
Tool: web_search(query="neural networks", site="arxiv.org")

Example 3 - Comparative search:
User: "Compare prices for laptops"
Tool: web_search(query="laptop prices 2024")
Note: Include year for price-sensitive searches""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search terms"
                },
                "date_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Optional: restrict to recent results"
                },
                "site": {
                    "type": "string",
                    "description": "Optional: restrict to specific domain"
                }
            },
            "required": ["query"]
        }
    }
    
    test_queries = [
        "Find recent news about climate change",
        "Look for academic papers on quantum computing",
        "What are current laptop prices?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                tools=[search_tool],
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Tool call: {block.name}")
                    print(f"Parameters: {block.input}")
                    
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("Few-Shot Learning for Tool Usage")
    print("=" * 70)
    print("This example shows how providing examples improves tool selection.")
    print()
    
    # Demonstrate impact on simple tool
    demonstrate_few_shot_impact()
    
    # Demonstrate with complex tool parameters
    demonstrate_complex_tool_examples()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. WITHOUT examples:
   - Agent may use tools for trivial tasks
   - Or may not use tools when it should
   - Inconsistent behavior

2. WITH positive examples:
   - Better judgment about when to use tools
   - Correct parameter construction
   - More consistent behavior

3. WITH positive AND negative examples:
   - Best judgment calls
   - Clear boundaries on tool usage
   - Most reliable behavior

Bottom line: Few-shot examples are one of the highest-impact
improvements you can make to prompt engineering.
""")
