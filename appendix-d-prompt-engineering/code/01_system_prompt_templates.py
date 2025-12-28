"""
System Prompt Templates Example

Demonstrates how different system prompt templates affect agent behavior.

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


# Template 1: Minimal (often insufficient)
MINIMAL_PROMPT = """You are a helpful assistant."""

# Template 2: Tool-aware with guidelines
TOOL_AWARE_PROMPT = """You are a helpful assistant with access to tools that allow you to take actions.

Your capabilities:
- You can search for information using the web_search tool
- You can perform calculations using the calculator tool

Guidelines:
1. Always use tools when you need information you don't have
2. If a tool call fails, try an alternative approach
3. Provide clear, concise responses

Important constraints:
- Never make up information—use tools to find facts
- If you cannot complete a task, explain why clearly
"""

# Template 3: Task-specific with explicit process
RESEARCH_PROMPT = """You are a research assistant.

Your primary task: Given a question, search for information and provide a well-sourced answer.

Available tools:
- web_search(query: str) -> list[SearchResult]
  Use this to find relevant sources. Try multiple searches if needed.

Research process:
1. Search for relevant sources (2-3 searches usually sufficient)
2. Review the search results
3. Synthesize findings into a clear answer
4. Cite sources used

Termination conditions:
- You have found sufficient information to answer confidently
- Additional searches are yielding redundant information

Quality standards:
- Prefer authoritative sources (.edu, .gov, established publications)
- Cross-reference facts across multiple sources
- Note any uncertainties
"""


def test_prompt_template(system_prompt: str, user_query: str, prompt_name: str) -> None:
    """Test a system prompt template with a query."""
    print(f"\n{'='*70}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*70}")
    print(f"\nSystem Prompt Length: {len(system_prompt)} characters")
    print(f"User Query: {user_query}")
    print(f"\n{'-'*70}")
    
    # Define a simple search tool
    tools = [
        {
            "name": "web_search",
            "description": "Searches the web for information. Returns a list of results.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculator",
            "description": "Performs basic arithmetic calculations.",
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
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=[
                {"role": "user", "content": user_query}
            ]
        )
        
        # Analyze response
        print("Agent Response:")
        for block in response.content:
            if block.type == "text":
                print(f"  Text: {block.text[:200]}...")
            elif block.type == "tool_use":
                print(f"  Tool Used: {block.name}")
                print(f"  Tool Input: {block.input}")
        
        print(f"\nStop Reason: {response.stop_reason}")
        
    except Exception as e:
        print(f"Error: {e}")


def compare_templates() -> None:
    """Compare how different templates handle the same query."""
    
    # Test query that requires tool use
    query = "What is the average temperature in Antarctica, and what's 32 degrees Fahrenheit in Celsius?"
    
    print("\n" + "="*70)
    print("COMPARING SYSTEM PROMPT TEMPLATES")
    print("="*70)
    print(f"\nWe'll test the same query with three different system prompts:")
    print(f"Query: {query}")
    
    # Test each template
    test_prompt_template(MINIMAL_PROMPT, query, "Minimal Prompt")
    test_prompt_template(TOOL_AWARE_PROMPT, query, "Tool-Aware Prompt")
    test_prompt_template(RESEARCH_PROMPT, query, "Task-Specific Research Prompt")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
What we typically observe:

1. Minimal Prompt:
   - May not use tools effectively
   - Might make up information or say "I don't know"
   - No clear decision-making process

2. Tool-Aware Prompt:
   - Better tool usage
   - Follows guidelines about when to use tools
   - More reliable behavior

3. Task-Specific Prompt:
   - Most structured behavior
   - Clear process adherence
   - Best for consistent, predictable results

Key Takeaway: More explicit instructions = more predictable behavior
""")


if __name__ == "__main__":
    print("System Prompt Templates Example")
    print("=" * 70)
    print("This example demonstrates how system prompt design affects agent behavior.")
    print()
    
    compare_templates()
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
When designing system prompts:

1. Start with a clear role definition
2. List specific capabilities (tools available)
3. Provide behavioral guidelines
4. Set explicit constraints
5. Include a process/workflow if applicable
6. Define termination conditions for loops
7. Specify output format requirements

Remember: The agent only knows what you tell it!
""")
