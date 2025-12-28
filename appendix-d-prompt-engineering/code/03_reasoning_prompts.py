"""
Reasoning Prompts Example

Demonstrates how explicit reasoning improves agent decision-making.

Appendix D: Prompt Engineering for Agents
"""

import os
from dotenv import load_dotenv
import anthropic
import json

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# Without reasoning - direct action
PROMPT_WITHOUT_REASONING = """You are a helpful assistant with access to tools.

Available tools:
- web_search: Search for information
- calculator: Perform calculations
- get_weather: Get weather information

Use tools as needed to answer user queries."""


# With think-then-act pattern
PROMPT_THINK_THEN_ACT = """You are a helpful assistant with access to tools.

For each user request, follow this process:

1. THINK: What does the user want to achieve?
2. PLAN: What tools/steps are needed?
3. ACT: Execute the plan
4. VERIFY: Did it work? Adjust if needed

Format your thinking as:
[THINKING]
User wants: [goal]
I need to: [plan]
[/THINKING]

Then take action.

Available tools:
- web_search: Search for information
- calculator: Perform calculations
- get_weather: Get weather information
"""


# With explicit tool selection reasoning
PROMPT_TOOL_SELECTION = """You are a helpful assistant with access to tools.

When choosing which tool to use, explicitly state your reasoning:

[TOOL SELECTION]
Available tools: [list]
User need: [what they're asking for]
Best tool: [chosen tool]
Reason: [why this tool]
Alternative: [if primary fails]
[/TOOL SELECTION]

Available tools:
- web_search: Search the web for current information
- calculator: Perform arithmetic calculations
- get_weather: Get weather forecasts for locations
"""


# With chain-of-thought for complex tasks
PROMPT_CHAIN_OF_THOUGHT = """You are a helpful assistant with access to tools.

For multi-step tasks, break down your reasoning:

Step 1: Understanding the task
[What is the user asking for? What are the components?]

Step 2: Planning tool usage
[Which tools will I need? In what order?]

Step 3: Executing plan
[Actual tool calls]

Step 4: Validating results
[Did I get what I needed? Any issues?]

Step 5: Presenting answer
[Synthesize the information]

Available tools:
- web_search: Search for information
- calculator: Perform calculations
- get_weather: Get weather information
"""


def test_reasoning_prompt(system_prompt: str, query: str, prompt_name: str) -> None:
    """Test a reasoning prompt with a query."""
    print(f"\n{'='*70}")
    print(f"Testing: {prompt_name}")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    tools = [
        {
            "name": "web_search",
            "description": "Searches the web for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "calculator",
            "description": "Performs arithmetic calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        },
        {
            "name": "get_weather",
            "description": "Gets weather forecast for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "date": {"type": "string", "description": "Date (YYYY-MM-DD)"}
                },
                "required": ["location"]
            }
        }
    ]
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        print("Agent Response:")
        print("-" * 70)
        
        for block in response.content:
            if block.type == "text":
                # Look for reasoning sections
                text = block.text
                if "[THINKING]" in text or "[TOOL SELECTION]" in text or "Step 1:" in text:
                    print("🧠 REASONING VISIBLE:")
                    print(text)
                else:
                    print("Response (no visible reasoning):")
                    print(text[:300])
            elif block.type == "tool_use":
                print(f"\n🔧 Tool Used: {block.name}")
                print(f"   Input: {json.dumps(block.input, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")


def demonstrate_reasoning_impact() -> None:
    """Show how explicit reasoning affects agent behavior."""
    
    # Complex query that benefits from reasoning
    query = """I'm planning a trip to Seattle next week. Can you tell me 
what the weather will be like, and based on that, calculate how much 
I should budget for rain gear if I need an umbrella ($15), rain jacket ($45), 
and waterproof shoes ($60)?"""
    
    print("\n" + "="*70)
    print("DEMONSTRATING REASONING PROMPT IMPACT")
    print("="*70)
    print("\nThis query requires:")
    print("1. Getting weather information")
    print("2. Determining what's needed based on weather")
    print("3. Calculating total cost")
    print()
    
    # Test without reasoning
    test_reasoning_prompt(
        PROMPT_WITHOUT_REASONING,
        query,
        "Without Explicit Reasoning"
    )
    
    # Test with think-then-act
    test_reasoning_prompt(
        PROMPT_THINK_THEN_ACT,
        query,
        "With Think-Then-Act Pattern"
    )
    
    # Test with tool selection reasoning
    test_reasoning_prompt(
        PROMPT_TOOL_SELECTION,
        query,
        "With Tool Selection Reasoning"
    )
    
    # Test with chain-of-thought
    test_reasoning_prompt(
        PROMPT_CHAIN_OF_THOUGHT,
        query,
        "With Chain-of-Thought"
    )


def demonstrate_debugging_benefit() -> None:
    """Show how reasoning helps with debugging."""
    
    print("\n" + "="*70)
    print("REASONING FOR DEBUGGING")
    print("="*70)
    
    print("""
Why explicit reasoning helps debugging:

1. WITHOUT REASONING:
   Agent: [calls weather tool] [calls calculator] "You'll need $120 for rain gear"
   Problem: WHY did it decide to buy all three items?
   
2. WITH REASONING:
   Agent: [THINKING]
          User wants weather and budget calculation
          Need to: 1) Get weather, 2) Decide what's needed, 3) Calculate
          [/THINKING]
          [calls weather] "Forecast: Heavy rain"
          [THINKING] 
          Heavy rain means all rain gear needed
          [/THINKING]
          [calls calculator for total]
          "You'll need $120 for rain gear (umbrella + jacket + shoes)"
   
   Benefit: You can SEE the decision-making process

3. DEBUGGING TIPS:
   - Reasoning reveals the agent's "thought process"
   - Makes it easy to spot logical errors
   - Shows which information influenced decisions
   - Helps identify missing context or instructions

4. TRADE-OFFS:
   + Easier to debug
   + More transparent behavior
   + Better for complex tasks
   - More tokens used
   - Slightly slower
   - May expose internal reasoning to users (can filter it out)
""")


def demonstrate_reasoning_templates() -> None:
    """Show practical reasoning templates."""
    
    print("\n" + "="*70)
    print("PRACTICAL REASONING TEMPLATES")
    print("="*70)
    
    templates = {
        "Simple Decision": """
Before taking action:
[DECISION]
Options: [A, B, C]
Chosen: [X]
Reason: [Why X is best]
[/DECISION]
""",
        
        "Multi-Step Task": """
[PLAN]
1. [First step and why]
2. [Second step and why]
3. [Third step and why]
[/PLAN]

[EXECUTION]
[Actual tool calls]
[/EXECUTION]

[VALIDATION]
Did it work? [Check results]
[/VALIDATION]
""",
        
        "Error Recovery": """
[ATTEMPT]
Tried: [what was tried]
Result: [success/failure]
[/ATTEMPT]

[RECOVERY]
Issue: [what went wrong]
Alternative: [backup plan]
[/RECOVERY]
""",
        
        "Confidence Assessment": """
[CONFIDENCE]
My answer: [the answer]
Confidence: [high/medium/low]
Reasoning: [why this confidence level]
Gaps: [what I'm uncertain about]
[/CONFIDENCE]
"""
    }
    
    for name, template in templates.items():
        print(f"\n{name}:")
        print("-" * 70)
        print(template)


if __name__ == "__main__":
    print("Reasoning Prompts Example")
    print("=" * 70)
    print("This example shows how explicit reasoning improves agents.")
    print()
    
    # Demonstrate impact on behavior
    demonstrate_reasoning_impact()
    
    # Explain debugging benefits
    demonstrate_debugging_benefit()
    
    # Show practical templates
    demonstrate_reasoning_templates()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Reasoning makes agent behavior OBSERVABLE
   - You can see why decisions were made
   - Easier to debug unexpected behavior
   - Helps identify prompt improvements

2. Reasoning improves RELIABILITY
   - Forces agent to "think through" problems
   - Reduces impulsive/incorrect actions
   - Better handling of complex tasks

3. WHEN TO USE REASONING:
   - Complex, multi-step tasks
   - When debugging agent behavior
   - When transparency is important
   - When reliability > speed

4. WHEN NOT TO USE REASONING:
   - Simple, single-action tasks
   - When response time is critical
   - When token budget is tight
   - When reasoning would confuse users

Start with reasoning during development, optimize later if needed.
""")
