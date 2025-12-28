"""
Workflow vs Agent Comparison

This module demonstrates the fundamental difference between workflows
and agents: who controls the execution flow.

Chapter 26: From Workflows to Agents
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# =============================================================================
# EXAMPLE 1: WORKFLOW APPROACH
# The developer controls the flow - steps are predetermined
# =============================================================================

def workflow_research(topic: str) -> str:
    """
    Research a topic using a WORKFLOW approach.
    
    Notice: The developer defines exactly what steps happen and in what order.
    The flow is predetermined and fixed.
    """
    print("=" * 60)
    print("WORKFLOW APPROACH: Developer controls the flow")
    print("=" * 60)
    
    # Step 1: ALWAYS generate initial research questions
    print("\nStep 1: Generating research questions (ALWAYS happens)...")
    questions_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"Generate 3 research questions about: {topic}. Be concise."
        }]
    )
    questions = questions_response.content[0].text
    print(f"  Generated questions:\n{questions[:200]}...")
    
    # Step 2: ALWAYS expand on each question (fixed behavior)
    print("\nStep 2: Expanding on questions (ALWAYS happens)...")
    expansion_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"Briefly expand on these questions:\n{questions}"
        }]
    )
    expansion = expansion_response.content[0].text
    print(f"  Expanded content:\n{expansion[:200]}...")
    
    # Step 3: ALWAYS synthesize into a summary
    print("\nStep 3: Synthesizing summary (ALWAYS happens)...")
    summary_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Summarize this research in 2-3 sentences:\n{expansion}"
        }]
    )
    summary = summary_response.content[0].text
    
    print("\n✓ Workflow complete: 3 steps, always in the same order")
    return summary


# =============================================================================
# EXAMPLE 2: AGENT APPROACH
# The LLM controls the flow - decides what to do next based on the situation
# =============================================================================

# Define tools the agent can use
research_tools = [
    {
        "name": "search_web",
        "description": "Search the web for information on a topic. Use this when you need current or factual information.",
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
        "name": "take_notes",
        "description": "Save important information to your notes. Use this to track key findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to save"
                }
            },
            "required": ["note"]
        }
    },
    {
        "name": "ask_clarification",
        "description": "Ask the user for clarification if the request is unclear.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarifying question to ask"
                }
            },
            "required": ["question"]
        }
    }
]


def simulate_tool_execution(tool_name: str, tool_input: dict) -> str:
    """Simulate tool execution for demonstration purposes."""
    if tool_name == "search_web":
        return f"[Simulated search results for: {tool_input['query']}]\n- Found 3 relevant articles\n- Key finding: Topic is well-researched"
    elif tool_name == "take_notes":
        return f"[Note saved: {tool_input['note'][:50]}...]"
    elif tool_name == "ask_clarification":
        return f"[User response: Please focus on recent developments]"
    return "[Unknown tool]"


def agent_research(topic: str, max_iterations: int = 5) -> str:
    """
    Research a topic using an AGENT approach.
    
    Notice: The LLM decides what to do at each step. The number of steps,
    which tools to use, and when to stop are all determined by the LLM.
    """
    print("=" * 60)
    print("AGENT APPROACH: LLM controls the flow")
    print("=" * 60)
    
    system_prompt = """You are a research assistant. You have tools available to help you research topics.
    
Your goal is to gather useful information and provide a helpful summary.
You decide:
- Whether to search for more information
- Whether to take notes on important findings
- Whether to ask for clarification
- When you have enough information to provide a final answer

Think about what would be most helpful at each step."""

    messages = [{
        "role": "user",
        "content": f"Please research this topic and provide a brief summary: {topic}"
    }]
    
    iteration = 0
    actions_taken = []
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print("LLM is thinking about what to do next...")
        
        # The LLM decides what to do
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            tools=research_tools,
            messages=messages
        )
        
        # Check what the LLM decided
        if response.stop_reason == "end_turn":
            # LLM decided it's done - this was ITS choice
            print("  → LLM decided: Task is complete, providing answer")
            final_response = response.content[0].text
            actions_taken.append("Provided final answer")
            break
            
        elif response.stop_reason == "tool_use":
            # LLM decided to use a tool - this was ITS choice
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → LLM decided: Use '{block.name}' tool")
                    print(f"    Input: {block.input}")
                    actions_taken.append(f"Used {block.name}")
                    
                    # Execute the tool
                    result = simulate_tool_execution(block.name, block.input)
                    print(f"    Result: {result[:100]}...")
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "user", "content": tool_results})
    
    else:
        final_response = "Max iterations reached"
    
    print(f"\n✓ Agent complete: {iteration} iterations, actions: {actions_taken}")
    print("Notice: The number and type of actions were decided by the LLM!")
    
    return final_response


# =============================================================================
# MAIN: Run both approaches to see the difference
# =============================================================================

if __name__ == "__main__":
    topic = "renewable energy storage solutions"
    
    print("\n" + "=" * 70)
    print("COMPARING WORKFLOW vs AGENT APPROACHES")
    print("=" * 70)
    
    print("\n\n" + "-" * 70)
    print("Running WORKFLOW approach...")
    print("-" * 70)
    workflow_result = workflow_research(topic)
    print(f"\nWorkflow result:\n{workflow_result}")
    
    print("\n\n" + "-" * 70)
    print("Running AGENT approach...")
    print("-" * 70)
    agent_result = agent_research(topic)
    print(f"\nAgent result:\n{agent_result}")
    
    print("\n\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print("""
1. WORKFLOW: Always 3 steps, always in the same order
   - Step 1: Generate questions (predetermined)
   - Step 2: Expand questions (predetermined)
   - Step 3: Synthesize (predetermined)
   
2. AGENT: Variable steps, LLM decides
   - Number of iterations: Determined by LLM
   - Which tools to use: Determined by LLM
   - When to stop: Determined by LLM
   
The workflow is predictable but inflexible.
The agent is flexible but less predictable.

Choose based on your needs!
""")
