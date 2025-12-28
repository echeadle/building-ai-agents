"""
Common Prompt Mistakes Example

Demonstrates common prompt engineering mistakes and how to fix them.

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


class PromptMistakeDemo:
    """Demonstrates common prompt mistakes and their fixes."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def demo_mistake_1_vague_tool_descriptions(self) -> None:
        """Mistake: Vague tool descriptions."""
        print("\n" + "="*70)
        print("MISTAKE 1: Vague Tool Descriptions")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_tool = {
            "name": "search",
            "description": "Searches for information",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
        print(f"Tool: {bad_tool['name']}")
        print(f"Description: '{bad_tool['description']}'")
        print("\nProblem: Agent doesn't know WHEN to use this or HOW to use it well")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_tool = {
            "name": "web_search",
            "description": """Searches the web for current information.

When to use:
- User asks about recent events or current facts
- You need to verify information beyond your training data
- Looking for specific statistics or sources

When NOT to use:
- General knowledge questions you can answer confidently
- Mathematical calculations (use calculator)
- Historical facts that don't change

Parameters:
- query: Use keywords, not full sentences (e.g., "python asyncio" not "how do I use asyncio")

Returns: List of results with titles, URLs, and snippets""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (not a full sentence)"
                    }
                },
                "required": ["query"]
            }
        }
        print(f"Tool: {good_tool['name']}")
        print(f"Description:\n{good_tool['description']}")
        print("\nBenefit: Agent knows exactly when and how to use this tool")
    
    def demo_mistake_2_no_termination(self) -> None:
        """Mistake: No termination conditions."""
        print("\n" + "="*70)
        print("MISTAKE 2: No Termination Conditions")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_prompt = """You are a research assistant. Search for information 
and read articles until you can answer the user's question."""
        print(bad_prompt)
        print("\nProblem: Agent never knows when it's 'enough' → infinite loop")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_prompt = """You are a research assistant.

Research process:
1. Perform 2-4 searches to find relevant sources
2. Read 3-5 of the most authoritative sources
3. Take notes on key findings
4. Synthesize a report

Termination conditions (stop researching when):
- You have read at least 3 high-quality sources
- Your notes address all aspects of the research question
- Additional searches return redundant information
- You've performed 5 searches (diminishing returns)

If you reach any termination condition, synthesize your report."""
        print(good_prompt)
        print("\nBenefit: Agent knows when to stop and produce output")
    
    def demo_mistake_3_conflicting_instructions(self) -> None:
        """Mistake: Conflicting instructions."""
        print("\n" + "="*70)
        print("MISTAKE 3: Conflicting Instructions")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_prompt = """Always use tools to verify information before responding.
Keep responses fast—don't make unnecessary tool calls.
Be thorough and check multiple sources."""
        print(bad_prompt)
        print("\nProblem: Instructions conflict - agent is confused")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_prompt = """Tool usage priorities:
1. Use tools for facts you cannot know (current events, statistics)
2. Answer directly for general knowledge within your training
3. Use tools when accuracy is critical
4. Don't use tools for trivial queries with obvious answers

When in doubt: Use the tool. Accuracy > speed."""
        print(good_prompt)
        print("\nBenefit: Clear priority order resolves conflicts")
    
    def demo_mistake_4_assuming_state(self) -> None:
        """Mistake: Assuming agent knows its state."""
        print("\n" + "="*70)
        print("MISTAKE 4: Assuming Agent Knows Its State")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_prompt = """Don't repeat information you've already provided."""
        print(bad_prompt)
        print("\nProblem: In long conversations, agent doesn't remember turn 20")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_prompt = """Conversation management:
- Before answering, check recent messages (last 5-10 messages)
- If you've recently provided information, refer to it
- If conversation is very long and you're unsure, ask:
  "Have we already discussed X?"
- It's okay to summarize previous points if helpful"""
        print(good_prompt)
        print("\nBenefit: Agent knows how to handle conversation history")
    
    def demo_mistake_5_no_error_handling(self) -> None:
        """Mistake: No error handling guidance."""
        print("\n" + "="*70)
        print("MISTAKE 5: No Error Handling Guidance")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_prompt = """You have access to a web_read tool to fetch webpage content."""
        print(bad_prompt)
        print("\nProblem: When web_read fails, agent doesn't know what to do")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_prompt = """web_read(url: str) -> str | ErrorResult

Error handling:
- If web_read returns an error, try an alternative source
- If multiple sources fail, inform user and suggest alternatives

Common errors:
  - 404: Page not found → Try different source
  - Timeout: Server slow → Try once more, then move on
  - Access denied: Paywall → Find free source

Example recovery:
If paywalled article fails, search for:
"[article title] summary" or "[article title] free version"
"""
        print(good_prompt)
        print("\nBenefit: Agent knows how to recover from failures")
    
    def demo_mistake_6_over_engineering(self) -> None:
        """Mistake: Over-engineering the prompt."""
        print("\n" + "="*70)
        print("MISTAKE 6: Over-Engineering the Prompt")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_prompt = """You are an advanced AI system with sophisticated 
reasoning capabilities, leveraging state-of-the-art natural language 
processing to provide comprehensive, nuanced responses that take into 
account multiple perspectives and edge cases...
[3000 more words covering every possible scenario]"""
        print(bad_prompt[:200] + "...\n[continues for 3000 words]")
        print("\nProblem: Too complex, agent gets confused or ignores most of it")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_prompt = """You are a helpful assistant with web search and reading capabilities.

Core behaviors:
1. Search when you need information
2. Read sources thoroughly
3. Cite your sources
4. Say "I don't know" if you can't find information

That's it. Keep it simple."""
        print(good_prompt)
        print("\nBenefit: Simple, clear, actually followed")
        print("\n💡 Tip: Start minimal, add complexity only when needed")
    
    def demo_mistake_7_no_examples(self) -> None:
        """Mistake: No examples for complex behavior."""
        print("\n" + "="*70)
        print("MISTAKE 7: No Examples for Complex Behavior")
        print("="*70)
        
        print("\n❌ BAD EXAMPLE:")
        print("-" * 70)
        bad_prompt = """When the user asks about comparisons, gather 
information about each option and present a balanced analysis."""
        print(bad_prompt)
        print("\nProblem: Agent's comparison structure varies wildly")
        
        print("\n✅ GOOD EXAMPLE:")
        print("-" * 70)
        good_prompt = """When asked for comparisons, follow this example:

User: "Should I learn Python or JavaScript?"

Your response structure:
---
## Python
**Best for:** Data science, automation, backend
**Learning curve:** Beginner-friendly
**Job market:** Strong in data/ML roles
**Sources:** [citations]

## JavaScript
**Best for:** Web development, full-stack
**Learning curve:** Moderate
**Job market:** Highest demand overall
**Sources:** [citations]

## Recommendation
[Based on user context or balanced advice]
---

Follow this structure for ALL comparison requests."""
        print(good_prompt)
        print("\nBenefit: Consistent, structured comparisons")


def demonstrate_checklist() -> None:
    """Provide a checklist for avoiding mistakes."""
    print("\n" + "="*70)
    print("PROMPT ENGINEERING CHECKLIST")
    print("="*70)
    
    checklist = """
Before deploying any agent prompt, verify:

TOOL DESCRIPTIONS:
□ Each tool has a clear purpose stated
□ "When to use" and "When NOT to use" are specified
□ Parameters are fully described
□ Examples of good/bad usage provided
□ Error conditions are documented

INSTRUCTIONS:
□ No conflicting instructions
□ Priorities are clear when instructions might conflict
□ Termination conditions are explicit
□ Process/workflow is described step-by-step
□ Error handling guidance is provided

STATE MANAGEMENT:
□ Assumptions about conversation history are explicit
□ Agent knows how to check what it already said
□ Long conversation handling is addressed
□ Tool call history tracking is clear

OUTPUT FORMAT:
□ Expected format is explicitly described
□ Examples of correct format provided
□ Strong language used ("MUST", "ONLY")
□ Validation strategy in place

COMPLEXITY:
□ Prompt is as simple as possible
□ Each instruction is necessary
□ No redundant or contradictory guidance
□ Complex behaviors have examples

TESTING:
□ Tested with edge cases
□ Tested with error conditions
□ Tested with conflicting inputs
□ Tested with long conversations
"""
    print(checklist)


def demonstrate_debugging_process() -> None:
    """Show how to debug prompt issues."""
    print("\n" + "="*70)
    print("DEBUGGING PROMPT ISSUES")
    print("="*70)
    
    process = """
When agent behavior is unexpected:

1. CHECK THE PROMPT FIRST (not the code!)
   - Most "bugs" are actually prompt issues
   - Read the prompt as if you're the agent
   - Look for ambiguity or conflicts

2. IDENTIFY THE SYMPTOM
   - Agent not using tools? → Check tool descriptions
   - Agent looping? → Check termination conditions
   - Agent using wrong tool? → Check examples and "when to use"
   - Inconsistent behavior? → Check for conflicting instructions
   - Wrong output format? → Check format instructions

3. ISOLATE THE ISSUE
   - Test with minimal prompt
   - Add instructions back one at a time
   - Find which instruction causes the problem

4. FIX SYSTEMATICALLY
   - Make ONE change at a time
   - Test after each change
   - Document what fixed it

5. PREVENT REGRESSION
   - Save the failing case as a test
   - Add it to your test suite
   - Check that fix didn't break other cases

Example debugging session:
Problem: Agent searches 20 times and never stops
Diagnosis: No termination condition
Fix: Add "Stop after 5 searches or 3 good sources"
Test: Verify agent now stops appropriately
Prevent: Add test case for "excessive searching"
"""
    print(process)


if __name__ == "__main__":
    print("Common Prompt Mistakes Example")
    print("=" * 70)
    print("This example shows common prompt mistakes and how to fix them.")
    print()
    
    demo = PromptMistakeDemo()
    
    # Demonstrate each mistake
    demo.demo_mistake_1_vague_tool_descriptions()
    demo.demo_mistake_2_no_termination()
    demo.demo_mistake_3_conflicting_instructions()
    demo.demo_mistake_4_assuming_state()
    demo.demo_mistake_5_no_error_handling()
    demo.demo_mistake_6_over_engineering()
    demo.demo_mistake_7_no_examples()
    
    # Provide checklist
    demonstrate_checklist()
    
    # Show debugging process
    demonstrate_debugging_process()
    
    print("\n" + "="*70)
    print("FINAL REMINDERS")
    print("="*70)
    print("""
1. Most agent bugs are PROMPT bugs, not code bugs
2. Start simple, add complexity only when needed
3. Be explicit - assume the agent knows nothing
4. Provide examples for complex behaviors
5. Test thoroughly with edge cases
6. Use the checklist before deploying
7. Document what works and what doesn't

Remember: Prompt engineering is ITERATIVE.
Your first prompt won't be perfect, and that's okay.
Refine based on observed behavior.
""")
