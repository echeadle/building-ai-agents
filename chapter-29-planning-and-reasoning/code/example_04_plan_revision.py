"""
Patterns for plan revision and adaptation.

This example demonstrates how to detect when a plan needs revision
and create revised plans based on new information or obstacles.

Chapter 29: Planning and Reasoning
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from enum import Enum

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


class RevisionTrigger(Enum):
    """Reasons a plan might need revision."""
    STEP_FAILED = "step_failed"
    NEW_INFORMATION = "new_information"
    GOAL_CHANGED = "goal_changed"
    OBSTACLE_FOUND = "obstacle_found"
    BETTER_PATH_FOUND = "better_path_found"


def detect_revision_needed(
    original_plan: dict,
    current_step: int,
    step_result: str,
    context: str
) -> tuple[bool, RevisionTrigger | None, str]:
    """
    Analyze whether a plan needs revision.
    
    Args:
        original_plan: The plan being executed
        current_step: Which step just completed
        step_result: Result of the current step
        context: Accumulated context from previous steps
        
    Returns:
        Tuple of (needs_revision, trigger_type, explanation)
    """
    analysis_prompt = f"""Analyze whether this plan needs revision.

ORIGINAL PLAN:
Goal: {original_plan['goal']}
Steps: {json.dumps(original_plan['steps'], indent=2)}

CURRENT PROGRESS:
Just completed step {current_step}
Result: {step_result}

CONTEXT FROM PREVIOUS STEPS:
{context if context else "No previous steps."}

Analyze:
1. Did the step succeed as expected?
2. Did we learn something that changes our approach?
3. Are the remaining steps still optimal?
4. Are there any obstacles or better paths now visible?

Respond in JSON:
{{
    "needs_revision": true/false,
    "trigger": "step_failed" | "new_information" | "obstacle_found" | "better_path_found" | null,
    "explanation": "Why revision is or isn't needed",
    "suggested_changes": "What changes to make, if any"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    analysis = json.loads(text.strip())
    
    trigger = None
    if analysis.get("trigger"):
        try:
            trigger = RevisionTrigger(analysis["trigger"])
        except ValueError:
            pass
    
    return (
        analysis.get("needs_revision", False),
        trigger,
        analysis.get("explanation", "")
    )


def revise_plan(
    original_plan: dict,
    completed_steps: list[dict],
    trigger: RevisionTrigger,
    context: str
) -> dict:
    """
    Create a revised plan based on what we've learned.
    
    Args:
        original_plan: The original plan
        completed_steps: Steps already completed with results
        trigger: Why we're revising
        context: What we've learned
        
    Returns:
        Revised plan dictionary
    """
    revision_prompt = f"""Revise this plan based on new circumstances.

ORIGINAL PLAN:
Goal: {original_plan['goal']}

COMPLETED STEPS:
{json.dumps(completed_steps, indent=2)}

REVISION TRIGGER: {trigger.value}

CONTEXT:
{context}

Create a revised plan that:
1. Preserves the original goal (unless it's now impossible)
2. Builds on completed work (don't repeat what's done)
3. Addresses the reason for revision
4. Remains achievable

Respond in JSON:
{{
    "goal": "Same or adjusted goal",
    "completed_steps": {len(completed_steps)},
    "new_steps": [
        {{"step_number": N, "action": "...", "expected_outcome": "...", "rationale": "Why this step"}}
    ],
    "revision_summary": "What changed and why"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": revision_prompt}]
    )
    
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    return json.loads(text.strip())


def should_revise(step_result: str, expected_outcome: str) -> tuple[bool, str]:
    """
    Quick check if a single step result warrants plan revision.
    
    Args:
        step_result: What actually happened
        expected_outcome: What we expected
        
    Returns:
        Tuple of (should_revise, reason)
    """
    check_prompt = f"""Compare the expected vs actual outcome of this step.

Expected outcome: {expected_outcome}
Actual result: {step_result}

Answer these questions:
1. Was the step successful? (Yes/No)
2. Does the result change our approach? (Yes/No)
3. Should we revise the plan? (Yes/No)

Respond in JSON:
{{
    "step_successful": true/false,
    "changes_approach": true/false,
    "should_revise": true/false,
    "reason": "Brief explanation"
}}"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[{"role": "user", "content": check_prompt}]
    )
    
    text = response.content[0].text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    result = json.loads(text.strip())
    return result.get("should_revise", False), result.get("reason", "")


def demonstrate_plan_revision():
    """Demonstrate plan revision in action."""
    
    # Simulate a scenario where revision is needed
    original_plan = {
        "goal": "Find the best restaurant for a team dinner in downtown Seattle",
        "steps": [
            {
                "step_number": 1, 
                "action": "Search for top-rated restaurants downtown", 
                "expected_outcome": "List of 5-10 candidates"
            },
            {
                "step_number": 2, 
                "action": "Check availability for party of 10 on Friday", 
                "expected_outcome": "List of available options"
            },
            {
                "step_number": 3, 
                "action": "Compare menus for dietary restrictions (vegetarian, gluten-free)", 
                "expected_outcome": "Filtered list accommodating all diets"
            },
            {
                "step_number": 4, 
                "action": "Make reservation at best option", 
                "expected_outcome": "Confirmed booking"
            }
        ]
    }
    
    # Simulate completing step 1 with unexpected results
    step_1_result = """Found 5 top-rated restaurants in downtown Seattle:
    1. The Walrus and the Carpenter - PERMANENTLY CLOSED
    2. Canlis - Available but $200+ per person
    3. Metropolitan Grill - PERMANENTLY CLOSED  
    4. Wild Ginger - Available, moderate pricing
    5. The Butcher's Table - PERMANENTLY CLOSED
    
    Note: Economic downturn has caused several closures. Only 2 options remain."""
    
    print("=" * 60)
    print("PLAN REVISION DEMONSTRATION")
    print("=" * 60)
    
    print("\nORIGINAL PLAN:")
    print(f"Goal: {original_plan['goal']}")
    for step in original_plan["steps"]:
        print(f"  {step['step_number']}. {step['action']}")
    
    print(f"\nSTEP 1 RESULT:\n{step_1_result}")
    
    # Check if revision is needed
    print("\n" + "-" * 40)
    print("CHECKING IF REVISION NEEDED...")
    print("-" * 40)
    
    needs_revision, trigger, explanation = detect_revision_needed(
        original_plan=original_plan,
        current_step=1,
        step_result=step_1_result,
        context=""
    )
    
    print(f"Revision needed: {needs_revision}")
    print(f"Trigger: {trigger.value if trigger else 'None'}")
    print(f"Explanation: {explanation}")
    
    if needs_revision and trigger:
        print("\n" + "-" * 40)
        print("REVISING PLAN...")
        print("-" * 40)
        
        completed = [{"step_number": 1, "result": step_1_result}]
        revised = revise_plan(original_plan, completed, trigger, step_1_result)
        
        print(f"\nRevised Goal: {revised['goal']}")
        print(f"Revision Summary: {revised.get('revision_summary', 'N/A')}")
        print("\nNew Steps:")
        for step in revised.get("new_steps", []):
            print(f"  {step['step_number']}. {step['action']}")
            if step.get('rationale'):
                print(f"      → Rationale: {step['rationale']}")


def demonstrate_quick_revision_check():
    """Demonstrate quick revision checking."""
    
    print("\n" + "=" * 60)
    print("QUICK REVISION CHECK EXAMPLES")
    print("=" * 60)
    
    test_cases = [
        {
            "expected": "List of 5 candidate restaurants",
            "actual": "Found 5 highly-rated restaurants matching criteria",
            "scenario": "Success - matches expectation"
        },
        {
            "expected": "List of 5 candidate restaurants",
            "actual": "Search failed - API rate limited, no results returned",
            "scenario": "Failure - step didn't complete"
        },
        {
            "expected": "List of 5 candidate restaurants",
            "actual": "Found only 2 restaurants, but discovered a food festival happening that night with many options",
            "scenario": "New opportunity - better path available"
        },
        {
            "expected": "Available time slots for meeting",
            "actual": "Calendar shows participant is on vacation all next week",
            "scenario": "Obstacle - blocking issue discovered"
        }
    ]
    
    for case in test_cases:
        print(f"\nScenario: {case['scenario']}")
        print(f"Expected: {case['expected']}")
        print(f"Actual: {case['actual']}")
        
        should_rev, reason = should_revise(case['actual'], case['expected'])
        
        print(f"Should revise? {should_rev}")
        print(f"Reason: {reason}")
        print("-" * 40)


if __name__ == "__main__":
    demonstrate_plan_revision()
    demonstrate_quick_revision_check()
