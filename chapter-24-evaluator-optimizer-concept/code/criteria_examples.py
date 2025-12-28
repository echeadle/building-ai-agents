"""
Evaluation Criteria Examples

This file demonstrates how to design clear, actionable evaluation criteria
for different use cases. Good criteria are the foundation of an effective
evaluator-optimizer system.

Chapter 24: Evaluator-Optimizer - Concept and Design
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationCriterion:
    """
    A well-designed evaluation criterion.
    
    Good criteria are:
    - Specific: Describes something observable in the output
    - Actionable: If failed, it's clear how to fix it
    - Independent: Can be evaluated on its own
    - Prioritized: Clear whether it's critical, important, or nice-to-have
    """
    name: str
    description: str
    priority: str  # "critical", "important", "nice_to_have"
    pass_description: str
    fail_description: str
    examples: list[dict] = field(default_factory=list)  # Example pass/fail cases


# =============================================================================
# EXAMPLE 1: PRODUCT DESCRIPTION CRITERIA
# =============================================================================

PRODUCT_DESCRIPTION_CRITERIA = [
    EvaluationCriterion(
        name="factual_accuracy",
        description="All stated features, specs, and claims are verifiable and correct",
        priority="critical",
        pass_description="Every fact can be verified from the product specifications",
        fail_description="Contains unverifiable claims, exaggerations, or errors",
        examples=[
            {
                "pass": "The laptop weighs 2.8 pounds and has a 14-inch display",
                "fail": "This revolutionary laptop is the lightest ever made"
            }
        ]
    ),
    EvaluationCriterion(
        name="completeness",
        description="Includes key features, target user, and primary use case",
        priority="critical",
        pass_description="Reader knows what it does, who it's for, and when to use it",
        fail_description="Missing essential information about features or use case",
        examples=[
            {
                "pass": "Perfect for students who need a portable laptop for note-taking and research",
                "fail": "A great laptop with many features"  # Who is it for? What features?
            }
        ]
    ),
    EvaluationCriterion(
        name="clarity",
        description="Technical terms explained, sentences under 20 words on average",
        priority="important",
        pass_description="General consumer understands without looking things up",
        fail_description="Contains jargon or complex sentences requiring expertise",
        examples=[
            {
                "pass": "The 512GB solid-state drive loads programs in seconds",
                "fail": "Features NVMe Gen4 SSD with 7000MB/s sequential read throughput"
            }
        ]
    ),
    EvaluationCriterion(
        name="persuasiveness",
        description="Focuses on benefits (not just features), creates desire",
        priority="important",
        pass_description="Reader can imagine using and benefiting from the product",
        fail_description="Reads like a spec sheet without emotional connection",
        examples=[
            {
                "pass": "Never worry about running out of battery during a long flight",
                "fail": "18-hour battery life"  # Feature without benefit context
            }
        ]
    ),
    EvaluationCriterion(
        name="length",
        description="Between 100-200 words",
        priority="important",
        pass_description="Word count is within the specified range",
        fail_description="Too short (incomplete) or too long (unfocused)",
        examples=[]
    ),
    EvaluationCriterion(
        name="brand_voice",
        description="Matches company tone: professional but approachable",
        priority="nice_to_have",
        pass_description="Sounds like our other product descriptions",
        fail_description="Too formal, too casual, or generic sounding",
        examples=[
            {
                "pass": "We designed this with busy professionals in mind",
                "fail": "This cutting-edge solution leverages synergies"
            }
        ]
    ),
]


# =============================================================================
# EXAMPLE 2: PROFESSIONAL EMAIL CRITERIA
# =============================================================================

EMAIL_CRITERIA = [
    EvaluationCriterion(
        name="purpose_clarity",
        description="Main point stated within first two sentences",
        priority="critical",
        pass_description="Reader immediately knows what the email is about/asking",
        fail_description="Purpose buried in middle or end, or unclear throughout",
        examples=[
            {
                "pass": "I'm writing to request your approval for the Q4 budget.",
                "fail": "Hope you're doing well. I wanted to touch base about a few things..."
            }
        ]
    ),
    EvaluationCriterion(
        name="appropriate_tone",
        description="Tone matches relationship and context",
        priority="critical",
        pass_description="Feels appropriate for the sender-recipient relationship",
        fail_description="Too casual for formal context, or too stiff for colleagues",
        examples=[
            {
                "pass": "(To colleague) Hey Sarah, quick question about the report...",
                "fail": "(To colleague) Dear Ms. Johnson, I hope this message finds you well..."
            },
            {
                "pass": "(To client) Thank you for your inquiry. We'd be happy to help.",
                "fail": "(To client) Hey! Got your message, here's the deal..."
            }
        ]
    ),
    EvaluationCriterion(
        name="action_clarity",
        description="If action needed: what, by whom, by when are explicit",
        priority="critical",
        pass_description="Reader knows exactly what they should do next",
        fail_description="Action is vague, missing deadline, or unclear owner",
        examples=[
            {
                "pass": "Please review the attached and send your feedback by Friday at 5pm.",
                "fail": "Let me know your thoughts when you get a chance."
            }
        ]
    ),
    EvaluationCriterion(
        name="conciseness",
        description="No unnecessary words, repetition, or padding",
        priority="important",
        pass_description="Every sentence serves a purpose",
        fail_description="Contains filler phrases or could be shortened 30%+",
        examples=[
            {
                "pass": "The project deadline is March 15.",
                "fail": "I wanted to take a moment to reach out and let you know that the deadline for the project we've been discussing is going to be March 15."
            }
        ]
    ),
    EvaluationCriterion(
        name="grammar_spelling",
        description="No errors that undermine professionalism",
        priority="critical",
        pass_description="Error-free",
        fail_description="Any grammatical or spelling errors present",
        examples=[]
    ),
    EvaluationCriterion(
        name="subject_line",
        description="Subject accurately summarizes content",
        priority="important",
        pass_description="Recipient knows what email is about before opening",
        fail_description="Vague ('Quick question'), misleading, or missing",
        examples=[
            {
                "pass": "Q4 Budget Approval Needed by Nov 15",
                "fail": "Following up"
            }
        ]
    ),
]


# =============================================================================
# EXAMPLE 3: CODE DOCUMENTATION CRITERIA
# =============================================================================

CODE_DOCUMENTATION_CRITERIA = [
    EvaluationCriterion(
        name="purpose_description",
        description="First line clearly states what the function/class does",
        priority="critical",
        pass_description="Reader understands purpose without reading code",
        fail_description="Purpose is vague, missing, or requires code reading",
        examples=[
            {
                "pass": "Calculate the compound interest for a given principal and rate.",
                "fail": "This function does calculations."
            }
        ]
    ),
    EvaluationCriterion(
        name="parameter_documentation",
        description="All parameters have type, description, and constraints",
        priority="critical",
        pass_description="Reader knows what to pass for each parameter",
        fail_description="Parameters missing, types unclear, or constraints unstated",
        examples=[
            {
                "pass": "principal (float): The initial investment amount. Must be positive.",
                "fail": "principal: the amount"
            }
        ]
    ),
    EvaluationCriterion(
        name="return_documentation",
        description="Return value has type and description",
        priority="critical",
        pass_description="Reader knows what they'll get back",
        fail_description="Return type or meaning unclear",
        examples=[
            {
                "pass": "Returns: float - The final amount after interest, rounded to 2 decimals",
                "fail": "Returns the result"
            }
        ]
    ),
    EvaluationCriterion(
        name="error_documentation",
        description="Raised exceptions are documented with conditions",
        priority="important",
        pass_description="Reader knows what errors to expect and why",
        fail_description="Exceptions missing or conditions unclear",
        examples=[
            {
                "pass": "Raises: ValueError - If principal is negative or rate is not between 0 and 1",
                "fail": "May raise errors"
            }
        ]
    ),
    EvaluationCriterion(
        name="example_usage",
        description="Includes at least one working example",
        priority="important",
        pass_description="Reader can copy example to understand basic usage",
        fail_description="No example, or example doesn't work",
        examples=[
            {
                "pass": "Example: >>> compound_interest(1000, 0.05, 10)  # Returns: 1628.89",
                "fail": "Usage: call the function with appropriate parameters"
            }
        ]
    ),
    EvaluationCriterion(
        name="format_compliance",
        description="Follows Google or NumPy docstring style consistently",
        priority="nice_to_have",
        pass_description="Formatting matches chosen style guide",
        fail_description="Mixed styles or non-standard formatting",
        examples=[]
    ),
]


# =============================================================================
# EXAMPLE 4: TECHNICAL EXPLANATION CRITERIA
# =============================================================================

TECHNICAL_EXPLANATION_CRITERIA = [
    EvaluationCriterion(
        name="accuracy",
        description="All technical claims are correct",
        priority="critical",
        pass_description="An expert would not find errors",
        fail_description="Contains technical inaccuracies or oversimplifications that mislead",
        examples=[]
    ),
    EvaluationCriterion(
        name="audience_appropriateness",
        description="Complexity matches target audience level",
        priority="critical",
        pass_description="Target audience can follow without being talked down to",
        fail_description="Too advanced for beginners or too simple for experts",
        examples=[
            {
                "pass": "(For beginners) Think of an API like a waiter in a restaurant...",
                "fail": "(For beginners) REST APIs use HTTP methods to perform CRUD operations..."
            }
        ]
    ),
    EvaluationCriterion(
        name="structure",
        description="Logical flow from simple to complex, or problem to solution",
        priority="important",
        pass_description="Each section builds on the previous",
        fail_description="Jumps around, assumes knowledge not yet introduced",
        examples=[]
    ),
    EvaluationCriterion(
        name="concrete_examples",
        description="Abstract concepts illustrated with specific examples",
        priority="important",
        pass_description="Every major concept has an example",
        fail_description="Concepts explained only abstractly",
        examples=[
            {
                "pass": "For example, when you search on Google, your browser sends an API request...",
                "fail": "API requests return data based on specified parameters."
            }
        ]
    ),
    EvaluationCriterion(
        name="completeness",
        description="Covers topic sufficiently for stated purpose",
        priority="important",
        pass_description="Reader can accomplish their goal with this information",
        fail_description="Missing key information needed for understanding",
        examples=[]
    ),
]


# =============================================================================
# UTILITY: FORMAT CRITERIA FOR EVALUATOR PROMPT
# =============================================================================

def format_criteria_for_prompt(criteria: list[EvaluationCriterion]) -> str:
    """
    Format criteria list into a string for use in an evaluator prompt.
    
    Args:
        criteria: List of evaluation criteria
        
    Returns:
        Formatted string ready to include in a prompt
    """
    lines = []
    
    # Group by priority
    critical = [c for c in criteria if c.priority == "critical"]
    important = [c for c in criteria if c.priority == "important"]
    nice_to_have = [c for c in criteria if c.priority == "nice_to_have"]
    
    if critical:
        lines.append("CRITICAL (must pass):")
        for i, c in enumerate(critical, 1):
            lines.append(f"  {i}. {c.name.upper()}")
            lines.append(f"     {c.description}")
            lines.append(f"     Pass: {c.pass_description}")
            lines.append(f"     Fail: {c.fail_description}")
            lines.append("")
    
    if important:
        lines.append("IMPORTANT (should pass):")
        for i, c in enumerate(important, 1):
            lines.append(f"  {i}. {c.name.upper()}")
            lines.append(f"     {c.description}")
            lines.append(f"     Pass: {c.pass_description}")
            lines.append(f"     Fail: {c.fail_description}")
            lines.append("")
    
    if nice_to_have:
        lines.append("NICE TO HAVE (polish):")
        for i, c in enumerate(nice_to_have, 1):
            lines.append(f"  {i}. {c.name.upper()}")
            lines.append(f"     {c.description}")
            lines.append(f"     Pass: {c.pass_description}")
            lines.append(f"     Fail: {c.fail_description}")
            lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN: DISPLAY EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION CRITERIA EXAMPLES")
    print("=" * 70)
    print()
    
    examples = [
        ("Product Description", PRODUCT_DESCRIPTION_CRITERIA),
        ("Professional Email", EMAIL_CRITERIA),
        ("Code Documentation", CODE_DOCUMENTATION_CRITERIA),
        ("Technical Explanation", TECHNICAL_EXPLANATION_CRITERIA),
    ]
    
    for name, criteria in examples:
        print(f"\n{'=' * 70}")
        print(f"{name.upper()} CRITERIA")
        print(f"{'=' * 70}")
        
        critical = [c for c in criteria if c.priority == "critical"]
        important = [c for c in criteria if c.priority == "important"]
        nice_to_have = [c for c in criteria if c.priority == "nice_to_have"]
        
        print(f"\nTotal: {len(criteria)} criteria")
        print(f"  - Critical: {len(critical)}")
        print(f"  - Important: {len(important)}")
        print(f"  - Nice to have: {len(nice_to_have)}")
        
        print("\nCriteria names:")
        for c in criteria:
            print(f"  [{c.priority[:4]}] {c.name}")
    
    # Show formatted output example
    print("\n" + "=" * 70)
    print("EXAMPLE: FORMATTED FOR EVALUATOR PROMPT")
    print("=" * 70)
    print("\nEmail criteria formatted for use in a prompt:")
    print("-" * 50)
    print(format_criteria_for_prompt(EMAIL_CRITERIA))
