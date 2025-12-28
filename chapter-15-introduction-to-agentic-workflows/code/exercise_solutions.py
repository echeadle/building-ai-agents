"""
Exercise Solutions for Chapter 15: Introduction to Agentic Workflows

This file contains the solutions to the practical exercise from the chapter,
analyzing three scenarios and determining the best workflow pattern for each.
"""

# =============================================================================
# EXERCISE: Analyze three scenarios and recommend workflow patterns
# =============================================================================

"""
SCENARIO 1: Email Triage System
--------------------------------
Requirements:
- Incoming emails to a support inbox
- Need to: classify priority, identify topic, route to correct department, 
  generate acknowledgment

ANALYSIS:

1. What's the task structure?
   - Multiple distinct operations on the same input
   - Clear categorization needed (by priority AND topic)
   - Different outcomes based on classification

2. Are subtasks sequential or independent?
   - Classification must happen first
   - Routing depends on classification
   - Acknowledgment can happen after routing
   → Mostly sequential with a classification gate

3. Do different inputs need different handling?
   - Yes! High-priority urgent issues vs. low-priority questions
   - Technical issues vs. billing issues need different departments
   → ROUTING is central to this workflow

4. Is the structure predictable?
   - Yes, we know the departments and priority levels
   → No need for orchestrator-workers

5. Does iteration help?
   - Not really - classification is either right or wrong
   - Acknowledgments don't need refinement
   → No need for evaluator-optimizer

RECOMMENDED PATTERN: ROUTING

Architecture:
```
Email Input
    │
    ▼
┌─────────────────┐
│  Classifier     │  ← Determine priority AND topic
│ (Multi-label)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────┐  ┌──────┐
│ High │  │ Low  │
│ Pri  │  │ Pri  │
└──┬───┘  └──┬───┘
   │         │
   ├─────────┤
   ▼         ▼
┌─────────────────┐
│  Topic Router   │
└────────┬────────┘
    ┌────┼────┐
    ▼    ▼    ▼
┌────┐┌────┐┌────┐
│Tech││Bill││Gen │
└─┬──┘└─┬──┘└─┬──┘
  │     │     │
  └─────┼─────┘
        ▼
  ┌───────────┐
  │Generate   │
  │Acknowledge│
  └───────────┘
```

Considerations:
- May want two-stage routing (priority first, then topic)
- Could parallelize topic classification with priority classification
- Include fallback for unclassifiable emails
- Track routing accuracy over time
"""

scenario_1_recommendation = {
    "scenario": "Email Triage System",
    "primary_pattern": "Routing",
    "secondary_pattern": "Prompt Chaining (for acknowledgment generation)",
    "confidence": "High",
    "reasoning": "Clear categorization with different handling per category is the textbook use case for routing. The two-dimensional classification (priority AND topic) suggests a two-stage router.",
    "architecture_notes": [
        "Two-stage routing: priority → topic",
        "Specialized handlers per department",
        "Chain acknowledgment after routing",
        "Include catch-all handler for edge cases"
    ]
}


"""
SCENARIO 2: Code Documentation Generator
-----------------------------------------
Requirements:
- Given a Python file, generate docstrings for all functions
- Each function is independent
- Need high-quality, consistent documentation

ANALYSIS:

1. What's the task structure?
   - Parse file → identify functions → document each → reassemble
   - Each function documentation is independent

2. Are subtasks sequential or independent?
   - Parsing must happen first
   - But documenting function A doesn't depend on function B
   → PARALLELIZATION (sectioning) for the documentation step

3. Do different inputs need different handling?
   - Not really - all functions get similar treatment
   → Routing not needed

4. Is the structure predictable?
   - Yes, we know we need to document each function
   → Orchestrator-workers not needed

5. Does iteration help?
   - Possibly - could evaluate docs for completeness
   - But likely overkill for this task
   → Maybe light evaluation, not full eval-optimize loop

RECOMMENDED PATTERN: PARALLELIZATION (Sectioning)

Architecture:
```
Python File Input
       │
       ▼
┌─────────────┐
│ Parse File  │  ← Extract function signatures and bodies
│ (Code step) │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│     Parallel Documentation           │
│  ┌──────┐  ┌──────┐  ┌──────┐       │
│  │Func A│  │Func B│  │Func C│  ...  │
│  └──┬───┘  └──┬───┘  └──┬───┘       │
│     │        │        │              │
└─────┼────────┼────────┼──────────────┘
      │        │        │
      └────────┼────────┘
               ▼
       ┌───────────┐
       │ Reassemble│
       │ into file │
       └───────────┘
```

Considerations:
- Need consistent style across all generated docs
- Include style guide in each parallel prompt
- Parsing step is code, not LLM
- Could add evaluation step for quality consistency
"""

scenario_2_recommendation = {
    "scenario": "Code Documentation Generator",
    "primary_pattern": "Parallelization (Sectioning)",
    "secondary_pattern": "Prompt Chaining (parse → parallelize → assemble)",
    "confidence": "High",
    "reasoning": "Each function's documentation is completely independent, making this ideal for parallel sectioning. The overall workflow is a chain: parse, parallelize, reassemble.",
    "architecture_notes": [
        "Parse functions with code (AST), not LLM",
        "Parallelize documentation calls",
        "Include style guide in system prompt for consistency",
        "Aggregate results while preserving file structure"
    ]
}


"""
SCENARIO 3: Essay Writing Assistant
------------------------------------
Requirements:
- Help students improve their essays
- Provide feedback on thesis, structure, evidence, and writing quality
- Students should receive actionable improvement suggestions

ANALYSIS:

1. What's the task structure?
   - Analyze essay from multiple angles
   - Provide specific feedback
   - Possibly help with revisions

2. Are subtasks sequential or independent?
   - Thesis, structure, evidence, and quality can be evaluated independently
   → PARALLELIZATION (sectioning) for evaluation

3. Do different inputs need different handling?
   - Maybe different essay types (argumentative vs. narrative)?
   - But core feedback categories apply to most essays
   → Light routing possible, but not primary pattern

4. Is the structure predictable?
   - Yes, we know the four evaluation categories
   → Orchestrator-workers not needed

5. Does iteration help?
   - YES! This is exactly the revision process:
   - Draft → Feedback → Revise → Re-evaluate → ...
   → EVALUATOR-OPTIMIZER for the improvement loop

RECOMMENDED PATTERN: COMBINATION

Architecture:
```
Essay Input
    │
    ▼
┌───────────────────────────────────────┐
│    Parallel Evaluation (Sectioning)   │
│  ┌───────┐┌───────┐┌────────┐┌─────┐ │
│  │Thesis ││Struct ││Evidence││Style│ │
│  └───┬───┘└───┬───┘└───┬────┘└──┬──┘ │
└──────┼────────┼────────┼────────┼─────┘
       │        │        │        │
       └────────┴────────┴────────┘
                │
                ▼
       ┌────────────────┐
       │ Synthesize     │
       │ Feedback       │
       └───────┬────────┘
               │
               ▼
  ┌────────────────────────────────┐
  │     Evaluator-Optimizer Loop   │
  │   (if student requests help    │
  │    with revisions)             │
  │  ┌───────┐      ┌─────────┐   │
  │  │Revise │◀────▶│Evaluate │   │
  │  │Essay  │      │Revision │   │
  │  └───────┘      └─────────┘   │
  └────────────────────────────────┘
```

Considerations:
- Parallel evaluation saves time and covers multiple angles
- Synthesize feedback into coherent, prioritized suggestions
- Offer optional revision loop for students who want more help
- Clear evaluation criteria essential for effective iteration
"""

scenario_3_recommendation = {
    "scenario": "Essay Writing Assistant",
    "primary_pattern": "Parallelization (Sectioning) + Evaluator-Optimizer",
    "secondary_pattern": None,
    "confidence": "High",
    "reasoning": "Essay analysis naturally splits into independent categories (parallel sectioning). If the student wants revision help, an evaluator-optimizer loop mimics the human revision process.",
    "architecture_notes": [
        "Parallel evaluation for thesis/structure/evidence/style",
        "Synthesize into prioritized, actionable feedback",
        "Optional eval-optimize loop for revision assistance",
        "Clear rubric criteria for evaluation"
    ]
}


# =============================================================================
# PRINT SOLUTIONS
# =============================================================================

def print_solution(rec: dict) -> None:
    """Print a formatted solution."""
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {rec['scenario']}")
    print(f"{'=' * 60}")
    print(f"\nPrimary Pattern: {rec['primary_pattern']}")
    if rec.get('secondary_pattern'):
        print(f"Secondary Pattern: {rec['secondary_pattern']}")
    print(f"Confidence: {rec['confidence']}")
    print(f"\nReasoning: {rec['reasoning']}")
    print("\nArchitecture Notes:")
    for note in rec['architecture_notes']:
        print(f"  • {note}")


def main():
    """Display all exercise solutions."""
    print("\n" + "#" * 60)
    print("# EXERCISE SOLUTIONS")
    print("# Chapter 15: Introduction to Agentic Workflows")
    print("#" * 60)
    
    print_solution(scenario_1_recommendation)
    print_solution(scenario_2_recommendation)
    print_solution(scenario_3_recommendation)
    
    print("\n" + "=" * 60)
    print("SUMMARY: Pattern Selection Principles")
    print("=" * 60)
    print("""
1. Start by identifying the task structure:
   - Sequential dependencies → Chaining
   - Independent subtasks → Parallelization
   - Different input types → Routing
   - Unpredictable subtasks → Orchestrator-Workers
   - Quality iteration needed → Evaluator-Optimizer

2. Look for natural combinations:
   - Routing + specialized workflows per route
   - Chaining with parallel steps
   - Parallelization with evaluation synthesis

3. Always ask: "Does this complexity add real value?"
   - More patterns = more latency, cost, and code
   - Start with the simplest approach that might work
   - Add complexity to address specific shortcomings

4. Design for observability:
   - Track which routes are used
   - Log parallel step results
   - Monitor iteration counts
   - Measure quality improvements
""")


if __name__ == "__main__":
    main()
