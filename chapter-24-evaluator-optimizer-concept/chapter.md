---
chapter: 24
title: "Evaluator-Optimizer - Concept and Design"
part: 3
date: 2025-01-14
draft: false
---

# Chapter 24: Evaluator-Optimizer - Concept and Design

## Introduction

Have you ever written something, read it back, and immediately thought "I can do better"? That moment of self-reflection—recognizing what's wrong and how to improve it—is at the heart of quality work. What if we could give our AI agents the same capability?

In the previous chapters, we explored orchestrator-workers, where a coordinator breaks down complex tasks and delegates them to specialized workers. That pattern excels when tasks can be decomposed into independent subtasks. But what about tasks that require *iterative refinement*—where the first attempt is rarely the best, and improvement comes through cycles of creation and critique?

This chapter introduces the **evaluator-optimizer pattern**, the fifth and final workflow pattern in our toolkit. This pattern creates a feedback loop where one LLM generates content while another evaluates it, driving continuous improvement until quality standards are met.

You've encountered this pattern before—it's how human writing works. Writers create drafts, editors provide feedback, writers revise. The evaluator-optimizer pattern automates this dance, enabling your agents to refine their outputs to remarkably high quality.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain what the evaluator-optimizer pattern is and how it differs from other workflow patterns
- Identify use cases where iterative refinement outperforms single-shot generation
- Design effective evaluation criteria that drive meaningful improvements
- Determine appropriate stopping conditions to prevent infinite loops
- Sketch the architecture of an evaluator-optimizer system before implementing it

## What Is Evaluator-Optimizer?

The **evaluator-optimizer pattern** creates a feedback loop between two LLM roles:

1. **Generator (Optimizer)**: Produces or improves content based on the task and any feedback received
2. **Evaluator**: Assesses the generated content against defined criteria and provides actionable feedback

These roles work in a loop:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌──────────────┐         ┌──────────────┐               │
│    │              │         │              │               │
│    │   Generator  │────────▶│  Evaluator   │               │
│    │              │         │              │               │
│    └──────────────┘         └──────────────┘               │
│           ▲                        │                        │
│           │                        │                        │
│           │    Feedback +          │                        │
│           │    "Keep Improving"    │                        │
│           │                        ▼                        │
│           │                 ┌──────────────┐               │
│           └─────────────────│   Quality    │               │
│              (if not done)  │    Gate      │               │
│                             │              │               │
│                             └──────────────┘               │
│                                    │                        │
│                                    │ (if done)              │
│                                    ▼                        │
│                             ┌──────────────┐               │
│                             │    Final     │               │
│                             │   Output     │               │
│                             └──────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The loop continues until either:
- The evaluator determines the output meets quality standards
- A maximum iteration count is reached (safety limit)

### The Key Insight

The power of this pattern comes from a simple observation:

> **If humans can give useful feedback on a task, LLMs can too.**

When you ask a human editor to review an article, they don't need to be able to write the article themselves—they just need to recognize what's good and what needs work. Similarly, an LLM can often evaluate content more reliably than it can generate perfect content on the first try.

This separation of concerns—generation vs. evaluation—leverages different strengths:
- **Generation** requires creativity and comprehensiveness
- **Evaluation** requires judgment and specificity

By splitting these roles, we get better results than asking a single LLM to "write something perfect."

## When to Use Evaluator-Optimizer

The evaluator-optimizer pattern shines in specific situations. Let's explore when it's the right choice—and when simpler approaches work better.

### Ideal Use Cases

**1. Quality-Critical Content Generation**

When output quality directly impacts outcomes, iterative refinement pays off:

- **Marketing copy**: Headlines, ad copy, email subject lines
- **Technical documentation**: API docs, user guides, tutorials  
- **Creative writing**: Stories, articles, scripts
- **Code generation**: Functions, algorithms, test cases

**2. Tasks with Clear Evaluation Criteria**

The pattern works best when you can define what "good" looks like:

- **Code review**: Does it handle edge cases? Is it efficient? Is it readable?
- **Writing quality**: Is it clear? Is it engaging? Is it accurate?
- **Compliance checking**: Does it meet legal/policy requirements?

**3. Complex Outputs That Benefit from Iteration**

Some tasks are simply too complex to get right on the first try:

- **Research summaries**: Synthesizing multiple sources accurately
- **Translation**: Capturing nuance across languages
- **Data transformation**: Complex format conversions

**4. When You Have an "Expert Reviewer" Persona**

If you can clearly describe what an expert reviewer would look for, you can encode that expertise in your evaluator:

- "A senior software engineer reviewing for production readiness"
- "An editor at a major publication checking for style guide compliance"
- "A legal reviewer ensuring regulatory compliance"

### When NOT to Use Evaluator-Optimizer

**1. Simple, Single-Step Tasks**

If a task can be done well in one shot, adding an evaluation loop just wastes tokens:

- Answering factual questions
- Simple format conversions
- Basic calculations

**2. Highly Subjective Tasks Without Clear Criteria**

If you can't define what "better" means, the evaluator can't help:

- "Make this more creative" (too vague)
- "Write something that will go viral" (unpredictable)

**3. Time-Critical Applications**

Each iteration adds latency. If speed matters more than perfection:

- Real-time chat responses
- Live customer support
- Rapid prototyping

**4. Cost-Sensitive Applications**

Each iteration costs tokens. For high-volume, low-value outputs:

- Bulk content generation
- Disposable drafts
- Internal notes

### Decision Framework

Ask yourself these questions:

```
┌─────────────────────────────────────────────────────────────┐
│                  Should I Use Evaluator-Optimizer?          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Can I clearly define what "good" looks like?            │
│     NO  → Use single-shot generation                        │
│     YES → Continue                                          │
│                                                             │
│  2. Would a human expert review improve the output?         │
│     NO  → Use single-shot generation                        │
│     YES → Continue                                          │
│                                                             │
│  3. Is quality worth extra latency and cost?                │
│     NO  → Use single-shot generation                        │
│     YES → Continue                                          │
│                                                             │
│  4. Is the task complex enough to benefit from iteration?   │
│     NO  → Use single-shot generation                        │
│     YES → Use evaluator-optimizer ✓                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## The Generator-Evaluator Loop

Let's examine each component of the loop in detail.

### The Generator's Role

The generator creates or improves content. It receives:

1. **The original task**: What needs to be created
2. **Previous output** (if any): The last version produced
3. **Feedback** (if any): The evaluator's critique

The generator's system prompt typically includes:

- The task description and constraints
- Output format requirements
- Instructions to incorporate feedback (after the first iteration)

Here's a conceptual example of a generator prompt:

```
You are an expert technical writer. Your task is to write clear, 
accurate documentation.

{task_description}

{if feedback exists}
Previous version:
{previous_output}

Reviewer feedback:
{feedback}

Please revise the content to address the feedback while maintaining 
the strengths of the previous version.
{endif}
```

### The Evaluator's Role

The evaluator assesses content against criteria. It produces:

1. **Assessment**: Does the content meet standards?
2. **Specific feedback**: What needs improvement and why
3. **Decision**: Continue iterating or accept as final

The evaluator's system prompt typically includes:

- Evaluation criteria (specific and measurable)
- Scoring rubric or checklist
- Instructions for providing actionable feedback

Here's a conceptual example of an evaluator prompt:

```
You are a senior editor reviewing technical documentation.

Evaluate the following content against these criteria:
1. Accuracy: Are all technical details correct?
2. Clarity: Can a beginner understand this?
3. Completeness: Are all necessary topics covered?
4. Structure: Is the content well-organized?

For each criterion, score 1-5 and explain your reasoning.

If ALL criteria score 4 or higher, respond with:
{"status": "approved", "feedback": "..."}

Otherwise, respond with:
{"status": "needs_revision", "feedback": "..."}

Be specific about what needs improvement and why.
```

### The Feedback Loop

The magic happens in how feedback flows:

```
Iteration 1:
  Generator → Creates initial draft
  Evaluator → "Good structure, but examples are unclear. Score: 3/5"

Iteration 2:
  Generator → Revises with clearer examples
  Evaluator → "Examples improved! But missing error handling. Score: 3.5/5"

Iteration 3:
  Generator → Adds error handling section
  Evaluator → "Comprehensive and clear. Score: 4.5/5. Approved!"
```

Each iteration builds on the previous one, with feedback guiding specific improvements.

## Designing Effective Evaluation Criteria

The quality of your evaluator-optimizer system depends entirely on your evaluation criteria. Vague criteria produce vague feedback, which produces vague improvements.

### Characteristics of Good Criteria

**1. Specific and Measurable**

```
# Bad: Vague
"Is the writing good?"

# Good: Specific
"Does each paragraph have a clear topic sentence?"
"Are all code examples syntactically correct?"
"Is the reading level appropriate for beginners (Flesch-Kincaid grade 8 or below)?"
```

**2. Actionable**

The feedback should tell the generator exactly what to fix:

```
# Bad: Not actionable
"The introduction needs work."

# Good: Actionable
"The introduction should start with a problem statement that 
resonates with the reader, not a definition. Lead with why 
this matters before explaining what it is."
```

**3. Prioritized**

Not all criteria are equally important. Rank them:

```
Critical (must fix):
- Factual accuracy
- Security vulnerabilities
- Breaking functionality

Important (should fix):
- Code readability
- Documentation completeness
- Edge case handling

Nice to have (fix if time permits):
- Variable naming
- Comment formatting
- Whitespace consistency
```

**4. Independent**

Each criterion should be evaluable on its own:

```
# Bad: Overlapping
"Is it clear?" and "Is it easy to understand?"

# Good: Independent
"Is the vocabulary appropriate for the audience?"
"Are complex concepts introduced with examples?"
"Is the logical flow easy to follow?"
```

### Evaluation Rubrics

A rubric provides consistent scoring across iterations:

```
CLARITY RUBRIC:

5 - Exceptional:
    - Every concept is immediately understandable
    - Examples illuminate rather than confuse
    - No jargon without explanation

4 - Good:
    - Most concepts are clear
    - Examples are helpful
    - Minimal unexplained jargon

3 - Adequate:
    - Core concepts are understandable
    - Some examples help
    - Some jargon needs explanation

2 - Needs Work:
    - Several concepts are confusing
    - Examples may confuse more than help
    - Significant unexplained jargon

1 - Poor:
    - Difficult to understand overall
    - Examples are missing or unhelpful
    - Heavy use of unexplained jargon
```

### Domain-Specific Criteria

Different domains require different evaluation focuses:

**Code Generation:**
```
- Correctness: Does it produce expected output?
- Efficiency: Is the time/space complexity appropriate?
- Readability: Can another developer understand it?
- Safety: Are inputs validated? Errors handled?
- Style: Does it follow language conventions?
```

**Marketing Copy:**
```
- Hook: Does the first line grab attention?
- Benefit-focused: Does it emphasize customer value?
- Call-to-action: Is the next step clear?
- Tone: Does it match brand voice?
- Length: Is it appropriate for the channel?
```

**Technical Documentation:**
```
- Accuracy: Are all facts correct?
- Completeness: Are all features documented?
- Examples: Is every concept illustrated?
- Structure: Can users find what they need?
- Currency: Is it up to date?
```

## Knowing When to Stop

An evaluator-optimizer loop without proper stopping conditions can run forever, burning tokens without meaningful improvement. Let's explore how to terminate gracefully.

### Stopping Conditions

**1. Quality Threshold Met**

The most desirable stopping condition—the output is good enough:

```python
# Conceptual stopping condition
if all_criteria_score >= 4:
    return "approved"
```

**2. Maximum Iterations Reached**

A safety limit prevents runaway loops:

```python
MAX_ITERATIONS = 5  # Rarely need more than this

for iteration in range(MAX_ITERATIONS):
    output = generator.generate(task, feedback)
    evaluation = evaluator.evaluate(output)
    
    if evaluation.status == "approved":
        break
    feedback = evaluation.feedback

# Return best output even if not fully approved
```

**3. Diminishing Returns**

When improvements become marginal, stop iterating:

```python
# Track scores across iterations
scores = [3.0, 3.5, 3.7, 3.75, 3.76]

# If improvement is less than threshold, stop
if scores[-1] - scores[-2] < 0.1:
    print("Diminishing returns detected, stopping")
```

**4. Convergence Detection**

When feedback becomes repetitive, further iteration won't help:

```python
# If evaluator gives same feedback twice, stop
if current_feedback == previous_feedback:
    print("Feedback converged, stopping")
```

### Setting the Right Maximum

How many iterations are enough? Consider:

| Task Complexity | Typical Iterations | Max Recommended |
|-----------------|-------------------|-----------------|
| Simple refinement | 1-2 | 3 |
| Standard tasks | 2-3 | 5 |
| Complex tasks | 3-4 | 7 |
| Critical tasks | 4-5 | 10 |

> **💡 Tip:** Start with a max of 3-5 iterations. Monitor actual iteration counts in production, and adjust if needed. Most tasks converge within 3 iterations if your criteria are well-designed.

### Handling Failure to Converge

Sometimes the generator simply can't meet the criteria. Plan for this:

```python
if iteration == MAX_ITERATIONS and not approved:
    # Options:
    # 1. Return best attempt with warning
    # 2. Escalate to human review
    # 3. Try different generator prompt
    # 4. Relax criteria slightly
    
    return {
        "output": best_output,
        "status": "best_effort",
        "message": f"Did not fully meet criteria after {MAX_ITERATIONS} iterations",
        "final_score": best_score
    }
```

## Evaluator-Optimizer Architecture

Before implementing (in the next chapter), let's visualize the complete architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EVALUATOR-OPTIMIZER SYSTEM                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                         INPUT                                   │ │
│  │  • Task description                                            │ │
│  │  • Evaluation criteria                                         │ │
│  │  • Max iterations                                              │ │
│  │  • Quality threshold                                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    GENERATOR LLM                                │ │
│  │                                                                 │ │
│  │  System Prompt:                                                 │ │
│  │  • Role and expertise                                          │ │
│  │  • Task-specific instructions                                  │ │
│  │  • Format requirements                                         │ │
│  │                                                                 │ │
│  │  Input: Task + Previous Output + Feedback                      │ │
│  │  Output: Generated/Revised Content                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    EVALUATOR LLM                                │ │
│  │                                                                 │ │
│  │  System Prompt:                                                 │ │
│  │  • Reviewer role and expertise                                 │ │
│  │  • Evaluation criteria + rubric                                │ │
│  │  • Output format (structured)                                  │ │
│  │                                                                 │ │
│  │  Input: Generated Content                                      │ │
│  │  Output: Score + Feedback + Decision                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    CONTROL LOGIC                                │ │
│  │                                                                 │ │
│  │  • Check if approved → Return output                           │ │
│  │  • Check iteration count → Stop if max reached                 │ │
│  │  • Check for convergence → Stop if no improvement              │ │
│  │  • Otherwise → Loop back to Generator with feedback            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                        OUTPUT                                   │ │
│  │  • Final content                                               │ │
│  │  • Approval status                                             │ │
│  │  • Iteration count                                             │ │
│  │  • Final evaluation scores                                     │ │
│  │  • Iteration history (for debugging)                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Input Handler:**
- Validates task description
- Parses evaluation criteria
- Sets defaults for missing parameters

**Generator:**
- Creates initial content (iteration 1)
- Incorporates feedback into revisions (iteration 2+)
- Maintains consistency across revisions

**Evaluator:**
- Applies criteria consistently
- Provides specific, actionable feedback
- Makes clear approve/revise decisions

**Control Logic:**
- Manages iteration count
- Detects convergence
- Handles edge cases (timeouts, errors)
- Tracks history for debugging

**Output Handler:**
- Packages final result
- Includes metadata (iterations, scores)
- Provides iteration history if requested

## Variations and Extensions

The basic pattern can be extended for specific needs:

### Multiple Evaluators

Use different evaluators for different criteria:

```
Generator → Content
    ├── Technical Evaluator → Accuracy feedback
    ├── Style Evaluator → Writing feedback  
    └── Compliance Evaluator → Policy feedback
        │
        ▼
    Aggregated Feedback → Generator
```

### Self-Evaluation

For simpler cases, the generator can evaluate its own work:

```
Generator → Create content → Self-evaluate → Revise if needed
```

This saves tokens but may miss blind spots.

### Human-in-the-Loop Evaluation

For high-stakes content, include human review:

```
Generator → Evaluator (LLM) → Human Review → Final Approval
```

### Parallel Generation with Evaluation

Generate multiple candidates, evaluate all, pick the best:

```
    ┌── Generator A ──┐
    ├── Generator B ──┼── Evaluator → Best output
    └── Generator C ──┘
```

## Common Pitfalls

Before we implement this pattern, let's review common mistakes:

### 1. Vague Evaluation Criteria

**Problem:** "Make it better" doesn't tell the generator what to improve.

**Solution:** Be specific: "Improve readability by breaking paragraphs longer than 5 sentences and replacing jargon with plain language."

### 2. Conflicting Criteria

**Problem:** "Be concise" and "Be comprehensive" pull in opposite directions.

**Solution:** Prioritize criteria and define acceptable tradeoffs: "Comprehensiveness takes priority over brevity, but no section should exceed 500 words."

### 3. Infinite Loops

**Problem:** The evaluator never approves, or criteria are impossible to meet.

**Solution:** Always set a maximum iteration count. Monitor for diminishing returns.

### 4. Over-Iteration

**Problem:** Spending 10 iterations to go from 95% to 97% quality.

**Solution:** Set appropriate quality thresholds. "Good enough" is often good enough.

### 5. Lost Context Across Iterations

**Problem:** The generator forgets earlier feedback or the original task.

**Solution:** Include the original task and key context in every iteration, not just the latest feedback.

## Practical Exercise

**Task:** Design an evaluator-optimizer system for improving product descriptions

Before implementing (which we'll do in Chapter 25), design the system on paper:

**Requirements:**
1. Define the task: What makes a good product description?
2. Create evaluation criteria: List 4-5 specific, measurable criteria
3. Design a rubric: Create a 1-5 scoring scale for each criterion
4. Set stopping conditions: When is a description "good enough"?
5. Plan for failure: What happens if the generator can't meet criteria?

**Deliverable:** A written design document answering each requirement.

**Hints:**
- Think about what makes you click "Buy" on a product page
- Consider both content (what it says) and form (how it's presented)
- Be specific enough that someone else could implement your criteria

**Example Criterion to Get You Started:**
```
BENEFIT CLARITY (1-5):
5 - Top 3 benefits are immediately clear within first 50 words
4 - Benefits are clear but take more than 50 words to convey
3 - Benefits are present but buried in features
2 - Benefits are vague or unclear
1 - Benefits are missing; only features listed
```

**Solution:** See `code/design_exercise.md`

## Key Takeaways

- **Evaluator-optimizer creates a feedback loop** between a generator (creates content) and an evaluator (critiques content), driving iterative improvement.

- **If humans can give useful feedback, LLMs can too.** This insight enables automated refinement workflows that were previously human-only.

- **Evaluation criteria quality determines system quality.** Specific, actionable, prioritized criteria produce meaningful improvements; vague criteria produce vague results.

- **Always include stopping conditions.** Maximum iterations, quality thresholds, and convergence detection prevent runaway loops and wasted tokens.

- **Match the pattern to the problem.** Use evaluator-optimizer for quality-critical tasks with clear criteria; use simpler approaches for routine tasks.

## What's Next

In Chapter 25, we'll implement a complete evaluator-optimizer system. You'll build a writing assistant that iteratively improves drafts based on configurable criteria. We'll create reusable classes for both generator and evaluator roles, implement robust stopping conditions, and see the pattern in action across multiple iterations.

The conceptual foundation you've built in this chapter—understanding the loop, designing criteria, and planning for edge cases—will make the implementation straightforward and maintainable.
