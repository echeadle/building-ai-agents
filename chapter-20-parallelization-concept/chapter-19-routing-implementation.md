---
chapter: 20
title: "Parallelization - Concept and Design"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 20: Parallelization - Concept and Design

## Introduction

So far, every workflow pattern we've built has been sequential—one LLM call follows another, waiting patiently for each to complete before moving on. But what happens when you need to analyze a document from multiple angles simultaneously? Or process ten customer reviews at once? Or get three different perspectives on a security vulnerability?

**Parallelization** is the answer. It's a workflow pattern that runs multiple LLM calls at the same time, then combines the results. This simple concept opens up powerful possibilities: faster processing, higher confidence through multiple viewpoints, and the ability to tackle problems that would be painfully slow if done one step at a time.

In the previous chapters, we built prompt chaining (Chapter 16-17) for sequential tasks and routing (Chapter 18-19) for directing inputs to specialized handlers. Parallelization complements both patterns—you can parallelize steps within a chain or run multiple specialized handlers simultaneously.

In this chapter, you'll learn the conceptual foundations of parallelization. We'll explore the two core patterns—**sectioning** and **voting**—understand when parallelization helps (and when it doesn't), and design strategies for aggregating results. Chapter 21 will put these concepts into action with complete Python implementations.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain the difference between sectioning and voting patterns
- Identify when parallelization will (and won't) improve your workflow
- Design aggregation strategies appropriate to your use case
- Recognize the cost-speed-confidence trade-offs in parallel workflows
- Sketch architecture diagrams for parallel LLM systems

## Understanding Parallelization

### What Is Parallelization?

Parallelization means running multiple LLM calls concurrently rather than sequentially. Instead of:

```
Call 1 → Wait → Call 2 → Wait → Call 3 → Wait → Combine Results
```

You get:

```
┌─ Call 1 ─┐
│          │
├─ Call 2 ─┼─→ Combine Results
│          │
└─ Call 3 ─┘
   (simultaneous)
```

The key insight is that many tasks contain independent subtasks—work that doesn't depend on the results of other work. When you identify these independent pieces, you can process them in parallel.

### Why Parallelize?

Parallelization offers three distinct benefits, and different use cases emphasize different benefits:

**1. Speed (Latency Reduction)**

If you need to process 10 documents and each takes 3 seconds, sequential processing takes 30 seconds. Parallel processing? Still about 3 seconds (plus a small overhead). For user-facing applications, this difference transforms the experience.

**2. Confidence (Multiple Perspectives)**

Getting three different LLM responses to the same question and finding consensus can be more reliable than a single response. If two out of three agree that code has a security vulnerability, you can be more confident than if you only asked once.

**3. Capacity (Throughput)**

When you need to process large volumes—thousands of support tickets, millions of log entries—parallelization lets you leverage API rate limits effectively rather than waiting for each call to complete before starting the next.

### The Two Core Patterns

Anthropic's "Building Effective Agents" identifies two main parallelization patterns:

1. **Sectioning**: Divide a task into independent subtasks and process them in parallel
2. **Voting**: Run the same task multiple times and aggregate the results

Let's explore each in depth.

## Pattern 1: Sectioning

### The Concept

**Sectioning** (also called "fan-out" or "parallel decomposition") splits a large task into independent pieces, processes each piece separately, and then combines the results.

Think of it like organizing a research team: instead of one person reading an entire 500-page report, five people each read 100 pages and then share their findings.

### When to Use Sectioning

Sectioning works when your task has **naturally independent subtasks**:

- **Document analysis**: Different sections of a document can be analyzed independently
- **Multi-aspect evaluation**: Checking code for style, bugs, and performance simultaneously
- **Batch processing**: Processing multiple inputs that don't depend on each other
- **Multi-source research**: Gathering information from multiple sources at once

### Sectioning Architecture

```
                    ┌─────────────────┐
                    │   Input Task    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    Splitter     │
                    │ (divide work)   │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Worker 1   │   │  Worker 2   │   │  Worker 3   │
    │ (subtask A) │   │ (subtask B) │   │ (subtask C) │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Aggregator    │
                    │ (combine work)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Output   │
                    └─────────────────┘
```

### Design Considerations for Sectioning

**1. How Do You Split the Work?**

The splitting strategy depends on your task:

| Task Type | Splitting Strategy |
|-----------|-------------------|
| Long document | By section, chapter, or page |
| Multiple files | One worker per file |
| Multi-criteria evaluation | One worker per criterion |
| Batch of items | Divide items into groups |

**2. Are the Subtasks Truly Independent?**

This is crucial. If subtask B needs information from subtask A, you can't parallelize them. Ask yourself:

- Does any worker need output from another worker?
- Does order matter?
- Is there shared state that workers would conflict on?

If you answer "yes" to any of these, you may need a hybrid approach (some parallel, some sequential).

**3. How Will You Combine Results?**

The aggregation strategy must match your task:

- **Concatenation**: For document summaries, join section summaries together
- **Structured merge**: Combine JSON results into a single structure
- **Synthesis**: Use another LLM call to synthesize parallel results into a coherent whole

### Example: Multi-Aspect Code Review

Imagine reviewing code for three independent concerns:

```
                         ┌────────────────┐
                         │  Source Code   │
                         └───────┬────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
   ┌──────▼──────┐       ┌───────▼───────┐      ┌──────▼──────┐
   │  Security   │       │  Performance  │      │   Style     │
   │  Reviewer   │       │   Reviewer    │      │  Reviewer   │
   └──────┬──────┘       └───────┬───────┘      └──────┬──────┘
          │                      │                      │
          │ vulnerabilities      │ bottlenecks          │ issues
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                         ┌───────▼────────┐
                         │  Merge Into    │
                         │ Unified Report │
                         └────────────────┘
```

Each reviewer focuses on one concern, runs in parallel, and the results merge into a comprehensive report. This is faster than sequential review and ensures each aspect gets specialized attention.

## Pattern 2: Voting

### The Concept

**Voting** (also called "ensemble" or "majority vote") runs the same task multiple times—often with the same prompt or slight variations—and aggregates the responses to increase confidence.

Think of it like asking three doctors for a diagnosis: if all three agree, you're more confident than if you only asked one.

### When to Use Voting

Voting works when you need **higher confidence** or **reliability**:

- **High-stakes classification**: Security vulnerability detection, medical triage
- **Subjective evaluation**: Content quality assessment, sentiment analysis
- **Reducing randomness**: When LLM responses vary significantly
- **Catching edge cases**: Different "runs" might catch different issues

### Voting Architecture

```
                    ┌─────────────────┐
                    │   Input Task    │
                    │ (same for all)  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Voter 1    │   │  Voter 2    │   │  Voter 3    │
    │ (same task) │   │ (same task) │   │ (same task) │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           │ response A      │ response B      │ response C
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Aggregator    │
                    │ (voting logic)  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Answer   │
                    └─────────────────┘
```

### Voting Strategies

Different aggregation strategies suit different needs:

**1. Majority Vote**

The simplest strategy: pick the answer that most voters agree on.

```
Voter 1: "Yes, vulnerable"
Voter 2: "Yes, vulnerable"  
Voter 3: "No, safe"

Result: "Yes, vulnerable" (2 out of 3)
```

Best for: Binary or categorical decisions with clear answers.

**2. Unanimous Agreement**

Require all voters to agree; otherwise, flag for human review or take conservative action.

```
Voter 1: "Approve"
Voter 2: "Approve"
Voter 3: "Reject"

Result: No consensus → Escalate to human
```

Best for: High-stakes decisions where false positives are costly.

**3. Weighted Voting**

Some voters count more than others, perhaps based on prompt sophistication or model capability.

```
Voter 1 (weight 1.0): "Category A"
Voter 2 (weight 1.5): "Category B"
Voter 3 (weight 1.0): "Category A"

Weighted scores: A = 2.0, B = 1.5
Result: "Category A"
```

Best for: When you have evidence that some approaches are more reliable.

**4. Threshold-Based**

Require a minimum confidence level based on agreement.

```
5 voters: 3 say "Critical", 2 say "Minor"
Threshold: 70% agreement required

Result: 60% agreement → "Uncertain, needs review"
```

Best for: When you need calibrated confidence levels.

**5. Union (Collect All)**

Combine all unique findings rather than choosing one.

```
Voter 1: Found issues [A, B]
Voter 2: Found issues [B, C]
Voter 3: Found issues [A, D]

Result: All issues [A, B, C, D]
```

Best for: Finding problems where any voter's discovery is valuable.

### Voting Variations

**Same Prompt, Multiple Runs**

The simplest voting setup: identical prompts, relying on LLM sampling randomness (temperature > 0) to get different perspectives.

```python
# Conceptual example - not runnable code
prompts = [same_prompt, same_prompt, same_prompt]
responses = run_parallel(prompts)
final_answer = majority_vote(responses)
```

**Varied Prompts**

Use slightly different prompts to get genuinely different perspectives:

```python
# Conceptual example
prompts = [
    "Analyze this code for security issues. Think step by step.",
    "You are a security expert. Review this code for vulnerabilities.",
    "Check this code: what could an attacker exploit?"
]
responses = run_parallel(prompts)
final_answer = aggregate(responses)
```

**Varied Temperatures**

Use different temperature settings to balance creativity and consistency:

```python
# Conceptual example
configs = [
    {"prompt": same_prompt, "temperature": 0.0},  # Deterministic
    {"prompt": same_prompt, "temperature": 0.5},  # Balanced
    {"prompt": same_prompt, "temperature": 1.0},  # Creative
]
```

## When Parallelization Helps (And When It Doesn't)

### Parallelization Helps When...

**✅ Subtasks are genuinely independent**

The golden rule. If tasks don't share dependencies, parallelize them.

**✅ Latency is critical**

User-facing applications where waiting 30 seconds versus 3 seconds matters significantly.

**✅ You need higher confidence**

When a single LLM response isn't reliable enough for your use case.

**✅ You're processing at scale**

Batch processing thousands of items where sequential would take hours.

**✅ You have diverse subtasks**

Different types of analysis that benefit from specialized prompts.

### Parallelization Doesn't Help When...

**❌ Tasks have sequential dependencies**

If step 2 needs output from step 1, you can't parallelize them. Use prompt chaining instead.

**❌ You're already rate-limited**

If you're hitting API rate limits with sequential calls, parallel calls will hit those limits even faster (and more chaotically).

**❌ Results require deep integration**

If the final output requires understanding relationships between subtasks, parallel workers can't see each other's work. You might need a synthesis step.

**❌ Costs outweigh benefits**

Voting triples your API costs for potentially modest confidence gains. Make sure the trade-off is worth it.

**❌ The task is already fast enough**

Don't add complexity for marginal gains. If a single call takes 500ms and that's acceptable, keep it simple.

## Designing Aggregation Strategies

The aggregation step is where parallel results become useful output. Poor aggregation can waste all the benefits of parallelization.

### Aggregation for Sectioning

When you've split work across sections, you need to combine the pieces:

**1. Simple Concatenation**

Join results in order. Works for document summaries, section analyses.

```
Section 1 Summary + Section 2 Summary + Section 3 Summary
→ Full Document Summary
```

**2. Structured Merge**

Combine structured data (JSON, objects) into a unified structure.

```python
# Conceptual example
results = {
    "security": security_findings,
    "performance": performance_findings,
    "style": style_findings
}
```

**3. LLM Synthesis**

Use another LLM call to synthesize parallel results into coherent output.

```
Worker outputs → Synthesis prompt → Final cohesive summary
```

This is especially useful when section summaries need to flow together naturally or when insights from one section relate to another.

### Aggregation for Voting

When you have multiple answers to the same question:

**1. Counting Votes**

Count occurrences and pick the winner.

```python
# Conceptual example
from collections import Counter

def majority_vote(responses: list[str]) -> str:
    counts = Counter(responses)
    winner, count = counts.most_common(1)[0]
    return winner
```

**2. Confidence Scoring**

Report both the answer and confidence level.

```python
# Conceptual example
def vote_with_confidence(responses: list[str]) -> dict:
    counts = Counter(responses)
    total = len(responses)
    winner, count = counts.most_common(1)[0]
    return {
        "answer": winner,
        "confidence": count / total,
        "agreement": f"{count}/{total}"
    }
```

**3. Fallback Logic**

Define what happens when there's no clear winner.

```python
# Conceptual example
def vote_with_fallback(responses: list[str], threshold: float = 0.6) -> dict:
    counts = Counter(responses)
    total = len(responses)
    winner, count = counts.most_common(1)[0]
    
    if count / total >= threshold:
        return {"answer": winner, "status": "confident"}
    else:
        return {"answer": None, "status": "needs_human_review"}
```

## Architecture Diagrams: Putting It Together

### Combined Pattern: Sectioning + Voting

For maximum reliability, you can combine patterns. Here's a code review system that uses sectioning for different aspects and voting within each aspect:

```
                         ┌────────────────┐
                         │  Source Code   │
                         └───────┬────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
   ┌──────────┐           ┌──────────┐           ┌──────────┐
   │ Security │           │  Perf    │           │  Style   │
   │ Aspect   │           │  Aspect  │           │  Aspect  │
   └────┬─────┘           └────┬─────┘           └────┬─────┘
        │                      │                      │
   ┌────┼────┐            ┌────┼────┐            ┌────┼────┐
   │    │    │            │    │    │            │    │    │
   ▼    ▼    ▼            ▼    ▼    ▼            ▼    ▼    ▼
  V1   V2   V3           V1   V2   V3           V1   V2   V3
   │    │    │            │    │    │            │    │    │
   └────┼────┘            └────┼────┘            └────┼────┘
        │                      │                      │
   ┌────▼────┐            ┌────▼────┐            ┌────▼────┐
   │  Vote   │            │  Vote   │            │  Vote   │
   └────┬────┘            └────┬────┘            └────┬────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                        ┌──────▼──────┐
                        │   Merge     │
                        │   Report    │
                        └─────────────┘
```

This architecture:
- Sections the task into three aspects (security, performance, style)
- Votes within each aspect (3 voters each) for reliability
- Merges the voted results into a final report

### Pattern: Fan-Out, Fan-In with Synthesis

For research or analysis tasks where you need to synthesize findings:

```
                    ┌─────────────────┐
                    │ Research Query  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Planner     │
                    │ (identify areas)│
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │ Research  │       │ Research  │       │ Research  │
   │ Worker 1  │       │ Worker 2  │       │ Worker 3  │
   │(subtopic A)│      │(subtopic B)│      │(subtopic C)│
   └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
         │                   │                   │
         │ findings          │ findings          │ findings
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Synthesizer   │
                    │ (LLM combines)  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Coherent Report │
                    └─────────────────┘
```

The synthesizer is itself an LLM call that takes all worker findings and creates a unified, coherent output—connecting insights, resolving contradictions, and creating a narrative.

## The Trade-offs

Every architecture decision involves trade-offs. Here's how parallelization trades stack up:

### Speed vs. Cost

| Approach | Time | Cost (API calls) |
|----------|------|------------------|
| Sequential (3 tasks) | 3x | 3 calls |
| Parallel (3 tasks) | 1x | 3 calls |

Parallel is faster, same cost. Clear win for independent tasks.

### Confidence vs. Cost (Voting)

| Approach | Confidence | Cost |
|----------|------------|------|
| Single call | Baseline | 1x |
| 3-vote majority | Higher | 3x |
| 5-vote majority | Even higher | 5x |

More votes = more confidence, but linear cost increase. Diminishing returns after 3-5 votes.

### Complexity vs. Maintainability

| Approach | Complexity | Debugging Difficulty |
|----------|------------|---------------------|
| Sequential | Low | Easy |
| Parallel (simple) | Medium | Moderate |
| Parallel + Voting | High | Harder |

More parallelism means more moving parts. Make sure the benefits justify the complexity.

## Planning Your Parallel Workflow

Before implementing parallelization, answer these questions:

### 1. What Are the Independent Pieces?

Map out your task. Can you identify subtasks that don't depend on each other?

```
Task: "Analyze customer feedback"

Subtasks:
- Sentiment analysis ← Independent!
- Topic extraction ← Independent!
- Urgency classification ← Independent!
- Entity extraction ← Independent!

All four can run in parallel.
```

### 2. What Type of Aggregation Do You Need?

- **Simple merge**: Results are structured data that combines naturally
- **Synthesis needed**: Results need an LLM to create coherent output
- **Voting needed**: You want consensus, not combination

### 3. What Are Your Constraints?

- **Latency budget**: How fast must responses be?
- **Cost budget**: How much can you spend per request?
- **Rate limits**: What are your API limits?
- **Reliability requirements**: How confident do results need to be?

### 4. What's Your Failure Strategy?

- What if one parallel worker fails?
- Do you retry, skip, or fail the whole request?
- How do you handle partial results?

## Common Pitfalls

### 1. Parallelizing Dependent Tasks

**The mistake**: Assuming tasks are independent when they're not.

**Example**: Trying to parallelize "extract entities" and "summarize focusing on those entities"—the second task needs the first task's output.

**The fix**: Map dependencies before parallelizing. If task B references task A's output, they must be sequential.

### 2. Ignoring Rate Limits

**The mistake**: Firing off 100 parallel requests and immediately hitting rate limits.

**Example**: Processing 1000 documents with 1000 simultaneous API calls.

**The fix**: Use controlled concurrency. Limit parallel requests to a reasonable number (e.g., 10-20) and process in batches.

### 3. Poor Aggregation Design

**The mistake**: Parallelizing work but not thinking through how to combine results.

**Example**: Three workers analyze different document sections, but their summaries are just concatenated without considering flow or contradictions.

**The fix**: Design aggregation as carefully as you design the parallel work. Consider using an LLM synthesis step for complex combinations.

## Practical Exercise

**Task**: Design a parallel workflow (on paper or pseudocode) for the following scenario:

You're building a content moderation system that needs to check user-submitted posts for:
- Hate speech
- Spam/advertising
- Personal information exposure
- Copyright violations

**Requirements**:
1. Each check should run in parallel (they're independent)
2. If ANY check fails, the post should be flagged
3. For hate speech specifically, use 3-vote majority for higher confidence
4. Produce a final report listing all detected issues

**Deliverables**:
1. Draw an architecture diagram showing the workflow
2. Specify the aggregation strategy for each parallel group
3. Define what the final output structure looks like
4. Identify potential failure modes and how you'd handle them

**Hints**:
- Consider combining sectioning (different checks) with voting (hate speech confidence)
- Think about what happens if one worker fails—does the whole system fail?
- The final output needs to clearly indicate pass/fail and list specific issues

**Solution**: See `code/exercise_design.md` for a sample solution.

## Key Takeaways

- **Parallelization runs multiple LLM calls simultaneously** to save time, increase confidence, or both

- **Sectioning divides work into independent subtasks** that run in parallel, then combines results—ideal for multi-aspect analysis or batch processing

- **Voting runs the same task multiple times** and aggregates answers—ideal for high-stakes decisions where confidence matters

- **Not all tasks can be parallelized**—dependencies between subtasks require sequential execution

- **Aggregation strategy is critical**—simple merging, voting, or LLM synthesis depending on your needs

- **Parallelization trades cost for speed and/or confidence**—make sure the trade-off is worth it for your use case

- **Design before implementing**—map dependencies, choose aggregation strategies, and plan failure handling before writing code

## What's Next

In Chapter 21, we'll implement everything we've designed here. You'll learn:

- Python's `asyncio` for parallel API calls
- Implementing the sectioning pattern with real tools
- Implementing the voting pattern with aggregation logic
- Building a complete code review system that combines both patterns
- Error handling strategies for parallel workflows

The concepts you've learned in this chapter will become working code in the next. Let's build it.
