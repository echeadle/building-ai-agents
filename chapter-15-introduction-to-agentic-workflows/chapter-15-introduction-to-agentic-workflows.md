---
chapter: 15
title: "Introduction to Agentic Workflows"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 15: Introduction to Agentic Workflows

## Introduction

Congratulations—you've built something powerful. In Part 2, you constructed the **Augmented LLM**: an LLM enhanced with tools, system prompts, and structured output capabilities. This building block can answer questions, perform calculations, fetch weather data, and respond in predictable formats. It's genuinely useful.

But here's the thing: many real-world tasks are too complex for a single LLM call to handle well. Consider these scenarios:

- A user submits a customer support ticket. Depending on whether it's a billing issue, technical problem, or general inquiry, it needs completely different handling.
- You need to generate a marketing email, but it must be translated into five languages and checked for cultural appropriateness.
- A code review requires checking for security vulnerabilities, performance issues, *and* style violations—each requiring different expertise.
- A research task involves breaking down a question, gathering information from multiple sources, and synthesizing a coherent report.

Each of these tasks requires *multiple* LLM calls working together in coordinated ways. This is where **agentic workflows** come in.

In this chapter, we'll step back and look at the landscape of workflow patterns. You'll learn the five fundamental patterns that can handle almost any complex task, understand when to use each one, and—just as importantly—learn when *not* to use them at all.

## Learning Objectives

By the end of this chapter, you will be able to:

- Distinguish between workflows and autonomous agents
- Identify and describe the five core workflow patterns
- Analyze a task to determine which pattern (if any) is appropriate
- Recognize when simple prompts are sufficient and workflows are overkill
- Plan how to combine patterns for complex real-world applications

## Workflows vs. Agents: A Critical Distinction

Before diving into patterns, let's clarify a distinction that will guide everything in Parts 3 and 4 of this book.

### Workflows: You Define the Path

A **workflow** is a system where *you*, the developer, define how LLM calls are orchestrated. You decide:

- Which LLM calls happen
- In what order they execute  
- What conditions trigger which paths
- When the process terminates

The LLM executes individual steps, but the *control flow* is predetermined by your code. Think of it like a flowchart: you draw the boxes and arrows, and the LLM fills in the boxes.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Step 1    │────▶│   Step 2    │────▶│   Step 3    │
│  (LLM call) │     │  (LLM call) │     │  (LLM call) │
└─────────────┘     └─────────────┘     └─────────────┘
        │                                      │
        │         Control flow is              │
        └──────── predetermined ───────────────┘
```

### Agents: The LLM Decides the Path

An **agent** is a system where *the LLM itself* directs the control flow. The LLM decides:

- What action to take next
- Whether to use a tool or respond directly
- When to continue or stop
- How to adapt when things don't go as expected

You provide the tools and constraints, but the LLM navigates through the problem space autonomously.

```
┌─────────────────────────────────────────────────┐
│                  Agent Loop                      │
│  ┌──────────┐                                   │
│  │  Think   │◀────────────────────┐             │
│  └────┬─────┘                     │             │
│       │                           │             │
│       ▼                           │             │
│  ┌──────────┐    ┌──────────┐     │             │
│  │  Decide  │───▶│   Act    │─────┘             │
│  └────┬─────┘    └──────────┘                   │
│       │                                         │
│       ▼                                         │
│  ┌──────────┐         LLM controls              │
│  │   Done   │         the flow                  │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

### Why This Distinction Matters

Workflows and agents exist on a spectrum, but understanding this distinction helps you make better design decisions:

| Aspect | Workflows | Agents |
|--------|-----------|--------|
| **Control** | Developer-defined | LLM-directed |
| **Predictability** | High—you know what will happen | Lower—behavior emerges from LLM decisions |
| **Debugging** | Easier—clear execution path | Harder—must trace LLM reasoning |
| **Flexibility** | Limited to designed paths | Can handle unexpected situations |
| **Risk** | Lower—bounded behavior | Higher—may do unexpected things |
| **Best for** | Well-understood, repeatable tasks | Open-ended, exploratory tasks |

In Part 3, we focus on **workflows**—patterns with predetermined control flow that you design. In Part 4, we'll tackle **autonomous agents** that direct their own behavior.

> **💡 Tip:** Start with workflows. They're easier to build, test, and debug. Graduate to agents only when you need the flexibility and can accept the added complexity.

## The Five Workflow Patterns

Anthropic's research on effective AI implementations identified five fundamental workflow patterns. These patterns aren't arbitrary—they emerge from the natural structure of complex tasks. Let's preview each one.

### Pattern 1: Prompt Chaining

**What it is:** Breaking a task into a sequence of steps, where each step's output becomes the next step's input.

**When to use it:** When a task has clear, sequential subtasks that benefit from focused attention at each stage.

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Step 1  │────▶│ Step 2  │────▶│ Step 3  │────▶│ Output  │
│Generate │     │ Review  │     │ Format  │     │         │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
```

**Example:** Writing a blog post → Editing for clarity → Translating to Spanish → Formatting as HTML

**Key benefit:** Each step can focus on one thing, improving quality while allowing validation between steps.

### Pattern 2: Routing

**What it is:** Classifying an input and directing it to a specialized handler.

**When to use it:** When different types of inputs require fundamentally different processing.

```
                    ┌──────────────┐
                    │  Classifier  │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Handler  │    │ Handler  │    │ Handler  │
    │    A     │    │    B     │    │    C     │
    └──────────┘    └──────────┘    └──────────┘
```

**Example:** Customer message → Classify as billing/technical/general → Route to specialized handler

**Key benefit:** Specialized handlers can be optimized for their specific task, improving overall quality.

### Pattern 3: Parallelization

**What it is:** Running multiple LLM calls simultaneously, then aggregating results.

**When to use it:** When subtasks are independent (sectioning) or when you want multiple perspectives on the same input (voting).

```
Sectioning:                          Voting:
                                     
     ┌─────────┐                          ┌─────────┐
     │ Task A  │                          │ Judge 1 │
     └────┬────┘                          └────┬────┘
          │                                    │
Input ────┼──── ┌─────────┐ ──── Aggregate    Same ──── ┌─────────┐ ──── Aggregate
          │     │ Task B  │                   Input     │ Judge 2 │
          │     └────┬────┘                             └────┬────┘
          │          │                                       │
     ┌────┴────┐     │                          ┌─────────┐  │
     │ Task C  │─────┘                          │ Judge 3 │──┘
     └─────────┘                                └─────────┘
```

**Example (Sectioning):** Analyze code for security, performance, and style simultaneously

**Example (Voting):** Have three models assess if content is appropriate, use majority vote

**Key benefit:** Reduces latency (parallelization) and/or increases confidence (voting).

### Pattern 4: Orchestrator-Workers

**What it is:** A central LLM dynamically breaks down a task and delegates to worker LLMs.

**When to use it:** When you can't predict in advance what subtasks are needed.

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│   "Given this task, I need to: A, B, C, D..."       │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │Worker A │    │Worker B │    │Worker C │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              ┌─────────────┐
              │ Synthesize  │
              └─────────────┘
```

**Example:** "Research the causes of the 2008 financial crisis" → Orchestrator identifies subtopics → Workers research each → Orchestrator synthesizes findings

**Key benefit:** Handles complex, unpredictable tasks by letting an LLM do the planning.

### Pattern 5: Evaluator-Optimizer

**What it is:** One LLM generates output, another evaluates it, and the cycle repeats until quality is sufficient.

**When to use it:** When you have clear evaluation criteria and iterative refinement improves results.

```
┌───────────────────────────────────────────────────┐
│                                                   │
│    ┌───────────┐         ┌───────────┐           │
│    │ Generator │────────▶│ Evaluator │           │
│    └───────────┘         └─────┬─────┘           │
│          ▲                     │                 │
│          │                     │                 │
│          │    Feedback         │                 │
│          └─────────────────────┘                 │
│                                                   │
│              Repeat until satisfied               │
└───────────────────────────────────────────────────┘
```

**Example:** Generate essay draft → Evaluate for clarity, evidence, flow → Revise based on feedback → Re-evaluate → Continue until passing

**Key benefit:** Mimics human revision process, producing progressively better output.

## Choosing the Right Pattern

Selecting the right pattern isn't just about capability—it's about matching the pattern to your problem's structure. Here's a decision framework:

### Decision Tree

```
Is the task simple enough for one LLM call?
├── YES → Use a simple prompt (no workflow needed)
└── NO → Continue...

Can the task be broken into sequential steps?
├── YES → Can each step be validated before proceeding?
│         ├── YES → Use PROMPT CHAINING
│         └── NO → Consider simpler prompting first
└── NO → Continue...

Are there distinct categories that need different handling?
├── YES → Use ROUTING
└── NO → Continue...

Are there independent subtasks or need for multiple perspectives?
├── YES → Independent subtasks? → Use PARALLELIZATION (Sectioning)
│         Multiple perspectives? → Use PARALLELIZATION (Voting)
└── NO → Continue...

Is the task breakdown unpredictable at design time?
├── YES → Use ORCHESTRATOR-WORKERS
└── NO → Continue...

Does the task benefit from iterative refinement?
├── YES → Do you have clear evaluation criteria?
│         ├── YES → Use EVALUATOR-OPTIMIZER
│         └── NO → Define criteria first, then use pattern
└── NO → Reconsider if you need a workflow at all
```

### Pattern Selection Table

| Situation | Best Pattern | Why |
|-----------|--------------|-----|
| Multi-step content creation | Prompt Chaining | Each step focuses on one aspect |
| Customer support tickets | Routing | Different issues need different expertise |
| Code review | Parallelization (Sectioning) | Security, performance, style are independent |
| Content moderation | Parallelization (Voting) | Multiple perspectives increase confidence |
| Research tasks | Orchestrator-Workers | Can't predict needed research paths |
| Writing refinement | Evaluator-Optimizer | Clear criteria enable iteration |
| Translation pipeline | Prompt Chaining | Translate → Review → Localize |
| Form processing | Routing | Different form types, different handlers |

## When Simple Prompts Are Enough

Here's a truth that might surprise you: **most tasks don't need workflows**.

Before reaching for a pattern, ask yourself:

### Can a single, well-crafted prompt do the job?

A good prompt with clear instructions often outperforms a poorly designed workflow. Consider:

```python
# Sometimes this is all you need
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system="""You are an expert technical writer. 
    When given a topic, write a clear, well-structured explanation.
    Include:
    - An introduction that motivates the topic
    - Clear sections with examples
    - A summary of key points
    Format your response with markdown headers.""",
    messages=[
        {"role": "user", "content": f"Explain {topic} to intermediate developers."}
    ]
)
```

### Signs you DON'T need a workflow:

1. **The task is straightforward** — One clear objective, no branching logic needed
2. **Quality is acceptable** — A single LLM call produces good-enough results
3. **Speed matters more than perfection** — Workflows add latency
4. **You're exploring** — Start simple, add complexity when you understand the problem
5. **The cost doesn't justify it** — Workflows multiply API calls

### Signs you DO need a workflow:

1. **Single calls produce inconsistent quality** — Breaking it down helps focus
2. **Different inputs need fundamentally different handling** — Routing improves results
3. **The task has clearly separable subtasks** — Parallelization saves time
4. **You can't predict the steps in advance** — Orchestration provides flexibility
5. **Iteration measurably improves output** — Evaluation loops help

> **⚠️ Warning:** The biggest mistake is over-engineering. A simple prompt that works 90% of the time is often better than a complex workflow that works 95% of the time but takes 5x longer and costs 5x more.

## Combining Patterns

Real-world systems often combine multiple patterns. Here are common combinations:

### Routing + Specialized Workflows

Route inputs to different workflows, each optimized for its category:

```
Input
  │
  ▼
┌─────────────┐
│   Router    │
└──────┬──────┘
       │
  ┌────┴────┬────────────┐
  ▼         ▼            ▼
┌─────┐  ┌─────┐    ┌─────────────┐
│Chain│  │Chain│    │Eval-Optimize│
│  A  │  │  B  │    │    Loop     │
└─────┘  └─────┘    └─────────────┘
```

**Example:** Customer message → Route by type → Billing uses simple chain, complaints use evaluator-optimizer for careful response crafting.

### Orchestrator + Parallel Workers

The orchestrator delegates to workers that run in parallel:

```
┌─────────────┐
│ Orchestrator│
└──────┬──────┘
       │ Delegate
       ▼
  ┌────┴────┐
  ▼    ▼    ▼
┌──┐ ┌──┐ ┌──┐  ← Workers run in parallel
└─┬┘ └─┬┘ └─┬┘
  │    │    │
  └────┼────┘
       ▼
  ┌─────────┐
  │Synthesize│
  └─────────┘
```

**Example:** Research orchestrator identifies 5 subtopics, workers research each in parallel, orchestrator synthesizes.

### Chaining + Evaluation

Add evaluation gates between chain steps:

```
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│ Step 1 │────▶│Evaluate│────▶│ Step 2 │────▶│Evaluate│
└────────┘     └───┬────┘     └────────┘     └───┬────┘
                   │                              │
               ┌───┴───┐                      ┌───┴───┐
               │ Pass? │                      │ Pass? │
               └───────┘                      └───────┘
                   │                              │
              Retry if fail                 Retry if fail
```

**Example:** Generate content → Evaluate for accuracy → Translate → Evaluate for fluency.

## Practical Example: Analyzing the Decision

Let's work through a realistic example to see how you'd choose a pattern.

**Scenario:** You're building a system to process user feedback from your app. Users submit free-form text feedback.

### Step 1: Understand the task

What needs to happen?
- Understand what type of feedback it is (bug report, feature request, praise, complaint)
- Extract key information
- Generate an appropriate response
- Route to the right team

### Step 2: Ask the key questions

1. **Is one LLM call enough?** 
   - Probably not—we need classification, extraction, response generation, and routing.

2. **Are there distinct categories needing different handling?**
   - Yes! Bug reports need technical details extracted, feature requests need product context, complaints need careful responses.
   - **→ Routing is appropriate**

3. **Within each category, what's needed?**
   - Bug reports: Extract details → Validate completeness → Format for engineering
   - Feature requests: Extract request → Check for duplicates → Prioritize
   - Complaints: Draft response → Evaluate tone → Refine if needed
   - **→ Each route uses a different pattern**

### Step 3: Design the architecture

```
User Feedback
      │
      ▼
┌─────────────┐
│   Router    │  ← Classify feedback type
│ (LLM-based) │
└──────┬──────┘
       │
   ┌───┴───┬───────────┬───────────┐
   ▼       ▼           ▼           ▼
┌─────┐ ┌─────┐   ┌─────────┐  ┌──────────┐
│ Bug │ │Feat │   │Complaint│  │  Praise  │
│Chain│ │Chain│   │Eval-Opt │  │ (Simple) │
└─────┘ └─────┘   └─────────┘  └──────────┘
```

This combines:
- **Routing** at the top level
- **Prompt Chaining** for bugs and features
- **Evaluator-Optimizer** for complaints (tone matters!)
- **Simple prompt** for praise (just needs acknowledgment)

### Step 4: Consider if it's worth it

Before building this:
- How many feedback messages per day?
- What's the cost of mishandling feedback?
- Could a simpler system work "well enough"?

If you get 10 messages a day, a single well-crafted prompt might be fine. If you get 10,000, the investment in a proper workflow pays off.

## The Code Pattern We'll Use

Throughout Part 3, we'll build each pattern as a reusable class. Here's a preview of the interface style we'll follow:

```python
"""
Preview of the pattern interfaces we'll build.

Each pattern follows a similar structure:
- Clear initialization
- A main execute/run method
- Consistent error handling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkflowResult:
    """Standard result container for all workflow patterns."""
    success: bool
    output: Any
    metadata: dict  # Steps taken, time, cost, etc.


class WorkflowPattern(ABC):
    """Base class for all workflow patterns."""
    
    @abstractmethod
    def execute(self, input_data: Any) -> WorkflowResult:
        """Execute the workflow and return results."""
        pass


# In upcoming chapters, we'll implement:
# - PromptChain(WorkflowPattern)
# - Router(WorkflowPattern)  
# - ParallelWorkflow(WorkflowPattern)
# - Orchestrator(WorkflowPattern)
# - EvaluatorOptimizer(WorkflowPattern)
```

Each pattern chapter will provide:
1. A concept chapter explaining the pattern in depth
2. An implementation chapter with complete, runnable code
3. Practical exercises to reinforce learning

## Common Pitfalls

### 1. Over-Engineering from the Start

**The mistake:** Building a complex workflow before validating that you need one.

**The fix:** Always start with the simplest solution. Build a basic prompt, see where it fails, then add complexity to address specific failures.

### 2. Choosing the Wrong Pattern

**The mistake:** Using orchestrator-workers when simple chaining would do, or using chaining when routing is more appropriate.

**The fix:** Use the decision tree. Be honest about your task's structure. If subtasks are predictable, you don't need an orchestrator.

### 3. Ignoring Latency and Cost

**The mistake:** Building a 5-step workflow that takes 30 seconds and costs $0.50 per request when users expect instant responses.

**The fix:** Consider latency and cost constraints *before* designing. Use parallel execution where possible. Cache repeated operations.

## Practical Exercise

**Task:** Analyze three real-world scenarios and determine the best workflow pattern for each.

**Scenarios:**

1. **Email Triage System**
   - Incoming emails to a support inbox
   - Need to: classify priority, identify topic, route to correct department, generate acknowledgment
   
2. **Code Documentation Generator**
   - Given a Python file, generate docstrings for all functions
   - Each function is independent
   - Need high-quality, consistent documentation

3. **Essay Writing Assistant**  
   - Help students improve their essays
   - Need to provide feedback on thesis, structure, evidence, and writing quality
   - Students should receive actionable improvement suggestions

**Requirements:**
- For each scenario, identify the most appropriate pattern(s)
- Explain why you chose that pattern
- Sketch the workflow architecture (boxes and arrows)
- Identify any places where patterns might combine

**Hints:**
- Consider what's independent vs. sequential
- Think about whether the task structure is predictable
- Ask: "Does this need refinement loops?"

**Solution:** See `code/exercise_solutions.py`

## Key Takeaways

1. **Workflows vs. Agents:** Workflows have developer-defined control flow; agents have LLM-directed control flow. Start with workflows.

2. **Five Patterns:** Prompt Chaining, Routing, Parallelization, Orchestrator-Workers, and Evaluator-Optimizer cover most complex tasks.

3. **Pattern Selection:** Match the pattern to your problem structure. Use the decision tree to guide your choice.

4. **Start Simple:** Most tasks don't need workflows. A well-crafted single prompt is often sufficient.

5. **Combine Thoughtfully:** Real systems often combine patterns, but each addition adds complexity. Add only what you need.

6. **Consider Constraints:** Latency, cost, and reliability requirements should influence pattern choice.

## What's Next

In Chapter 16, we'll dive deep into our first pattern: **Prompt Chaining**. You'll learn how to break complex tasks into focused steps, design quality gates between steps, and implement a reusable chaining system. We'll build a practical example that generates, reviews, and translates content—demonstrating how each link in the chain makes the overall output better.

The pattern is simple, but the principles you'll learn apply to all the workflows we'll build. Let's start connecting those links.
