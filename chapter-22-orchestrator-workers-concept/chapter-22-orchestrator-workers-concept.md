---
chapter: 22
title: "Orchestrator-Workers - Concept and Design"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 22: Orchestrator-Workers - Concept and Design

## Introduction

Imagine you're managing a complex home renovation project. You don't know exactly what needs to be done until you inspect each room. The kitchen might need new cabinets, the bathroom might need re-tiling, and the living room might just need a fresh coat of paint. You can't create a detailed plan upfront—you need to assess, delegate to specialists, and then combine their work into a finished home.

This is exactly the problem the **orchestrator-workers** pattern solves in AI agents. Unlike prompt chaining (Chapter 16-17) where we predefined each step, or parallelization (Chapter 20-21) where we knew the subtasks in advance, orchestrator-workers handles situations where we *can't predict* what work needs to be done until we analyze the input.

In the previous chapters, we've built workflows that follow predetermined paths. Routing directs input to one of several known handlers. Chaining moves through a fixed sequence of steps. But what happens when the task itself determines what steps are needed? That's where orchestrator-workers becomes essential.

In this chapter, we'll explore the orchestrator-workers pattern conceptually—understanding when to use it, how to design the orchestrator and workers, and the tradeoffs involved. Chapter 23 will then implement these concepts in working code.

## Learning Objectives

By the end of this chapter, you will be able to:

- Explain what the orchestrator-workers pattern is and how it differs from other workflow patterns
- Identify use cases where orchestrator-workers is the right choice
- Design an effective orchestrator that can decompose tasks dynamically
- Create workers that are focused, independent, and composable
- Understand the tradeoffs between dynamic and static task breakdown
- Sketch an architecture for an orchestrator-workers system

## What Is the Orchestrator-Workers Pattern?

The orchestrator-workers pattern consists of two types of components:

1. **Orchestrator**: A central LLM that analyzes the task, breaks it down into subtasks, delegates work to specialized workers, and synthesizes the final result.

2. **Workers**: Specialized components (often LLMs with specific prompts or tools) that handle individual subtasks and return results to the orchestrator.

Here's the key insight: **the orchestrator decides at runtime what work needs to be done**. It doesn't follow a script—it analyzes the input and creates a plan dynamically.

```
┌─────────────────────────────────────────────────────────────┐
│                      USER REQUEST                           │
│            "Analyze this codebase for issues"               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│                                                             │
│  1. Analyze request                                         │
│  2. Identify what subtasks are needed                       │
│  3. Delegate to appropriate workers                         │
│  4. Collect and synthesize results                          │
└───────┬─────────────────┬─────────────────┬─────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   WORKER 1    │ │   WORKER 2    │ │   WORKER 3    │
│               │ │               │ │               │
│  Security     │ │  Performance  │ │  Code Style   │
│  Analysis     │ │  Analysis     │ │  Analysis     │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └────────────────┬┴─────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                           │
│                                                             │
│  Synthesize worker results into final response              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FINAL RESPONSE                           │
│         Comprehensive analysis with all findings            │
└─────────────────────────────────────────────────────────────┘
```

### Comparison with Other Patterns

Let's see how orchestrator-workers differs from patterns we've already learned:

| Pattern | Task Structure | Control Flow | Best For |
|---------|---------------|--------------|----------|
| **Prompt Chaining** | Fixed sequence | Predetermined | Known multi-step processes |
| **Routing** | Single handler selection | Based on classification | Categorizable requests |
| **Parallelization** | Known parallel tasks | Simultaneous execution | Predictable independent work |
| **Orchestrator-Workers** | Dynamic decomposition | LLM-directed | Unpredictable task structure |

The key distinction is **predictability**. If you know the subtasks ahead of time, use chaining or parallelization. If you don't—if the input determines what work is needed—use orchestrator-workers.

## When to Use Orchestrator-Workers

The orchestrator-workers pattern shines in specific scenarios. Let's explore when it's the right choice.

### Ideal Use Cases

**1. Complex Analysis Tasks**

When analyzing documents, code, or data where the scope of analysis depends on what's found:

```
User: "Review this contract and identify any issues."

Orchestrator thinks: "This is a software licensing contract. I need to:
- Check intellectual property clauses
- Review liability limitations  
- Analyze termination conditions
- Verify compliance requirements
The specific clauses present determine what analysis is needed."
```

**2. Research and Information Gathering**

When the research path depends on initial findings:

```
User: "Research the market opportunity for our new product."

Orchestrator thinks: "I need to:
- Identify the target market (depends on product type)
- Analyze competitors (depends on which market)
- Assess market size (depends on competitors found)
- Identify trends (depends on market identified)
Each step informs what the next steps should be."
```

**3. Content Creation with Multiple Components**

When creating content requires different specialized skills:

```
User: "Create a comprehensive report on climate change for our board."

Orchestrator thinks: "This report needs:
- Executive summary (synthesis skill)
- Data analysis section (analytical skill)
- Visualization recommendations (design skill)
- Action items (strategic skill)
The specific sections depend on board's needs and available data."
```

**4. Code Generation and Refactoring**

When the structure of the solution depends on the problem:

```
User: "Build a REST API for managing user accounts."

Orchestrator thinks: "I need to create:
- User model (depends on requirements)
- Authentication endpoints (depends on auth method)
- CRUD operations (depends on model)
- Validation logic (depends on operations)
The architecture emerges from the requirements."
```

### When NOT to Use Orchestrator-Workers

Not every problem needs this pattern. Avoid it when:

**Simple tasks**: If the task is straightforward, the orchestration overhead adds latency without benefit.

```
# Don't use orchestrator-workers for:
"What's the capital of France?"
"Translate 'hello' to Spanish."
"Calculate 15% of 200."
```

**Predictable structures**: If you always need the same subtasks, use chaining instead.

```
# Use chaining instead:
"Always: extract data → validate → transform → load"
```

**Independent parallel work**: If subtasks don't depend on analysis, use simple parallelization.

```
# Use parallelization instead:
"Translate this document into French, Spanish, and German."
```

**Real-time requirements**: Orchestration adds latency. For instant responses, simpler patterns work better.

> **Decision Rule**: Use orchestrator-workers when you genuinely can't know what subtasks are needed until you've analyzed the input. If you're tempted to hardcode the subtask list, a simpler pattern is probably better.

## The Orchestrator's Role

The orchestrator is the brain of the system. It has three primary responsibilities:

### 1. Planning: Understanding and Decomposing Tasks

The orchestrator first analyzes the request to understand what needs to be done:

```python
# Conceptual orchestrator planning prompt
PLANNING_PROMPT = """
You are a task orchestrator. Analyze the user's request and break it down
into specific subtasks that can be delegated to specialized workers.

For each subtask, specify:
1. task_id: A unique identifier
2. task_type: The type of worker needed
3. task_description: What specifically needs to be done
4. dependencies: Which other tasks must complete first (if any)
5. priority: How important this task is (high/medium/low)

Available worker types:
- researcher: Finds and summarizes information
- analyzer: Performs detailed analysis on data
- writer: Creates polished content
- coder: Writes or reviews code
- reviewer: Checks work for quality and errors

User Request: {user_request}

Respond with a JSON array of subtasks.
"""
```

Good orchestrator planning exhibits these qualities:

- **Comprehensive**: Identifies all necessary subtasks
- **Specific**: Each subtask is concrete and actionable
- **Appropriately granular**: Not too broad, not too detailed
- **Dependency-aware**: Understands what must happen in what order

### 2. Delegating: Assigning Work to Workers

Once the plan exists, the orchestrator delegates tasks to workers:

```python
# Conceptual delegation logic
def delegate_tasks(plan, workers):
    """
    Assign each task in the plan to an appropriate worker.
    Handle dependencies by ordering execution correctly.
    """
    results = {}
    
    # Sort tasks by dependencies
    ordered_tasks = topological_sort(plan)
    
    for task in ordered_tasks:
        # Wait for dependencies
        dependency_results = {
            dep: results[dep] 
            for dep in task.dependencies
        }
        
        # Find appropriate worker
        worker = workers[task.task_type]
        
        # Execute task with context
        result = worker.execute(
            task.task_description,
            context=dependency_results
        )
        
        results[task.task_id] = result
    
    return results
```

Effective delegation requires:

- **Worker matching**: Choosing the right worker for each task
- **Context passing**: Giving workers the information they need
- **Dependency management**: Ensuring prerequisites complete first
- **Error handling**: Dealing with worker failures gracefully

### 3. Synthesizing: Combining Results

After workers complete their tasks, the orchestrator combines results:

```python
# Conceptual synthesis prompt
SYNTHESIS_PROMPT = """
You are synthesizing the results from multiple specialized workers
into a coherent final response.

Original Request: {user_request}

Worker Results:
{formatted_results}

Create a comprehensive response that:
1. Addresses the original request completely
2. Integrates insights from all workers
3. Resolves any conflicts between worker outputs
4. Presents information in a clear, organized manner

If any worker reported errors or incomplete results, acknowledge
this and explain what information may be missing.
"""
```

Good synthesis requires:

- **Completeness**: All relevant worker outputs are included
- **Coherence**: Results flow together logically
- **Conflict resolution**: Contradictions are addressed
- **Quality control**: Poor worker outputs are handled appropriately

## Worker Design Principles

Workers are the specialists that do the actual work. Designing effective workers is crucial for system success.

### Principle 1: Single Responsibility

Each worker should do one thing well:

```python
# Good: Focused workers
class SecurityAnalyzer:
    """Analyzes code for security vulnerabilities only."""
    pass

class PerformanceAnalyzer:
    """Analyzes code for performance issues only."""
    pass

# Bad: Unfocused worker
class CodeAnalyzer:
    """Analyzes code for security, performance, style, 
    documentation, testing, and architecture issues."""
    pass
```

Why single responsibility matters:
- Easier to optimize and test
- Clearer prompts lead to better outputs
- Simpler to replace or upgrade individual workers
- Failures are isolated to specific capabilities

### Principle 2: Self-Contained Execution

Workers should be independent—they receive input and produce output without needing external state:

```python
# Good: Self-contained worker
class ResearchWorker:
    def execute(self, query: str, context: dict) -> dict:
        """
        Receives everything needed in parameters.
        Returns complete result.
        """
        # All needed information comes from parameters
        search_results = self.search(query)
        summary = self.summarize(search_results, context)
        return {
            "query": query,
            "findings": summary,
            "sources": search_results.sources
        }

# Bad: Worker with external dependencies
class ResearchWorker:
    def execute(self):
        """
        Relies on self.current_query being set elsewhere.
        Stores results in self.results instead of returning.
        """
        # Depends on external state
        results = self.search(self.current_query)
        self.results = results  # Side effect instead of return
```

### Principle 3: Clear Interfaces

Workers should have predictable inputs and outputs:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class WorkerInput:
    """Standard input format for all workers."""
    task_description: str
    context: dict
    constraints: Optional[dict] = None

@dataclass  
class WorkerOutput:
    """Standard output format for all workers."""
    task_id: str
    status: str  # "success", "partial", "failed"
    result: dict
    errors: Optional[list] = None
    metadata: Optional[dict] = None
```

Clear interfaces enable:
- Consistent orchestrator logic
- Easy worker substitution
- Reliable error handling
- Simple testing

### Principle 4: Graceful Failure

Workers should fail informatively, not catastrophically:

```python
class RobustWorker:
    def execute(self, task: WorkerInput) -> WorkerOutput:
        try:
            result = self._do_work(task)
            return WorkerOutput(
                task_id=task.task_id,
                status="success",
                result=result
            )
        except PartialResultError as e:
            return WorkerOutput(
                task_id=task.task_id,
                status="partial",
                result=e.partial_result,
                errors=[str(e)]
            )
        except Exception as e:
            return WorkerOutput(
                task_id=task.task_id,
                status="failed",
                result={},
                errors=[f"Worker failed: {str(e)}"]
            )
```

## Dynamic vs. Static Task Breakdown

A key design decision is how much flexibility to give the orchestrator.

### Fully Dynamic Decomposition

The orchestrator decides everything at runtime:

```python
# Fully dynamic: Orchestrator has complete freedom
DYNAMIC_PROMPT = """
Analyze this request and create whatever subtasks you think
are necessary. You have complete flexibility in how you
break down the work.

Request: {request}
"""
```

**Pros:**
- Maximum flexibility
- Can handle novel situations
- Adapts to input complexity

**Cons:**
- Unpredictable costs (might create many tasks)
- Inconsistent behavior
- Harder to test and debug
- May miss important subtasks

### Constrained Dynamic Decomposition

The orchestrator chooses from predefined task types:

```python
# Constrained: Choose from predefined options
CONSTRAINED_PROMPT = """
Analyze this request and select which of the following
analyses are needed:

Available task types:
- security_scan: Check for security vulnerabilities
- performance_check: Identify performance issues  
- style_review: Check code style and formatting
- test_coverage: Analyze test coverage
- documentation: Review documentation completeness

For each task type you select, explain why it's needed
for this specific request.

Request: {request}
"""
```

**Pros:**
- Predictable costs
- Consistent behavior
- Known worker requirements
- Easier to test

**Cons:**
- Less flexible
- May not fit novel situations
- Requires upfront worker design

### Hybrid Approach (Recommended)

Combine constrained selection with dynamic refinement:

```python
# Hybrid: Constrained selection with dynamic details
HYBRID_PROMPT = """
Analyze this request using our standard task types, but
customize each task's specifics based on the input.

Standard task types (select all that apply):
- security_scan
- performance_check
- style_review
- test_coverage
- documentation

For each selected task, specify:
1. Why it's needed for this request
2. Specific focus areas based on the input
3. Priority level (high/medium/low)

You may also suggest ONE custom task if the standard
types don't cover something critical.

Request: {request}
"""
```

This hybrid approach gives you:
- Predictability with flexibility
- Cost control with adaptability
- Testable behavior with customization

> **💡 Tip**: Start with a constrained approach. Add flexibility only when you encounter situations that genuinely need it. Most use cases work fine with a fixed set of well-designed workers.

## Architecture Diagram

Here's a complete view of the orchestrator-workers architecture:

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATOR-WORKERS SYSTEM                  │
└──────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   Input     │
                              │  Request    │
                              └──────┬──────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                        PLANNING PHASE                          │  │
│  │                                                                │  │
│  │  • Analyze input request                                       │  │
│  │  • Identify required subtasks                                  │  │
│  │  • Determine dependencies                                      │  │
│  │  • Assign priorities                                           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                │                                      │
│                                ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                      TASK QUEUE                                │  │
│  │                                                                │  │
│  │  [Task 1: Security] ──depends on──▶ [Task 3: Report]          │  │
│  │  [Task 2: Performance] ──depends on──▶ [Task 3: Report]       │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │   WORKER 1   │ │   WORKER 2   │ │   WORKER N   │
           │              │ │              │ │              │
           │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │
           │  │ System │  │ │  │ System │  │ │  │ System │  │
           │  │ Prompt │  │ │  │ Prompt │  │ │  │ Prompt │  │
           │  └────────┘  │ │  └────────┘  │ │  └────────┘  │
           │  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │
           │  │ Tools  │  │ │  │ Tools  │  │ │  │ Tools  │  │
           │  └────────┘  │ │  └────────┘  │ │  └────────┘  │
           │              │ │              │ │              │
           │  Security    │ │ Performance  │ │  Custom      │
           │  Specialist  │ │ Specialist   │ │  Specialist  │
           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                  │                │                │
                  │   ┌────────────┴────────────┐   │
                  │   │                         │   │
                  ▼   ▼                         ▼   ▼
           ┌───────────────────────────────────────────┐
           │              RESULTS COLLECTION            │
           │                                           │
           │  {task_1: result_1, task_2: result_2, ...}│
           └─────────────────────┬─────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATOR                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                      SYNTHESIS PHASE                           │  │
│  │                                                                │  │
│  │  • Collect all worker results                                  │  │
│  │  • Resolve conflicts                                           │  │
│  │  • Integrate findings                                          │  │
│  │  • Generate final response                                     │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │    Final     │
                         │   Response   │
                         └──────────────┘
```

### Component Interactions

Let's trace through a concrete example:

```
User Request: "Analyze this Python file for potential issues."

1. PLANNING PHASE
   Orchestrator analyzes the file and determines:
   - Task 1: security_scan (high priority, no dependencies)
   - Task 2: performance_check (medium priority, no dependencies)
   - Task 3: style_review (low priority, no dependencies)
   - Task 4: synthesis (high priority, depends on 1, 2, 3)

2. DELEGATION PHASE  
   Tasks 1, 2, 3 execute in parallel (no dependencies)
   
   Worker 1 (Security): "Found SQL injection risk on line 45"
   Worker 2 (Performance): "Inefficient loop on line 23"
   Worker 3 (Style): "Missing docstrings in 3 functions"

3. SYNTHESIS PHASE
   Orchestrator combines results:
   
   "Analysis of example.py:
   
   🔴 Critical: Security vulnerability
   - SQL injection risk at line 45
   - Recommendation: Use parameterized queries
   
   🟡 Warning: Performance issue
   - Inefficient loop at line 23
   - Recommendation: Use list comprehension
   
   🟢 Info: Style suggestions
   - 3 functions missing docstrings
   - Recommendation: Add documentation"
```

## Design Considerations

### How Many Workers?

More workers means more specialization but also more complexity:

| Worker Count | Pros | Cons |
|--------------|------|------|
| 2-3 workers | Simple, fast, easy to manage | Limited specialization |
| 4-6 workers | Good balance, covers most needs | Moderate complexity |
| 7+ workers | High specialization | Complex orchestration, harder to test |

> **💡 Tip**: Start with 3-4 workers covering your core use cases. Add workers only when you identify clear gaps that can't be filled by existing workers.

### Worker Communication

Should workers communicate with each other?

**Direct Worker Communication** (Not Recommended):
```
Worker A ←──────→ Worker B
         sharing
         results
```

**Orchestrator-Mediated Communication** (Recommended):
```
Worker A ───results───→ Orchestrator ───context───→ Worker B
```

Keep all communication through the orchestrator. This:
- Maintains clear control flow
- Enables logging and debugging
- Prevents circular dependencies
- Simplifies testing

### Handling Long-Running Tasks

Some worker tasks take longer than others. Strategies:

**Sequential with Timeouts**:
```python
for task in tasks:
    result = worker.execute(task, timeout=30)
```

**Parallel with Aggregation**:
```python
results = await asyncio.gather(
    *[worker.execute(task) for task in parallel_tasks],
    return_exceptions=True
)
```

**Priority-Based Execution**:
```python
# Execute high-priority tasks first
high_priority = [t for t in tasks if t.priority == "high"]
low_priority = [t for t in tasks if t.priority == "low"]

critical_results = execute_all(high_priority)
# Return early if critical tasks provide enough information
if sufficient(critical_results):
    return synthesize(critical_results)
    
# Otherwise, continue with lower priority
all_results = critical_results + execute_all(low_priority)
```

## Designing Your Orchestrator Prompt

The orchestrator prompt is the most critical piece. Here's a template:

```python
ORCHESTRATOR_SYSTEM_PROMPT = """
You are an intelligent task orchestrator. Your role is to:

1. ANALYZE requests to understand what needs to be accomplished
2. DECOMPOSE complex requests into specific, manageable subtasks
3. DELEGATE subtasks to specialized workers
4. SYNTHESIZE worker results into coherent responses

## Available Workers

{worker_descriptions}

## Planning Guidelines

When creating a task plan:
- Identify ALL subtasks needed (err on the side of thoroughness)
- Specify dependencies between tasks (what must complete first?)
- Assign appropriate workers based on task requirements
- Set priorities (high/medium/low) based on importance
- Keep subtasks focused - one clear objective per task

## Output Format

For planning, respond with JSON:
{{
    "analysis": "Brief analysis of the request",
    "tasks": [
        {{
            "id": "task_1",
            "worker": "worker_type",
            "description": "Specific task description",
            "dependencies": [],
            "priority": "high"
        }}
    ]
}}

For synthesis, create a coherent response that integrates all worker 
outputs and directly addresses the original request.
"""
```

### Tips for Effective Orchestrator Prompts

**Be explicit about worker capabilities**:
```python
# Good: Specific capabilities
"""
researcher: Can search the web and summarize findings.
           Best for: fact-finding, background research, comparisons.
           Not for: code analysis, creative writing.
"""

# Bad: Vague capabilities  
"""
researcher: Does research.
"""
```

**Provide examples for complex cases**:
```python
"""
## Example

Request: "Compare Python and JavaScript for web development"

Task Plan:
- Task 1: researcher - "Research Python web frameworks (Django, Flask)"
- Task 2: researcher - "Research JavaScript web frameworks (React, Vue)"
- Task 3: analyzer - "Compare findings from tasks 1 and 2"
- Task 4: writer - "Create summary report" (depends on task 3)
"""
```

**Set clear constraints**:
```python
"""
## Constraints

- Maximum 5 tasks per request (prioritize the most important)
- Each task should complete in under 30 seconds
- Only use available workers (don't invent new types)
- If a request is too simple for decomposition, respond directly
"""
```

## Common Pitfalls

### Pitfall 1: Over-Orchestration

Not everything needs orchestration. Watch for this anti-pattern:

```python
# Over-orchestrated simple request
Request: "What's the weather in Tokyo?"

Bad Plan:
- Task 1: researcher - "Find weather services for Tokyo"
- Task 2: analyzer - "Compare weather data accuracy"
- Task 3: writer - "Format weather report"

Good Approach:
# Just use a weather tool directly - no orchestration needed
```

**Fix**: Add logic to bypass orchestration for simple requests.

### Pitfall 2: Unclear Task Boundaries

When tasks overlap, workers do redundant work:

```python
# Overlapping tasks
- Task 1: "Analyze the code for issues"
- Task 2: "Review the code quality"
- Task 3: "Check the code for problems"

# Clear boundaries
- Task 1: "Analyze for security vulnerabilities"
- Task 2: "Analyze for performance issues"
- Task 3: "Check style and formatting"
```

**Fix**: Define non-overlapping responsibilities for each worker.

### Pitfall 3: Missing Dependency Handling

Tasks with implicit dependencies cause incorrect results:

```python
# Missing dependency
- Task 1: "Summarize the research findings"  # But no research done yet!
- Task 2: "Research the topic"

# Correct dependencies
- Task 1: "Research the topic"
- Task 2: "Summarize the research findings" (depends on Task 1)
```

**Fix**: Always explicitly model task dependencies.

## Practical Exercise

**Task:** Design an orchestrator-workers system for a "Document Analyzer" that can analyze uploaded documents (contracts, reports, articles) and provide comprehensive summaries.

**Requirements:**

1. Design 3-5 specialized workers, each with:
   - A clear name and responsibility
   - Input requirements
   - Output format
   - Example use cases

2. Write the orchestrator system prompt that:
   - Describes each worker's capabilities
   - Explains how to decompose document analysis tasks
   - Includes an example task plan

3. Create a dependency diagram showing how tasks might flow for:
   - A legal contract
   - A technical report
   - A news article

**Hints:**
- Consider what different document types need (structure, content, implications)
- Think about which analyses can run in parallel vs. which have dependencies
- Remember: not every document needs every type of analysis

**Solution:** See `code/exercise_design.py` for a complete design document.

## Key Takeaways

- **Orchestrator-workers is for dynamic decomposition**—use it when you can't predict subtasks in advance

- **The orchestrator has three roles**: planning (decomposing tasks), delegating (assigning to workers), and synthesizing (combining results)

- **Workers should be focused and independent**—single responsibility, self-contained, clear interfaces, graceful failure

- **Balance flexibility and predictability**—start constrained, add flexibility only when needed

- **Keep communication through the orchestrator**—don't let workers communicate directly

- **Watch for over-orchestration**—simple requests don't need complex workflows

## What's Next

Now that you understand the orchestrator-workers pattern conceptually, Chapter 23 will implement it in working Python code. We'll build a research orchestrator that can dynamically delegate to specialized workers, handle dependencies, and synthesize comprehensive results. You'll see how the design principles from this chapter translate into actual code.
