---
chapter: 26
title: "From Workflows to Agents"
part: 4
date: 2025-01-15
draft: false
---

# Chapter 26: From Workflows to Agents

## Introduction

Congratulations! You've made it to a pivotal moment in your journey. Over the past ten chapters, you've mastered the five workflow patterns: chaining, routing, parallelization, orchestrator-workers, and evaluator-optimizer. These patterns are powerful, and honestly, they'll solve most problems you'll encounter in production.

But sometimes they're not enough.

Imagine you're building a system to help users debug their code. You could create a workflow that runs a fixed sequence: analyze the error, search documentation, suggest fixes. But what if the error requires checking the database schema first? What if the user's description is vague and you need to ask clarifying questions? What if the fix you suggest doesn't work and you need to try a different approach?

Workflows follow predetermined paths. **Agents choose their own path.**

This chapter marks your transition from building workflows to building true agents—systems where the LLM itself decides what to do next. This is both incredibly powerful and genuinely risky, which is why we're going to approach it thoughtfully.

## Learning Objectives

By the end of this chapter, you will be able to:

- Articulate the key difference between workflows and agents (LLM-directed control flow)
- Identify scenarios where workflows fall short and agents excel
- Describe the fundamental agent loop: perceive → think → act → repeat
- Evaluate the trust and autonomy tradeoffs when deploying agents
- Sketch the architecture of an autonomous agent system

## The Key Difference: Who's in Control?

Let's start with the fundamental distinction that separates workflows from agents.

### Workflows: You're in Control

In every workflow pattern we've built, **you** (the developer) determined the control flow:

```python
# Prompt Chaining - YOU define the sequence
def content_workflow(topic: str) -> str:
    draft = generate_draft(topic)      # Step 1: Always happens
    translated = translate(draft)       # Step 2: Always happens
    return translated                   # Fixed sequence
```

```python
# Routing - YOU define the routes
def route_query(query: str) -> str:
    category = classify(query)          # Classification happens
    if category == "billing":           # YOU define the branches
        return handle_billing(query)
    elif category == "technical":
        return handle_technical(query)
    # Routes are predetermined
```

```python
# Orchestrator-Workers - YOU define when to delegate
def orchestrate(task: str) -> str:
    subtasks = plan_subtasks(task)      # Planning happens once
    results = execute_workers(subtasks)  # Workers execute in parallel
    return synthesize(results)          # Synthesis happens once
    # The structure is fixed: plan → execute → synthesize
```

Even in our most sophisticated workflow, the orchestrator-workers pattern, the *structure* is fixed. Yes, the LLM decides *what* subtasks to create, but *you* decided that there would be exactly one planning phase, one execution phase, and one synthesis phase.

### Agents: The LLM is in Control

An agent is fundamentally different:

```python
# Agent - THE LLM defines what happens next
def agent_loop(initial_task: str) -> str:
    messages = [{"role": "user", "content": initial_task}]
    
    while True:
        # The LLM decides: respond? use a tool? ask a question?
        response = call_llm(messages)
        
        if response.wants_to_use_tool:
            # LLM chose to act
            result = execute_tool(response.tool_call)
            messages.append(tool_result(result))
            # Loop continues - LLM will decide what's next
            
        elif response.is_final_answer:
            # LLM chose to stop
            return response.content
            
        # What happens next? The LLM decides.
```

The critical insight: **In an agent, the LLM's output determines the control flow.** The LLM might:
- Use one tool, then stop
- Use five tools in sequence
- Use a tool, realize it needs different information, use another tool
- Decide it needs clarification and ask a question
- Determine the task is complete and provide a final answer

You don't know in advance how many iterations will occur or which path will be taken. The LLM is genuinely directing its own behavior.

### A Visual Comparison

**Workflow (Prompt Chain):**
```
[Input] → [Step 1] → [Step 2] → [Step 3] → [Output]
              ↓           ↓           ↓
         (predetermined path, fixed steps)
```

**Agent:**
```
[Input] → [Think] → [Act?] → [Think] → [Act?] → ... → [Output]
              ↓         ↓         ↓         ↓
         (LLM decides)  ↓    (LLM decides)  ↓
                   [Tool A]            [Tool B]
                        ↓                   ↓
                   [Result]            [Result]
                        ↓                   ↓
                   (back to Think)    (back to Think)
```

The agent's path emerges from its reasoning, not from your code structure.

## When Workflows Aren't Enough

Workflows are excellent—and you should prefer them when they work! But certain scenarios genuinely require agent-level autonomy.

### Scenario 1: Unknown Number of Steps

**Problem:** You're building a research assistant. The user asks: "What's the relationship between company X's stock price and their product announcements over the past year?"

**Why workflows struggle:** You don't know in advance:
- How many product announcements there were
- How many searches you'll need to find them all
- Whether the initial search strategy will work
- If you'll need to adjust your approach based on what you find

**Why agents excel:** An agent can:
1. Search for product announcements
2. Realize results are incomplete, try a different search
3. Find stock price data for relevant dates
4. Notice a discrepancy, investigate further
5. Decide when it has enough information to synthesize

The agent keeps going until *it* decides the task is complete.

### Scenario 2: Dynamic Error Recovery

**Problem:** You're building a code execution assistant. The user provides code that needs to run, but it might have bugs, missing dependencies, or environment issues.

**Why workflows struggle:** A fixed sequence like "run code → if error → suggest fix → done" doesn't handle:
- Multiple cascading errors
- Errors that require different types of fixes
- Situations where the fix introduces new problems

**Why agents excel:** An agent can:
1. Run the code
2. See an import error, install the package
3. Run again, see a syntax error, fix it
4. Run again, see a runtime error, debug it
5. Run again, success!

The agent adapts its strategy based on what actually happens.

### Scenario 3: Open-Ended Exploration

**Problem:** A user asks: "Help me understand why my application is slow."

**Why workflows struggle:** "Slow" could mean:
- Database queries are inefficient
- API calls are blocking
- Memory leaks are causing garbage collection
- The algorithm itself is O(n²)
- Network latency is high

You can't pre-determine which investigation path to follow.

**Why agents excel:** An agent can explore:
1. Check logs → nothing obvious
2. Profile the code → database seems slow
3. Analyze database queries → found a missing index
4. But wait, there's also an N+1 query problem
5. Continue investigating until performance is acceptable

### When to Stick with Workflows

Agents aren't always the answer. Prefer workflows when:

| Situation | Why Workflows Win |
|-----------|-------------------|
| Task has predictable steps | Workflows are faster and cheaper |
| High reliability is critical | Workflows have deterministic behavior |
| You need auditability | Workflow paths are easy to log and explain |
| Cost is a major concern | Workflows use fewer LLM calls |
| The task is well-understood | Why add complexity? |

> **💡 Tip:** Start with the simplest workflow that might work. Only upgrade to an agent when you've hit a wall that workflows can't overcome.

## The Agent Loop: Perceive → Think → Act → Repeat

Every agent, regardless of complexity, follows the same fundamental loop. Understanding this loop is essential for everything we'll build in the coming chapters.

### The Four Phases

```
┌─────────────────────────────────────────────────────────┐
│                     AGENT LOOP                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    ┌──────────┐                                        │
│    │ PERCEIVE │ ◄─── Gather information                │
│    └────┬─────┘      (user input, tool results,        │
│         │             environment state)                │
│         ▼                                              │
│    ┌──────────┐                                        │
│    │  THINK   │ ◄─── Reason about situation            │
│    └────┬─────┘      (what do I know? what should      │
│         │             I do next?)                       │
│         ▼                                              │
│    ┌──────────┐                                        │
│    │   ACT    │ ◄─── Take action OR provide answer     │
│    └────┬─────┘      (use tool, ask question,          │
│         │             give final response)              │
│         ▼                                              │
│    ┌──────────┐                                        │
│    │  REPEAT? │ ◄─── Is the task complete?             │
│    └────┬─────┘      (if no, loop back to PERCEIVE)    │
│         │                                              │
│         ▼                                              │
│    [COMPLETE]                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

Let's break down each phase:

### Phase 1: Perceive

The agent gathers all available information:
- The original user request
- Conversation history
- Results from previous tool calls
- Any relevant context or state

```python
def perceive(conversation_history: list, tool_results: list) -> list:
    """Compile all available information into messages."""
    messages = []
    
    # Original request and conversation
    messages.extend(conversation_history)
    
    # Results from any tools we've used
    for result in tool_results:
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", ...}]
        })
    
    return messages
```

### Phase 2: Think

The LLM processes the information and reasons about what to do:
- What is the current state of the task?
- What information am I missing?
- What actions are available to me?
- What should I do next?

This happens inside the LLM call—Claude reasons through the situation and decides on a course of action.

### Phase 3: Act

Based on its thinking, the agent takes action:
- **Use a tool:** Execute a function to get information or make changes
- **Ask for clarification:** Request more information from the user
- **Provide an answer:** Return the final response to the user

```python
def act(response) -> tuple[str, any]:
    """Execute the agent's chosen action."""
    if response.stop_reason == "tool_use":
        # Agent wants to use a tool
        tool_call = extract_tool_call(response)
        result = execute_tool(tool_call)
        return ("continue", result)
    
    elif response.stop_reason == "end_turn":
        # Agent is providing a final answer
        return ("complete", response.content[0].text)
```

### Phase 4: Repeat (or Complete)

The agent evaluates whether the task is complete:
- If **complete:** Return the final answer to the user
- If **not complete:** Go back to Phase 1 with new information

The key insight is that **the LLM itself signals completion** by choosing not to use any more tools and providing a final response.

### A Minimal Agent Loop in Code

Here's what the complete loop looks like in Python:

```python
def run_agent(user_request: str, tools: list, max_iterations: int = 10) -> str:
    """
    Run an agent loop until completion or max iterations.
    
    This is the fundamental pattern all agents follow.
    """
    messages = [{"role": "user", "content": user_request}]
    
    for iteration in range(max_iterations):
        # THINK: Let the LLM reason and decide
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )
        
        # Check what the agent decided to do
        if response.stop_reason == "end_turn":
            # Agent decided task is complete
            return response.content[0].text
        
        elif response.stop_reason == "tool_use":
            # ACT: Agent wants to use a tool
            
            # Add assistant's response (with tool call) to history
            messages.append({"role": "assistant", "content": response.content})
            
            # Execute each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            
            # PERCEIVE: Add results to messages for next iteration
            messages.append({"role": "user", "content": tool_results})
        
        # REPEAT: Loop continues with updated context
    
    # Safety: Max iterations reached
    return "I wasn't able to complete the task within the allowed steps."
```

This simple loop is the foundation of everything we'll build. In the coming chapters, we'll add sophistication, but the core pattern remains the same.

## Trust and Autonomy Considerations

Here's the honest truth about agents: **they can do things you didn't anticipate.** This is both their power and their danger.

### The Autonomy Spectrum

Think of agent autonomy as a spectrum:

```
Low Autonomy                                              High Autonomy
     │                                                           │
     ▼                                                           ▼
[Confirmation    [Checkpoint     [Bounded        [Fully
 Required]        Mode]           Actions]        Autonomous]
     │               │               │                │
Every action    Pause at key    Limited tool     Complete
needs human     milestones      access/scope     freedom
approval                                         to act
```

Where you position your agent on this spectrum depends on:

1. **Risk Level:** What's the worst thing that could happen?
2. **Reversibility:** Can mistakes be undone?
3. **Trust in the Model:** How reliable is the LLM's judgment?
4. **User Expectations:** What level of autonomy do users expect?

### Risk Assessment Framework

Before building an agent, assess the risks:

| Risk Category | Low Risk | High Risk |
|--------------|----------|-----------|
| **Data Access** | Read-only access | Write/delete access |
| **External Actions** | No side effects | Sends emails, makes purchases |
| **Scope** | Single, narrow task | Broad, open-ended tasks |
| **Reversibility** | All actions reversible | Permanent consequences |
| **Sensitivity** | Non-sensitive data | Financial, medical, personal data |

> **⚠️ Warning:** Start with low autonomy and increase it gradually as you build trust in your agent's behavior. It's much easier to grant more freedom than to recover from an agent that went off the rails.

### Practical Trust Patterns

Based on your risk assessment, implement appropriate guardrails:

**Pattern 1: Confirmation Required (Low Autonomy)**
```python
# Every significant action requires human approval
if agent_wants_to_send_email:
    if not get_user_confirmation(f"Send email to {recipient}?"):
        continue_without_sending()
```

**Pattern 2: Checkpoint Mode (Medium Autonomy)**
```python
# Agent can work autonomously but pauses at key points
if task_stage == "about_to_make_changes":
    show_user_summary_of_planned_changes()
    if not get_user_approval():
        revise_plan()
```

**Pattern 3: Bounded Actions (Medium-High Autonomy)**
```python
# Agent is autonomous but within strict limits
allowed_actions = ["read_file", "search", "calculate"]
if requested_action not in allowed_actions:
    deny_and_explain()
```

**Pattern 4: Fully Autonomous (High Autonomy)**
```python
# Agent can do anything with available tools
# Use only when: low stakes, reversible, trusted context
```

We'll implement all of these patterns in Chapter 31 (Human-in-the-Loop). For now, remember: **autonomy should be earned, not assumed.**

## Agent Architecture Overview

Let's look at the complete architecture of an agent system. This is what we'll build over the next several chapters.

### The Components

```
┌──────────────────────────────────────────────────────────────────┐
│                         AGENT SYSTEM                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                     AGENT CORE                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │   System     │  │   Agentic    │  │  Planning    │      │ │
│  │  │   Prompt     │  │    Loop      │  │   Module     │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    STATE MANAGEMENT                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │ Conversation │  │   Working    │  │    Long      │      │ │
│  │  │   History    │  │   Memory     │  │    Term      │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      TOOL LAYER                              │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ │
│  │  │  Tool 1  │ │  Tool 2  │ │  Tool 3  │ │   ...    │       │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    SAFETY & CONTROL                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │ │
│  │  │  Guardrails  │  │   Human in   │  │    Error     │      │ │
│  │  │              │  │   the Loop   │  │   Handling   │      │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**Agent Core:**
- **System Prompt:** Defines the agent's identity, capabilities, and constraints
- **Agentic Loop:** The perceive → think → act → repeat cycle
- **Planning Module:** Optional step-by-step reasoning before acting

**State Management:**
- **Conversation History:** All messages in the current session
- **Working Memory:** Current task context, intermediate results
- **Long-term Memory:** Persistent information across sessions (optional)

**Tool Layer:**
- The collection of tools available to the agent
- Tool registry for dynamic tool selection
- Input validation and output formatting

**Safety & Control:**
- **Guardrails:** Input/output validation, action constraints
- **Human in the Loop:** Approval gates, confirmation requests
- **Error Handling:** Graceful failure, retry logic, fallbacks

### How Components Interact

Here's a trace of a single agent iteration:

1. **User sends request** → Added to conversation history
2. **System prompt** → Loaded and prepended to messages
3. **Agentic loop begins** → LLM called with full context
4. **LLM decides to use a tool** → Tool call extracted
5. **Guardrails check** → Is this action allowed?
6. **Human approval** → (if required) User confirms
7. **Tool executed** → Result obtained
8. **State updated** → Tool result added to working memory
9. **Loop continues** → Back to step 3 with new context
10. **LLM decides task is complete** → Final answer returned

### What We'll Build

Over the next chapters, you'll implement each component:

| Chapter | Component | What You'll Build |
|---------|-----------|-------------------|
| 27 | Agentic Loop | The core execution cycle |
| 28 | State Management | Memory and context handling |
| 29 | Planning | Think-before-you-act patterns |
| 30 | Error Handling | Robust failure recovery |
| 31 | Human in the Loop | Approval and confirmation gates |
| 32 | Guardrails | Safety constraints and validation |
| 33 | Complete Agent | All components integrated |

By Chapter 33, you'll have a production-ready `Agent` class that incorporates everything.

## Common Pitfalls

### Pitfall 1: Reaching for Agents Too Early

**The mistake:** Building an agent when a simple workflow would suffice.

**Why it happens:** Agents feel more sophisticated and flexible.

**The consequence:** Higher costs, slower responses, less predictable behavior, harder to debug.

**The fix:** Always ask: "Can I solve this with a workflow?" If yes, use the workflow. Upgrade to an agent only when you've hit a genuine limitation.

### Pitfall 2: Insufficient Termination Conditions

**The mistake:** Not properly detecting when the agent should stop.

**Why it happens:** Relying solely on the LLM to decide when it's done.

**The consequence:** Agents that loop forever, run up massive API bills, or never provide an answer.

**The fix:** Implement multiple termination conditions:
```python
# Good: Multiple termination checks
if iteration >= max_iterations:
    return "Could not complete within allowed steps"
if cost_so_far >= budget_limit:
    return "Reached budget limit"
if response.stop_reason == "end_turn":
    return response.content[0].text
```

### Pitfall 3: Over-Trusting Agent Decisions

**The mistake:** Giving agents access to dangerous tools without oversight.

**Why it happens:** It's easier to build, and "it usually works."

**The consequence:** When it doesn't work, the consequences can be severe—deleted files, sent emails, exposed data.

**The fix:** Match autonomy to risk. High-stakes actions require human confirmation. Always have guardrails. Chapter 31 and 32 cover this in depth.

## Practical Exercise

**Task:** Analyze a workflow you've built (or design a hypothetical one) and determine whether it should be upgraded to an agent.

**Requirements:**

1. Choose a workflow pattern you've implemented (from Part 3) or imagine one
2. Write a brief description of what it does
3. List three scenarios where the workflow would handle the task well
4. List three scenarios where the workflow would struggle
5. For each struggling scenario, explain why an agent would handle it better
6. Make a recommendation: Should this be an agent? Why or why not?

**Example Analysis:**

*Workflow: Customer Service Router (from Chapter 19)*

*Handles well:*
- Simple billing questions → routes to billing handler
- Technical support requests → routes to tech handler
- General inquiries → routes to general handler

*Struggles with:*
- User starts with billing question, but root cause is technical
- Complex issue spanning multiple categories
- User needs back-and-forth troubleshooting

*Agent advantage:*
- Can switch strategies when initial approach fails
- Can combine knowledge from multiple domains
- Can engage in multi-step troubleshooting

*Recommendation:* Upgrade to agent for complex support scenarios, keep workflow for simple categorizable queries.

**Hints:**
- Think about edge cases in your workflow
- Consider what happens when the "happy path" fails
- Ask: "Does this require dynamic adaptation?"

**Solution:** See `code/workflow_analysis_template.py` for a structured template.

## Key Takeaways

- **Agents differ from workflows in one key way:** The LLM controls the flow, not your code
- **The agent loop is simple:** Perceive → Think → Act → Repeat until done
- **Use workflows first:** Agents are powerful but add complexity—use them only when needed
- **Trust must be earned:** Start with low autonomy and guardrails, increase freedom gradually
- **Architecture matters:** Good agent design separates concerns (core, state, tools, safety)

## What's Next

In Chapter 27, we'll implement the agentic loop in detail. You'll build a minimal but complete agent that can decide when to use tools, chain multiple actions together, and determine when a task is complete. This will be the foundation for everything else we build in Part 4.

The loop itself is deceptively simple. The complexity—and the opportunity—lies in the details we'll add: state management, planning, error handling, and safety controls. But first, let's make sure you deeply understand the basic loop.

Let's build your first true agent.
