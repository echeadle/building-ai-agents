---
chapter: 42
title: "Project - Research Assistant Agent"
part: 6
date: 2025-01-15
draft: false
---

# Chapter 42: Project - Research Assistant Agent

## Introduction

You've learned the foundations, mastered the patterns, and built the infrastructure. Now it's time to build something real: a research assistant that can investigate topics, read sources, synthesize information, and produce comprehensive reports—all autonomously.

This isn't a toy example. The agent you'll build in this chapter can actually help you research topics, compare options, analyze trends, or investigate questions that require gathering information from multiple sources. It demonstrates what becomes possible when you combine the patterns from earlier chapters into a cohesive system.

Here's what makes a research assistant compelling: it performs a task that takes humans hours in a matter of minutes. Ask it to "compare the top three note-taking apps based on features, pricing, and user reviews," and it will search, read multiple sources, organize findings, identify patterns, and write a structured report—while you grab coffee.

This chapter walks through the complete implementation: requirements, design decisions, tool suite, workflow logic, and the final code. By the end, you'll have a fully functional research assistant and understand how to apply these patterns to build your own agents.

## Learning Objectives

By the end of this chapter, you will be able to:

-   Design a capstone project from requirements to implementation
-   Build a complete tool suite for research tasks (search, read, organize)
-   Implement multi-step research workflows with proper state management
-   Synthesize findings from multiple sources into coherent reports
-   Deploy a production-ready agent that solves real problems

## Project Requirements

Let's start by defining what our research assistant needs to do. Clear requirements guide every design decision.

### Functional Requirements

**The research assistant must:**

1. **Accept natural language research queries**

    - "What are the best practices for API rate limiting?"
    - "Compare React, Vue, and Svelte for a new project"
    - "Summarize recent developments in AI agent architectures"

2. **Search the web for relevant sources**

    - Use search APIs to find authoritative content
    - Evaluate relevance before reading full pages
    - Handle multiple searches if initial results are insufficient

3. **Read and extract information from web pages**

    - Fetch full content from URLs
    - Extract meaningful text (not just snippets)
    - Handle different content types (articles, documentation, blog posts)

4. **Organize findings as research progresses**

    - Keep notes on important facts and insights
    - Track which sources have been consulted
    - Build up knowledge incrementally

5. **Synthesize findings into a coherent report**

    - Structure information logically
    - Cite sources appropriately
    - Identify patterns and draw conclusions
    - Present findings in a clear, readable format

6. **Know when to stop researching**
    - Detect when sufficient information has been gathered
    - Avoid infinite search loops
    - Produce useful reports even with incomplete information

### Non-Functional Requirements

**The research assistant must also:**

-   **Be cost-effective**: Minimize unnecessary API calls
-   **Be observable**: Log decisions and progress
-   **Be reliable**: Handle errors gracefully (API failures, bad URLs, etc.)
-   **Be fast enough**: Complete typical research in 2-5 minutes
-   **Be secure**: Validate inputs, sanitize outputs

## Design Overview

Before writing code, let's design the system architecture. Good design makes implementation straightforward.

### The Research Loop

Research is fundamentally iterative:

```
1. Understand the question
2. Search for relevant sources
3. Read the most promising sources
4. Take notes on findings
5. Assess if more research is needed
   ↓ Yes? Go to step 2
   ↓ No? Go to step 6
6. Synthesize findings into a report
```

This maps naturally to an agentic loop where the LLM decides each next step.

### Tool Suite

Our agent needs three core tools:

**1. `web_search`**

-   Searches the web for a query
-   Returns titles, URLs, and snippets
-   The agent uses this to find sources

**2. `web_read`**

-   Fetches and extracts content from a URL
-   Returns the full text of the page
-   The agent uses this to read sources in detail

**3. `save_note`**

-   Saves a note about a finding
-   Organizes information as research progresses
-   The agent uses this to remember what it learned

### State Management

The agent maintains three pieces of state:

```python
{
    "research_question": "The original query",
    "notes": [
        {"source": "url", "finding": "what we learned"},
        ...
    ],
    "sources_read": ["url1", "url2", ...]
}
```

This state persists across iterations, allowing the agent to remember what it has already done.

### Workflow

The complete workflow:

```
START
  ↓
Understand research question
  ↓
┌─────────────────────────┐
│  Search for sources     │←─────┐
│  (web_search)           │      │
└──────────┬──────────────┘      │
           ↓                      │
┌─────────────────────────┐      │
│  Read promising sources │      │
│  (web_read)             │      │
└──────────┬──────────────┘      │
           ↓                      │
┌─────────────────────────┐      │
│  Save key findings      │      │
│  (save_note)            │      │
└──────────┬──────────────┘      │
           ↓                      │
    Enough info? ─────No──────────┘
           │
          Yes
           ↓
┌─────────────────────────┐
│  Synthesize report      │
└──────────┬──────────────┘
           ↓
          END
```

### Design Decisions

**Why this tool set?**

-   **Minimal but complete**: These three tools cover the full research workflow
-   **Composable**: Can be combined in flexible ways
-   **Reusable**: Useful beyond just this agent

**Why save notes incrementally?**

-   Prevents context window overflow on long research sessions
-   Helps the agent stay focused on key findings
-   Creates an intermediate representation for report generation

**Why let the LLM control the loop?**

-   The LLM can adapt based on what it finds
-   No rigid workflow steps that might not fit every query
-   The agent can decide when it has enough information

## Building the Tool Suite

Let's implement each tool. These will be real, working functions that interact with external APIs.

See the `code/` directory for the complete implementation of:

-   `web_search_tool.py` - Web search using Brave Search API
-   `web_read_tool.py` - Web page content extraction
-   `save_note_tool.py` - Research note management

## The Complete Research Assistant

The full implementation brings together:

-   The agentic loop from Chapter 27
-   Tool integration from Chapters 8-12
-   State management from Chapter 28
-   Proper error handling and observability

See `code/research_assistant.py` for the complete implementation.

## How It Works: A Walkthrough

Let's trace through what happens when you ask: "What are best practices for API rate limiting?"

**Iteration 1:**

-   Claude receives the query and decides to search first
-   Calls `web_search("API rate limiting best practices")`
-   Gets back 5 search results with titles, URLs, and snippets
-   Examines the results and identifies 2-3 promising sources

**Iteration 2:**

-   Claude calls `web_read(url)` on the first promising source
-   Gets back the full text of an article about rate limiting
-   Reads and understands the content

**Iteration 3:**

-   Claude calls `save_note()` to record key findings:
    -   "Token bucket algorithm is commonly used for rate limiting"
    -   "Rate limits should be communicated in response headers"
    -   etc.

**Iterations 4-8:**

-   Claude reads additional sources
-   Saves more notes on different aspects:
    -   Different rate limiting algorithms
    -   Implementation strategies
    -   Common pitfalls
    -   Real-world examples

**Iteration 9:**

-   Claude decides it has enough information
-   Doesn't call any tools
-   Instead, produces a comprehensive report synthesizing all findings

**Result:**
A well-structured report with:

-   Executive summary
-   Main rate limiting strategies
-   Implementation details
-   Best practices from multiple sources
-   Cited sources
-   Recommendations

The key insight: **Claude controls the entire research process**. It decides what to search for, which sources to read, what findings to save, and when to stop researching. You just provide the question.

## Practical Considerations

### Cost Management

Research can be expensive if not managed:

**Token usage:**

-   Each search: ~200 tokens (results are compact)
-   Each page read: 1,000-3,000 tokens (depending on content)
-   Each iteration: ~500-1,000 tokens (Claude's reasoning)
-   Final report generation: 2,000-4,000 tokens

A typical research session (10 iterations, 3 pages read):

-   ~15,000-20,000 tokens total
-   ~$0.30-$0.60 at current Claude pricing

**Cost optimization strategies:**

1. Set reasonable `max_iterations` limits
2. Truncate page content intelligently
3. Use cheaper models for simple research
4. Cache frequently-read pages

### Reliability Improvements

The basic implementation can be enhanced:

**Error handling:**

```python
def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
    try:
        # ... existing tool execution ...
    except Exception as e:
        # Return error message instead of crashing
        return f"Tool {tool_name} failed: {str(e)}"
```

**Loop detection:**

```python
# Track tool calls to detect loops
self.tool_history = []

def _detect_loop(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
    """Check if we're stuck in a loop."""
    call = (tool_name, json.dumps(tool_input, sort_keys=True))

    # Count occurrences
    count = self.tool_history.count(call)
    return count >= 3  # Same call 3+ times = loop
```

**Quality gates:**

```python
def _should_continue_research(self) -> bool:
    """Decide if more research is warranted."""
    # Stop if we have enough notes
    if len(self.notes.notes) >= 15:
        return False

    # Stop if we've read many sources
    if len(self.notes.sources_read) >= 8:
        return False

    return True
```

## Common Pitfalls

**1. Infinite Search Loops**

**Problem:** The agent keeps searching but never reads sources or produces a report.

**Why it happens:** The system prompt doesn't clearly guide the research process.

**Solution:** Be explicit in the system prompt:

```python
"After searching, you MUST read at least 2-3 sources before searching again.
Save notes as you read. After reading 5+ quality sources, produce your report."
```

**2. Reading Every Search Result**

**Problem:** The agent tries to read all 20 search results, making research slow and expensive.

**Why it happens:** No guidance on source selection.

**Solution:** Add selection criteria to the prompt:

```python
"Be selective about which sources to read. Prefer:
- Official documentation
- Well-known technical blogs
- Recent content
Skip: forums, old content, non-authoritative sources"
```

**3. Poor Note Quality**

**Problem:** Notes are too vague: "This source talks about rate limiting."

**Why it happens:** No examples of good notes in the prompt.

**Solution:** Provide examples:

```python
"Save specific, actionable findings. Good examples:
✓ 'Token bucket algorithm: allows burst traffic up to bucket size'
✓ 'Return 429 status code with Retry-After header'
✗ 'This article discusses rate limiting' (too vague)"
```

**4. Incomplete Reports**

**Problem:** Reports lack structure or depth.

**Why it happens:** No template or format specified.

**Solution:** Include a report template in the system prompt (as we did above).

## Practical Exercise

**Task:** Enhance the research assistant with parallel reading

Currently, the agent reads sources one at a time. Modify it to read multiple sources in parallel for faster research.

**Requirements:**

1. After searching, identify 2-3 sources to read
2. Read them all in parallel using threading or asyncio
3. Process all the content before the next iteration
4. Maintain the same note-taking workflow

**Hints:**

-   Use `concurrent.futures.ThreadPoolExecutor` for parallel requests
-   Modify the tool execution to batch `web_read` calls
-   Return all results together to Claude

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

-   **Capstone projects integrate multiple patterns**: The research assistant combines tools, agentic loops, state management, and workflow patterns into one cohesive system

-   **Real agents need real infrastructure**: Search APIs, web scraping, error handling, and cost management are essential for production systems

-   **The LLM controls the workflow**: Unlike rigid pipelines, the agent decides what to search, what to read, when to take notes, and when to stop

-   **State management enables complex tasks**: Persistent notes allow the agent to build knowledge across iterations without overwhelming the context window

-   **System prompts guide behavior**: Clear instructions, examples, and constraints in the prompt determine the agent's research quality

-   **Iteration limits prevent runaway costs**: Always cap iterations and build in termination conditions

## What's Next

You've built a complete, functional research assistant that can investigate real topics and produce useful reports. This demonstrates what's possible when you combine simple patterns into sophisticated systems.

In Chapter 43, we'll build another capstone project: a code analysis agent that can read, understand, and analyze codebases. This will show how agents can work with structured data and perform technical analysis tasks.

The patterns are the same, but the application is different—which is exactly the point. Master the patterns, and you can build agents for any domain.
