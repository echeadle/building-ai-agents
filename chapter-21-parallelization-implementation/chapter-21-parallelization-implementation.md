---
chapter: 21
title: "Parallelization - Implementation"
part: 3
date: 2025-01-15
draft: false
---

# Chapter 21: Parallelization - Implementation

## Introduction

In Chapter 20, we explored *why* parallelization matters and *when* to use it. You learned about sectioning (dividing work into independent subtasks) and voting (getting multiple perspectives on the same problem). Now it's time to build these patterns.

This chapter transforms concepts into working code. We'll start with the essentials of Python's `asyncio` module—just enough to run LLM calls concurrently. Then we'll implement both parallelization patterns and build a practical code review system that uses voting to detect security vulnerabilities.

By the end, you'll have reusable parallel workflow classes that can dramatically speed up your agents and improve their reliability through consensus.

## Learning Objectives

By the end of this chapter, you will be able to:

- Use Python asyncio to run multiple LLM calls concurrently
- Implement the sectioning pattern for independent parallel subtasks
- Implement the voting pattern for multiple perspectives on the same task
- Apply different aggregation strategies (majority vote, consensus, merge)
- Handle errors gracefully in parallel workflows
- Build a code review system that uses voting for vulnerability detection

## Python Asyncio Essentials

Before we can run LLM calls in parallel, you need to understand how Python's `asyncio` works. Don't worry—we'll cover just what's necessary for our purposes.

### Why Asyncio?

When you make an API call to Claude, most of the time is spent *waiting*—waiting for the network, waiting for Claude to generate a response. During this waiting time, your program could be doing other useful work, like making additional API calls.

**Synchronous code** (what we've written so far) waits for each operation to complete before starting the next:

```python
# Synchronous: ~9 seconds total (3 calls × 3 seconds each)
response1 = make_api_call()  # Wait 3 seconds
response2 = make_api_call()  # Wait 3 seconds  
response3 = make_api_call()  # Wait 3 seconds
```

**Asynchronous code** can start multiple operations and wait for them all together:

```python
# Asynchronous: ~3 seconds total (all 3 run concurrently)
response1, response2, response3 = await asyncio.gather(
    make_api_call(),
    make_api_call(),
    make_api_call()
)
```

### The Core Concepts

There are only four things you need to know:

**1. `async def` defines an asynchronous function (called a "coroutine"):**

```python
async def fetch_data():
    # This function can use await
    return "data"
```

**2. `await` pauses the function until an operation completes:**

```python
async def main():
    result = await fetch_data()  # Pause here until fetch_data returns
    print(result)
```

**3. `asyncio.gather()` runs multiple coroutines concurrently:**

```python
async def main():
    # All three run at the same time
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data()
    )
    # results is a list: [result1, result2, result3]
```

**4. `asyncio.run()` starts the async event loop from regular code:**

```python
# In your main script
if __name__ == "__main__":
    asyncio.run(main())
```

### The Anthropic SDK's Async Client

The Anthropic SDK provides an async client specifically for this purpose:

```python
import anthropic

# Synchronous client (what we've used so far)
client = anthropic.Anthropic()

# Asynchronous client (for parallel calls)
async_client = anthropic.AsyncAnthropic()
```

The async client has the same interface, but its methods are coroutines that you `await`:

```python
async def get_response(prompt: str) -> str:
    """Make an async API call to Claude."""
    response = await async_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

Let's see this in action with a complete example:

```python
"""
Demonstrating async API calls with the Anthropic SDK.
"""

import asyncio
import os
import time
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Create the async client
async_client = anthropic.AsyncAnthropic()


async def get_response(prompt: str, label: str) -> dict:
    """Make an async API call and return the result with timing."""
    start = time.time()
    
    response = await async_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    
    elapsed = time.time() - start
    return {
        "label": label,
        "response": response.content[0].text,
        "time": elapsed
    }


async def main():
    """Run three API calls in parallel."""
    prompts = [
        ("What is 2+2? Reply with just the number.", "math"),
        ("What color is the sky? Reply in one word.", "color"),
        ("Name a planet. Reply with just the name.", "planet")
    ]
    
    print("Starting parallel API calls...")
    start = time.time()
    
    # Run all calls concurrently
    results = await asyncio.gather(*[
        get_response(prompt, label) 
        for prompt, label in prompts
    ])
    
    total_time = time.time() - start
    
    # Display results
    for result in results:
        print(f"\n{result['label']}: {result['response'][:50]}")
        print(f"  Individual time: {result['time']:.2f}s")
    
    print(f"\nTotal wall-clock time: {total_time:.2f}s")
    sum_individual = sum(r['time'] for r in results)
    print(f"Sum of individual times: {sum_individual:.2f}s")
    print(f"Time saved: {sum_individual - total_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
```

When you run this, you'll see that the total time is roughly equal to the *longest* single call, not the sum of all calls. That's the power of parallelization.

## Building the Parallel Base Class

Before implementing specific patterns, let's create a base class that handles the common functionality all parallel workflows need:

```python
"""
Base class for parallel workflow patterns.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class ParallelResult:
    """Container for a single parallel task result."""
    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0


@dataclass 
class ParallelWorkflowResult:
    """Container for the complete parallel workflow result."""
    results: list[ParallelResult] = field(default_factory=list)
    aggregated_result: Any = None
    total_time: float = 0.0
    successful_count: int = 0
    failed_count: int = 0


class ParallelWorkflow(ABC):
    """
    Abstract base class for parallel workflow patterns.
    
    Subclasses implement specific patterns (sectioning, voting)
    by overriding the abstract methods.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        max_concurrent: int = 5
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _call_llm(
        self,
        messages: list[dict],
        system: str | None = None
    ) -> str:
        """Make a single async LLM call."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages
        }
        if system:
            kwargs["system"] = system
            
        response = await self.async_client.messages.create(**kwargs)
        return response.content[0].text
    
    async def _execute_task(
        self,
        task_id: str,
        task_data: Any
    ) -> ParallelResult:
        """
        Execute a single task with error handling and timing.
        
        Subclasses define what 'execute' means by implementing
        _process_task().
        """
        import time
        start = time.time()
        
        try:
            result = await self._process_task(task_id, task_data)
            return ParallelResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=time.time() - start
            )
        except Exception as e:
            return ParallelResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start
            )
    
    @abstractmethod
    async def _process_task(self, task_id: str, task_data: Any) -> Any:
        """Process a single task. Implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _aggregate_results(
        self, 
        results: list[ParallelResult]
    ) -> Any:
        """Aggregate results from parallel tasks. Implemented by subclasses."""
        pass
    
    async def run(self, tasks: list[Any]) -> ParallelWorkflowResult:
        """
        Execute all tasks in parallel and aggregate results.
        
        Args:
            tasks: List of task data to process
            
        Returns:
            ParallelWorkflowResult with individual and aggregated results
        """
        import time
        start = time.time()
        
        # Create task tuples with IDs
        task_items = [
            (f"task_{i}", task_data) 
            for i, task_data in enumerate(tasks)
        ]
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_execute(task_id: str, task_data: Any):
            async with semaphore:
                return await self._execute_task(task_id, task_data)
        
        # Execute all tasks
        results = await asyncio.gather(*[
            bounded_execute(task_id, task_data)
            for task_id, task_data in task_items
        ])
        
        # Aggregate results
        aggregated = await self._aggregate_results(results)
        
        # Build final result
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        return ParallelWorkflowResult(
            results=results,
            aggregated_result=aggregated,
            total_time=time.time() - start,
            successful_count=len(successful),
            failed_count=len(failed)
        )
```

This base class provides:

- **Async LLM calls** with the `_call_llm()` method
- **Error handling** that captures failures without crashing the whole workflow
- **Timing** for performance analysis
- **Concurrency limiting** with a semaphore to avoid overwhelming the API
- **Abstract methods** that subclasses implement for their specific pattern

## Implementing Sectioning

**Sectioning** divides a large task into independent subtasks that run in parallel. Each subtask handles a different portion of the work, and results are merged at the end.

Common use cases:
- Processing multiple documents simultaneously
- Analyzing different aspects of the same input
- Generating content for different sections

```python
"""
Sectioning pattern: parallel independent subtasks.

Chapter 21: Parallelization - Implementation
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class Section:
    """Defines a section to process in parallel."""
    name: str
    prompt_template: str
    system_prompt: str | None = None


@dataclass
class SectionResult:
    """Result from processing a single section."""
    name: str
    content: str
    success: bool
    error: str | None = None


class SectioningWorkflow:
    """
    Implements the sectioning pattern for parallel subtasks.
    
    Divides work into independent sections, processes them in parallel,
    and merges the results.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        max_concurrent: int = 5
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _process_section(
        self,
        section: Section,
        input_data: str
    ) -> SectionResult:
        """Process a single section with the input data."""
        try:
            # Format the prompt with the input
            prompt = section.prompt_template.format(input=input_data)
            
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            if section.system_prompt:
                kwargs["system"] = section.system_prompt
            
            response = await self.async_client.messages.create(**kwargs)
            
            return SectionResult(
                name=section.name,
                content=response.content[0].text,
                success=True
            )
        except Exception as e:
            return SectionResult(
                name=section.name,
                content="",
                success=False,
                error=str(e)
            )
    
    async def run(
        self,
        sections: list[Section],
        input_data: str,
        merge_results: bool = True
    ) -> dict:
        """
        Process all sections in parallel.
        
        Args:
            sections: List of Section definitions
            input_data: The input to process across all sections
            merge_results: Whether to combine results into final output
            
        Returns:
            Dictionary with individual results and optional merged output
        """
        import time
        start = time.time()
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_process(section: Section):
            async with semaphore:
                return await self._process_section(section, input_data)
        
        # Process all sections in parallel
        results = await asyncio.gather(*[
            bounded_process(section) for section in sections
        ])
        
        # Build output
        output = {
            "sections": {r.name: r for r in results},
            "successful": [r.name for r in results if r.success],
            "failed": [r.name for r in results if not r.success],
            "execution_time": time.time() - start
        }
        
        # Optionally merge results
        if merge_results:
            successful_results = [r for r in results if r.success]
            merged = "\n\n".join([
                f"## {r.name}\n\n{r.content}" 
                for r in successful_results
            ])
            output["merged"] = merged
        
        return output


# Example: Document analysis with parallel sections
async def analyze_document_example():
    """Analyze a document from multiple angles in parallel."""
    
    workflow = SectioningWorkflow()
    
    # Define the sections to analyze
    sections = [
        Section(
            name="Summary",
            prompt_template="Provide a 2-3 sentence summary of this text:\n\n{input}",
            system_prompt="You are a concise summarizer. Be brief and accurate."
        ),
        Section(
            name="Key Points",
            prompt_template="List the 3-5 most important points from this text:\n\n{input}",
            system_prompt="You extract key information as clear bullet points."
        ),
        Section(
            name="Sentiment",
            prompt_template="Analyze the overall sentiment and tone of this text:\n\n{input}",
            system_prompt="You are a sentiment analyst. Describe tone and emotion."
        ),
        Section(
            name="Questions",
            prompt_template="What are 3 questions a reader might have after reading this?\n\n{input}",
            system_prompt="You anticipate reader questions and curiosities."
        )
    ]
    
    # Sample document to analyze
    document = """
    The company announced record quarterly earnings today, exceeding analyst 
    expectations by 15%. Revenue grew 23% year-over-year, driven primarily by 
    strong performance in the cloud services division. However, the CEO noted 
    challenges in the hardware segment, which saw a 5% decline. Looking ahead, 
    management provided cautious guidance for the next quarter, citing 
    macroeconomic uncertainties and supply chain constraints. Despite these 
    concerns, the company plans to increase R&D spending by 20% to accelerate 
    product development in AI and machine learning.
    """
    
    print("Analyzing document in parallel...\n")
    result = await workflow.run(sections, document)
    
    print(f"Completed in {result['execution_time']:.2f}s")
    print(f"Successful sections: {result['successful']}")
    print(f"Failed sections: {result['failed']}")
    print("\n" + "="*60 + "\n")
    print(result['merged'])


if __name__ == "__main__":
    asyncio.run(analyze_document_example())
```

### How Sectioning Works

1. **Define sections**: Each section has a name, prompt template, and optional system prompt
2. **Process in parallel**: All sections receive the same input and run concurrently
3. **Collect results**: Each section returns independently
4. **Merge output**: Successful results are combined into a cohesive document

The key insight is that each section is *independent*—the summary doesn't need the sentiment analysis, and vice versa. This independence makes parallelization safe and effective.

## Implementing Voting

**Voting** runs the same task multiple times (possibly with different prompts or temperatures) and aggregates the results. This increases confidence through consensus.

Common use cases:
- Classification tasks where accuracy is critical
- Detecting issues that might be missed by a single pass
- Generating diverse options before selecting the best

```python
"""
Voting pattern: multiple perspectives on the same task.

Chapter 21: Parallelization - Implementation
"""

import asyncio
import os
from collections import Counter
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class Voter:
    """Defines a single voter configuration."""
    name: str
    system_prompt: str
    temperature: float = 1.0


@dataclass
class Vote:
    """A single vote from one voter."""
    voter_name: str
    decision: str
    confidence: str | None = None
    reasoning: str | None = None
    success: bool = True
    error: str | None = None


@dataclass
class VotingResult:
    """Aggregated voting results."""
    votes: list[Vote]
    winner: str | None
    vote_counts: dict[str, int]
    consensus: bool
    consensus_threshold: float
    execution_time: float


class VotingWorkflow:
    """
    Implements the voting pattern for parallel consensus.
    
    Runs the same task with multiple voters and aggregates
    their decisions to determine a consensus result.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        consensus_threshold: float = 0.6
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.consensus_threshold = consensus_threshold
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _get_vote(
        self,
        voter: Voter,
        prompt: str,
        options: list[str] | None = None
    ) -> Vote:
        """Get a single vote from a voter."""
        try:
            # Build the voting prompt
            if options:
                options_str = ", ".join(options)
                full_prompt = f"""{prompt}

You must choose exactly one of these options: {options_str}

Respond in this format:
DECISION: [your choice]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation]"""
            else:
                full_prompt = f"""{prompt}

Respond in this format:
DECISION: [your choice]
CONFIDENCE: [high/medium/low]  
REASONING: [brief explanation]"""
            
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=voter.temperature,
                system=voter.system_prompt,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Parse the response
            text = response.content[0].text
            decision = self._extract_field(text, "DECISION")
            confidence = self._extract_field(text, "CONFIDENCE")
            reasoning = self._extract_field(text, "REASONING")
            
            # Normalize decision to match options if provided
            if options and decision:
                decision = self._normalize_to_options(decision, options)
            
            return Vote(
                voter_name=voter.name,
                decision=decision or "UNKNOWN",
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return Vote(
                voter_name=voter.name,
                decision="ERROR",
                success=False,
                error=str(e)
            )
    
    def _extract_field(self, text: str, field: str) -> str | None:
        """Extract a field value from formatted response."""
        for line in text.split("\n"):
            if line.strip().upper().startswith(f"{field}:"):
                return line.split(":", 1)[1].strip()
        return None
    
    def _normalize_to_options(
        self, 
        decision: str, 
        options: list[str]
    ) -> str:
        """Normalize a decision to match one of the valid options."""
        decision_lower = decision.lower().strip()
        for option in options:
            if option.lower() in decision_lower or decision_lower in option.lower():
                return option
        return decision
    
    def _aggregate_votes(self, votes: list[Vote]) -> tuple[str | None, dict, bool]:
        """
        Aggregate votes and determine consensus.
        
        Returns:
            (winner, vote_counts, has_consensus)
        """
        # Filter successful votes
        valid_votes = [v for v in votes if v.success and v.decision != "UNKNOWN"]
        
        if not valid_votes:
            return None, {}, False
        
        # Count votes
        decisions = [v.decision for v in valid_votes]
        vote_counts = dict(Counter(decisions))
        
        # Find winner
        winner = max(vote_counts, key=vote_counts.get)
        winner_count = vote_counts[winner]
        
        # Check consensus
        consensus = (winner_count / len(valid_votes)) >= self.consensus_threshold
        
        return winner, vote_counts, consensus
    
    async def run(
        self,
        voters: list[Voter],
        prompt: str,
        options: list[str] | None = None
    ) -> VotingResult:
        """
        Run voting with all voters in parallel.
        
        Args:
            voters: List of Voter configurations
            prompt: The decision prompt
            options: Optional list of valid choices
            
        Returns:
            VotingResult with individual votes and consensus
        """
        import time
        start = time.time()
        
        # Get all votes in parallel
        votes = await asyncio.gather(*[
            self._get_vote(voter, prompt, options)
            for voter in voters
        ])
        
        # Aggregate
        winner, vote_counts, consensus = self._aggregate_votes(votes)
        
        return VotingResult(
            votes=votes,
            winner=winner,
            vote_counts=vote_counts,
            consensus=consensus,
            consensus_threshold=self.consensus_threshold,
            execution_time=time.time() - start
        )


# Example: Content moderation with voting
async def content_moderation_example():
    """Use voting to classify content with high confidence."""
    
    workflow = VotingWorkflow(consensus_threshold=0.6)
    
    # Define diverse voters with different perspectives
    voters = [
        Voter(
            name="strict_moderator",
            system_prompt="""You are a strict content moderator. 
            You err on the side of caution and flag anything 
            that could potentially be problematic.""",
            temperature=0.3
        ),
        Voter(
            name="balanced_moderator", 
            system_prompt="""You are a balanced content moderator.
            You carefully weigh context and intent when making decisions.""",
            temperature=0.5
        ),
        Voter(
            name="lenient_moderator",
            system_prompt="""You are a lenient content moderator.
            You focus on clear violations and give benefit of the doubt
            for ambiguous content.""",
            temperature=0.3
        ),
        Voter(
            name="context_analyst",
            system_prompt="""You analyze content with deep attention to context.
            Consider the full picture before making a judgment.""",
            temperature=0.7
        ),
        Voter(
            name="policy_expert",
            system_prompt="""You are an expert in content policies.
            You apply rules consistently and fairly.""",
            temperature=0.3
        )
    ]
    
    # Test content
    content = """
    I can't believe how stupid the new policy is. The people who made it 
    must have rocks for brains. We should organize a protest!
    """
    
    prompt = f"""Classify the following user content:

{content}

Is this content acceptable, needs_review, or should be rejected?"""
    
    options = ["acceptable", "needs_review", "rejected"]
    
    print("Running content moderation vote...\n")
    result = await workflow.run(voters, prompt, options)
    
    print(f"Execution time: {result.execution_time:.2f}s\n")
    print("Individual votes:")
    for vote in result.votes:
        status = "✓" if vote.success else "✗"
        print(f"  {status} {vote.voter_name}: {vote.decision} ({vote.confidence})")
        if vote.reasoning:
            print(f"      Reasoning: {vote.reasoning[:80]}...")
    
    print(f"\nVote counts: {result.vote_counts}")
    print(f"Winner: {result.winner}")
    print(f"Consensus reached: {result.consensus}")


if __name__ == "__main__":
    asyncio.run(content_moderation_example())
```

### How Voting Works

1. **Define voters**: Each voter has a different system prompt and potentially different temperature
2. **Parallel execution**: All voters process the same prompt simultaneously
3. **Normalize decisions**: Match responses to the valid options
4. **Count and aggregate**: Determine the winner and whether consensus was reached

The diversity among voters is intentional—different perspectives catch different issues and reduce the chance of systematic blind spots.

## Aggregation Strategies

Different tasks require different ways of combining parallel results. Here are the three main strategies:

### Majority Vote

Best for classification tasks with discrete options:

```python
def majority_vote(votes: list[str]) -> str:
    """Return the most common vote."""
    from collections import Counter
    counts = Counter(votes)
    return counts.most_common(1)[0][0]
```

### Consensus with Threshold

Best when you need high confidence:

```python
def consensus_vote(
    votes: list[str], 
    threshold: float = 0.7
) -> tuple[str | None, bool]:
    """
    Return winner only if it exceeds the threshold.
    
    Returns:
        (winner, reached_consensus)
    """
    from collections import Counter
    counts = Counter(votes)
    total = len(votes)
    
    winner, count = counts.most_common(1)[0]
    ratio = count / total
    
    if ratio >= threshold:
        return winner, True
    return winner, False
```

### Merge and Synthesize

Best for content generation where you want to combine multiple perspectives:

```python
async def merge_and_synthesize(
    results: list[str],
    async_client: anthropic.AsyncAnthropic,
    task_description: str
) -> str:
    """
    Use an LLM to intelligently merge multiple results.
    
    This is useful when parallel tasks generate content
    that should be synthesized rather than voted on.
    """
    combined = "\n\n---\n\n".join([
        f"Result {i+1}:\n{result}" 
        for i, result in enumerate(results)
    ])
    
    prompt = f"""You have received multiple results for the same task.
Task: {task_description}

Here are the results:

{combined}

Please synthesize these into a single, coherent response that:
1. Incorporates the best elements from each result
2. Resolves any contradictions thoughtfully
3. Maintains consistency in tone and style

Synthesized result:"""
    
    response = await async_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text
```

## Error Handling in Parallel Workflows

When running tasks in parallel, individual failures shouldn't crash the entire workflow. Here's a robust error handling pattern:

```python
"""
Error handling patterns for parallel workflows.

Chapter 21: Parallelization - Implementation
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum


class FailurePolicy(Enum):
    """How to handle failures in parallel execution."""
    IGNORE = "ignore"           # Continue with successful results only
    FAIL_FAST = "fail_fast"     # Fail immediately on first error
    RETRY = "retry"             # Retry failed tasks
    REQUIRE_ALL = "require_all" # Fail if any task fails


@dataclass
class TaskResult:
    """Result of a parallel task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: str | None = None
    attempts: int = 1


async def execute_with_retry(
    task_id: str,
    task_func: Callable,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> TaskResult:
    """
    Execute a task with automatic retries on failure.
    
    Uses exponential backoff between retries.
    """
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            result = await task_func()
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                attempts=attempt
            )
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                # Exponential backoff
                delay = retry_delay * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
    
    return TaskResult(
        task_id=task_id,
        success=False,
        error=last_error,
        attempts=max_retries
    )


async def parallel_with_policy(
    tasks: list[tuple[str, Callable]],
    policy: FailurePolicy = FailurePolicy.IGNORE,
    max_retries: int = 3
) -> list[TaskResult]:
    """
    Execute tasks in parallel with specified failure policy.
    
    Args:
        tasks: List of (task_id, async_function) tuples
        policy: How to handle failures
        max_retries: Retries for RETRY policy
        
    Returns:
        List of TaskResult objects
    """
    if policy == FailurePolicy.FAIL_FAST:
        return await _execute_fail_fast(tasks)
    elif policy == FailurePolicy.RETRY:
        return await _execute_with_retries(tasks, max_retries)
    else:
        # IGNORE or REQUIRE_ALL - execute all
        results = await _execute_all(tasks)
        
        if policy == FailurePolicy.REQUIRE_ALL:
            failures = [r for r in results if not r.success]
            if failures:
                raise RuntimeError(
                    f"{len(failures)} tasks failed: "
                    f"{[f.task_id for f in failures]}"
                )
        
        return results


async def _execute_all(
    tasks: list[tuple[str, Callable]]
) -> list[TaskResult]:
    """Execute all tasks, capturing any errors."""
    async def safe_execute(task_id: str, func: Callable):
        try:
            result = await func()
            return TaskResult(task_id=task_id, success=True, result=result)
        except Exception as e:
            return TaskResult(task_id=task_id, success=False, error=str(e))
    
    return await asyncio.gather(*[
        safe_execute(task_id, func) 
        for task_id, func in tasks
    ])


async def _execute_fail_fast(
    tasks: list[tuple[str, Callable]]
) -> list[TaskResult]:
    """Execute tasks, failing immediately on first error."""
    results = []
    pending = [
        asyncio.create_task(func(), name=task_id)
        for task_id, func in tasks
    ]
    
    try:
        # Wait for all, but propagate first exception
        completed = await asyncio.gather(*pending, return_exceptions=False)
        for (task_id, _), result in zip(tasks, completed):
            results.append(TaskResult(
                task_id=task_id, success=True, result=result
            ))
    except Exception as e:
        # Cancel remaining tasks
        for task in pending:
            if not task.done():
                task.cancel()
        raise
    
    return results


async def _execute_with_retries(
    tasks: list[tuple[str, Callable]],
    max_retries: int
) -> list[TaskResult]:
    """Execute tasks with automatic retries."""
    return await asyncio.gather(*[
        execute_with_retry(task_id, func, max_retries)
        for task_id, func in tasks
    ])


# Example usage
async def error_handling_example():
    """Demonstrate different failure policies."""
    import random
    
    async def unreliable_task():
        """A task that fails 50% of the time."""
        if random.random() < 0.5:
            raise RuntimeError("Random failure!")
        return "success"
    
    tasks = [
        (f"task_{i}", unreliable_task)
        for i in range(5)
    ]
    
    print("Testing IGNORE policy:")
    results = await parallel_with_policy(tasks, FailurePolicy.IGNORE)
    success = len([r for r in results if r.success])
    print(f"  Completed: {success}/5 succeeded\n")
    
    print("Testing RETRY policy:")
    results = await parallel_with_policy(tasks, FailurePolicy.RETRY, max_retries=5)
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.task_id}: {r.attempts} attempts")


if __name__ == "__main__":
    asyncio.run(error_handling_example())
```

Choose your failure policy based on your use case:

| Policy | Use When |
|--------|----------|
| `IGNORE` | Partial results are acceptable |
| `FAIL_FAST` | Any failure means the whole task is invalid |
| `RETRY` | Failures are likely transient (network issues) |
| `REQUIRE_ALL` | You need all results, but want to try everything first |

## Building a Code Review System with Voting

Now let's put everything together into a practical application: a code review system that uses voting to detect security vulnerabilities with high confidence.

```python
"""
Code review system using voting for vulnerability detection.

Chapter 21: Parallelization - Implementation
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()


@dataclass
class Vulnerability:
    """A detected security vulnerability."""
    type: str
    severity: str  # critical, high, medium, low
    location: str
    description: str
    recommendation: str


@dataclass
class ReviewerVote:
    """A single reviewer's assessment."""
    reviewer_name: str
    found_vulnerabilities: list[Vulnerability]
    overall_risk: str
    success: bool = True
    error: str | None = None


@dataclass
class CodeReviewResult:
    """Aggregated code review results."""
    votes: list[ReviewerVote]
    confirmed_vulnerabilities: list[dict]
    risk_consensus: str
    confidence: float
    execution_time: float


class CodeReviewSystem:
    """
    Multi-perspective code review system using voting.
    
    Multiple specialized reviewers analyze code in parallel,
    then results are aggregated to identify confirmed issues.
    """
    
    # Different reviewer perspectives
    REVIEWERS = [
        {
            "name": "injection_specialist",
            "system": """You are a security expert specializing in injection attacks.
            Focus on: SQL injection, command injection, XSS, template injection.
            Be thorough but avoid false positives."""
        },
        {
            "name": "auth_specialist",
            "system": """You are a security expert specializing in authentication and authorization.
            Focus on: broken authentication, session management, access control issues.
            Be thorough but avoid false positives."""
        },
        {
            "name": "crypto_specialist", 
            "system": """You are a security expert specializing in cryptography.
            Focus on: weak encryption, hardcoded secrets, insecure random, key management.
            Be thorough but avoid false positives."""
        },
        {
            "name": "data_specialist",
            "system": """You are a security expert specializing in data protection.
            Focus on: sensitive data exposure, logging issues, data validation.
            Be thorough but avoid false positives."""
        },
        {
            "name": "general_security",
            "system": """You are a general application security expert.
            Look for any security issues not covered by specialists.
            Consider the overall security posture of the code."""
        }
    ]
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        confirmation_threshold: int = 2
    ):
        """
        Initialize the code review system.
        
        Args:
            model: The model to use for reviews
            confirmation_threshold: Minimum votes to confirm a vulnerability
        """
        self.model = model
        self.confirmation_threshold = confirmation_threshold
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _get_review(
        self,
        reviewer: dict,
        code: str,
        context: str
    ) -> ReviewerVote:
        """Get a security review from a single reviewer."""
        prompt = f"""Review the following code for security vulnerabilities.

Context: {context}

Code:
```
{code}
```

For each vulnerability found, provide:
1. TYPE: The category of vulnerability
2. SEVERITY: critical/high/medium/low
3. LOCATION: Where in the code (line number or function)
4. DESCRIPTION: What the vulnerability is
5. RECOMMENDATION: How to fix it

Also provide an OVERALL_RISK assessment: critical/high/medium/low/none

Format your response as:
VULNERABILITY 1:
TYPE: ...
SEVERITY: ...
LOCATION: ...
DESCRIPTION: ...
RECOMMENDATION: ...

VULNERABILITY 2:
... (if more vulnerabilities found)

OVERALL_RISK: ...

If no vulnerabilities found, state "NO VULNERABILITIES FOUND" and set OVERALL_RISK."""

        try:
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=reviewer["system"],
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            vulnerabilities = self._parse_vulnerabilities(text)
            overall_risk = self._extract_overall_risk(text)
            
            return ReviewerVote(
                reviewer_name=reviewer["name"],
                found_vulnerabilities=vulnerabilities,
                overall_risk=overall_risk
            )
            
        except Exception as e:
            return ReviewerVote(
                reviewer_name=reviewer["name"],
                found_vulnerabilities=[],
                overall_risk="unknown",
                success=False,
                error=str(e)
            )
    
    def _parse_vulnerabilities(self, text: str) -> list[Vulnerability]:
        """Parse vulnerability entries from response text."""
        vulnerabilities = []
        
        # Split by VULNERABILITY markers
        parts = text.split("VULNERABILITY")
        
        for part in parts[1:]:  # Skip first part (before any vulnerability)
            vuln = {}
            for field in ["TYPE", "SEVERITY", "LOCATION", "DESCRIPTION", "RECOMMENDATION"]:
                value = self._extract_field(part, field)
                if value:
                    vuln[field.lower()] = value
            
            if vuln.get("type") and vuln.get("description"):
                vulnerabilities.append(Vulnerability(
                    type=vuln.get("type", "unknown"),
                    severity=vuln.get("severity", "medium"),
                    location=vuln.get("location", "unknown"),
                    description=vuln.get("description", ""),
                    recommendation=vuln.get("recommendation", "")
                ))
        
        return vulnerabilities
    
    def _extract_field(self, text: str, field: str) -> str | None:
        """Extract a field value from text."""
        import re
        pattern = rf"{field}:\s*(.+?)(?=\n[A-Z_]+:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_overall_risk(self, text: str) -> str:
        """Extract overall risk from response."""
        import re
        match = re.search(r"OVERALL_RISK:\s*(\w+)", text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "unknown"
    
    def _aggregate_vulnerabilities(
        self, 
        votes: list[ReviewerVote]
    ) -> list[dict]:
        """
        Aggregate vulnerabilities across reviewers.
        
        Vulnerabilities found by multiple reviewers are confirmed.
        """
        # Collect all vulnerabilities with their sources
        all_vulns = []
        for vote in votes:
            if vote.success:
                for vuln in vote.found_vulnerabilities:
                    all_vulns.append({
                        "vulnerability": vuln,
                        "reporter": vote.reviewer_name
                    })
        
        # Group similar vulnerabilities
        grouped = {}
        for item in all_vulns:
            vuln = item["vulnerability"]
            # Create a key based on type and location
            key = f"{vuln.type.lower()}:{vuln.location.lower()}"
            
            if key not in grouped:
                grouped[key] = {
                    "type": vuln.type,
                    "severity": vuln.severity,
                    "location": vuln.location,
                    "descriptions": [],
                    "recommendations": [],
                    "reporters": []
                }
            
            grouped[key]["descriptions"].append(vuln.description)
            grouped[key]["recommendations"].append(vuln.recommendation)
            grouped[key]["reporters"].append(item["reporter"])
            
            # Upgrade severity if any reporter says higher
            severity_order = ["low", "medium", "high", "critical"]
            current = grouped[key]["severity"].lower()
            new = vuln.severity.lower()
            if severity_order.index(new) > severity_order.index(current):
                grouped[key]["severity"] = vuln.severity
        
        # Filter to confirmed vulnerabilities
        confirmed = []
        for key, vuln_data in grouped.items():
            vote_count = len(vuln_data["reporters"])
            confirmed.append({
                **vuln_data,
                "vote_count": vote_count,
                "confirmed": vote_count >= self.confirmation_threshold
            })
        
        # Sort by confirmation then severity
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        confirmed.sort(
            key=lambda x: (
                x["confirmed"],
                severity_order.get(x["severity"].lower(), 0)
            ),
            reverse=True
        )
        
        return confirmed
    
    def _calculate_risk_consensus(
        self, 
        votes: list[ReviewerVote]
    ) -> tuple[str, float]:
        """Calculate consensus risk level and confidence."""
        from collections import Counter
        
        valid_votes = [
            v.overall_risk.lower() 
            for v in votes 
            if v.success and v.overall_risk != "unknown"
        ]
        
        if not valid_votes:
            return "unknown", 0.0
        
        counts = Counter(valid_votes)
        winner, count = counts.most_common(1)[0]
        confidence = count / len(valid_votes)
        
        return winner, confidence
    
    async def review(
        self,
        code: str,
        context: str = "General code review"
    ) -> CodeReviewResult:
        """
        Perform a comprehensive security review of the code.
        
        Args:
            code: The code to review
            context: Description of what the code does
            
        Returns:
            CodeReviewResult with aggregated findings
        """
        import time
        start = time.time()
        
        # Get reviews from all reviewers in parallel
        votes = await asyncio.gather(*[
            self._get_review(reviewer, code, context)
            for reviewer in self.REVIEWERS
        ])
        
        # Aggregate results
        confirmed_vulns = self._aggregate_vulnerabilities(votes)
        risk_consensus, confidence = self._calculate_risk_consensus(votes)
        
        return CodeReviewResult(
            votes=votes,
            confirmed_vulnerabilities=confirmed_vulns,
            risk_consensus=risk_consensus,
            confidence=confidence,
            execution_time=time.time() - start
        )


def format_review_report(result: CodeReviewResult) -> str:
    """Format the review result as a readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("SECURITY CODE REVIEW REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Overall assessment
    lines.append(f"Overall Risk: {result.risk_consensus.upper()}")
    lines.append(f"Confidence: {result.confidence:.0%}")
    lines.append(f"Review Time: {result.execution_time:.2f}s")
    lines.append("")
    
    # Reviewer summary
    lines.append("-" * 40)
    lines.append("Reviewer Assessments:")
    for vote in result.votes:
        if vote.success:
            vuln_count = len(vote.found_vulnerabilities)
            lines.append(f"  {vote.reviewer_name}: {vote.overall_risk} ({vuln_count} issues)")
        else:
            lines.append(f"  {vote.reviewer_name}: ERROR - {vote.error}")
    lines.append("")
    
    # Confirmed vulnerabilities
    confirmed = [v for v in result.confirmed_vulnerabilities if v["confirmed"]]
    if confirmed:
        lines.append("-" * 40)
        lines.append(f"CONFIRMED VULNERABILITIES ({len(confirmed)}):")
        lines.append("")
        
        for i, vuln in enumerate(confirmed, 1):
            lines.append(f"{i}. [{vuln['severity'].upper()}] {vuln['type']}")
            lines.append(f"   Location: {vuln['location']}")
            lines.append(f"   Votes: {vuln['vote_count']} reviewers")
            lines.append(f"   Description: {vuln['descriptions'][0][:200]}")
            lines.append(f"   Fix: {vuln['recommendations'][0][:200]}")
            lines.append("")
    else:
        lines.append("-" * 40)
        lines.append("No confirmed vulnerabilities (threshold not met)")
        lines.append("")
    
    # Potential issues (not confirmed)
    potential = [v for v in result.confirmed_vulnerabilities if not v["confirmed"]]
    if potential:
        lines.append("-" * 40)
        lines.append(f"POTENTIAL ISSUES ({len(potential)}) - needs further review:")
        for vuln in potential:
            lines.append(f"  - [{vuln['severity']}] {vuln['type']} at {vuln['location']}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


async def main():
    """Demonstrate the code review system."""
    
    # Sample code with intentional vulnerabilities
    vulnerable_code = '''
def login(username, password):
    """Authenticate a user."""
    # Build SQL query
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    
    if result:
        # Store password in session for later
        session['password'] = password
        session['user'] = username
        return True
    return False

def get_user_data(user_id):
    """Get user profile data."""
    # Admin bypass for testing
    if user_id == "admin_debug":
        return get_all_users()
    
    data = db.query(f"SELECT * FROM profiles WHERE id={user_id}")
    return data

def reset_password(email):
    """Send password reset email."""
    import random
    token = random.randint(1000, 9999)  # 4-digit reset code
    send_email(email, f"Your reset code is: {token}")
    cache.set(f"reset_{email}", token)
    
def render_profile(user_data):
    """Render user profile page."""
    template = f"<h1>Welcome {user_data['name']}</h1>"
    template += f"<p>Email: {user_data['email']}</p>"
    return template
'''
    
    context = "User authentication and profile management module for a web application"
    
    print("Starting parallel security review...\n")
    
    reviewer = CodeReviewSystem(confirmation_threshold=2)
    result = await reviewer.review(vulnerable_code, context)
    
    report = format_review_report(result)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
```

### How the Code Review System Works

1. **Specialized Reviewers**: Five experts, each focused on different security domains
2. **Parallel Analysis**: All reviewers examine the code simultaneously
3. **Structured Output**: Each reviewer reports vulnerabilities in a consistent format
4. **Voting Aggregation**: Similar vulnerabilities are grouped and counted
5. **Confirmation Threshold**: Only issues found by multiple reviewers are "confirmed"
6. **Risk Consensus**: Overall risk is determined by majority vote

This approach catches more issues than a single pass while reducing false positives through consensus.

## Common Pitfalls

### 1. Not Limiting Concurrency

**Problem**: Making too many API calls at once can trigger rate limits.

```python
# Bad: Unlimited concurrency
results = await asyncio.gather(*[make_call() for _ in range(100)])

# Good: Use a semaphore
semaphore = asyncio.Semaphore(5)
async def bounded_call():
    async with semaphore:
        return await make_call()

results = await asyncio.gather(*[bounded_call() for _ in range(100)])
```

### 2. Ignoring Partial Failures

**Problem**: One failed task shouldn't crash everything.

```python
# Bad: Exceptions propagate
results = await asyncio.gather(*tasks)  # One failure = all fail

# Good: Capture exceptions
results = await asyncio.gather(*tasks, return_exceptions=True)
# Now results contains either values or Exception objects
```

### 3. Forgetting That Order Matters

**Problem**: `asyncio.gather()` returns results in task order, but tasks complete in arbitrary order.

```python
# Results are in the same order as input tasks
tasks = [analyze(doc) for doc in documents]
results = await asyncio.gather(*tasks)

# results[0] corresponds to documents[0], etc.
# Even though documents[2] might have finished first
```

## Practical Exercise

**Task:** Build a parallel translation system that translates text into multiple languages simultaneously and uses voting to select the best translation for ambiguous phrases.

**Requirements:**

1. Create a `TranslationWorkflow` class that:
   - Accepts a source text and list of target languages
   - Translates to all languages in parallel using sectioning
   - For each translation, uses 3 different "translator personas" (formal, casual, technical) with voting
   - Returns translations with confidence scores

2. The system should handle:
   - At least 3 target languages
   - Graceful handling of translation failures
   - A merged report showing all translations

3. Use proper async patterns:
   - Semaphore for concurrency limiting
   - Error handling that doesn't crash on single failures
   - Timing information

**Hints:**
- Combine sectioning (for languages) with voting (for translator personas)
- Consider nesting the patterns: each "section" runs its own voting workflow
- Track which translations had high consensus vs. disagreement

**Solution:** See `code/exercise_translation.py`

## Key Takeaways

- **Asyncio enables parallel LLM calls**: Use `async def`, `await`, and `asyncio.gather()` to run operations concurrently
- **Sectioning divides work**: Independent subtasks run in parallel and results are merged
- **Voting builds consensus**: Multiple perspectives on the same task increase confidence
- **Aggregation strategy matters**: Choose majority vote, consensus threshold, or synthesis based on your needs
- **Error handling is essential**: Use semaphores for rate limiting and capture failures gracefully
- **Parallel execution trades cost for speed/confidence**: You're making more API calls, but getting results faster or more reliably

## What's Next

In Chapter 22, we'll explore the **Orchestrator-Workers** pattern—a more sophisticated approach where an orchestrator LLM dynamically decides what tasks to delegate and which workers should handle them. Unlike the static parallelization we built here, orchestrator-workers adapts to the specific needs of each request.
