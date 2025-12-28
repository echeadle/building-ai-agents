"""
Parallel Orchestrator - Exercise Solution

Chapter 23: Orchestrator-Workers - Implementation

This module extends the ResearchOrchestrator to support parallel
worker execution using asyncio, significantly reducing total
execution time.

Exercise Task:
- Add an `async_research` method that runs workers in parallel
- Track and report time savings from parallelization
- Handle errors in parallel execution gracefully
"""

import os
import json
import re
import time
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Subtask:
    """Represents a subtask created by the orchestrator."""
    id: str
    type: str
    title: str
    description: str
    focus_areas: list[str]
    result: Optional[str] = None
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class TaskPlan:
    """The orchestrator's plan for handling a complex query."""
    original_query: str
    query_analysis: str
    subtasks: list[Subtask]
    synthesis_guidance: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkerResult:
    """Result from a worker executing a subtask."""
    subtask_id: str
    subtask_title: str
    content: str
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class ParallelOrchestratorResult:
    """Result from parallel orchestrator with timing comparison."""
    query: str
    plan: TaskPlan
    worker_results: list[WorkerResult]
    synthesis: str
    success: bool
    total_time: float
    sequential_estimate: float  # Estimated time if run sequentially
    time_saved: float  # Difference (savings from parallelization)
    speedup_factor: float  # Sequential / Parallel
    subtasks_completed: int
    subtasks_failed: int


# =============================================================================
# System Prompts
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert task orchestrator. Analyze complex queries and break them down into focused, independent subtasks.

Create 3-6 subtasks that:
- Are independent (can be completed separately)
- Don't overlap significantly
- Collectively address the full query

Subtask types: research, analysis, comparison

Output JSON only:
{
    "query_analysis": "Brief analysis",
    "subtasks": [
        {"id": "task_1", "type": "research|analysis|comparison", "title": "Title", "description": "Description", "focus_areas": ["area1", "area2"]}
    ],
    "synthesis_guidance": "How to combine results"
}"""


WORKER_PROMPTS = {
    "research": "You are a research assistant. Provide factual, comprehensive information with key findings, examples, and caveats.",
    "analysis": "You are an analyst. Evaluate implications, trade-offs, pros/cons, and provide meaningful conclusions.",
    "comparison": "You are a comparison expert. Compare options using clear criteria and provide fair, contextual recommendations."
}


SYNTHESIS_PROMPT = """Synthesize the research results into a coherent response that:
- Directly addresses the original query
- Integrates insights from all sources
- Uses clear structure with headers
- Ends with conclusions/recommendations"""


# =============================================================================
# Async Worker Class
# =============================================================================

class AsyncWorker:
    """
    Async worker that executes subtasks using the async Anthropic client.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic()  # Async client!
        self.model = model
    
    async def execute(self, subtask: Subtask, context: str = "") -> WorkerResult:
        """
        Execute a single subtask asynchronously.
        
        Args:
            subtask: The subtask to execute
            context: Optional additional context
            
        Returns:
            WorkerResult from the execution
        """
        start_time = time.time()
        
        system_prompt = WORKER_PROMPTS.get(subtask.type, WORKER_PROMPTS["research"])
        
        task_message = f"## Task: {subtask.title}\n\n{subtask.description}\n\n"
        task_message += "## Focus Areas\n"
        task_message += "\n".join(f"- {area}" for area in subtask.focus_areas)
        
        if context:
            task_message += f"\n\n## Context\n{context}"
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": task_message}]
            )
            
            execution_time = time.time() - start_time
            
            return WorkerResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                content=response.content[0].text,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkerResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                content="",
                success=False,
                error=str(e),
                execution_time=execution_time
            )


# =============================================================================
# Async Synthesizer
# =============================================================================

class AsyncSynthesizer:
    """Async synthesizer for combining results."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic()
        self.model = model
    
    async def synthesize(
        self,
        original_query: str,
        results: list[WorkerResult],
        synthesis_guidance: str = ""
    ) -> str:
        """Synthesize results asynchronously."""
        successful = [r for r in results if r.success]
        
        if not successful:
            return "Unable to generate response. All subtasks failed."
        
        request = f"## Original Query\n{original_query}\n\n## Results\n\n"
        for i, r in enumerate(successful, 1):
            request += f"### {i}. {r.subtask_title}\n{r.content}\n\n---\n\n"
        
        if synthesis_guidance:
            request += f"## Guidance\n{synthesis_guidance}\n\n"
        
        request += "Synthesize these into a comprehensive response."
        
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYNTHESIS_PROMPT,
            messages=[{"role": "user", "content": request}]
        )
        
        return response.content[0].text


# =============================================================================
# Parallel Research Orchestrator
# =============================================================================

class ParallelResearchOrchestrator:
    """
    Research orchestrator with parallel worker execution.
    
    This extends the basic orchestrator to run workers concurrently
    using asyncio, providing significant speedup for multi-subtask
    queries.
    
    Example:
        orchestrator = ParallelResearchOrchestrator()
        result = await orchestrator.async_research("Your complex query")
        print(f"Speedup: {result.speedup_factor:.1f}x")
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_subtasks: int = 6,
        verbose: bool = True
    ):
        self.client = anthropic.Anthropic()  # Sync for planning
        self.model = model
        self.max_subtasks = max_subtasks
        self.verbose = verbose
        self.async_worker = AsyncWorker(model=model)
        self.async_synthesizer = AsyncSynthesizer(model=model)
    
    def _log(self, message: str) -> None:
        """Print if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def _create_plan(self, query: str) -> TaskPlan:
        """Create task plan (sync - runs once)."""
        self._log("\n📋 Creating task plan...")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Break down:\n\n{query}"}]
        )
        
        response_text = response.content[0].text
        
        try:
            plan_data = json.loads(response_text)
        except json.JSONDecodeError:
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                plan_data = json.loads(match.group())
            else:
                raise ValueError("Invalid JSON from orchestrator")
        
        subtasks = []
        for task_data in plan_data.get("subtasks", [])[:self.max_subtasks]:
            subtasks.append(Subtask(
                id=task_data.get("id", f"task_{len(subtasks)+1}"),
                type=task_data.get("type", "research"),
                title=task_data.get("title", "Untitled"),
                description=task_data.get("description", ""),
                focus_areas=task_data.get("focus_areas", [])
            ))
        
        plan = TaskPlan(
            original_query=query,
            query_analysis=plan_data.get("query_analysis", ""),
            subtasks=subtasks,
            synthesis_guidance=plan_data.get("synthesis_guidance", "")
        )
        
        self._log(f"   ✓ Created {len(subtasks)} subtasks")
        return plan
    
    async def _execute_subtask(
        self,
        subtask: Subtask,
        context: str,
        index: int,
        total: int
    ) -> WorkerResult:
        """Execute a single subtask with logging."""
        self._log(f"   🚀 Started [{index}/{total}]: {subtask.title}")
        
        subtask.status = "in_progress"
        subtask.started_at = datetime.now()
        
        result = await self.async_worker.execute(subtask, context)
        
        subtask.completed_at = datetime.now()
        
        if result.success:
            subtask.result = result.content
            subtask.status = "completed"
            self._log(f"   ✓ Finished [{index}/{total}]: {subtask.title} ({result.execution_time:.1f}s)")
        else:
            subtask.status = "failed"
            self._log(f"   ✗ Failed [{index}/{total}]: {subtask.title}")
        
        return result
    
    async def _execute_all_parallel(
        self,
        plan: TaskPlan,
        context: str = ""
    ) -> list[WorkerResult]:
        """Execute all subtasks in parallel."""
        self._log(f"\n🔧 Executing {len(plan.subtasks)} subtasks in PARALLEL...")
        
        # Create tasks for all subtasks
        tasks = [
            self._execute_subtask(subtask, context, i, len(plan.subtasks))
            for i, subtask in enumerate(plan.subtasks, 1)
        ]
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that were returned
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to failed WorkerResult
                subtask = plan.subtasks[i]
                processed_results.append(WorkerResult(
                    subtask_id=subtask.id,
                    subtask_title=subtask.title,
                    content="",
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def async_research(
        self,
        query: str,
        context: str = ""
    ) -> ParallelOrchestratorResult:
        """
        Execute research with parallel worker execution.
        
        Args:
            query: The research question
            context: Optional additional context
            
        Returns:
            ParallelOrchestratorResult with timing comparison
        """
        total_start = time.time()
        
        self._log(f"\n{'='*60}")
        self._log("🔍 PARALLEL RESEARCH")
        self._log(f"{'='*60}")
        self._log(f"\n{query}")
        
        # Step 1: Create plan (sync)
        plan = self._create_plan(query)
        
        # Step 2: Execute in parallel (async)
        parallel_start = time.time()
        results = await self._execute_all_parallel(plan, context)
        parallel_time = time.time() - parallel_start
        
        # Calculate sequential estimate (sum of all execution times)
        sequential_estimate = sum(
            r.execution_time or 0 for r in results
        )
        
        # Step 3: Synthesize (async)
        self._log("\n📝 Synthesizing results...")
        synthesis = await self.async_synthesizer.synthesize(
            plan.original_query,
            results,
            plan.synthesis_guidance
        )
        self._log("   ✓ Synthesis complete")
        
        total_time = time.time() - total_start
        completed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        # Calculate speedup
        time_saved = sequential_estimate - parallel_time
        speedup = sequential_estimate / parallel_time if parallel_time > 0 else 1.0
        
        self._log(f"\n{'='*60}")
        self._log("⚡ PARALLEL EXECUTION COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"   Subtasks: {completed}/{len(results)} successful")
        self._log(f"   Parallel time: {parallel_time:.1f}s")
        self._log(f"   Sequential estimate: {sequential_estimate:.1f}s")
        self._log(f"   Time saved: {time_saved:.1f}s")
        self._log(f"   Speedup: {speedup:.2f}x")
        
        return ParallelOrchestratorResult(
            query=query,
            plan=plan,
            worker_results=results,
            synthesis=synthesis,
            success=completed > 0,
            total_time=total_time,
            sequential_estimate=sequential_estimate,
            time_saved=time_saved,
            speedup_factor=speedup,
            subtasks_completed=completed,
            subtasks_failed=failed
        )
    
    def research(self, query: str, context: str = "") -> ParallelOrchestratorResult:
        """
        Synchronous wrapper for async_research.
        
        Use this if you're not in an async context.
        """
        return asyncio.run(self.async_research(query, context))


# =============================================================================
# Comparison Demo
# =============================================================================

async def compare_sequential_vs_parallel():
    """
    Demonstrate the time savings from parallel execution.
    """
    print("\n" + "=" * 70)
    print("SEQUENTIAL vs PARALLEL COMPARISON")
    print("=" * 70)
    
    query = "What are the major challenges and opportunities in renewable energy adoption?"
    
    # Run parallel version
    orchestrator = ParallelResearchOrchestrator(verbose=True)
    result = await orchestrator.async_research(query)
    
    # Display comparison
    print("\n" + "=" * 70)
    print("TIMING COMPARISON")
    print("=" * 70)
    print(f"Query: {query[:60]}...")
    print(f"\nSubtasks executed: {len(result.worker_results)}")
    print(f"\nSequential estimate: {result.sequential_estimate:.1f}s")
    print(f"Parallel actual:     {result.total_time:.1f}s")
    print(f"Time saved:          {result.time_saved:.1f}s")
    print(f"Speedup factor:      {result.speedup_factor:.2f}x")
    
    # Individual task times
    print("\nIndividual task execution times:")
    for wr in result.worker_results:
        status = "✓" if wr.success else "✗"
        time_str = f"{wr.execution_time:.1f}s" if wr.execution_time else "N/A"
        print(f"  {status} {wr.subtask_title}: {time_str}")
    
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)
    print(result.synthesis)
    
    return result


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PARALLEL ORCHESTRATOR - Exercise Solution")
    print("=" * 70)
    print("""
This solution demonstrates:
1. Async worker execution using asyncio
2. Parallel subtask processing with asyncio.gather()
3. Time savings measurement and speedup calculation
4. Graceful error handling in parallel execution

Running comparison demo...
""")
    
    # Run the comparison
    asyncio.run(compare_sequential_vs_parallel())
