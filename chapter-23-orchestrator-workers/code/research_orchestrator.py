"""
Complete Research Orchestrator implementation.

Chapter 23: Orchestrator-Workers - Implementation

This module provides a complete orchestrator-workers implementation
for handling complex research queries. It combines the orchestrator,
workers, delegator, and synthesizer into a single, easy-to-use class.
"""

import os
import json
import time
import re
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
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
    type: str  # research, analysis, comparison
    title: str
    description: str
    focus_areas: list[str]
    result: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
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
class OrchestratorResult:
    """Complete result from the orchestrator."""
    query: str
    plan: TaskPlan
    worker_results: list[WorkerResult]
    synthesis: str
    success: bool
    total_time: float
    subtasks_completed: int
    subtasks_failed: int


# =============================================================================
# System Prompts
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert task orchestrator. Your role is to analyze complex queries and break them down into focused, independent subtasks.

## Your Responsibilities
1. Analyze the query to understand what the user truly needs
2. Identify distinct aspects, perspectives, or components
3. Create 3-6 focused subtasks that collectively address the query
4. Ensure subtasks are independent (can be completed separately)

## Subtask Types
- research: Gather factual information on a topic
- analysis: Evaluate implications, trade-offs, significance
- comparison: Compare options, approaches, or perspectives

## Critical Rules
- Subtasks must NOT overlap significantly in scope
- Each subtask should address a UNIQUE aspect of the query
- Include at least one analysis or comparison task for perspective

## Output Format (JSON only)
{
    "query_analysis": "Brief analysis of what the user is asking",
    "subtasks": [
        {
            "id": "task_1",
            "type": "research|analysis|comparison",
            "title": "Short descriptive title",
            "description": "What this subtask should accomplish",
            "focus_areas": ["specific", "areas", "to", "cover"]
        }
    ],
    "synthesis_guidance": "How to combine the subtask results"
}

Respond ONLY with valid JSON. No other text."""


WORKER_PROMPTS = {
    "research": """You are a thorough research assistant. Provide comprehensive, factual information.

Guidelines:
- Focus on factual, verifiable information
- Organize with clear key points (3-5 main findings)
- Include specific examples or data
- Note important caveats or limitations

Structure your response with:
1. Key findings
2. Supporting details for each
3. Notable examples or evidence
4. Important caveats""",

    "analysis": """You are an analytical expert. Evaluate implications, trade-offs, and significance.

Guidelines:
- Consider multiple perspectives
- Identify pros and cons, benefits and risks
- Support analysis with reasoning
- Draw meaningful conclusions

Structure your response with:
1. Overview of the issue
2. Key factors to consider
3. Analysis of implications (positive and negative)
4. Conclusions and insights""",

    "comparison": """You are a comparison specialist. Compare options fairly using clear criteria.

Guidelines:
- Use clear comparison criteria
- Evaluate each option fairly
- Highlight key differences and similarities
- Provide contextual recommendations

Structure your response with:
1. Options being compared
2. Comparison criteria
3. Evaluation of each option
4. Key differences and recommendations"""
}


SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing information into coherent responses.

Guidelines:
- Directly address the original query
- Integrate insights from all sources naturally
- Maintain logical flow and clear structure
- Highlight key findings and conclusions
- Note any conflicts or caveats
- Provide actionable insights where appropriate

Output Structure:
- Use headers to organize major sections
- Lead with the most important findings
- Support claims with details from the research
- End with clear conclusions or recommendations"""


# =============================================================================
# Component Classes
# =============================================================================

class Worker:
    """Executes individual subtasks assigned by the orchestrator."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def execute(self, subtask: Subtask, context: str = "") -> WorkerResult:
        """Execute a single subtask."""
        start_time = time.time()
        
        system_prompt = WORKER_PROMPTS.get(subtask.type, WORKER_PROMPTS["research"])
        
        task_message = f"## Task: {subtask.title}\n\n{subtask.description}\n\n"
        task_message += "## Focus Areas\n"
        task_message += "\n".join(f"- {area}" for area in subtask.focus_areas)
        
        if context:
            task_message += f"\n\n## Additional Context\n{context}"
        
        task_message += "\n\nPlease complete this task thoroughly."
        
        try:
            response = self.client.messages.create(
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
            
        except anthropic.APIError as e:
            execution_time = time.time() - start_time
            return WorkerResult(
                subtask_id=subtask.id,
                subtask_title=subtask.title,
                content="",
                success=False,
                error=str(e),
                execution_time=execution_time
            )


class Synthesizer:
    """Synthesizes multiple worker results into a coherent response."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def synthesize(
        self,
        original_query: str,
        results: list[WorkerResult],
        synthesis_guidance: str = ""
    ) -> str:
        """Synthesize worker results into a final response."""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return "Unable to generate a response. All subtasks failed."
        
        # Build synthesis request
        synthesis_request = f"## Original Query\n{original_query}\n\n"
        synthesis_request += "## Research Results\n\n"
        
        for i, result in enumerate(successful_results, 1):
            synthesis_request += f"### {i}. {result.subtask_title}\n\n"
            synthesis_request += f"{result.content}\n\n---\n\n"
        
        if synthesis_guidance:
            synthesis_request += f"## Synthesis Guidance\n{synthesis_guidance}\n\n"
        
        synthesis_request += "Please synthesize these results into a comprehensive response."
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": synthesis_request}]
        )
        
        return response.content[0].text


# =============================================================================
# Main Orchestrator Class
# =============================================================================

class ResearchOrchestrator:
    """
    A complete orchestrator-workers implementation for research queries.
    
    This class coordinates the breakdown of complex queries into subtasks,
    delegates work to specialized workers, and synthesizes results into
    coherent responses.
    
    Example:
        orchestrator = ResearchOrchestrator()
        result = orchestrator.research("What are the impacts of AI on healthcare?")
        print(result.synthesis)
    
    Attributes:
        model: Claude model used for all LLM calls
        max_subtasks: Maximum number of subtasks to create
        verbose: Whether to print progress updates
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_subtasks: int = 6,
        verbose: bool = True
    ):
        """
        Initialize the orchestrator.
        
        Args:
            model: Claude model to use for all LLM calls
            max_subtasks: Maximum number of subtasks to create
            verbose: Whether to print progress updates
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_subtasks = max_subtasks
        self.verbose = verbose
        self.worker = Worker(model=model)
        self.synthesizer = Synthesizer(model=model)
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _create_plan(self, query: str) -> TaskPlan:
        """Create a task plan by analyzing the query."""
        self._log("\n📋 Creating task plan...")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Break down this query into subtasks:\n\n{query}"
                }
            ]
        )
        
        response_text = response.content[0].text
        
        # Parse JSON from response
        try:
            plan_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                raise ValueError("Orchestrator returned invalid JSON")
        
        # Convert to Subtask objects
        subtasks = []
        for task_data in plan_data.get("subtasks", [])[:self.max_subtasks]:
            subtask = Subtask(
                id=task_data.get("id", f"task_{len(subtasks)+1}"),
                type=task_data.get("type", "research"),
                title=task_data.get("title", "Untitled"),
                description=task_data.get("description", ""),
                focus_areas=task_data.get("focus_areas", [])
            )
            subtasks.append(subtask)
        
        plan = TaskPlan(
            original_query=query,
            query_analysis=plan_data.get("query_analysis", ""),
            subtasks=subtasks,
            synthesis_guidance=plan_data.get("synthesis_guidance", "")
        )
        
        self._log(f"   ✓ Created plan with {len(subtasks)} subtasks")
        
        if self.verbose:
            for st in subtasks:
                self._log(f"     - [{st.type}] {st.title}")
        
        return plan
    
    def _execute_subtasks(
        self,
        plan: TaskPlan,
        context: str = ""
    ) -> list[WorkerResult]:
        """Execute all subtasks using workers."""
        results = []
        total = len(plan.subtasks)
        
        self._log(f"\n🔧 Executing {total} subtasks...")
        
        for i, subtask in enumerate(plan.subtasks, 1):
            self._log(f"\n   [{i}/{total}] {subtask.title}")
            
            subtask.status = "in_progress"
            subtask.started_at = datetime.now()
            
            result = self.worker.execute(subtask, context)
            
            subtask.completed_at = datetime.now()
            
            if result.success:
                subtask.result = result.content
                subtask.status = "completed"
                self._log(f"           ✓ Completed ({result.execution_time:.1f}s)")
            else:
                subtask.status = "failed"
                self._log(f"           ✗ Failed: {result.error}")
            
            results.append(result)
        
        return results
    
    def _synthesize_results(
        self,
        plan: TaskPlan,
        results: list[WorkerResult]
    ) -> str:
        """Synthesize worker results into final response."""
        self._log("\n📝 Synthesizing results...")
        
        synthesis = self.synthesizer.synthesize(
            original_query=plan.original_query,
            results=results,
            synthesis_guidance=plan.synthesis_guidance
        )
        
        self._log("   ✓ Synthesis complete")
        return synthesis
    
    def research(
        self,
        query: str,
        context: str = ""
    ) -> OrchestratorResult:
        """
        Execute a complete research workflow for a complex query.
        
        This method:
        1. Creates a task plan by breaking down the query
        2. Executes all subtasks using specialized workers
        3. Synthesizes the results into a coherent response
        
        Args:
            query: The research question to answer
            context: Optional additional context to provide to workers
            
        Returns:
            OrchestratorResult containing the plan, results, and synthesis
        """
        start_time = time.time()
        
        self._log(f"\n{'='*60}")
        self._log(f"🔍 RESEARCH QUERY")
        self._log(f"{'='*60}")
        self._log(f"\n{query}")
        
        # Step 1: Create plan
        plan = self._create_plan(query)
        
        # Step 2: Execute subtasks
        results = self._execute_subtasks(plan, context)
        
        # Step 3: Synthesize
        synthesis = self._synthesize_results(plan, results)
        
        total_time = time.time() - start_time
        completed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        self._log(f"\n{'='*60}")
        self._log(f"✅ COMPLETE")
        self._log(f"{'='*60}")
        self._log(f"   Subtasks: {completed}/{len(results)} successful")
        self._log(f"   Total time: {total_time:.1f}s")
        
        return OrchestratorResult(
            query=query,
            plan=plan,
            worker_results=results,
            synthesis=synthesis,
            success=completed > 0,
            total_time=total_time,
            subtasks_completed=completed,
            subtasks_failed=failed
        )
    
    def get_plan_only(self, query: str) -> TaskPlan:
        """
        Get just the task plan without executing.
        
        Useful for previewing how a query will be broken down
        before committing to execution.
        
        Args:
            query: The query to analyze
            
        Returns:
            TaskPlan without execution
        """
        return self._create_plan(query)
    
    def execute_plan(
        self,
        plan: TaskPlan,
        context: str = ""
    ) -> OrchestratorResult:
        """
        Execute a pre-existing plan.
        
        Use this after reviewing a plan from get_plan_only().
        
        Args:
            plan: TaskPlan to execute
            context: Optional additional context
            
        Returns:
            OrchestratorResult with execution results
        """
        start_time = time.time()
        
        self._log(f"\n{'='*60}")
        self._log(f"🔧 EXECUTING EXISTING PLAN")
        self._log(f"{'='*60}")
        
        results = self._execute_subtasks(plan, context)
        synthesis = self._synthesize_results(plan, results)
        
        total_time = time.time() - start_time
        completed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        
        return OrchestratorResult(
            query=plan.original_query,
            plan=plan,
            worker_results=results,
            synthesis=synthesis,
            success=completed > 0,
            total_time=total_time,
            subtasks_completed=completed,
            subtasks_failed=failed
        )


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create orchestrator
    orchestrator = ResearchOrchestrator(verbose=True)
    
    # Example research query
    query = """What are the environmental and economic impacts of 
    electric vehicle adoption in urban areas?"""
    
    # Execute research
    result = orchestrator.research(query)
    
    # Display final synthesis
    print("\n" + "=" * 60)
    print("FINAL SYNTHESIS")
    print("=" * 60)
    print(result.synthesis)
    
    # Show execution metrics
    print("\n" + "=" * 60)
    print("EXECUTION METRICS")
    print("=" * 60)
    print(f"Query: {result.query}")
    print(f"Subtasks created: {len(result.plan.subtasks)}")
    print(f"Subtasks completed: {result.subtasks_completed}")
    print(f"Subtasks failed: {result.subtasks_failed}")
    print(f"Total execution time: {result.total_time:.1f}s")
    print(f"Overall success: {result.success}")
    
    # Show individual worker results summary
    print("\n" + "=" * 60)
    print("WORKER RESULTS SUMMARY")
    print("=" * 60)
    for wr in result.worker_results:
        status = "✓" if wr.success else "✗"
        time_str = f"{wr.execution_time:.1f}s" if wr.execution_time else "N/A"
        print(f"  {status} {wr.subtask_title} ({time_str})")
