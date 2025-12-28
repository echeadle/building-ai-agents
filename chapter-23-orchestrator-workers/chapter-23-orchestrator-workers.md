---
chapter: 23
title: "Orchestrator-Workers - Implementation"
part: 3
part_title: "Workflow Patterns"
date: 2025-01-15
draft: false
---

# Chapter 23: Orchestrator-Workers - Implementation

## Introduction

In Chapter 22, we explored the orchestrator-workers pattern conceptually—understanding when to use it and how to design the relationship between an orchestrator that plans and workers that execute. Now it's time to build it.

The orchestrator-workers pattern shines when you can't predict in advance how to break down a task. A research question like "What are the environmental and economic impacts of electric vehicle adoption?" requires different subtasks than "How has remote work affected urban real estate markets?" The orchestrator must analyze each query, determine what aspects need investigation, and delegate accordingly.

In this chapter, we'll build a complete, working orchestrator-workers system. By the end, you'll have a research orchestrator that can take complex questions, break them into focused subtasks, delegate to specialized workers, and synthesize their findings into coherent reports.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement an orchestrator LLM that dynamically breaks down complex tasks
- Create flexible worker tasks that can handle various subtask types
- Build a task delegation system that tracks and collects worker results
- Synthesize multiple worker outputs into a coherent final result
- Design a reusable `Orchestrator` class for your own applications

## Prerequisites

Before diving in, make sure you have:

- Completed Chapters 1-14 (especially Chapter 14's `AugmentedLLM` class)
- Your `.env` file configured with `ANTHROPIC_API_KEY`
- Understanding of the orchestrator-workers concept from Chapter 22

## The Architecture We're Building

Let's visualize what we're implementing:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
│         "What are the impacts of AI on healthcare?"              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                              │
│                                                                  │
│  1. Analyze the query                                            │
│  2. Break into subtasks:                                         │
│     - "Research diagnostic AI applications"                      │
│     - "Research treatment AI applications"                       │
│     - "Research administrative AI applications"                  │
│     - "Analyze challenges and risks"                             │
│  3. Dispatch to workers                                          │
│  4. Collect results                                              │
│  5. Synthesize final report                                      │
└─────────────────────────────────────────────────────────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
      ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
      │Worker 1│    │Worker 2│    │Worker 3│    │Worker 4│
      │Research│    │Research│    │Research│    │Analysis│
      └────────┘    └────────┘    └────────┘    └────────┘
           │              │              │              │
           └──────────────┴──────────────┴──────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │   Final Synthesis    │
                    │   by Orchestrator    │
                    └─────────────────────┘
```

## Implementing the Orchestrator LLM

The orchestrator's job is to understand a complex task and break it into manageable pieces. This requires careful prompt engineering—the orchestrator must know how to decompose tasks effectively.

### The Task Decomposition Prompt

The orchestrator needs clear instructions on how to analyze and break down tasks:

```python
"""
Orchestrator implementation for the orchestrator-workers pattern.

Chapter 23: Orchestrator-Workers - Implementation
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert task orchestrator. Your role is to analyze complex queries and break them down into focused, independent subtasks that can be researched or analyzed separately.

## Your Responsibilities

1. **Analyze the Query**: Understand what the user is truly asking for
2. **Identify Dimensions**: Find the distinct aspects, perspectives, or components
3. **Create Subtasks**: Break the query into 3-6 focused subtasks
4. **Ensure Coverage**: Make sure subtasks collectively address the full query
5. **Maintain Independence**: Each subtask should be completable on its own

## Subtask Guidelines

- Each subtask should be specific and focused
- Subtasks should not overlap significantly
- Include both research tasks (gather information) and analysis tasks (evaluate/compare)
- Consider different perspectives: technical, economic, social, practical
- Aim for 3-6 subtasks (fewer for simple queries, more for complex ones)

## Output Format

You must respond with a JSON object in this exact format:
{
    "query_analysis": "Brief analysis of what the user is asking",
    "subtasks": [
        {
            "id": "task_1",
            "type": "research|analysis|comparison",
            "title": "Short descriptive title",
            "description": "Detailed description of what this subtask should accomplish",
            "focus_areas": ["specific", "areas", "to", "cover"]
        }
    ],
    "synthesis_guidance": "How the subtask results should be combined"
}

Respond ONLY with the JSON object, no other text."""


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


@dataclass
class TaskPlan:
    """The orchestrator's plan for handling a complex query."""
    original_query: str
    query_analysis: str
    subtasks: list[Subtask]
    synthesis_guidance: str


class Orchestrator:
    """
    Orchestrates complex tasks by breaking them into subtasks 
    and delegating to workers.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_subtasks: int = 6
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_subtasks = max_subtasks
    
    def create_plan(self, query: str) -> TaskPlan:
        """
        Analyze a complex query and create a plan of subtasks.
        
        Args:
            query: The complex query to break down
            
        Returns:
            TaskPlan containing subtasks to be executed
        """
        # Ask the orchestrator to decompose the task
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
        
        # Parse the response
        response_text = response.content[0].text
        
        try:
            plan_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Orchestrator returned invalid JSON: {e}")
        
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
        
        return TaskPlan(
            original_query=query,
            query_analysis=plan_data.get("query_analysis", ""),
            subtasks=subtasks,
            synthesis_guidance=plan_data.get("synthesis_guidance", "")
        )


if __name__ == "__main__":
    # Test the orchestrator's planning capability
    orchestrator = Orchestrator()
    
    query = "What are the environmental and economic impacts of electric vehicle adoption in urban areas?"
    
    print(f"Query: {query}\n")
    print("Creating task plan...")
    
    plan = orchestrator.create_plan(query)
    
    print(f"\nQuery Analysis: {plan.query_analysis}\n")
    print(f"Created {len(plan.subtasks)} subtasks:\n")
    
    for subtask in plan.subtasks:
        print(f"  [{subtask.id}] {subtask.title}")
        print(f"      Type: {subtask.type}")
        print(f"      Description: {subtask.description}")
        print(f"      Focus Areas: {', '.join(subtask.focus_areas)}")
        print()
    
    print(f"Synthesis Guidance: {plan.synthesis_guidance}")
```

When you run this, the orchestrator analyzes your query and produces a structured plan. For a query about electric vehicles, you might see subtasks like:

- Research environmental benefits (emissions reduction, air quality)
- Research environmental challenges (battery production, electricity sources)
- Analyze economic impacts for consumers (costs, savings)
- Analyze economic impacts for cities (infrastructure, jobs)
- Compare urban vs suburban adoption patterns

The key insight here is that **the orchestrator prompt is doing heavy lifting**. A well-designed prompt produces good task decomposition; a vague prompt produces overlapping or missing subtasks.

## Creating Flexible Worker Tasks

Workers receive subtasks and produce results. They need to be flexible enough to handle different types of tasks while maintaining consistency in their output format.

### The Worker Implementation

```python
"""
Worker implementation for executing subtasks.

Chapter 23: Orchestrator-Workers - Implementation
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# Different worker prompts for different task types
WORKER_PROMPTS = {
    "research": """You are a thorough research assistant. Your task is to provide comprehensive, factual information on a specific topic.

## Guidelines
- Focus on factual, verifiable information
- Cover the topic thoroughly but stay focused
- Organize information clearly with key points
- Note any important caveats or limitations
- Cite specific examples or data when relevant

## Output Format
Provide a well-organized research summary with:
1. Key findings (3-5 main points)
2. Supporting details for each finding
3. Notable examples or evidence
4. Important caveats or considerations""",

    "analysis": """You are an analytical expert. Your task is to analyze a topic, evaluating its implications, trade-offs, and significance.

## Guidelines
- Provide balanced analysis considering multiple perspectives
- Identify pros and cons, benefits and risks
- Evaluate significance and implications
- Support analysis with reasoning
- Draw meaningful conclusions

## Output Format
Provide a structured analysis with:
1. Overview of the issue
2. Key factors to consider
3. Analysis of implications (positive and negative)
4. Conclusions and insights""",

    "comparison": """You are a comparison specialist. Your task is to compare different options, approaches, or perspectives on a topic.

## Guidelines
- Identify clear criteria for comparison
- Evaluate each option fairly
- Highlight key differences and similarities
- Note context-dependent factors
- Provide actionable insights

## Output Format
Provide a structured comparison with:
1. Options/perspectives being compared
2. Comparison criteria
3. Evaluation of each option
4. Summary of key differences
5. Contextual recommendations"""
}


@dataclass
class WorkerResult:
    """Result from a worker executing a subtask."""
    subtask_id: str
    subtask_title: str
    content: str
    success: bool
    error: Optional[str] = None


class Worker:
    """
    Executes individual subtasks assigned by the orchestrator.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def execute(
        self,
        subtask_id: str,
        subtask_type: str,
        title: str,
        description: str,
        focus_areas: list[str],
        context: str = ""
    ) -> WorkerResult:
        """
        Execute a single subtask.
        
        Args:
            subtask_id: Unique identifier for the subtask
            subtask_type: Type of task (research, analysis, comparison)
            title: Title of the subtask
            description: Detailed description of what to do
            focus_areas: Specific areas to focus on
            context: Optional additional context
            
        Returns:
            WorkerResult containing the output
        """
        # Select appropriate system prompt
        system_prompt = WORKER_PROMPTS.get(
            subtask_type, 
            WORKER_PROMPTS["research"]  # Default to research
        )
        
        # Build the task message
        task_message = f"""## Task: {title}

{description}

## Focus Areas
{chr(10).join(f'- {area}' for area in focus_areas)}
"""
        
        if context:
            task_message += f"\n## Additional Context\n{context}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": task_message}
                ]
            )
            
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content=response.content[0].text,
                success=True
            )
            
        except anthropic.APIError as e:
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content="",
                success=False,
                error=str(e)
            )


# Import Optional for type hints
from typing import Optional


if __name__ == "__main__":
    # Test a single worker
    worker = Worker()
    
    result = worker.execute(
        subtask_id="task_1",
        subtask_type="research",
        title="Environmental Benefits of Electric Vehicles",
        description="Research the positive environmental impacts of widespread electric vehicle adoption, including emissions reduction and air quality improvements.",
        focus_areas=[
            "Reduction in tailpipe emissions",
            "Impact on urban air quality",
            "Comparison with internal combustion engines",
            "Lifecycle environmental considerations"
        ]
    )
    
    if result.success:
        print(f"Worker Result for: {result.subtask_title}")
        print("=" * 50)
        print(result.content)
    else:
        print(f"Worker failed: {result.error}")
```

Notice how each worker type has a specialized prompt. Research workers focus on gathering facts; analysis workers evaluate trade-offs; comparison workers contrast options. This specialization helps produce more focused outputs.

> **💡 Tip:** You can add more worker types for your specific use cases. A "creative" worker might generate ideas, a "technical" worker might focus on implementation details, or a "critique" worker might find problems.

## Task Delegation and Result Collection

Now we need to connect the orchestrator to workers and manage the flow of subtasks and results. This is where the coordination happens.

### The Delegation System

```python
"""
Task delegation and result collection system.

Chapter 23: Orchestrator-Workers - Implementation
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Import our orchestrator and worker classes
# In a real project, these would be proper imports from your package
# For this example, we'll define everything needed inline

import anthropic
import json


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


@dataclass
class TaskPlan:
    """The orchestrator's plan for handling a complex query."""
    original_query: str
    query_analysis: str
    subtasks: list[Subtask]
    synthesis_guidance: str


@dataclass
class WorkerResult:
    """Result from a worker executing a subtask."""
    subtask_id: str
    subtask_title: str
    content: str
    success: bool
    error: Optional[str] = None


class TaskDelegator:
    """
    Manages the delegation of subtasks to workers and 
    collection of results.
    """
    
    def __init__(self, worker, verbose: bool = True):
        """
        Initialize the delegator.
        
        Args:
            worker: Worker instance to execute subtasks
            verbose: Whether to print progress updates
        """
        self.worker = worker
        self.verbose = verbose
        self.results: list[WorkerResult] = []
    
    def delegate_all(
        self,
        plan: TaskPlan,
        context: str = ""
    ) -> list[WorkerResult]:
        """
        Delegate all subtasks in a plan to workers.
        
        Args:
            plan: The TaskPlan containing subtasks
            context: Optional additional context for all workers
            
        Returns:
            List of WorkerResults
        """
        self.results = []
        total = len(plan.subtasks)
        
        if self.verbose:
            print(f"\nExecuting {total} subtasks...")
            print("-" * 50)
        
        for i, subtask in enumerate(plan.subtasks, 1):
            if self.verbose:
                print(f"\n[{i}/{total}] Executing: {subtask.title}")
            
            # Update status
            subtask.status = "in_progress"
            
            # Execute the subtask
            result = self.worker.execute(
                subtask_id=subtask.id,
                subtask_type=subtask.type,
                title=subtask.title,
                description=subtask.description,
                focus_areas=subtask.focus_areas,
                context=context
            )
            
            # Update subtask with result
            if result.success:
                subtask.result = result.content
                subtask.status = "completed"
                if self.verbose:
                    print(f"    ✓ Completed successfully")
            else:
                subtask.status = "failed"
                if self.verbose:
                    print(f"    ✗ Failed: {result.error}")
            
            self.results.append(result)
        
        if self.verbose:
            successful = sum(1 for r in self.results if r.success)
            print(f"\n{'-' * 50}")
            print(f"Completed: {successful}/{total} subtasks successful")
        
        return self.results
    
    def get_successful_results(self) -> list[WorkerResult]:
        """Return only successful results."""
        return [r for r in self.results if r.success]
    
    def get_failed_results(self) -> list[WorkerResult]:
        """Return only failed results."""
        return [r for r in self.results if not r.success]


# Worker prompts and class (from previous section)
WORKER_PROMPTS = {
    "research": """You are a thorough research assistant. Provide comprehensive, factual information on a specific topic.
Focus on key findings, supporting details, examples, and important caveats.""",
    
    "analysis": """You are an analytical expert. Analyze the topic, evaluating implications, trade-offs, and significance.
Provide balanced analysis with pros/cons and meaningful conclusions.""",
    
    "comparison": """You are a comparison specialist. Compare different options or perspectives fairly.
Use clear criteria, evaluate each option, and provide contextual recommendations."""
}


class Worker:
    """Executes individual subtasks."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def execute(
        self,
        subtask_id: str,
        subtask_type: str,
        title: str,
        description: str,
        focus_areas: list[str],
        context: str = ""
    ) -> WorkerResult:
        """Execute a single subtask."""
        system_prompt = WORKER_PROMPTS.get(subtask_type, WORKER_PROMPTS["research"])
        
        task_message = f"## Task: {title}\n\n{description}\n\n"
        task_message += f"## Focus Areas\n"
        task_message += "\n".join(f"- {area}" for area in focus_areas)
        
        if context:
            task_message += f"\n\n## Additional Context\n{context}"
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": task_message}]
            )
            
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content=response.content[0].text,
                success=True
            )
        except anthropic.APIError as e:
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content="",
                success=False,
                error=str(e)
            )


if __name__ == "__main__":
    # Create a simple test plan
    plan = TaskPlan(
        original_query="What are the pros and cons of remote work?",
        query_analysis="User wants to understand benefits and drawbacks of remote work",
        subtasks=[
            Subtask(
                id="task_1",
                type="research",
                title="Benefits of Remote Work",
                description="Research the key benefits of remote work for employees",
                focus_areas=["Flexibility", "Work-life balance", "Cost savings"]
            ),
            Subtask(
                id="task_2",
                type="research",
                title="Challenges of Remote Work",
                description="Research the challenges and drawbacks of remote work",
                focus_areas=["Isolation", "Communication", "Work-life boundaries"]
            ),
            Subtask(
                id="task_3",
                type="analysis",
                title="Remote Work Trade-offs Analysis",
                description="Analyze the trade-offs between remote and office work",
                focus_areas=["Productivity", "Collaboration", "Career growth"]
            )
        ],
        synthesis_guidance="Combine into balanced view showing both sides"
    )
    
    # Create worker and delegator
    worker = Worker()
    delegator = TaskDelegator(worker, verbose=True)
    
    # Execute all subtasks
    results = delegator.delegate_all(plan)
    
    # Show summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    for result in delegator.get_successful_results():
        print(f"\n### {result.subtask_title}")
        print(result.content[:500] + "..." if len(result.content) > 500 else result.content)
```

The `TaskDelegator` class handles the coordination between the orchestrator's plan and the workers. It:

1. Iterates through all subtasks in the plan
2. Updates status as each subtask progresses
3. Collects results (successful and failed)
4. Provides progress feedback

## Synthesizing Worker Outputs

The final piece is synthesis—taking all the worker results and combining them into a coherent response. This is where the orchestrator comes back into play.

### The Synthesis Implementation

```python
"""
Synthesis of worker outputs into final response.

Chapter 23: Orchestrator-Workers - Implementation
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing information from multiple sources into coherent, comprehensive responses.

## Your Task
You will receive:
1. An original query from a user
2. Multiple research/analysis results from specialized workers
3. Guidance on how to combine the results

Your job is to synthesize these inputs into a single, well-organized response that:
- Directly addresses the original query
- Integrates insights from all worker results
- Maintains a logical flow and structure
- Highlights key findings and conclusions
- Notes any conflicting information or caveats
- Provides actionable insights where appropriate

## Output Guidelines
- Write in a clear, professional style
- Use headers to organize major sections
- Lead with the most important findings
- Support claims with details from the research
- End with clear conclusions or recommendations
- Keep the response comprehensive but focused"""


@dataclass
class WorkerResult:
    """Result from a worker executing a subtask."""
    subtask_id: str
    subtask_title: str
    content: str
    success: bool
    error: Optional[str] = None


class Synthesizer:
    """
    Synthesizes multiple worker results into a coherent final response.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def synthesize(
        self,
        original_query: str,
        results: list[WorkerResult],
        synthesis_guidance: str = ""
    ) -> str:
        """
        Synthesize worker results into a final response.
        
        Args:
            original_query: The user's original question
            results: List of WorkerResults to synthesize
            synthesis_guidance: Optional guidance on how to combine
            
        Returns:
            Synthesized response as a string
        """
        # Filter to successful results only
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return "Unable to generate a response. All subtasks failed."
        
        # Build the synthesis request
        synthesis_request = f"""## Original Query
{original_query}

## Research Results

"""
        
        for i, result in enumerate(successful_results, 1):
            synthesis_request += f"""### {i}. {result.subtask_title}

{result.content}

---

"""
        
        if synthesis_guidance:
            synthesis_request += f"""## Synthesis Guidance
{synthesis_guidance}

"""
        
        synthesis_request += "Please synthesize these results into a comprehensive response to the original query."
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": synthesis_request}
            ]
        )
        
        return response.content[0].text


if __name__ == "__main__":
    # Test synthesis with mock results
    synthesizer = Synthesizer()
    
    mock_results = [
        WorkerResult(
            subtask_id="task_1",
            subtask_title="Benefits of Electric Vehicles",
            content="""Key findings on EV benefits:
            
1. **Zero Direct Emissions**: EVs produce no tailpipe emissions, reducing urban air pollution.

2. **Lower Operating Costs**: Electricity is cheaper than gasoline per mile, with fewer maintenance needs.

3. **Reduced Noise Pollution**: Electric motors are significantly quieter than combustion engines.

4. **Energy Efficiency**: EVs convert 85-90% of energy to motion vs. 20-30% for gas vehicles.""",
            success=True
        ),
        WorkerResult(
            subtask_id="task_2",
            subtask_title="Challenges of Electric Vehicles",
            content="""Key challenges for EV adoption:

1. **Range Anxiety**: Limited range compared to gas vehicles, though improving rapidly.

2. **Charging Infrastructure**: Insufficient public charging stations in many areas.

3. **Higher Upfront Cost**: EVs typically cost more to purchase, though prices are declining.

4. **Battery Production**: Mining for battery materials has environmental and ethical concerns.""",
            success=True
        ),
        WorkerResult(
            subtask_id="task_3",
            subtask_title="Economic Impact Analysis",
            content="""Economic analysis of EV adoption:

1. **Job Transformation**: Shift from traditional auto jobs to EV manufacturing and charging infrastructure.

2. **Energy Grid Impact**: Increased electricity demand requires grid upgrades and planning.

3. **Consumer Savings**: 5-year total cost of ownership often favors EVs despite higher purchase price.

4. **Urban Planning**: Cities rethinking parking and infrastructure for charging needs.""",
            success=True
        )
    ]
    
    result = synthesizer.synthesize(
        original_query="What are the impacts of electric vehicle adoption?",
        results=mock_results,
        synthesis_guidance="Balance environmental benefits against practical challenges, and address economic factors."
    )
    
    print("SYNTHESIZED RESPONSE")
    print("=" * 50)
    print(result)
```

The synthesizer doesn't just concatenate results—it intelligently weaves them together, resolves overlaps, and creates a narrative that directly addresses the original query.

## The Complete Orchestrator Class

Now let's bring everything together into a single, reusable `ResearchOrchestrator` class:

```python
"""
Complete Research Orchestrator implementation.

Chapter 23: Orchestrator-Workers - Implementation

This module provides a complete orchestrator-workers implementation
for handling complex research queries.
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import anthropic

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
# Prompts
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

Respond ONLY with valid JSON."""


WORKER_PROMPTS = {
    "research": """You are a thorough research assistant. Provide comprehensive, factual information.

Guidelines:
- Focus on factual, verifiable information
- Organize with clear key points
- Include specific examples or data
- Note important caveats

Structure your response with:
1. Key findings (3-5 main points)
2. Supporting details
3. Notable examples
4. Important caveats""",

    "analysis": """You are an analytical expert. Evaluate implications, trade-offs, and significance.

Guidelines:
- Consider multiple perspectives
- Identify pros and cons
- Support analysis with reasoning
- Draw meaningful conclusions

Structure your response with:
1. Overview
2. Key factors
3. Implications (positive/negative)
4. Conclusions""",

    "comparison": """You are a comparison specialist. Compare options fairly using clear criteria.

Guidelines:
- Use clear comparison criteria
- Evaluate each option fairly
- Highlight differences and similarities
- Provide contextual recommendations

Structure your response with:
1. Options being compared
2. Comparison criteria
3. Evaluation of each
4. Key differences
5. Recommendations"""
}


SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing information into coherent responses.

Guidelines:
- Directly address the original query
- Integrate insights from all sources
- Maintain logical flow
- Highlight key findings
- Note conflicts or caveats
- Provide actionable insights

Structure:
- Use headers to organize sections
- Lead with most important findings
- Support claims with research details
- End with clear conclusions"""


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
        import time
        start_time = time.time()
        
        system_prompt = WORKER_PROMPTS.get(subtask.type, WORKER_PROMPTS["research"])
        
        task_message = f"## Task: {subtask.title}\n\n{subtask.description}\n\n"
        task_message += "## Focus Areas\n"
        task_message += "\n".join(f"- {area}" for area in subtask.focus_areas)
        
        if context:
            task_message += f"\n\n## Additional Context\n{context}"
        
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
        
        synthesis_request = f"## Original Query\n{original_query}\n\n## Research Results\n\n"
        
        for i, result in enumerate(successful_results, 1):
            synthesis_request += f"### {i}. {result.subtask_title}\n\n{result.content}\n\n---\n\n"
        
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
    delegates work to specialized workers, and synthesizes results.
    
    Example:
        orchestrator = ResearchOrchestrator()
        result = orchestrator.research("What are the impacts of AI on healthcare?")
        print(result.synthesis)
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
    
    def _log(self, message: str):
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
                {"role": "user", "content": f"Break down this query into subtasks:\n\n{query}"}
            ]
        )
        
        response_text = response.content[0].text
        
        try:
            plan_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                raise ValueError("Orchestrator returned invalid JSON")
        
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
        
        self._log(f"   Created plan with {len(subtasks)} subtasks")
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
        
        Args:
            query: The research question to answer
            context: Optional additional context
            
        Returns:
            OrchestratorResult containing plan, results, and synthesis
        """
        import time
        start_time = time.time()
        
        self._log(f"\n{'='*60}")
        self._log(f"🔍 Research Query: {query}")
        self._log(f"{'='*60}")
        
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
        self._log(f"✅ Complete! ({completed}/{len(results)} subtasks successful)")
        self._log(f"   Total time: {total_time:.1f}s")
        self._log(f"{'='*60}\n")
        
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
        Useful for previewing how a query will be broken down.
        """
        return self._create_plan(query)


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
    
    # Display results
    print("\n" + "=" * 60)
    print("FINAL SYNTHESIS")
    print("=" * 60)
    print(result.synthesis)
    
    # Show metrics
    print("\n" + "=" * 60)
    print("EXECUTION METRICS")
    print("=" * 60)
    print(f"Query: {result.query}")
    print(f"Subtasks created: {len(result.plan.subtasks)}")
    print(f"Subtasks completed: {result.subtasks_completed}")
    print(f"Subtasks failed: {result.subtasks_failed}")
    print(f"Total execution time: {result.total_time:.1f}s")
```

This complete class provides:

- **Clean API**: Just call `orchestrator.research(query)` 
- **Full visibility**: Track plans, results, and timing
- **Error resilience**: Continues even if some subtasks fail
- **Configurability**: Adjust model, subtask limits, verbosity

## Using the Orchestrator

Here's how to use the `ResearchOrchestrator` in practice:

```python
"""
Example usage of the Research Orchestrator.

Chapter 23: Orchestrator-Workers - Implementation
"""

from research_orchestrator import ResearchOrchestrator

# Create orchestrator
orchestrator = ResearchOrchestrator(verbose=True)

# Simple usage - just pass a query
result = orchestrator.research(
    "How is artificial intelligence transforming the healthcare industry?"
)

# Access the synthesized answer
print(result.synthesis)

# Access individual worker results if needed
for worker_result in result.worker_results:
    if worker_result.success:
        print(f"\n--- {worker_result.subtask_title} ---")
        print(worker_result.content)

# Check metrics
print(f"\nCompleted in {result.total_time:.1f} seconds")
print(f"Success rate: {result.subtasks_completed}/{len(result.worker_results)}")

# Preview a plan without executing
plan = orchestrator.get_plan_only(
    "What are the long-term effects of social media on mental health?"
)

for subtask in plan.subtasks:
    print(f"- [{subtask.type}] {subtask.title}")
```

## Common Pitfalls

### 1. Overlapping Subtasks

**Problem:** The orchestrator creates subtasks that cover the same ground.

```python
# Bad: These subtasks overlap significantly
subtasks = [
    {"title": "Benefits of remote work", ...},
    {"title": "Advantages of working from home", ...},  # Same thing!
]
```

**Solution:** Improve the orchestrator prompt to emphasize distinct, non-overlapping subtasks:

```python
ORCHESTRATOR_SYSTEM_PROMPT = """...
## Critical Requirements
- Subtasks must NOT overlap in scope
- Each subtask should address a UNIQUE aspect
- If two subtasks seem similar, combine them
..."""
```

### 2. Missing Synthesis Context

**Problem:** The synthesizer doesn't understand how pieces fit together.

**Solution:** Always provide synthesis guidance and include the original query:

```python
synthesis = self.synthesizer.synthesize(
    original_query=plan.original_query,  # Always include!
    results=results,
    synthesis_guidance=plan.synthesis_guidance  # Guide the synthesis
)
```

### 3. Ignoring Failed Subtasks

**Problem:** If critical subtasks fail, the synthesis may be incomplete.

**Solution:** Check for failures and handle gracefully:

```python
if result.subtasks_failed > 0:
    print(f"Warning: {result.subtasks_failed} subtasks failed")
    for wr in result.worker_results:
        if not wr.success:
            print(f"  - {wr.subtask_title}: {wr.error}")
```

## Practical Exercise

**Task:** Extend the `ResearchOrchestrator` to support parallel worker execution.

**Requirements:**
1. Add an `async_research` method that runs workers in parallel using `asyncio`
2. Workers should execute simultaneously rather than sequentially
3. Track and report time savings from parallelization
4. Handle errors in parallel execution gracefully

**Hints:**
- Use `asyncio.gather()` to run multiple coroutines
- Create an async version of the `Worker.execute` method
- The Anthropic SDK supports async with `AsyncAnthropic`
- Be mindful of rate limits when running parallel requests

**Solution:** See `code/exercise_parallel_orchestrator.py`

## Key Takeaways

- **The orchestrator prompt is critical**—it determines the quality of task decomposition. Invest time in crafting clear instructions for how to break down queries.

- **Workers should be specialized**—different task types (research, analysis, comparison) benefit from different prompts and approaches.

- **Synthesis transforms data into insight**—don't just concatenate results; use an LLM to intelligently weave findings together.

- **Track everything**—execution times, success rates, and individual results help you debug and optimize.

- **The pattern is flexible**—adapt worker types, subtask limits, and synthesis approaches to your specific use case.

## What's Next

In Chapter 24, we'll explore the final workflow pattern: **Evaluator-Optimizer**. This pattern uses one LLM to generate content and another to evaluate and improve it—creating an iterative refinement loop. You'll learn when iterative improvement beats single-shot generation and how to design effective evaluation criteria.
