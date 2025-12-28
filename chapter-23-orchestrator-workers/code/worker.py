"""
Worker implementation for executing subtasks.

Chapter 23: Orchestrator-Workers - Implementation

This module implements workers that execute individual subtasks
assigned by the orchestrator.
"""

import os
import time
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# Worker System Prompts
# =============================================================================

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
4. Important caveats or considerations

Write in a clear, professional style. Be comprehensive but concise.""",

    "analysis": """You are an analytical expert. Your task is to analyze a topic, evaluating its implications, trade-offs, and significance.

## Guidelines
- Provide balanced analysis considering multiple perspectives
- Identify pros and cons, benefits and risks
- Evaluate significance and implications
- Support analysis with reasoning
- Draw meaningful conclusions

## Output Format
Provide a structured analysis with:
1. Overview of the issue being analyzed
2. Key factors to consider
3. Analysis of implications (both positive and negative)
4. Your conclusions and insights

Be balanced and objective in your assessment.""",

    "comparison": """You are a comparison specialist. Your task is to compare different options, approaches, or perspectives on a topic.

## Guidelines
- Identify clear criteria for comparison
- Evaluate each option fairly
- Highlight key differences and similarities
- Note context-dependent factors
- Provide actionable insights

## Output Format
Provide a structured comparison with:
1. Options or perspectives being compared
2. Comparison criteria used
3. Evaluation of each option against criteria
4. Summary of key differences
5. Contextual recommendations

Be fair and thorough in your comparisons.""",

    "default": """You are a helpful assistant. Complete the assigned task thoroughly and accurately.

## Guidelines
- Focus on the task at hand
- Be comprehensive but concise
- Organize your response clearly
- Note any limitations or caveats

Provide a well-structured response that directly addresses the task."""
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WorkerResult:
    """
    Result from a worker executing a subtask.
    
    Attributes:
        subtask_id: ID of the subtask that was executed
        subtask_title: Title of the subtask
        content: The worker's output content
        success: Whether execution succeeded
        error: Error message if execution failed
        execution_time: Time taken to execute (seconds)
    """
    subtask_id: str
    subtask_title: str
    content: str
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None


# =============================================================================
# Worker Class
# =============================================================================

class Worker:
    """
    Executes individual subtasks assigned by the orchestrator.
    
    Workers are specialized by task type (research, analysis, comparison)
    and produce structured outputs suitable for synthesis.
    
    Example:
        worker = Worker()
        result = worker.execute(
            subtask_id="task_1",
            subtask_type="research",
            title="Benefits of Electric Vehicles",
            description="Research the environmental benefits of EVs",
            focus_areas=["emissions", "air quality", "energy efficiency"]
        )
        print(result.content)
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the worker.
        
        Args:
            model: Claude model to use for task execution
        """
        self.client = anthropic.Anthropic()
        self.model = model
    
    def get_system_prompt(self, subtask_type: str) -> str:
        """
        Get the appropriate system prompt for a subtask type.
        
        Args:
            subtask_type: Type of subtask (research, analysis, comparison)
            
        Returns:
            System prompt string for the worker
        """
        return WORKER_PROMPTS.get(subtask_type, WORKER_PROMPTS["default"])
    
    def build_task_message(
        self,
        title: str,
        description: str,
        focus_areas: list[str],
        context: str = ""
    ) -> str:
        """
        Build the task message to send to the LLM.
        
        Args:
            title: Task title
            description: Task description
            focus_areas: Areas to focus on
            context: Additional context
            
        Returns:
            Formatted task message
        """
        message = f"""## Task: {title}

{description}

## Focus Areas
"""
        for area in focus_areas:
            message += f"- {area}\n"
        
        if context:
            message += f"\n## Additional Context\n{context}\n"
        
        message += "\nPlease complete this task thoroughly, addressing all focus areas."
        
        return message
    
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
        start_time = time.time()
        
        # Get appropriate system prompt
        system_prompt = self.get_system_prompt(subtask_type)
        
        # Build the task message
        task_message = self.build_task_message(
            title=title,
            description=description,
            focus_areas=focus_areas,
            context=context
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": task_message}
                ]
            )
            
            execution_time = time.time() - start_time
            
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content=response.content[0].text,
                success=True,
                execution_time=execution_time
            )
            
        except anthropic.APIConnectionError as e:
            execution_time = time.time() - start_time
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content="",
                success=False,
                error=f"Connection error: {e}",
                execution_time=execution_time
            )
            
        except anthropic.RateLimitError as e:
            execution_time = time.time() - start_time
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content="",
                success=False,
                error=f"Rate limit exceeded: {e}",
                execution_time=execution_time
            )
            
        except anthropic.APIStatusError as e:
            execution_time = time.time() - start_time
            return WorkerResult(
                subtask_id=subtask_id,
                subtask_title=title,
                content="",
                success=False,
                error=f"API error ({e.status_code}): {e.message}",
                execution_time=execution_time
            )


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create a worker
    worker = Worker()
    
    # Test different task types
    test_tasks = [
        {
            "subtask_id": "task_1",
            "subtask_type": "research",
            "title": "Environmental Benefits of Electric Vehicles",
            "description": "Research the positive environmental impacts of widespread electric vehicle adoption, including emissions reduction and air quality improvements.",
            "focus_areas": [
                "Reduction in tailpipe emissions",
                "Impact on urban air quality",
                "Energy efficiency compared to ICE vehicles",
                "Lifecycle environmental considerations"
            ]
        },
        {
            "subtask_id": "task_2",
            "subtask_type": "analysis",
            "title": "Economic Trade-offs of EV Adoption",
            "description": "Analyze the economic implications of transitioning to electric vehicles, considering both costs and benefits.",
            "focus_areas": [
                "Upfront costs vs long-term savings",
                "Impact on auto industry jobs",
                "Infrastructure investment needs",
                "Consumer financial considerations"
            ]
        },
        {
            "subtask_id": "task_3",
            "subtask_type": "comparison",
            "title": "Urban vs Suburban EV Adoption",
            "description": "Compare the challenges and benefits of electric vehicle adoption in urban versus suburban areas.",
            "focus_areas": [
                "Charging infrastructure availability",
                "Driving patterns and range needs",
                "Parking and home charging options",
                "Public transportation alternatives"
            ]
        }
    ]
    
    # Execute first task as demonstration
    task = test_tasks[0]
    
    print(f"Executing task: {task['title']}")
    print(f"Type: {task['subtask_type']}")
    print("-" * 50)
    
    result = worker.execute(**task)
    
    if result.success:
        print(f"\n✓ Completed in {result.execution_time:.1f}s\n")
        print("RESULT:")
        print("=" * 50)
        print(result.content)
    else:
        print(f"\n✗ Failed: {result.error}")
    
    # Show available task types
    print("\n" + "=" * 50)
    print("AVAILABLE WORKER TYPES:")
    print("=" * 50)
    for task_type in WORKER_PROMPTS.keys():
        if task_type != "default":
            print(f"  - {task_type}")
