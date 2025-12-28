"""
Exercise Solution: Document Analyzer Design

This file contains the complete design for a Document Analyzer
orchestrator-workers system, demonstrating how to think through
and design such a system.

Chapter 22: Orchestrator-Workers - Concept and Design
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# =============================================================================
# Worker Definitions
# =============================================================================

@dataclass
class WorkerDesign:
    """Design specification for a worker."""
    name: str
    responsibility: str
    input_requirements: list[str]
    output_format: dict
    example_use_cases: list[str]


# Define the workers for our Document Analyzer system

STRUCTURE_ANALYZER = WorkerDesign(
    name="structure_analyzer",
    responsibility=(
        "Analyzes the document's structure and organization. "
        "Identifies sections, headings, document type, and overall layout."
    ),
    input_requirements=[
        "document_text: The full text of the document",
        "document_type_hint: Optional hint about expected document type"
    ],
    output_format={
        "document_type": "contract | report | article | letter | other",
        "sections": [
            {
                "title": "Section title or identifier",
                "start_position": "Character position where section starts",
                "content_summary": "Brief summary of section contents"
            }
        ],
        "structure_quality": "well-organized | adequate | poor",
        "suggested_improvements": ["List of structural improvements"]
    },
    example_use_cases=[
        "Identifying the sections of a legal contract",
        "Mapping the chapters of a technical report",
        "Understanding the flow of a news article"
    ]
)


CONTENT_EXTRACTOR = WorkerDesign(
    name="content_extractor",
    responsibility=(
        "Extracts key information from the document. "
        "Identifies main points, key entities, dates, numbers, and facts."
    ),
    input_requirements=[
        "document_text: The full text of the document",
        "structure_info: Output from structure_analyzer (optional)",
        "extraction_focus: Specific types of information to prioritize"
    ],
    output_format={
        "main_points": ["List of key points or arguments"],
        "entities": {
            "people": ["Named individuals"],
            "organizations": ["Companies, institutions"],
            "locations": ["Places mentioned"],
            "dates": ["Important dates and deadlines"]
        },
        "numbers_and_figures": [
            {
                "value": "The number or amount",
                "context": "What this number represents"
            }
        ],
        "key_terms": ["Important terminology used"]
    },
    example_use_cases=[
        "Extracting party names and dates from a contract",
        "Finding key statistics in a research report",
        "Identifying sources cited in a news article"
    ]
)


IMPLICATIONS_ANALYZER = WorkerDesign(
    name="implications_analyzer",
    responsibility=(
        "Analyzes implications, risks, and important considerations. "
        "Identifies what the document means for the reader and potential issues."
    ),
    input_requirements=[
        "document_text: The full text of the document",
        "extracted_content: Output from content_extractor",
        "reader_context: Information about who is reading and why"
    ],
    output_format={
        "implications": [
            {
                "implication": "What this means for the reader",
                "severity": "high | medium | low",
                "relevant_section": "Which part of document this relates to"
            }
        ],
        "risks": [
            {
                "risk": "Potential issue or concern",
                "likelihood": "high | medium | low",
                "mitigation": "How to address this risk"
            }
        ],
        "opportunities": ["Positive aspects or opportunities identified"],
        "questions_to_consider": ["Questions the reader should think about"]
    },
    example_use_cases=[
        "Identifying unfavorable terms in a contract",
        "Finding gaps in a project report",
        "Assessing bias in a news article"
    ]
)


SUMMARY_WRITER = WorkerDesign(
    name="summary_writer",
    responsibility=(
        "Creates clear, concise summaries of documents. "
        "Synthesizes information into readable overviews of varying lengths."
    ),
    input_requirements=[
        "document_text: The full text of the document (for reference)",
        "structure_info: Output from structure_analyzer",
        "extracted_content: Output from content_extractor",
        "implications: Output from implications_analyzer",
        "summary_length: Target length (brief | standard | detailed)"
    ],
    output_format={
        "executive_summary": "1-2 paragraph overview for quick reading",
        "detailed_summary": "Comprehensive summary covering all key points",
        "section_summaries": [
            {
                "section": "Section name",
                "summary": "Summary of that section"
            }
        ],
        "one_sentence_summary": "The document in one sentence"
    },
    example_use_cases=[
        "Creating an executive summary of a contract",
        "Summarizing a lengthy research report",
        "Creating a brief for a news article"
    ]
)


RECOMMENDATION_ENGINE = WorkerDesign(
    name="recommendation_engine",
    responsibility=(
        "Provides actionable recommendations based on document analysis. "
        "Suggests next steps, decisions, and actions for the reader."
    ),
    input_requirements=[
        "document_summary: Output from summary_writer",
        "implications: Output from implications_analyzer",
        "reader_context: Information about the reader's goals and constraints"
    ],
    output_format={
        "primary_recommendation": "The most important action to take",
        "additional_recommendations": [
            {
                "recommendation": "Suggested action",
                "rationale": "Why this is recommended",
                "priority": "high | medium | low"
            }
        ],
        "decision_points": ["Key decisions the reader needs to make"],
        "follow_up_actions": ["Actions to take after initial review"]
    },
    example_use_cases=[
        "Suggesting negotiation points for a contract",
        "Recommending action items from a report",
        "Suggesting follow-up research for an article"
    ]
)


# =============================================================================
# Orchestrator System Prompt
# =============================================================================

DOCUMENT_ANALYZER_ORCHESTRATOR_PROMPT = """
You are a Document Analysis Orchestrator. Your role is to analyze documents
comprehensively by coordinating specialized workers.

## Your Responsibilities

1. ANALYZE the document and user's request to understand what analysis is needed
2. CREATE a task plan that delegates to appropriate workers
3. SYNTHESIZE worker results into a coherent, useful response

## Available Workers

### structure_analyzer
Analyzes the document's structure and organization. Identifies sections,
headings, document type, and overall layout.
Best for: Understanding document organization, identifying sections.

### content_extractor  
Extracts key information from the document. Identifies main points, key
entities, dates, numbers, and facts.
Best for: Finding specific information, extracting data points.

### implications_analyzer
Analyzes implications, risks, and important considerations. Identifies
what the document means for the reader.
Best for: Risk assessment, understanding consequences.

### summary_writer
Creates clear, concise summaries of documents. Synthesizes information
into readable overviews.
Best for: Creating summaries, executive briefs.

### recommendation_engine
Provides actionable recommendations based on document analysis.
Best for: Suggesting next steps, decision support.

## Planning Guidelines

1. Always start with structure_analyzer for unfamiliar documents
2. content_extractor should run after structure_analyzer when both are needed
3. implications_analyzer requires content_extractor output
4. summary_writer works best with all previous analyses
5. recommendation_engine is only needed when user wants action items

## Document Type Considerations

### Legal Contracts
Required: structure_analyzer, content_extractor, implications_analyzer
Optional: summary_writer, recommendation_engine

### Technical Reports  
Required: structure_analyzer, content_extractor, summary_writer
Optional: implications_analyzer, recommendation_engine

### News Articles
Required: content_extractor, summary_writer
Optional: structure_analyzer, implications_analyzer

## Task Plan Format

Respond with JSON:
{
    "document_type": "Identified document type",
    "analysis_needed": "Why this analysis is appropriate",
    "tasks": [
        {
            "id": "task_1",
            "worker": "worker_name",
            "description": "Specific instructions for the worker",
            "dependencies": [],
            "priority": "high | medium | low"
        }
    ]
}

## Example Task Plan

For request: "Review this employment contract and highlight any concerns"

{
    "document_type": "legal_contract",
    "analysis_needed": "User wants to understand contract terms and potential issues",
    "tasks": [
        {
            "id": "task_1",
            "worker": "structure_analyzer",
            "description": "Identify all sections of this employment contract",
            "dependencies": [],
            "priority": "high"
        },
        {
            "id": "task_2", 
            "worker": "content_extractor",
            "description": "Extract key terms: salary, benefits, termination clauses, non-compete, intellectual property",
            "dependencies": ["task_1"],
            "priority": "high"
        },
        {
            "id": "task_3",
            "worker": "implications_analyzer",
            "description": "Identify concerning terms, unfavorable clauses, and potential risks for the employee",
            "dependencies": ["task_2"],
            "priority": "high"
        },
        {
            "id": "task_4",
            "worker": "summary_writer",
            "description": "Create a brief summary highlighting the most important terms",
            "dependencies": ["task_2"],
            "priority": "medium"
        },
        {
            "id": "task_5",
            "worker": "recommendation_engine",
            "description": "Suggest negotiation points and questions to ask the employer",
            "dependencies": ["task_3"],
            "priority": "medium"
        }
    ]
}

## Synthesis Guidelines

When combining worker results:
1. Lead with the most important findings
2. Group related information together
3. Be specific about which document sections findings relate to
4. Clearly distinguish facts from implications
5. End with actionable recommendations if appropriate
"""


# =============================================================================
# Dependency Diagrams
# =============================================================================

def print_dependency_diagram(document_type: str, tasks: list[dict]):
    """Print an ASCII dependency diagram for a task plan."""
    print(f"\nDependency Diagram: {document_type}")
    print("=" * 50)
    
    # Build dependency map
    task_names = {t["id"]: t["worker"] for t in tasks}
    
    # Group tasks by level (based on dependencies)
    levels: dict[int, list[str]] = {}
    task_levels: dict[str, int] = {}
    
    def get_level(task_id: str, tasks_dict: dict) -> int:
        task = tasks_dict[task_id]
        if not task["dependencies"]:
            return 0
        return max(get_level(dep, tasks_dict) for dep in task["dependencies"]) + 1
    
    tasks_dict = {t["id"]: t for t in tasks}
    for task in tasks:
        level = get_level(task["id"], tasks_dict)
        task_levels[task["id"]] = level
        if level not in levels:
            levels[level] = []
        levels[level].append(task["id"])
    
    # Print levels
    max_level = max(levels.keys())
    for level in range(max_level + 1):
        if level > 0:
            print("        │")
            print("        ▼")
        
        level_tasks = levels.get(level, [])
        task_strs = [f"[{task_names[t]}]" for t in level_tasks]
        print(f"Level {level}: {' + '.join(task_strs)}")
    
    print()


# Legal Contract Example
LEGAL_CONTRACT_TASKS = [
    {
        "id": "task_1",
        "worker": "structure_analyzer",
        "description": "Identify contract sections",
        "dependencies": [],
        "priority": "high"
    },
    {
        "id": "task_2",
        "worker": "content_extractor",
        "description": "Extract key terms and clauses",
        "dependencies": ["task_1"],
        "priority": "high"
    },
    {
        "id": "task_3",
        "worker": "implications_analyzer",
        "description": "Identify risks and concerns",
        "dependencies": ["task_2"],
        "priority": "high"
    },
    {
        "id": "task_4",
        "worker": "recommendation_engine",
        "description": "Suggest negotiation points",
        "dependencies": ["task_3"],
        "priority": "medium"
    }
]

# Technical Report Example
TECHNICAL_REPORT_TASKS = [
    {
        "id": "task_1",
        "worker": "structure_analyzer",
        "description": "Map report sections",
        "dependencies": [],
        "priority": "high"
    },
    {
        "id": "task_2",
        "worker": "content_extractor",
        "description": "Extract key findings and data",
        "dependencies": ["task_1"],
        "priority": "high"
    },
    {
        "id": "task_3",
        "worker": "summary_writer",
        "description": "Create executive summary",
        "dependencies": ["task_2"],
        "priority": "high"
    }
]

# News Article Example
NEWS_ARTICLE_TASKS = [
    {
        "id": "task_1",
        "worker": "content_extractor",
        "description": "Extract key facts and sources",
        "dependencies": [],
        "priority": "high"
    },
    {
        "id": "task_2",
        "worker": "implications_analyzer",
        "description": "Assess bias and perspective",
        "dependencies": ["task_1"],
        "priority": "medium"
    },
    {
        "id": "task_3",
        "worker": "summary_writer",
        "description": "Create brief summary",
        "dependencies": ["task_1"],
        "priority": "high"
    }
]


# =============================================================================
# Demonstration
# =============================================================================

def demonstrate_design():
    """Display the complete Document Analyzer design."""
    
    print("=" * 70)
    print("DOCUMENT ANALYZER SYSTEM DESIGN")
    print("Exercise Solution - Chapter 22")
    print("=" * 70)
    
    # Display workers
    print("\n" + "=" * 70)
    print("WORKER DESIGNS")
    print("=" * 70)
    
    workers = [
        STRUCTURE_ANALYZER,
        CONTENT_EXTRACTOR,
        IMPLICATIONS_ANALYZER,
        SUMMARY_WRITER,
        RECOMMENDATION_ENGINE
    ]
    
    for i, worker in enumerate(workers, 1):
        print(f"\n{i}. {worker.name.upper()}")
        print("-" * 50)
        print(f"Responsibility: {worker.responsibility}")
        print(f"\nInput Requirements:")
        for req in worker.input_requirements:
            print(f"  • {req}")
        print(f"\nOutput Format:")
        for key, value in worker.output_format.items():
            print(f"  • {key}: {type(value).__name__}")
        print(f"\nExample Use Cases:")
        for case in worker.example_use_cases:
            print(f"  • {case}")
    
    # Display dependency diagrams
    print("\n" + "=" * 70)
    print("DEPENDENCY DIAGRAMS")
    print("=" * 70)
    
    print_dependency_diagram("Legal Contract", LEGAL_CONTRACT_TASKS)
    print_dependency_diagram("Technical Report", TECHNICAL_REPORT_TASKS)
    print_dependency_diagram("News Article", NEWS_ARTICLE_TASKS)
    
    # Display orchestrator prompt summary
    print("\n" + "=" * 70)
    print("ORCHESTRATOR PROMPT")
    print("=" * 70)
    
    print("\nThe orchestrator prompt includes:")
    print("  ✓ Worker descriptions with capabilities")
    print("  ✓ Planning guidelines for task creation")
    print("  ✓ Document type considerations")
    print("  ✓ Task plan JSON format")
    print("  ✓ Example task plan")
    print("  ✓ Synthesis guidelines")
    
    print(f"\nFull prompt length: {len(DOCUMENT_ANALYZER_ORCHESTRATOR_PROMPT)} characters")
    
    # Design rationale
    print("\n" + "=" * 70)
    print("DESIGN RATIONALE")
    print("=" * 70)
    
    print("""
Why these five workers?

1. STRUCTURE_ANALYZER - Foundation for understanding any document
   - Must run first for complex documents
   - Enables other workers to reference specific sections
   
2. CONTENT_EXTRACTOR - Core information gathering
   - Works best with structure context
   - Provides data for all other analyses
   
3. IMPLICATIONS_ANALYZER - Risk/opportunity identification
   - Requires extracted content to analyze
   - Critical for contracts and business documents
   
4. SUMMARY_WRITER - Synthesis and presentation
   - Can work with partial or complete analysis
   - Produces the main deliverable for many use cases
   
5. RECOMMENDATION_ENGINE - Actionable output
   - Only needed when user wants guidance
   - Requires implications analysis for good recommendations

Key Design Decisions:

• Workers are SPECIALISTS, not generalists
  - Each does one thing well
  - Easy to replace or upgrade individual workers
  
• Clear DEPENDENCIES modeled
  - Prevents running workers without needed context
  - Enables parallel execution where possible
  
• DOCUMENT TYPE influences worker selection
  - Not all workers needed for all document types
  - Optimizes cost and latency
  
• SYNTHESIS happens in orchestrator
  - Workers return structured data
  - Orchestrator combines for final response
""")
    
    print("=" * 70)
    print("See Chapter 23 for the implementation of this design!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_design()
