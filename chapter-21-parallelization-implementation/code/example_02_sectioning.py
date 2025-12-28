"""
Sectioning pattern: parallel independent subtasks.

Chapter 21: Parallelization - Implementation

Sectioning divides a large task into independent subtasks that
run in parallel. Each subtask handles a different portion of
the work, and results are merged at the end.
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class Section:
    """
    Defines a section to process in parallel.
    
    Attributes:
        name: Identifier for this section
        prompt_template: Template with {input} placeholder
        system_prompt: Optional system prompt for this section
    """
    name: str
    prompt_template: str
    system_prompt: str | None = None


@dataclass
class SectionResult:
    """
    Result from processing a single section.
    
    Attributes:
        name: The section identifier
        content: The generated content
        success: Whether processing succeeded
        error: Error message if failed
        execution_time: How long this section took
    """
    name: str
    content: str
    success: bool
    error: str | None = None
    execution_time: float = 0.0


class SectioningWorkflow:
    """
    Implements the sectioning pattern for parallel subtasks.
    
    Divides work into independent sections, processes them in parallel,
    and merges the results into a cohesive output.
    
    Example usage:
        workflow = SectioningWorkflow()
        sections = [
            Section("intro", "Write an introduction about: {input}"),
            Section("details", "Provide detailed analysis of: {input}"),
            Section("conclusion", "Write a conclusion for: {input}")
        ]
        result = await workflow.run(sections, "artificial intelligence")
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        max_concurrent: int = 5
    ):
        """
        Initialize the sectioning workflow.
        
        Args:
            model: The Claude model to use
            max_tokens: Maximum tokens per section response
            max_concurrent: Maximum parallel requests
        """
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _process_section(
        self,
        section: Section,
        input_data: str
    ) -> SectionResult:
        """
        Process a single section with the input data.
        
        Args:
            section: The section definition
            input_data: The input to process
            
        Returns:
            SectionResult with the generated content
        """
        import time
        start = time.time()
        
        try:
            # Format the prompt with the input
            prompt = section.prompt_template.format(input=input_data)
            
            # Build API call arguments
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            if section.system_prompt:
                kwargs["system"] = section.system_prompt
            
            # Make the API call
            response = await self.async_client.messages.create(**kwargs)
            
            return SectionResult(
                name=section.name,
                content=response.content[0].text,
                success=True,
                execution_time=time.time() - start
            )
            
        except Exception as e:
            return SectionResult(
                name=section.name,
                content="",
                success=False,
                error=str(e),
                execution_time=time.time() - start
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
            Dictionary containing:
            - sections: Dict mapping section name to SectionResult
            - successful: List of successful section names
            - failed: List of failed section names
            - execution_time: Total wall-clock time
            - merged: Combined output (if merge_results=True)
        """
        import time
        start = time.time()
        
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_process(section: Section) -> SectionResult:
            """Process a section with concurrency limiting."""
            async with semaphore:
                return await self._process_section(section, input_data)
        
        # Process all sections in parallel
        results = await asyncio.gather(*[
            bounded_process(section) for section in sections
        ])
        
        # Build output dictionary
        output = {
            "sections": {r.name: r for r in results},
            "successful": [r.name for r in results if r.success],
            "failed": [r.name for r in results if not r.success],
            "execution_time": time.time() - start
        }
        
        # Optionally merge results in section order
        if merge_results:
            # Maintain original section order
            section_order = [s.name for s in sections]
            successful_results = [
                output["sections"][name]
                for name in section_order
                if output["sections"][name].success
            ]
            merged = "\n\n".join([
                f"## {r.name}\n\n{r.content}" 
                for r in successful_results
            ])
            output["merged"] = merged
        
        return output


# =============================================================================
# Example: Document Analysis with Parallel Sections
# =============================================================================

async def document_analysis_example():
    """
    Analyze a document from multiple angles in parallel.
    
    This example shows how to extract different types of
    information from the same document simultaneously.
    """
    workflow = SectioningWorkflow()
    
    # Define the analysis sections
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
            name="Sentiment Analysis",
            prompt_template="Analyze the overall sentiment and tone of this text:\n\n{input}",
            system_prompt="You are a sentiment analyst. Describe tone, emotion, and attitude."
        ),
        Section(
            name="Questions Raised",
            prompt_template="What are 3 questions a reader might have after reading this?\n\n{input}",
            system_prompt="You anticipate reader questions and curiosities."
        ),
        Section(
            name="Action Items",
            prompt_template="What actions or next steps does this text suggest or imply?\n\n{input}",
            system_prompt="You identify actionable items and recommendations."
        )
    ]
    
    # Sample document to analyze
    document = """
    The company announced record quarterly earnings today, exceeding analyst 
    expectations by 15%. Revenue grew 23% year-over-year, driven primarily by 
    strong performance in the cloud services division. However, the CEO noted 
    challenges in the hardware segment, which saw a 5% decline. 
    
    Looking ahead, management provided cautious guidance for the next quarter, 
    citing macroeconomic uncertainties and supply chain constraints. Despite 
    these concerns, the company plans to increase R&D spending by 20% to 
    accelerate product development in AI and machine learning.
    
    The board also approved a new stock buyback program worth $5 billion and 
    increased the quarterly dividend by 10%. Analysts remain divided on the 
    outlook, with some praising the strategic investments while others express 
    concern about the slowing hardware business.
    """
    
    print("=" * 60)
    print("DOCUMENT ANALYSIS WITH PARALLEL SECTIONS")
    print("=" * 60)
    print("\nAnalyzing document from 5 different angles in parallel...\n")
    
    result = await workflow.run(sections, document)
    
    # Display timing information
    print(f"Completed in {result['execution_time']:.2f}s")
    print(f"Successful sections: {len(result['successful'])}/{len(sections)}")
    
    if result['failed']:
        print(f"Failed sections: {result['failed']}")
    
    # Show individual section timings
    print("\nSection timings:")
    for name, section_result in result['sections'].items():
        status = "✓" if section_result.success else "✗"
        print(f"  {status} {name}: {section_result.execution_time:.2f}s")
    
    # Display the merged output
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(result['merged'])


# =============================================================================
# Example: Content Generation with Parallel Sections
# =============================================================================

async def content_generation_example():
    """
    Generate a blog post with sections written in parallel.
    
    This example shows how to use sectioning for creative
    content generation where sections are independent.
    """
    workflow = SectioningWorkflow(max_tokens=512)
    
    topic = "The Future of Remote Work"
    
    sections = [
        Section(
            name="Hook",
            prompt_template="""Write an attention-grabbing opening paragraph (2-3 sentences) 
for a blog post about: {input}

Make it engaging and relevant to today's workforce.""",
            system_prompt="You write compelling blog introductions that hook readers."
        ),
        Section(
            name="Current State",
            prompt_template="""Write a paragraph describing the current state of: {input}

Include relevant context and recent developments.""",
            system_prompt="You provide clear, factual context for blog posts."
        ),
        Section(
            name="Benefits",
            prompt_template="""Write a section about the key benefits of: {input}

Focus on 3 main advantages with brief explanations.""",
            system_prompt="You highlight benefits in an engaging, balanced way."
        ),
        Section(
            name="Challenges",
            prompt_template="""Write a section about the challenges related to: {input}

Be honest about obstacles while maintaining a constructive tone.""",
            system_prompt="You discuss challenges thoughtfully and constructively."
        ),
        Section(
            name="Conclusion",
            prompt_template="""Write a concluding paragraph for a blog post about: {input}

End with a forward-looking perspective and call to action.""",
            system_prompt="You write memorable conclusions that inspire action."
        )
    ]
    
    print("\n" + "=" * 60)
    print("CONTENT GENERATION WITH PARALLEL SECTIONS")
    print("=" * 60)
    print(f"\nGenerating blog post about: {topic}")
    print("Writing 5 sections in parallel...\n")
    
    result = await workflow.run(sections, topic)
    
    print(f"Completed in {result['execution_time']:.2f}s")
    
    # Display the complete blog post
    print("\n" + "=" * 60)
    print(f"BLOG POST: {topic.upper()}")
    print("=" * 60 + "\n")
    
    # Custom merge that removes the section headers for cleaner output
    for name in ["Hook", "Current State", "Benefits", "Challenges", "Conclusion"]:
        if name in result['sections'] and result['sections'][name].success:
            print(result['sections'][name].content)
            print()


async def main():
    """Run all sectioning examples."""
    await document_analysis_example()
    await content_generation_example()


if __name__ == "__main__":
    asyncio.run(main())
