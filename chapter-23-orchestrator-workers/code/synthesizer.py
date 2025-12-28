"""
Synthesis of worker outputs into final response.

Chapter 23: Orchestrator-Workers - Implementation

This module synthesizes multiple worker results into a coherent,
comprehensive final response.
"""

import os
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
# Synthesis System Prompt
# =============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are an expert at synthesizing information from multiple sources into coherent, comprehensive responses.

## Your Task

You will receive:
1. An original query from a user
2. Multiple research/analysis results from specialized workers
3. Optional guidance on how to combine the results

Your job is to synthesize these inputs into a single, well-organized response that:
- Directly addresses the original query
- Integrates insights from all worker results
- Maintains a logical flow and clear structure
- Highlights key findings and conclusions
- Notes any conflicting information or important caveats
- Provides actionable insights where appropriate

## Output Guidelines

1. **Structure**: Use clear headers to organize major sections
2. **Priority**: Lead with the most important findings
3. **Integration**: Weave together insights from different sources naturally
4. **Evidence**: Support key claims with details from the research
5. **Balance**: Present multiple perspectives fairly
6. **Conclusion**: End with clear takeaways or recommendations

## Writing Style

- Clear and professional
- Comprehensive but focused
- Accessible to a general audience
- Objective and balanced
- Action-oriented where appropriate"""


# =============================================================================
# Data Classes
# =============================================================================

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
class SynthesisResult:
    """Result from the synthesis process."""
    content: str
    sources_used: int
    success: bool
    error: Optional[str] = None


# =============================================================================
# Synthesizer Class
# =============================================================================

class Synthesizer:
    """
    Synthesizes multiple worker results into a coherent final response.
    
    The Synthesizer takes outputs from multiple workers and combines them
    into a single, well-organized response that directly addresses the
    original query.
    
    Example:
        synthesizer = Synthesizer()
        result = synthesizer.synthesize(
            original_query="What are the impacts of AI?",
            results=[worker_result_1, worker_result_2, worker_result_3],
            synthesis_guidance="Focus on practical implications"
        )
        print(result.content)
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the synthesizer.
        
        Args:
            model: Claude model to use for synthesis
        """
        self.client = anthropic.Anthropic()
        self.model = model
    
    def _build_synthesis_request(
        self,
        original_query: str,
        results: list[WorkerResult],
        synthesis_guidance: str = ""
    ) -> str:
        """
        Build the synthesis request message.
        
        Args:
            original_query: The user's original question
            results: List of WorkerResults to synthesize
            synthesis_guidance: Optional guidance on how to combine
            
        Returns:
            Formatted synthesis request
        """
        request = f"""## Original Query

{original_query}

## Research Results from Specialized Workers

"""
        
        for i, result in enumerate(results, 1):
            request += f"""### {i}. {result.subtask_title}

{result.content}

---

"""
        
        if synthesis_guidance:
            request += f"""## Synthesis Guidance

{synthesis_guidance}

"""
        
        request += """Please synthesize these results into a comprehensive, well-organized response that directly addresses the original query. Integrate insights from all sources and provide clear conclusions."""
        
        return request
    
    def synthesize(
        self,
        original_query: str,
        results: list[WorkerResult],
        synthesis_guidance: str = ""
    ) -> SynthesisResult:
        """
        Synthesize worker results into a final response.
        
        Args:
            original_query: The user's original question
            results: List of WorkerResults to synthesize
            synthesis_guidance: Optional guidance on how to combine
            
        Returns:
            SynthesisResult containing the synthesized content
        """
        # Filter to successful results only
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return SynthesisResult(
                content="Unable to generate a response. All worker subtasks failed.",
                sources_used=0,
                success=False,
                error="No successful worker results to synthesize"
            )
        
        # Build the synthesis request
        synthesis_request = self._build_synthesis_request(
            original_query=original_query,
            results=successful_results,
            synthesis_guidance=synthesis_guidance
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYNTHESIS_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": synthesis_request}
                ]
            )
            
            return SynthesisResult(
                content=response.content[0].text,
                sources_used=len(successful_results),
                success=True
            )
            
        except anthropic.APIError as e:
            return SynthesisResult(
                content="",
                sources_used=0,
                success=False,
                error=str(e)
            )
    
    def synthesize_with_custom_prompt(
        self,
        original_query: str,
        results: list[WorkerResult],
        custom_system_prompt: str
    ) -> SynthesisResult:
        """
        Synthesize with a custom system prompt.
        
        Use this for specialized synthesis needs like:
        - Executive summaries
        - Technical reports
        - Action item lists
        - Specific formatting requirements
        
        Args:
            original_query: The user's original question
            results: List of WorkerResults to synthesize
            custom_system_prompt: Custom system prompt for synthesis
            
        Returns:
            SynthesisResult containing the synthesized content
        """
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return SynthesisResult(
                content="Unable to generate a response. No successful results.",
                sources_used=0,
                success=False,
                error="No successful worker results"
            )
        
        synthesis_request = self._build_synthesis_request(
            original_query=original_query,
            results=successful_results
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=custom_system_prompt,
                messages=[
                    {"role": "user", "content": synthesis_request}
                ]
            )
            
            return SynthesisResult(
                content=response.content[0].text,
                sources_used=len(successful_results),
                success=True
            )
            
        except anthropic.APIError as e:
            return SynthesisResult(
                content="",
                sources_used=0,
                success=False,
                error=str(e)
            )


# =============================================================================
# Preset Synthesis Prompts
# =============================================================================

EXECUTIVE_SUMMARY_PROMPT = """You are an expert at creating executive summaries. 
Synthesize the provided research into a concise executive summary with:
- Key findings (3-5 bullet points)
- Critical implications
- Recommended actions
Keep it brief and actionable. Maximum 500 words."""

TECHNICAL_REPORT_PROMPT = """You are a technical writer creating a detailed report.
Synthesize the provided research into a technical report with:
- Abstract
- Methodology overview
- Detailed findings
- Technical analysis
- Conclusions and recommendations
Be thorough and precise."""

ACTION_ITEMS_PROMPT = """You are a project manager extracting actionable items.
From the provided research, create a prioritized list of:
- Immediate actions (do now)
- Short-term actions (within 30 days)
- Long-term actions (within 90 days)
Each item should be specific and assignable."""


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create test worker results
    mock_results = [
        WorkerResult(
            subtask_id="task_1",
            subtask_title="Benefits of Electric Vehicles",
            content="""## Key Findings on EV Benefits

### 1. Environmental Impact
- **Zero Direct Emissions**: EVs produce no tailpipe emissions, significantly reducing urban air pollution.
- **Lower Lifecycle Emissions**: Even accounting for electricity generation, EVs typically produce 50-70% fewer emissions over their lifetime.

### 2. Operating Cost Advantages
- **Fuel Savings**: Electricity costs approximately $0.03-0.05 per mile vs $0.10-0.15 for gasoline.
- **Reduced Maintenance**: Fewer moving parts mean less maintenance (no oil changes, fewer brake replacements due to regenerative braking).

### 3. Energy Efficiency
- EVs convert 85-90% of electrical energy to motion, compared to 20-30% for internal combustion engines.

### Caveats
- Environmental benefits depend heavily on the local electricity grid mix.
- Manufacturing batteries does have significant environmental impact.""",
            success=True,
            execution_time=3.2
        ),
        WorkerResult(
            subtask_id="task_2",
            subtask_title="Challenges of Electric Vehicle Adoption",
            content="""## Key Challenges for EV Adoption

### 1. Range Limitations
- Average EV range is 200-300 miles per charge
- "Range anxiety" remains a significant psychological barrier
- Long-distance travel requires careful planning

### 2. Charging Infrastructure
- Insufficient public charging stations in many areas
- Home charging requires dedicated equipment ($500-2000)
- Fast charging can take 30-60 minutes

### 3. Cost Barriers
- Higher upfront purchase price ($10,000-20,000 premium over comparable ICE vehicles)
- Battery replacement costs ($5,000-15,000)
- Charging infrastructure investment

### 4. Environmental Concerns
- Battery production requires lithium, cobalt, nickel mining
- Battery recycling infrastructure is still developing
- Electricity source affects overall environmental impact

### Important Context
- Many challenges are rapidly improving with technology advancement
- Government incentives help offset cost barriers in many regions""",
            success=True,
            execution_time=2.8
        ),
        WorkerResult(
            subtask_id="task_3",
            subtask_title="Economic Impact Analysis",
            content="""## Economic Analysis of EV Adoption

### Consumer Economics
| Factor | EVs | ICE Vehicles |
|--------|-----|--------------|
| Purchase Price | Higher | Lower |
| Fuel Cost | Lower | Higher |
| Maintenance | Lower | Higher |
| 5-Year TCO | Often lower | Often higher |

### Macro-Economic Impacts

#### Job Market Transformation
- **Losses**: Traditional auto manufacturing, gas stations, oil industry
- **Gains**: EV manufacturing, battery production, charging infrastructure, software

#### Energy Sector Shifts
- Decreased oil demand
- Increased electricity demand
- Grid modernization investments needed

#### Urban Planning Implications
- Redesign of parking infrastructure
- Changes to city tax revenue (gas taxes)
- New zoning for charging stations

### Key Insight
The transition creates winners and losers economically, requiring policy interventions to manage workforce transitions and infrastructure investments.""",
            success=True,
            execution_time=3.5
        )
    ]
    
    # Create synthesizer
    synthesizer = Synthesizer()
    
    # Run standard synthesis
    print("STANDARD SYNTHESIS")
    print("=" * 60)
    
    result = synthesizer.synthesize(
        original_query="What are the impacts of electric vehicle adoption?",
        results=mock_results,
        synthesis_guidance="Balance environmental benefits against practical challenges, and address economic factors for both consumers and society."
    )
    
    if result.success:
        print(f"Sources used: {result.sources_used}")
        print("-" * 60)
        print(result.content)
    else:
        print(f"Synthesis failed: {result.error}")
    
    # Run executive summary synthesis
    print("\n\n" + "=" * 60)
    print("EXECUTIVE SUMMARY SYNTHESIS")
    print("=" * 60)
    
    exec_result = synthesizer.synthesize_with_custom_prompt(
        original_query="What are the impacts of electric vehicle adoption?",
        results=mock_results,
        custom_system_prompt=EXECUTIVE_SUMMARY_PROMPT
    )
    
    if exec_result.success:
        print(exec_result.content)
    else:
        print(f"Synthesis failed: {exec_result.error}")
