"""
Research Assistant Agent

Chapter 42: Project - Research Assistant Agent

A fully autonomous agent that:
1. Searches for information
2. Reads relevant sources
3. Takes structured notes
4. Synthesizes findings into reports
"""

import os
import anthropic
from typing import List, Dict, Any
from dotenv import load_dotenv
from tools import (
    search_web,
    read_page,
    take_note,
    create_report,
    reset_research_session,
    get_final_report
)

load_dotenv()


class ResearchAgent:
    """
    An autonomous research assistant agent
    
    Capabilities:
    - Web search via SerpAPI
    - Content reading from URLs
    - Structured note-taking
    - Report synthesis with citations
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 25
    ):
        """
        Initialize the research agent
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_turns: Maximum conversation turns (safety limit)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_turns = max_turns
        
        # Tool definitions
        self.tools = [
            {
                "name": "search_web",
                "description": (
                    "Search the web for information. Returns titles, URLs, and "
                    "snippets from top results. Use this to find sources to read. "
                    "Start with broad queries, then refine based on results."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query. Be specific and use keywords."
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (1-10)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "read_page",
                "description": (
                    "Read the full text content of a web page. Use this after "
                    "searching to get detailed information from promising sources. "
                    "Only read pages that seem highly relevant based on search results."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to read"
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "take_note",
                "description": (
                    "Record an important finding from your research. Include the "
                    "source URL and a clear summary. Take notes as you research - "
                    "these will be used to create the final report."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "finding": {
                            "type": "string",
                            "description": "The key information or insight discovered"
                        },
                        "source_url": {
                            "type": "string",
                            "description": "URL where this information was found"
                        },
                        "source_title": {
                            "type": "string",
                            "description": "Title of the source"
                        },
                        "credibility": {
                            "type": "string",
                            "description": "Assessment of source credibility",
                            "enum": ["high", "medium", "low"]
                        }
                    },
                    "required": ["finding", "source_url", "source_title", "credibility"]
                }
            },
            {
                "name": "create_report",
                "description": (
                    "Create the final research report based on all notes taken. "
                    "Call this when you have gathered sufficient information to "
                    "comprehensively answer the research question. The report should "
                    "be in markdown format with proper citations."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "report_content": {
                            "type": "string",
                            "description": (
                                "The complete research report in markdown format, "
                                "with proper citations in [Source](URL) format"
                            )
                        }
                    },
                    "required": ["report_content"]
                }
            }
        ]
        
        # Tool function mapping
        self.tool_functions = {
            "search_web": search_web,
            "read_page": read_page,
            "take_note": take_note,
            "create_report": create_report
        }
    
    def research(
        self,
        question: str,
        max_searches: int = 10,
        max_reads: int = 15,
        verbose: bool = True
    ) -> str:
        """
        Conduct research on a question
        
        Args:
            question: The research question to answer
            max_searches: Maximum number of search queries allowed
            max_reads: Maximum number of pages to read
            verbose: Whether to print progress
            
        Returns:
            The final research report
        """
        # Reset session state
        reset_research_session()
        
        # Initialize conversation
        messages = [
            {
                "role": "user",
                "content": self._create_research_prompt(question, max_searches, max_reads)
            }
        ]
        
        turn_count = 0
        searches_used = 0
        pages_read = 0
        
        if verbose:
            print(f"\n🔍 Starting research on: {question}\n")
        
        while turn_count < self.max_turns:
            turn_count += 1
            if verbose:
                print(f"Turn {turn_count}/{self.max_turns}")
            
            # Make API call
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    tools=self.tools,
                    messages=messages
                )
            except anthropic.APIConnectionError as e:
                if verbose:
                    print(f"❌ API connection error: {e}")
                return "Error: Could not connect to Anthropic API. Check your internet connection."
            except anthropic.RateLimitError:
                if verbose:
                    print("❌ Rate limit reached. Please wait and try again.")
                return "Error: API rate limit reached."
            except Exception as e:
                if verbose:
                    print(f"❌ Unexpected API error: {e}")
                return f"Error: {str(e)}"
            
            # Process response
            if response.stop_reason == "end_turn":
                # Agent decided it's done without creating a report
                if verbose:
                    print("\n⚠️ Agent stopped without creating a report.")
                    print("This might indicate the research question couldn't be answered.")
                break
            
            # Check for tool use
            tool_uses = [block for block in response.content if block.type == "tool_use"]
            
            if not tool_uses:
                # No tools used, agent is finished
                if verbose:
                    print("\n✅ Research complete!")
                break
            
            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Process each tool use
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input
                
                if verbose:
                    print(f"  🔧 Using tool: {tool_name}")
                
                # Apply constraints
                if tool_name == "search_web":
                    if searches_used >= max_searches:
                        result = (
                            f"Error: Maximum searches ({max_searches}) reached. "
                            "Use existing information or create report with what you have."
                        )
                    else:
                        searches_used += 1
                        result = self.tool_functions[tool_name](**tool_input)
                        if verbose:
                            print(f"     Searches used: {searches_used}/{max_searches}")
                
                elif tool_name == "read_page":
                    if pages_read >= max_reads:
                        result = (
                            f"Error: Maximum pages ({max_reads}) read. "
                            "Use existing notes or create report."
                        )
                    else:
                        pages_read += 1
                        result = self.tool_functions[tool_name](**tool_input)
                        if verbose:
                            print(f"     Pages read: {pages_read}/{max_reads}")
                
                else:
                    # Other tools have no limits
                    result = self.tool_functions[tool_name](**tool_input)
                    
                    if tool_name == "create_report" and verbose:
                        print("  📄 Final report created!")
                
                # Add result to conversation
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })
            
            # Add tool results as user message
            messages.append({"role": "user", "content": tool_results})
            
            # Check if report was created
            final_report = get_final_report()
            if final_report:
                if verbose:
                    print(f"\n✅ Research complete! Report generated ({len(final_report)} chars)")
                return final_report
        
        # If we hit max turns without a report
        if verbose:
            print(f"\n⚠️ Reached maximum turns ({self.max_turns}) without completing research.")
        
        final_report = get_final_report()
        if final_report:
            return final_report
        else:
            return "Research incomplete: Maximum turns reached without generating a report."
    
    def _create_research_prompt(
        self,
        question: str,
        max_searches: int,
        max_reads: int
    ) -> str:
        """Create the initial research prompt"""
        return f"""You are a research assistant. Your task is to thoroughly research this question and provide a comprehensive, well-cited answer:

**Research Question:** {question}

**Your Process:**
1. **Search**: Use search_web to find relevant sources. Start broad, then refine based on results.
2. **Read**: Use read_page to read the most promising sources. Prioritize credible, authoritative sites.
3. **Take Notes**: Use take_note to record key findings as you research. Include source citations.
4. **Synthesize**: When you have sufficient information, use create_report to create your final report.

**Requirements:**
- Your report must cite all sources in [Source Title](URL) format
- Evaluate source credibility (prefer .edu, .gov, established publications)
- Focus on recent information (last 2 years when possible)
- Be comprehensive but concise
- Structure your report with clear sections using markdown headers

**Constraints:**
- Maximum {max_searches} searches allowed
- Maximum {max_reads} pages to read
- Take notes as you research - they inform your final report
- If you hit resource limits, work with what you have

Begin your research now. Search for relevant information, read the best sources, take detailed notes, and synthesize a comprehensive report."""


# Example usage
if __name__ == "__main__":
    # Verify API keys
    api_key = os.getenv("ANTHROPIC_API_KEY")
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")
    
    if not serpapi_key:
        print("⚠️ Warning: SERPAPI_API_KEY not found.")
        print("Get a free key at https://serpapi.com/")
        print("Without it, the agent cannot search the web.\n")
    
    # Create agent
    agent = ResearchAgent(api_key=api_key)
    
    # Conduct research
    question = "What are the main applications of large language models in 2024?"
    
    print("="*80)
    print(f"Research Question: {question}")
    print("="*80)
    
    report = agent.research(question, max_searches=6, max_reads=10)
    
    # Save report
    output_file = "research_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"---\n\n")
        f.write(report)
    
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)
    print(report)
    print("\n" + "="*80)
    print(f"✓ Report saved to {output_file}")
    print("="*80)
