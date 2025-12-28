"""
Complete Research Assistant Agent.

Chapter 42: Project - Research Assistant Agent

This agent can:
- Search the web for information
- Read web pages in detail
- Take notes on findings
- Synthesize research into comprehensive reports
"""

import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import anthropic

load_dotenv()

# Verify API keys
api_key = os.getenv("ANTHROPIC_API_KEY")
brave_key = os.getenv("BRAVE_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
if not brave_key:
    print("⚠️  BRAVE_API_KEY not found. Get one at: https://brave.com/search/api/")
    print("   The free tier includes 2,000 queries/month\n")

# Import our tools
from web_search_tool import web_search, WEB_SEARCH_TOOL
from web_read_tool import web_read, WEB_READ_TOOL
from save_note_tool import ResearchNotes, SAVE_NOTE_TOOL


class ResearchAssistant:
    """
    An autonomous research assistant that can investigate topics,
    read sources, and produce comprehensive reports.
    
    Usage:
        assistant = ResearchAssistant()
        report = assistant.research("What are best practices for API design?")
        print(report)
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the research assistant.
        
        Args:
            model: The Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.notes = ResearchNotes()
        
        # Configure tools
        self.tools = [
            WEB_SEARCH_TOOL,
            WEB_READ_TOOL,
            SAVE_NOTE_TOOL
        ]
        
        # System prompt defines the agent's behavior
        self.system_prompt = """You are an expert research assistant. Your job is to investigate topics thoroughly and produce comprehensive, well-sourced reports.

Research Process:
1. Search for relevant sources using web_search
2. Read the most promising sources using web_read
3. Save key findings using save_note as you go
4. When you have sufficient information, produce a final report

Guidelines:
- Be thorough: Search multiple times with different queries if needed
- Be selective: Read only the most relevant sources (not every search result)
- Be organized: Save notes on key findings as you research
- Be conclusive: Your final report should synthesize findings and draw insights
- Cite sources: Reference URLs in your report
- Know when to stop: Don't over-research; 5-10 quality sources is usually enough

Report Format:
When you have enough information, produce a report with:
- Executive Summary: Brief overview of findings
- Main Findings: Key insights organized by theme/category
- Detailed Analysis: In-depth discussion of important points
- Sources: List of all URLs consulted
- Conclusion: Summary and recommendations if applicable

Begin researching now."""
    
    def research(self, query: str, max_iterations: int = 20) -> str:
        """
        Research a topic and produce a comprehensive report.
        
        Args:
            query: The research question or topic
            max_iterations: Maximum agentic loop iterations
            
        Returns:
            The final research report
        """
        print(f"\n🔍 Starting research on: {query}\n")
        
        # Clear notes from any previous research
        self.notes.clear()
        
        # Initialize conversation
        messages = [
            {
                "role": "user",
                "content": f"Research this topic thoroughly and produce a comprehensive report: {query}"
            }
        ]
        
        # Agentic loop
        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration + 1} ---")
            
            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages
            )
            
            # Check if Claude is done (no more tool calls)
            if response.stop_reason == "end_turn":
                # Extract the final report
                report = self._extract_text(response.content)
                print("\n✅ Research complete!")
                return report
            
            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add Claude's response to messages
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Execute tools and collect results
                tool_results = []
                
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        
                        print(f"🔧 Using tool: {tool_name}")
                        
                        # Execute the tool
                        result = self._execute_tool(tool_name, tool_input)
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": str(result)
                        })
                
                # Add tool results to messages
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            
            else:
                # Unexpected stop reason
                print(f"⚠️  Unexpected stop reason: {response.stop_reason}")
                break
        
        # Max iterations reached
        print("\n⚠️  Max iterations reached. Generating partial report...")
        
        # Ask Claude to produce a report with what it has
        messages.append({
            "role": "user",
            "content": "You've reached the iteration limit. Please produce your research report based on the information you've gathered so far."
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.system_prompt,
            messages=messages
        )
        
        return self._extract_text(response.content)
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Execute a tool and return its result."""
        
        if tool_name == "web_search":
            query = tool_input["query"]
            max_results = tool_input.get("max_results", 5)
            results = web_search(query, max_results)
            
            # Format results nicely
            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"{i}. {result['title']}\n"
                    f"   URL: {result['url']}\n"
                    f"   {result['snippet']}"
                )
            
            return "\n\n".join(formatted) if formatted else "No results found"
        
        elif tool_name == "web_read":
            url = tool_input["url"]
            content = web_read(url)
            return content if content else "Failed to read page"
        
        elif tool_name == "save_note":
            finding = tool_input["finding"]
            source = tool_input.get("source", "")
            return self.notes.save_note(finding, source)
        
        else:
            return f"Unknown tool: {tool_name}"
    
    def _extract_text(self, content_blocks: List[Any]) -> str:
        """Extract text from Claude's response content blocks."""
        text_parts = []
        for block in content_blocks:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return "\n".join(text_parts)


def main():
    """Demonstrate the research assistant."""
    
    assistant = ResearchAssistant()
    
    # Example research queries
    queries = [
        "What are the key differences between REST and GraphQL APIs?",
        "Compare the top 3 Python web frameworks in 2024",
        "What are best practices for API rate limiting?"
    ]
    
    # Pick one to research
    print("Select a research topic:")
    for i, query in enumerate(queries, 1):
        print(f"{i}. {query}")
    
    choice = input("\nEnter number (1-3, or type your own query): ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(queries):
        query = queries[int(choice) - 1]
    elif choice:
        query = choice
    else:
        query = queries[0]  # Default
    
    try:
        report = assistant.research(query)
        
        print("\n" + "="*60)
        print("RESEARCH REPORT")
        print("="*60)
        print(report)
        print("\n" + "="*60)
        
        # Show research progress
        print("\n" + assistant.notes.get_summary())
        
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user")
    except Exception as e:
        print(f"\n\nError during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
