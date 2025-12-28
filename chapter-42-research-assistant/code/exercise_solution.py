"""
Exercise Solution: Research Assistant with Parallel Reading

Chapter 42: Project - Research Assistant Agent

This enhanced version reads multiple sources in parallel for faster research.
Key improvements:
- Batch web_read operations
- Use ThreadPoolExecutor for parallel requests
- Maintain the same workflow and state management
"""

import os
import json
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Import our tools
from web_search_tool import web_search, WEB_SEARCH_TOOL
from web_read_tool import web_read, WEB_READ_TOOL
from save_note_tool import ResearchNotes, SAVE_NOTE_TOOL


# New tool for parallel reading
READ_MULTIPLE_TOOL = {
    "name": "read_multiple",
    "description": (
        "Read multiple web pages in parallel for faster research. "
        "Provide a list of URLs to read simultaneously. Returns the "
        "content of all pages. Use this after searching to read "
        "2-3 promising sources at once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to read in parallel (2-5 URLs recommended)"
            }
        },
        "required": ["urls"]
    }
}


def read_multiple(urls: List[str], max_workers: int = 3) -> Dict[str, Optional[str]]:
    """
    Read multiple web pages in parallel.
    
    Args:
        urls: List of URLs to read
        max_workers: Maximum number of concurrent requests
        
    Returns:
        Dictionary mapping URL to content (or None if failed)
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all read tasks
        future_to_url = {
            executor.submit(web_read, url): url 
            for url in urls
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                results[url] = content
            except Exception as e:
                print(f"Error reading {url}: {e}")
                results[url] = None
    
    return results


class ParallelResearchAssistant:
    """
    Enhanced research assistant that reads multiple sources in parallel.
    
    This version is faster for research that requires reading many sources.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the parallel research assistant.
        
        Args:
            model: The Claude model to use
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.notes = ResearchNotes()
        
        # Configure tools - includes the new parallel reading tool
        self.tools = [
            WEB_SEARCH_TOOL,
            WEB_READ_TOOL,
            READ_MULTIPLE_TOOL,  # New!
            SAVE_NOTE_TOOL
        ]
        
        # Enhanced system prompt that explains parallel reading
        self.system_prompt = """You are an expert research assistant. Your job is to investigate topics thoroughly and produce comprehensive, well-sourced reports.

Research Process:
1. Search for relevant sources using web_search
2. Read sources using either:
   - web_read: For a single source
   - read_multiple: For 2-3 sources at once (FASTER - use this when possible!)
3. Save key findings using save_note as you go
4. When you have sufficient information, produce a final report

Guidelines:
- Be thorough: Search multiple times with different queries if needed
- Be efficient: Use read_multiple to read 2-3 sources at once when you identify multiple promising URLs
- Be selective: Focus on the most relevant sources
- Be organized: Save notes on key findings as you research
- Be conclusive: Your final report should synthesize findings and draw insights
- Cite sources: Reference URLs in your report
- Know when to stop: 5-10 quality sources is usually enough

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
        print(f"\n🔍 Starting parallel research on: {query}\n")
        
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
            
            # Check if Claude is done
            if response.stop_reason == "end_turn":
                report = self._extract_text(response.content)
                print("\n✅ Research complete!")
                return report
            
            # Process tool calls
            if response.stop_reason == "tool_use":
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
                print(f"⚠️  Unexpected stop reason: {response.stop_reason}")
                break
        
        # Max iterations reached
        print("\n⚠️  Max iterations reached. Generating partial report...")
        
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
        
        elif tool_name == "read_multiple":
            urls = tool_input["urls"]
            print(f"   Reading {len(urls)} URLs in parallel...")
            
            results = read_multiple(urls)
            
            # Format results
            formatted = []
            for url, content in results.items():
                if content:
                    formatted.append(f"=== {url} ===\n{content}\n")
                else:
                    formatted.append(f"=== {url} ===\nFailed to read\n")
            
            return "\n\n".join(formatted)
        
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
    """Demonstrate the parallel research assistant."""
    
    assistant = ParallelResearchAssistant()
    
    # Example research query
    query = "What are the best practices for writing Python async code?"
    
    print("This version uses parallel reading for faster research.")
    print("Watch for 'read_multiple' tool calls that fetch multiple sources at once.\n")
    
    try:
        report = assistant.research(query)
        
        print("\n" + "="*60)
        print("RESEARCH REPORT")
        print("="*60)
        print(report)
        print("\n" + "="*60)
        
        # Show research progress
        print("\n" + assistant.notes.get_summary())
        
        print("\nℹ️  Parallel reading typically reduces research time by 30-50%")
        
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user")
    except Exception as e:
        print(f"\n\nError during research: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
