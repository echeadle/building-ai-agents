"""
Research Assistant Tools

Chapter 42: Project - Research Assistant Agent

Each tool provides one capability:
- search_web: Find relevant sources using SerpAPI
- read_page: Extract content from URLs
- take_note: Record structured findings
- create_report: Synthesize final report
"""

import os
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

# Global state for the current research session
research_notes: List[Dict[str, str]] = []
final_report: str = ""


def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using SerpAPI
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
        
    Returns:
        Formatted string with search results
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not found. Please add it to your .env file. Get a free key at https://serpapi.com/"
    
    # Validate num_results
    num_results = max(1, min(10, num_results))
    
    try:
        # SerpAPI endpoint
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "engine": "google"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "error" in data:
            return f"Search API error: {data['error']}"
        
        # Extract organic results
        results = data.get("organic_results", [])
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format results for the agent
        formatted_results = [f"Search results for: '{query}'\n"]
        for idx, result in enumerate(results[:num_results], 1):
            title = result.get("title", "No title")
            url = result.get("link", "")
            snippet = result.get("snippet", "No description")
            
            formatted_results.append(
                f"{idx}. {title}\n"
                f"   URL: {url}\n"
                f"   {snippet}\n"
            )
        
        return "\n".join(formatted_results)
        
    except requests.exceptions.Timeout:
        return f"Error: Search request timed out for query: {query}"
    except requests.exceptions.RequestException as e:
        return f"Error searching web: {str(e)}"
    except Exception as e:
        return f"Unexpected error during search: {str(e)}"


def read_page(url: str) -> str:
    """
    Read and extract text content from a web page
    
    Args:
        url: The URL to read
        
    Returns:
        Extracted text content
    """
    try:
        # Add user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit length to avoid overwhelming the context
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[Content truncated to {max_chars} characters for context limits]"
        
        return f"Content from {url}:\n\n{text}"
        
    except requests.exceptions.Timeout:
        return f"Error: Request timed out when trying to read {url}"
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} when reading {url}"
    except requests.exceptions.RequestException as e:
        return f"Error reading page {url}: {str(e)}"
    except Exception as e:
        return f"Unexpected error reading page {url}: {str(e)}"


def take_note(
    finding: str,
    source_url: str,
    source_title: str,
    credibility: str
) -> str:
    """
    Record a research finding
    
    Args:
        finding: The key information discovered
        source_url: URL of the source
        source_title: Title of the source
        credibility: Source credibility assessment (high/medium/low)
        
    Returns:
        Confirmation message
    """
    global research_notes
    
    # Validate credibility
    if credibility not in ["high", "medium", "low"]:
        return f"Error: credibility must be 'high', 'medium', or 'low', got '{credibility}'"
    
    note = {
        "finding": finding,
        "source_url": source_url,
        "source_title": source_title,
        "credibility": credibility
    }
    
    research_notes.append(note)
    
    # Provide feedback to agent
    return (
        f"✓ Note recorded (total notes: {len(research_notes)}).\n"
        f"Finding: {finding[:100]}{'...' if len(finding) > 100 else ''}\n"
        f"Source: {source_title}\n"
        f"Credibility: {credibility}\n"
        f"Continue researching or create your report when ready."
    )


def create_report(report_content: str) -> str:
    """
    Create the final research report
    
    Args:
        report_content: The complete report in markdown format
        
    Returns:
        Confirmation message
    """
    global final_report
    
    if not report_content or len(report_content.strip()) < 100:
        return "Error: Report content is too short. Please provide a comprehensive report."
    
    final_report = report_content
    
    return (
        f"✓ Research report created successfully!\n"
        f"Report length: {len(report_content)} characters\n"
        f"Based on {len(research_notes)} research notes.\n"
        f"The report is complete and ready to be saved."
    )


def get_research_notes() -> List[Dict[str, str]]:
    """Get all research notes from the current session"""
    return research_notes.copy()


def get_final_report() -> str:
    """Get the final research report"""
    return final_report


def reset_research_session():
    """Clear all notes and report for a new research task"""
    global research_notes, final_report
    research_notes = []
    final_report = ""


# Example usage
if __name__ == "__main__":
    print("Testing research tools...")
    print("\n1. Testing search_web:")
    print("-" * 60)
    result = search_web("Python programming", num_results=3)
    print(result)
    
    print("\n2. Testing read_page:")
    print("-" * 60)
    result = read_page("https://www.python.org")
    print(result[:500] + "...")
    
    print("\n3. Testing take_note:")
    print("-" * 60)
    result = take_note(
        finding="Python is a high-level programming language",
        source_url="https://www.python.org",
        source_title="Python Official Website",
        credibility="high"
    )
    print(result)
    
    print("\n4. Testing get_research_notes:")
    print("-" * 60)
    notes = get_research_notes()
    print(f"Total notes: {len(notes)}")
    print(f"First note: {notes[0] if notes else 'None'}")
    
    print("\n5. Testing create_report:")
    print("-" * 60)
    report = """# Python Programming Language

## Overview
Python is a high-level, interpreted programming language known for its simplicity and readability.

## Key Features
- Easy to learn and use
- Large standard library
- Cross-platform compatibility

## Sources
- [Python Official Website](https://www.python.org)
"""
    result = create_report(report)
    print(result)
    
    print("\n6. Testing reset_research_session:")
    print("-" * 60)
    reset_research_session()
    print(f"Notes after reset: {len(get_research_notes())}")
    print(f"Report after reset: {len(get_final_report())} chars")
    
    print("\n✓ All tools tested successfully!")
