"""
Web search tool for the research assistant.

Chapter 42: Project - Research Assistant Agent
"""

import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
if not BRAVE_API_KEY:
    print("⚠️  BRAVE_API_KEY not found. Get one at: https://brave.com/search/api/")
    print("   The free tier includes 2,000 queries/month")


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using Brave Search API.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
        
    Example:
        >>> results = web_search("python async programming")
        >>> for result in results:
        ...     print(f"{result['title']}: {result['url']}")
    """
    if not BRAVE_API_KEY:
        return []
    
    url = "https://api.search.brave.com/res/v1/web/search"
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    
    params = {
        "q": query,
        "count": max_results
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information
        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", "")
            })
        
        return results
        
    except requests.RequestException as e:
        print(f"Search error: {e}")
        return []


# Tool definition for the agent
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for information. Use this to find relevant sources "
        "on a topic. Returns a list of search results with titles, URLs, "
        "and snippets. Use specific queries for better results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific for better results."
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}


if __name__ == "__main__":
    # Test the search
    print("Testing web_search tool...\n")
    
    if not BRAVE_API_KEY:
        print("Cannot test: BRAVE_API_KEY not set")
        print("\nTo use this tool:")
        print("1. Get a free API key from https://brave.com/search/api/")
        print("2. Add to your .env file: BRAVE_API_KEY=your-key-here")
    else:
        results = web_search("Python async best practices", max_results=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   {result['url']}")
                print(f"   {result['snippet'][:100]}...")
                print()
        else:
            print("No results returned")
