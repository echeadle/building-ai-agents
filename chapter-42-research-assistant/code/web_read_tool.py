"""
Web page reading tool for the research assistant.

Chapter 42: Project - Research Assistant Agent
"""

import os
import requests
from typing import Optional
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


def web_read(url: str, max_chars: int = 10000) -> Optional[str]:
    """
    Read the content of a web page and extract the main text.
    
    Args:
        url: The URL to read
        max_chars: Maximum characters to return (to avoid huge pages)
        
    Returns:
        The extracted text content, or None if the page couldn't be read
        
    Example:
        >>> content = web_read("https://example.com/article")
        >>> print(content[:200])
    """
    try:
        # Fetch the page
        headers = {
            "User-Agent": "ResearchAssistant/1.0 (Educational Project)"
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
        
        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Content truncated]"
        
        return text
        
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None
    except Exception as e:
        print(f"Error parsing {url}: {e}")
        return None


# Tool definition for the agent
WEB_READ_TOOL = {
    "name": "web_read",
    "description": (
        "Read the full content of a web page. Use this after searching "
        "to read promising sources in detail. Returns the main text "
        "content of the page. Only read URLs from search results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to read. Must be a valid HTTP/HTTPS URL."
            }
        },
        "required": ["url"]
    }
}


if __name__ == "__main__":
    # Test the reader
    print("Testing web_read tool...\n")
    
    test_url = "https://docs.python.org/3/library/asyncio.html"
    content = web_read(test_url, max_chars=500)
    
    if content:
        print(f"Successfully read {len(content)} characters from:")
        print(f"{test_url}\n")
        print("First 500 characters:")
        print(content[:500])
    else:
        print("Failed to read the page")
