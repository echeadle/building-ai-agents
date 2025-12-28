"""
Exercise Solution: Book Search Tool

This is the solution to the Chapter 8 practical exercise.

Task: Define a tool for looking up book information in a library database.

Requirements:
- Named 'search_books'
- Accepts a search query (required)
- Accepts optional filters: author, genre, year_from, year_to
- Clear description for when Claude should use it

Chapter 8: Defining Your First Tool
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the Anthropic client
client = anthropic.Anthropic()


# The book search tool definition
search_books_tool = {
    "name": "search_books",
    "description": """Searches for books in the library database.

Use this tool when the user wants to:
- Find books by title, author, or subject
- Discover books in a specific genre
- Look for books published within a certain time period
- Get book recommendations based on search criteria

The search query is required and will match against titles, authors, 
and descriptions. Optional filters can narrow down results.

If no filters are provided, the search returns all matching books 
sorted by relevance.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search term to look for in book titles, authors, and descriptions"
            },
            "author": {
                "type": "string",
                "description": "Filter results to books by this author (optional)"
            },
            "genre": {
                "type": "string",
                "description": "Filter results to a specific genre (optional)",
                "enum": [
                    "fiction",
                    "non-fiction",
                    "mystery",
                    "science-fiction",
                    "fantasy",
                    "romance",
                    "thriller",
                    "biography",
                    "history",
                    "science",
                    "self-help",
                    "children"
                ]
            },
            "year_from": {
                "type": "integer",
                "description": "Filter to books published in or after this year (optional)"
            },
            "year_to": {
                "type": "integer",
                "description": "Filter to books published in or before this year (optional)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (optional, defaults to 10)",
                "minimum": 1,
                "maximum": 50
            }
        },
        "required": ["query"]
    }
}


def main():
    """Demonstrate the book search tool with various queries."""
    
    print("=" * 60)
    print("Exercise Solution: Book Search Tool")
    print("=" * 60)
    print()
    
    # Display the tool definition
    print("Tool Definition:")
    print(json.dumps(search_books_tool, indent=2))
    print()
    
    # Test queries that demonstrate different use cases
    test_queries = [
        "Find science fiction books about artificial intelligence",
        "What mystery novels did Agatha Christie write?",
        "Show me popular fantasy books from the 2010s",
        "I'm looking for self-help books about productivity",
        "Find books about World War 2 history published after 2000",
    ]
    
    for query in test_queries:
        print(f"User: {query}")
        print("-" * 40)
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[search_books_tool],
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        
        for block in response.content:
            if block.type == "tool_use":
                print(f"Tool: {block.name}")
                print(f"Arguments: {json.dumps(block.input, indent=2)}")
            elif block.type == "text" and block.text.strip():
                print(f"Text: {block.text}")
        
        print()
    
    print("=" * 60)
    print("Key observations:")
    print("- Claude extracts the search query from natural language")
    print("- Genre is correctly identified and matched to enum values")
    print("- Author names are extracted when mentioned")
    print("- Year ranges are inferred from phrases like 'from the 2010s'")
    print("=" * 60)


if __name__ == "__main__":
    main()
