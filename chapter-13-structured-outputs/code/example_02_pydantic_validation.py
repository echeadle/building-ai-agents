"""
Validating LLM responses with Pydantic.

This example demonstrates how to use Pydantic to define schemas
and validate that Claude's responses match your expectations.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


# Define schemas using Pydantic
class BookInfo(BaseModel):
    """Schema for extracted book information."""
    
    title: str = Field(description="The book's title")
    author: str = Field(description="The author's full name")
    year: Optional[int] = Field(
        default=None, 
        description="Publication year",
        ge=1000,  # Reasonable minimum year
        le=2100   # Reasonable maximum year
    )
    genres: list[str] = Field(
        default_factory=list, 
        description="List of genres"
    )


class MovieInfo(BaseModel):
    """Schema for movie information."""
    
    title: str = Field(min_length=1, description="The movie's title")
    director: str = Field(min_length=1, description="The director's name")
    year: int = Field(
        ge=1888,  # First movie was made in 1888
        le=2100,
        description="Release year"
    )
    genres: list[str] = Field(
        default_factory=list,
        description="List of genres"
    )
    rating: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Rating out of 10"
    )
    runtime_minutes: Optional[int] = Field(
        default=None,
        ge=1,
        description="Runtime in minutes"
    )


def schema_to_prompt_instructions(model_class: type[BaseModel]) -> str:
    """
    Convert a Pydantic model to prompt instructions for Claude.
    
    Args:
        model_class: A Pydantic BaseModel class
        
    Returns:
        A string with JSON format instructions
    """
    schema = model_class.model_json_schema()
    
    lines = ["Required JSON format:", "{"]
    
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    items = list(properties.items())
    for i, (name, prop) in enumerate(items):
        prop_type = prop.get("type", "any")
        
        # Handle arrays
        if prop_type == "array":
            items_type = prop.get("items", {}).get("type", "any")
            prop_type = f"array of {items_type}s"
        
        # Handle nullable types
        any_of = prop.get("anyOf", [])
        if any_of:
            types = [t.get("type") for t in any_of if t.get("type") != "null"]
            if types:
                prop_type = types[0]
                prop_type += " or null"
        
        description = prop.get("description", "")
        is_required = name in required
        req_str = "required" if is_required else "optional"
        
        comma = "," if i < len(items) - 1 else ""
        lines.append(f'    "{name}": {prop_type}{comma}  // {req_str} - {description}')
    
    lines.append("}")
    
    return "\n".join(lines)


def extract_and_validate_book_info(description: str) -> BookInfo:
    """
    Extract book information and validate it against our schema.
    
    Args:
        description: A text description of a book
        
    Returns:
        A validated BookInfo object
        
    Raises:
        ValidationError: If the response doesn't match the schema
        json.JSONDecodeError: If the response isn't valid JSON
    """
    # Generate prompt instructions from schema
    format_instructions = schema_to_prompt_instructions(BookInfo)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a data extraction assistant. Respond only with valid JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""Extract book information from this description.

{format_instructions}

Description: {description}"""
            }
        ]
    )
    
    response_text = message.content[0].text
    
    # Parse JSON
    data = json.loads(response_text)
    
    # Validate against schema - returns BookInfo or raises ValidationError
    return BookInfo.model_validate(data)


def extract_and_validate_movie_info(description: str) -> MovieInfo:
    """
    Extract movie information and validate it against our schema.
    
    Args:
        description: A text description of a movie
        
    Returns:
        A validated MovieInfo object
    """
    format_instructions = schema_to_prompt_instructions(MovieInfo)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a data extraction assistant. Respond only with valid JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""Extract movie information from this description.

{format_instructions}

Description: {description}"""
            }
        ]
    )
    
    response_text = message.content[0].text
    data = json.loads(response_text)
    return MovieInfo.model_validate(data)


def demonstrate_validation_errors():
    """
    Demonstrate what happens when validation fails.
    """
    print("\n" + "=" * 60)
    print("Demonstrating Validation Errors")
    print("=" * 60)
    
    # Missing required field
    print("\n1. Missing required field:")
    try:
        BookInfo.model_validate({"title": "1984"})  # Missing author
    except ValidationError as e:
        print(f"   Error: {e.errors()[0]['msg']}")
        print(f"   Field: {e.errors()[0]['loc']}")
    
    # Wrong type
    print("\n2. Wrong type (string instead of int):")
    try:
        BookInfo.model_validate({
            "title": "1984",
            "author": "George Orwell",
            "year": "nineteen eighty four"  # Should be int
        })
    except ValidationError as e:
        print(f"   Error: {e.errors()[0]['msg']}")
        print(f"   Field: {e.errors()[0]['loc']}")
    
    # Value out of range
    print("\n3. Value out of range:")
    try:
        MovieInfo.model_validate({
            "title": "Future Movie",
            "director": "Someone",
            "year": 2500,  # Too far in future
            "rating": 15   # Max is 10
        })
    except ValidationError as e:
        for err in e.errors():
            print(f"   Error: {err['msg']} at {err['loc']}")


if __name__ == "__main__":
    # Example 1: Book extraction with validation
    print("=" * 60)
    print("Example 1: Book Extraction with Validation")
    print("=" * 60)
    
    book_description = """
    "1984" by George Orwell, published in 1949, is a dystopian novel 
    that explores themes of totalitarianism and surveillance. It has
    become one of the most influential works of the 20th century.
    """
    
    print("\nGenerated prompt instructions for BookInfo:")
    print(schema_to_prompt_instructions(BookInfo))
    
    try:
        book = extract_and_validate_book_info(book_description)
        print(f"\nExtracted and validated:")
        print(f"  Title: {book.title}")
        print(f"  Author: {book.author}")
        print(f"  Year: {book.year}")
        print(f"  Genres: {book.genres}")
    except ValidationError as e:
        print(f"\nValidation failed: {e}")
    except json.JSONDecodeError as e:
        print(f"\nInvalid JSON: {e}")
    
    # Example 2: Movie extraction with validation
    print("\n" + "=" * 60)
    print("Example 2: Movie Extraction with Validation")
    print("=" * 60)
    
    movie_description = """
    Inception (2010), directed by Christopher Nolan, is a mind-bending
    science fiction thriller about dream infiltration. It runs for 
    148 minutes and has an 8.8 rating on IMDB.
    """
    
    print("\nGenerated prompt instructions for MovieInfo:")
    print(schema_to_prompt_instructions(MovieInfo))
    
    try:
        movie = extract_and_validate_movie_info(movie_description)
        print(f"\nExtracted and validated:")
        print(f"  Title: {movie.title}")
        print(f"  Director: {movie.director}")
        print(f"  Year: {movie.year}")
        print(f"  Genres: {movie.genres}")
        print(f"  Rating: {movie.rating}")
        print(f"  Runtime: {movie.runtime_minutes} minutes")
    except ValidationError as e:
        print(f"\nValidation failed: {e}")
    except json.JSONDecodeError as e:
        print(f"\nInvalid JSON: {e}")
    
    # Example 3: Demonstrate validation errors
    demonstrate_validation_errors()
