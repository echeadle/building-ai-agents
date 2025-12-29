---
chapter: 13
title: "Structured Outputs and Response Parsing"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 13: Structured Outputs and Response Parsing

## Introduction

In the previous chapters, you've built tools that Claude can use and learned how to handle tool calls in a loop. But there's a challenge we haven't addressed yet: what happens when you need Claude to respond in a specific format that your code can reliably parse?

Consider this scenario: you're building an agent that extracts information from documents. You ask Claude to identify the author, date, and main topics from a text. Claude responds with beautifully written prose—but your code needs structured data it can store in a database. How do you bridge that gap?

This is where **structured outputs** come in. By requesting responses in a specific format (typically JSON) and validating those responses, you transform Claude from a text generator into a component that fits neatly into your data pipeline.

In this chapter, you'll learn how to request structured responses from Claude, define schemas for the data you expect, validate that responses match your requirements, and handle the inevitable cases where something goes wrong.

## Learning Objectives

By the end of this chapter, you will be able to:

- Request JSON-formatted responses from Claude using system prompts and clear instructions
- Define response schemas that specify exactly what data you expect
- Validate LLM responses against your schemas using Pydantic
- Handle malformed responses gracefully without crashing your application
- Decide when to use structured outputs versus freeform text responses

## Why Structured Outputs Matter for Agents

When you're building agents, structured outputs serve three critical purposes:

**1. Programmatic Processing**

Freeform text is great for humans but terrible for code. If Claude responds with "The meeting is scheduled for next Tuesday at 3 PM," your code can't easily extract that into a datetime object. But if Claude responds with `{"date": "2025-01-21", "time": "15:00"}`, parsing is trivial.

**2. Reliability**

Structured outputs with validation catch errors early. If you expect a number but Claude returns "approximately five," validation fails immediately rather than causing mysterious bugs downstream.

**3. Composability**

When building complex agents with multiple steps, structured outputs allow each step to produce data that the next step can consume reliably. This is the foundation of the workflow patterns we'll explore in Part 3.

## Requesting JSON Output

The simplest way to get structured output from Claude is to ask for it. Let's start with a basic example.

### The Direct Approach

```python
"""
Requesting JSON output from Claude - basic approach.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()


def extract_book_info(description: str) -> dict:
    """
    Extract structured book information from a text description.
    
    Args:
        description: A text description of a book
        
    Returns:
        A dictionary containing extracted book information
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Extract information from this book description and return it as JSON.

The JSON should have these fields:
- title: the book's title
- author: the author's name
- year: publication year (as a number)
- genres: a list of genres

Book description:
{description}

Return ONLY the JSON object, no other text."""
            }
        ]
    )
    
    response_text = message.content[0].text
    return json.loads(response_text)


if __name__ == "__main__":
    description = """
    "The Hitchhiker's Guide to the Galaxy" is a comedic science fiction novel 
    written by Douglas Adams, first published in 1979. It's a blend of science 
    fiction and comedy that has become a cult classic.
    """
    
    result = extract_book_info(description)
    print(json.dumps(result, indent=2))
```

When you run this, Claude typically responds with clean JSON:

```json
{
  "title": "The Hitchhiker's Guide to the Galaxy",
  "author": "Douglas Adams",
  "year": 1979,
  "genres": ["science fiction", "comedy"]
}
```

### Using System Prompts for Consistent Formatting

For more reliable results, especially when you'll make many similar requests, use a system prompt:

```python
def extract_book_info_with_system_prompt(description: str) -> dict:
    """
    Extract book information using a system prompt for consistency.
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a data extraction assistant. You always respond with valid JSON and nothing else.
Never include explanations, markdown formatting, or additional text.
If you cannot extract a field, use null for that field.""",
        messages=[
            {
                "role": "user",
                "content": f"""Extract book information from this description.

Required JSON format:
{{
    "title": "string",
    "author": "string", 
    "year": number or null,
    "genres": ["string"]
}}

Description: {description}"""
            }
        ]
    )
    
    return json.loads(message.content[0].text)
```

The system prompt establishes Claude's role as a data extraction assistant that always outputs JSON. This consistency pays off when you're making many similar requests.

## Defining Response Schemas

Showing Claude an example of the JSON structure you want is helpful, but for production code, you need proper schemas. This is where **Pydantic** shines.

### Introduction to Pydantic

Pydantic is a data validation library that uses Python type hints to define schemas. If you haven't used it before, here's a quick introduction:

```python
from pydantic import BaseModel, Field
from typing import Optional

class BookInfo(BaseModel):
    """Schema for extracted book information."""
    
    title: str = Field(description="The book's title")
    author: str = Field(description="The author's full name")
    year: Optional[int] = Field(default=None, description="Publication year")
    genres: list[str] = Field(default_factory=list, description="List of genres")
```

This defines a schema where:
- `title` and `author` are required strings
- `year` is an optional integer (can be `None`)
- `genres` is a list of strings (defaults to empty list)

### Converting Schemas to Prompt Instructions

You can generate prompt instructions from your Pydantic schema:

```python
def schema_to_prompt_instructions(model_class: type[BaseModel]) -> str:
    """
    Convert a Pydantic model to prompt instructions for Claude.
    
    Args:
        model_class: A Pydantic BaseModel class
        
    Returns:
        A string with JSON format instructions
    """
    schema = model_class.model_json_schema()
    
    lines = ["Required JSON format:"]
    lines.append("{")
    
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    for name, prop in properties.items():
        prop_type = prop.get("type", "any")
        description = prop.get("description", "")
        is_required = name in required
        
        req_str = "required" if is_required else "optional"
        lines.append(f'    "{name}": {prop_type},  // {req_str} - {description}')
    
    lines.append("}")
    
    return "\n".join(lines)
```

Using this with our `BookInfo` class:

```python
print(schema_to_prompt_instructions(BookInfo))
```

Outputs:

```
Required JSON format:
{
    "title": string,  // required - The book's title
    "author": string,  // required - The author's full name
    "year": integer,  // optional - Publication year
    "genres": array,  // optional - List of genres
}
```

## Validating LLM Responses

Now for the crucial part: validating that Claude's response actually matches your schema.

### Basic Validation with Pydantic

```python
"""
Validating LLM responses with Pydantic.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import Optional

load_dotenv()

client = anthropic.Anthropic()


class BookInfo(BaseModel):
    """Schema for extracted book information."""
    
    title: str = Field(description="The book's title")
    author: str = Field(description="The author's full name")
    year: Optional[int] = Field(default=None, description="Publication year")
    genres: list[str] = Field(default_factory=list, description="List of genres")


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
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a data extraction assistant. Respond only with valid JSON.",
        messages=[
            {
                "role": "user",
                "content": f"""Extract book information from this description.

Required JSON format:
{{
    "title": "string (required)",
    "author": "string (required)", 
    "year": number or null,
    "genres": ["string"]
}}

Description: {description}"""
            }
        ]
    )
    
    response_text = message.content[0].text
    
    # Parse JSON
    data = json.loads(response_text)
    
    # Validate against schema
    return BookInfo.model_validate(data)


if __name__ == "__main__":
    description = """
    "1984" by George Orwell, published in 1949, is a dystopian novel 
    that explores themes of totalitarianism and surveillance.
    """
    
    try:
        book = extract_and_validate_book_info(description)
        print(f"Title: {book.title}")
        print(f"Author: {book.author}")
        print(f"Year: {book.year}")
        print(f"Genres: {book.genres}")
    except ValidationError as e:
        print(f"Validation failed: {e}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
```

### What Validation Catches

Pydantic validation catches several types of problems:

**Missing required fields:**
```python
# This will fail validation
data = {"title": "1984"}  # missing required 'author'
BookInfo.model_validate(data)  # ValidationError!
```

**Wrong types:**
```python
# This will fail validation
data = {"title": "1984", "author": "Orwell", "year": "nineteen forty-nine"}
BookInfo.model_validate(data)  # ValidationError! year must be int
```

**Invalid values:**
```python
# Using Field constraints
class BookInfo(BaseModel):
    title: str = Field(min_length=1)  # Can't be empty
    year: Optional[int] = Field(default=None, ge=1000, le=2100)  # Reasonable year range
```

## Handling Malformed Responses

No matter how good your prompts are, sometimes Claude's response won't be valid JSON. Here's how to handle that gracefully.

### The Robust Parsing Function

```python
"""
Robust response parsing with multiple fallback strategies.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
import re
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, ValidationError
from typing import TypeVar, Optional

load_dotenv()

client = anthropic.Anthropic()

T = TypeVar("T", bound=BaseModel)


def clean_json_response(text: str) -> str:
    """
    Clean up common issues in JSON responses from LLMs.
    
    Args:
        text: Raw response text that should contain JSON
        
    Returns:
        Cleaned text more likely to be valid JSON
    """
    # Remove markdown code blocks if present
    if "```json" in text:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Remove any text before the first { or [
    json_start = min(
        text.find("{") if text.find("{") != -1 else len(text),
        text.find("[") if text.find("[") != -1 else len(text)
    )
    if json_start < len(text):
        text = text[json_start:]
    
    # Remove any text after the last } or ]
    json_end = max(text.rfind("}"), text.rfind("]"))
    if json_end != -1:
        text = text[:json_end + 1]
    
    return text


def parse_with_retry(
    response_text: str,
    schema: type[T],
    client: anthropic.Anthropic,
    original_prompt: str,
    max_retries: int = 2
) -> T:
    """
    Parse a response with automatic retry on failure.
    
    Args:
        response_text: The raw response from Claude
        schema: Pydantic model class to validate against
        client: Anthropic client for retry requests
        original_prompt: The original user prompt (for context in retries)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Validated instance of the schema
        
    Raises:
        ValueError: If parsing fails after all retries
    """
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Clean the response
            cleaned = clean_json_response(response_text)
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Validate against schema
            return schema.model_validate(data)
            
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
        except ValidationError as e:
            last_error = f"Validation failed: {e}"
        
        # If we have retries left, ask Claude to fix the response
        if attempt < max_retries:
            print(f"Parse attempt {attempt + 1} failed: {last_error}")
            print("Requesting corrected response...")
            
            correction_message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system="You are a data extraction assistant. Respond only with valid JSON.",
                messages=[
                    {"role": "user", "content": original_prompt},
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": f"""Your previous response had an error: {last_error}

Please provide a corrected JSON response. Return ONLY the JSON object, no explanations or markdown."""
                    }
                ]
            )
            
            response_text = correction_message.content[0].text
    
    raise ValueError(f"Failed to parse response after {max_retries + 1} attempts: {last_error}")
```

### A Complete Parsing Utility

Let's put everything together into a reusable utility class:

```python
"""
Complete response parsing utility with validation.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, ValidationError
from typing import TypeVar, Generic, Optional

load_dotenv()

T = TypeVar("T", bound=BaseModel)


@dataclass
class ParseResult(Generic[T]):
    """Result of a parsing attempt."""
    
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    raw_response: str = ""
    attempts: int = 1


class ResponseParser:
    """
    Utility for parsing and validating LLM responses.
    
    This class provides robust parsing of JSON responses with:
    - Automatic cleanup of common formatting issues
    - Validation against Pydantic schemas
    - Retry logic for failed parses
    """
    
    def __init__(self, client: anthropic.Anthropic, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the parser.
        
        Args:
            client: Anthropic client instance
            model: Model to use for retry requests
        """
        self.client = client
        self.model = model
    
    def clean_response(self, text: str) -> str:
        """
        Clean common issues in JSON responses.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned text
        """
        # Remove markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
        
        text = text.strip()
        
        # Find JSON boundaries
        json_start = min(
            text.find("{") if text.find("{") != -1 else len(text),
            text.find("[") if text.find("[") != -1 else len(text)
        )
        json_end = max(text.rfind("}"), text.rfind("]"))
        
        if json_start < len(text) and json_end != -1:
            text = text[json_start:json_end + 1]
        
        return text
    
    def parse(
        self,
        response_text: str,
        schema: type[T],
        original_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> ParseResult[T]:
        """
        Parse and validate a response.
        
        Args:
            response_text: Raw response text from Claude
            schema: Pydantic model to validate against
            original_prompt: Original prompt for retry context
            max_retries: Maximum retry attempts
            
        Returns:
            ParseResult with success status and data or error
        """
        current_text = response_text
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                cleaned = self.clean_response(current_text)
                data = json.loads(cleaned)
                validated = schema.model_validate(data)
                
                return ParseResult(
                    success=True,
                    data=validated,
                    raw_response=response_text,
                    attempts=attempt + 1
                )
                
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e.msg}"
            except ValidationError as e:
                errors = "; ".join(
                    f"{err['loc']}: {err['msg']}" 
                    for err in e.errors()
                )
                last_error = f"Validation failed: {errors}"
            
            # Retry if we have attempts left and an original prompt
            if attempt < max_retries and original_prompt:
                correction = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system="You are a data extraction assistant. Respond only with valid JSON.",
                    messages=[
                        {"role": "user", "content": original_prompt},
                        {"role": "assistant", "content": current_text},
                        {
                            "role": "user",
                            "content": f"Error: {last_error}\n\nPlease provide corrected JSON only."
                        }
                    ]
                )
                current_text = correction.content[0].text
        
        return ParseResult(
            success=False,
            error=last_error,
            raw_response=response_text,
            attempts=max_retries + 1
        )


# Example usage
if __name__ == "__main__":
    from pydantic import Field
    
    class MovieInfo(BaseModel):
        """Schema for movie information."""
        title: str
        director: str
        year: int = Field(ge=1888, le=2100)  # Movies started in 1888
        genres: list[str] = Field(default_factory=list)
        rating: Optional[float] = Field(default=None, ge=0, le=10)
    
    client = anthropic.Anthropic()
    parser = ResponseParser(client)
    
    # Simulate a response that needs cleaning
    messy_response = '''Here's the movie information:
    
```json
{
    "title": "Inception",
    "director": "Christopher Nolan",
    "year": 2010,
    "genres": ["Sci-Fi", "Thriller"],
    "rating": 8.8
}
```

I hope this helps!'''
    
    result = parser.parse(
        messy_response,
        MovieInfo,
        original_prompt="Extract movie info from: Inception directed by Christopher Nolan..."
    )
    
    if result.success:
        print(f"Parsed successfully in {result.attempts} attempt(s):")
        print(f"  Title: {result.data.title}")
        print(f"  Director: {result.data.director}")
        print(f"  Year: {result.data.year}")
        print(f"  Genres: {result.data.genres}")
        print(f"  Rating: {result.data.rating}")
    else:
        print(f"Parsing failed: {result.error}")
```

## When to Use Structured vs Freeform Output

Not every response needs to be structured. Here's a decision framework:

### Use Structured Output When:

**Your code needs to process the response:**
```python
# Structured: code can reliably extract and use the data
{"sentiment": "positive", "confidence": 0.92, "keywords": ["great", "loved"]}
```

**The response feeds into another system:**
- Database storage
- API responses
- Next step in a workflow chain

**You need consistent format across many requests:**
- Batch processing
- Comparative analysis
- Automated testing

**The data has a clear, predictable shape:**
- Form fields
- Entity extraction
- Classification results

### Use Freeform Output When:

**The response is for human consumption:**
```python
# Freeform: natural language is more readable
"The sentiment is positive with high confidence. The reviewer particularly 
appreciated the product's durability and value for money."
```

**The output needs to be creative or flexible:**
- Creative writing
- Open-ended explanations
- Brainstorming responses

**You don't know the output structure in advance:**
- Exploratory analysis
- General Q&A
- Conversational interfaces

### The Hybrid Approach

Sometimes you want both—structured data for your code and a human-readable explanation:

```python
class AnalysisResult(BaseModel):
    """Schema for analysis with both structured data and explanation."""
    
    # Structured fields for programmatic use
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1)
    keywords: list[str]
    
    # Freeform field for human consumption
    explanation: str = Field(description="Human-readable analysis summary")
```

## Putting It All Together

Let's create a complete example that demonstrates everything we've learned:

```python
"""
Complete structured output example: Document analyzer.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
from datetime import date

load_dotenv()


# Define our schemas
class Person(BaseModel):
    """A person mentioned in a document."""
    name: str
    role: Optional[str] = None


class DocumentAnalysis(BaseModel):
    """Structured analysis of a document."""
    
    title: Optional[str] = Field(
        default=None,
        description="The document's title if identifiable"
    )
    document_type: str = Field(
        description="Type of document (email, report, article, memo, etc.)"
    )
    date_written: Optional[str] = Field(
        default=None,
        description="Date the document was written, if mentioned (YYYY-MM-DD format)"
    )
    author: Optional[Person] = Field(
        default=None,
        description="The document's author"
    )
    recipients: list[Person] = Field(
        default_factory=list,
        description="People the document is addressed to"
    )
    main_topics: list[str] = Field(
        description="Main topics discussed (2-5 topics)"
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Any action items or requests mentioned"
    )
    sentiment: str = Field(
        description="Overall tone: positive, negative, neutral, or mixed"
    )
    summary: str = Field(
        description="A 1-2 sentence summary of the document"
    )


class DocumentAnalyzer:
    """Analyzes documents and extracts structured information."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"
    
    def analyze(self, document: str, max_retries: int = 2) -> DocumentAnalysis:
        """
        Analyze a document and return structured information.
        
        Args:
            document: The document text to analyze
            max_retries: Number of retry attempts for parsing errors
            
        Returns:
            DocumentAnalysis object with extracted information
            
        Raises:
            ValueError: If analysis fails after all retries
        """
        prompt = f"""Analyze this document and extract structured information.

Return a JSON object with these fields:
- title: string or null
- document_type: string (email, report, article, memo, letter, etc.)
- date_written: string in YYYY-MM-DD format or null
- author: {{"name": string, "role": string or null}} or null
- recipients: array of {{"name": string, "role": string or null}}
- main_topics: array of 2-5 strings
- action_items: array of strings (empty if none)
- sentiment: one of "positive", "negative", "neutral", "mixed"
- summary: string (1-2 sentences)

Document:
---
{document}
---

Return ONLY the JSON object."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system="You are a document analysis assistant. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Clean potential markdown formatting
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(lines[1:-1])
                
                data = json.loads(cleaned)
                return DocumentAnalysis.model_validate(data)
                
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                
                if attempt < max_retries:
                    # Ask for correction
                    correction = self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system="You are a document analysis assistant. Always respond with valid JSON only.",
                        messages=[
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response_text},
                            {
                                "role": "user", 
                                "content": f"Error parsing response: {last_error}\n\nPlease return corrected JSON only."
                            }
                        ]
                    )
                    response_text = correction.content[0].text
        
        raise ValueError(f"Failed to analyze document: {last_error}")


if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    
    # Sample document to analyze
    document = """
    From: Sarah Chen, Project Manager
    To: Development Team
    Date: January 10, 2025
    Subject: Q1 Sprint Planning Update
    
    Hi team,
    
    I wanted to share some exciting updates about our Q1 sprint planning. 
    After reviewing last quarter's velocity, I'm confident we can tackle 
    the authentication refactor and the new dashboard features.
    
    Action items:
    1. Please review the updated Jira board by Friday
    2. Schedule your 1:1s with me for sprint capacity planning
    3. Mark any PTO in the shared calendar
    
    Looking forward to a great quarter!
    
    Best,
    Sarah
    """
    
    try:
        analysis = analyzer.analyze(document)
        
        print("Document Analysis Results")
        print("=" * 40)
        print(f"Type: {analysis.document_type}")
        print(f"Date: {analysis.date_written}")
        print(f"Author: {analysis.author.name if analysis.author else 'Unknown'}")
        print(f"Recipients: {[r.name for r in analysis.recipients]}")
        print(f"Topics: {analysis.main_topics}")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"\nSummary: {analysis.summary}")
        
        if analysis.action_items:
            print(f"\nAction Items:")
            for item in analysis.action_items:
                print(f"  • {item}")
                
    except ValueError as e:
        print(f"Analysis failed: {e}")
```

## Common Pitfalls

### 1. Not Handling Markdown Code Blocks

Claude often wraps JSON in markdown code blocks, even when you ask it not to:

```
Here's the JSON:

```json
{"key": "value"}
```
```

**Solution:** Always strip markdown formatting before parsing:

```python
if "```json" in text:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
```

### 2. Being Too Strict with Validation

If your schema is too strict, valid responses may fail validation:

```python
# Too strict: requires exact values
class Rating(BaseModel):
    score: Literal[1, 2, 3, 4, 5]  # Claude might return 4.5

# Better: allow reasonable flexibility
class Rating(BaseModel):
    score: float = Field(ge=1, le=5)
```

### 3. Not Providing Enough Schema Context

Claude does better when it understands the purpose of each field:

```python
# Minimal - Claude guesses at format
class Event(BaseModel):
    date: str
    
# Better - Claude knows exactly what you want
class Event(BaseModel):
    date: str = Field(
        description="Event date in ISO 8601 format (YYYY-MM-DD)"
    )
```

## Practical Exercise

**Task:** Build a resume parser that extracts structured information from resume text.

**Requirements:**

1. Create a Pydantic schema for resume data including:
   - Name and contact information
   - Work experience (list of positions with company, title, dates)
   - Education (list of degrees with school, degree, year)
   - Skills (categorized into technical and soft skills)

2. Implement a `ResumeParser` class that:
   - Takes resume text as input
   - Returns validated structured data
   - Handles parsing errors gracefully
   - Includes retry logic for malformed responses

3. Test with at least two different resume formats

**Hints:**
- Use nested Pydantic models for complex structures like work experience
- Make date fields flexible (people write dates many different ways)
- Consider making most fields optional since resumes vary widely

**Solution:** See `code/exercise_resume_parser.py`

## Key Takeaways

- **Structured outputs transform text into data** — JSON responses let your code reliably extract and use information from Claude's responses.

- **Pydantic provides validation** — Define schemas with type hints and get automatic validation, helpful error messages, and documentation.

- **Always handle malformed responses** — No matter how good your prompts are, plan for parsing failures. Clean up common issues and implement retry logic.

- **Include field descriptions in your schema** — Claude reads your schema instructions, so detailed descriptions improve output quality.

- **Match the output format to the use case** — Use structured outputs for programmatic processing, freeform for human consumption, or hybrid approaches when you need both.

## What's Next

In Chapter 14, we'll bring together everything from Part 2—tools, system prompts, and structured outputs—into a complete `AugmentedLLM` class. This class will serve as the foundation for all the workflow patterns and agents we build in the rest of the book. You'll have a single, reusable building block that handles the complexity of tool use and response parsing behind a clean interface.
