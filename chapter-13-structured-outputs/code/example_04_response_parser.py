"""
Complete response parsing utility with validation.

This module provides a reusable ResponseParser class that handles
all aspects of parsing and validating LLM responses.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field, ValidationError
from typing import TypeVar, Generic, Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

T = TypeVar("T", bound=BaseModel)


@dataclass
class ParseResult(Generic[T]):
    """
    Result of a parsing attempt.
    
    Attributes:
        success: Whether parsing succeeded
        data: The validated data (if successful)
        error: Error message (if failed)
        raw_response: The original response text
        attempts: Number of parsing attempts made
    """
    
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    raw_response: str = ""
    attempts: int = 1
    
    def __str__(self) -> str:
        if self.success:
            return f"ParseResult(success=True, attempts={self.attempts})"
        return f"ParseResult(success=False, error='{self.error}', attempts={self.attempts})"


class ResponseParser:
    """
    Utility for parsing and validating LLM responses.
    
    This class provides robust parsing of JSON responses with:
    - Automatic cleanup of common formatting issues
    - Validation against Pydantic schemas
    - Retry logic for failed parses
    
    Example:
        parser = ResponseParser(client)
        result = parser.parse(response_text, MySchema)
        if result.success:
            print(result.data)
        else:
            print(f"Error: {result.error}")
    """
    
    def __init__(
        self, 
        client: anthropic.Anthropic, 
        model: str = "claude-sonnet-4-20250514",
        verbose: bool = False
    ):
        """
        Initialize the parser.
        
        Args:
            client: Anthropic client instance
            model: Model to use for retry requests
            verbose: Whether to print debug information
        """
        self.client = client
        self.model = model
        self.verbose = verbose
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ResponseParser] {message}")
    
    def clean_response(self, text: str) -> str:
        """
        Clean common issues in JSON responses.
        
        Handles:
        - Markdown code blocks (```json ... ```)
        - Text before/after the JSON object
        - Extra whitespace
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned text
        """
        original_len = len(text)
        
        # Remove markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
                self._log("Extracted content from ```json block")
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
                self._log("Extracted content from ``` block")
        
        text = text.strip()
        
        # Find JSON boundaries
        json_start_brace = text.find("{")
        json_start_bracket = text.find("[")
        
        if json_start_brace == -1:
            json_start = json_start_bracket
        elif json_start_bracket == -1:
            json_start = json_start_brace
        else:
            json_start = min(json_start_brace, json_start_bracket)
        
        json_end = max(text.rfind("}"), text.rfind("]"))
        
        if json_start != -1 and json_start < len(text) and json_end != -1:
            if json_start > 0:
                self._log(f"Removed {json_start} chars of prefix")
            if json_end < len(text) - 1:
                self._log(f"Removed {len(text) - json_end - 1} chars of suffix")
            text = text[json_start:json_end + 1]
        
        if len(text) != original_len:
            self._log(f"Cleaned: {original_len} -> {len(text)} chars")
        
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
            max_retries: Maximum retry attempts (0 for no retries)
            
        Returns:
            ParseResult with success status and data or error
        """
        current_text = response_text
        last_error = None
        
        for attempt in range(max_retries + 1):
            self._log(f"Parse attempt {attempt + 1}/{max_retries + 1}")
            
            try:
                # Clean the response
                cleaned = self.clean_response(current_text)
                
                # Parse JSON
                data = json.loads(cleaned)
                
                # Validate against schema
                validated = schema.model_validate(data)
                
                self._log("Parse successful!")
                return ParseResult(
                    success=True,
                    data=validated,
                    raw_response=response_text,
                    attempts=attempt + 1
                )
                
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON at position {e.pos}: {e.msg}"
                self._log(f"JSON error: {last_error}")
            except ValidationError as e:
                errors = "; ".join(
                    f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" 
                    for err in e.errors()
                )
                last_error = f"Validation failed: {errors}"
                self._log(f"Validation error: {last_error}")
            
            # Retry if we have attempts left and an original prompt
            if attempt < max_retries and original_prompt:
                self._log("Requesting correction from Claude...")
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
                self._log(f"Got corrected response ({len(current_text)} chars)")
        
        return ParseResult(
            success=False,
            error=last_error,
            raw_response=response_text,
            attempts=max_retries + 1
        )
    
    def parse_or_raise(
        self,
        response_text: str,
        schema: type[T],
        original_prompt: Optional[str] = None,
        max_retries: int = 2
    ) -> T:
        """
        Parse and validate, raising an exception on failure.
        
        This is a convenience method for when you want to handle
        errors with try/except rather than checking result.success.
        
        Args:
            response_text: Raw response text from Claude
            schema: Pydantic model to validate against
            original_prompt: Original prompt for retry context
            max_retries: Maximum retry attempts
            
        Returns:
            Validated instance of the schema
            
        Raises:
            ValueError: If parsing fails after all retries
        """
        result = self.parse(response_text, schema, original_prompt, max_retries)
        
        if result.success:
            return result.data
        
        raise ValueError(f"Failed to parse response: {result.error}")


# Example schemas for demonstration
class ContactInfo(BaseModel):
    """Schema for contact information."""
    
    name: str = Field(description="Person's full name")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    company: Optional[str] = Field(default=None, description="Company name")


class MeetingDetails(BaseModel):
    """Schema for meeting information."""
    
    title: str = Field(description="Meeting title")
    date: str = Field(description="Meeting date (YYYY-MM-DD)")
    time: str = Field(description="Meeting time (HH:MM)")
    attendees: list[str] = Field(description="List of attendee names")
    agenda: list[str] = Field(default_factory=list, description="Agenda items")
    location: Optional[str] = Field(default=None, description="Meeting location")


if __name__ == "__main__":
    client = anthropic.Anthropic()
    
    # Create parser with verbose output for demonstration
    parser = ResponseParser(client, verbose=True)
    
    # Example 1: Clean response that parses immediately
    print("=" * 60)
    print("Example 1: Clean Response")
    print("=" * 60)
    
    clean_response = '{"name": "John Smith", "email": "john@example.com", "phone": "555-1234"}'
    result = parser.parse(clean_response, ContactInfo)
    print(f"\nResult: {result}")
    if result.success:
        print(f"Data: name={result.data.name}, email={result.data.email}")
    
    # Example 2: Response with markdown formatting
    print("\n" + "=" * 60)
    print("Example 2: Markdown-Wrapped Response")
    print("=" * 60)
    
    markdown_response = '''Here's the contact information:

```json
{
    "name": "Jane Doe",
    "email": "jane@company.com",
    "company": "Tech Corp"
}
```

Let me know if you need more details!'''
    
    result = parser.parse(markdown_response, ContactInfo)
    print(f"\nResult: {result}")
    if result.success:
        print(f"Data: name={result.data.name}, company={result.data.company}")
    
    # Example 3: Real extraction with potential retry
    print("\n" + "=" * 60)
    print("Example 3: Real Extraction with Retry Logic")
    print("=" * 60)
    
    prompt = """Extract meeting details from this text:

"Let's schedule the Q1 Planning Meeting for January 15, 2025 at 2:30 PM.
Attendees will be Sarah, Mike, and Lisa. We'll meet in Conference Room A.
Agenda items: Review Q4 results, Set Q1 goals, Assign responsibilities."

Return as JSON with fields: title, date (YYYY-MM-DD), time (HH:MM), 
attendees (array), agenda (array), location."""

    # Make the actual API call
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="You are a data extraction assistant. Respond only with valid JSON.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = parser.parse(
        message.content[0].text,
        MeetingDetails,
        original_prompt=prompt,
        max_retries=2
    )
    
    print(f"\nResult: {result}")
    if result.success:
        meeting = result.data
        print(f"\nExtracted Meeting Details:")
        print(f"  Title: {meeting.title}")
        print(f"  Date: {meeting.date}")
        print(f"  Time: {meeting.time}")
        print(f"  Location: {meeting.location}")
        print(f"  Attendees: {', '.join(meeting.attendees)}")
        print(f"  Agenda:")
        for item in meeting.agenda:
            print(f"    - {item}")
    else:
        print(f"Failed to parse: {result.error}")
    
    # Example 4: Using parse_or_raise
    print("\n" + "=" * 60)
    print("Example 4: Using parse_or_raise")
    print("=" * 60)
    
    try:
        contact = parser.parse_or_raise(
            '{"name": "Bob Wilson", "email": "bob@test.com"}',
            ContactInfo
        )
        print(f"Parsed contact: {contact.name}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Demonstrate failure
    try:
        parser.parse_or_raise(
            '{"not_valid": "missing required name field"}',
            ContactInfo
        )
    except ValueError as e:
        print(f"Expected error: {e}")
