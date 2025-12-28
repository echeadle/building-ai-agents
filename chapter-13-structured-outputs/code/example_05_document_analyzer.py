"""
Complete structured output example: Document analyzer.

This example brings together everything from Chapter 13 to create
a practical document analysis tool with complex nested schemas.

Chapter 13: Structured Outputs and Response Parsing
"""

import os
import json
import re
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


# Define our schemas
class Person(BaseModel):
    """A person mentioned in a document."""
    
    name: str = Field(description="Person's name")
    role: Optional[str] = Field(
        default=None, 
        description="Person's role or title if mentioned"
    )


class DocumentAnalysis(BaseModel):
    """Structured analysis of a document."""
    
    title: Optional[str] = Field(
        default=None,
        description="The document's title if identifiable"
    )
    document_type: str = Field(
        description="Type of document (email, report, article, memo, letter, etc.)"
    )
    date_written: Optional[str] = Field(
        default=None,
        description="Date the document was written (YYYY-MM-DD format if possible)"
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
    key_dates: list[str] = Field(
        default_factory=list,
        description="Important dates mentioned in the document"
    )
    urgency: str = Field(
        default="normal",
        description="Urgency level: low, normal, high, or urgent"
    )


class DocumentAnalyzer:
    """
    Analyzes documents and extracts structured information.
    
    This class demonstrates production-ready structured output
    extraction with validation and error handling.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            verbose: Whether to print debug information
        """
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"
        self.verbose = verbose
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[Analyzer] {message}")
    
    def _clean_response(self, text: str) -> str:
        """Clean potential markdown formatting from response."""
        # Remove markdown code blocks
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        elif "```" in text:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Find JSON boundaries
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1:
            return text[start:end + 1]
        
        return text
    
    def _build_prompt(self, document: str) -> str:
        """Build the extraction prompt."""
        return f"""Analyze this document and extract structured information.

Return a JSON object with these fields:
- title: string or null (document title if identifiable)
- document_type: string (email, report, article, memo, letter, etc.)
- date_written: string in YYYY-MM-DD format or null
- author: object with "name" and "role" (string or null), or null if unknown
- recipients: array of objects with "name" and "role" fields
- main_topics: array of 2-5 strings (main topics discussed)
- action_items: array of strings (tasks, requests, or to-dos)
- sentiment: one of "positive", "negative", "neutral", "mixed"
- summary: string (1-2 sentence summary)
- key_dates: array of strings (important dates/deadlines mentioned)
- urgency: one of "low", "normal", "high", "urgent"

Document:
---
{document}
---

Return ONLY the JSON object, no other text or markdown formatting."""

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
        prompt = self._build_prompt(document)
        self._log(f"Analyzing document ({len(document)} chars)")
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system="You are a document analysis assistant. Always respond with valid JSON only, no markdown or explanations.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        self._log(f"Received response ({len(response_text)} chars)")
        last_error = None
        
        for attempt in range(max_retries + 1):
            self._log(f"Parse attempt {attempt + 1}/{max_retries + 1}")
            
            try:
                cleaned = self._clean_response(response_text)
                data = json.loads(cleaned)
                result = DocumentAnalysis.model_validate(data)
                self._log("Successfully parsed and validated!")
                return result
                
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e.msg}"
                self._log(f"JSON error: {last_error}")
            except ValidationError as e:
                errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                last_error = f"Validation failed: {'; '.join(errors)}"
                self._log(f"Validation error: {last_error}")
            
            # Retry if we have attempts left
            if attempt < max_retries:
                self._log("Requesting corrected response...")
                correction = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system="You are a document analysis assistant. Always respond with valid JSON only.",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user", 
                            "content": f"Error parsing response: {last_error}\n\nPlease return corrected JSON only, no markdown."
                        }
                    ]
                )
                response_text = correction.content[0].text
        
        raise ValueError(f"Failed to analyze document: {last_error}")
    
    def analyze_batch(
        self, 
        documents: list[str]
    ) -> list[tuple[Optional[DocumentAnalysis], Optional[str]]]:
        """
        Analyze multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of (result, error) tuples. If successful, result is 
            DocumentAnalysis and error is None. If failed, result is 
            None and error contains the error message.
        """
        results = []
        
        for i, doc in enumerate(documents):
            self._log(f"Processing document {i + 1}/{len(documents)}")
            try:
                analysis = self.analyze(doc)
                results.append((analysis, None))
            except ValueError as e:
                results.append((None, str(e)))
        
        return results


def print_analysis(analysis: DocumentAnalysis) -> None:
    """Pretty print a document analysis."""
    print("\n" + "=" * 50)
    print("DOCUMENT ANALYSIS")
    print("=" * 50)
    
    if analysis.title:
        print(f"Title: {analysis.title}")
    print(f"Type: {analysis.document_type}")
    print(f"Date: {analysis.date_written or 'Unknown'}")
    print(f"Sentiment: {analysis.sentiment}")
    print(f"Urgency: {analysis.urgency}")
    
    if analysis.author:
        author_str = analysis.author.name
        if analysis.author.role:
            author_str += f" ({analysis.author.role})"
        print(f"Author: {author_str}")
    
    if analysis.recipients:
        recipients = []
        for r in analysis.recipients:
            r_str = r.name
            if r.role:
                r_str += f" ({r.role})"
            recipients.append(r_str)
        print(f"Recipients: {', '.join(recipients)}")
    
    print(f"\nSummary: {analysis.summary}")
    
    print(f"\nMain Topics:")
    for topic in analysis.main_topics:
        print(f"  • {topic}")
    
    if analysis.action_items:
        print(f"\nAction Items:")
        for item in analysis.action_items:
            print(f"  ☐ {item}")
    
    if analysis.key_dates:
        print(f"\nKey Dates:")
        for date in analysis.key_dates:
            print(f"  📅 {date}")


if __name__ == "__main__":
    # Create analyzer with verbose output
    analyzer = DocumentAnalyzer(verbose=True)
    
    # Sample documents to analyze
    documents = [
        # Document 1: Business email
        """
From: Sarah Chen, Project Manager
To: Development Team, Marketing Team
Date: January 10, 2025
Subject: URGENT: Q1 Sprint Planning Update

Hi team,

I wanted to share some exciting updates about our Q1 sprint planning. 
After reviewing last quarter's velocity, I'm confident we can tackle 
the authentication refactor and the new dashboard features.

Key dates to remember:
- Sprint kickoff: January 13, 2025
- Mid-sprint review: January 24, 2025
- Sprint demo: February 3, 2025

Action items:
1. Please review the updated Jira board by Friday
2. Schedule your 1:1s with me for sprint capacity planning
3. Mark any PTO in the shared calendar

This is time-sensitive as we need to finalize the scope by Monday.

Looking forward to a great quarter!

Best,
Sarah
        """,
        
        # Document 2: Formal memo
        """
MEMORANDUM

TO: All Department Heads
FROM: Robert Williams, Chief Operating Officer
DATE: January 8, 2025
RE: Budget Freeze Implementation

Due to current market conditions and the need to preserve capital through Q1, 
I am announcing an immediate freeze on all non-essential expenditures effective 
January 15, 2025.

This freeze applies to:
- New equipment purchases over $500
- Travel requests (exceptions require VP approval)
- External contractor engagements
- Software license expansions

Essential operations and previously approved projects will continue as planned. 
Please review your departmental budgets and identify any items that can be 
deferred to Q2.

Submit your revised budget projections to Finance by January 20, 2025.

This situation is temporary, and we expect to resume normal operations by 
April 1, 2025 pending Q1 results.

Questions should be directed to the Finance department.
        """,
        
        # Document 3: Informal team update
        """
Hey everyone!

Quick update from the hackathon last weekend - we won second place! 🎉

The judges really liked our AI-powered accessibility tool. They especially 
mentioned the voice navigation feature that Mike built.

A few things to share:
- We got $2,500 in AWS credits
- TechCrunch wants to do a small feature on us
- The event organizers invited us to present at their March conference

Nothing urgent, but let's sync up next week to discuss if we want to 
develop this further. Could be a cool side project or maybe even something 
more.

Pizza's on me at the next team lunch!

- Alex
        """
    ]
    
    # Analyze each document
    for i, doc in enumerate(documents):
        print(f"\n{'#' * 60}")
        print(f"# DOCUMENT {i + 1}")
        print(f"{'#' * 60}")
        
        # Show first 100 chars of document
        preview = doc.strip()[:100].replace('\n', ' ')
        print(f"Preview: {preview}...")
        
        try:
            analysis = analyzer.analyze(doc)
            print_analysis(analysis)
        except ValueError as e:
            print(f"\nFailed to analyze: {e}")
        
        print()
