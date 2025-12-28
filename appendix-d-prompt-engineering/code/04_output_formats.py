"""
Output Format Instructions Example

Demonstrates how to get structured, predictable outputs from agents.

Appendix D: Prompt Engineering for Agents
"""

import os
from dotenv import load_dotenv
import anthropic
import json

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()


def test_json_output_format() -> None:
    """Demonstrate requesting JSON output with schema."""
    
    print("\n" + "="*70)
    print("JSON OUTPUT FORMAT")
    print("="*70)
    
    system_prompt = """You are a research assistant.

When you complete your research, respond with ONLY a JSON object (no preamble, no explanation).

Use this schema:
{
    "summary": "2-3 sentence overview",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "sources": [
        {
            "url": "https://example.com",
            "title": "Source title",
            "relevance": "Why this source matters"
        }
    ],
    "confidence": "high|medium|low",
    "gaps": ["Information that couldn't be found"]
}

Example:
{
    "summary": "Quantum computing made significant progress in 2024...",
    "key_findings": [
        "IBM achieved 127-qubit processor",
        "Error correction remains main challenge"
    ],
    "sources": [
        {
            "url": "https://ibm.com/quantum",
            "title": "IBM Quantum Update",
            "relevance": "Primary source for hardware progress"
        }
    ],
    "confidence": "high",
    "gaps": ["Commercial timeline still uncertain"]
}

CRITICAL: Respond ONLY with valid JSON. No text before or after."""
    
    query = "Research the current state of quantum computing (use mock data for this example)"
    
    print(f"\nQuery: {query}\n")
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        result_text = response.content[0].text
        print("Raw Response:")
        print("-" * 70)
        print(result_text)
        
        # Try to parse as JSON
        print("\n\nParsing as JSON:")
        print("-" * 70)
        try:
            data = json.loads(result_text)
            print("✓ Successfully parsed!")
            print(json.dumps(data, indent=2))
        except json.JSONDecodeError as e:
            print(f"✗ JSON parsing failed: {e}")
            
    except Exception as e:
        print(f"Error: {e}")


def test_markdown_output_format() -> None:
    """Demonstrate requesting Markdown formatted output."""
    
    print("\n" + "="*70)
    print("MARKDOWN OUTPUT FORMAT")
    print("="*70)
    
    system_prompt = """You are a research assistant.

Format your final report as Markdown with this EXACT structure:

# [Research Topic]

## Executive Summary
[2-3 paragraphs]

## Key Findings

### Finding 1: [Title]
[Details]
*Source: [URL]*

### Finding 2: [Title]
[Details]
*Source: [URL]*

### Finding 3: [Title]
[Details]
*Source: [URL]*

## Methodology
- Searches performed: [X]
- Sources reviewed: [Y]
- Date range: [Z]

## Conclusions
[Synthesis paragraph]

## References
1. [Title] - [URL]
2. [Title] - [URL]

---
*Report generated: [DATE]*

Follow this structure EXACTLY. Use this Markdown formatting, not any other format."""
    
    query = "Research electric vehicle adoption trends (use mock data for this example)"
    
    print(f"\nQuery: {query}\n")
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        result_text = response.content[0].text
        print("Formatted Report:")
        print("="*70)
        print(result_text)
        
        # Validate structure
        print("\n\nValidating Structure:")
        print("-" * 70)
        required_sections = [
            "# ",  # Title
            "## Executive Summary",
            "## Key Findings",
            "## Methodology",
            "## Conclusions",
            "## References"
        ]
        
        for section in required_sections:
            if section in result_text:
                print(f"✓ Found: {section}")
            else:
                print(f"✗ Missing: {section}")
                
    except Exception as e:
        print(f"Error: {e}")


def test_conversational_with_citations() -> None:
    """Demonstrate inline citation format."""
    
    print("\n" + "="*70)
    print("CONVERSATIONAL WITH INLINE CITATIONS")
    print("="*70)
    
    system_prompt = """You are a research assistant.

When presenting information from research, cite sources inline using this format:

According to recent research [1], quantum computers achieved a significant 
milestone in 2024. IBM's 127-qubit processor [2] represents a major advance, 
though error correction remains challenging [1][3].

[1] Nature: "Quantum Computing Progress Report 2024" - https://...
[2] IBM Blog: "Introducing 127-Qubit Processor" - https://...
[3] MIT Review: "The Error Correction Challenge" - https://...

Guidelines:
- Number citations as you introduce new sources
- Provide full citation list at the end
- Use multiple citations for well-supported claims
- Note conflicting sources explicitly

Write naturally, but cite everything."""
    
    query = "Explain recent progress in quantum computing (use mock sources)"
    
    print(f"\nQuery: {query}\n")
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        result_text = response.content[0].text
        print("Response with Citations:")
        print("="*70)
        print(result_text)
        
        # Check for citations
        print("\n\nCitation Check:")
        print("-" * 70)
        if "[1]" in result_text:
            print("✓ Contains numbered citations")
        else:
            print("✗ No numbered citations found")
            
        if "https://" in result_text:
            print("✓ Contains URLs")
        else:
            print("✗ No URLs found")
            
    except Exception as e:
        print(f"Error: {e}")


def demonstrate_format_enforcement() -> None:
    """Show techniques for enforcing output format."""
    
    print("\n" + "="*70)
    print("FORMAT ENFORCEMENT TECHNIQUES")
    print("="*70)
    
    techniques = {
        "Explicit Schema": """
Provide the EXACT schema/structure you want:
- Use code blocks to show structure
- Include example output
- Mark required vs optional fields
""",
        
        "Negative Instructions": """
Tell the agent what NOT to do:
- "ONLY JSON, no preamble"
- "No explanatory text before the report"
- "Do not use HTML tags, use Markdown"
""",
        
        "Format Validation": """
In your code, validate the output:
```python
def validate_format(text: str) -> bool:
    try:
        data = json.loads(text)
        required = ['summary', 'findings', 'sources']
        return all(key in data for key in required)
    except:
        return False
```
""",
        
        "Escape Hatches": """
Allow the agent to explain issues:
"If you cannot format the response as JSON because [reason],
respond with: ERROR: [explanation]"
""",
        
        "Progressive Refinement": """
First ask for content, then ask for formatting:
1. "Research the topic and list findings"
2. "Now format those findings as JSON using this schema..."
"""
    }
    
    for name, technique in techniques.items():
        print(f"\n{name}:")
        print("-" * 70)
        print(technique)


def demonstrate_common_format_pitfalls() -> None:
    """Show common mistakes with format instructions."""
    
    print("\n" + "="*70)
    print("COMMON FORMAT INSTRUCTION PITFALLS")
    print("="*70)
    
    pitfalls = [
        {
            "mistake": "Vague format request",
            "bad": "Please format your response nicely",
            "good": "Use Markdown with headers (##) and bullet lists (-)"
        },
        {
            "mistake": "Conflicting instructions",
            "bad": "Respond in JSON. Also, write a natural paragraph summary.",
            "good": "Respond in JSON with a 'summary' field containing a paragraph"
        },
        {
            "mistake": "No example provided",
            "bad": "Return data as JSON with sources and findings",
            "good": 'Return JSON like: {"findings": [...], "sources": [...]}'
        },
        {
            "mistake": "Format not enforced",
            "bad": "Preferably use JSON format",
            "good": "You MUST respond ONLY with valid JSON"
        },
        {
            "mistake": "Ambiguous structure",
            "bad": "Include sources somewhere",
            "good": "Include a 'sources' array at the end of the JSON"
        }
    ]
    
    for i, pitfall in enumerate(pitfalls, 1):
        print(f"\n{i}. {pitfall['mistake']}:")
        print(f"   ❌ Bad: {pitfall['bad']}")
        print(f"   ✅ Good: {pitfall['good']}")


if __name__ == "__main__":
    print("Output Format Instructions Example")
    print("=" * 70)
    print("This example shows how to get structured outputs from agents.")
    print()
    
    # Test JSON output
    test_json_output_format()
    
    # Test Markdown output
    test_markdown_output_format()
    
    # Test conversational with citations
    test_conversational_with_citations()
    
    # Show enforcement techniques
    demonstrate_format_enforcement()
    
    # Show common pitfalls
    demonstrate_common_format_pitfalls()
    
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
1. BE EXPLICIT about format requirements
   - Show the exact structure you want
   - Provide examples
   - Use code blocks for schemas

2. USE STRONG LANGUAGE for enforcement
   - "MUST respond with JSON"
   - "ONLY valid JSON, no other text"
   - "EXACTLY this structure"

3. VALIDATE in your code
   - Don't assume format compliance
   - Parse and validate output
   - Handle format errors gracefully

4. START SIMPLE, then refine
   - Basic format first
   - Add structure requirements
   - Iterate based on failures

5. PROVIDE ESCAPE HATCHES
   - Allow agent to report formatting issues
   - Don't force invalid formats

Remember: The more specific your format instructions,
the more reliably the agent will follow them.
""")
