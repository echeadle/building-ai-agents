"""
Prompt chain with quality gates between steps.

This example demonstrates how to add validation between chain steps
to catch and retry bad outputs before they propagate through the chain.

Features:
- Rule-based validation (length, content checks)
- LLM-based quality assessment
- Retry logic for failed validations
- Detailed result tracking

Chapter 17: Prompt Chaining - Implementation
"""

import os
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class QualityCheckResult:
    """Result of a quality gate check."""
    passed: bool
    message: str
    score: Optional[float] = None


class ChainError(Exception):
    """Raised when a chain step fails validation after all retries."""
    def __init__(self, step: str, message: str, output: str):
        self.step = step
        self.output = output
        super().__init__(f"Chain failed at '{step}': {message}")


def check_content_quality(content: str, min_length: int = 50) -> QualityCheckResult:
    """
    Quality gate: Check if generated content meets basic requirements.
    
    This is a simple rule-based check that validates:
    - Minimum content length
    - Presence of a call to action
    
    Args:
        content: The generated content to validate
        min_length: Minimum acceptable character count
    
    Returns:
        QualityCheckResult indicating whether the content passed
    """
    # Check minimum length
    if len(content) < min_length:
        return QualityCheckResult(
            passed=False,
            message=f"Content too short: {len(content)} chars (minimum: {min_length})"
        )
    
    # Check for call to action (simple heuristic)
    cta_indicators = [
        "try", "start", "get", "discover", "learn", 
        "join", "contact", "visit", "sign up", "download",
        "explore", "experience", "transform", "unlock"
    ]
    has_cta = any(word in content.lower() for word in cta_indicators)
    
    if not has_cta:
        return QualityCheckResult(
            passed=False,
            message="Content missing call to action"
        )
    
    return QualityCheckResult(
        passed=True,
        message="Content meets quality requirements"
    )


def check_translation_quality(
    original: str, 
    translated: str, 
    target_language: str
) -> QualityCheckResult:
    """
    Quality gate: Use LLM to verify translation quality.
    
    This demonstrates using an LLM as a quality checker—a powerful pattern
    for validating complex outputs that can't be checked with simple rules.
    
    Args:
        original: The original English content
        translated: The translated content
        target_language: The target language
    
    Returns:
        QualityCheckResult with quality score and assessment
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"""Evaluate this translation. Rate it 1-10 and identify any issues.

Original (English):
{original}

Translation ({target_language}):
{translated}

Consider:
- Accuracy of meaning
- Natural phrasing in target language
- Preservation of tone and intent
- Appropriate adaptation of idioms

Respond in this exact format:
SCORE: [1-10]
PASSED: [YES/NO] (YES if score >= 7)
ISSUES: [Brief description or "None"]"""
            }
        ]
    )
    
    result_text = response.content[0].text
    
    # Parse the response
    try:
        lines = result_text.strip().split("\n")
        score_line = [l for l in lines if l.startswith("SCORE:")][0]
        passed_line = [l for l in lines if l.startswith("PASSED:")][0]
        issues_line = [l for l in lines if l.startswith("ISSUES:")][0]
        
        score = float(score_line.split(":")[1].strip())
        passed = "YES" in passed_line.upper()
        issues = issues_line.split(":", 1)[1].strip()
        
        return QualityCheckResult(
            passed=passed,
            message=issues if issues != "None" else "Translation approved",
            score=score
        )
    except (IndexError, ValueError) as e:
        # If parsing fails, be conservative and pass
        return QualityCheckResult(
            passed=True,
            message=f"Could not parse quality check (assuming pass): {result_text[:100]}",
            score=None
        )


def generate_content(topic: str, style: str = "professional") -> str:
    """
    Step 1: Generate marketing copy.
    
    Args:
        topic: The subject to write about
        style: The tone (professional, casual, enthusiastic)
    
    Returns:
        Generated marketing copy
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Write a short marketing paragraph (3-4 sentences) about: {topic}
                
Style: {style}

Requirements:
- Focus on benefits
- Include a clear call to action
- Keep it concise but compelling"""
            }
        ]
    )
    return response.content[0].text


def translate_content(content: str, target_language: str) -> str:
    """
    Step 2: Translate content to another language.
    
    Args:
        content: The text to translate
        target_language: The language to translate into
    
    Returns:
        Translated content
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Translate the following marketing copy to {target_language}.

Maintain the same tone, style, and persuasive intent. Adapt idioms naturally.

Content to translate:
{content}"""
            }
        ]
    )
    return response.content[0].text


def content_chain_with_gates(
    topic: str, 
    target_language: str, 
    style: str = "professional",
    max_retries: int = 2
) -> dict:
    """
    Execute content chain with quality gates and retry logic.
    
    This function:
    1. Generates content and validates it passes quality checks
    2. Translates the content and validates translation quality
    3. Retries failed steps up to max_retries times
    4. Tracks all attempts and quality checks for inspection
    
    Args:
        topic: The subject to write about
        target_language: The language to translate into
        style: The tone of the content
        max_retries: Number of times to retry a failed step
    
    Returns:
        Dictionary containing original and translated content with quality scores
    
    Raises:
        ChainError: If a step fails validation after all retries
    """
    results = {
        "topic": topic,
        "style": style,
        "target_language": target_language,
        "steps": []
    }
    
    # Step 1: Generate content (with retries)
    print(f"Step 1: Generating {style} content about '{topic}'...")
    
    for attempt in range(max_retries + 1):
        original_content = generate_content(topic, style)
        quality_check = check_content_quality(original_content)
        
        # Track this attempt
        results["steps"].append({
            "step": "generate",
            "attempt": attempt + 1,
            "output_preview": original_content[:100] + "...",
            "quality_check": {
                "passed": quality_check.passed,
                "message": quality_check.message
            }
        })
        
        if quality_check.passed:
            print(f"  ✓ Quality gate passed: {quality_check.message}")
            break
        else:
            print(f"  ✗ Quality gate failed (attempt {attempt + 1}): {quality_check.message}")
            if attempt == max_retries:
                raise ChainError("generate", quality_check.message, original_content)
    
    results["original"] = original_content
    
    # Step 2: Translate content (with retries)
    print(f"\nStep 2: Translating to {target_language}...")
    
    for attempt in range(max_retries + 1):
        translated_content = translate_content(original_content, target_language)
        quality_check = check_translation_quality(
            original_content, 
            translated_content, 
            target_language
        )
        
        # Track this attempt
        results["steps"].append({
            "step": "translate",
            "attempt": attempt + 1,
            "output_preview": translated_content[:100] + "...",
            "quality_check": {
                "passed": quality_check.passed,
                "message": quality_check.message,
                "score": quality_check.score
            }
        })
        
        if quality_check.passed:
            print(f"  ✓ Quality gate passed (score: {quality_check.score}): {quality_check.message}")
            break
        else:
            print(f"  ✗ Quality gate failed (attempt {attempt + 1}): {quality_check.message}")
            if attempt == max_retries:
                raise ChainError("translate", quality_check.message, translated_content)
    
    results["translated"] = translated_content
    results["translation_score"] = quality_check.score
    
    return results


if __name__ == "__main__":
    try:
        result = content_chain_with_gates(
            topic="AI-powered customer support chatbots",
            target_language="French",
            style="professional"
        )
        
        print("\n" + "="*50)
        print("CHAIN COMPLETED SUCCESSFULLY")
        print("="*50)
        
        print(f"\nTopic: {result['topic']}")
        print(f"Style: {result['style']}")
        
        print(f"\nOriginal:\n{result['original']}")
        
        print(f"\nTranslated ({result['target_language']}):")
        print(f"Score: {result['translation_score']}/10")
        print(f"{result['translated']}")
        
        print(f"\nStep History:")
        for step in result["steps"]:
            status = "✓" if step["quality_check"]["passed"] else "✗"
            print(f"  {status} {step['step']} (attempt {step['attempt']}): {step['quality_check']['message']}")
        
    except ChainError as e:
        print(f"\n❌ Chain failed at step '{e.step}'")
        print(f"Reason: {e}")
        print(f"Last output: {e.output[:200]}...")
