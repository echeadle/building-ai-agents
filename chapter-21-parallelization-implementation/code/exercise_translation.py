"""
Exercise Solution: Parallel Translation System with Voting

Chapter 21: Parallelization - Implementation

This system combines sectioning (for multiple target languages) with
voting (for translation quality through different translator personas).

Features:
- Translates to multiple languages in parallel (sectioning)
- Uses 3 translator personas per language for quality voting
- Reports confidence scores based on translator agreement
- Handles failures gracefully
"""

import asyncio
import os
from collections import Counter
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class TranslatorPersona:
    """
    A translator with a specific style and approach.
    
    Attributes:
        name: Identifier for this persona
        style: Description of translation style
        system_prompt: Instructions for this translator
        temperature: Sampling temperature
    """
    name: str
    style: str
    system_prompt: str
    temperature: float = 0.7


@dataclass
class Translation:
    """
    A single translation from one persona.
    
    Attributes:
        persona_name: Which translator produced this
        text: The translated text
        style: The style used
        success: Whether translation succeeded
        error: Error message if failed
    """
    persona_name: str
    text: str
    style: str
    success: bool = True
    error: str | None = None


@dataclass
class LanguageTranslationResult:
    """
    Translation results for a single target language.
    
    Attributes:
        language: Target language
        translations: Individual translations from each persona
        best_translation: The voted-best translation
        confidence: Agreement level among translators
        execution_time: Time for this language's translations
    """
    language: str
    translations: list[Translation] = field(default_factory=list)
    best_translation: str = ""
    confidence: float = 0.0
    execution_time: float = 0.0


@dataclass
class TranslationReport:
    """
    Complete translation report for all languages.
    
    Attributes:
        source_text: Original text
        source_language: Detected/specified source language
        language_results: Results for each target language
        total_execution_time: Total time for all translations
    """
    source_text: str
    source_language: str
    language_results: dict[str, LanguageTranslationResult] = field(default_factory=dict)
    total_execution_time: float = 0.0


class TranslationWorkflow:
    """
    Parallel translation system with voting for quality.
    
    Combines two parallelization patterns:
    1. Sectioning: Different languages are translated in parallel
    2. Voting: Each language gets 3 translations from different personas,
       then a vote determines the best one
    
    Example usage:
        workflow = TranslationWorkflow()
        result = await workflow.translate(
            "Hello, how are you?",
            source_language="English",
            target_languages=["Spanish", "French", "German"]
        )
    """
    
    # Different translator personas for voting
    PERSONAS = [
        TranslatorPersona(
            name="formal",
            style="Formal/Professional",
            system_prompt="""You are a professional translator specializing in formal, 
business-appropriate translations. Use proper grammar, formal vocabulary, 
and maintain a professional tone. Avoid colloquialisms and slang.""",
            temperature=0.3
        ),
        TranslatorPersona(
            name="casual",
            style="Casual/Conversational", 
            system_prompt="""You are a translator who captures the natural, everyday 
way native speakers talk. Use conversational language, common expressions, 
and make it sound natural - like a native speaker having a casual conversation.""",
            temperature=0.7
        ),
        TranslatorPersona(
            name="accurate",
            style="Literal/Accurate",
            system_prompt="""You are a translator focused on accuracy and precision.
Translate as closely to the original meaning as possible while maintaining
grammatical correctness. Preserve the structure and intent of the original.""",
            temperature=0.3
        )
    ]
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        max_concurrent: int = 10
    ):
        """
        Initialize the translation workflow.
        
        Args:
            model: Claude model to use
            max_tokens: Max tokens per translation
            max_concurrent: Max parallel API calls
        """
        self.model = model
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.async_client = anthropic.AsyncAnthropic()
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _translate_single(
        self,
        text: str,
        source_language: str,
        target_language: str,
        persona: TranslatorPersona
    ) -> Translation:
        """
        Perform a single translation with one persona.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            persona: Translator persona to use
            
        Returns:
            Translation result
        """
        async with self._semaphore:
            prompt = f"""Translate the following text from {source_language} to {target_language}.

Text to translate:
"{text}"

Provide ONLY the translated text, nothing else. No explanations, no notes, just the translation."""

            try:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=persona.temperature,
                    system=persona.system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                translated_text = response.content[0].text.strip()
                # Remove quotes if the model added them
                if translated_text.startswith('"') and translated_text.endswith('"'):
                    translated_text = translated_text[1:-1]
                
                return Translation(
                    persona_name=persona.name,
                    text=translated_text,
                    style=persona.style
                )
                
            except Exception as e:
                return Translation(
                    persona_name=persona.name,
                    text="",
                    style=persona.style,
                    success=False,
                    error=str(e)
                )
    
    async def _vote_best_translation(
        self,
        translations: list[Translation],
        target_language: str
    ) -> tuple[str, float]:
        """
        Use LLM to vote for the best translation.
        
        When translators disagree, we ask a judge to pick the best one.
        
        Args:
            translations: List of translations to compare
            target_language: The target language
            
        Returns:
            (best_translation, confidence_score)
        """
        successful = [t for t in translations if t.success]
        
        if not successful:
            return "", 0.0
        
        if len(successful) == 1:
            return successful[0].text, 1.0
        
        # Check if all translations are identical (high confidence)
        unique_translations = set(t.text.lower().strip() for t in successful)
        if len(unique_translations) == 1:
            return successful[0].text, 1.0
        
        # Ask a judge to pick the best
        options = "\n".join([
            f"Option {i+1} ({t.style}): {t.text}"
            for i, t in enumerate(successful)
        ])
        
        prompt = f"""You are a {target_language} language expert. Compare these translations and select the best one.

{options}

Consider:
1. Accuracy of meaning
2. Natural flow in {target_language}
3. Appropriate tone and register

Respond with ONLY the number of the best option (1, 2, or 3) and a confidence score.
Format: [number] [confidence: high/medium/low]

Example: 2 high"""

        try:
            async with self._semaphore:
                response = await self.async_client.messages.create(
                    model=self.model,
                    max_tokens=50,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            result = response.content[0].text.strip()
            
            # Parse the response
            parts = result.split()
            choice = int(parts[0]) - 1
            confidence_word = parts[-1].lower() if len(parts) > 1 else "medium"
            
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
            confidence = confidence_map.get(confidence_word, 0.7)
            
            if 0 <= choice < len(successful):
                return successful[choice].text, confidence
            else:
                return successful[0].text, 0.5
                
        except Exception:
            # Fallback: pick the formal translation
            for t in successful:
                if t.persona_name == "formal":
                    return t.text, 0.6
            return successful[0].text, 0.5
    
    async def _translate_to_language(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> LanguageTranslationResult:
        """
        Translate to a single language using all personas.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            
        Returns:
            LanguageTranslationResult with all translations and voting result
        """
        import time
        start = time.time()
        
        # Get translations from all personas in parallel
        translations = await asyncio.gather(*[
            self._translate_single(text, source_language, target_language, persona)
            for persona in self.PERSONAS
        ])
        
        # Vote for the best translation
        best_text, confidence = await self._vote_best_translation(
            list(translations), 
            target_language
        )
        
        return LanguageTranslationResult(
            language=target_language,
            translations=list(translations),
            best_translation=best_text,
            confidence=confidence,
            execution_time=time.time() - start
        )
    
    async def translate(
        self,
        text: str,
        source_language: str = "English",
        target_languages: list[str] = None
    ) -> TranslationReport:
        """
        Translate text to multiple languages in parallel.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_languages: List of target languages
            
        Returns:
            TranslationReport with all results
        """
        import time
        
        if target_languages is None:
            target_languages = ["Spanish", "French", "German"]
        
        start = time.time()
        
        # Translate to all languages in parallel (sectioning)
        results = await asyncio.gather(*[
            self._translate_to_language(text, source_language, lang)
            for lang in target_languages
        ])
        
        # Build the report
        language_results = {r.language: r for r in results}
        
        return TranslationReport(
            source_text=text,
            source_language=source_language,
            language_results=language_results,
            total_execution_time=time.time() - start
        )


def format_translation_report(report: TranslationReport) -> str:
    """Format the translation report as a readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("                    TRANSLATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Source Language: {report.source_language}")
    lines.append(f"Source Text: \"{report.source_text}\"")
    lines.append(f"Total Time: {report.total_execution_time:.2f}s")
    lines.append("")
    
    for lang, result in report.language_results.items():
        lines.append("-" * 70)
        lines.append(f"TARGET: {lang.upper()}")
        lines.append("-" * 70)
        
        # Show individual translations
        lines.append("\nIndividual Translations:")
        for t in result.translations:
            status = "✓" if t.success else "✗"
            if t.success:
                lines.append(f"  {status} {t.style:25} | {t.text}")
            else:
                lines.append(f"  {status} {t.style:25} | ERROR: {t.error}")
        
        # Show voted best
        lines.append(f"\n  → Best Translation: \"{result.best_translation}\"")
        lines.append(f"  → Confidence: {result.confidence:.0%}")
        lines.append(f"  → Time: {result.execution_time:.2f}s")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Main Example
# =============================================================================

async def main():
    """Demonstrate the parallel translation system."""
    
    workflow = TranslationWorkflow()
    
    # Test texts with varying complexity
    test_texts = [
        {
            "text": "The early bird catches the worm, but the second mouse gets the cheese.",
            "note": "Idiom with cultural reference"
        },
        {
            "text": "Please submit your quarterly report by end of business Friday.",
            "note": "Business communication"
        },
        {
            "text": "I can't believe you did that! That's absolutely hilarious!",
            "note": "Emotional/casual expression"
        }
    ]
    
    target_languages = ["Spanish", "French", "German", "Japanese"]
    
    for test in test_texts:
        print(f"\n{'='*70}")
        print(f"Translating: {test['note']}")
        print(f"Text: \"{test['text']}\"")
        print(f"Target Languages: {', '.join(target_languages)}")
        print("="*70)
        print("\nProcessing... (using 3 translators per language in parallel)\n")
        
        report = await workflow.translate(
            text=test["text"],
            source_language="English",
            target_languages=target_languages
        )
        
        formatted = format_translation_report(report)
        print(formatted)
        
        # Summary statistics
        total_translations = len(target_languages) * len(workflow.PERSONAS)
        successful = sum(
            1 for r in report.language_results.values()
            for t in r.translations if t.success
        )
        avg_confidence = sum(
            r.confidence for r in report.language_results.values()
        ) / len(report.language_results)
        
        print(f"\nSummary:")
        print(f"  - Total API calls: {total_translations}")
        print(f"  - Successful: {successful}/{total_translations}")
        print(f"  - Average confidence: {avg_confidence:.0%}")
        print(f"  - Total time: {report.total_execution_time:.2f}s")
        print(f"  - Sequential would be: ~{report.total_execution_time * 3:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
