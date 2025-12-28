"""
Quality Gate Examples for Prompt Chains

Chapter 16: Prompt Chaining - Concept and Design

This file provides practical examples of quality gate validators that can
be used between chain steps. These validators check structure, semantics,
and completeness of chain step outputs.
"""

import json
import re
from typing import Any, Callable, Optional
from dataclasses import dataclass


# =============================================================================
# STRUCTURAL VALIDATORS
# =============================================================================

def validate_json_structure(output: str, required_fields: list[str]) -> tuple[bool, str]:
    """
    Validate that output is valid JSON with required fields.
    
    Args:
        output: The string output to validate
        required_fields: List of field names that must be present
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Missing required fields: {missing}"
    
    return True, ""


def validate_list_length(
    output: Any, 
    min_items: int = 1, 
    max_items: Optional[int] = None
) -> tuple[bool, str]:
    """
    Validate that output is a list with appropriate length.
    
    Args:
        output: The output to validate (should be a list)
        min_items: Minimum number of items required
        max_items: Maximum number of items allowed (None for no limit)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(output, list):
        return False, f"Expected list, got {type(output).__name__}"
    
    if len(output) < min_items:
        return False, f"Too few items: got {len(output)}, need at least {min_items}"
    
    if max_items is not None and len(output) > max_items:
        return False, f"Too many items: got {len(output)}, max is {max_items}"
    
    return True, ""


def validate_word_count(
    text: str, 
    min_words: int = 0, 
    max_words: Optional[int] = None
) -> tuple[bool, str]:
    """
    Validate that text has appropriate word count.
    
    Args:
        text: The text to validate
        min_words: Minimum word count
        max_words: Maximum word count (None for no limit)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    word_count = len(text.split())
    
    if word_count < min_words:
        return False, f"Too short: {word_count} words, need at least {min_words}"
    
    if max_words is not None and word_count > max_words:
        return False, f"Too long: {word_count} words, max is {max_words}"
    
    return True, ""


# =============================================================================
# COMPLETENESS VALIDATORS
# =============================================================================

def validate_all_items_addressed(
    input_items: list[str], 
    output_text: str,
    case_sensitive: bool = False
) -> tuple[bool, str]:
    """
    Validate that all input items appear in the output.
    
    Useful for ensuring nothing was dropped during processing.
    
    Args:
        input_items: List of items that should appear in output
        output_text: The text to check
        case_sensitive: Whether matching should be case-sensitive
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    search_text = output_text if case_sensitive else output_text.lower()
    
    missing = []
    for item in input_items:
        search_item = item if case_sensitive else item.lower()
        if search_item not in search_text:
            missing.append(item)
    
    if missing:
        return False, f"Missing items in output: {missing}"
    
    return True, ""


def validate_sections_present(
    markdown_text: str, 
    expected_sections: list[str]
) -> tuple[bool, str]:
    """
    Validate that markdown text contains expected section headers.
    
    Args:
        markdown_text: Markdown text to check
        expected_sections: List of section titles (without # prefix)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Find all headers in the markdown
    header_pattern = r'^#{1,6}\s+(.+)$'
    found_headers = re.findall(header_pattern, markdown_text, re.MULTILINE)
    found_headers_lower = [h.lower().strip() for h in found_headers]
    
    missing = []
    for section in expected_sections:
        if section.lower() not in found_headers_lower:
            missing.append(section)
    
    if missing:
        return False, f"Missing sections: {missing}"
    
    return True, ""


def validate_no_truncation(text: str) -> tuple[bool, str]:
    """
    Check for signs that output was truncated mid-sentence.
    
    Args:
        text: Text to check
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    text = text.strip()
    
    # Check for incomplete sentences (ending with common truncation patterns)
    truncation_patterns = [
        r'\.\.\.$',           # Ends with ellipsis
        r'[,;:]\s*$',         # Ends with comma, semicolon, colon
        r'\b(and|or|the|a|an|to|of|in)\s*$',  # Ends with common words
        r'[a-z]\s*$',         # Ends with lowercase (usually mid-sentence)
    ]
    
    for pattern in truncation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "Output appears to be truncated"
    
    return True, ""


# =============================================================================
# SEMANTIC VALIDATORS
# =============================================================================

def validate_language(text: str, expected_language: str) -> tuple[bool, str]:
    """
    Basic check that text appears to be in the expected language.
    
    Note: This is a simple heuristic. For production use, consider
    using a proper language detection library.
    
    Args:
        text: Text to check
        expected_language: Expected language code ('en', 'es', 'fr', etc.)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Simple heuristic based on common words
    language_markers = {
        'en': ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has'],
        'es': ['el', 'la', 'los', 'las', 'es', 'son', 'está', 'están'],
        'fr': ['le', 'la', 'les', 'est', 'sont', 'dans', 'avec', 'pour'],
        'de': ['der', 'die', 'das', 'und', 'ist', 'sind', 'mit', 'für'],
    }
    
    if expected_language not in language_markers:
        return True, ""  # Can't validate, assume OK
    
    text_lower = text.lower()
    markers = language_markers[expected_language]
    
    # Count how many markers appear
    found = sum(1 for m in markers if f' {m} ' in f' {text_lower} ')
    
    # Expect at least half the markers in a reasonably long text
    if len(text.split()) > 50 and found < len(markers) // 2:
        return False, f"Text does not appear to be in {expected_language}"
    
    return True, ""


def validate_sentiment_appropriate(text: str, expected_sentiment: str) -> tuple[bool, str]:
    """
    Basic check that text sentiment matches expectations.
    
    Note: This is a simple heuristic. For production use, consider
    using a sentiment analysis library or LLM-based validation.
    
    Args:
        text: Text to check
        expected_sentiment: 'positive', 'negative', or 'neutral'
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'best', 
                      'love', 'fantastic', 'perfect', 'outstanding']
    negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 
                      'horrible', 'poor', 'disappointing', 'fail']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if expected_sentiment == 'positive' and neg_count > pos_count:
        return False, "Text appears more negative than expected"
    elif expected_sentiment == 'negative' and pos_count > neg_count:
        return False, "Text appears more positive than expected"
    
    return True, ""


# =============================================================================
# COMPOSITE VALIDATOR
# =============================================================================

@dataclass
class ValidationResult:
    """Result of running a validation gate."""
    is_valid: bool
    errors: list[str]
    warnings: list[str]


class QualityGate:
    """
    A quality gate that runs multiple validators on chain step output.
    
    Combines structural, completeness, and semantic validation into
    a single reusable component.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.validators: list[tuple[Callable, dict, bool]] = []  # (func, kwargs, required)
    
    def add_validator(
        self, 
        validator: Callable[..., tuple[bool, str]], 
        required: bool = True,
        **kwargs
    ) -> "QualityGate":
        """
        Add a validator to this gate.
        
        Args:
            validator: A validator function that returns (is_valid, error_message)
            required: If True, failure fails the gate. If False, failure is a warning.
            **kwargs: Additional arguments to pass to the validator
        
        Returns:
            self (for chaining)
        """
        self.validators.append((validator, kwargs, required))
        return self
    
    def validate(self, output: Any) -> ValidationResult:
        """
        Run all validators on the output.
        
        Args:
            output: The chain step output to validate
        
        Returns:
            ValidationResult with overall status and any errors/warnings
        """
        errors = []
        warnings = []
        
        for validator, kwargs, required in self.validators:
            # Determine what to pass to the validator
            # If it expects 'output', pass it; otherwise pass output as first arg
            try:
                is_valid, message = validator(output, **kwargs)
            except TypeError:
                # Validator might expect different argument name
                is_valid, message = validator(output)
            
            if not is_valid:
                if required:
                    errors.append(message)
                else:
                    warnings.append(message)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# =============================================================================
# EXAMPLE QUALITY GATES
# =============================================================================

def create_glossary_gate() -> QualityGate:
    """Create a quality gate for validating translation glossary output."""
    gate = QualityGate("Glossary Validation")
    
    # Structural validation
    gate.add_validator(
        validate_json_structure,
        required_fields=["terms"]
    )
    
    # Completeness validation (assuming we know input terms)
    # This would need to be configured with actual input terms
    
    return gate


def create_blog_draft_gate(min_words: int = 500) -> QualityGate:
    """Create a quality gate for validating blog post draft output."""
    gate = QualityGate("Blog Draft Validation")
    
    # Check word count
    gate.add_validator(
        validate_word_count,
        min_words=min_words,
        max_words=3000
    )
    
    # Check for truncation
    gate.add_validator(
        validate_no_truncation,
        required=True
    )
    
    return gate


def create_translation_gate(target_language: str, source_terms: list[str]) -> QualityGate:
    """Create a quality gate for validating translation output."""
    gate = QualityGate("Translation Validation")
    
    # Check language
    gate.add_validator(
        validate_language,
        expected_language=target_language
    )
    
    # Check completeness (all terms should be addressed)
    # Note: For translation, we'd check translated terms, not source terms
    # This is simplified for illustration
    gate.add_validator(
        validate_no_truncation,
        required=True
    )
    
    return gate


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QUALITY GATE EXAMPLES - CHAPTER 16")
    print("=" * 70)
    
    # Example 1: Structural validation
    print("\n1. Structural Validation")
    print("-" * 40)
    
    good_json = '{"terms": [{"original": "API", "translation": "API"}]}'
    bad_json = '{"items": []}'
    
    result1 = validate_json_structure(good_json, ["terms"])
    print(f"Good JSON: valid={result1[0]}, error='{result1[1]}'")
    
    result2 = validate_json_structure(bad_json, ["terms"])
    print(f"Bad JSON: valid={result2[0]}, error='{result2[1]}'")
    
    # Example 2: Completeness validation
    print("\n2. Completeness Validation")
    print("-" * 40)
    
    input_features = ["waterproof", "lightweight", "durable"]
    complete_output = "This waterproof jacket is lightweight and extremely durable."
    incomplete_output = "This jacket is lightweight and stylish."
    
    result3 = validate_all_items_addressed(input_features, complete_output)
    print(f"Complete: valid={result3[0]}, error='{result3[1]}'")
    
    result4 = validate_all_items_addressed(input_features, incomplete_output)
    print(f"Incomplete: valid={result4[0]}, error='{result4[1]}'")
    
    # Example 3: Composite quality gate
    print("\n3. Composite Quality Gate")
    print("-" * 40)
    
    gate = create_blog_draft_gate(min_words=10)
    
    good_draft = "This is a complete blog post draft. " * 20  # ~100 words
    truncated_draft = "This is a blog post that ends with and"
    
    result5 = gate.validate(good_draft)
    print(f"Good draft: valid={result5.is_valid}, errors={result5.errors}")
    
    result6 = gate.validate(truncated_draft)
    print(f"Truncated: valid={result6.is_valid}, errors={result6.errors}")
    
    # Example 4: Word count validation
    print("\n4. Word Count Validation")
    print("-" * 40)
    
    short_text = "Too short."
    good_text = " ".join(["word"] * 150)
    long_text = " ".join(["word"] * 500)
    
    result7 = validate_word_count(short_text, min_words=100, max_words=200)
    print(f"Short text: valid={result7[0]}, error='{result7[1]}'")
    
    result8 = validate_word_count(good_text, min_words=100, max_words=200)
    print(f"Good text: valid={result8[0]}, error='{result8[1]}'")
    
    result9 = validate_word_count(long_text, min_words=100, max_words=200)
    print(f"Long text: valid={result9[0]}, error='{result9[1]}'")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
