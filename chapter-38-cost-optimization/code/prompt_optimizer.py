"""
System prompt optimization techniques.

Chapter 38: Cost Optimization
"""

import re


def analyze_system_prompt(prompt: str) -> dict:
    lines = prompt.strip().split('\n')
    words = prompt.split()
    estimated_tokens = len(words) + len(lines)
    
    analysis = {
        "original_length": len(prompt),
        "word_count": len(words),
        "estimated_tokens": estimated_tokens,
        "issues": [],
        "suggestions": [],
    }
    
    if "  " in prompt or "\n\n\n" in prompt:
        analysis["issues"].append("Contains excessive whitespace")
    
    if estimated_tokens > 1000:
        analysis["issues"].append(f"Prompt is very long ({estimated_tokens} tokens)")
    
    verbose_phrases = ["you should always", "please make sure to", "it is important that you"]
    for phrase in verbose_phrases:
        if phrase in prompt.lower():
            analysis["issues"].append(f"Verbose phrasing: '{phrase}'")
            break
    
    return analysis


def optimize_system_prompt(prompt: str) -> str:
    optimized = re.sub(r' +', ' ', prompt)
    optimized = re.sub(r'\n{3,}', '\n\n', optimized)
    
    replacements = [
        ("you should always", "Always"),
        ("please make sure to", ""),
        ("it is important that you", ""),
        ("in order to", "to"),
    ]
    
    for verbose, concise in replacements:
        optimized = re.sub(re.escape(verbose), concise, optimized, flags=re.IGNORECASE)
    
    return optimized.strip()


if __name__ == "__main__":
    verbose = """
    You are a helpful assistant. It is important that you always help users.
    You should always be polite. Please make sure to think carefully.
    """
    
    print("Prompt Optimization Demo")
    print("=" * 40)
    print(f"Original: {analyze_system_prompt(verbose)['estimated_tokens']} tokens")
    
    optimized = optimize_system_prompt(verbose)
    print(f"Optimized: {analyze_system_prompt(optimized)['estimated_tokens']} tokens")
