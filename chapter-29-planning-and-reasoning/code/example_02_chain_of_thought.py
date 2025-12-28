"""
Chain-of-thought reasoning patterns.

This example demonstrates how to use chain-of-thought (CoT) prompting to 
encourage step-by-step reasoning in LLM responses. CoT helps with tasks
that require logical reasoning, calculations, or weighing multiple factors.

Chapter 29: Planning and Reasoning
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


def ask_with_chain_of_thought(question: str) -> dict:
    """
    Ask a question using chain-of-thought prompting.
    
    Args:
        question: The question to answer
        
    Returns:
        Dictionary with reasoning and final answer
    """
    cot_prompt = f"""Question: {question}

Please think through this step by step:
1. First, identify what information is needed
2. Then, reason through each relevant consideration  
3. Finally, provide your conclusion

Format your response as:
REASONING:
[Your step-by-step thinking]

ANSWER:
[Your final answer]"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": cot_prompt}]
    )
    
    text = response.content[0].text
    
    # Parse the response
    if "REASONING:" in text and "ANSWER:" in text:
        parts = text.split("ANSWER:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        answer = parts[1].strip()
    else:
        reasoning = text
        answer = text
    
    return {
        "reasoning": reasoning,
        "answer": answer,
        "full_response": text
    }


def ask_direct(question: str) -> str:
    """
    Ask a question directly without chain-of-thought prompting.
    
    Args:
        question: The question to answer
        
    Returns:
        The response text
    """
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text


def compare_with_and_without_cot(question: str) -> None:
    """
    Compare responses with and without chain-of-thought.
    
    Args:
        question: The question to answer both ways
    """
    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    
    # Without CoT
    print("\n--- WITHOUT CHAIN-OF-THOUGHT ---")
    direct_response = ask_direct(question)
    print(direct_response)
    
    # With CoT
    print("\n--- WITH CHAIN-OF-THOUGHT ---")
    cot_result = ask_with_chain_of_thought(question)
    print(f"Reasoning:\n{cot_result['reasoning']}")
    print(f"\nFinal Answer:\n{cot_result['answer']}")


def ask_with_structured_cot(question: str, considerations: list[str]) -> dict:
    """
    Ask a question with structured chain-of-thought, specifying what to consider.
    
    Args:
        question: The question to answer
        considerations: List of aspects to think about
        
    Returns:
        Dictionary with structured reasoning and answer
    """
    considerations_text = "\n".join(f"- {c}" for c in considerations)
    
    structured_prompt = f"""Question: {question}

Think through this systematically by considering each of these aspects:
{considerations_text}

For each consideration, provide your analysis, then synthesize a final answer.

Format your response as:
ANALYSIS:
[Your analysis of each consideration]

SYNTHESIS:
[How these factors combine]

FINAL ANSWER:
[Your conclusion]"""

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": structured_prompt}]
    )
    
    text = response.content[0].text
    
    # Parse sections
    sections = {}
    current_section = "full"
    sections[current_section] = []
    
    for line in text.split('\n'):
        if line.strip() in ["ANALYSIS:", "SYNTHESIS:", "FINAL ANSWER:"]:
            current_section = line.strip().rstrip(':').lower().replace(' ', '_')
            sections[current_section] = []
        else:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
    
    return {
        "analysis": '\n'.join(sections.get('analysis', [])).strip(),
        "synthesis": '\n'.join(sections.get('synthesis', [])).strip(),
        "final_answer": '\n'.join(sections.get('final_answer', [])).strip(),
        "full_response": text
    }


if __name__ == "__main__":
    # Example 1: A classic puzzle where CoT helps
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Logic Puzzle")
    print("=" * 60)
    
    puzzle = """
    A farmer has 17 sheep. All but 9 run away. How many sheep does 
    the farmer have left?
    """
    
    compare_with_and_without_cot(puzzle)
    
    # Example 2: A more complex reasoning question
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Technical Decision")
    print("=" * 60)
    
    complex_question = """
    Should a small startup use microservices architecture from day one?
    Consider development speed, operational complexity, and future scaling.
    """
    
    compare_with_and_without_cot(complex_question)
    
    # Example 3: Structured CoT with specific considerations
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Structured Chain-of-Thought")
    print("=" * 60)
    
    decision_question = "Should I learn TypeScript or stick with JavaScript?"
    considerations = [
        "Type safety benefits",
        "Learning curve",
        "Job market demand",
        "Project size where it matters",
        "Tooling and IDE support"
    ]
    
    result = ask_with_structured_cot(decision_question, considerations)
    
    print(f"\nQuestion: {decision_question}")
    print(f"\nConsiderations analyzed: {considerations}")
    print(f"\nAnalysis:\n{result['analysis'][:500]}...")
    print(f"\nSynthesis:\n{result['synthesis']}")
    print(f"\nFinal Answer:\n{result['final_answer']}")
