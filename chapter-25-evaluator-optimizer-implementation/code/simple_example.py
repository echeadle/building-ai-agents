"""
Simple Evaluator-Optimizer Example

A minimal, standalone implementation of the evaluator-optimizer pattern.
This file is designed to be easy to understand and modify.

Chapter 25: Evaluator-Optimizer - Implementation

This example creates a simple slogan optimizer that iteratively
improves a marketing slogan based on feedback.
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize client
client = anthropic.Anthropic()

# Configuration
MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 4
QUALITY_THRESHOLD = 0.8


def generate_slogan(product_description: str) -> str:
    """Generate an initial marketing slogan."""
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        temperature=0.8,  # Creative for slogans
        messages=[{
            "role": "user",
            "content": f"""Create a short, catchy marketing slogan (under 10 words) 
for this product:

{product_description}

Output only the slogan, nothing else."""
        }]
    )
    
    return response.content[0].text.strip()


def revise_slogan(current_slogan: str, feedback: str, product_description: str) -> str:
    """Revise a slogan based on feedback."""
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": f"""Product: {product_description}

Current slogan: {current_slogan}

Feedback: {feedback}

Create an improved slogan that addresses this feedback.
Keep it under 10 words. Output only the new slogan."""
        }]
    )
    
    return response.content[0].text.strip()


def evaluate_slogan(slogan: str, product_description: str) -> dict:
    """
    Evaluate a slogan and return score and feedback.
    
    Returns:
        dict with 'score' (0-1), 'passed' (bool), and 'feedback' (str)
    """
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0.3,  # Consistent evaluation
        system=f"""You evaluate marketing slogans. Score them 0.0 to 1.0 based on:
- Memorability: Is it catchy and easy to remember?
- Clarity: Does it communicate the product's value?
- Brevity: Is it concise (under 10 words)?
- Emotion: Does it evoke a feeling or desire?

A score of {QUALITY_THRESHOLD} or higher passes.

ALWAYS respond with this exact JSON format:
{{"score": 0.75, "passed": false, "feedback": "specific improvement suggestion"}}

No other text, just the JSON.""",
        messages=[{
            "role": "user",
            "content": f"""Product: {product_description}

Slogan: "{slogan}"

Evaluate this slogan."""
        }]
    )
    
    # Parse response
    response_text = response.content[0].text.strip()
    
    try:
        # Handle markdown code blocks
        if "```" in response_text:
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        return {
            "score": float(result.get("score", 0.5)),
            "passed": bool(result.get("passed", False)),
            "feedback": str(result.get("feedback", "No feedback provided"))
        }
    except (json.JSONDecodeError, KeyError):
        print(f"Warning: Could not parse evaluation: {response_text[:100]}")
        return {
            "score": 0.5,
            "passed": False,
            "feedback": "Evaluation failed - trying again"
        }


def optimize_slogan(product_description: str) -> dict:
    """
    Run the full evaluator-optimizer loop.
    
    Returns:
        dict with 'slogan', 'score', 'iterations', and 'history'
    """
    
    print(f"\n{'='*50}")
    print("SLOGAN OPTIMIZER")
    print(f"{'='*50}")
    print(f"Product: {product_description}")
    
    history = []
    current_slogan = ""
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Generate or revise
        if iteration == 1:
            print("Generating initial slogan...")
            current_slogan = generate_slogan(product_description)
        else:
            print("Revising based on feedback...")
            current_slogan = revise_slogan(
                current_slogan,
                history[-1]["feedback"],
                product_description
            )
        
        print(f'Slogan: "{current_slogan}"')
        
        # Evaluate
        print("Evaluating...")
        evaluation = evaluate_slogan(current_slogan, product_description)
        
        print(f"Score: {evaluation['score']:.2f}")
        
        # Record history
        history.append({
            "iteration": iteration,
            "slogan": current_slogan,
            "score": evaluation["score"],
            "feedback": evaluation["feedback"]
        })
        
        # Check if passed
        if evaluation["passed"]:
            print(f"\n✓ Slogan passed! (score >= {QUALITY_THRESHOLD})")
            break
        else:
            print(f"Feedback: {evaluation['feedback']}")
    
    # Return results
    best = max(history, key=lambda x: x["score"])
    
    return {
        "slogan": best["slogan"],
        "score": best["score"],
        "iterations": len(history),
        "history": history
    }


def main():
    """Run the slogan optimizer demo."""
    
    # Test products
    products = [
        "A smartphone app that uses AI to help people learn new languages through conversations with virtual native speakers",
        "Eco-friendly reusable coffee cups that keep drinks hot for 8 hours and collapse flat for easy storage",
        "A subscription service that delivers personalized vitamin packs based on your health goals and blood test results",
    ]
    
    # Pick one to optimize
    product = products[0]
    
    result = optimize_slogan(product)
    
    print(f"\n{'='*50}")
    print("FINAL RESULT")
    print(f"{'='*50}")
    print(f'Best Slogan: "{result["slogan"]}"')
    print(f"Final Score: {result['score']:.2f}")
    print(f"Iterations Used: {result['iterations']}")
    
    print("\n--- Score History ---")
    for record in result["history"]:
        print(f"  {record['iteration']}: {record['score']:.2f} - \"{record['slogan']}\"")


if __name__ == "__main__":
    main()
