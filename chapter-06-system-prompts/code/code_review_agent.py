"""
A code review assistant with a comprehensive system prompt.

Demonstrates all four components: identity, capabilities, behavior, and boundaries.

Chapter 6: System Prompts and Persona Design
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

# Comprehensive system prompt for a code review assistant
SYSTEM_PROMPT = """You are a Python code review assistant. Your purpose is to help 
developers improve their code quality by providing constructive feedback.

## Your Expertise
You specialize in:
- Python best practices and PEP 8 style guidelines
- Common bugs and error patterns
- Performance optimization opportunities
- Code readability and maintainability
- Security considerations for Python code

## How You Review Code
When reviewing code:
1. Start with what's done well—acknowledge good practices
2. Identify issues in order of severity (bugs first, then style)
3. Explain WHY something is a problem, not just what
4. Provide specific, actionable suggestions with code examples
5. Keep feedback constructive and educational

## Communication Style
- Be encouraging but honest
- Use clear, technical language appropriate for developers
- Format code suggestions as proper Python code blocks
- Be concise—developers are busy

## Boundaries
- Only review Python code (politely decline other languages)
- Don't write entire applications—focus on reviewing what's provided
- If code is too complex for a thorough review, say so
- Don't make assumptions about business requirements—ask if unclear"""


def review_code(code: str) -> str:
    """
    Send code to be reviewed by the assistant.
    
    Args:
        code: The Python code to review
        
    Returns:
        The code review feedback
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Please review this Python code:\n\n```python\n{code}\n```"
            }
        ]
    )
    
    return response.content[0].text


def main():
    """Demonstrate the code review assistant."""
    
    # Example 1: Code with style and performance issues
    code_sample_1 = '''
def calculate_average(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    average = total / len(numbers)
    return average
'''
    
    print("=" * 60)
    print("CODE REVIEW EXAMPLE 1: Basic function with issues")
    print("=" * 60)
    print("\nCode submitted:")
    print(code_sample_1)
    print("\nReview:")
    print(review_code(code_sample_1))
    
    print("\n" + "=" * 60)
    
    # Example 2: Code with a potential bug
    code_sample_2 = '''
def get_user_data(user_id):
    users = {"1": "Alice", "2": "Bob", "3": "Charlie"}
    return users[user_id]
    
def process_users(user_ids):
    results = []
    for id in user_ids:
        results.append(get_user_data(id))
    return results
'''
    
    print("CODE REVIEW EXAMPLE 2: Potential bug")
    print("=" * 60)
    print("\nCode submitted:")
    print(code_sample_2)
    print("\nReview:")
    print(review_code(code_sample_2))
    
    print("\n" + "=" * 60)
    
    # Example 3: Testing boundary - non-Python code
    non_python_code = '''
function calculateSum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}
'''
    
    print("CODE REVIEW EXAMPLE 3: Testing boundary (JavaScript)")
    print("=" * 60)
    print("\nCode submitted:")
    print(non_python_code)
    print("\nReview:")
    print(review_code(non_python_code))


if __name__ == "__main__":
    main()
