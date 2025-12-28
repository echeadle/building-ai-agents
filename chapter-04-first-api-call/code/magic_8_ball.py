"""
Magic 8-Ball: An interactive yes/no question answerer using Claude.

Chapter 4: Your First API Call to Claude - Exercise Solution
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()

# The system-like prompt that makes Claude act like a Magic 8-Ball
MAGIC_8_BALL_PROMPT = """You are a mystical Magic 8-Ball. When asked a yes/no question, you must respond 
with ONLY one of the following classic Magic 8-Ball responses (choose randomly and appropriately):

Positive answers:
- It is certain.
- It is decidedly so.
- Without a doubt.
- Yes definitely.
- You may rely on it.
- As I see it, yes.
- Most likely.
- Outlook good.
- Yes.
- Signs point to yes.

Neutral answers:
- Reply hazy, try again.
- Ask again later.
- Better not tell you now.
- Cannot predict now.
- Concentrate and ask again.

Negative answers:
- Don't count on it.
- My reply is no.
- My sources say no.
- Outlook not so good.
- Very doubtful.

Respond with ONLY the Magic 8-Ball phrase, nothing else. Be mystical and dramatic in your selection,
as if you are a magical oracle peering into the future.

The question is: """


def ask_magic_8_ball(question: str) -> tuple[str, dict]:
    """
    Ask the Magic 8-Ball a yes/no question.
    
    Args:
        question: A yes/no question to ask the mystical orb
        
    Returns:
        Tuple of (magic_response, usage_info)
    """
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,  # Magic 8-Ball responses are short
        messages=[
            {"role": "user", "content": MAGIC_8_BALL_PROMPT + question}
        ]
    )
    
    usage_info = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens
    }
    
    return message.content[0].text.strip(), usage_info


def main():
    """Run the interactive Magic 8-Ball."""
    print("=" * 50)
    print("🎱 Welcome to the Mystical Magic 8-Ball! 🎱")
    print("=" * 50)
    print()
    print("Ask me any yes/no question, and I shall reveal")
    print("the secrets of the universe...")
    print()
    print("(Type 'quit' to exit)")
    print("-" * 50)
    
    total_tokens_used = 0
    questions_asked = 0
    
    while True:
        print()
        question = input("🔮 Your question: ").strip()
        
        if question.lower() in ("quit", "exit", "q"):
            print()
            print("=" * 50)
            print("The Magic 8-Ball fades into the mist...")
            print(f"Questions answered: {questions_asked}")
            print(f"Total tokens used: {total_tokens_used}")
            print("Farewell, seeker of truth! 🌟")
            print("=" * 50)
            break
        
        if not question:
            print("The orb remains silent... You must ask a question!")
            continue
        
        try:
            response, usage = ask_magic_8_ball(question)
            questions_asked += 1
            total_tokens_used += usage["total_tokens"]
            
            print()
            print("   *the orb glows mysteriously*")
            print()
            print(f"   🎱 {response}")
            print()
            print(f"   (tokens used: {usage['total_tokens']})")
            
        except anthropic.APIError as e:
            print(f"The mystic connection was broken! Error: {e}")
            print("Please try again...")


if __name__ == "__main__":
    main()
