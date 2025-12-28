"""
Glossary Scavenger Hunt Exercise

Test your understanding of key agent development concepts by finding
connections between terms in the glossary.

Appendix F: Glossary
"""

from glossary_lookup import GlossaryLookup
from typing import List, Dict


class GlossaryScavengerHunt:
    """
    An interactive exercise to help learn agent terminology.
    """
    
    def __init__(self):
        self.glossary = GlossaryLookup()
        self.score = 0
        self.total_questions = 0
    
    def question_1(self) -> bool:
        """
        Question 1: Find the three message roles.
        
        What are the three roles used in messages to Claude?
        """
        print("\n" + "="*60)
        print("QUESTION 1: The Three Roles")
        print("="*60)
        print("\nIn the Anthropic API, messages have three possible roles.")
        print("Can you find them using the glossary?")
        print("\nHint: Search for 'role' in the glossary")
        
        correct_answers = {"system", "user", "assistant"}
        
        print("\nYour answer (comma-separated):")
        answer = input("> ").strip().lower()
        answers = {a.strip() for a in answer.split(",")}
        
        if answers == correct_answers:
            print("\n✓ Correct! The three roles are: system, user, and assistant")
            self.glossary.print_term("role")
            return True
        else:
            print(f"\n✗ Not quite. You said: {answers}")
            print(f"The correct answer is: {correct_answers}")
            self.glossary.print_term("role")
            return False
    
    def question_2(self) -> bool:
        """
        Question 2: Understanding the agentic loop.
        
        What are the 5 steps in the agentic loop?
        """
        print("\n" + "="*60)
        print("QUESTION 2: The Agentic Loop")
        print("="*60)
        print("\nThe agentic loop is the core pattern of agent execution.")
        print("How many steps does it have?")
        print("\nHint: Look up 'agentic loop' in the glossary")
        
        print("\nYour answer:")
        answer = input("> ").strip()
        
        if answer == "5":
            print("\n✓ Correct! The agentic loop has 5 steps:")
            self.glossary.print_term("agentic_loop")
            return True
        else:
            print(f"\n✗ Not quite. The loop has 5 steps, not {answer}.")
            self.glossary.print_term("agentic_loop")
            return False
    
    def question_3(self) -> bool:
        """
        Question 3: Find related concepts.
        
        What concepts are related to "tool"?
        """
        print("\n" + "="*60)
        print("QUESTION 3: Connected Concepts")
        print("="*60)
        print("\nTools are central to agent functionality.")
        print("Using the glossary, find THREE terms related to 'tool'")
        
        related = self.glossary.get_related("tool")
        print(f"\nHint: There are {len(related)} related terms")
        
        print("\nYour answer (comma-separated):")
        answer = input("> ").strip().lower()
        answers = {a.strip().replace(" ", "_") for a in answer.split(",")}
        
        # Accept if they got at least 3 correct related terms
        correct_related = set(related)
        matches = answers.intersection(correct_related)
        
        if len(matches) >= 3:
            print(f"\n✓ Correct! You found: {matches}")
            print(f"\nAll related terms: {correct_related}")
            self.glossary.print_term("tool")
            return True
        else:
            print(f"\n✗ You found {len(matches)} correct terms: {matches}")
            print(f"The related terms are: {correct_related}")
            self.glossary.print_term("tool")
            return False
    
    def question_4(self) -> bool:
        """
        Question 4: Workflow patterns.
        
        Name the five core workflow patterns.
        """
        print("\n" + "="*60)
        print("QUESTION 4: The Five Workflows")
        print("="*60)
        print("\nThere are five core agentic workflow patterns covered in Part 3.")
        print("Can you name them all?")
        print("\nHint: Look up 'workflow' in the glossary")
        
        correct = {
            "chaining", "routing", "parallel", 
            "orchestrator-workers", "evaluator-optimizer"
        }
        
        print("\nYour answer (comma-separated):")
        answer = input("> ").strip().lower()
        
        # Normalize answers
        answers = set()
        for a in answer.split(","):
            normalized = a.strip().replace("prompt chaining", "chaining")
            normalized = normalized.replace("parallel execution", "parallel")
            answers.add(normalized)
        
        matches = answers.intersection(correct)
        
        if len(matches) >= 4:  # Accept if they got 4 out of 5
            print(f"\n✓ Great! You got {len(matches)} out of 5!")
            print(f"The five patterns are: {correct}")
            self.glossary.print_term("workflow")
            return True
        else:
            print(f"\n✗ You found {len(matches)}: {matches}")
            print(f"The five patterns are: {correct}")
            self.glossary.print_term("workflow")
            return False
    
    def question_5(self) -> bool:
        """
        Question 5: Security fundamentals.
        
        Why should you never hardcode API keys?
        """
        print("\n" + "="*60)
        print("QUESTION 5: Security Best Practices")
        print("="*60)
        print("\nAPI keys should never be hardcoded in your code.")
        print("What should you use instead?")
        print("\nHint: Look up 'api_key' in the glossary")
        
        print("\nYour answer:")
        answer = input("> ").strip().lower()
        
        correct_keywords = ["environment", "variable", ".env", "dotenv"]
        
        if any(keyword in answer for keyword in correct_keywords):
            print("\n✓ Correct! API keys should be loaded from environment variables.")
            print("This is typically done using python-dotenv and a .env file.")
            self.glossary.print_term("api_key")
            self.glossary.print_term("environment_variable")
            return True
        else:
            print("\n✗ Not quite. API keys should be stored in environment variables")
            print("and loaded using python-dotenv from a .env file.")
            self.glossary.print_term("api_key")
            return False
    
    def run(self):
        """Run the complete scavenger hunt."""
        print("\n" + "="*70)
        print(" "*15 + "GLOSSARY SCAVENGER HUNT")
        print("="*70)
        print("\nWelcome! This exercise will help you learn key agent development terms.")
        print("Use the glossary to find answers to the questions.")
        print("\nYou can look up terms at any time during the questions.")
        print("Just type 'help <term>' to see its definition.")
        print("\nLet's begin!\n")
        
        questions = [
            self.question_1,
            self.question_2,
            self.question_3,
            self.question_4,
            self.question_5,
        ]
        
        for i, question in enumerate(questions, 1):
            self.total_questions += 1
            
            if question():
                self.score += 1
            
            if i < len(questions):
                input("\nPress Enter to continue to the next question...")
        
        # Final score
        print("\n" + "="*70)
        print(" "*25 + "RESULTS")
        print("="*70)
        print(f"\nYou got {self.score} out of {self.total_questions} questions correct!")
        
        percentage = (self.score / self.total_questions) * 100
        
        if percentage == 100:
            print("\n🎉 Perfect score! You know your agent terminology!")
        elif percentage >= 80:
            print("\n✓ Great job! You have a solid understanding of the concepts.")
        elif percentage >= 60:
            print("\n👍 Good work! Review the glossary to strengthen your knowledge.")
        else:
            print("\n📚 Keep learning! The glossary is your friend.")
        
        print("\n" + "="*70)


def main():
    """Run the scavenger hunt."""
    hunt = GlossaryScavengerHunt()
    hunt.run()
    
    print("\n\nBonus Challenge:")
    print("-" * 70)
    print("Try these additional exercises:")
    print("1. Pick any term and trace its related terms 3 levels deep")
    print("2. Find all terms introduced in the same chapter")
    print("3. Create your own definition for a new concept you've learned")
    print("4. Find terms that have NO related terms listed")
    print("\nUse glossary_lookup.py to explore these challenges!")


if __name__ == "__main__":
    main()
