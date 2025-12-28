"""
Feedback Collector

Collects and stores human feedback on agent outputs for
analysis and potential improvement of agent behavior.

Chapter 31: Human-in-the-Loop
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from enum import Enum
import json

load_dotenv()


class FeedbackType(Enum):
    """Types of feedback humans can provide."""
    RATING = "rating"           # 1-5 star rating
    CORRECTION = "correction"   # Corrected output
    PREFERENCE = "preference"   # A vs B preference
    FLAG = "flag"               # Flag problematic output
    COMMENT = "comment"         # Free-form comment
    THUMBS = "thumbs"           # Simple thumbs up/down


@dataclass
class Feedback:
    """A single piece of human feedback."""
    feedback_id: str
    feedback_type: FeedbackType
    context: dict[str, Any]      # What the agent was doing
    agent_output: str            # What the agent produced
    human_input: Any             # The feedback itself
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "context": self.context,
            "agent_output": self.agent_output,
            "human_input": self.human_input,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Feedback':
        """Create Feedback from dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            feedback_type=FeedbackType(data["feedback_type"]),
            context=data["context"],
            agent_output=data["agent_output"],
            human_input=data["human_input"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class FeedbackCollector:
    """
    Collects and stores human feedback on agent outputs.
    
    This class provides various methods for collecting different
    types of feedback (ratings, corrections, preferences, flags).
    
    In production, this would persist to a database and potentially
    feed into fine-tuning or prompt improvement pipelines.
    
    Example:
        collector = FeedbackCollector(storage_path="feedback.json")
        
        # Collect a rating
        collector.collect_rating(
            context={"task": "summarization"},
            agent_output="Here is the summary..."
        )
        
        # Get summary statistics
        print(collector.get_summary())
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the feedback collector.
        
        Args:
            storage_path: Path to store feedback JSON (optional)
        """
        self.feedback_log: list[Feedback] = []
        self.storage_path = storage_path
        self._feedback_counter = 0
        
        # Load existing feedback if storage path exists
        if storage_path and os.path.exists(storage_path):
            self._load_from_file()
    
    def _generate_id(self) -> str:
        """Generate a unique feedback ID."""
        self._feedback_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"FB-{timestamp}-{self._feedback_counter:04d}"
    
    def _store_feedback(self, feedback: Feedback) -> None:
        """Store feedback in memory and optionally to file."""
        self.feedback_log.append(feedback)
        
        if self.storage_path:
            self._persist_to_file()
    
    def _persist_to_file(self) -> None:
        """Save all feedback to JSON file."""
        data = [fb.to_dict() for fb in self.feedback_log]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_file(self) -> None:
        """Load feedback from JSON file."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.feedback_log = [Feedback.from_dict(d) for d in data]
                self._feedback_counter = len(self.feedback_log)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load feedback file: {e}")
            self.feedback_log = []
    
    def collect_thumbs(
        self,
        context: dict[str, Any],
        agent_output: str,
        prompt: str = "Was this response helpful? (👍/👎): "
    ) -> Feedback:
        """
        Collect simple thumbs up/down feedback.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
            prompt: The prompt to show the user
        
        Returns:
            Feedback object with thumbs rating
        """
        print(f"\n🤖 Agent output:\n{self._truncate(agent_output, 300)}")
        
        while True:
            response = input(prompt).strip().lower()
            if response in ['👍', 'up', 'y', 'yes', '+', '1']:
                rating = "up"
                break
            elif response in ['👎', 'down', 'n', 'no', '-', '0']:
                rating = "down"
                break
            else:
                print("Please enter 👍 (up/yes) or 👎 (down/no)")
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.THUMBS,
            context=context,
            agent_output=agent_output,
            human_input=rating
        )
        
        self._store_feedback(feedback)
        print(f"{'👍' if rating == 'up' else '👎'} Thanks for your feedback!")
        return feedback
    
    def collect_rating(
        self,
        context: dict[str, Any],
        agent_output: str,
        prompt: str = "Rate this response (1-5 stars): ",
        allow_skip: bool = True
    ) -> Optional[Feedback]:
        """
        Collect a star rating for an agent output.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
            prompt: The prompt to show the user
            allow_skip: Whether to allow skipping (Enter to skip)
        
        Returns:
            Feedback object with the rating, or None if skipped
        """
        print(f"\n🤖 Agent output:\n{self._truncate(agent_output, 300)}")
        
        while True:
            response = input(prompt).strip()
            
            if allow_skip and response == "":
                print("⏭️  Skipped")
                return None
            
            try:
                rating = int(response)
                if 1 <= rating <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                if allow_skip:
                    print("Please enter 1-5 or press Enter to skip.")
                else:
                    print("Please enter a number between 1 and 5.")
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.RATING,
            context=context,
            agent_output=agent_output,
            human_input=rating
        )
        
        self._store_feedback(feedback)
        print(f"{'⭐' * rating} Thanks for rating!")
        return feedback
    
    def collect_correction(
        self,
        context: dict[str, Any],
        agent_output: str
    ) -> Feedback:
        """
        Collect a corrected version of the agent's output.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
        
        Returns:
            Feedback object with the correction
        """
        print(f"\n🤖 Agent output:\n{agent_output}")
        print("\n" + "-" * 40)
        
        print("Provide the corrected version (or press Enter to skip):")
        print("(For multi-line input, end with a line containing only '.')")
        
        lines = []
        while True:
            line = input()
            if line == '.':
                break
            if line == '' and not lines:
                # Empty first line means skip
                break
            lines.append(line)
        
        correction = '\n'.join(lines) if lines else None
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.CORRECTION,
            context=context,
            agent_output=agent_output,
            human_input=correction
        )
        
        self._store_feedback(feedback)
        
        if correction:
            print("📝 Correction recorded. Thank you!")
        else:
            print("⏭️  Skipped")
        
        return feedback
    
    def collect_preference(
        self,
        context: dict[str, Any],
        option_a: str,
        option_b: str
    ) -> Feedback:
        """
        Collect A/B preference between two options.
        
        Args:
            context: The context of the interaction
            option_a: First option
            option_b: Second option
        
        Returns:
            Feedback object with the preference
        """
        print("\n📋 Which response is better?")
        print("\n[A]:")
        print("-" * 40)
        print(self._truncate(option_a, 400))
        print("-" * 40)
        print("\n[B]:")
        print("-" * 40)
        print(self._truncate(option_b, 400))
        print("-" * 40)
        
        while True:
            choice = input("\nPrefer A, B, or Neither? ").strip().upper()
            if choice in ['A', 'B', 'NEITHER', 'N']:
                break
            print("Please enter A, B, or Neither")
        
        preference = choice if choice not in ['N', 'NEITHER'] else 'NEITHER'
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.PREFERENCE,
            context=context,
            agent_output=json.dumps({"A": option_a, "B": option_b}),
            human_input=preference
        )
        
        self._store_feedback(feedback)
        print(f"✅ Preference recorded: {preference}")
        return feedback
    
    def collect_flag(
        self,
        context: dict[str, Any],
        agent_output: str,
        flag_categories: Optional[list[str]] = None
    ) -> Feedback:
        """
        Flag problematic output with categorization.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
            flag_categories: Available flag categories
        
        Returns:
            Feedback object with the flag
        """
        if flag_categories is None:
            flag_categories = [
                "incorrect",
                "inappropriate",
                "unhelpful",
                "off_topic",
                "harmful",
                "other"
            ]
        
        print(f"\n🤖 Agent output:\n{self._truncate(agent_output, 300)}")
        print("\n🚩 Flag this output? Categories:")
        
        for i, category in enumerate(flag_categories, 1):
            print(f"  {i}. {category}")
        print(f"  0. Don't flag (skip)")
        
        while True:
            try:
                choice = int(input("\nSelect category (0 to skip): "))
                if 0 <= choice <= len(flag_categories):
                    break
                print(f"Please enter 0-{len(flag_categories)}")
            except ValueError:
                print("Please enter a valid number.")
        
        if choice == 0:
            flag_info = None
            print("⏭️  Skipped")
        else:
            category = flag_categories[choice - 1]
            reason = input("Briefly explain the issue: ").strip()
            flag_info = {"category": category, "reason": reason}
            print(f"🚩 Flagged as '{category}'. Thank you!")
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.FLAG,
            context=context,
            agent_output=agent_output,
            human_input=flag_info
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def collect_comment(
        self,
        context: dict[str, Any],
        agent_output: str,
        prompt: str = "Any comments on this response? "
    ) -> Optional[Feedback]:
        """
        Collect a free-form comment.
        
        Args:
            context: The context of the interaction
            agent_output: What the agent produced
            prompt: The prompt to show the user
        
        Returns:
            Feedback object with the comment, or None if empty
        """
        print(f"\n🤖 Agent output:\n{self._truncate(agent_output, 300)}")
        
        comment = input(f"\n{prompt}").strip()
        
        if not comment:
            return None
        
        feedback = Feedback(
            feedback_id=self._generate_id(),
            feedback_type=FeedbackType.COMMENT,
            context=context,
            agent_output=agent_output,
            human_input=comment
        )
        
        self._store_feedback(feedback)
        print("💬 Comment recorded. Thank you!")
        return feedback
    
    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text if it exceeds max_length."""
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    def get_feedback_by_type(
        self,
        feedback_type: FeedbackType
    ) -> list[Feedback]:
        """Get all feedback of a specific type."""
        return [
            fb for fb in self.feedback_log
            if fb.feedback_type == feedback_type
        ]
    
    def get_recent_feedback(self, n: int = 10) -> list[Feedback]:
        """Get the n most recent feedback items."""
        return sorted(
            self.feedback_log,
            key=lambda x: x.timestamp,
            reverse=True
        )[:n]
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected feedback."""
        summary = {
            "total_feedback": len(self.feedback_log),
            "by_type": {},
            "ratings": {
                "count": 0,
                "average": None,
                "distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            },
            "thumbs": {
                "up": 0,
                "down": 0
            },
            "preferences": {
                "A": 0,
                "B": 0,
                "NEITHER": 0
            },
            "flag_categories": {},
            "corrections_provided": 0
        }
        
        for fb in self.feedback_log:
            # Count by type
            type_name = fb.feedback_type.value
            summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
            
            # Process ratings
            if fb.feedback_type == FeedbackType.RATING:
                summary["ratings"]["count"] += 1
                rating = fb.human_input
                if isinstance(rating, int) and 1 <= rating <= 5:
                    summary["ratings"]["distribution"][rating] += 1
            
            # Process thumbs
            if fb.feedback_type == FeedbackType.THUMBS:
                if fb.human_input == "up":
                    summary["thumbs"]["up"] += 1
                else:
                    summary["thumbs"]["down"] += 1
            
            # Process preferences
            if fb.feedback_type == FeedbackType.PREFERENCE:
                pref = fb.human_input
                if pref in summary["preferences"]:
                    summary["preferences"][pref] += 1
            
            # Process flags
            if fb.feedback_type == FeedbackType.FLAG and fb.human_input:
                category = fb.human_input.get("category", "unknown")
                summary["flag_categories"][category] = (
                    summary["flag_categories"].get(category, 0) + 1
                )
            
            # Count corrections
            if fb.feedback_type == FeedbackType.CORRECTION and fb.human_input:
                summary["corrections_provided"] += 1
        
        # Calculate average rating
        if summary["ratings"]["count"] > 0:
            total = sum(
                rating * count
                for rating, count in summary["ratings"]["distribution"].items()
            )
            summary["ratings"]["average"] = round(
                total / summary["ratings"]["count"], 2
            )
        
        return summary
    
    def export_for_training(self) -> list[dict[str, Any]]:
        """
        Export feedback in a format suitable for training/fine-tuning.
        
        Returns:
            List of feedback items formatted for training
        """
        training_data = []
        
        for fb in self.feedback_log:
            if fb.feedback_type == FeedbackType.CORRECTION and fb.human_input:
                # Corrections are valuable for training
                training_data.append({
                    "context": fb.context,
                    "original": fb.agent_output,
                    "corrected": fb.human_input,
                    "type": "correction"
                })
            elif fb.feedback_type == FeedbackType.PREFERENCE:
                # Preferences can be used for RLHF-style training
                options = json.loads(fb.agent_output)
                training_data.append({
                    "context": fb.context,
                    "chosen": options.get(fb.human_input, ""),
                    "rejected": options.get(
                        "B" if fb.human_input == "A" else "A", ""
                    ),
                    "type": "preference"
                })
            elif fb.feedback_type == FeedbackType.RATING:
                # High-rated responses are positive examples
                if fb.human_input >= 4:
                    training_data.append({
                        "context": fb.context,
                        "response": fb.agent_output,
                        "rating": fb.human_input,
                        "type": "positive_example"
                    })
        
        return training_data


def main():
    """Demonstrate the feedback collector."""
    print("=" * 60)
    print("Feedback Collector Demo")
    print("=" * 60)
    
    # Create collector (in-memory for demo)
    collector = FeedbackCollector()
    
    # Sample agent output for feedback
    sample_output = """Based on my analysis, here are the key findings:

1. Revenue increased by 15% year-over-year
2. Customer satisfaction scores improved from 4.2 to 4.5
3. The new product line contributed 30% of total sales

I recommend focusing on expanding the new product line while 
maintaining service quality to sustain this growth trajectory."""
    
    context = {
        "task": "quarterly_analysis",
        "user_query": "Summarize the Q3 results"
    }
    
    # Demo different feedback types
    print("\n--- THUMBS UP/DOWN ---")
    collector.collect_thumbs(context, sample_output)
    
    print("\n--- STAR RATING ---")
    collector.collect_rating(context, sample_output)
    
    print("\n--- FLAG OUTPUT ---")
    collector.collect_flag(context, sample_output)
    
    print("\n--- FREE COMMENT ---")
    collector.collect_comment(context, sample_output)
    
    # Show summary
    print("\n" + "=" * 60)
    print("Feedback Summary")
    print("=" * 60)
    summary = collector.get_summary()
    print(f"Total feedback items: {summary['total_feedback']}")
    print(f"\nBy type: {summary['by_type']}")
    print(f"\nThumbs: 👍 {summary['thumbs']['up']} / 👎 {summary['thumbs']['down']}")
    
    if summary['ratings']['average']:
        print(f"\nAverage rating: {'⭐' * int(summary['ratings']['average'])} "
              f"({summary['ratings']['average']}/5)")


if __name__ == "__main__":
    main()
