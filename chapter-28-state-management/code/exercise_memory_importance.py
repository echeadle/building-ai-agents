"""
Exercise Solution: Memory Importance System

This exercise extends long-term memory with importance scores
and decay over time.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import math

# Load environment variables from .env file
load_dotenv()


@dataclass
class ImportantMemory:
    """
    A memory entry with importance scoring.
    
    Attributes:
        key: Unique identifier for this memory
        value: The stored information
        category: Category for organization
        importance: Base importance score (1-10)
        created_at: When the memory was created
        last_accessed: When the memory was last accessed
        access_count: How many times the memory has been accessed
    """
    
    key: str
    value: Any
    category: str = "general"
    importance: int = 5  # 1-10 scale
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed = datetime.now().isoformat()
        self.access_count += 1
    
    def get_effective_importance(self, decay_rate: float = 0.1) -> float:
        """
        Calculate importance with time-based decay.
        
        Importance decays based on time since last access.
        decay_rate controls how quickly importance decreases.
        
        Args:
            decay_rate: How much to decay per day (0.1 = 10% per day)
            
        Returns:
            Effective importance score (can be < 1)
        """
        # Calculate days since last access
        last_access = datetime.fromisoformat(self.last_accessed)
        days_since_access = (datetime.now() - last_access).days
        
        # Apply exponential decay
        # Formula: importance * e^(-decay_rate * days)
        decay_factor = math.exp(-decay_rate * days_since_access)
        
        # Also boost based on access count (frequently accessed = more important)
        access_boost = min(1.5, 1 + (self.access_count * 0.05))
        
        return self.importance * decay_factor * access_boost
    
    def days_since_access(self) -> int:
        """Get days since last access."""
        last_access = datetime.fromisoformat(self.last_accessed)
        return (datetime.now() - last_access).days


class ImportantMemoryStore:
    """
    Long-term memory store with importance-based retrieval.
    
    Features:
    - Importance scores (1-10)
    - Time-based decay
    - Retrieval by importance threshold
    - Automatic cleanup of unimportant memories
    """
    
    def __init__(
        self,
        storage_path: str = "important_memories.json",
        decay_rate: float = 0.1
    ):
        """
        Initialize the memory store.
        
        Args:
            storage_path: Path to storage file
            decay_rate: Daily decay rate (0.1 = 10% per day)
        """
        self.storage_path = Path(storage_path)
        self.decay_rate = decay_rate
        self.memories: dict[str, ImportantMemory] = {}
        self._load()
    
    def _load(self) -> None:
        """Load memories from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self.memories[key] = ImportantMemory(**entry_data)
                print(f"Loaded {len(self.memories)} memories")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load memories: {e}")
                self.memories = {}
    
    def _save(self) -> None:
        """Save memories to disk."""
        data = {
            key: asdict(entry) 
            for key, entry in self.memories.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def remember(
        self, 
        key: str, 
        value: Any, 
        importance: int = 5,
        category: str = "general"
    ) -> None:
        """
        Store a memory with importance score.
        
        Args:
            key: Unique identifier
            value: Information to store
            importance: Importance score (1-10)
            category: Category for organization
        """
        # Clamp importance to valid range
        importance = max(1, min(10, importance))
        
        if key in self.memories:
            # Update existing memory
            self.memories[key].value = value
            self.memories[key].importance = importance
            self.memories[key].last_accessed = datetime.now().isoformat()
        else:
            # Create new memory
            self.memories[key] = ImportantMemory(
                key=key,
                value=value,
                category=category,
                importance=importance
            )
        
        self._save()
        print(f"Remembered '{key}' with importance {importance}")
    
    def recall(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory by key.
        
        Updates access tracking.
        
        Args:
            key: The memory key
            
        Returns:
            The stored value, or None if not found
        """
        if key in self.memories:
            self.memories[key].touch()
            self._save()
            return self.memories[key].value
        return None
    
    def recall_important(self, threshold: float = 7.0) -> dict[str, Any]:
        """
        Retrieve only high-importance memories.
        
        Args:
            threshold: Minimum effective importance score
            
        Returns:
            Dict of key-value pairs meeting threshold
        """
        result = {}
        
        for key, memory in self.memories.items():
            effective_importance = memory.get_effective_importance(self.decay_rate)
            if effective_importance >= threshold:
                memory.touch()
                result[key] = {
                    "value": memory.value,
                    "importance": memory.importance,
                    "effective_importance": round(effective_importance, 2)
                }
        
        if result:
            self._save()
        
        return result
    
    def forget_unimportant(self, threshold: float = 2.0) -> list[str]:
        """
        Remove memories below importance threshold.
        
        Args:
            threshold: Memories with effective importance below
                      this will be removed
                      
        Returns:
            List of keys that were forgotten
        """
        forgotten = []
        
        for key in list(self.memories.keys()):
            effective_importance = self.memories[key].get_effective_importance(
                self.decay_rate
            )
            if effective_importance < threshold:
                forgotten.append(key)
                del self.memories[key]
        
        if forgotten:
            self._save()
            print(f"Forgot {len(forgotten)} unimportant memories")
        
        return forgotten
    
    def boost_importance(self, key: str, amount: int = 1) -> bool:
        """
        Increase a memory's importance.
        
        Useful when a memory proves useful.
        
        Args:
            key: Memory key
            amount: How much to increase importance
            
        Returns:
            True if memory was found and boosted
        """
        if key in self.memories:
            self.memories[key].importance = min(
                10, 
                self.memories[key].importance + amount
            )
            self.memories[key].touch()
            self._save()
            return True
        return False
    
    def get_all_with_scores(self) -> list[dict]:
        """
        Get all memories with their effective importance scores.
        
        Returns:
            List of dicts with memory info, sorted by effective importance
        """
        result = []
        
        for key, memory in self.memories.items():
            effective = memory.get_effective_importance(self.decay_rate)
            result.append({
                "key": key,
                "value": memory.value,
                "category": memory.category,
                "base_importance": memory.importance,
                "effective_importance": round(effective, 2),
                "access_count": memory.access_count,
                "days_since_access": memory.days_since_access()
            })
        
        # Sort by effective importance (highest first)
        return sorted(result, key=lambda x: x["effective_importance"], reverse=True)
    
    def get_context_for_prompt(
        self, 
        max_entries: int = 10,
        min_importance: float = 3.0
    ) -> str:
        """
        Generate context string for LLM prompts.
        
        Only includes memories above importance threshold.
        
        Args:
            max_entries: Maximum memories to include
            min_importance: Minimum effective importance
            
        Returns:
            Formatted string of important memories
        """
        memories_with_scores = self.get_all_with_scores()
        
        # Filter by importance
        relevant = [
            m for m in memories_with_scores 
            if m["effective_importance"] >= min_importance
        ][:max_entries]
        
        if not relevant:
            return "No important memories to recall."
        
        lines = ["## Important Remembered Information"]
        
        for mem in relevant:
            importance_indicator = "★" * min(5, int(mem["effective_importance"] / 2))
            lines.append(
                f"- {mem['key']}: {mem['value']} {importance_indicator}"
            )
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get statistics about stored memories."""
        if not self.memories:
            return {"total": 0}
        
        scores = [
            m.get_effective_importance(self.decay_rate) 
            for m in self.memories.values()
        ]
        
        return {
            "total": len(self.memories),
            "avg_importance": round(sum(scores) / len(scores), 2),
            "max_importance": round(max(scores), 2),
            "min_importance": round(min(scores), 2),
            "high_importance_count": sum(1 for s in scores if s >= 7)
        }


def demonstrate_important_memory():
    """Demonstrate the importance-based memory system."""
    print("Demonstrating Importance-Based Memory System")
    print("=" * 60)
    
    # Use a test file
    test_file = "test_important_memories.json"
    
    # Create memory store
    store = ImportantMemoryStore(storage_path=test_file, decay_rate=0.1)
    
    # Store memories with different importance levels
    print("\n1. Storing memories with importance scores...")
    store.remember("user_name", "Alice", importance=10, category="identity")
    store.remember("birthday", "March 15", importance=8, category="identity")
    store.remember("favorite_color", "blue", importance=3, category="preference")
    store.remember("last_restaurant", "Pizza Hut", importance=2, category="context")
    store.remember("project_deadline", "2025-06-01", importance=9, category="work")
    store.remember("coffee_preference", "oat milk latte", importance=4, category="preference")
    
    # Show all memories with scores
    print("\n2. All memories with effective importance:")
    print("-" * 60)
    for mem in store.get_all_with_scores():
        print(f"   {mem['key']}: importance={mem['base_importance']}, "
              f"effective={mem['effective_importance']}, "
              f"accesses={mem['access_count']}")
    
    # Retrieve high-importance memories only
    print("\n3. High-importance memories (threshold >= 7):")
    important = store.recall_important(threshold=7.0)
    for key, info in important.items():
        print(f"   {key}: {info['value']} (effective: {info['effective_importance']})")
    
    # Access some memories to boost their effective importance
    print("\n4. Accessing some memories (boosts effective importance)...")
    store.recall("user_name")
    store.recall("user_name")
    store.recall("user_name")
    store.recall("coffee_preference")
    
    # Manually boost importance
    print("\n5. Boosting importance of 'coffee_preference'...")
    store.boost_importance("coffee_preference", amount=3)
    
    # Show updated scores
    print("\n6. Updated scores after access and boost:")
    print("-" * 60)
    for mem in store.get_all_with_scores():
        print(f"   {mem['key']}: effective={mem['effective_importance']}, "
              f"accesses={mem['access_count']}")
    
    # Get context for prompts
    print("\n7. Context string for LLM prompts:")
    print("-" * 60)
    print(store.get_context_for_prompt(min_importance=3.0))
    
    # Forget unimportant memories
    print("\n8. Forgetting unimportant memories (threshold < 2)...")
    forgotten = store.forget_unimportant(threshold=2.0)
    print(f"   Forgotten: {forgotten}")
    
    # Final stats
    print("\n9. Memory statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Simulate time passing (for demonstration)
    print("\n10. Simulating time passage (modifying last_accessed)...")
    # Manually set some memories to appear old
    if "favorite_color" in store.memories:
        old_date = (datetime.now() - timedelta(days=30)).isoformat()
        store.memories["favorite_color"].last_accessed = old_date
        store._save()
    
    print("    Set 'favorite_color' to last accessed 30 days ago")
    print("\n    Updated effective importance:")
    for mem in store.get_all_with_scores():
        print(f"    {mem['key']}: effective={mem['effective_importance']}, "
              f"days_old={mem['days_since_access']}")
    
    # Clean up
    print("\n11. Cleaning up test file...")
    os.remove(test_file)
    print("    Done!")


if __name__ == "__main__":
    demonstrate_important_memory()
