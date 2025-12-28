"""
Long-term memory with file-based persistence.

This example shows how to store and retrieve memories
that persist across sessions.

Chapter 28: State Management
"""

import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


@dataclass
class MemoryEntry:
    """A single long-term memory entry."""
    
    key: str
    value: Any
    category: str  # e.g., "preference", "fact", "context"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.updated_at = datetime.now().isoformat()
        self.access_count += 1


class LongTermMemory:
    """
    File-based long-term memory storage.
    
    Persists memories to a JSON file for retrieval across sessions.
    """
    
    def __init__(self, storage_path: str = "agent_memory.json"):
        """
        Initialize long-term memory.
        
        Args:
            storage_path: Path to the JSON file for storing memories
        """
        self.storage_path = Path(storage_path)
        self.memories: dict[str, MemoryEntry] = {}
        self._load()
    
    def _load(self) -> None:
        """Load memories from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for key, entry_data in data.items():
                        self.memories[key] = MemoryEntry(**entry_data)
                print(f"Loaded {len(self.memories)} memories from {self.storage_path}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load memories: {e}")
                self.memories = {}
        else:
            print(f"No existing memory file. Starting fresh.")
    
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
        category: str = "general"
    ) -> None:
        """
        Store a memory.
        
        Args:
            key: Unique identifier for this memory
            value: The information to remember
            category: Category for organizing memories
        """
        if key in self.memories:
            # Update existing memory
            self.memories[key].value = value
            self.memories[key].updated_at = datetime.now().isoformat()
        else:
            # Create new memory
            self.memories[key] = MemoryEntry(
                key=key,
                value=value,
                category=category
            )
        self._save()
        print(f"Remembered: {key} = {value}")
    
    def recall(self, key: str) -> Optional[Any]:
        """
        Retrieve a memory by key.
        
        Args:
            key: The memory key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        if key in self.memories:
            self.memories[key].touch()
            self._save()
            return self.memories[key].value
        return None
    
    def recall_by_category(self, category: str) -> dict[str, Any]:
        """
        Retrieve all memories in a category.
        
        Args:
            category: The category to filter by
            
        Returns:
            Dictionary of key-value pairs in the category
        """
        result = {}
        for key, entry in self.memories.items():
            if entry.category == category:
                entry.touch()
                result[key] = entry.value
        if result:
            self._save()
        return result
    
    def forget(self, key: str) -> bool:
        """
        Remove a memory.
        
        Args:
            key: The memory key to remove
            
        Returns:
            True if memory was removed, False if not found
        """
        if key in self.memories:
            del self.memories[key]
            self._save()
            print(f"Forgot: {key}")
            return True
        return False
    
    def search(self, query: str) -> dict[str, Any]:
        """
        Search memories by key or value content.
        
        Args:
            query: Search string
            
        Returns:
            Matching memories
        """
        query_lower = query.lower()
        results = {}
        
        for key, entry in self.memories.items():
            if query_lower in key.lower() or query_lower in str(entry.value).lower():
                results[key] = entry.value
        
        return results
    
    def get_context_for_prompt(self, max_entries: int = 10) -> str:
        """
        Generate a context string of recent/relevant memories for prompts.
        
        Args:
            max_entries: Maximum number of memories to include
            
        Returns:
            Formatted string of memories
        """
        if not self.memories:
            return "No long-term memories stored."
        
        # Sort by access count and recency
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.access_count, m.updated_at),
            reverse=True
        )[:max_entries]
        
        lines = ["## Remembered Information"]
        
        # Group by category
        by_category: dict[str, list] = {}
        for mem in sorted_memories:
            if mem.category not in by_category:
                by_category[mem.category] = []
            by_category[mem.category].append(mem)
        
        for category, mems in by_category.items():
            lines.append(f"\n### {category.title()}")
            for mem in mems:
                value_str = str(mem.value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                lines.append(f"- {mem.key}: {value_str}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get statistics about stored memories."""
        if not self.memories:
            return {"total": 0}
        
        categories = {}
        for entry in self.memories.values():
            categories[entry.category] = categories.get(entry.category, 0) + 1
        
        return {
            "total": len(self.memories),
            "by_category": categories,
            "most_accessed": max(self.memories.values(), key=lambda m: m.access_count).key
        }
    
    def clear_all(self) -> None:
        """Clear all memories."""
        self.memories = {}
        self._save()
        print("All memories cleared.")


def demonstrate_long_term_memory():
    """Show how long-term memory works."""
    print("Demonstrating Long-Term Memory")
    print("=" * 50)
    
    # Use a test file
    test_file = "test_memory.json"
    
    # Create memory store
    print("\n1. Creating memory store...")
    memory = LongTermMemory(test_file)
    
    # Store some memories
    print("\n2. Storing memories...")
    memory.remember("user_name", "Alice", category="preference")
    memory.remember("favorite_color", "blue", category="preference")
    memory.remember("timezone", "America/New_York", category="preference")
    memory.remember("last_topic", "machine learning", category="context")
    memory.remember("project_deadline", "2025-03-15", category="fact")
    memory.remember("favorite_foods", ["pizza", "sushi", "tacos"], category="preference")
    
    # Retrieve a specific memory
    print("\n3. Retrieving specific memory...")
    name = memory.recall("user_name")
    print(f"User's name: {name}")
    
    # Get all preferences
    print("\n4. Getting all preferences...")
    prefs = memory.recall_by_category("preference")
    print(f"Preferences: {json.dumps(prefs, indent=2)}")
    
    # Search memories
    print("\n5. Searching memories...")
    results = memory.search("blue")
    print(f"Search results for 'blue': {results}")
    
    # Get context for prompts
    print("\n6. Context for LLM prompts:")
    print("-" * 40)
    print(memory.get_context_for_prompt())
    print("-" * 40)
    
    # Get stats
    print("\n7. Memory statistics:")
    stats = memory.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    
    # Demonstrate persistence
    print("\n8. Demonstrating persistence...")
    del memory  # Delete the object
    
    # Create a new instance - should load from file
    memory2 = LongTermMemory(test_file)
    recalled = memory2.recall("user_name")
    print(f"After reload, user_name = {recalled}")
    
    # Clean up test file
    print("\n9. Cleaning up...")
    os.remove(test_file)
    print("Test file removed.")


if __name__ == "__main__":
    demonstrate_long_term_memory()
