"""
Simple JSON storage for productivity agent.

Chapter 44: Project - Personal Productivity Agent
"""

import json
import os
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime


class Storage:
    """Simple JSON-based storage for tasks, notes, and context."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize storage with data directory.
        
        Args:
            data_dir: Directory to store JSON files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.tasks_file = self.data_dir / "tasks.json"
        self.notes_file = self.data_dir / "notes.json"
        self.context_file = self.data_dir / "context.json"
        
        # Initialize files if they don't exist
        self._init_file(self.tasks_file, [])
        self._init_file(self.notes_file, [])
        self._init_file(self.context_file, {
            "current_project": None,
            "recent_projects": [],
            "preferences": {},
            "important_dates": {}
        })
    
    def _init_file(self, filepath: Path, default_data: Any) -> None:
        """
        Initialize a JSON file if it doesn't exist.
        
        Args:
            filepath: Path to the file
            default_data: Default data structure for the file
        """
        if not filepath.exists():
            with open(filepath, 'w') as f:
                json.dump(default_data, f, indent=2)
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        Load all tasks from storage.
        
        Returns:
            List of task dictionaries
        """
        with open(self.tasks_file, 'r') as f:
            return json.load(f)
    
    def save_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """
        Save all tasks to storage.
        
        Args:
            tasks: List of task dictionaries to save
        """
        with open(self.tasks_file, 'w') as f:
            json.dump(tasks, f, indent=2)
    
    def load_notes(self) -> List[Dict[str, Any]]:
        """
        Load all notes from storage.
        
        Returns:
            List of note dictionaries
        """
        with open(self.notes_file, 'r') as f:
            return json.load(f)
    
    def save_notes(self, notes: List[Dict[str, Any]]) -> None:
        """
        Save all notes to storage.
        
        Args:
            notes: List of note dictionaries to save
        """
        with open(self.notes_file, 'w') as f:
            json.dump(notes, f, indent=2)
    
    def load_context(self) -> Dict[str, Any]:
        """
        Load user context from storage.
        
        Returns:
            User context dictionary
        """
        with open(self.context_file, 'r') as f:
            return json.load(f)
    
    def save_context(self, context: Dict[str, Any]) -> None:
        """
        Save user context to storage.
        
        Args:
            context: User context dictionary to save
        """
        with open(self.context_file, 'w') as f:
            json.dump(context, f, indent=2)


if __name__ == "__main__":
    # Demo storage functionality
    print("Storage Module Demo")
    print("===================\n")
    
    # Initialize storage
    storage = Storage("demo_data")
    
    # Load initial (empty) data
    tasks = storage.load_tasks()
    notes = storage.load_notes()
    context = storage.load_context()
    
    print(f"Initial tasks: {len(tasks)}")
    print(f"Initial notes: {len(notes)}")
    print(f"Initial context: {context}\n")
    
    # Add a sample task
    sample_task = {
        "id": "task_demo",
        "title": "Test the storage module",
        "description": "Verify that JSON storage works correctly",
        "status": "done",
        "priority": "high",
        "project": "Chapter 44",
        "due_date": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "tags": ["testing", "demo"]
    }
    
    tasks.append(sample_task)
    storage.save_tasks(tasks)
    
    print("✓ Saved sample task")
    
    # Reload to verify persistence
    reloaded_tasks = storage.load_tasks()
    print(f"✓ Reloaded {len(reloaded_tasks)} task(s)")
    print(f"\nTask details:")
    print(f"  Title: {reloaded_tasks[0]['title']}")
    print(f"  Status: {reloaded_tasks[0]['status']}")
    print(f"  Priority: {reloaded_tasks[0]['priority']}")
    
    print("\n✓ Storage module working correctly!")
