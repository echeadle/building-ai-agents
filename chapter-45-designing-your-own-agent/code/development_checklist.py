"""
Development Checklist Utility

A practical tool to track your progress through the development rungs.
Save this as a JSON file and update it as you work.

Chapter 45: Designing Your Own Agent
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DevelopmentChecklist:
    """Track progress through the development rungs"""
    
    def __init__(self, project_name: str, filepath: Optional[Path] = None):
        """
        Initialize checklist for a project
        
        Args:
            project_name: Name of your agent project
            filepath: Optional path to save/load checklist
        """
        self.project_name = project_name
        self.filepath = filepath or Path(f"{project_name}_checklist.json")
        
        # Default checklist structure
        self.checklist = {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "current_rung": 1,
            "rungs": {
                "1": {
                    "name": "Hello World",
                    "estimate": "30 minutes",
                    "status": "not_started",
                    "tasks": [
                        "Basic API call working",
                        "Environment setup verified",
                        "Secrets loading correctly"
                    ],
                    "completed": []
                },
                "2": {
                    "name": "Single Tool",
                    "estimate": "2 hours",
                    "status": "not_started",
                    "tasks": [
                        "Implement core tool",
                        "Agent successfully calls it",
                        "Error handling works",
                        "Output validated"
                    ],
                    "completed": []
                },
                "3": {
                    "name": "Core Flow",
                    "estimate": "1 day",
                    "status": "not_started",
                    "tasks": [
                        "All core tools implemented",
                        "Main workflow working",
                        "End-to-end test with one example",
                        "Obvious bugs fixed"
                    ],
                    "completed": []
                },
                "4": {
                    "name": "Edge Cases",
                    "estimate": "2-3 days",
                    "status": "not_started",
                    "tasks": [
                        "Tested with varied inputs",
                        "Errors handled gracefully",
                        "Input validation added",
                        "Prompts improved"
                    ],
                    "completed": []
                },
                "5": {
                    "name": "Production Hardening",
                    "estimate": "1 week",
                    "status": "not_started",
                    "tasks": [
                        "Observability added",
                        "Proper error handling",
                        "Tests written",
                        "Cost/latency optimized",
                        "Documentation complete"
                    ],
                    "completed": []
                },
                "6": {
                    "name": "Nice-to-Have Features",
                    "estimate": "ongoing",
                    "status": "not_started",
                    "tasks": [
                        "Feature 1 (customize this)",
                        "Feature 2 (customize this)",
                        "User feedback gathered"
                    ],
                    "completed": []
                }
            }
        }
        
        # Load existing checklist if it exists
        if self.filepath.exists():
            self.load()
    
    def complete_task(self, rung: int, task_index: int) -> None:
        """
        Mark a task as complete
        
        Args:
            rung: Rung number (1-6)
            task_index: Task index within the rung (0-based)
        """
        rung_key = str(rung)
        rung_data = self.checklist["rungs"][rung_key]
        
        if task_index < len(rung_data["tasks"]):
            task = rung_data["tasks"][task_index]
            if task not in rung_data["completed"]:
                rung_data["completed"].append(task)
                self.checklist["last_updated"] = datetime.now().isoformat()
                
                # Update rung status
                if len(rung_data["completed"]) == len(rung_data["tasks"]):
                    rung_data["status"] = "complete"
                    print(f"🎉 Rung {rung} complete! Moving to Rung {rung + 1}")
                    if rung < 6:
                        self.checklist["current_rung"] = rung + 1
                        self.checklist["rungs"][str(rung + 1)]["status"] = "in_progress"
                elif len(rung_data["completed"]) > 0:
                    rung_data["status"] = "in_progress"
                
                self.save()
                print(f"✅ Completed: {task}")
    
    def uncomplete_task(self, rung: int, task_index: int) -> None:
        """
        Mark a task as incomplete
        
        Args:
            rung: Rung number (1-6)
            task_index: Task index within the rung (0-based)
        """
        rung_key = str(rung)
        rung_data = self.checklist["rungs"][rung_key]
        
        if task_index < len(rung_data["tasks"]):
            task = rung_data["tasks"][task_index]
            if task in rung_data["completed"]:
                rung_data["completed"].remove(task)
                self.checklist["last_updated"] = datetime.now().isoformat()
                
                # Update rung status
                if len(rung_data["completed"]) == 0:
                    rung_data["status"] = "not_started"
                else:
                    rung_data["status"] = "in_progress"
                
                self.save()
                print(f"⬜ Uncompleted: {task}")
    
    def show_current_rung(self) -> None:
        """Display current rung status"""
        current = self.checklist["current_rung"]
        rung_data = self.checklist["rungs"][str(current)]
        
        print(f"\n{'='*80}")
        print(f"CURRENT RUNG: {current} - {rung_data['name']}")
        print(f"Estimated time: {rung_data['estimate']}")
        print(f"Status: {rung_data['status']}")
        print(f"{'='*80}\n")
        
        for i, task in enumerate(rung_data["tasks"]):
            status = "✅" if task in rung_data["completed"] else "⬜"
            print(f"{status} {i}. {task}")
        
        completed_count = len(rung_data["completed"])
        total_count = len(rung_data["tasks"])
        progress = (completed_count / total_count) * 100
        
        print(f"\nProgress: {completed_count}/{total_count} ({progress:.0f}%)")
        print()
    
    def show_all_rungs(self) -> None:
        """Display all rungs with their status"""
        print(f"\n{'='*80}")
        print(f"PROJECT: {self.project_name}")
        print(f"Created: {self.checklist['created_at'][:10]}")
        print(f"Last Updated: {self.checklist['last_updated'][:10]}")
        print(f"{'='*80}\n")
        
        for rung_num in range(1, 7):
            rung_key = str(rung_num)
            rung_data = self.checklist["rungs"][rung_key]
            
            status_icon = {
                "not_started": "⬜",
                "in_progress": "🔄",
                "complete": "✅"
            }.get(rung_data["status"], "❓")
            
            completed = len(rung_data["completed"])
            total = len(rung_data["tasks"])
            
            print(f"{status_icon} Rung {rung_num}: {rung_data['name']} "
                  f"({completed}/{total} tasks)")
        
        print()
    
    def add_custom_task(self, rung: int, task: str) -> None:
        """
        Add a custom task to a rung
        
        Args:
            rung: Rung number (1-6)
            task: Task description
        """
        rung_key = str(rung)
        self.checklist["rungs"][rung_key]["tasks"].append(task)
        self.checklist["last_updated"] = datetime.now().isoformat()
        self.save()
        print(f"Added task to Rung {rung}: {task}")
    
    def save(self) -> None:
        """Save checklist to file"""
        with open(self.filepath, "w") as f:
            json.dump(self.checklist, f, indent=2)
    
    def load(self) -> None:
        """Load checklist from file"""
        with open(self.filepath, "r") as f:
            self.checklist = json.load(f)


# Example usage
if __name__ == "__main__":
    print("Development Checklist Utility")
    print("=" * 80)
    print()
    print("This utility helps you track progress through the development rungs.")
    print()
    print("Example usage:")
    print()
    print("  # Create a new checklist")
    print("  checklist = DevelopmentChecklist('my-agent')")
    print()
    print("  # Show current rung")
    print("  checklist.show_current_rung()")
    print()
    print("  # Complete a task (rung 1, task 0)")
    print("  checklist.complete_task(1, 0)")
    print()
    print("  # Show all rungs")
    print("  checklist.show_all_rungs()")
    print()
    print("  # Add custom task")
    print("  checklist.add_custom_task(6, 'Add email integration')")
    print()
    print("=" * 80)
    print()
    
    # Interactive demo
    print("Running interactive demo...")
    print()
    
    demo = DevelopmentChecklist("demo-agent")
    demo.show_all_rungs()
    
    # Simulate some progress
    print("Completing some tasks...")
    demo.complete_task(1, 0)  # First task of rung 1
    demo.complete_task(1, 1)  # Second task
    demo.complete_task(1, 2)  # Third task - completes rung 1
    
    demo.show_current_rung()
    
    print("\nFull progress:")
    demo.show_all_rungs()
    
    print(f"\nChecklist saved to: {demo.filepath}")
