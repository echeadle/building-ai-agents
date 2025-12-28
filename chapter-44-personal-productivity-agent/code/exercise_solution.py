"""
Exercise Solution: Time Tracking Feature

This solution adds time tracking to the productivity agent, allowing users to:
- Start timers for tasks
- Stop timers and record duration
- View time entries
- Get time reports

Chapter 44: Project - Personal Productivity Agent
"""

import os
from dotenv import load_dotenv
import anthropic
from typing import Dict, List, Any, Optional
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# Enhanced Storage with Time Entries
class StorageWithTimeTracking:
    """Extended storage that includes time tracking."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize storage with data directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.tasks_file = self.data_dir / "tasks.json"
        self.notes_file = self.data_dir / "notes.json"
        self.context_file = self.data_dir / "context.json"
        self.time_entries_file = self.data_dir / "time_entries.json"
        self.active_timers_file = self.data_dir / "active_timers.json"
        
        # Initialize files
        self._init_file(self.tasks_file, [])
        self._init_file(self.notes_file, [])
        self._init_file(self.context_file, {
            "current_project": None,
            "recent_projects": [],
            "preferences": {},
            "important_dates": {}
        })
        self._init_file(self.time_entries_file, [])
        self._init_file(self.active_timers_file, {})
    
    def _init_file(self, filepath: Path, default_data: Any) -> None:
        """Initialize a JSON file if it doesn't exist."""
        if not filepath.exists():
            with open(filepath, 'w') as f:
                json.dump(default_data, f, indent=2)
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load all tasks."""
        with open(self.tasks_file, 'r') as f:
            return json.load(f)
    
    def save_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        """Save all tasks."""
        with open(self.tasks_file, 'w') as f:
            json.dump(tasks, f, indent=2)
    
    def load_notes(self) -> List[Dict[str, Any]]:
        """Load all notes."""
        with open(self.notes_file, 'r') as f:
            return json.load(f)
    
    def save_notes(self, notes: List[Dict[str, Any]]) -> None:
        """Save all notes."""
        with open(self.notes_file, 'w') as f:
            json.dump(notes, f, indent=2)
    
    def load_context(self) -> Dict[str, Any]:
        """Load user context."""
        with open(self.context_file, 'r') as f:
            return json.load(f)
    
    def save_context(self, context: Dict[str, Any]) -> None:
        """Save user context."""
        with open(self.context_file, 'w') as f:
            json.dump(context, f, indent=2)
    
    def load_time_entries(self) -> List[Dict[str, Any]]:
        """Load all time entries."""
        with open(self.time_entries_file, 'r') as f:
            return json.load(f)
    
    def save_time_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Save all time entries."""
        with open(self.time_entries_file, 'w') as f:
            json.dump(entries, f, indent=2)
    
    def load_active_timers(self) -> Dict[str, Any]:
        """Load active timers."""
        with open(self.active_timers_file, 'r') as f:
            return json.load(f)
    
    def save_active_timers(self, timers: Dict[str, Any]) -> None:
        """Save active timers."""
        with open(self.active_timers_file, 'w') as f:
            json.dump(timers, f, indent=2)


# Time Tracking Tools
def start_timer(storage: StorageWithTimeTracking, task_id: str) -> Dict[str, Any]:
    """
    Start a timer for a task.
    
    Args:
        storage: Storage instance
        task_id: ID of task to track time for
    
    Returns:
        Timer start confirmation
    """
    # Check if task exists
    tasks = storage.load_tasks()
    task = next((t for t in tasks if t["id"] == task_id), None)
    
    if not task:
        return {"success": False, "error": f"Task {task_id} not found"}
    
    # Check if timer already running for this task
    active_timers = storage.load_active_timers()
    if task_id in active_timers:
        return {
            "success": False,
            "error": f"Timer already running for task: {task['title']}"
        }
    
    # Start timer
    start_time = datetime.now().isoformat()
    active_timers[task_id] = {
        "task_id": task_id,
        "task_title": task["title"],
        "start_time": start_time
    }
    
    storage.save_active_timers(active_timers)
    
    return {
        "success": True,
        "message": f"Timer started for: {task['title']}",
        "start_time": start_time
    }


def stop_timer(storage: StorageWithTimeTracking, task_id: str) -> Dict[str, Any]:
    """
    Stop a running timer and record the time entry.
    
    Args:
        storage: Storage instance
        task_id: ID of task to stop timer for
    
    Returns:
        Time entry with duration
    """
    # Check if timer is running
    active_timers = storage.load_active_timers()
    if task_id not in active_timers:
        return {
            "success": False,
            "error": "No active timer for this task"
        }
    
    # Get timer info
    timer = active_timers[task_id]
    start_time = datetime.fromisoformat(timer["start_time"])
    end_time = datetime.now()
    duration_seconds = int((end_time - start_time).total_seconds())
    
    # Create time entry
    time_entries = storage.load_time_entries()
    entry = {
        "id": f"time_{uuid.uuid4().hex[:8]}",
        "task_id": task_id,
        "task_title": timer["task_title"],
        "start_time": timer["start_time"],
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_seconds,
        "duration_formatted": _format_duration(duration_seconds)
    }
    
    time_entries.append(entry)
    storage.save_time_entries(time_entries)
    
    # Remove active timer
    del active_timers[task_id]
    storage.save_active_timers(active_timers)
    
    return {
        "success": True,
        "entry": entry,
        "message": f"Logged {entry['duration_formatted']} for: {timer['task_title']}"
    }


def list_active_timers(storage: StorageWithTimeTracking) -> Dict[str, Any]:
    """
    List all currently running timers.
    
    Args:
        storage: Storage instance
    
    Returns:
        List of active timers with elapsed time
    """
    active_timers = storage.load_active_timers()
    
    timers_with_elapsed = []
    for task_id, timer in active_timers.items():
        start_time = datetime.fromisoformat(timer["start_time"])
        elapsed_seconds = int((datetime.now() - start_time).total_seconds())
        
        timers_with_elapsed.append({
            "task_id": task_id,
            "task_title": timer["task_title"],
            "start_time": timer["start_time"],
            "elapsed": _format_duration(elapsed_seconds)
        })
    
    return {
        "success": True,
        "active_timers": timers_with_elapsed,
        "count": len(timers_with_elapsed)
    }


def time_report(
    storage: StorageWithTimeTracking,
    task_id: Optional[str] = None,
    days: int = 7
) -> Dict[str, Any]:
    """
    Generate a time tracking report.
    
    Args:
        storage: Storage instance
        task_id: Optional task ID to filter by
        days: Number of days to include (default 7)
    
    Returns:
        Time report with statistics
    """
    time_entries = storage.load_time_entries()
    
    # Filter by date range
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_entries = [
        e for e in time_entries
        if datetime.fromisoformat(e["start_time"]) >= cutoff_date
    ]
    
    # Filter by task if specified
    if task_id:
        recent_entries = [e for e in recent_entries if e["task_id"] == task_id]
    
    # Calculate statistics
    total_seconds = sum(e["duration_seconds"] for e in recent_entries)
    
    # Group by task
    by_task = {}
    for entry in recent_entries:
        task_id = entry["task_id"]
        if task_id not in by_task:
            by_task[task_id] = {
                "task_title": entry["task_title"],
                "entries": 0,
                "total_seconds": 0
            }
        by_task[task_id]["entries"] += 1
        by_task[task_id]["total_seconds"] += entry["duration_seconds"]
    
    # Format task summaries
    task_summaries = [
        {
            "task_title": data["task_title"],
            "sessions": data["entries"],
            "total_time": _format_duration(data["total_seconds"])
        }
        for data in by_task.values()
    ]
    
    # Sort by total time (descending)
    task_summaries.sort(key=lambda x: by_task[next(
        k for k, v in by_task.items() if v["task_title"] == x["task_title"]
    )]["total_seconds"], reverse=True)
    
    return {
        "success": True,
        "period_days": days,
        "total_time": _format_duration(total_seconds),
        "total_sessions": len(recent_entries),
        "by_task": task_summaries
    }


def _format_duration(seconds: int) -> str:
    """Format duration in seconds to human-readable string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


# Enhanced Agent with Time Tracking
class ProductivityAgentWithTimeTracking:
    """Productivity agent with time tracking capabilities."""
    
    def __init__(self, storage: StorageWithTimeTracking):
        """Initialize the agent."""
        self.storage = storage
        self.client = anthropic.Anthropic()
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Import base tools
        from tools import (
            create_task, list_tasks, update_task,
            save_note, search_notes, get_context, update_context
        )
        
        # Add time tracking tools
        self.tool_functions = {
            "create_task": create_task,
            "list_tasks": list_tasks,
            "update_task": update_task,
            "save_note": save_note,
            "search_notes": search_notes,
            "get_context": get_context,
            "update_context": update_context,
            "start_timer": start_timer,
            "stop_timer": stop_timer,
            "list_active_timers": list_active_timers,
            "time_report": time_report
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return """You are a helpful personal productivity assistant with time tracking capabilities.

Your capabilities:
- Create, list, and update tasks
- Save and search notes
- Track projects and context
- Start and stop timers for tasks
- Generate time reports
- Provide context-aware assistance

Guidelines for time tracking:
1. When users start working on a task, offer to start a timer
2. Remind users about active timers periodically
3. When users finish a task, suggest stopping the timer
4. Proactively offer time reports to show productivity

Always help users stay organized and aware of how they spend their time."""
    
    def _get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions including time tracking."""
        base_tools = [
            {
                "name": "create_task",
                "description": "Create a new task.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Task title"},
                        "description": {"type": "string", "description": "Task description"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"},
                        "project": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["title"]
                }
            },
            {
                "name": "list_tasks",
                "description": "List tasks with optional filters.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["todo", "in_progress", "done", "archived"]},
                        "project": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        "tag": {"type": "string"}
                    }
                }
            },
            {
                "name": "start_timer",
                "description": "Start a timer for a task. Use this when the user begins working on something.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to track time for"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "stop_timer",
                "description": "Stop a running timer and record the time entry.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "ID of the task to stop tracking"
                        }
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "list_active_timers",
                "description": "List all currently running timers with elapsed time.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "time_report",
                "description": "Generate a time tracking report for recent work.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Optional task ID to filter by"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days to include (default 7)"
                        }
                    }
                }
            }
        ]
        
        return base_tools
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool."""
        try:
            tool_func = self.tool_functions[tool_name]
            result = tool_func(self.storage, **tool_input)
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def chat(self, user_message: str) -> str:
        """Process a user message."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self._get_system_prompt(),
                messages=self.conversation_history,
                tools=self._get_tools()
            )
            
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            if response.stop_reason == "end_turn":
                text_response = ""
                for block in response.content:
                    if block.type == "text":
                        text_response += block.text
                return text_response
            
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
                continue
        
        return "Processing limit reached. Please rephrase."


def main():
    """Demo the time tracking feature."""
    storage = StorageWithTimeTracking()
    agent = ProductivityAgentWithTimeTracking(storage)
    
    print("Productivity Agent with Time Tracking")
    print("=====================================")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
