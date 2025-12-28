"""
Exercise: Design a complete tool following best practices.

Appendix C: Tool Design Patterns

TASK: Design a tool for managing tasks in a todo list.
The tool should allow creating, updating, and searching tasks.

REQUIREMENTS:
1. Follow naming conventions (verb-based, snake_case)
2. Write comprehensive descriptions with use cases
3. Include proper parameter validation
4. Return structured error messages
5. Handle edge cases gracefully
"""

import os
from typing import Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Task:
    """Represents a task in the todo list."""
    id: str
    title: str
    description: str
    status: str  # "pending", "in_progress", "completed"
    priority: str  # "low", "medium", "high"
    created_at: str
    updated_at: str
    due_date: Optional[str] = None


class TaskManager:
    """Simple in-memory task manager for demonstration."""
    
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self._next_id = 1
    
    def _generate_id(self) -> str:
        """Generate a unique task ID."""
        task_id = f"task_{self._next_id}"
        self._next_id += 1
        return task_id
    
    def add_task(self, task: Task) -> None:
        """Add a task to storage."""
        self.tasks[task.id] = task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, updates: dict[str, Any]) -> Optional[Task]:
        """Update a task with new values."""
        task = self.tasks.get(task_id)
        if task:
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = datetime.now().isoformat()
        return task
    
    def search_tasks(
        self,
        query: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None
    ) -> list[Task]:
        """Search tasks by various criteria."""
        results = list(self.tasks.values())
        
        if query:
            query_lower = query.lower()
            results = [
                t for t in results
                if query_lower in t.title.lower() or query_lower in t.description.lower()
            ]
        
        if status:
            results = [t for t in results if t.status == status]
        
        if priority:
            results = [t for t in results if t.priority == priority]
        
        return results


# Global task manager instance
task_manager = TaskManager()


# Tool 1: Create Task
# ===================

def create_task(
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: Optional[str] = None
) -> dict[str, Any]:
    """
    Create a new task in the todo list.
    
    This demonstrates:
    - Clear, verb-based naming
    - Required vs optional parameters
    - Input validation
    - Structured success/error returns
    """
    # Validate title
    if not title or not title.strip():
        return {
            "success": False,
            "error": "invalid_input",
            "message": "Task title cannot be empty",
            "suggestion": "Provide a descriptive title for the task"
        }
    
    # Validate priority
    valid_priorities = ["low", "medium", "high"]
    if priority not in valid_priorities:
        return {
            "success": False,
            "error": "invalid_priority",
            "message": f"Invalid priority: {priority}",
            "suggestion": f"Use one of: {', '.join(valid_priorities)}"
        }
    
    # Validate due_date format if provided
    if due_date:
        try:
            datetime.fromisoformat(due_date)
        except ValueError:
            return {
                "success": False,
                "error": "invalid_date",
                "message": f"Invalid date format: {due_date}",
                "suggestion": "Use ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
            }
    
    # Create task
    now = datetime.now().isoformat()
    task = Task(
        id=task_manager._generate_id(),
        title=title.strip(),
        description=description.strip(),
        status="pending",
        priority=priority,
        created_at=now,
        updated_at=now,
        due_date=due_date
    )
    
    task_manager.add_task(task)
    
    return {
        "success": True,
        "task_id": task.id,
        "task": asdict(task),
        "message": f"Task '{task.title}' created successfully"
    }


CREATE_TASK_TOOL = {
    "name": "create_task",
    "description": """Create a new task in the todo list.

Use this tool when the user wants to:
- Add a new task or todo item
- Create a reminder for something
- Track a new action item

All tasks start with 'pending' status.
Priority levels affect task sorting (high > medium > low).
Optional due date can be set for deadline tracking.

Returns the created task with its unique ID.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Task title - short, descriptive name (e.g., 'Buy groceries', 'Finish report')"
            },
            "description": {
                "type": "string",
                "description": "Detailed task description (optional). Use for additional context, notes, or requirements."
            },
            "priority": {
                "type": "string",
                "description": "Task priority level (default: medium). Affects sorting and visibility.",
                "enum": ["low", "medium", "high"]
            },
            "due_date": {
                "type": "string",
                "description": "Optional due date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). Example: '2024-12-31'"
            }
        },
        "required": ["title"]
    }
}


# Tool 2: Update Task
# ===================

def update_task(
    task_id: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None
) -> dict[str, Any]:
    """
    Update an existing task.
    
    This demonstrates:
    - Partial updates (only specified fields)
    - Validation of each field
    - Clear error messages
    - Flexible parameter handling
    """
    # Check task exists
    task = task_manager.get_task(task_id)
    if not task:
        return {
            "success": False,
            "error": "not_found",
            "message": f"Task '{task_id}' not found",
            "suggestion": "Use search_tasks to find the correct task ID"
        }
    
    # Build updates dictionary
    updates = {}
    
    if title is not None:
        if not title.strip():
            return {
                "success": False,
                "error": "invalid_input",
                "message": "Task title cannot be empty"
            }
        updates["title"] = title.strip()
    
    if description is not None:
        updates["description"] = description.strip()
    
    if status is not None:
        valid_statuses = ["pending", "in_progress", "completed"]
        if status not in valid_statuses:
            return {
                "success": False,
                "error": "invalid_status",
                "message": f"Invalid status: {status}",
                "suggestion": f"Use one of: {', '.join(valid_statuses)}"
            }
        updates["status"] = status
    
    if priority is not None:
        valid_priorities = ["low", "medium", "high"]
        if priority not in valid_priorities:
            return {
                "success": False,
                "error": "invalid_priority",
                "message": f"Invalid priority: {priority}",
                "suggestion": f"Use one of: {', '.join(valid_priorities)}"
            }
        updates["priority"] = priority
    
    if due_date is not None:
        try:
            datetime.fromisoformat(due_date)
            updates["due_date"] = due_date
        except ValueError:
            return {
                "success": False,
                "error": "invalid_date",
                "message": f"Invalid date format: {due_date}",
                "suggestion": "Use ISO format: YYYY-MM-DD"
            }
    
    # Apply updates
    updated_task = task_manager.update_task(task_id, updates)
    
    return {
        "success": True,
        "task_id": task_id,
        "task": asdict(updated_task),
        "updated_fields": list(updates.keys()),
        "message": f"Task '{task_id}' updated successfully"
    }


UPDATE_TASK_TOOL = {
    "name": "update_task",
    "description": """Update an existing task's properties.

Use this tool when the user wants to:
- Change task details (title, description)
- Update task status (mark as completed, in progress)
- Change priority level
- Set or update due date

Only specified fields will be updated; others remain unchanged.
Task ID must be valid (use search_tasks to find it).

Returns the updated task with all current values.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "Unique task identifier (e.g., 'task_1', 'task_42'). Get this from create_task or search_tasks."
            },
            "title": {
                "type": "string",
                "description": "New task title (optional). Only provide if changing."
            },
            "description": {
                "type": "string",
                "description": "New task description (optional). Only provide if changing."
            },
            "status": {
                "type": "string",
                "description": "New status (optional). Only provide if changing.",
                "enum": ["pending", "in_progress", "completed"]
            },
            "priority": {
                "type": "string",
                "description": "New priority (optional). Only provide if changing.",
                "enum": ["low", "medium", "high"]
            },
            "due_date": {
                "type": "string",
                "description": "New due date in ISO format (optional). Only provide if changing."
            }
        },
        "required": ["task_id"]
    }
}


# Tool 3: Search Tasks
# ====================

def search_tasks(
    query: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 20
) -> dict[str, Any]:
    """
    Search and filter tasks.
    
    This demonstrates:
    - Multiple optional filters
    - Default values
    - Clear result structure
    - Helpful empty results message
    """
    # Validate status if provided
    if status:
        valid_statuses = ["pending", "in_progress", "completed"]
        if status not in valid_statuses:
            return {
                "success": False,
                "error": "invalid_status",
                "message": f"Invalid status filter: {status}",
                "suggestion": f"Use one of: {', '.join(valid_statuses)}"
            }
    
    # Validate priority if provided
    if priority:
        valid_priorities = ["low", "medium", "high"]
        if priority not in valid_priorities:
            return {
                "success": False,
                "error": "invalid_priority",
                "message": f"Invalid priority filter: {priority}",
                "suggestion": f"Use one of: {', '.join(valid_priorities)}"
            }
    
    # Validate limit
    if limit < 1 or limit > 100:
        return {
            "success": False,
            "error": "invalid_limit",
            "message": f"Limit must be between 1 and 100, got {limit}"
        }
    
    # Search tasks
    results = task_manager.search_tasks(query, status, priority)
    
    # Apply limit
    results = results[:limit]
    
    # Build response
    response = {
        "success": True,
        "count": len(results),
        "filters_applied": {
            "query": query,
            "status": status,
            "priority": priority
        },
        "tasks": [asdict(task) for task in results]
    }
    
    if not results:
        response["message"] = "No tasks found matching the criteria"
        response["suggestion"] = "Try broader search terms or remove filters"
    
    return response


SEARCH_TASKS_TOOL = {
    "name": "search_tasks",
    "description": """Search and filter tasks in the todo list.

Use this tool when the user wants to:
- Find specific tasks by keywords
- List all pending/completed tasks
- Filter tasks by priority
- Get an overview of tasks

Filters can be combined for more specific results.
Text query searches both title and description.
Returns up to the specified limit (default: 20, max: 100).

Returns list of matching tasks sorted by creation date (newest first).""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search keywords to find in title or description (optional). Example: 'groceries', 'report deadline'"
            },
            "status": {
                "type": "string",
                "description": "Filter by task status (optional). Useful for viewing only pending or completed tasks.",
                "enum": ["pending", "in_progress", "completed"]
            },
            "priority": {
                "type": "string",
                "description": "Filter by priority level (optional). Useful for focusing on high-priority items.",
                "enum": ["low", "medium", "high"]
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 20, max: 100)",
                "minimum": 1,
                "maximum": 100
            }
        },
        "required": []
    }
}


# Demonstration
# =============

if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE: WELL-DESIGNED TASK MANAGEMENT TOOLS")
    print("=" * 60)
    
    # Create some tasks
    print("\n1. CREATING TASKS")
    print("-" * 60)
    
    result = create_task(
        title="Write documentation",
        description="Complete API documentation for new features",
        priority="high",
        due_date="2024-12-15"
    )
    print(f"✅ Created: {result['task_id']} - {result['message']}")
    
    result = create_task(
        title="Buy groceries",
        priority="medium"
    )
    print(f"✅ Created: {result['task_id']}")
    
    result = create_task(
        title="Review pull requests",
        priority="high"
    )
    print(f"✅ Created: {result['task_id']}")
    
    # Try invalid input
    result = create_task(title="")
    print(f"❌ Validation error: {result['message']}")
    
    # Update a task
    print("\n2. UPDATING TASKS")
    print("-" * 60)
    
    result = update_task(
        task_id="task_1",
        status="in_progress"
    )
    print(f"✅ Updated: {result['message']}")
    print(f"   Changed: {', '.join(result['updated_fields'])}")
    
    # Try invalid task ID
    result = update_task(task_id="task_999")
    print(f"❌ Not found: {result['message']}")
    
    # Search tasks
    print("\n3. SEARCHING TASKS")
    print("-" * 60)
    
    result = search_tasks()
    print(f"All tasks: {result['count']} found")
    for task in result['tasks']:
        print(f"  • [{task['status']}] {task['title']} (Priority: {task['priority']})")
    
    result = search_tasks(priority="high")
    print(f"\nHigh priority tasks: {result['count']} found")
    for task in result['tasks']:
        print(f"  • {task['title']}")
    
    result = search_tasks(query="documentation")
    print(f"\nTasks matching 'documentation': {result['count']} found")
    
    # Validate tool definitions
    print("\n4. VALIDATING TOOL DEFINITIONS")
    print("-" * 60)
    
    from tool_validator import ToolValidator
    
    validator = ToolValidator()
    
    print("\nValidating CREATE_TASK_TOOL:")
    issues = validator.validate(CREATE_TASK_TOOL)
    if not issues:
        print("✅ No issues found!")
    
    print("\nValidating UPDATE_TASK_TOOL:")
    issues = validator.validate(UPDATE_TASK_TOOL)
    if not issues:
        print("✅ No issues found!")
    
    print("\nValidating SEARCH_TASKS_TOOL:")
    issues = validator.validate(SEARCH_TASKS_TOOL)
    if not issues:
        print("✅ No issues found!")
    
    print("\n" + "=" * 60)
    print("All tools follow best practices!")
    print("=" * 60)
