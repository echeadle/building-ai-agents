"""
Tools for the productivity agent.

Chapter 44: Project - Personal Productivity Agent
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any


def create_task(
    storage: 'Storage',
    title: str,
    description: str = "",
    priority: str = "medium",
    due_date: Optional[str] = None,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a new task.
    
    Args:
        storage: Storage instance
        title: Task title
        description: Task description
        priority: Priority level (low, medium, high)
        due_date: Due date in ISO format (YYYY-MM-DD)
        project: Project name
        tags: List of tags
    
    Returns:
        Created task with ID
    """
    tasks = storage.load_tasks()
    
    task = {
        "id": f"task_{uuid.uuid4().hex[:8]}",
        "title": title,
        "description": description,
        "status": "todo",
        "priority": priority,
        "project": project,
        "due_date": due_date,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "tags": tags or []
    }
    
    tasks.append(task)
    storage.save_tasks(tasks)
    
    return {"success": True, "task": task}


def list_tasks(
    storage: 'Storage',
    status: Optional[str] = None,
    project: Optional[str] = None,
    priority: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    List tasks with optional filters.
    
    Args:
        storage: Storage instance
        status: Filter by status
        project: Filter by project
        priority: Filter by priority
        tag: Filter by tag
    
    Returns:
        List of matching tasks
    """
    tasks = storage.load_tasks()
    
    # Apply filters
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if project:
        tasks = [t for t in tasks if t["project"] == project]
    if priority:
        tasks = [t for t in tasks if t["priority"] == priority]
    if tag:
        tasks = [t for t in tasks if tag in t["tags"]]
    
    return {
        "success": True,
        "tasks": tasks,
        "count": len(tasks)
    }


def update_task(
    storage: 'Storage',
    task_id: str,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_date: Optional[str] = None,
    description: Optional[str] = None,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Update an existing task.
    
    Args:
        storage: Storage instance
        task_id: ID of task to update
        status: New status
        priority: New priority
        due_date: New due date
        description: New description
        project: New project
        tags: New tags
    
    Returns:
        Updated task or error
    """
    tasks = storage.load_tasks()
    
    # Find the task
    task_index = None
    for i, task in enumerate(tasks):
        if task["id"] == task_id:
            task_index = i
            break
    
    if task_index is None:
        return {"success": False, "error": f"Task {task_id} not found"}
    
    # Update fields
    task = tasks[task_index]
    if status:
        task["status"] = status
        if status == "done" and not task["completed_at"]:
            task["completed_at"] = datetime.now().isoformat()
    if priority:
        task["priority"] = priority
    if due_date:
        task["due_date"] = due_date
    if description:
        task["description"] = description
    if project:
        task["project"] = project
    if tags:
        task["tags"] = tags
    
    storage.save_tasks(tasks)
    
    return {"success": True, "task": task}


def save_note(
    storage: 'Storage',
    title: str,
    content: str,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    linked_tasks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Save a new note.
    
    Args:
        storage: Storage instance
        title: Note title
        content: Note content
        project: Associated project
        tags: List of tags
        linked_tasks: List of linked task IDs
    
    Returns:
        Created note with ID
    """
    notes = storage.load_notes()
    
    note = {
        "id": f"note_{uuid.uuid4().hex[:8]}",
        "title": title,
        "content": content,
        "project": project,
        "tags": tags or [],
        "linked_tasks": linked_tasks or [],
        "created_at": datetime.now().isoformat()
    }
    
    notes.append(note)
    storage.save_notes(notes)
    
    return {"success": True, "note": note}


def search_notes(
    storage: 'Storage',
    query: Optional[str] = None,
    tags: Optional[List[str]] = None,
    project: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search notes by keyword or tag.
    
    Args:
        storage: Storage instance
        query: Search query (matches title and content)
        tags: Filter by tags
        project: Filter by project
    
    Returns:
        List of matching notes
    """
    notes = storage.load_notes()
    
    # Apply filters
    if query:
        query_lower = query.lower()
        notes = [
            n for n in notes
            if query_lower in n["title"].lower() or 
               query_lower in n["content"].lower()
        ]
    
    if tags:
        notes = [
            n for n in notes
            if any(tag in n["tags"] for tag in tags)
        ]
    
    if project:
        notes = [n for n in notes if n["project"] == project]
    
    return {
        "success": True,
        "notes": notes,
        "count": len(notes)
    }


def get_context(storage: 'Storage') -> Dict[str, Any]:
    """
    Get current user context.
    
    Args:
        storage: Storage instance
    
    Returns:
        User context including current project and preferences
    """
    context = storage.load_context()
    
    # Also include recent task and note statistics
    tasks = storage.load_tasks()
    notes = storage.load_notes()
    
    return {
        "success": True,
        "context": context,
        "stats": {
            "total_tasks": len(tasks),
            "active_tasks": len([t for t in tasks if t["status"] in ["todo", "in_progress"]]),
            "total_notes": len(notes)
        }
    }


def update_context(
    storage: 'Storage',
    current_project: Optional[str] = None,
    preferences: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Update user context or preferences.
    
    Args:
        storage: Storage instance
        current_project: Set current project
        preferences: Update preferences
    
    Returns:
        Updated context
    """
    context = storage.load_context()
    
    if current_project:
        context["current_project"] = current_project
        # Track recent projects
        if current_project not in context["recent_projects"]:
            context["recent_projects"].insert(0, current_project)
            context["recent_projects"] = context["recent_projects"][:5]
    
    if preferences:
        context["preferences"].update(preferences)
    
    storage.save_context(context)
    
    return {"success": True, "context": context}


if __name__ == "__main__":
    # Demo tools functionality
    from storage import Storage
    
    print("Tools Module Demo")
    print("=================\n")
    
    # Initialize storage
    storage = Storage("demo_data")
    
    # Create a task
    print("1. Creating a task...")
    result = create_task(
        storage,
        title="Write Chapter 44",
        description="Complete the productivity agent chapter",
        priority="high",
        project="Book Writing",
        tags=["writing", "ai-agents"]
    )
    print(f"✓ Created task: {result['task']['id']}")
    task_id = result['task']['id']
    
    # List tasks
    print("\n2. Listing all tasks...")
    result = list_tasks(storage)
    print(f"✓ Found {result['count']} task(s)")
    
    # Save a note
    print("\n3. Saving a note...")
    result = save_note(
        storage,
        title="Agent Design Patterns",
        content="Remember to emphasize simplicity and composability",
        project="Book Writing",
        tags=["ideas", "patterns"],
        linked_tasks=[task_id]
    )
    print(f"✓ Saved note: {result['note']['id']}")
    
    # Search notes
    print("\n4. Searching notes...")
    result = search_notes(storage, query="patterns")
    print(f"✓ Found {result['count']} note(s) matching 'patterns'")
    
    # Update context
    print("\n5. Updating context...")
    result = update_context(
        storage,
        current_project="Book Writing",
        preferences={"work_hours": "9am-5pm"}
    )
    print(f"✓ Current project: {result['context']['current_project']}")
    
    # Get context
    print("\n6. Getting context...")
    result = get_context(storage)
    print(f"✓ Active tasks: {result['stats']['active_tasks']}")
    print(f"✓ Total notes: {result['stats']['total_notes']}")
    
    # Update task status
    print("\n7. Marking task as done...")
    result = update_task(storage, task_id, status="done")
    print(f"✓ Task completed at: {result['task']['completed_at']}")
    
    print("\n✓ All tools working correctly!")
