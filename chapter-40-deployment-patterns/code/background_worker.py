"""
Background worker pattern for long-running agent tasks.

Chapter 40: Deployment Patterns

Some agent tasks take too long for synchronous HTTP requests.
This pattern accepts requests immediately and processes them in the background.

Run with: python background_worker.py
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found")


# ----- Task Status -----

class TaskStatus(str, Enum):
    """Status of a background task."""
    PENDING = "pending"      # Task created but not started
    RUNNING = "running"      # Task is being processed
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"        # Task failed with an error


class TaskRecord(BaseModel):
    """Complete record of a background task."""
    task_id: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_message: str
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time_ms: Optional[float] = None


# ----- Task Storage -----
# In production, use Redis, PostgreSQL, or another persistent store
# This in-memory storage is lost on restart

task_store: dict[str, TaskRecord] = {}


# ----- Agent Worker -----

def process_agent_task(task_id: str, message: str) -> None:
    """
    Process an agent task in the background.
    
    This function runs asynchronously and updates the task store.
    It simulates a long-running research task.
    """
    # Get the task record
    task = task_store.get(task_id)
    if not task:
        return
    
    # Update status to running
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now(timezone.utc).isoformat()
    task_store[task_id] = task
    
    start_time = time.perf_counter()
    
    try:
        client = anthropic.Anthropic()
        
        # Use a more elaborate prompt for "research" tasks
        system_prompt = """You are a research assistant. When given a topic:

1. Provide a comprehensive overview
2. List key facts and figures
3. Discuss different perspectives
4. Include relevant examples
5. Conclude with a summary

Take your time to be thorough and accurate."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"Research the following topic thoroughly:\n\n{message}"
            }]
        )
        
        # Update with successful result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.result = response.content[0].text
        task.tokens_used = response.usage.input_tokens + response.usage.output_tokens
        task.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
    except anthropic.APIError as e:
        # Update with error
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.error = f"API Error: {str(e)}"
        task.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
    except Exception as e:
        # Update with unexpected error
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.error = f"Unexpected error: {str(e)}"
        task.processing_time_ms = (time.perf_counter() - start_time) * 1000
    
    # Save updated task
    task_store[task_id] = task


# ----- Request/Response Models -----

class TaskRequest(BaseModel):
    """Request to create a background task."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The research topic or task description"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Research the history of artificial intelligence"}
            ]
        }
    }


class TaskCreatedResponse(BaseModel):
    """Response when a task is successfully created."""
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Initial task status")
    status_url: str = Field(..., description="URL to check task status")
    message: str = Field(..., description="Instructions for the user")


# ----- FastAPI Application -----

app = FastAPI(
    title="Agent Background Worker API",
    description="""
    API for long-running agent tasks.
    
    Use this when tasks might take more than a few seconds:
    1. POST /tasks to create a task
    2. Poll GET /tasks/{task_id} until status is 'completed' or 'failed'
    3. Retrieve the result from the response
    """,
    version="1.0.0"
)


# ----- Endpoints -----

@app.post(
    "/tasks",
    response_model=TaskCreatedResponse,
    status_code=202,  # 202 Accepted - request accepted for processing
    tags=["Tasks"]
)
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a background task for the agent to process.
    
    Returns immediately with a task ID. The actual processing
    happens in the background.
    
    Poll the status endpoint to check for completion.
    """
    # Generate unique task ID
    task_id = str(uuid.uuid4())[:12]
    
    # Create task record
    task = TaskRecord(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now(timezone.utc).isoformat(),
        input_message=request.message
    )
    task_store[task_id] = task
    
    # Schedule background processing
    background_tasks.add_task(process_agent_task, task_id, request.message)
    
    return TaskCreatedResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        status_url=f"/tasks/{task_id}",
        message="Task created. Poll the status_url to check for completion."
    )


@app.get(
    "/tasks/{task_id}",
    response_model=TaskRecord,
    tags=["Tasks"]
)
async def get_task_status(task_id: str):
    """
    Get the status and result of a task.
    
    Poll this endpoint until status is 'completed' or 'failed'.
    
    Typical polling strategy:
    - Poll every 1-2 seconds
    - Implement exponential backoff for long tasks
    - Set a maximum timeout on the client side
    """
    task = task_store.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return task


@app.get("/tasks", tags=["Tasks"])
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 20,
    offset: int = 0
):
    """
    List tasks, optionally filtered by status.
    
    Useful for monitoring and debugging.
    """
    tasks = list(task_store.values())
    
    # Filter by status if specified
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    # Sort by creation time, newest first
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    
    # Paginate
    total = len(tasks)
    tasks = tasks[offset:offset + limit]
    
    return {
        "tasks": tasks,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@app.delete("/tasks/{task_id}", tags=["Tasks"])
async def delete_task(task_id: str):
    """
    Delete a task record.
    
    Note: This only removes the record. It does not cancel
    a running task (that would require additional infrastructure).
    """
    if task_id not in task_store:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    del task_store[task_id]
    return {"deleted": task_id, "message": "Task record deleted"}


@app.get("/health", tags=["System"])
async def health():
    """
    Health check with task statistics.
    """
    pending = sum(1 for t in task_store.values() if t.status == TaskStatus.PENDING)
    running = sum(1 for t in task_store.values() if t.status == TaskStatus.RUNNING)
    completed = sum(1 for t in task_store.values() if t.status == TaskStatus.COMPLETED)
    failed = sum(1 for t in task_store.values() if t.status == TaskStatus.FAILED)
    
    return {
        "status": "healthy",
        "task_stats": {
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": len(task_store)
        }
    }


@app.get("/", tags=["System"])
async def root():
    """API information."""
    return {
        "name": "Agent Background Worker API",
        "version": "1.0.0",
        "usage": {
            "1": "POST /tasks with {'message': 'your task'}",
            "2": "Poll GET /tasks/{task_id} until completed",
            "3": "Result is in the 'result' field"
        }
    }


# ----- Demo: Polling Example -----

@app.get("/demo", tags=["Demo"])
async def demo_page():
    """Serve a demo page showing how to poll for results."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Background Task Demo</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .task { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
            .pending { border-left: 4px solid #ffa500; }
            .running { border-left: 4px solid #2196f3; }
            .completed { border-left: 4px solid #4caf50; }
            .failed { border-left: 4px solid #f44336; }
            input, button { padding: 10px; font-size: 16px; margin: 5px; }
            input { width: 60%; }
            pre { white-space: pre-wrap; background: #fff; padding: 10px; }
        </style>
    </head>
    <body>
        <h1>Background Task Demo</h1>
        <div>
            <input type="text" id="message" placeholder="Enter research topic..." 
                   value="The history of Python programming language">
            <button onclick="createTask()">Create Task</button>
        </div>
        <div id="tasks"></div>
        
        <script>
        let activeTasks = {};
        
        async function createTask() {
            const message = document.getElementById('message').value;
            
            const response = await fetch('/tasks', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            });
            
            const data = await response.json();
            activeTasks[data.task_id] = data;
            renderTasks();
            pollTask(data.task_id);
        }
        
        async function pollTask(taskId) {
            const response = await fetch(`/tasks/${taskId}`);
            const task = await response.json();
            activeTasks[taskId] = task;
            renderTasks();
            
            if (task.status === 'pending' || task.status === 'running') {
                setTimeout(() => pollTask(taskId), 1000);
            }
        }
        
        function renderTasks() {
            const container = document.getElementById('tasks');
            container.innerHTML = '';
            
            for (const task of Object.values(activeTasks)) {
                const div = document.createElement('div');
                div.className = `task ${task.status}`;
                
                let content = `<strong>Task ${task.task_id}</strong> - ${task.status.toUpperCase()}<br>`;
                content += `Input: ${task.input_message.substring(0, 100)}...<br>`;
                
                if (task.status === 'completed') {
                    content += `<pre>${task.result}</pre>`;
                    content += `Tokens: ${task.tokens_used}, Time: ${Math.round(task.processing_time_ms)}ms`;
                } else if (task.status === 'failed') {
                    content += `<pre style="color:red">${task.error}</pre>`;
                }
                
                div.innerHTML = content;
                container.appendChild(div);
            }
        }
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
