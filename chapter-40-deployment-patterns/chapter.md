---
chapter: 40
title: "Deployment Patterns"
part: 5
date: 2025-01-15
draft: false
---

# Chapter 40: Deployment Patterns

## Introduction

Your agent works perfectly on your laptop. It handles requests gracefully, manages costs efficiently, and responds quickly. Now you want others to use it. How do you get it from your development machine to a production environment where real users can interact with it?

This is the gap between building and shipping. Many excellent agents never reach users because their creators don't know how to deploy them. The leap from `python agent.py` to a production service feels daunting—there's Docker, REST APIs, environment variables, health checks, scaling, and a dozen other concerns that seem far removed from the agent code itself.

Here's the good news: deployment patterns for agents aren't fundamentally different from deploying any Python application. If you can build an agent, you can deploy one. This chapter demystifies production deployment by showing you concrete, copy-paste-ready patterns that work.

In the previous chapters, we optimized our agents for cost (Chapter 38) and latency (Chapter 39). Now we'll make them accessible. By the end of this chapter, you'll have a production-ready FastAPI service wrapping your agent, containerized with Docker, and ready to scale.

## Learning Objectives

By the end of this chapter, you will be able to:

- Wrap an AI agent in a REST API using FastAPI
- Implement background workers for long-running agent tasks
- Containerize agents with Docker for consistent deployment
- Configure environments properly across development, staging, and production
- Add health checks and monitoring endpoints
- Apply scaling strategies for handling increased load

## Deployment Options Overview

Before diving into code, let's understand the main deployment patterns and when to use each:

| Pattern | Best For | Latency | Complexity |
|---------|----------|---------|------------|
| REST API (Sync) | Quick queries, real-time chat | Low | Low |
| REST API (Streaming) | Long responses, chat interfaces | Low perceived | Medium |
| Background Workers | Long tasks, batch processing | High (async) | Medium |
| Serverless Functions | Sporadic traffic, cost optimization | Variable | Medium |
| WebSockets | Real-time bidirectional communication | Very low | High |

For most agent deployments, you'll start with a REST API. It's simple, well-understood, and works with any client. We'll focus primarily on this pattern while also covering background workers for tasks that take too long for a synchronous HTTP request.

## Agents as REST APIs with FastAPI

FastAPI is the modern choice for Python APIs. It's fast, includes automatic documentation, has built-in validation, and supports async operations natively. Let's wrap our agent in a FastAPI service.

### Basic Agent API

First, let's create a minimal API that exposes an agent:

```python
"""
Basic FastAPI wrapper for an AI agent.

Chapter 40: Deployment Patterns
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# ----- Request/Response Models -----

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "What is Python?",
                    "conversation_id": "conv_123"
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent's response")
    conversation_id: str = Field(..., description="Conversation ID for follow-ups")
    tokens_used: int = Field(..., description="Total tokens used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    uptime_seconds: float


# ----- Agent Implementation -----

class SimpleAgent:
    """
    A simple agent for demonstration purposes.
    
    In production, you'd use your full Agent class from Chapter 33.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = """You are a helpful AI assistant. Be concise and accurate.
When you don't know something, say so clearly."""
        
        # Simple in-memory conversation storage (use Redis/database in production)
        self.conversations: dict[str, list[dict]] = {}
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ) -> tuple[str, str, int]:
        """
        Process a chat message.
        
        Args:
            message: User's message
            conversation_id: Optional ID to continue a conversation
        
        Returns:
            Tuple of (response, conversation_id, tokens_used)
        """
        import uuid
        
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            messages = self.conversations[conversation_id].copy()
        else:
            conversation_id = str(uuid.uuid4())[:8]
            messages = []
        
        # Add user message
        messages.append({"role": "user", "content": message})
        
        # Call the API
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system_prompt,
            messages=messages
        )
        
        # Extract response text
        assistant_message = response.content[0].text
        
        # Update conversation history
        messages.append({"role": "assistant", "content": assistant_message})
        self.conversations[conversation_id] = messages
        
        # Limit conversation history to prevent memory bloat
        if len(self.conversations) > 1000:
            oldest = list(self.conversations.keys())[0]
            del self.conversations[oldest]
        
        total_tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return assistant_message, conversation_id, total_tokens


# ----- Application Setup -----

# Global state
start_time: float = 0
agent: Optional[SimpleAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    This runs on startup and shutdown.
    """
    global start_time, agent
    
    # Startup
    start_time = time.time()
    agent = SimpleAgent()
    print("🚀 Agent initialized and ready")
    
    yield
    
    # Shutdown
    print("👋 Shutting down agent")


# Create FastAPI app
app = FastAPI(
    title="AI Agent API",
    description="REST API for interacting with an AI agent",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware (configure appropriately for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Endpoints -----

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Use this for load balancers and container orchestration.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time() - start_time
    )


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Send a message to the agent and get a response.
    
    Optionally include a conversation_id to continue a previous conversation.
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    start = time.perf_counter()
    
    try:
        response_text, conv_id, tokens = agent.chat(
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        processing_time = (time.perf_counter() - start) * 1000
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            tokens_used=tokens,
            processing_time_ms=round(processing_time, 2)
        )
        
    except anthropic.APIConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to AI service"
        )
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limited. Please try again later."
        )
    except anthropic.APIStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI service error: {e.message}"
        )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "AI Agent API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }


# ----- Run directly for development -----

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "basic_api:app",  # module:app format for reload
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
```

Run this with `python basic_api.py` and visit `http://localhost:8000/docs` to see the automatic API documentation.

### Streaming Responses

For chat interfaces, streaming responses feel much faster. Here's how to add streaming support:

```python
"""
FastAPI agent with streaming responses.

Chapter 40: Deployment Patterns
"""

import os
import time
import json
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found")


class StreamingChatRequest(BaseModel):
    """Request for streaming chat."""
    message: str = Field(..., min_length=1, max_length=10000)
    system_prompt: Optional[str] = Field(None, max_length=5000)


app = FastAPI(title="Streaming Agent API", version="1.0.0")


async def generate_stream(
    message: str,
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from the agent.
    
    Yields Server-Sent Events (SSE) formatted data.
    """
    client = anthropic.Anthropic()
    
    system = system_prompt or "You are a helpful assistant."
    
    try:
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": message}]
        ) as stream:
            for text in stream.text_stream:
                # Format as Server-Sent Event
                data = json.dumps({"type": "text", "content": text})
                yield f"data: {data}\n\n"
            
            # Send completion event
            final = stream.get_final_message()
            completion_data = json.dumps({
                "type": "done",
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens
            })
            yield f"data: {completion_data}\n\n"
            
    except anthropic.APIError as e:
        error_data = json.dumps({"type": "error", "message": str(e)})
        yield f"data: {error_data}\n\n"


@app.post("/chat/stream")
async def stream_chat(request: StreamingChatRequest):
    """
    Stream a chat response using Server-Sent Events.
    
    The response is a stream of JSON objects, each prefixed with "data: ".
    
    Event types:
    - {"type": "text", "content": "..."} - Text chunk
    - {"type": "done", "input_tokens": N, "output_tokens": N} - Completion
    - {"type": "error", "message": "..."} - Error occurred
    """
    return StreamingResponse(
        generate_stream(request.message, request.system_prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

To consume this stream in JavaScript:

```javascript
const eventSource = new EventSource('/chat/stream', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: "Hello!"})
});

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'text') {
        document.getElementById('output').textContent += data.content;
    } else if (data.type === 'done') {
        console.log(`Tokens used: ${data.input_tokens + data.output_tokens}`);
    }
};
```

## Background Workers for Long Tasks

Some agent tasks take too long for synchronous HTTP requests. Users don't want to wait 60 seconds for a research report. Instead, accept the request immediately and process it in the background.

```python
"""
Background worker pattern for long-running agent tasks.

Chapter 40: Deployment Patterns

Uses Redis for task queue (install with: pip install redis rq)
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
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
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    """Record of a background task."""
    task_id: str
    status: TaskStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_message: str
    result: Optional[str] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None


# ----- In-Memory Task Storage -----
# In production, use Redis, PostgreSQL, or another persistent store

task_store: dict[str, TaskRecord] = {}


# ----- Agent Worker -----

def process_agent_task(task_id: str, message: str) -> None:
    """
    Process an agent task in the background.
    
    This function runs asynchronously and updates the task store.
    """
    # Update status to running
    task = task_store.get(task_id)
    if not task:
        return
    
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now(timezone.utc).isoformat()
    task_store[task_id] = task
    
    try:
        # Simulate long-running agent work
        client = anthropic.Anthropic()
        
        # Use a more elaborate prompt for "research" tasks
        system_prompt = """You are a research assistant. When given a topic:
1. Provide a comprehensive overview
2. List key facts and figures
3. Discuss different perspectives
4. Conclude with a summary

Take your time to be thorough."""
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Research the following topic thoroughly: {message}"}]
        )
        
        # Update with result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.result = response.content[0].text
        task.tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.error = str(e)
    
    task_store[task_id] = task


# ----- Request/Response Models -----

class TaskRequest(BaseModel):
    """Request to create a background task."""
    message: str = Field(..., min_length=1, max_length=10000)


class TaskCreatedResponse(BaseModel):
    """Response when a task is created."""
    task_id: str
    status: TaskStatus
    status_url: str


# ----- FastAPI Application -----

app = FastAPI(
    title="Agent Background Worker API",
    description="API for long-running agent tasks",
    version="1.0.0"
)


@app.post("/tasks", response_model=TaskCreatedResponse, status_code=202)
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a background task for the agent to process.
    
    Returns immediately with a task ID. Poll the status endpoint
    to check for completion.
    """
    # Create task record
    task_id = str(uuid.uuid4())[:12]
    
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
        status_url=f"/tasks/{task_id}"
    )


@app.get("/tasks/{task_id}", response_model=TaskRecord)
async def get_task_status(task_id: str):
    """
    Get the status and result of a task.
    
    Poll this endpoint until status is 'completed' or 'failed'.
    """
    task = task_store.get(task_id)
    
    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return task


@app.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    limit: int = 20
):
    """List recent tasks, optionally filtered by status."""
    tasks = list(task_store.values())
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    # Sort by creation time, newest first
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    
    return {"tasks": tasks[:limit], "total": len(tasks)}


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task record."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    del task_store[task_id]
    return {"deleted": task_id}


@app.get("/health")
async def health():
    pending = sum(1 for t in task_store.values() if t.status == TaskStatus.PENDING)
    running = sum(1 for t in task_store.values() if t.status == TaskStatus.RUNNING)
    
    return {
        "status": "healthy",
        "pending_tasks": pending,
        "running_tasks": running,
        "total_tasks": len(task_store)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

For higher throughput, replace the in-memory store and `BackgroundTasks` with a proper task queue like Redis Queue (RQ) or Celery:

```python
"""
Redis Queue (RQ) worker for agent tasks.

Install: pip install redis rq

Run worker: rq worker agent-tasks
"""

import os
from redis import Redis
from rq import Queue
from dotenv import load_dotenv
import anthropic

load_dotenv()


# Connect to Redis
redis_conn = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None)
)

# Create queue
task_queue = Queue("agent-tasks", connection=redis_conn)


def process_research_task(message: str) -> dict:
    """
    Worker function for research tasks.
    
    This runs in a separate RQ worker process.
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": message}]
    )
    
    return {
        "result": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }


# To enqueue a task from your API:
# job = task_queue.enqueue(process_research_task, message)
# return {"job_id": job.id}
```

## Containerization with Docker

Docker ensures your agent runs the same way everywhere—on your laptop, in CI/CD, and in production. Here's a production-ready Dockerfile:

```dockerfile
# Dockerfile for AI Agent API
# Build: docker build -t agent-api .
# Run: docker run -p 8000:8000 --env-file .env agent-api

# Use Python slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 agent && \
    useradd --uid 1000 --gid agent --shell /bin/bash --create-home agent

# Set working directory
WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=agent:agent . .

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

And the corresponding `requirements.txt`:

```text
# requirements.txt
anthropic>=0.18.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-dotenv>=1.0.0
pydantic>=2.5.0
```

For development, use Docker Compose to orchestrate multiple services:

```yaml
# docker-compose.yml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=debug
    volumes:
      # Mount code for hot-reloading in development
      - .:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  worker:
    build: .
    env_file:
      - .env
    environment:
      - REDIS_HOST=redis
    depends_on:
      - redis
    command: rq worker agent-tasks --url redis://redis:6379

volumes:
  redis-data:
```

Run with `docker-compose up` for the full stack.

## Environment Configuration

Different environments need different configurations. Here's a pattern for managing this cleanly:

```python
"""
Environment configuration management.

Chapter 40: Deployment Patterns
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic Settings automatically loads from environment variables
    and validates types. Add a .env file for local development.
    """
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=1)
    
    # Anthropic
    anthropic_api_key: str = Field(
        ...,  # Required
        description="Anthropic API key"
    )
    default_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default Claude model to use"
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens per response"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Max requests per minute per client"
    )
    
    # Redis (for background tasks)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: Optional[str] = Field(default=None)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")  # json or text
    
    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    # Timeouts
    request_timeout_seconds: int = Field(default=60)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Allow both ANTHROPIC_API_KEY and anthropic_api_key
        populate_by_name = True
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT
    
    def get_uvicorn_config(self) -> dict:
        """Get configuration dict for uvicorn."""
        config = {
            "host": self.api_host,
            "port": self.api_port,
            "workers": self.api_workers,
            "log_level": self.log_level.lower(),
        }
        
        if self.is_development:
            config["reload"] = True
            config["workers"] = 1  # Reload doesn't work with multiple workers
        
        return config


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings.
    
    Uses lru_cache to load settings only once.
    """
    return Settings()


# Usage in FastAPI
def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    settings = get_settings()
    
    app = FastAPI(
        title="AI Agent API",
        version="1.0.0",
        debug=settings.debug,
        docs_url="/docs" if not settings.is_production else None,  # Disable docs in prod
        redoc_url="/redoc" if not settings.is_production else None,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Example .env file for different environments:

# .env.development
"""
ENVIRONMENT=development
DEBUG=true
ANTHROPIC_API_KEY=sk-ant-...
LOG_LEVEL=DEBUG
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
"""

# .env.production
"""
ENVIRONMENT=production
DEBUG=false
ANTHROPIC_API_KEY=sk-ant-...
API_WORKERS=4
LOG_LEVEL=INFO
LOG_FORMAT=json
CORS_ORIGINS=["https://myapp.com", "https://www.myapp.com"]
RATE_LIMIT_REQUESTS=60
"""
```

## Health Checks and Monitoring

Production services need health checks for load balancers and monitoring. Here's a comprehensive health check implementation:

```python
"""
Health checks and monitoring endpoints.

Chapter 40: Deployment Patterns
"""

import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import anthropic


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Complete health check response."""
    status: HealthStatus
    timestamp: str
    version: str
    uptime_seconds: float
    components: list[ComponentHealth]


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    ready: bool
    message: str


# Track application start time
_start_time = time.time()


async def check_anthropic_health() -> ComponentHealth:
    """
    Check Anthropic API connectivity.
    
    Makes a minimal API call to verify the service is reachable.
    """
    start = time.perf_counter()
    
    try:
        client = anthropic.Anthropic()
        
        # Minimal API call - just count tokens
        response = client.messages.count_tokens(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "test"}]
        )
        
        latency = (time.perf_counter() - start) * 1000
        
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2)
        )
        
    except anthropic.AuthenticationError:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.UNHEALTHY,
            message="Invalid API key"
        )
    except anthropic.RateLimitError:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.DEGRADED,
            message="Rate limited"
        )
    except Exception as e:
        return ComponentHealth(
            name="anthropic_api",
            status=HealthStatus.UNHEALTHY,
            message=str(e)[:100]
        )


async def check_redis_health(
    host: str = "localhost",
    port: int = 6379
) -> ComponentHealth:
    """Check Redis connectivity if configured."""
    try:
        import redis
        
        start = time.perf_counter()
        r = redis.Redis(host=host, port=port, socket_timeout=5)
        r.ping()
        latency = (time.perf_counter() - start) * 1000
        
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2)
        )
        
    except ImportError:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Redis not configured"
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=str(e)[:100]
        )


def check_memory_health() -> ComponentHealth:
    """Check memory usage."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        
        if used_percent > 90:
            status = HealthStatus.UNHEALTHY
        elif used_percent > 80:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return ComponentHealth(
            name="memory",
            status=status,
            message=f"{used_percent:.1f}% used"
        )
        
    except ImportError:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.HEALTHY,
            message="psutil not installed"
        )


def setup_health_routes(app: FastAPI, version: str = "1.0.0"):
    """
    Add health check routes to a FastAPI app.
    
    Routes:
    - GET /health - Detailed health check
    - GET /health/live - Liveness probe (is the process running?)
    - GET /health/ready - Readiness probe (can it handle requests?)
    """
    
    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        tags=["Health"],
        summary="Detailed health check"
    )
    async def health_check(response: Response):
        """
        Comprehensive health check including all dependencies.
        
        Returns 200 if healthy, 503 if unhealthy or degraded.
        """
        # Run health checks concurrently
        components = await asyncio.gather(
            check_anthropic_health(),
            check_redis_health(),
            asyncio.to_thread(check_memory_health)
        )
        
        # Determine overall status
        statuses = [c.status for c in components]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            overall = HealthStatus.HEALTHY
        
        return HealthCheckResponse(
            status=overall,
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=version,
            uptime_seconds=round(time.time() - _start_time, 2),
            components=list(components)
        )
    
    @app.get(
        "/health/live",
        tags=["Health"],
        summary="Liveness probe"
    )
    async def liveness():
        """
        Simple liveness probe.
        
        Returns 200 if the process is running. Used by Kubernetes
        to know when to restart the container.
        """
        return {"alive": True}
    
    @app.get(
        "/health/ready",
        response_model=ReadinessResponse,
        tags=["Health"],
        summary="Readiness probe"
    )
    async def readiness(response: Response):
        """
        Readiness probe for load balancers.
        
        Returns 200 if ready to accept requests, 503 otherwise.
        Used by Kubernetes to know when to send traffic.
        """
        # Check critical dependencies
        anthropic_health = await check_anthropic_health()
        
        if anthropic_health.status == HealthStatus.UNHEALTHY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadinessResponse(
                ready=False,
                message=f"Anthropic API unhealthy: {anthropic_health.message}"
            )
        
        return ReadinessResponse(
            ready=True,
            message="Service is ready"
        )


# Metrics endpoint (basic implementation)
class Metrics:
    """Simple metrics collector."""
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0
    
    def record_request(self, tokens: int, latency_ms: float, error: bool = False):
        self.request_count += 1
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        if error:
            self.error_count += 1
    
    def get_stats(self) -> dict:
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(avg_latency, 2)
        }


def setup_metrics_route(app: FastAPI, metrics: Metrics):
    """Add metrics endpoint."""
    
    @app.get("/metrics", tags=["Monitoring"])
    async def get_metrics():
        """Get application metrics."""
        return {
            "uptime_seconds": round(time.time() - _start_time, 2),
            **metrics.get_stats()
        }
```

## Scaling Strategies

As traffic grows, you'll need to scale. Here are the key patterns:

### Horizontal Scaling

Run multiple instances behind a load balancer:

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - agent-api

  agent-api:
    build: .
    env_file:
      - .env
    deploy:
      replicas: 4  # Run 4 instances
      resources:
        limits:
          cpus: '1'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/live"]
      interval: 10s
      timeout: 5s
      retries: 3
```

With this nginx configuration:

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream agent_api {
        least_conn;  # Send to least loaded server
        server agent-api:8000;
    }
    
    server {
        listen 80;
        
        location / {
            proxy_pass http://agent_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Timeouts for long-running requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # Disable buffering for streaming
            proxy_buffering off;
        }
        
        location /health {
            proxy_pass http://agent_api/health;
            # Don't retry health checks
            proxy_next_upstream off;
        }
    }
}
```

### Kubernetes Deployment

For production-grade scaling, use Kubernetes:

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-api
  labels:
    app: agent-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-api
  template:
    metadata:
      labels:
        app: agent-api
    spec:
      containers:
      - name: agent-api
        image: your-registry/agent-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: anthropic-api-key
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            cpu: "250m"
            memory: "256Mi"
          limits:
            cpu: "1000m"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: agent-api
spec:
  selector:
    app: agent-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Rate Limiting

Protect your API from overload:

```python
"""
Rate limiting middleware.

Chapter 40: Deployment Patterns
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import FastAPI, Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production, use Redis-based limiting for distributed systems.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.requests: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if a client is allowed to make a request."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if t > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True
    
    def remaining(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        now = time.time()
        window_start = now - self.window_seconds
        
        current = len([
            t for t in self.requests[client_id]
            if t > window_start
        ])
        
        return max(0, self.requests_per_minute - current)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits."""
    
    def __init__(self, app: FastAPI, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Get client identifier (IP address or API key)
        client_id = request.headers.get(
            "X-API-Key",
            request.client.host if request.client else "unknown"
        )
        
        # Check rate limit
        if not self.limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.limiter.remaining(client_id))
        
        return response


# Usage:
# app = FastAPI()
# limiter = RateLimiter(requests_per_minute=60)
# app.add_middleware(RateLimitMiddleware, limiter=limiter)
```

## Complete Production-Ready API

Let's bring everything together into a complete, production-ready implementation:

```python
"""
Complete production-ready AI Agent API.

Chapter 40: Deployment Patterns

This brings together all the patterns:
- FastAPI with proper configuration
- Health checks and monitoring
- Rate limiting
- Streaming support
- Background tasks
- Proper error handling
"""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import anthropic

# Load environment
load_dotenv()


# ----- Configuration -----

class Settings(BaseSettings):
    """Application configuration."""
    
    environment: str = "development"
    debug: bool = False
    
    anthropic_api_key: str
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    
    rate_limit_per_minute: int = 60
    request_timeout: int = 60
    
    cors_origins: list[str] = ["*"]
    
    class Config:
        env_file = ".env"


settings = Settings()


# ----- Logging -----

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)


# ----- Models -----

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: int
    processing_time_ms: float


class TaskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    status_url: str


# ----- Agent -----

class ProductionAgent:
    """Production-ready agent with proper resource management."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.conversations: dict[str, list[dict]] = {}
        self.system_prompt = "You are a helpful AI assistant. Be concise and accurate."
    
    def chat(self, message: str, conversation_id: Optional[str] = None) -> tuple[str, str, int]:
        """Synchronous chat."""
        if conversation_id and conversation_id in self.conversations:
            messages = self.conversations[conversation_id].copy()
        else:
            conversation_id = str(uuid.uuid4())[:8]
            messages = []
        
        messages.append({"role": "user", "content": message})
        
        response = self.client.messages.create(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            system=self.system_prompt,
            messages=messages
        )
        
        assistant_message = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_message})
        
        self.conversations[conversation_id] = messages
        self._cleanup_old_conversations()
        
        return (
            assistant_message,
            conversation_id,
            response.usage.input_tokens + response.usage.output_tokens
        )
    
    async def stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        """Streaming chat."""
        with self.client.messages.stream(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": message}]
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
            
            final = stream.get_final_message()
            yield f"data: {json.dumps({'type': 'done', 'tokens': final.usage.input_tokens + final.usage.output_tokens})}\n\n"
    
    def _cleanup_old_conversations(self):
        """Remove old conversations to prevent memory leaks."""
        if len(self.conversations) > 1000:
            oldest = list(self.conversations.keys())[:100]
            for key in oldest:
                del self.conversations[key]


# ----- Rate Limiting -----

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = {}
    
    def check(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id] = [t for t in self.requests[client_id] if t > now - 60]
        
        if len(self.requests[client_id]) >= self.rpm:
            return False
        
        self.requests[client_id].append(now)
        return True


# ----- Metrics -----

class Metrics:
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.tokens = 0
    
    def record(self, tokens: int = 0, error: bool = False):
        self.requests += 1
        self.tokens += tokens
        if error:
            self.errors += 1


# ----- Application -----

agent: Optional[ProductionAgent] = None
rate_limiter = RateLimiter(settings.rate_limit_per_minute)
metrics = Metrics()
start_time = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, start_time
    start_time = time.time()
    agent = ProductionAgent()
    logger.info("Agent initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Middleware -----

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.check(client_id):
        return Response(
            content='{"detail": "Rate limit exceeded"}',
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": "60"}
        )
    
    return await call_next(request)


# ----- Endpoints -----

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - start_time, 2),
        "version": "1.0.0"
    }


@app.get("/health/live")
async def liveness():
    return {"alive": True}


@app.get("/health/ready")
async def readiness():
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {"ready": True}


@app.get("/metrics")
async def get_metrics():
    return {
        "requests": metrics.requests,
        "errors": metrics.errors,
        "tokens": metrics.tokens,
        "uptime": round(time.time() - start_time, 2)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    if request.stream:
        return StreamingResponse(
            agent.stream_chat(request.message),
            media_type="text/event-stream"
        )
    
    start = time.perf_counter()
    
    try:
        response_text, conv_id, tokens = agent.chat(
            request.message,
            request.conversation_id
        )
        
        metrics.record(tokens=tokens)
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            tokens_used=tokens,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2)
        )
        
    except anthropic.APIError as e:
        metrics.record(error=True)
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/")
async def root():
    return {
        "service": "AI Agent API",
        "version": "1.0.0",
        "docs": "/docs" if settings.debug else None,
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
```

## Common Pitfalls

**1. Not handling graceful shutdown**

When your service receives a shutdown signal, it should finish processing current requests before exiting. FastAPI's lifespan context manager helps with this, but long-running tasks need explicit handling.

**2. Storing state in memory without persistence**

In-memory conversation storage is lost when the container restarts. For production, use Redis or a database for any state that needs to survive restarts.

**3. Missing health checks**

Load balancers need health checks to route traffic properly. Without them, traffic goes to unhealthy instances. Always implement `/health/live` and `/health/ready` endpoints.

**4. Hardcoding configuration**

Never hardcode API keys, URLs, or settings. Use environment variables for everything that might change between environments. Pydantic Settings makes this easy.

**5. No rate limiting**

Without rate limits, a single user can exhaust your API quota or overwhelm your service. Implement rate limiting from day one.

## Practical Exercise

**Task:** Deploy your agent to a cloud platform

**Requirements:**

1. Create a Dockerfile for your agent API
2. Add health check endpoints (`/health`, `/health/live`, `/health/ready`)
3. Configure environment variables properly
4. Deploy to one of:
   - Railway (easiest)
   - Render
   - Google Cloud Run
   - AWS App Runner

**Steps:**

1. Create a `Dockerfile` using the template from this chapter
2. Create a `requirements.txt` with your dependencies
3. Push to GitHub
4. Connect your repository to your chosen platform
5. Add your `ANTHROPIC_API_KEY` as an environment secret
6. Deploy and test the health endpoints

**Verification:**

- `curl https://your-app.platform.com/health` returns healthy
- `curl -X POST https://your-app.platform.com/chat -d '{"message": "Hello!"}'` returns a response

**Solution:** See `code/exercise_solution.py` for a complete deployable example

## Key Takeaways

- **FastAPI is the modern choice** for Python APIs—it's fast, has great documentation, and supports async natively
- **Health checks are essential**—implement `/health/live` for liveness and `/health/ready` for readiness probes
- **Use background workers** for tasks longer than 30 seconds—users shouldn't wait for synchronous responses
- **Docker ensures consistency**—containerize your agent so it runs the same everywhere
- **Configuration should be external**—use environment variables, never hardcode secrets
- **Scale horizontally** behind a load balancer—multiple stateless instances beat one big server
- **Rate limiting protects you**—implement it before you need it

## What's Next

Your agent is deployed and serving users. But with production exposure comes security concerns. In the next chapter, **Security Considerations**, you'll learn to protect your agents from attacks: API key management, input injection, output security, rate limiting for abuse prevention, and the principle of least privilege for tools. Security isn't optional—it's essential for any production system.
