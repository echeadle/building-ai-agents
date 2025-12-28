"""
Basic FastAPI wrapper for an AI agent.

Chapter 40: Deployment Patterns

Run with: python basic_api.py
Then visit: http://localhost:8000/docs
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

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
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message"
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for context"
    )
    
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
        
        # Simple in-memory conversation storage
        # In production, use Redis or a database
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
        uptime_seconds=round(time.time() - start_time, 2)
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
        "basic_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
