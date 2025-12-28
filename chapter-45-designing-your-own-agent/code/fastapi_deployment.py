"""
Minimal FastAPI Deployment Template

A production-ready template for deploying your agent as a web API.
Includes health checks, error handling, and observability.

Chapter 45: Designing Your Own Agent
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# =============================================================================
# API MODELS
# =============================================================================

class AgentRequest(BaseModel):
    """Request model for agent processing"""
    input: str = Field(..., description="Input text to process")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context for the agent"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "input": "Your input here",
                "context": {
                    "user_id": "user123",
                    "session_id": "session456"
                }
            }
        }


class AgentResponse(BaseModel):
    """Response model from agent"""
    result: str = Field(..., description="Agent's output")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the processing"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "result": "Agent's response here",
                "metadata": {
                    "tokens_used": 1234,
                    "cost": 0.05
                },
                "processing_time": 2.34
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    api_connected: bool


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Agent API",
    description="Production-ready API for your agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# YOUR AGENT CLASS (REPLACE WITH ACTUAL IMPLEMENTATION)
# =============================================================================

class YourAgent:
    """
    Replace this with your actual agent implementation
    
    This is a placeholder that shows the expected interface.
    """
    
    def __init__(self):
        """Initialize your agent"""
        logger.info("Initializing agent...")
        # Initialize your agent here
        # self.llm = ...
        # self.tools = ...
    
    def process(self, input_text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process input with your agent
        
        Args:
            input_text: Input to process
            context: Optional context dictionary
            
        Returns:
            Dictionary with 'result' and optional 'metadata'
        """
        # This is a placeholder - replace with your actual agent logic
        logger.info(f"Processing input: {input_text[:50]}...")
        
        # Simulate processing
        time.sleep(0.5)
        
        return {
            "result": f"Processed: {input_text}",
            "metadata": {
                "tokens_used": 100,
                "cost": 0.01
            }
        }


# Initialize agent (singleton)
try:
    agent = YourAgent()
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    raise


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns API status and connectivity information.
    """
    try:
        # Check if we can access the API
        api_connected = bool(api_key)
        
        return HealthResponse(
            status="healthy" if api_connected else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            api_connected=api_connected
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/process", response_model=AgentResponse, tags=["Agent"])
async def process_request(request: AgentRequest):
    """
    Process input with the agent
    
    Args:
        request: Agent request with input and optional context
        
    Returns:
        Agent response with result and metadata
    """
    start_time = time.time()
    
    try:
        # Log request
        logger.info(f"Received request: {request.input[:100]}...")
        
        # Validate input
        if not request.input or len(request.input.strip()) == 0:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        if len(request.input) > 100000:  # 100K character limit
            raise HTTPException(
                status_code=400,
                detail="Input too long (max 100,000 characters)"
            )
        
        # Process with agent
        result = agent.process(
            input_text=request.input,
            context=request.context
        )
        
        processing_time = time.time() - start_time
        
        # Log success
        logger.info(f"Request processed successfully in {processing_time:.2f}s")
        
        return AgentResponse(
            result=result["result"],
            metadata=result.get("metadata", {}),
            processing_time=processing_time
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/process-async", tags=["Agent"])
async def process_request_async(
    request: AgentRequest,
    background_tasks: BackgroundTasks
):
    """
    Process input asynchronously (for long-running tasks)
    
    Returns immediately with a task ID, process continues in background.
    Use this for tasks that take >30 seconds.
    
    Args:
        request: Agent request
        background_tasks: FastAPI background tasks
        
    Returns:
        Task ID for checking status later
    """
    import uuid
    task_id = str(uuid.uuid4())
    
    def process_in_background(task_id: str, input_text: str, context: Optional[Dict]):
        """Background processing function"""
        try:
            logger.info(f"Background task {task_id} started")
            result = agent.process(input_text, context)
            logger.info(f"Background task {task_id} completed")
            # Store result somewhere (database, cache, etc.)
            # For this example, we just log it
        except Exception as e:
            logger.error(f"Background task {task_id} failed: {e}")
    
    # Add to background tasks
    background_tasks.add_task(
        process_in_background,
        task_id,
        request.input,
        request.context
    )
    
    logger.info(f"Queued background task: {task_id}")
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Task queued for processing"
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught errors"""
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(
        status_code=500,
        detail="Internal server error"
    )


# =============================================================================
# STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    logger.info("API starting up...")
    logger.info(f"Docs available at: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    logger.info("API shutting down...")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Reload: {reload}")
    
    uvicorn.run(
        "fastapi_deployment:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
