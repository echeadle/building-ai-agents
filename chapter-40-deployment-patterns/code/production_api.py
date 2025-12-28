"""
Complete production-ready AI Agent API.

Chapter 40: Deployment Patterns

This brings together all deployment patterns:
- FastAPI with proper configuration
- Health checks and monitoring
- Rate limiting
- Streaming support
- Background tasks
- Proper error handling
- Metrics collection

Run with: python production_api.py
"""

import os
import time
import uuid
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import anthropic

# Load environment
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application configuration from environment."""
    
    # Environment
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    version: str = Field(default="1.0.0")
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    
    # Anthropic
    anthropic_api_key: str
    default_model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=1024)
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=60)
    
    # Timeouts
    request_timeout: int = Field(default=60)
    
    # CORS
    cors_origins: list[str] = Field(default=["*"])
    
    class Config:
        env_file = ".env"
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    """Configure structured logging."""
    log_format = (
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        if settings.is_production
        else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format=log_format
    )

setup_logging()
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    stream: bool = False
    
    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "Hello!", "stream": False}]
        }
    }


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    conversation_id: str
    tokens_used: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    uptime_seconds: float


# ============================================================================
# AGENT
# ============================================================================

class ProductionAgent:
    """Production-ready agent with proper resource management."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.conversations: dict[str, list[dict]] = {}
        self.system_prompt = """You are a helpful AI assistant. 
Be concise, accurate, and friendly.
If you don't know something, say so clearly."""
    
    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None
    ) -> tuple[str, str, int]:
        """Synchronous chat."""
        # Get or create conversation
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
        
        # Store conversation
        self.conversations[conversation_id] = messages
        self._cleanup_old_conversations()
        
        return (
            assistant_message,
            conversation_id,
            response.usage.input_tokens + response.usage.output_tokens
        )
    
    async def stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        """Streaming chat using Server-Sent Events."""
        with self.client.messages.stream(
            model=settings.default_model,
            max_tokens=settings.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": message}]
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
            
            final = stream.get_final_message()
            tokens = final.usage.input_tokens + final.usage.output_tokens
            yield f"data: {json.dumps({'type': 'done', 'tokens': tokens})}\n\n"
    
    def _cleanup_old_conversations(self):
        """Remove old conversations to prevent memory leaks."""
        max_conversations = 1000
        if len(self.conversations) > max_conversations:
            # Remove oldest 10%
            to_remove = list(self.conversations.keys())[:max_conversations // 10]
            for key in to_remove:
                del self.conversations[key]


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.requests: dict[str, list[float]] = {}
    
    def check(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id]
            if t > now - 60
        ]
        
        if len(self.requests[client_id]) >= self.rpm:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def remaining(self, client_id: str) -> int:
        now = time.time()
        current = len([
            t for t in self.requests.get(client_id, [])
            if t > now - 60
        ])
        return max(0, self.rpm - current)


# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    """Simple metrics collector."""
    
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.tokens = 0
        self.total_latency_ms = 0.0
    
    def record(self, tokens: int = 0, latency_ms: float = 0, error: bool = False):
        self.requests += 1
        self.tokens += tokens
        self.total_latency_ms += latency_ms
        if error:
            self.errors += 1
    
    def get_stats(self) -> dict:
        return {
            "total_requests": self.requests,
            "total_errors": self.errors,
            "error_rate": round(self.errors / max(self.requests, 1), 4),
            "total_tokens": self.tokens,
            "avg_latency_ms": round(
                self.total_latency_ms / max(self.requests, 1), 2
            )
        }


# ============================================================================
# APPLICATION
# ============================================================================

# Global state
agent: Optional[ProductionAgent] = None
rate_limiter: Optional[RateLimiter] = None
metrics = Metrics()
start_time = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global agent, rate_limiter, start_time
    
    # Startup
    start_time = time.time()
    agent = ProductionAgent()
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    
    logger.info(f"Agent API started (env={settings.environment})")
    
    yield
    
    # Shutdown
    logger.info("Agent API shutting down")


# Create app
app = FastAPI(
    title="AI Agent API",
    description="Production-ready AI Agent REST API",
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url=None
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    # Skip health endpoints
    if request.url.path.startswith("/health"):
        return await call_next(request)
    
    if not settings.rate_limit_enabled:
        return await call_next(request)
    
    # Get client ID
    client_id = request.client.host if request.client else "unknown"
    
    if not rate_limiter.check(client_id):
        return Response(
            content='{"detail": "Rate limit exceeded"}',
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": "60"}
        )
    
    response = await call_next(request)
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.rpm)
    response.headers["X-RateLimit-Remaining"] = str(rate_limiter.remaining(client_id))
    
    return response


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "AI Agent API",
        "version": settings.version,
        "environment": settings.environment,
        "endpoints": {
            "/chat": "POST - Chat with the agent",
            "/health": "GET - Health check",
            "/metrics": "GET - Application metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - start_time, 2)
    )


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Liveness probe for Kubernetes."""
    return {"alive": True}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Readiness probe for Kubernetes."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return {"ready": True}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Application metrics."""
    return {
        "uptime_seconds": round(time.time() - start_time, 2),
        **metrics.get_stats()
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Chat with the AI agent.
    
    Set `stream: true` for streaming responses (Server-Sent Events).
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            agent.stream_chat(request.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    
    # Synchronous response
    start = time.perf_counter()
    
    try:
        response_text, conv_id, tokens = agent.chat(
            request.message,
            request.conversation_id
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record(tokens=tokens, latency_ms=latency_ms)
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            tokens_used=tokens,
            processing_time_ms=round(latency_ms, 2)
        )
        
    except anthropic.APIConnectionError:
        metrics.record(error=True)
        logger.error("Failed to connect to Anthropic API")
        raise HTTPException(status_code=503, detail="AI service unavailable")
        
    except anthropic.RateLimitError:
        metrics.record(error=True)
        logger.warning("Anthropic rate limit hit")
        raise HTTPException(status_code=429, detail="AI service rate limited")
        
    except anthropic.APIError as e:
        metrics.record(error=True)
        logger.error(f"Anthropic API error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("PRODUCTION AI AGENT API")
    print("=" * 60)
    print()
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Rate limit: {settings.rate_limit_per_minute}/min")
    print()
    print(f"Starting server at http://{settings.api_host}:{settings.api_port}")
    print()
    
    uvicorn.run(
        "production_api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
