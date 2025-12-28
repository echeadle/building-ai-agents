"""
Exercise Solution: Deploy Your Agent to a Cloud Platform

Chapter 40: Deployment Patterns

This is a complete, deployable AI Agent API ready for cloud deployment.

Deployment steps for various platforms:

1. RAILWAY (Easiest):
   - Push to GitHub
   - Connect Railway to your repo
   - Add ANTHROPIC_API_KEY as environment variable
   - Deploy automatically

2. RENDER:
   - Push to GitHub
   - Create new Web Service
   - Connect to repo
   - Add environment variables
   - Deploy

3. GOOGLE CLOUD RUN:
   - Build: gcloud builds submit --tag gcr.io/PROJECT_ID/agent-api
   - Deploy: gcloud run deploy agent-api --image gcr.io/PROJECT_ID/agent-api
   - Set env vars in Cloud Console

4. AWS APP RUNNER:
   - Push to ECR or connect GitHub
   - Create new service
   - Configure environment variables
   - Deploy

Test your deployment:
   curl https://your-app.platform.com/health
   curl -X POST https://your-app.platform.com/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Hello!"}'
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
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import anthropic

# Load environment
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration from environment variables."""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    VERSION = os.getenv("VERSION", "1.0.0")
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Anthropic
    MODEL = os.getenv("MODEL", "claude-sonnet-4-20250514")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    @classmethod
    def is_production(cls) -> bool:
        return cls.ENVIRONMENT == "production"


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    stream: bool = False


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
    checks: dict


# ============================================================================
# AGENT
# ============================================================================

class DeployableAgent:
    """A simple, deployable AI agent."""
    
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
        """Process a chat message."""
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            messages = self.conversations[conversation_id].copy()
        else:
            conversation_id = str(uuid.uuid4())[:8]
            messages = []
        
        messages.append({"role": "user", "content": message})
        
        response = self.client.messages.create(
            model=Config.MODEL,
            max_tokens=Config.MAX_TOKENS,
            system=self.system_prompt,
            messages=messages
        )
        
        assistant_message = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_message})
        
        # Store (with cleanup)
        self.conversations[conversation_id] = messages
        if len(self.conversations) > 1000:
            oldest = list(self.conversations.keys())[0]
            del self.conversations[oldest]
        
        return (
            assistant_message,
            conversation_id,
            response.usage.input_tokens + response.usage.output_tokens
        )
    
    async def stream_chat(self, message: str) -> AsyncGenerator[str, None]:
        """Stream a chat response."""
        with self.client.messages.stream(
            model=Config.MODEL,
            max_tokens=Config.MAX_TOKENS,
            system=self.system_prompt,
            messages=[{"role": "user", "content": message}]
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
            
            final = stream.get_final_message()
            yield f"data: {json.dumps({'type': 'done', 'tokens': final.usage.input_tokens + final.usage.output_tokens})}\n\n"


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.requests: dict[str, list[float]] = {}
    
    def check(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > now - 60
        ]
        
        if len(self.requests[client_id]) >= self.rpm:
            return False
        
        self.requests[client_id].append(now)
        return True


# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    """Simple metrics collector."""
    
    def __init__(self):
        self.requests = 0
        self.errors = 0
        self.tokens = 0
    
    def record(self, tokens: int = 0, error: bool = False):
        self.requests += 1
        self.tokens += tokens
        if error:
            self.errors += 1


# ============================================================================
# APPLICATION
# ============================================================================

agent: Optional[DeployableAgent] = None
rate_limiter: Optional[RateLimiter] = None
metrics = Metrics()
start_time = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle."""
    global agent, rate_limiter, start_time
    
    start_time = time.time()
    agent = DeployableAgent()
    rate_limiter = RateLimiter(Config.RATE_LIMIT_PER_MINUTE)
    
    logger.info(f"Agent API started (env={Config.ENVIRONMENT})")
    yield
    logger.info("Agent API shutting down")


app = FastAPI(
    title="AI Agent API",
    description="Deployable AI Agent REST API",
    version=Config.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if Config.DEBUG or not Config.is_production() else None,
    redoc_url=None
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limit middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    
    client_id = request.client.host if request.client else "unknown"
    
    if rate_limiter and not rate_limiter.check(client_id):
        return Response(
            content='{"detail": "Rate limit exceeded"}',
            status_code=429,
            media_type="application/json"
        )
    
    return await call_next(request)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information and links."""
    return {
        "name": "AI Agent API",
        "version": Config.VERSION,
        "environment": Config.ENVIRONMENT,
        "docs": "/docs" if not Config.is_production() else None,
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "metrics": "GET /metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Comprehensive health check."""
    checks = {
        "agent": agent is not None,
        "api_key": bool(os.getenv("ANTHROPIC_API_KEY"))
    }
    
    all_healthy = all(checks.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        version=Config.VERSION,
        environment=Config.ENVIRONMENT,
        uptime_seconds=round(time.time() - start_time, 2),
        checks=checks
    )


@app.get("/health/live", tags=["Health"])
async def liveness():
    """Liveness probe - is the process running?"""
    return {"alive": True}


@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Readiness probe - can we handle requests?"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    return {"ready": True}


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Application metrics."""
    return {
        "uptime_seconds": round(time.time() - start_time, 2),
        "total_requests": metrics.requests,
        "total_errors": metrics.errors,
        "total_tokens": metrics.tokens,
        "error_rate": round(metrics.errors / max(metrics.requests, 1), 4)
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest):
    """
    Chat with the AI agent.
    
    - Set `stream: true` for streaming responses
    - Include `conversation_id` to continue a conversation
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Streaming response
    if request.stream:
        return StreamingResponse(
            agent.stream_chat(request.message),
            media_type="text/event-stream"
        )
    
    # Synchronous response
    start = time.perf_counter()
    
    try:
        response_text, conv_id, tokens = agent.chat(
            request.message, request.conversation_id
        )
        
        metrics.record(tokens=tokens)
        
        return ChatResponse(
            response=response_text,
            conversation_id=conv_id,
            tokens_used=tokens,
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2)
        )
        
    except anthropic.APIConnectionError:
        metrics.record(error=True)
        raise HTTPException(status_code=503, detail="AI service unavailable")
    except anthropic.RateLimitError:
        metrics.record(error=True)
        raise HTTPException(status_code=429, detail="AI service rate limited")
    except anthropic.APIError as e:
        metrics.record(error=True)
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/demo", response_class=HTMLResponse, tags=["Demo"])
async def demo():
    """Interactive demo page."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Demo</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .chat { background: #f5f5f5; padding: 20px; border-radius: 8px; min-height: 300px; margin: 20px 0; }
        .message { margin: 10px 0; padding: 10px; border-radius: 8px; }
        .user { background: #007bff; color: white; margin-left: 20%; }
        .assistant { background: white; margin-right: 20%; border: 1px solid #ddd; }
        input { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        .stats { color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <h1>🤖 AI Agent Demo</h1>
    <p>Chat with the deployed AI agent.</p>
    
    <div class="chat" id="chat"></div>
    
    <div>
        <input type="text" id="input" placeholder="Type your message..." onkeypress="if(event.key==='Enter')send()">
        <button onclick="send()">Send</button>
        <label><input type="checkbox" id="stream"> Stream</label>
    </div>
    
    <div class="stats" id="stats"></div>
    
    <script>
    let conversationId = null;
    
    async function send() {
        const input = document.getElementById('input');
        const chat = document.getElementById('chat');
        const stream = document.getElementById('stream').checked;
        const message = input.value.trim();
        
        if (!message) return;
        input.value = '';
        
        // Show user message
        chat.innerHTML += `<div class="message user">${message}</div>`;
        
        // Create assistant message placeholder
        const assistantDiv = document.createElement('div');
        assistantDiv.className = 'message assistant';
        assistantDiv.textContent = '...';
        chat.appendChild(assistantDiv);
        chat.scrollTop = chat.scrollHeight;
        
        try {
            if (stream) {
                // Streaming request
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message, stream: true, conversation_id: conversationId})
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                assistantDiv.textContent = '';
                
                while (true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value);
                    for (const line of text.split('\\n')) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            if (data.type === 'text') {
                                assistantDiv.textContent += data.content;
                            } else if (data.type === 'done') {
                                document.getElementById('stats').textContent = `Tokens: ${data.tokens}`;
                            }
                        }
                    }
                    chat.scrollTop = chat.scrollHeight;
                }
            } else {
                // Regular request
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message, conversation_id: conversationId})
                });
                
                const data = await response.json();
                assistantDiv.textContent = data.response;
                conversationId = data.conversation_id;
                document.getElementById('stats').textContent = 
                    `Tokens: ${data.tokens_used} | Time: ${data.processing_time_ms}ms | Conv: ${data.conversation_id}`;
            }
        } catch (e) {
            assistantDiv.textContent = `Error: ${e.message}`;
            assistantDiv.style.background = '#ffebee';
        }
        
        chat.scrollTop = chat.scrollHeight;
    }
    </script>
</body>
</html>
"""


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("AI AGENT API - READY FOR DEPLOYMENT")
    print("=" * 60)
    print()
    print(f"Environment: {Config.ENVIRONMENT}")
    print(f"Version: {Config.VERSION}")
    print(f"Debug: {Config.DEBUG}")
    print()
    print("Deployment checklist:")
    print("  ✓ Health endpoints: /health, /health/live, /health/ready")
    print("  ✓ Rate limiting enabled")
    print("  ✓ CORS configured")
    print("  ✓ Metrics endpoint: /metrics")
    print("  ✓ Demo page: /demo")
    print()
    print(f"Starting at http://{Config.HOST}:{Config.PORT}")
    print()
    
    uvicorn.run(
        "exercise_solution:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG
    )
