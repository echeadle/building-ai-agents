"""
FastAPI agent with streaming responses.

Chapter 40: Deployment Patterns

Streaming responses feel much faster because users see progress immediately.
This uses Server-Sent Events (SSE) for real-time streaming.

Run with: python streaming_api.py
Test with: curl -X POST http://localhost:8000/chat/stream \
    -H "Content-Type: application/json" \
    -d '{"message": "Write a haiku about coding"}'
"""

import os
import json
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found")


# ----- Request Models -----

class StreamingChatRequest(BaseModel):
    """Request for streaming chat."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message to send to the agent"
    )
    system_prompt: Optional[str] = Field(
        None,
        max_length=5000,
        description="Optional custom system prompt"
    )


class NonStreamingChatRequest(BaseModel):
    """Request for non-streaming chat (for comparison)."""
    message: str = Field(..., min_length=1, max_length=10000)


# ----- Application -----

app = FastAPI(
    title="Streaming Agent API",
    description="API with streaming response support",
    version="1.0.0"
)


# ----- Streaming Generator -----

async def generate_stream(
    message: str,
    system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response from the agent.
    
    Yields Server-Sent Events (SSE) formatted data.
    
    Event format:
    - {"type": "text", "content": "..."} - Text chunk
    - {"type": "done", "input_tokens": N, "output_tokens": N} - Completion
    - {"type": "error", "message": "..."} - Error occurred
    """
    client = anthropic.Anthropic()
    
    system = system_prompt or "You are a helpful assistant. Be concise."
    
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
            
            # Send completion event with token usage
            final = stream.get_final_message()
            completion_data = json.dumps({
                "type": "done",
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens
            })
            yield f"data: {completion_data}\n\n"
            
    except anthropic.APIConnectionError:
        error_data = json.dumps({
            "type": "error",
            "message": "Failed to connect to AI service"
        })
        yield f"data: {error_data}\n\n"
        
    except anthropic.RateLimitError:
        error_data = json.dumps({
            "type": "error",
            "message": "Rate limited. Please try again later."
        })
        yield f"data: {error_data}\n\n"
        
    except anthropic.APIError as e:
        error_data = json.dumps({
            "type": "error",
            "message": str(e)
        })
        yield f"data: {error_data}\n\n"


# ----- Endpoints -----

@app.post("/chat/stream")
async def stream_chat(request: StreamingChatRequest):
    """
    Stream a chat response using Server-Sent Events.
    
    The response is a stream of JSON objects, each prefixed with "data: ".
    
    Event types:
    - {"type": "text", "content": "..."} - Text chunk
    - {"type": "done", "input_tokens": N, "output_tokens": N} - Completion
    - {"type": "error", "message": "..."} - Error occurred
    
    Example JavaScript client:
    ```javascript
    const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: "Hello!"})
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        
        const text = decoder.decode(value);
        const lines = text.split('\\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                if (data.type === 'text') {
                    console.log(data.content);
                }
            }
        }
    }
    ```
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


@app.post("/chat")
async def non_streaming_chat(request: NonStreamingChatRequest):
    """
    Non-streaming chat for comparison.
    
    This waits for the full response before returning.
    """
    client = anthropic.Anthropic()
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": request.message}]
    )
    
    return {
        "response": response.content[0].text,
        "tokens_used": response.usage.input_tokens + response.usage.output_tokens
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Streaming Agent API",
        "endpoints": {
            "/chat/stream": "POST - Streaming chat (SSE)",
            "/chat": "POST - Non-streaming chat",
            "/health": "GET - Health check"
        }
    }


# ----- HTML Demo Page -----

@app.get("/demo")
async def demo_page():
    """Serve a simple HTML demo page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streaming Chat Demo</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            #output { 
                white-space: pre-wrap; 
                background: #f5f5f5; 
                padding: 20px; 
                min-height: 200px;
                border-radius: 8px;
            }
            input, button { padding: 10px; font-size: 16px; }
            input { width: 70%; }
            button { cursor: pointer; }
            .stats { color: #666; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Streaming Chat Demo</h1>
        <div>
            <input type="text" id="message" placeholder="Type your message..." value="Write a haiku about programming">
            <button onclick="sendMessage()">Send</button>
        </div>
        <h3>Response:</h3>
        <div id="output"></div>
        <div id="stats" class="stats"></div>
        
        <script>
        async function sendMessage() {
            const message = document.getElementById('message').value;
            const output = document.getElementById('output');
            const stats = document.getElementById('stats');
            
            output.textContent = '';
            stats.textContent = 'Streaming...';
            
            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const {done, value} = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                const lines = text.split('\\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'text') {
                            output.textContent += data.content;
                        } else if (data.type === 'done') {
                            stats.textContent = `Tokens: ${data.input_tokens} in, ${data.output_tokens} out`;
                        } else if (data.type === 'error') {
                            stats.textContent = `Error: ${data.message}`;
                        }
                    }
                }
            }
        }
        
        // Send on Enter key
        document.getElementById('message').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') sendMessage();
        });
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
