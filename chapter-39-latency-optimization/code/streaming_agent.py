"""
Streaming responses for improved perceived latency.

Chapter 39: Latency Optimization

Streaming provides immediate feedback to users, making the agent
feel much faster even when total response time is unchanged.
"""

import os
import sys
import time
from typing import Any, Callable, Generator, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class StreamMetrics:
    """Metrics from a streaming response."""
    time_to_first_token_ms: float
    total_duration_ms: float
    tokens_generated: int
    tokens_per_second: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": round(self.tokens_per_second, 1)
        }


class StreamingAgent:
    """
    An agent that streams responses for better perceived latency.
    
    Streaming provides immediate feedback to users, making the agent
    feel much faster even when total response time is unchanged.
    
    Usage:
        agent = StreamingAgent()
        
        # Stream to console
        for chunk in agent.stream("Tell me about Python"):
            print(chunk, end="", flush=True)
        
        # Or use callbacks
        agent.stream_with_callback(
            "Tell me about Python",
            on_text=lambda text: print(text, end=""),
            on_complete=lambda metrics: print(f"\\n\\nDone in {metrics.total_duration_ms}ms")
        )
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the streaming agent.
        
        Args:
            model: Claude model to use
            system_prompt: Optional system prompt
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful assistant."
    
    def stream(
        self,
        user_message: str,
        max_tokens: int = 1024
    ) -> Generator[str, None, StreamMetrics]:
        """
        Stream a response, yielding text chunks as they arrive.
        
        Args:
            user_message: The user's input
            max_tokens: Maximum tokens to generate
        
        Yields:
            Text chunks as they're generated
        
        Returns:
            StreamMetrics with timing information
        """
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens_generated = 0
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}]
        ) as stream:
            for text in stream.text_stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                tokens_generated += 1  # Approximate: 1 chunk ≈ 1 token
                yield text
        
        end_time = time.perf_counter()
        total_duration = (end_time - start_time) * 1000
        ttft = (first_token_time - start_time) * 1000 if first_token_time else total_duration
        
        return StreamMetrics(
            time_to_first_token_ms=ttft,
            total_duration_ms=total_duration,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_generated / (total_duration / 1000) if total_duration > 0 else 0
        )
    
    def stream_with_callback(
        self,
        user_message: str,
        on_text: Callable[[str], None],
        on_complete: Optional[Callable[[StreamMetrics], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        max_tokens: int = 1024
    ) -> Optional[StreamMetrics]:
        """
        Stream a response using callbacks.
        
        This is useful for integrating with async frameworks or UI updates.
        
        Args:
            user_message: The user's input
            on_text: Called for each text chunk
            on_complete: Called when streaming finishes
            on_error: Called if an error occurs
            max_tokens: Maximum tokens to generate
        
        Returns:
            StreamMetrics if successful, None if error
        """
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        tokens_generated = 0
        
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}]
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    
                    tokens_generated += 1
                    on_text(text)
            
            end_time = time.perf_counter()
            total_duration = (end_time - start_time) * 1000
            ttft = (first_token_time - start_time) * 1000 if first_token_time else total_duration
            
            metrics = StreamMetrics(
                time_to_first_token_ms=ttft,
                total_duration_ms=total_duration,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_generated / (total_duration / 1000) if total_duration > 0 else 0
            )
            
            if on_complete:
                on_complete(metrics)
            
            return metrics
            
        except Exception as e:
            if on_error:
                on_error(e)
            else:
                raise
            return None
    
    def stream_with_tools(
        self,
        user_message: str,
        tools: list[dict[str, Any]],
        tool_executor: Callable[[str, dict], str],
        max_tokens: int = 1024,
        max_iterations: int = 10
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream responses with tool use support.
        
        This is more complex because tool calls interrupt streaming.
        We yield events to indicate what's happening.
        
        Args:
            user_message: The user's input
            tools: Tool definitions
            tool_executor: Function to execute tools (name, input) -> result
            max_tokens: Maximum tokens per response
            max_iterations: Maximum tool use iterations
        
        Yields:
            Events with different types:
            - {"type": "text", "content": "..."}
            - {"type": "tool_call", "name": "...", "input": {...}}
            - {"type": "tool_result", "name": "...", "result": "..."}
            - {"type": "complete", "metrics": {...}}
        """
        messages = [{"role": "user", "content": user_message}]
        iteration = 0
        total_start = time.perf_counter()
        
        while iteration < max_iterations:
            iteration += 1
            
            # Stream the response
            collected_content = []
            tool_uses = []
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=self.system_prompt,
                tools=tools,
                messages=messages
            ) as stream:
                current_tool_use = None
                
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_start':
                            if hasattr(event, 'content_block'):
                                if event.content_block.type == 'tool_use':
                                    current_tool_use = {
                                        'id': event.content_block.id,
                                        'name': event.content_block.name,
                                        'input': ''
                                    }
                        elif event.type == 'content_block_delta':
                            if hasattr(event, 'delta'):
                                if hasattr(event.delta, 'text'):
                                    yield {"type": "text", "content": event.delta.text}
                                    collected_content.append({
                                        "type": "text",
                                        "text": event.delta.text
                                    })
                                elif hasattr(event.delta, 'partial_json'):
                                    if current_tool_use:
                                        current_tool_use['input'] += event.delta.partial_json
                        elif event.type == 'content_block_stop':
                            if current_tool_use:
                                try:
                                    import json
                                    current_tool_use['input'] = json.loads(current_tool_use['input'])
                                except:
                                    current_tool_use['input'] = {}
                                tool_uses.append(current_tool_use)
                                current_tool_use = None
                
                final_message = stream.get_final_message()
            
            # Check if we're done
            if final_message.stop_reason == "end_turn":
                total_duration = (time.perf_counter() - total_start) * 1000
                yield {
                    "type": "complete",
                    "metrics": {
                        "total_duration_ms": round(total_duration, 2),
                        "iterations": iteration
                    }
                }
                return
            
            # Process tool uses
            if final_message.stop_reason == "tool_use" and tool_uses:
                messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })
                
                tool_results = []
                for tool_use in tool_uses:
                    yield {
                        "type": "tool_call",
                        "name": tool_use['name'],
                        "input": tool_use['input']
                    }
                    
                    result = tool_executor(tool_use['name'], tool_use['input'])
                    
                    yield {
                        "type": "tool_result",
                        "name": tool_use['name'],
                        "result": result
                    }
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use['id'],
                        "content": result
                    })
                
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                break
        
        total_duration = (time.perf_counter() - total_start) * 1000
        yield {
            "type": "complete",
            "metrics": {
                "total_duration_ms": round(total_duration, 2),
                "iterations": iteration,
                "max_iterations_reached": iteration >= max_iterations
            }
        }


class ProgressIndicator:
    """
    Shows progress during long operations.
    
    When streaming isn't possible (e.g., during tool execution),
    use progress indicators to maintain user engagement.
    """
    
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(self, message: str = "Processing"):
        """
        Initialize the progress indicator.
        
        Args:
            message: Default message to display
        """
        self.message = message
        self.frame = 0
        self.start_time = time.perf_counter()
        self._running = False
    
    def update(self, status: Optional[str] = None) -> str:
        """
        Get the next frame of the progress indicator.
        
        Args:
            status: Optional status message to override default
        
        Returns:
            Formatted progress string
        """
        self.frame = (self.frame + 1) % len(self.SPINNER_FRAMES)
        elapsed = time.perf_counter() - self.start_time
        
        msg = status or self.message
        return f"\r{self.SPINNER_FRAMES[self.frame]} {msg} ({elapsed:.1f}s)"
    
    def print_update(self, status: Optional[str] = None) -> None:
        """Print progress update to console."""
        sys.stdout.write(self.update(status))
        sys.stdout.flush()
    
    def complete(self, final_message: str = "Done") -> str:
        """
        Mark the operation as complete.
        
        Args:
            final_message: Message to display on completion
        
        Returns:
            Formatted completion string
        """
        elapsed = time.perf_counter() - self.start_time
        return f"\r✓ {final_message} ({elapsed:.1f}s)\n"
    
    def print_complete(self, final_message: str = "Done") -> None:
        """Print completion message to console."""
        sys.stdout.write(self.complete(final_message))
        sys.stdout.flush()
    
    def reset(self) -> None:
        """Reset the progress indicator."""
        self.frame = 0
        self.start_time = time.perf_counter()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING RESPONSE DEMO")
    print("=" * 60)
    print()
    
    agent = StreamingAgent()
    
    print("Streaming response:\n")
    print("-" * 40)
    
    # Simple streaming
    gen = agent.stream("Write a haiku about programming.")
    for chunk in gen:
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Small delay to see streaming effect
    
    print("\n" + "-" * 40)
    
    # With callbacks and metrics
    print("\n\nWith metrics tracking:\n")
    print("-" * 40)
    
    def on_text(text: str) -> None:
        print(text, end="", flush=True)
    
    def on_complete(metrics: StreamMetrics) -> None:
        print(f"\n\n--- Metrics ---")
        print(f"Time to first token: {metrics.time_to_first_token_ms:.2f}ms")
        print(f"Total duration: {metrics.total_duration_ms:.2f}ms")
        print(f"Tokens/second: {metrics.tokens_per_second:.1f}")
    
    agent.stream_with_callback(
        "Explain async programming in one paragraph.",
        on_text=on_text,
        on_complete=on_complete
    )
    
    print("-" * 40)
    
    # Progress indicator demo
    print("\n\nProgress Indicator Demo:")
    print("-" * 40)
    
    progress = ProgressIndicator("Processing request")
    
    for i in range(20):
        progress.print_update()
        time.sleep(0.1)
    
    progress.print_complete("Request completed successfully")
