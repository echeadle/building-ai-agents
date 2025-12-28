"""
Introduction to asyncio with parallel API calls.

Chapter 21: Parallelization - Implementation

This example demonstrates the fundamentals of async/await
and how to make parallel API calls to Claude.
"""

import asyncio
import os
import time
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Create the async client for parallel calls
async_client = anthropic.AsyncAnthropic()


async def get_response(prompt: str, label: str) -> dict:
    """
    Make an async API call and return the result with timing.
    
    Args:
        prompt: The prompt to send to Claude
        label: A label to identify this request
        
    Returns:
        Dictionary with label, response text, and execution time
    """
    start = time.time()
    
    response = await async_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    
    elapsed = time.time() - start
    return {
        "label": label,
        "response": response.content[0].text,
        "time": elapsed
    }


async def sequential_example():
    """
    Make API calls sequentially (one after another).
    
    This shows the baseline: total time = sum of all call times.
    """
    prompts = [
        ("What is 2+2? Reply with just the number.", "math"),
        ("What color is the sky? Reply in one word.", "color"),
        ("Name a planet. Reply with just the name.", "planet")
    ]
    
    print("Sequential execution:")
    print("-" * 40)
    
    start = time.time()
    results = []
    
    # Process one at a time
    for prompt, label in prompts:
        result = await get_response(prompt, label)
        results.append(result)
        print(f"  Completed: {label}")
    
    total_time = time.time() - start
    
    # Display results
    for result in results:
        response_preview = result['response'][:50].replace('\n', ' ')
        print(f"\n  {result['label']}: {response_preview}")
        print(f"    Time: {result['time']:.2f}s")
    
    print(f"\n  Total wall-clock time: {total_time:.2f}s")
    return total_time


async def parallel_example():
    """
    Make API calls in parallel (all at once).
    
    This demonstrates the power of asyncio: total time ≈ longest call time.
    """
    prompts = [
        ("What is 2+2? Reply with just the number.", "math"),
        ("What color is the sky? Reply in one word.", "color"),
        ("Name a planet. Reply with just the name.", "planet")
    ]
    
    print("\nParallel execution:")
    print("-" * 40)
    
    start = time.time()
    
    # Process all at once using asyncio.gather()
    results = await asyncio.gather(*[
        get_response(prompt, label) 
        for prompt, label in prompts
    ])
    
    total_time = time.time() - start
    
    # Display results
    for result in results:
        response_preview = result['response'][:50].replace('\n', ' ')
        print(f"\n  {result['label']}: {response_preview}")
        print(f"    Time: {result['time']:.2f}s")
    
    print(f"\n  Total wall-clock time: {total_time:.2f}s")
    sum_individual = sum(r['time'] for r in results)
    print(f"  Sum of individual times: {sum_individual:.2f}s")
    print(f"  Time saved: {sum_individual - total_time:.2f}s")
    
    return total_time


async def bounded_parallel_example():
    """
    Make parallel calls with a concurrency limit.
    
    This is important for avoiding rate limits when making many calls.
    """
    # More prompts to demonstrate bounded concurrency
    prompts = [
        (f"Count to {i}. Reply with just the numbers.", f"count_{i}")
        for i in range(1, 8)
    ]
    
    print("\nBounded parallel execution (max 3 concurrent):")
    print("-" * 40)
    
    # Semaphore limits concurrent executions
    semaphore = asyncio.Semaphore(3)
    
    async def bounded_call(prompt: str, label: str) -> dict:
        """Execute a call with semaphore limiting."""
        async with semaphore:
            print(f"  Starting: {label}")
            result = await get_response(prompt, label)
            print(f"  Finished: {label}")
            return result
    
    start = time.time()
    
    # All tasks start, but only 3 run at a time
    results = await asyncio.gather(*[
        bounded_call(prompt, label)
        for prompt, label in prompts
    ])
    
    total_time = time.time() - start
    
    print(f"\n  Completed {len(results)} calls in {total_time:.2f}s")
    print(f"  With unbounded parallel, would be ~{max(r['time'] for r in results):.2f}s")
    print(f"  With sequential, would be ~{sum(r['time'] for r in results):.2f}s")


async def main():
    """Run all examples and compare results."""
    print("=" * 60)
    print("ASYNCIO BASICS: Sequential vs Parallel API Calls")
    print("=" * 60)
    
    # Run sequential first
    seq_time = await sequential_example()
    
    # Run parallel
    par_time = await parallel_example()
    
    # Show improvement
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel time:   {par_time:.2f}s")
    print(f"Speedup:         {seq_time / par_time:.1f}x faster")
    
    # Demonstrate bounded concurrency
    print("\n" + "=" * 60)
    await bounded_parallel_example()


if __name__ == "__main__":
    asyncio.run(main())
