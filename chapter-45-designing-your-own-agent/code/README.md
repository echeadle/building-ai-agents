# Chapter 45: Code Examples

This directory contains utilities and templates for designing your own agents.

## Files

### `design_template.py`
A comprehensive template for documenting your agent design from start to finish.

**Usage:**
```bash
# See the blank template
uv run python design_template.py template

# See a filled-out example (Meeting Notes Agent)
uv run python design_template.py example
```

**What it includes:**
- Problem analysis framework
- Requirements gathering template
- Tool design documentation
- Architecture planning
- Development roadmap
- Testing strategy
- Deployment planning

Use this template to plan your agent before writing any code!

---

### `development_checklist.py`
A progress tracking utility to manage your development through the six rungs.

**Usage:**
```python
from development_checklist import DevelopmentChecklist

# Create a new checklist for your project
checklist = DevelopmentChecklist("my-agent")

# Show current rung
checklist.show_current_rung()

# Complete a task (rung number, task index)
checklist.complete_task(1, 0)

# Show all rungs
checklist.show_all_rungs()

# Add custom task
checklist.add_custom_task(6, "Add email integration")
```

**Features:**
- Tracks all six development rungs
- Saves progress to JSON file
- Shows completion percentage
- Auto-advances to next rung when complete

Run the file directly to see an interactive demo:
```bash
uv run python development_checklist.py
```

---

### `fastapi_deployment.py`
A production-ready template for deploying your agent as a REST API.

**Features:**
- Health check endpoint
- Request/response validation
- Error handling
- Logging and observability
- Background task support (for long-running operations)
- CORS configuration
- Interactive API docs (Swagger)

**Setup:**

1. Install dependencies:
```bash
pip install fastapi uvicorn python-dotenv --break-system-packages
```

2. Set environment variables:
```bash
# Create .env file
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

3. Replace the `YourAgent` placeholder class with your actual agent implementation

4. Run the server:
```bash
uv run python fastapi_deployment.py
```

5. Visit the docs:
- Interactive API docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

**API Endpoints:**

- `GET /` - API information
- `GET /health` - Health check
- `POST /process` - Process input (synchronous)
- `POST /process-async` - Process input (asynchronous)

**Example request:**
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Your input here",
    "context": {"user_id": "user123"}
  }'
```

**Configuration via environment variables:**
```bash
# .env file
ANTHROPIC_API_KEY=your-key-here
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false
```

---

## How to Use These Files

### 1. Start with Design
Use `design_template.py` to plan your agent:
```bash
uv run python design_template.py template > my_agent_design.md
```
Fill out the template completely before writing code.

### 2. Track Development
Use `development_checklist.py` to manage progress:
```python
checklist = DevelopmentChecklist("my-agent")
# Update as you work through the rungs
```

### 3. Deploy Your Agent
Use `fastapi_deployment.py` as a starting point:
1. Copy the file to your project
2. Replace `YourAgent` with your implementation
3. Test locally
4. Deploy to your hosting platform

---

## Tips for Success

**Design Phase:**
- Spend time on the design document—it saves debugging time later
- Be realistic about scope—start with must-haves only
- Test your problem analysis—is an agent really needed?

**Development Phase:**
- Commit after each rung—you can always roll back
- Test with real data, not just examples
- Track costs from day one

**Deployment Phase:**
- Start simple (CLI or basic API)
- Add complexity only when needed
- Monitor everything: costs, latency, errors

---

## Additional Resources

- Chapter 42: Research Assistant Agent (complete example)
- Chapter 43: Code Analysis Agent (complete example)  
- Chapter 44: Personal Productivity Agent (complete example)
- Chapter 40: Deployment Patterns
- Appendix F: Security Checklist

---

## Next Steps

1. Choose a problem to solve with an agent
2. Fill out the design template
3. Create a development checklist
4. Start building (Rung 1: Hello World!)
5. Deploy using the FastAPI template when ready

**Good luck building your agent!** 🚀
