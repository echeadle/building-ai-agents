# Chapter 44 Code: Personal Productivity Agent

This directory contains the complete implementation of a personal productivity agent.

## Files

### Core Implementation

- **`storage.py`** - JSON-based storage layer for tasks, notes, and context
- **`tools.py`** - All productivity tools (create_task, save_note, etc.)
- **`agent.py`** - Main ProductivityAgent class with agentic loop
- **`.env.example`** - Example environment variables file

### Exercise

- **`exercise_solution.py`** - Complete solution adding time tracking feature

## Setup

1. **Install dependencies:**
   ```bash
   uv add anthropic python-dotenv
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your ANTHROPIC_API_KEY
   ```

3. **Run the agent:**
   ```bash
   python agent.py
   ```

## Usage Examples

### Basic Task Management

```
You: I need to review the Q2 budget by Friday
Agent: I've created a task:
  • Title: Review Q2 budget
  • Due: 2025-01-17
  • Priority: High
```

### Note Taking

```
You: The new API design uses REST instead of GraphQL
Agent: I've saved that note for you. Would you like to link it to any tasks?
```

### Context Switching

```
You: I'm switching to the API Redesign project
Agent: Okay, I've updated your current project to API Redesign. 
      You have 3 active tasks for this project.
```

### Viewing Tasks

```
You: Show me my high priority tasks
Agent: Here are your high priority tasks:

1. Review Q2 budget
   Due: 2025-01-17
   Status: To Do

2. Prepare investor deck
   Due: 2025-01-20
   Status: In Progress
```

## Architecture

The agent consists of three main components:

1. **Storage Layer** (`storage.py`)
   - Handles all data persistence
   - JSON-based for simplicity
   - Separate files for tasks, notes, and context

2. **Tools** (`tools.py`)
   - Individual functions for each capability
   - Each tool loads data, performs operation, saves data
   - Returns structured results

3. **Agent** (`agent.py`)
   - Maintains conversation history
   - Implements agentic loop
   - Routes tool calls to appropriate functions
   - Provides natural language interface

## Data Storage

All data is stored in the `data/` directory:

```
data/
├── tasks.json          # All tasks
├── notes.json          # All notes
└── context.json        # User context and preferences
```

For the time tracking exercise:
```
data/
├── time_entries.json   # Completed time entries
└── active_timers.json  # Currently running timers
```

## Extending the Agent

To add new functionality:

1. **Add a tool function** to `tools.py`:
   ```python
   def your_tool(storage: Storage, param: str) -> Dict[str, Any]:
       # Load data
       # Do operation
       # Save data
       return {"success": True, "result": ...}
   ```

2. **Register it** in `agent.py`:
   ```python
   self.tool_functions["your_tool"] = your_tool
   ```

3. **Add tool definition** in `_get_tools()`:
   ```python
   {
       "name": "your_tool",
       "description": "What it does and when to use it",
       "input_schema": {...}
   }
   ```

## Common Commands

- **quit** - Exit the agent
- **reset** - Clear conversation history (keeps data)

## Notes

- Data persists across sessions
- Conversation history resets when you restart
- All API calls use `claude-sonnet-4-20250514`
- Storage is single-user (no authentication)

## Exercise Solution

The `exercise_solution.py` file adds complete time tracking:

- Start/stop timers for tasks
- Track time entries with durations
- Generate time reports by task or period
- List active timers

Run it with:
```bash
python exercise_solution.py
```

## Production Considerations

This implementation is educational. For production:

- Use a real database (PostgreSQL, MongoDB)
- Add user authentication
- Implement proper error handling
- Add request validation
- Use async operations
- Add rate limiting
- Implement backup/restore
- Add logging and monitoring
