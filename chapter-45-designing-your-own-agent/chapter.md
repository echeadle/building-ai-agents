---
chapter: 45
title: "Designing Your Own Agent"
part: 6
date: 2025-12-09
draft: false
---

# Chapter 45: Designing Your Own Agent

## Introduction

You've reached the end of the book—but really, you're standing at a beginning. You've built three complete agents: a research assistant, a code analyzer, and a productivity helper. You understand the patterns, the pitfalls, and the possibilities. Now it's time to build something of your own.

This chapter isn't a tutorial with code you'll copy. Instead, it's a framework—a methodology for taking an idea and turning it into a working agent. Think of it as a repeatable process you can apply to any agent project, whether it's for work, a side project, or pure experimentation.

The goal is simple: by the end of this chapter, you'll know exactly how to go from "I wish I had an agent that could..." to a deployed, working system.

## Learning Objectives

By the end of this chapter, you will be able to:

- Identify whether a problem is a good fit for an agent solution
- Gather and document requirements systematically
- Design an appropriate tool suite for your agent's domain
- Plan an incremental development approach
- Create a testing strategy for your agent
- Prepare a deployment plan

## The Agent Design Process

Building an agent isn't like building traditional software. The LLM's flexibility means you can't specify every behavior upfront. Instead, you design constraints, tools, and prompts that guide the agent toward desired behaviors.

Here's the process we'll use:

```
1. Problem Analysis → Is this a good agent use case?
2. Requirements → What must the agent do?
3. Tool Design → What capabilities does it need?
4. Architecture → How will it all fit together?
5. Incremental Development → Build one piece at a time
6. Testing Strategy → How will we know it works?
7. Deployment Planning → How will users access it?
```

Let's walk through each step with examples.

## Step 1: Problem Analysis

Not every problem needs an agent. Sometimes a simple script or a single LLM call is better. Use this checklist to evaluate whether an agent is the right solution:

### When Agents Are a Good Fit

✅ **Multi-step tasks** - The task requires multiple actions in sequence
✅ **Dynamic workflows** - Steps depend on results of previous steps
✅ **Tool use** - The task needs to interact with external systems
✅ **Ambiguous input** - Users describe goals, not specific steps
✅ **Natural language** - The task benefits from language understanding

### When Agents Are Overkill

❌ **Single API call** - If one LLM call solves it, don't use an agent
❌ **Fully deterministic** - If you can write exact rules, write a script
❌ **Real-time critical** - Agents add latency; use for async tasks
❌ **Perfect accuracy required** - LLMs are probabilistic; use deterministic code
❌ **Simple automation** - Don't use an agent to do what cron can do

### Example Evaluation

Let's evaluate a few ideas:

**Idea: Email classifier that tags and files emails**
- ✅ Needs to understand natural language
- ✅ Dynamic (different rules for different emails)
- ❌ But: Each email is independent—this is a routing problem, not an agent
- **Verdict**: Use a router (Chapter 19), not an agent

**Idea: Personal finance advisor that analyzes spending and suggests budgets**
- ✅ Multi-step (fetch transactions, categorize, analyze, suggest)
- ✅ Tool use (connect to bank API, create budget)
- ✅ Ambiguous input ("help me save money")
- **Verdict**: Good agent use case

**Idea: Code formatter**
- ❌ Fully deterministic (Black does this perfectly)
- **Verdict**: Don't build an agent for this

> 💡 **Rule of thumb**: If you can write exact pseudocode for every case, you don't need an agent.

## Step 2: Requirements Gathering

Once you've confirmed an agent is appropriate, document what it needs to do. Good requirements are specific, testable, and prioritized.

### The Requirements Template

```markdown
# Agent Requirements: [Agent Name]

## Purpose
[One paragraph: What problem does this agent solve? For whom?]

## Core Capabilities (Must Have)
1. [Capability 1]
2. [Capability 2]
3. [Capability 3]

## Nice-to-Have Features
1. [Feature 1]
2. [Feature 2]

## Out of Scope
1. [What the agent will NOT do]
2. [Common requests to explicitly exclude]

## Success Criteria
- [How will we know if this agent is successful?]
- [What metrics matter?]

## Constraints
- Budget: [Token costs, API limits]
- Latency: [Response time requirements]
- Security: [What data can/cannot be accessed?]
```

### Example: Meeting Notes Agent

Let's design a meeting notes agent as an example. Here's how we'd fill out the template:

```markdown
# Agent Requirements: Meeting Notes Agent

## Purpose
Automatically generate structured meeting notes from transcripts or recordings,
identifying action items, decisions, and key discussion points. For busy teams
who want to focus on the meeting, not note-taking.

## Core Capabilities (Must Have)
1. Accept meeting transcript as input
2. Identify and list action items with owners
3. Identify and list decisions made
4. Summarize key discussion points by topic
5. Output structured markdown notes
6. Tag action items with deadlines if mentioned

## Nice-to-Have Features
1. Email action items to owners automatically
2. Integrate with task management tools (Asana, Jira)
3. Generate meeting summaries for different audiences (exec vs. team)
4. Track recurring meeting themes over time

## Out of Scope
1. Audio transcription (use existing service like Whisper)
2. Real-time note-taking during meeting
3. Calendar integration
4. Video analysis

## Success Criteria
- 95%+ of action items correctly identified and assigned
- Notes ready within 2 minutes of receiving transcript
- Users prefer agent notes to manual notes

## Constraints
- Budget: <$0.50 per meeting (estimate 4000 tokens)
- Latency: <2 minutes for 1-hour meeting transcript
- Security: Meeting content stays internal; no external APIs except Claude
```

> 💡 **Start with "must have" only**. You can add nice-to-have features after the core works.

## Step 3: Tool Design

Now that you know what the agent needs to do, design the tools it needs to do it. Good tool design is the difference between an agent that works and one that flails.

### Tool Design Principles

1. **One tool, one purpose** - Don't create a "do_everything" tool
2. **Clear boundaries** - Each tool should have well-defined inputs and outputs
3. **Error messages that guide** - Help the LLM recover from mistakes
4. **Minimal side effects** - Tools should be as pure/functional as possible
5. **Observable operations** - Log what tools do for debugging

### The Tool Design Template

For each tool your agent needs, document:

```python
"""
Tool Name: [name]

Purpose: [What this tool does and when to use it]

Inputs:
- param1 (type): description
- param2 (type): description

Outputs:
- Returns: description of return value
- Raises: exceptions that might occur

Side Effects:
- [What changes this tool makes, if any]

Example Usage:
[Show how the agent would use this tool]
"""
```

### Example: Meeting Notes Agent Tools

For our meeting notes agent, we'd need:

**Tool 1: parse_transcript**
```python
"""
Tool Name: parse_transcript

Purpose: Break meeting transcript into structured segments with speakers and timestamps

Inputs:
- transcript (str): Raw transcript text (from Whisper or other service)
- format (str): Input format ("whisper_json", "plain_text", "webvtt")

Outputs:
- Returns: List of segments, each with {speaker, timestamp, text}
- Raises: ValueError if format is unsupported

Side Effects: None (pure function)

Example Usage:
segments = parse_transcript(
    transcript=raw_text,
    format="plain_text"
)
# Returns: [
#   {"speaker": "Alice", "timestamp": "00:00:32", "text": "Let's review Q3 numbers"},
#   ...
# ]
"""
```

**Tool 2: extract_action_items**
```python
"""
Tool Name: extract_action_items

Purpose: Extract action items from meeting segments using structured output

Inputs:
- segments (list): Meeting segments from parse_transcript
- context (str): Optional meeting context (agenda, previous decisions)

Outputs:
- Returns: List of action items with {task, owner, deadline, priority}
- Raises: None (returns empty list if no action items)

Side Effects: None (pure function)

Example Usage:
action_items = extract_action_items(
    segments=segments,
    context="Sprint planning meeting for Q1 2025"
)
# Returns: [
#   {
#     "task": "Update design mockups based on feedback",
#     "owner": "Sarah",
#     "deadline": "Friday",
#     "priority": "high"
#   },
#   ...
# ]
"""
```

**Tool 3: generate_notes_markdown**
```python
"""
Tool Name: generate_notes_markdown

Purpose: Format extracted information into structured markdown notes

Inputs:
- action_items (list): From extract_action_items
- decisions (list): From extract_decisions
- summary (str): Meeting summary text
- metadata (dict): Meeting title, date, attendees

Outputs:
- Returns: Formatted markdown string
- Raises: None

Side Effects: None (pure function)

Example Usage:
markdown = generate_notes_markdown(
    action_items=action_items,
    decisions=decisions,
    summary=summary,
    metadata={
        "title": "Sprint Planning",
        "date": "2025-03-15",
        "attendees": ["Alice", "Bob", "Carol"]
    }
)
"""
```

> ⚠️ **Don't give the agent a tool it doesn't need**. Each tool increases complexity and the chance of mistakes.

## Step 4: Architecture

Now sketch out how the pieces fit together. You don't need UML diagrams—a simple flow is fine.

### Architecture Questions

1. **What pattern?** - Is this a workflow or true agent?
2. **State management?** - Does it need memory between steps?
3. **Human-in-the-loop?** - Any approval gates?
4. **Error handling?** - What happens if tools fail?
5. **Observability?** - How will you debug it?

### Example: Meeting Notes Agent Architecture

```
Input: Meeting Transcript
    ↓
[Agent Loop Starts]
    ↓
Tool: parse_transcript
    ↓
Tool: extract_action_items (in parallel with ↓)
Tool: extract_decisions      (in parallel with ↓)
Tool: summarize_discussion   (in parallel)
    ↓
[Wait for all parallel calls]
    ↓
Tool: generate_notes_markdown
    ↓
[Agent Loop Ends]
    ↓
Output: Structured Markdown Notes
```

**Key decisions:**
- **Pattern**: Actually a workflow (orchestrator-workers), not true agent
- **State**: None needed (single session)
- **Human-in-the-loop**: No (trust the output, user can edit)
- **Error handling**: Retry on tool failure, graceful degradation if extraction fails
- **Observability**: Log each tool call with execution time

> 💡 **Simplest architecture that works**. You can always add complexity later.

## Step 5: Incremental Development

Don't try to build everything at once. Here's a proven approach:

### The Development Ladder

**Rung 1: Hello World** (30 minutes)
- Get basic API call working
- Load secrets from `.env`
- Print simple response
- Verify your environment works

**Rung 2: Single Tool** (2 hours)
- Implement one core tool
- Get the agent to call it successfully
- Handle tool errors
- Verify outputs are correct

**Rung 3: Core Flow** (1 day)
- Add remaining core tools
- Implement the main workflow
- Test with one example end-to-end
- Fix obvious bugs

**Rung 4: Edge Cases** (2-3 days)
- Test with varied inputs
- Handle errors gracefully
- Add input validation
- Improve prompts based on failures

**Rung 5: Production Hardening** (1 week)
- Add observability
- Implement proper error handling
- Add tests
- Optimize cost and latency
- Write documentation

**Rung 6: Nice-to-Have Features** (ongoing)
- Add non-critical features
- Gather user feedback
- Iterate on UX

### Development Checklist

For each rung, check off:

**Code Quality**
- [ ] Type hints on all functions
- [ ] Docstrings on all public APIs
- [ ] No hardcoded secrets
- [ ] Error handling for all tool calls
- [ ] Comments explaining non-obvious logic

**Testing**
- [ ] Manual testing with example inputs
- [ ] Edge cases tested (empty input, malformed data)
- [ ] Tool failures handled gracefully
- [ ] Automated tests for critical paths

**Observability**
- [ ] Structured logging for debugging
- [ ] Token usage tracking
- [ ] Execution time measurement
- [ ] Tool call tracing

> 💡 **Git commit after each rung**. It's motivating to see progress, and you can always roll back.

## Step 6: Testing Strategy

Agent testing is different from traditional testing. You can't assert exact outputs, but you can test behaviors.

### What to Test

**Tool Tests (Unit)**
- Tools work with valid inputs
- Tools fail gracefully with invalid inputs
- Tool outputs match expected schema
- Tools handle edge cases (empty lists, null values)

**Workflow Tests (Integration)**
- Agent completes simple tasks end-to-end
- Agent handles tool failures appropriately
- Agent terminates correctly
- Agent produces expected output structure

**Behavior Tests (End-to-End)**
- Agent identifies action items correctly (>90% accuracy)
- Agent assigns owners when mentioned
- Agent doesn't hallucinate action items
- Agent handles meetings with no action items

### Testing Template

```python
"""
Test Suite: [Agent Name]

Purpose: Verify agent behavior meets requirements
"""

import pytest
from your_agent import MeetingNotesAgent

class TestTools:
    """Test individual tools work correctly"""
    
    def test_parse_transcript_valid_input(self):
        # Arrange
        transcript = "Alice: Let's start. Bob: Sounds good."
        
        # Act
        segments = parse_transcript(transcript, format="plain_text")
        
        # Assert
        assert len(segments) == 2
        assert segments[0]["speaker"] == "Alice"
    
    def test_parse_transcript_empty_input(self):
        # Should handle gracefully
        segments = parse_transcript("", format="plain_text")
        assert segments == []

class TestWorkflow:
    """Test agent completes workflows"""
    
    def test_agent_produces_structured_output(self):
        # Arrange
        agent = MeetingNotesAgent()
        transcript = load_example_transcript("simple_meeting.txt")
        
        # Act
        result = agent.process(transcript)
        
        # Assert
        assert "# Meeting Notes" in result
        assert "## Action Items" in result
        assert "## Decisions" in result

class TestBehavior:
    """Test agent behavior with real-world examples"""
    
    def test_identifies_action_items(self):
        # Arrange
        agent = MeetingNotesAgent()
        transcript = load_example_transcript("meeting_with_action_items.txt")
        
        # Act
        result = agent.process(transcript)
        
        # Assert - Check for known action items in transcript
        assert "Update design mockups" in result
        assert "Sarah" in result  # Owner mentioned
        assert "Friday" in result  # Deadline mentioned
    
    def test_handles_meeting_with_no_action_items(self):
        # Arrange
        agent = MeetingNotesAgent()
        transcript = load_example_transcript("informational_meeting.txt")
        
        # Act
        result = agent.process(transcript)
        
        # Assert - Should not hallucinate action items
        action_items_section = extract_section(result, "Action Items")
        assert len(action_items_section) < 50  # Empty or minimal
```

> 💡 **Build a test dataset** with 10-20 example inputs representing common cases and edge cases.

## Step 7: Deployment Planning

Finally, plan how users will access your agent. The deployment strategy depends on your use case.

### Deployment Options

**Option 1: Command-Line Tool**
- Best for: Personal use, developer tools
- Implementation: `python agent.py --input transcript.txt --output notes.md`
- Pros: Simple, no infrastructure
- Cons: Not user-friendly for non-developers

**Option 2: Web API (FastAPI)**
- Best for: Internal team tools, integrations
- Implementation: REST endpoint, accepts POST with transcript, returns notes
- Pros: Easy to integrate with other systems
- Cons: Requires hosting

**Option 3: Slack/Discord Bot**
- Best for: Team collaboration tools
- Implementation: Bot listens for transcript uploads, responds with notes
- Pros: Natural integration into existing workflows
- Cons: Platform-specific code

**Option 4: Web UI**
- Best for: Non-technical users, external customers
- Implementation: Simple web form, upload transcript, display notes
- Pros: Most user-friendly
- Cons: Most complex to build and maintain

### Deployment Checklist

Regardless of deployment option:

**Security**
- [ ] API keys in environment variables, not code
- [ ] Input validation and sanitization
- [ ] Rate limiting to prevent abuse
- [ ] Audit logging for sensitive operations
- [ ] HTTPS in production

**Reliability**
- [ ] Health check endpoint
- [ ] Graceful error handling
- [ ] Retry logic for transient failures
- [ ] Timeout on long-running operations

**Observability**
- [ ] Structured logging
- [ ] Error tracking (Sentry, etc.)
- [ ] Performance monitoring
- [ ] Cost tracking

**Documentation**
- [ ] README with setup instructions
- [ ] API documentation if relevant
- [ ] Example usage
- [ ] Troubleshooting guide

### Example: Minimal FastAPI Deployment

```python
"""
Minimal production deployment for Meeting Notes Agent
"""

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from your_agent import MeetingNotesAgent
import logging

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Meeting Notes Agent API")

# Initialize agent
agent = MeetingNotesAgent()

# Request model
class TranscriptRequest(BaseModel):
    transcript: str
    meeting_title: str = "Untitled Meeting"
    attendees: list[str] = []

# Response model
class NotesResponse(BaseModel):
    notes: str
    action_items_count: int
    processing_time: float

@app.post("/process", response_model=NotesResponse)
async def process_transcript(request: TranscriptRequest):
    """Process meeting transcript and return structured notes"""
    try:
        import time
        start_time = time.time()
        
        # Log request
        logger.info(f"Processing transcript: {request.meeting_title}")
        
        # Process with agent
        result = agent.process(
            transcript=request.transcript,
            metadata={
                "title": request.meeting_title,
                "attendees": request.attendees
            }
        )
        
        processing_time = time.time() - start_time
        
        # Log success
        logger.info(f"Completed in {processing_time:.2f}s")
        
        return NotesResponse(
            notes=result["notes"],
            action_items_count=len(result["action_items"]),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## A Complete Example: From Idea to Deployment

Let's walk through the entire process with a different example: a **Code Documentation Agent** that reads code files and generates/updates documentation.

### Step 1: Problem Analysis

- ✅ Multi-step: Read code, understand it, generate docs, update files
- ✅ Tool use: File reading, writing, code parsing
- ✅ Ambiguous input: "Document this function" vs "Update all docs"
- **Verdict**: Good agent use case

### Step 2: Requirements

```markdown
# Code Documentation Agent

## Purpose
Automatically generate and maintain documentation for Python codebases,
keeping docstrings and README files up-to-date with code changes.

## Core Capabilities
1. Read Python source files
2. Identify undocumented functions/classes
3. Generate docstrings in NumPy/Google style
4. Update existing docstrings if code changed
5. Generate README sections from code

## Success Criteria
- 90%+ of generated docstrings are accurate and helpful
- Maintains consistent documentation style
- Runs in <30 seconds for typical module

## Constraints
- Budget: <$0.10 per module
- Latency: <30 seconds
- Security: Read-only access to code (no execution)
```

### Step 3: Tool Design

```python
# Tool 1: read_python_file
"""Read Python file and return AST + source"""

# Tool 2: extract_function_signature
"""Get function signature, parameters, return type"""

# Tool 3: generate_docstring
"""Generate docstring given function code and signature"""

# Tool 4: update_file_with_docstring
"""Insert/update docstring in file"""

# Tool 5: analyze_module_structure
"""Get module-level overview for README generation"""
```

### Step 4: Architecture

```
Workflow Pattern (Orchestrator-Workers)

Orchestrator:
- Decides what needs documentation
- Delegates to specialized workers
- Aggregates results

Workers:
- Function docstring generator
- Class docstring generator
- Module overview generator
- README generator
```

### Step 5: Development Plan

1. **Rung 1**: Read a single Python file and print its functions
2. **Rung 2**: Generate docstring for one function
3. **Rung 3**: Update the file with the docstring
4. **Rung 4**: Handle entire module
5. **Rung 5**: Add error handling and tests
6. **Rung 6**: Add README generation

### Step 6: Testing

```python
def test_generates_accurate_docstring():
    # Test with simple function
    code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    docstring = generate_docstring(code)
    assert "Add two integers" in docstring or "Sum of" in docstring
    assert "Parameters" in docstring
    assert "Returns" in docstring

def test_preserves_existing_code():
    # Ensure we don't break working code
    original = read_file("example.py")
    agent.document_file("example.py")
    updated = read_file("example.py")
    
    # Code should still work
    exec(updated)  # Should not raise
```

### Step 7: Deployment

Deploy as CLI tool first:

```bash
# Document a single file
python -m doc_agent document src/utils.py

# Document entire project
python -m doc_agent document src/ --recursive

# Check what needs documentation
python -m doc_agent check src/ --report
```

Later, add as pre-commit hook:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: auto-document
      name: Auto-document Python files
      entry: python -m doc_agent document --staged
      language: python
```

## Common Pitfalls

### Pitfall 1: Overscoping the First Version

**Problem**: Trying to build all features at once
**Solution**: Start with absolute minimum. Get one workflow working end-to-end before adding features.

### Pitfall 2: Under-specifying Tool Descriptions

**Problem**: Agent doesn't know when to use which tool
**Solution**: Tool descriptions should include:
- What the tool does
- When to use it (and when NOT to)
- What the inputs mean
- What the outputs mean

### Pitfall 3: Not Testing with Real Data

**Problem**: Agent works on simple examples but fails on real inputs
**Solution**: Test with actual data from your domain, not sanitized examples.

### Pitfall 4: Ignoring Costs During Development

**Problem**: Developing with expensive model, massive prompts
**Solution**: Track token usage from day one. Optimize early.

### Pitfall 5: No Observability

**Problem**: Agent fails and you have no idea why
**Solution**: Log everything: tool calls, LLM responses, errors, execution time.

## Practical Exercise

**Task**: Design your own agent from start to finish.

**Requirements:**

1. **Choose a problem** from your own work or life
2. **Complete the problem analysis** - Is it a good agent use case?
3. **Write requirements** - Use the template from this chapter
4. **Design 3-5 tools** - Use the tool design template
5. **Sketch the architecture** - Flow diagram or text description
6. **Create a development plan** - Define your rungs
7. **Write 3 tests** - One for tools, workflow, and behavior

**Don't write code yet**—just the design documents.

**Deliverable**: A design document following this structure:

```markdown
# Agent Design: [Your Agent Name]

## Problem Analysis
[Is this a good agent use case?]

## Requirements
[Use the template]

## Tool Design
[3-5 tools with full documentation]

## Architecture
[Flow diagram or description]

## Development Plan
[Your 6 rungs]

## Testing Strategy
[3 example tests]

## Deployment Plan
[How will users access this?]
```

## Key Takeaways

- Not every problem needs an agent—evaluate carefully before building
- Good requirements prevent scope creep and guide development
- Tool design is the most important architecture decision
- Build incrementally—get something working quickly, then improve
- Test behaviors, not exact outputs—agents are probabilistic
- Plan deployment from the start—it influences architecture decisions
- The patterns in this book apply to any domain—master them and you can build anything

## What's Next

You've reached the end of the main chapters. You now have:

- ✅ A solid foundation in LLM APIs and conversations
- ✅ Complete understanding of tool use and function calling
- ✅ Five workflow patterns for complex tasks
- ✅ The ability to build true autonomous agents
- ✅ Production skills: testing, observability, optimization, deployment
- ✅ Three complete agent projects as references
- ✅ A framework for designing your own agents

**The appendices** provide additional references:
- Appendix A: Python refresher
- Appendix B: Anthropic API complete reference
- Appendix C: Prompt engineering best practices
- Appendix D: Common agent architectures
- Appendix E: Token and cost optimization
- Appendix F: Security checklist
- Appendix G: Further reading and resources

But really, the next step is yours. Take what you've learned and build something. Start small, test often, and iterate. The patterns are simple, but the possibilities are endless.

**Go build something amazing.**

---

*Thank you for reading "Building AI Agents from Scratch with Python." If you found this book helpful, please share it with others who might benefit. And if you build something cool, I'd love to hear about it.*

*— The Author*
