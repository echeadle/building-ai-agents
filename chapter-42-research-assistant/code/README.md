# Chapter 42 Code: Research Assistant Agent

This directory contains the complete implementation of an autonomous research assistant that can search, read, synthesize, and report on any topic.

## Files

### Core Tools

**`web_search_tool.py`** - Web search capability
- Uses Brave Search API to find relevant sources
- Returns titles, URLs, and snippets
- Requires `BRAVE_API_KEY` (free tier: 2,000 queries/month)
- Get your key at: https://brave.com/search/api/

**`web_read_tool.py`** - Web page content extraction
- Fetches and parses HTML pages
- Extracts main text content
- Removes navigation, scripts, and other noise
- Handles errors gracefully

**`save_note_tool.py`** - Research note management
- Tracks findings as research progresses
- Maintains list of consulted sources
- Provides research progress summary

### Main Implementation

**`research_assistant.py`** - The complete research assistant
- Integrates all three tools
- Implements the agentic loop
- Manages conversation state
- Produces comprehensive reports
- **Run this file to try the agent!**

### Exercise Solution

**`exercise_solution.py`** - Parallel reading enhancement
- Reads multiple sources simultaneously
- Uses ThreadPoolExecutor for concurrency
- Reduces research time by 30-50%
- Demonstrates advanced optimization

## Setup

1. **Install dependencies:**
```bash
pip install anthropic requests beautifulsoup4 python-dotenv --break-system-packages
```

2. **Configure API keys:**

Create a `.env` file in this directory:
```
ANTHROPIC_API_KEY=your-anthropic-key-here
BRAVE_API_KEY=your-brave-search-key-here
```

Get your keys:
- Anthropic: https://console.anthropic.com/
- Brave Search: https://brave.com/search/api/ (free tier available)

## Usage

### Basic Research Assistant

```bash
uv run python research_assistant.py
```

This will:
1. Show example research queries
2. Let you pick one or enter your own
3. Run autonomous research
4. Display a comprehensive report

### Parallel Research Assistant

```bash
uv run python exercise_solution.py
```

This enhanced version reads multiple sources in parallel for faster research.

### Testing Individual Tools

Test each tool separately:

```bash
# Test web search
uv run python web_search_tool.py

# Test web reading
uv run python web_read_tool.py

# Test note-taking
uv run python save_note_tool.py
```

## How It Works

The research assistant uses an agentic loop:

```
1. Receive research query
2. LOOP:
   a. Claude decides what to do next
   b. Execute tool (search, read, or save note)
   c. Claude processes results
   d. Repeat until research is complete
3. Generate final report
```

Claude autonomously decides:
- What to search for
- Which sources to read
- What findings to save
- When to stop researching

## Example Research Flow

For query: "What are best practices for API rate limiting?"

1. **Search**: `web_search("API rate limiting best practices")`
2. **Read**: `web_read(top_result_url)`
3. **Note**: `save_note("Token bucket algorithm commonly used", url)`
4. **Search**: `web_search("rate limiting algorithms comparison")`
5. **Read**: `web_read(another_url)`
6. **Note**: `save_note("429 status code should include Retry-After", url)`
7. ... continue research ...
8. **Report**: Generate comprehensive report with citations

## Customization

### Adjust Research Depth

```python
assistant = ResearchAssistant()
report = assistant.research(
    query="your query",
    max_iterations=15  # Default: 20
)
```

### Change Model

```python
assistant = ResearchAssistant(
    model="claude-sonnet-4-20250514"  # Use a different model
)
```

### Modify System Prompt

Edit the `system_prompt` in `ResearchAssistant.__init__()` to:
- Change research focus (depth vs. breadth)
- Adjust source selection criteria
- Modify report format
- Add domain-specific instructions

## Cost Estimation

Typical research session:
- 10-15 iterations
- 3-5 pages read
- 15,000-25,000 tokens total
- **~$0.30-$0.75** at current Claude pricing

Brave Search is free tier: 2,000 queries/month.

## Common Issues

**"BRAVE_API_KEY not found"**
- Get a free key at https://brave.com/search/api/
- Add to `.env` file: `BRAVE_API_KEY=your-key-here`

**Research loops or doesn't complete**
- System prompt guides behavior - make it more explicit
- Lower `max_iterations` if needed
- Check that notes are being saved (agent might be stuck searching)

**Poor quality reports**
- Add more specific instructions to system prompt
- Provide example report format
- Adjust source selection criteria

**Timeout errors when reading pages**
- Some sites block automated requests
- Agent handles this gracefully and moves on
- Increase timeout in `web_read_tool.py` if needed

## Next Steps

After mastering this capstone:

1. **Customize for your domain**
   - Add domain-specific tools
   - Adjust research process for your needs
   - Create specialized report templates

2. **Add more tools**
   - PDF reading for academic papers
   - Database queries for internal data
   - API calls for real-time information

3. **Improve reliability**
   - Add loop detection (see Chapter 37)
   - Implement quality gates
   - Add caching for frequently accessed pages

4. **Build a UI**
   - FastAPI backend (see Chapter 40)
   - Streaming progress updates
   - Save research sessions

## Learn More

- Chapter 27: The Agentic Loop
- Chapter 28: State Management  
- Chapter 36: Observability and Logging
- Chapter 38: Cost Optimization
- Chapter 41: Security Considerations
