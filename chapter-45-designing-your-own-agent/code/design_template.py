"""
Agent Design Template

Use this template to design your own agents from scratch.
This is a complete example showing all sections filled out.

Chapter 45: Designing Your Own Agent
"""

# =============================================================================
# AGENT DESIGN DOCUMENT
# =============================================================================

DESIGN_TEMPLATE = """
# Agent Design: [Agent Name]

## 1. Problem Analysis

### Is this a good agent use case?

**Multi-step task?** [Yes/No + explanation]

**Dynamic workflow?** [Yes/No + explanation]

**Needs tools?** [Yes/No + what tools]

**Ambiguous input?** [Yes/No + examples]

**Natural language benefits?** [Yes/No + explanation]

**Verdict:** [Good agent use case / Use simpler approach / Not suitable]

**Reasoning:** [1-2 sentences explaining your decision]

---

## 2. Requirements

### Purpose
[What problem does this solve? For whom?]

### Core Capabilities (Must Have)
1. [Capability 1]
2. [Capability 2]
3. [Capability 3]

### Nice-to-Have Features
1. [Feature 1]
2. [Feature 2]

### Out of Scope
1. [Explicitly exclude common requests]
2. [Things that might be expected but aren't included]

### Success Criteria
- [Measurable criterion 1]
- [Measurable criterion 2]
- [User satisfaction metric]

### Constraints
- **Budget:** [Token costs, API limits]
- **Latency:** [Response time requirements]
- **Security:** [Data access restrictions]

---

## 3. Tool Design

### Tool 1: [tool_name]

**Purpose:** [What this tool does and when to use it]

**Inputs:**
- `param1` (type): description
- `param2` (type): description

**Outputs:**
- Returns: description
- Raises: exceptions

**Side Effects:** [None / List changes this tool makes]

**Example Usage:**
```python
result = tool_name(param1="value", param2="value")
```

### Tool 2: [tool_name]

[Repeat structure for each tool]

---

## 4. Architecture

### Pattern
[Workflow / True Agent / Hybrid]

### Flow Diagram
```
Input
  ↓
[Step 1: Tool call]
  ↓
[Step 2: Tool call]
  ↓
Output
```

### State Management
[None / In-memory / Persistent]

### Human-in-the-Loop
[None / Approval gates / Feedback loops]

### Error Handling
[How will failures be handled?]

### Observability
[What will be logged?]

---

## 5. Development Plan

### Rung 1: Hello World (30 min)
- [ ] Basic API call working
- [ ] Environment setup verified
- [ ] Secrets loading correctly

### Rung 2: Single Tool (2 hours)
- [ ] Implement [core tool name]
- [ ] Agent successfully calls it
- [ ] Error handling works
- [ ] Output validated

### Rung 3: Core Flow (1 day)
- [ ] All core tools implemented
- [ ] Main workflow working
- [ ] End-to-end test with one example
- [ ] Obvious bugs fixed

### Rung 4: Edge Cases (2-3 days)
- [ ] Tested with varied inputs
- [ ] Errors handled gracefully
- [ ] Input validation added
- [ ] Prompts improved

### Rung 5: Production Hardening (1 week)
- [ ] Observability added
- [ ] Proper error handling
- [ ] Tests written
- [ ] Cost/latency optimized
- [ ] Documentation complete

### Rung 6: Nice-to-Have Features (ongoing)
- [ ] [Feature 1]
- [ ] [Feature 2]
- [ ] User feedback gathered

---

## 6. Testing Strategy

### Tool Tests (Unit)
```python
def test_tool_name_valid_input():
    \"\"\"Test tool works with valid inputs\"\"\"
    # Arrange
    input_data = ...
    
    # Act
    result = tool_name(input_data)
    
    # Assert
    assert result is not None
    assert result["expected_field"] == expected_value
```

### Workflow Tests (Integration)
```python
def test_agent_completes_task():
    \"\"\"Test agent completes simple task end-to-end\"\"\"
    # Arrange
    agent = YourAgent()
    input_data = ...
    
    # Act
    result = agent.process(input_data)
    
    # Assert
    assert result contains expected structure
```

### Behavior Tests (End-to-End)
```python
def test_agent_behavior():
    \"\"\"Test agent produces expected behavior with real data\"\"\"
    # Arrange
    agent = YourAgent()
    real_world_input = load_example("real_case.json")
    
    # Act
    result = agent.process(real_world_input)
    
    # Assert
    assert result meets behavioral requirements
```

### Test Dataset
- [ ] 10-20 example inputs collected
- [ ] Common cases represented
- [ ] Edge cases included
- [ ] Expected outputs documented

---

## 7. Deployment Plan

### Deployment Option
[CLI / Web API / Bot / Web UI]

### Infrastructure Requirements
- [ ] Hosting environment
- [ ] Environment variables
- [ ] API keys secured
- [ ] Monitoring setup

### Security Checklist
- [ ] Secrets in environment variables
- [ ] Input validation implemented
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] HTTPS in production

### Reliability Checklist
- [ ] Health check endpoint
- [ ] Error handling complete
- [ ] Retry logic implemented
- [ ] Timeouts configured

### Documentation
- [ ] README with setup
- [ ] API documentation (if applicable)
- [ ] Example usage
- [ ] Troubleshooting guide

---

## 8. Timeline and Milestones

**Week 1:** Rungs 1-3 (working prototype)
**Week 2:** Rung 4 (edge cases and refinement)
**Week 3:** Rung 5 (production hardening)
**Week 4+:** Rung 6 (nice-to-have features)

**Go/No-Go Decision Points:**
- After Rung 2: Does the core tool work reliably?
- After Rung 3: Does the workflow make sense?
- After Rung 4: Is accuracy acceptable?

---

## 9. Risk Assessment

### Technical Risks
1. [Risk 1] - Mitigation: [strategy]
2. [Risk 2] - Mitigation: [strategy]

### Cost Risks
1. [Risk 1] - Mitigation: [strategy]

### Timeline Risks
1. [Risk 1] - Mitigation: [strategy]

---

## 10. Success Metrics

### Launch Metrics (Week 1)
- [ ] Agent successfully completes 90% of test cases
- [ ] Average response time < [target]
- [ ] Cost per operation < [budget]

### Growth Metrics (Month 1)
- [ ] [Number] of successful operations
- [ ] User satisfaction > [threshold]
- [ ] Error rate < [threshold]

### Optimization Metrics (Month 3)
- [ ] Cost reduced by [target]%
- [ ] Latency reduced by [target]%
- [ ] Accuracy improved to [target]%

"""


# =============================================================================
# EXAMPLE: FILLED OUT DESIGN DOCUMENT
# =============================================================================

EXAMPLE_DESIGN = """
# Agent Design: Meeting Notes Agent

## 1. Problem Analysis

### Is this a good agent use case?

**Multi-step task?** Yes - Parse transcript, extract action items, identify 
decisions, summarize discussion, format output.

**Dynamic workflow?** Somewhat - The extraction steps can run in parallel, but 
the overall flow is predictable.

**Needs tools?** Yes - Transcript parser, information extractors, formatter.

**Ambiguous input?** Yes - Users provide raw transcripts in various formats.

**Natural language benefits?** Yes - Understanding action items and decisions 
requires NLP.

**Verdict:** Good agent use case, though could be simplified to orchestrator-
workers pattern rather than full autonomous agent.

**Reasoning:** Multiple steps with tool use and NLP requirements make this a 
good fit for an agent, though the predictable workflow suggests a structured 
orchestrator approach rather than fully autonomous agent.

---

## 2. Requirements

### Purpose
Automatically generate structured meeting notes from transcripts, identifying 
action items, decisions, and key discussion points. For busy teams who want to 
focus on the meeting, not note-taking.

### Core Capabilities (Must Have)
1. Accept meeting transcript as input (text or JSON)
2. Identify and list action items with owners
3. Identify and list decisions made
4. Summarize key discussion points by topic
5. Output structured markdown notes
6. Tag action items with deadlines if mentioned

### Nice-to-Have Features
1. Email action items to owners automatically
2. Integrate with task management tools (Asana, Jira)
3. Generate summaries for different audiences (exec vs. team)
4. Track recurring meeting themes over time

### Out of Scope
1. Audio transcription (use Whisper API separately)
2. Real-time note-taking during meeting
3. Calendar integration
4. Video analysis (body language, engagement)

### Success Criteria
- 95%+ of action items correctly identified and assigned
- Notes ready within 2 minutes of receiving transcript
- Users prefer agent notes to manual notes (survey)
- Cost < $0.50 per hour of meeting

### Constraints
- **Budget:** <$0.50 per meeting (estimate 4000 tokens @ $0.003/1K = $0.012)
- **Latency:** <2 minutes for 1-hour meeting transcript
- **Security:** Meeting content stays internal; no external APIs except Claude

---

## 3. Tool Design

### Tool 1: parse_transcript

**Purpose:** Break raw transcript into structured segments with speakers and 
timestamps for easier analysis.

**Inputs:**
- `transcript` (str): Raw transcript text
- `format` (str): Input format ("whisper_json", "plain_text", "webvtt")

**Outputs:**
- Returns: List[dict] with {"speaker": str, "timestamp": str, "text": str}
- Raises: ValueError if format unsupported

**Side Effects:** None (pure function)

**Example Usage:**
```python
segments = parse_transcript(
    transcript=raw_text,
    format="plain_text"
)
# Returns: [
#   {"speaker": "Alice", "timestamp": "00:00:32", "text": "Let's review Q3"},
#   {"speaker": "Bob", "timestamp": "00:01:15", "text": "I can send numbers"},
# ]
```

### Tool 2: extract_action_items

**Purpose:** Extract action items from meeting segments using Claude's 
structured output capabilities.

**Inputs:**
- `segments` (List[dict]): Meeting segments from parse_transcript
- `context` (str): Optional meeting context (agenda, previous decisions)

**Outputs:**
- Returns: List[dict] with {"task": str, "owner": str, "deadline": str, 
  "priority": str}
- Raises: None (returns empty list if no action items found)

**Side Effects:** None (pure function)

**Example Usage:**
```python
action_items = extract_action_items(
    segments=segments,
    context="Sprint planning for Q1 2025"
)
# Returns: [
#   {
#     "task": "Update design mockups based on feedback",
#     "owner": "Sarah",
#     "deadline": "Friday",
#     "priority": "high"
#   }
# ]
```

### Tool 3: extract_decisions

**Purpose:** Identify explicit decisions made during the meeting.

**Inputs:**
- `segments` (List[dict]): Meeting segments
- `context` (str): Optional context

**Outputs:**
- Returns: List[dict] with {"decision": str, "reasoning": str, "stakeholders": 
  List[str]}
- Raises: None

**Side Effects:** None

**Example Usage:**
```python
decisions = extract_decisions(segments=segments)
# Returns: [
#   {
#     "decision": "Switch to weekly sprints starting next month",
#     "reasoning": "More frequent feedback will help with current project",
#     "stakeholders": ["Engineering", "Product"]
#   }
# ]
```

### Tool 4: summarize_discussion

**Purpose:** Generate high-level summary of meeting topics and outcomes.

**Inputs:**
- `segments` (List[dict]): Meeting segments
- `action_items` (List[dict]): From extract_action_items
- `decisions` (List[dict]): From extract_decisions

**Outputs:**
- Returns: str (2-3 paragraph summary)
- Raises: None

**Side Effects:** None

**Example Usage:**
```python
summary = summarize_discussion(
    segments=segments,
    action_items=action_items,
    decisions=decisions
)
```

### Tool 5: generate_notes_markdown

**Purpose:** Format all extracted information into clean, structured markdown.

**Inputs:**
- `action_items` (List[dict])
- `decisions` (List[dict])
- `summary` (str)
- `metadata` (dict): {"title": str, "date": str, "attendees": List[str]}

**Outputs:**
- Returns: str (formatted markdown)
- Raises: None

**Side Effects:** None

**Example Usage:**
```python
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
```

---

## 4. Architecture

### Pattern
Orchestrator-Workers (not full autonomous agent)

### Flow Diagram
```
Input: Meeting Transcript + Metadata
    ↓
Tool: parse_transcript
    ↓
    ├─→ Tool: extract_action_items ─→┐
    ├─→ Tool: extract_decisions    ─→┤ (parallel)
    └─→ Tool: summarize_discussion ─→┘
    ↓
[Aggregate Results]
    ↓
Tool: generate_notes_markdown
    ↓
Output: Formatted Markdown Notes
```

### State Management
None needed - single session, no memory between meetings

### Human-in-the-Loop
None - trust the output, user can edit manually if needed

### Error Handling
- Retry transient API failures (with exponential backoff)
- Graceful degradation: if extraction fails, still produce summary
- Log all errors with context for debugging

### Observability
- Log each tool call with execution time
- Track token usage per meeting
- Record any API errors or warnings

---

## 5. Development Plan

### Rung 1: Hello World (30 min)
- [x] Basic Claude API call working
- [x] Environment and API key verified
- [x] Secrets loading from .env

### Rung 2: Single Tool (2 hours)
- [ ] Implement parse_transcript
- [ ] Get agent to call it successfully
- [ ] Handle parsing errors
- [ ] Verify output format correct

### Rung 3: Core Flow (1 day)
- [ ] Implement extract_action_items
- [ ] Implement extract_decisions
- [ ] Implement summarize_discussion
- [ ] Implement generate_notes_markdown
- [ ] Test end-to-end with one example meeting

### Rung 4: Edge Cases (2-3 days)
- [ ] Test with 10 different meeting transcripts
- [ ] Handle meetings with no action items
- [ ] Handle meetings with no decisions
- [ ] Handle malformed transcripts
- [ ] Improve prompts based on failures

### Rung 5: Production Hardening (1 week)
- [ ] Add structured logging
- [ ] Implement retry logic
- [ ] Add cost tracking
- [ ] Write automated tests
- [ ] Optimize for latency
- [ ] Write user documentation

### Rung 6: Nice-to-Have Features (ongoing)
- [ ] Email integration for action items
- [ ] Asana/Jira integration
- [ ] Multiple output formats (PDF, email)
- [ ] Meeting theme tracking

---

## 6. Testing Strategy

### Tool Tests (Unit)
```python
def test_parse_transcript_plain_text():
    transcript = "Alice: Hello\\nBob: Hi there"
    segments = parse_transcript(transcript, format="plain_text")
    assert len(segments) == 2
    assert segments[0]["speaker"] == "Alice"

def test_parse_transcript_empty():
    segments = parse_transcript("", format="plain_text")
    assert segments == []

def test_extract_action_items_with_owner():
    segments = [
        {"speaker": "Alice", "text": "Bob, can you send the report by Friday?"}
    ]
    items = extract_action_items(segments)
    assert len(items) >= 1
    assert items[0]["owner"] == "Bob"
    assert "Friday" in items[0]["deadline"]
```

### Workflow Tests (Integration)
```python
def test_agent_produces_structured_output():
    agent = MeetingNotesAgent()
    transcript = load_example("simple_meeting.txt")
    
    result = agent.process(transcript)
    
    assert "# Meeting Notes" in result
    assert "## Action Items" in result
    assert "## Decisions" in result
    assert "## Summary" in result

def test_agent_handles_tool_failures():
    agent = MeetingNotesAgent()
    # Force a tool to fail
    with patch('agent.extract_action_items', side_effect=Exception):
        result = agent.process(transcript)
        # Should still produce output
        assert "# Meeting Notes" in result
```

### Behavior Tests (End-to-End)
```python
def test_identifies_action_items_accurately():
    agent = MeetingNotesAgent()
    transcript = load_example("meeting_with_known_action_items.txt")
    
    result = agent.process(transcript)
    
    # Check for known action items in transcript
    assert "Update design mockups" in result
    assert "Sarah" in result  # Owner
    assert "Friday" in result  # Deadline

def test_handles_no_action_items():
    agent = MeetingNotesAgent()
    transcript = load_example("informational_meeting.txt")
    
    result = agent.process(transcript)
    
    # Should not hallucinate action items
    action_section = extract_section(result, "Action Items")
    assert len(action_section) < 50  # Empty or minimal

def test_identifies_decisions():
    agent = MeetingNotesAgent()
    transcript = load_example("meeting_with_decision.txt")
    
    result = agent.process(transcript)
    
    assert "Decisions" in result
    assert "Switch to weekly sprints" in result
```

### Test Dataset
- [x] 5 simple meeting transcripts
- [x] 3 complex multi-topic meetings
- [x] 2 meetings with no action items
- [x] 2 meetings with no decisions
- [x] 3 edge cases (malformed, very short, very long)

---

## 7. Deployment Plan

### Deployment Option
Start with CLI, then add web API

### Phase 1: CLI Tool
```bash
python -m meeting_notes process transcript.txt --output notes.md
python -m meeting_notes process transcript.json --format whisper_json
```

### Phase 2: Web API (FastAPI)
```python
POST /api/notes
{
  "transcript": "...",
  "meeting_title": "Sprint Planning",
  "attendees": ["Alice", "Bob"]
}

Response:
{
  "notes": "# Meeting Notes\\n...",
  "action_items_count": 5,
  "processing_time": 12.3
}
```

### Infrastructure Requirements
- [ ] Python 3.10+ runtime
- [ ] ANTHROPIC_API_KEY in environment
- [ ] Optional: Redis for caching
- [ ] Optional: PostgreSQL for meeting history

### Security Checklist
- [x] API key in .env, not code
- [ ] Input validation (max transcript length)
- [ ] Rate limiting (10 requests/minute)
- [ ] Audit logging for all operations
- [ ] HTTPS in production

### Reliability Checklist
- [ ] Health check endpoint (/health)
- [ ] Graceful error handling
- [ ] Retry logic with exponential backoff
- [ ] Timeout on long operations (5 minute max)

### Documentation
- [ ] README with setup instructions
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Example meeting transcripts
- [ ] Troubleshooting guide

---

## 8. Timeline and Milestones

**Week 1:** Rungs 1-3 (working prototype)
- Day 1-2: Environment setup and basic tool
- Day 3-5: All tools implemented, end-to-end working

**Week 2:** Rung 4 (edge cases and refinement)
- Test with varied inputs
- Fix prompt issues
- Handle edge cases

**Week 3:** Rung 5 (production hardening)
- Add logging and monitoring
- Write comprehensive tests
- Optimize for cost and latency

**Week 4:** Deploy and iterate
- Deploy CLI version
- Gather user feedback
- Begin web API if needed

**Go/No-Go Decision Points:**
- After Rung 2: Does parse_transcript work reliably?
- After Rung 3: Is action item extraction accurate enough (>90%)?
- After Rung 4: Can we handle real-world transcripts?

---

## 9. Risk Assessment

### Technical Risks
1. **Action item extraction accuracy < 90%**
   - Mitigation: Build test dataset early, iterate on prompts, consider 
     few-shot examples
   
2. **Latency > 2 minutes for long meetings**
   - Mitigation: Use parallel tool execution, optimize prompts for conciseness

3. **Cost exceeds budget**
   - Mitigation: Monitor token usage, optimize prompt lengths, cache common 
     extractions

### Cost Risks
1. **Unexpected high usage**
   - Mitigation: Implement rate limiting, monitor daily costs, set budget alerts

### Timeline Risks
1. **Prompt engineering takes longer than expected**
   - Mitigation: Start with simple prompts, iterate based on real examples, 
     allow extra week buffer

---

## 10. Success Metrics

### Launch Metrics (Week 1)
- [ ] Agent completes 90% of test cases successfully
- [ ] Average response time < 2 minutes
- [ ] Cost per meeting < $0.50
- [ ] Zero critical bugs

### Growth Metrics (Month 1)
- [ ] 50+ meetings processed
- [ ] User satisfaction > 4.0/5.0
- [ ] Action item accuracy > 90%
- [ ] Error rate < 5%

### Optimization Metrics (Month 3)
- [ ] Cost reduced by 20% through prompt optimization
- [ ] Latency reduced by 30% through parallelization
- [ ] Accuracy improved to >95%
- [ ] 5+ teams using regularly

"""


def print_template():
    """Print the design template for reference"""
    print(DESIGN_TEMPLATE)


def print_example():
    """Print the filled-out example"""
    print(EXAMPLE_DESIGN)


if __name__ == "__main__":
    print("=" * 80)
    print("AGENT DESIGN TEMPLATE")
    print("=" * 80)
    print()
    print("Use this template to design your own agents from scratch.")
    print()
    print("To see the blank template:")
    print("  python design_template.py template")
    print()
    print("To see a filled-out example:")
    print("  python design_template.py example")
    print()
    print("=" * 80)
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "template":
            print_template()
        elif sys.argv[1] == "example":
            print_example()
        else:
            print("Usage: python design_template.py [template|example]")
    else:
        print("Add 'template' or 'example' as argument to see the content")
