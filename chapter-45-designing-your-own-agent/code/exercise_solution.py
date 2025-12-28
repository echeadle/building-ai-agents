"""
Exercise Solution: Design Your Own Agent

This is an example solution showing how to complete the design exercise
from Chapter 45. Your design will be different based on your chosen problem.

Chapter 45: Designing Your Own Agent
"""

# =============================================================================
# EXAMPLE: EMAIL ASSISTANT AGENT DESIGN
# =============================================================================

EXAMPLE_DESIGN = """
# Agent Design: Email Assistant Agent

## 1. Problem Analysis

### Is this a good agent use case?

**Multi-step task?** Yes - Check email, identify important messages, draft 
replies, schedule follow-ups, categorize and file.

**Dynamic workflow?** Yes - Actions depend on email content, sender, urgency, 
and user preferences. Can't script exact steps.

**Needs tools?** Yes - Read email, send email, access calendar, create tasks, 
search previous conversations.

**Ambiguous input?** Yes - Users describe goals like "handle my morning emails" 
or "draft a polite decline to this meeting invite" without specific steps.

**Natural language benefits?** Yes - Understanding context, tone, and intent 
requires NLP. Determining urgency and importance is subjective.

**Verdict:** Good agent use case

**Reasoning:** Email management is inherently multi-step with dynamic workflows 
that depend on context. The combination of tool use, natural language 
understanding, and decision-making makes this ideal for an agent.

---

## 2. Requirements

### Purpose
Help busy professionals manage their inbox by automatically triaging emails, 
drafting replies, and ensuring important messages don't get lost. For people 
who receive 50+ emails daily and struggle to stay on top of their inbox.

### Core Capabilities (Must Have)
1. Read and analyze emails from user's inbox
2. Classify emails by importance (urgent, important, FYI, spam)
3. Identify action items and deadlines in emails
4. Draft replies to common types of emails
5. Suggest which emails to respond to first
6. Move emails to appropriate folders

### Nice-to-Have Features
1. Learn from user's email patterns over time
2. Auto-schedule meeting time based on calendar availability
3. Summarize long email threads
4. Proactive reminders for emails needing follow-up
5. Integration with task management tools (Todoist, Asana)

### Out of Scope
1. Actually sending emails without approval (human-in-the-loop required)
2. Accessing attachments or files
3. Managing multiple email accounts simultaneously
4. Real-time monitoring (batch processing only)

### Success Criteria
- Accurately classifies email importance 85%+ of the time
- Drafted replies require minimal editing (user accepts 70%+ as-is)
- Reduces time spent on email triage by 50%
- Zero false positives on "urgent" classification
- Processes daily inbox in <5 minutes

### Constraints
- **Budget:** <$2.00 per day (assuming 50 emails @ $0.04 each)
- **Latency:** Process entire inbox in <5 minutes
- **Security:** Email content is sensitive; never store long-term, use only 
  approved APIs with proper authentication

---

## 3. Tool Design

### Tool 1: fetch_emails

**Purpose:** Retrieve emails from user's inbox using IMAP or email API 
(Gmail API, Outlook Graph API).

**Inputs:**
- `mailbox` (str): Which mailbox to fetch from ("inbox", "sent", etc.)
- `limit` (int): Maximum number of emails to fetch (default 50)
- `unread_only` (bool): Only fetch unread emails (default True)
- `since_date` (str): Only fetch emails after this date (ISO format)

**Outputs:**
- Returns: List[dict] with {"id", "from", "subject", "body", "date", "thread_id"}
- Raises: ConnectionError if can't connect to email server, AuthenticationError 
  if credentials invalid

**Side Effects:** None (read-only)

**Example Usage:**
```python
emails = fetch_emails(
    mailbox="inbox",
    limit=20,
    unread_only=True,
    since_date="2025-03-01"
)
# Returns: [
#   {
#     "id": "msg123",
#     "from": "boss@company.com",
#     "subject": "Urgent: Need Q1 report",
#     "body": "Can you send me the Q1 report by EOD?",
#     "date": "2025-03-15T09:30:00Z",
#     "thread_id": "thread789"
#   },
#   ...
# ]
```

### Tool 2: classify_email

**Purpose:** Classify email importance and suggest appropriate action using 
Claude's analysis capabilities.

**Inputs:**
- `email` (dict): Email object from fetch_emails
- `user_context` (dict): User's role, priorities, calendar, etc.

**Outputs:**
- Returns: dict with {"importance": str, "action": str, "reasoning": str}
  - importance: "urgent", "important", "fyi", "spam"
  - action: "reply_now", "reply_today", "read", "archive", "delete"
  - reasoning: explanation of classification
- Raises: None (always returns a classification)

**Side Effects:** None (pure function)

**Example Usage:**
```python
classification = classify_email(
    email=email_obj,
    user_context={
        "role": "Engineering Manager",
        "current_projects": ["Q1 Planning", "Hiring"],
        "calendar_today": ["1-on-1 with Sarah at 2pm"],
    }
)
# Returns: {
#   "importance": "urgent",
#   "action": "reply_now",
#   "reasoning": "From your manager requesting deliverable by EOD"
# }
```

### Tool 3: draft_reply

**Purpose:** Generate a draft reply to an email based on context and user's 
writing style.

**Inputs:**
- `email` (dict): Email to reply to
- `intent` (str): What the reply should accomplish ("accept", "decline", 
  "request_info", "provide_update", etc.)
- `tone` (str): Desired tone ("professional", "casual", "formal")
- `additional_context` (str): Any additional info to include

**Outputs:**
- Returns: str (draft email text)
- Raises: None

**Side Effects:** None (pure function)

**Example Usage:**
```python
draft = draft_reply(
    email=email_obj,
    intent="decline_politely",
    tone="professional",
    additional_context="Already have meeting scheduled at that time"
)
# Returns: "Hi John,\\n\\nThank you for the invitation. Unfortunately, I have 
# a conflict at that time..."
```

### Tool 4: extract_action_items

**Purpose:** Identify action items and deadlines from email content.

**Inputs:**
- `email` (dict): Email to analyze
- `include_implicit` (bool): Also extract implied action items (default False)

**Outputs:**
- Returns: List[dict] with {"action": str, "deadline": str, "owner": str}
- Raises: None (returns empty list if no action items)

**Side Effects:** None (pure function)

**Example Usage:**
```python
actions = extract_action_items(email=email_obj)
# Returns: [
#   {
#     "action": "Send Q1 report",
#     "deadline": "Today EOD",
#     "owner": "me"
#   }
# ]
```

### Tool 5: move_email

**Purpose:** Move email to specified folder/label.

**Inputs:**
- `email_id` (str): ID of email to move
- `folder` (str): Destination folder ("archive", "projects/q1", etc.)
- `mark_read` (bool): Also mark as read (default False)

**Outputs:**
- Returns: bool (success/failure)
- Raises: ValueError if folder doesn't exist

**Side Effects:** Modifies email in user's mailbox

**Example Usage:**
```python
success = move_email(
    email_id="msg123",
    folder="projects/q1",
    mark_read=True
)
```

---

## 4. Architecture

### Pattern
True agent (autonomous decision-making about which emails to prioritize)

### Flow Diagram
```
Input: "Process my inbox"
    ↓
[Agent Loop Starts]
    ↓
Tool: fetch_emails (get unread emails)
    ↓
For each email:
    ↓
    Tool: classify_email (determine importance/action)
    ↓
    Decision: Is urgent?
        Yes → Tool: draft_reply → Present to user for approval
        No → Continue
    ↓
    Tool: extract_action_items
    ↓
    Tool: move_email (file appropriately)
    ↓
[Next email]
    ↓
[Agent Loop Ends]
    ↓
Output: Summary report with:
- Drafted replies for approval
- Extracted action items
- Emails processed and their status
```

### State Management
- **Short-term**: Current batch of emails being processed
- **Long-term**: User preferences, writing style examples (for draft_reply), 
  classification patterns (for learning)

### Human-in-the-Loop
Yes - Critical approval gates:
1. Before sending any email (show draft, wait for approval)
2. Before classifying something as "spam" (show, wait for confirmation)
3. Before moving emails marked "urgent" (confirm they're not false positives)

### Error Handling
- API connection failures: Retry 3 times with exponential backoff
- Classification failures: Default to "important" + "reply_today" (safe default)
- Draft generation failures: Notify user, don't block other emails
- Gracefully handle partial batch processing (some emails succeed, some fail)

### Observability
- Log every email processed with classification decision
- Track time spent per email
- Monitor classification accuracy (track when user overrides)
- Alert if processing takes >5 minutes or costs exceed budget

---

## 5. Development Plan

### Rung 1: Hello World (30 min)
- [x] Basic Claude API call working
- [x] Environment setup with ANTHROPIC_API_KEY
- [x] Test email API connection (Gmail/Outlook)

### Rung 2: Single Tool (2 hours)
- [ ] Implement fetch_emails tool
- [ ] Get agent to successfully fetch inbox
- [ ] Handle authentication errors
- [ ] Verify email data structure correct

### Rung 3: Core Flow (1 day)
- [ ] Implement classify_email tool
- [ ] Implement extract_action_items tool
- [ ] Implement draft_reply tool
- [ ] Implement move_email tool
- [ ] Process 5 test emails end-to-end

### Rung 4: Edge Cases (2-3 days)
- [ ] Test with 50+ diverse emails
- [ ] Handle emails with no clear sender
- [ ] Handle email threads (multiple messages)
- [ ] Handle HTML formatting in emails
- [ ] Improve classification prompts

### Rung 5: Production Hardening (1 week)
- [ ] Add human-in-the-loop approval gates
- [ ] Implement state persistence
- [ ] Add comprehensive logging
- [ ] Write test suite
- [ ] Optimize for cost (<$2/day)
- [ ] Security review (email data handling)

### Rung 6: Nice-to-Have Features (ongoing)
- [ ] Learn from user's classification overrides
- [ ] Calendar integration for scheduling
- [ ] Thread summarization
- [ ] Proactive follow-up reminders

---

## 6. Testing Strategy

### Tool Tests (Unit)
```python
def test_fetch_emails_returns_correct_structure():
    emails = fetch_emails(mailbox="inbox", limit=1)
    assert len(emails) >= 0
    if len(emails) > 0:
        assert "id" in emails[0]
        assert "from" in emails[0]
        assert "subject" in emails[0]

def test_classify_email_returns_valid_importance():
    email = create_test_email(subject="Urgent: Need help")
    classification = classify_email(email, user_context={})
    assert classification["importance"] in ["urgent", "important", "fyi", "spam"]
    assert classification["action"] in ["reply_now", "reply_today", "read", 
                                        "archive", "delete"]

def test_draft_reply_generates_text():
    email = create_test_email(subject="Meeting request")
    draft = draft_reply(email, intent="accept", tone="professional")
    assert len(draft) > 0
    assert "thank" in draft.lower() or "yes" in draft.lower()
```

### Workflow Tests (Integration)
```python
def test_agent_processes_inbox_end_to_end():
    agent = EmailAssistantAgent()
    # Use test email account
    result = agent.process_inbox(mailbox="test_inbox")
    
    assert result["emails_processed"] > 0
    assert "drafted_replies" in result
    assert "action_items" in result
    assert "processing_time" in result

def test_agent_handles_classification_failure():
    agent = EmailAssistantAgent()
    # Mock classify_email to fail
    with patch('agent.classify_email', side_effect=Exception("API error")):
        result = agent.process_inbox(mailbox="test_inbox")
        # Should still process other emails
        assert result["emails_processed"] >= 0
        assert result["errors"] > 0
```

### Behavior Tests (End-to-End)
```python
def test_correctly_identifies_urgent_emails():
    agent = EmailAssistantAgent()
    
    # Test email clearly marked urgent from boss
    urgent_email = create_test_email(
        from_addr="boss@company.com",
        subject="URGENT: Need Q1 report by EOD",
        body="Please send me the Q1 report by end of day."
    )
    
    classification = agent.classify_email(urgent_email)
    assert classification["importance"] == "urgent"
    assert classification["action"] == "reply_now"

def test_drafts_appropriate_reply_for_meeting_decline():
    agent = EmailAssistantAgent()
    
    meeting_email = create_test_email(
        subject="Meeting on Friday at 2pm?",
        body="Can you make it to a meeting Friday at 2pm?"
    )
    
    draft = agent.draft_reply(
        meeting_email,
        intent="decline_politely",
        tone="professional",
        additional_context="Have conflict at that time"
    )
    
    # Check for polite decline language
    draft_lower = draft.lower()
    assert "unfortunately" in draft_lower or "sorry" in draft_lower
    assert "conflict" in draft_lower or "available" in draft_lower
    # Should not be rude
    assert "no" not in draft_lower or "can't" not in draft_lower[:20]

def test_does_not_hallucinate_action_items():
    agent = EmailAssistantAgent()
    
    # FYI email with no action items
    fyi_email = create_test_email(
        subject="FYI: Team update",
        body="Just wanted to keep you in the loop about the project status."
    )
    
    action_items = agent.extract_action_items(fyi_email)
    assert len(action_items) == 0  # Should not invent action items
```

### Test Dataset
- [x] 10 urgent emails (boss, clients, deadlines)
- [x] 10 important emails (team, projects)
- [x] 10 FYI emails (newsletters, updates)
- [x] 5 spam/promotional emails
- [x] 5 ambiguous emails (test edge cases)
- [x] 5 email threads (multi-message conversations)

---

## 7. Deployment Plan

### Deployment Option
Start with CLI, then add scheduled batch processing

### Phase 1: CLI Tool (Week 1)
```bash
# Process inbox once
python -m email_assistant process

# Process and show drafts for approval
python -m email_assistant process --interactive

# Dry run (don't move emails, just show what would happen)
python -m email_assistant process --dry-run
```

### Phase 2: Scheduled Batch (Week 2)
```bash
# Cron job to run every morning at 8am
0 8 * * * /usr/bin/python3 /path/to/email_assistant/main.py process --auto
```

### Phase 3: Web Dashboard (Month 2+)
- View drafted replies
- Approve/edit/reject drafts
- See action items extracted
- Review classification decisions
- Override and provide feedback

### Infrastructure Requirements
- [ ] Python 3.10+ runtime
- [ ] ANTHROPIC_API_KEY in environment
- [ ] Email API credentials (Gmail OAuth or Outlook)
- [ ] Optional: Database for storing preferences and learning
- [ ] Optional: Task queue (Celery) for async processing

### Security Checklist
- [x] API keys in .env, never committed
- [ ] Email credentials stored securely (OAuth preferred)
- [ ] Email content never logged (PII protection)
- [ ] Rate limiting on email sending (prevent spam)
- [ ] Audit log of all emails sent
- [ ] User must approve any email before sending

### Reliability Checklist
- [ ] Health check for email API connection
- [ ] Graceful degradation if classification fails
- [ ] Retry logic for transient failures
- [ ] Timeout on batch processing (15 minute max)
- [ ] Alert if daily cost exceeds budget

### Documentation
- [ ] README with setup (OAuth flow)
- [ ] User guide (how to review drafts, train classification)
- [ ] Troubleshooting (common auth issues)
- [ ] Privacy policy (how email data is handled)

---

## 8. Timeline and Milestones

**Week 1:** Rungs 1-3 (working prototype that processes test inbox)

**Week 2:** Rung 4 (handle real inbox, edge cases)

**Week 3:** Rung 5 (production-ready with approval gates)

**Week 4+:** Deploy and iterate based on usage

**Go/No-Go Decision Points:**
- After Rung 2: Can we reliably fetch emails?
- After Rung 3: Is classification accurate enough (>80%)?
- After Rung 4: Do drafted replies sound natural?

---

## 9. Risk Assessment

### Technical Risks
1. **Email API rate limits**
   - Mitigation: Batch processing, caching, respect limits
   
2. **Classification accuracy below 85%**
   - Mitigation: Extensive testing, user feedback loop, few-shot examples
   
3. **Draft replies don't match user's voice**
   - Mitigation: Collect writing samples, include tone/style in prompts

### Cost Risks
1. **Processing 100+ emails/day exceeds budget**
   - Mitigation: Optimize prompts, use shorter models for simple tasks, 
     implement daily spending cap

### Security Risks
1. **Email credentials compromised**
   - Mitigation: Use OAuth, never store passwords, implement 2FA

2. **Sensitive email content in logs**
   - Mitigation: Scrub PII from logs, encrypt sensitive data, regular audits

### Timeline Risks
1. **OAuth integration more complex than expected**
   - Mitigation: Use established libraries (Google API client), allocate extra 
     time for auth

---

## 10. Success Metrics

### Launch Metrics (Week 1)
- [ ] Successfully processes 50 test emails
- [ ] Classification accuracy > 80%
- [ ] Draft replies require <30 seconds editing
- [ ] Processes inbox in <5 minutes
- [ ] Zero emails lost or corrupted

### Growth Metrics (Month 1)
- [ ] 500+ emails processed
- [ ] Classification accuracy > 85%
- [ ] User accepts 70%+ of drafted replies as-is
- [ ] Time spent on email reduced by 40%+
- [ ] Cost stays under $2/day

### Optimization Metrics (Month 3)
- [ ] Classification accuracy > 90% (through learning)
- [ ] Cost reduced by 25% through optimization
- [ ] Processing time reduced to <3 minutes
- [ ] User satisfaction > 4.0/5.0
- [ ] Zero false positives on urgent classification
"""

# Print the example
if __name__ == "__main__":
    print(EXAMPLE_DESIGN)
