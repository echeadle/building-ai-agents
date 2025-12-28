"""
Exercise Solution: CustomerServiceAgent

Chapter 33: The Complete Agent Class

This exercise demonstrates how to extend the base Agent class
to create a specialized customer service agent with:
- Ticket tracking
- Customer lookup (with approval)
- Support ticket creation
- Appropriate guardrails for customer data
"""

import os
from datetime import datetime
from typing import Any
from dotenv import load_dotenv

from agent import Agent
from config import AgentConfig, PlanningMode, HumanApprovalMode
from tools import ToolRegistry

# Load environment variables
load_dotenv()


class CustomerServiceAgent(Agent):
    """
    A specialized agent for customer service tasks.
    
    Extends the base Agent class with:
    - Ticket history tracking
    - Customer data handling with appropriate safety
    - Support ticket lifecycle management
    
    Example:
        >>> agent = CustomerServiceAgent()
        >>> result = agent.handle_ticket(
        ...     customer_id="C123",
        ...     issue="Unable to login to my account"
        ... )
        >>> print(result["resolution"])
    """
    
    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the customer service agent.
        
        Args:
            config: Optional custom configuration. Uses safe defaults if not provided.
        """
        # Use safe configuration as base - customer data is sensitive!
        if config is None:
            config = AgentConfig.for_safe_agent()
            config.system_prompt = """You are a helpful customer service agent.

Your responsibilities:
1. Help customers resolve issues efficiently
2. Look up customer information when needed (requires approval)
3. Create support tickets for tracking
4. Provide clear, empathetic responses

Guidelines:
- Always verify customer identity before accessing their data
- Be empathetic and professional
- Escalate complex issues appropriately
- Document all interactions in tickets

Never share sensitive customer information like passwords or payment details."""
            
            config.max_iterations = 10
            config.approval_mode = HumanApprovalMode.HIGH_RISK
            config.verbose = True
        
        # Initialize base agent
        super().__init__(config)
        
        # Customer service specific state
        self.ticket_history: list[dict] = []
        self.current_ticket_id: str | None = None
        self._next_ticket_number = 1000
        
        # Mock customer database (in real app, this would be a database)
        self._customers = {
            "C001": {
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "account_type": "Premium",
                "created": "2023-01-15",
                "issues_count": 2
            },
            "C002": {
                "name": "Bob Smith",
                "email": "bob@example.com", 
                "account_type": "Basic",
                "created": "2023-06-20",
                "issues_count": 5
            },
            "C003": {
                "name": "Carol White",
                "email": "carol@example.com",
                "account_type": "Enterprise",
                "created": "2022-11-01",
                "issues_count": 1
            }
        }
        
        # Register customer service tools
        self._register_customer_service_tools()
    
    def _register_customer_service_tools(self) -> None:
        """Register tools specific to customer service."""
        
        # Customer lookup - REQUIRES APPROVAL (sensitive data)
        self.register_tool(
            name="lookup_customer",
            description="""Look up customer information by their ID.
            Returns: name, email, account type, and account creation date.
            Use this to verify customer identity and understand their account.
            IMPORTANT: This accesses sensitive customer data and requires approval.""",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID (e.g., 'C001', 'C002')"
                    }
                },
                "required": ["customer_id"]
            },
            handler=self._lookup_customer,
            requires_approval=True  # Sensitive operation!
        )
        
        # Create support ticket
        self.register_tool(
            name="create_ticket",
            description="""Create a new support ticket for tracking an issue.
            Use this to document customer issues and their resolutions.
            Returns the ticket ID for reference.""",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID"
                    },
                    "issue_summary": {
                        "type": "string",
                        "description": "Brief summary of the issue"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                        "description": "Issue priority level"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["billing", "technical", "account", "general"],
                        "description": "Issue category"
                    }
                },
                "required": ["customer_id", "issue_summary", "priority", "category"]
            },
            handler=self._create_ticket,
            requires_approval=False
        )
        
        # Update ticket status
        self.register_tool(
            name="update_ticket",
            description="""Update the status of an existing support ticket.
            Use this to track progress and resolution of issues.""",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket ID to update"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "waiting_customer", "resolved", "closed"],
                        "description": "New status for the ticket"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Notes about the update"
                    }
                },
                "required": ["ticket_id", "status"]
            },
            handler=self._update_ticket,
            requires_approval=False
        )
        
        # Search knowledge base
        self.register_tool(
            name="search_knowledge_base",
            description="""Search the customer service knowledge base for solutions.
            Use this to find answers to common issues and procedures.""",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the knowledge base"
                    }
                },
                "required": ["query"]
            },
            handler=self._search_knowledge_base,
            requires_approval=False
        )
    
    def _lookup_customer(self, customer_id: str) -> str:
        """Look up customer by ID."""
        if customer_id not in self._customers:
            return f"Customer '{customer_id}' not found in database."
        
        customer = self._customers[customer_id]
        
        # Return formatted customer info (not sensitive details)
        return f"""Customer Found:
- ID: {customer_id}
- Name: {customer["name"]}
- Email: {customer["email"]}
- Account Type: {customer["account_type"]}
- Customer Since: {customer["created"]}
- Previous Issues: {customer["issues_count"]}"""
    
    def _create_ticket(
        self,
        customer_id: str,
        issue_summary: str,
        priority: str,
        category: str
    ) -> str:
        """Create a new support ticket."""
        ticket_id = f"TKT-{self._next_ticket_number}"
        self._next_ticket_number += 1
        
        ticket = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "issue_summary": issue_summary,
            "priority": priority,
            "category": category,
            "status": "open",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "notes": []
        }
        
        self.ticket_history.append(ticket)
        self.current_ticket_id = ticket_id
        
        return f"""Ticket Created Successfully:
- Ticket ID: {ticket_id}
- Customer: {customer_id}
- Priority: {priority.upper()}
- Category: {category}
- Status: OPEN

Please reference this ticket ID in all communications."""
    
    def _update_ticket(
        self,
        ticket_id: str,
        status: str,
        notes: str = ""
    ) -> str:
        """Update an existing ticket."""
        # Find the ticket
        ticket = None
        for t in self.ticket_history:
            if t["ticket_id"] == ticket_id:
                ticket = t
                break
        
        if not ticket:
            return f"Ticket '{ticket_id}' not found."
        
        old_status = ticket["status"]
        ticket["status"] = status
        ticket["updated_at"] = datetime.now().isoformat()
        
        if notes:
            ticket["notes"].append({
                "timestamp": datetime.now().isoformat(),
                "note": notes
            })
        
        return f"""Ticket Updated:
- Ticket ID: {ticket_id}
- Status: {old_status} → {status.upper()}
- Updated: {ticket["updated_at"]}"""
    
    def _search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base (mock implementation)."""
        # Mock knowledge base entries
        knowledge_base = {
            "password": """Password Reset Procedure:
1. Go to login page and click "Forgot Password"
2. Enter your email address
3. Check email for reset link (valid for 24 hours)
4. Create new password (min 8 chars, 1 uppercase, 1 number)
5. If issues persist, verify email is correct in account settings""",
            
            "login": """Login Troubleshooting:
1. Verify email address is correct
2. Check Caps Lock is off
3. Try password reset if forgotten
4. Clear browser cache and cookies
5. Try incognito/private browsing mode
6. Check if account is locked (5 failed attempts)""",
            
            "billing": """Billing FAQ:
- Invoices are sent on the 1st of each month
- Payment methods: Credit card, PayPal, bank transfer
- To update payment method: Settings > Billing > Payment Methods
- Refund requests: Submit within 30 days of charge
- Enterprise customers: Contact account manager""",
            
            "upgrade": """Account Upgrade Process:
1. Log into your account
2. Go to Settings > Subscription
3. Select desired plan
4. Enter payment information
5. Confirm upgrade
- Upgrades take effect immediately
- Prorated charges apply
- Enterprise upgrades require sales contact"""
        }
        
        query_lower = query.lower()
        
        # Find matching entries
        matches = []
        for keyword, content in knowledge_base.items():
            if keyword in query_lower or any(word in query_lower for word in keyword.split()):
                matches.append(content)
        
        if matches:
            return "\n\n---\n\n".join(matches)
        
        return "No matching articles found. Consider escalating to a specialist."
    
    # Custom methods for the customer service domain
    
    def handle_ticket(
        self,
        customer_id: str,
        issue: str,
        priority: str = "medium",
        category: str = "general"
    ) -> dict:
        """
        Handle a customer support ticket from start to resolution.
        
        This is the main entry point for processing customer issues.
        It will:
        1. Look up the customer (with approval)
        2. Create a ticket
        3. Search for solutions
        4. Provide a resolution
        
        Args:
            customer_id: The customer's ID
            issue: Description of the issue
            priority: Priority level (low, medium, high, urgent)
            category: Issue category
            
        Returns:
            Dictionary with ticket details and resolution
        """
        prompt = f"""A customer needs help with the following issue:

Customer ID: {customer_id}
Issue: {issue}
Priority: {priority}
Category: {category}

Please:
1. Look up the customer to understand their account
2. Create a support ticket for this issue
3. Search the knowledge base for relevant solutions
4. Provide a helpful response to resolve the issue
5. Update the ticket with the resolution

Be empathetic and professional in your response."""

        response = self.run(prompt)
        
        return {
            "customer_id": customer_id,
            "issue": issue,
            "ticket_id": self.current_ticket_id,
            "resolution": response,
            "ticket_history": [
                t for t in self.ticket_history 
                if t.get("customer_id") == customer_id
            ],
            "metrics": self.get_metrics()
        }
    
    def get_ticket(self, ticket_id: str) -> dict | None:
        """
        Get a specific ticket by ID.
        
        Args:
            ticket_id: The ticket ID to look up
            
        Returns:
            Ticket dictionary or None if not found
        """
        for ticket in self.ticket_history:
            if ticket["ticket_id"] == ticket_id:
                return ticket
        return None
    
    def get_open_tickets(self) -> list[dict]:
        """
        Get all open tickets.
        
        Returns:
            List of tickets with status 'open' or 'in_progress'
        """
        return [
            t for t in self.ticket_history
            if t["status"] in ("open", "in_progress", "waiting_customer")
        ]
    
    def get_customer_tickets(self, customer_id: str) -> list[dict]:
        """
        Get all tickets for a customer.
        
        Args:
            customer_id: Customer ID to look up
            
        Returns:
            List of tickets for that customer
        """
        return [
            t for t in self.ticket_history
            if t["customer_id"] == customer_id
        ]
    
    def get_ticket_summary(self) -> str:
        """
        Get a summary of all tickets.
        
        Returns:
            Human-readable summary
        """
        if not self.ticket_history:
            return "No tickets in history."
        
        by_status = {}
        by_priority = {}
        
        for ticket in self.ticket_history:
            status = ticket["status"]
            priority = ticket["priority"]
            by_status[status] = by_status.get(status, 0) + 1
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        lines = [f"Total Tickets: {len(self.ticket_history)}"]
        lines.append("\nBy Status:")
        for status, count in sorted(by_status.items()):
            lines.append(f"  {status}: {count}")
        lines.append("\nBy Priority:")
        for priority, count in sorted(by_priority.items()):
            lines.append(f"  {priority}: {count}")
        
        return "\n".join(lines)


def main():
    """Demonstrate the CustomerServiceAgent."""
    print("=" * 60)
    print("Customer Service Agent - Exercise Solution")
    print("=" * 60)
    
    # Create the agent
    agent = CustomerServiceAgent()
    
    print(f"\nAgent: {agent}")
    print(f"Available tools: {agent.tools.list_names()}")
    
    # Handle a customer issue
    print("\n" + "=" * 60)
    print("Handling Customer Issue...")
    print("=" * 60)
    
    result = agent.handle_ticket(
        customer_id="C001",
        issue="I can't log into my account. I've tried resetting my password but the reset email never arrives.",
        priority="high",
        category="account"
    )
    
    print("\n" + "=" * 60)
    print("RESOLUTION")
    print("=" * 60)
    print(f"\nTicket ID: {result['ticket_id']}")
    print(f"\n{result['resolution']}")
    
    # Show ticket summary
    print("\n" + "=" * 60)
    print("TICKET SUMMARY")
    print("=" * 60)
    print(agent.get_ticket_summary())
    
    # Show metrics
    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    for key, value in result['metrics'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
