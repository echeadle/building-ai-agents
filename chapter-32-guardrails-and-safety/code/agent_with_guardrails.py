"""
Example: Integrating guardrails into an agent.

Chapter 32: Guardrails and Safety

This example demonstrates how to integrate the complete guardrails
module into a working AI agent with:
- Input validation before processing
- Action constraint checking before tool execution
- Output filtering before returning responses
- Resource tracking throughout
"""

import os
import json
from dotenv import load_dotenv
import anthropic

from guardrails import Guardrails, GuardrailsConfig, ActionDecision
from resource_manager import ResourceLimitExceeded

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


# Define tools for the agent
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the specified path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 2')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "search_web",
        "description": "Search the web for information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Email recipient"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
]


def execute_tool(tool_name: str, args: dict) -> str:
    """
    Execute a tool and return the result.
    
    In a real agent, these would perform actual operations.
    For this example, we return mock results.
    """
    if tool_name == "read_file":
        # Mock file reading
        path = args.get("path", "")
        if "secret" in path.lower():
            return "Error: Access denied to sensitive files"
        return f"Contents of {path}:\nThis is sample file content.\nLine 2 of the file."
    
    elif tool_name == "calculate":
        expression = args.get("expression", "")
        try:
            # Simple safe evaluation for basic math
            allowed_chars = set("0123456789+-*/.(). ")
            if all(c in allowed_chars for c in expression):
                result = eval(expression)  # Safe for simple math only
                return f"Result: {result}"
            else:
                return "Error: Invalid expression"
        except Exception as e:
            return f"Error: {e}"
    
    elif tool_name == "search_web":
        query = args.get("query", "")
        return f"Search results for '{query}':\n1. Result one\n2. Result two\n3. Result three"
    
    elif tool_name == "send_email":
        to = args.get("to", "")
        subject = args.get("subject", "")
        return f"Email would be sent to {to} with subject '{subject}'"
    
    return f"Unknown tool: {tool_name}"


def request_approval(tool_name: str, args: dict, approver: str) -> bool:
    """
    Request human approval for a sensitive action.
    
    In a real application, this would present a UI or send a notification.
    For this example, we simulate approval via console input.
    """
    print(f"\n{'='*50}")
    print(f"⚠️  APPROVAL REQUIRED")
    print(f"{'='*50}")
    print(f"Tool: {tool_name}")
    print(f"Arguments: {json.dumps(args, indent=2)}")
    print(f"Approver: {approver}")
    print(f"{'='*50}")
    
    response = input("Approve this action? (y/n): ").strip().lower()
    return response == 'y'


class GuardedAgent:
    """An AI agent with comprehensive guardrails."""
    
    def __init__(self):
        """Initialize the guarded agent."""
        self.client = anthropic.Anthropic()
        
        # Create guardrails with specific configuration
        self.guardrails = Guardrails(GuardrailsConfig(
            # Input validation
            max_message_length=5000,
            
            # Output filtering
            redact_api_keys=True,
            redact_pii=True,
            
            # Action constraints
            allowed_tools=["read_file", "calculate", "search_web", "send_email"],
            blocked_tools=["execute_shell", "delete_file"],
            tools_requiring_approval=["send_email"],
            
            # Resource limits
            max_api_calls=20,
            max_tokens=50000,
            max_tool_calls=10,
            max_cost_dollars=0.50,
            max_duration_seconds=120,
        ))
        
        # Add file path constraints
        self.guardrails.add_file_path_constraint(
            "read_file",
            allowed_directories=["/home/user/documents", "/tmp", "./"],
            blocked_directories=["/etc", "/root", "/home/user/.ssh"]
        )
        
        self.system_prompt = """You are a helpful AI assistant with access to tools.
You can read files, perform calculations, search the web, and send emails.
Always be helpful and accurate. If you cannot perform an action, explain why."""
    
    def run(self, user_input: str) -> str:
        """
        Run the agent with the given user input.
        
        Args:
            user_input: The user's message
            
        Returns:
            The agent's final response
        """
        # Step 1: Validate input
        print("\n📥 Validating input...")
        input_result = self.guardrails.validate_input(user_input)
        if not input_result.is_valid:
            return f"❌ Invalid input: {', '.join(input_result.violations)}"
        
        sanitized_input = input_result.sanitized_value
        print(f"   ✓ Input validated")
        
        # Initialize conversation
        messages = [{"role": "user", "content": sanitized_input}]
        
        # Step 2: Agent loop
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}")
            
            try:
                # Check resource limits
                self.guardrails.check_resources()
                
                # Make API call
                print("   📡 Calling Claude API...")
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    system=self.system_prompt,
                    messages=messages,
                    tools=TOOLS,
                )
                
                # Record usage
                self.guardrails.record_api_call(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                print(f"   📊 Tokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
                
                # Check if done
                if response.stop_reason == "end_turn":
                    # Get final text response
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text += block.text
                    
                    # Step 3: Filter output
                    print("\n📤 Filtering output...")
                    output_result = self.guardrails.filter_output(final_text)
                    
                    if output_result.redactions:
                        print(f"   ⚠️  Applied redactions: {output_result.redactions}")
                    
                    if not output_result.is_safe:
                        print(f"   ⚠️  Safety concerns: {output_result.concerns}")
                    
                    return output_result.filtered_value
                
                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"\n   🔧 Tool call: {block.name}")
                        print(f"      Args: {json.dumps(block.input, indent=2)}")
                        
                        # Step 4: Validate tool arguments
                        args_result = self.guardrails.validate_tool_args(block.name, block.input)
                        if not args_result.is_valid:
                            print(f"      ❌ Invalid args: {args_result.violations}")
                            tool_result = f"Error: Invalid arguments - {args_result.violations}"
                        else:
                            # Step 5: Check action constraints
                            constraint_result = self.guardrails.check_action(
                                block.name, 
                                block.input
                            )
                            
                            if constraint_result.decision == ActionDecision.DENY:
                                print(f"      ❌ Denied: {constraint_result.reason}")
                                tool_result = f"Error: Action denied - {constraint_result.reason}"
                            
                            elif constraint_result.decision == ActionDecision.REQUIRE_APPROVAL:
                                print(f"      ⏸️  Requires approval from {constraint_result.requires_approval_from}")
                                approved = request_approval(
                                    block.name,
                                    block.input,
                                    constraint_result.requires_approval_from
                                )
                                
                                if approved:
                                    print(f"      ✓ Approved")
                                    self.guardrails.record_tool_call()
                                    tool_result = execute_tool(block.name, block.input)
                                else:
                                    print(f"      ❌ Rejected by user")
                                    tool_result = "Error: Action not approved by user"
                            
                            else:
                                print(f"      ✓ Allowed")
                                self.guardrails.record_tool_call()
                                tool_result = execute_tool(block.name, block.input)
                        
                        print(f"      Result: {tool_result[:100]}...")
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(tool_result),
                        })
                
                # Add assistant message and tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            
            except ResourceLimitExceeded as e:
                print(f"\n❌ Resource limit exceeded: {e}")
                return f"I had to stop because: {e}"
            
            except anthropic.APIError as e:
                print(f"\n❌ API error: {e}")
                self.guardrails.record_error()
                return f"An error occurred: {e}"
        
        return "I reached the maximum number of iterations without completing the task."
    
    def print_summary(self):
        """Print usage and activity summary."""
        print("\n" + "="*50)
        print("📊 SESSION SUMMARY")
        print("="*50)
        
        print("\nResource Usage:")
        for key, value in self.guardrails.get_usage_summary().items():
            print(f"  {key}: {value}")
        
        print("\nGuardrails Activity:")
        report = self.guardrails.get_report()
        for key, value in report.to_dict().items():
            if value > 0:
                print(f"  {key}: {value}")


def main():
    """Run the guarded agent demo."""
    print("="*50)
    print("🛡️  GUARDED AGENT DEMO")
    print("="*50)
    
    agent = GuardedAgent()
    
    # Demo queries
    queries = [
        "What is 25 * 17?",
        "Read the file /home/user/documents/report.txt",
        # "Read /etc/passwd",  # This would be blocked
        # "Send an email to boss@company.com about the project status",  # This requires approval
    ]
    
    for query in queries:
        print(f"\n{'='*50}")
        print(f"👤 USER: {query}")
        print("="*50)
        
        response = agent.run(query)
        
        print(f"\n{'='*50}")
        print(f"🤖 AGENT: {response}")
        print("="*50)
    
    # Print session summary
    agent.print_summary()
    
    # Interactive mode
    print("\n" + "="*50)
    print("💬 INTERACTIVE MODE (type 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
            
            response = agent.run(user_input)
            print(f"\n🤖 Agent: {response}")
        
        except KeyboardInterrupt:
            break
    
    # Final summary
    agent.print_summary()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
