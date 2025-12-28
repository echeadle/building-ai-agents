"""
Trip Planning Agent - Exercise Solution

A planning agent that creates day-by-day vacation itineraries with:
- Visible planning process
- Plan revision when constraints are discovered
- Transparent reasoning throughout

Chapter 29: Planning and Reasoning
"""

import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class TripPreferences:
    """User's trip preferences."""
    destination: str
    duration_days: int
    interests: list[str]
    budget_level: str  # "budget", "moderate", "luxury"
    pace: str  # "relaxed", "moderate", "packed"
    dietary_restrictions: list[str] = field(default_factory=list)
    mobility_concerns: str = ""


@dataclass 
class DayPlan:
    """Plan for a single day of the trip."""
    day_number: int
    date: str
    theme: str
    morning: str
    afternoon: str
    evening: str
    meals: dict
    notes: str = ""
    estimated_cost: str = ""


@dataclass
class PlanStep:
    """A step in the planning process."""
    step_number: int
    action: str
    status: str = "pending"  # pending, in_progress, completed, revised
    result: str = ""


# Simulated tools for the trip planner
TRIP_TOOLS = [
    {
        "name": "search_attractions",
        "description": "Search for attractions and activities at a destination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "The destination city"},
                "category": {"type": "string", "description": "Category: museums, nature, food, nightlife, shopping, landmarks"}
            },
            "required": ["destination"]
        }
    },
    {
        "name": "check_opening_hours",
        "description": "Check if an attraction is open on a specific day.",
        "input_schema": {
            "type": "object",
            "properties": {
                "attraction": {"type": "string", "description": "Name of the attraction"},
                "day_of_week": {"type": "string", "description": "Day of the week"}
            },
            "required": ["attraction", "day_of_week"]
        }
    },
    {
        "name": "search_restaurants",
        "description": "Search for restaurants at a destination.",
        "input_schema": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "The destination city"},
                "cuisine": {"type": "string", "description": "Type of cuisine"},
                "price_range": {"type": "string", "description": "budget, moderate, or fine_dining"}
            },
            "required": ["destination"]
        }
    },
    {
        "name": "estimate_travel_time",
        "description": "Estimate travel time between two locations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "from_location": {"type": "string"},
                "to_location": {"type": "string"},
                "mode": {"type": "string", "description": "walking, transit, or driving"}
            },
            "required": ["from_location", "to_location"]
        }
    },
    {
        "name": "save_to_itinerary",
        "description": "Save an activity to the trip itinerary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "integer"},
                "time_slot": {"type": "string", "description": "morning, afternoon, or evening"},
                "activity": {"type": "string"},
                "notes": {"type": "string"}
            },
            "required": ["day", "time_slot", "activity"]
        }
    }
]


def execute_trip_tool(tool_name: str, tool_input: dict, context: dict) -> str:
    """
    Execute a trip planning tool (simulated).
    
    Args:
        tool_name: Name of the tool
        tool_input: Tool input parameters
        context: Shared context including itinerary
        
    Returns:
        Tool result string
    """
    destination = context.get("destination", "the destination")
    
    if tool_name == "search_attractions":
        category = tool_input.get("category", "general")
        # Simulated search results
        attractions = {
            "museums": ["Art Museum", "History Museum", "Science Center"],
            "nature": ["Botanical Garden", "City Park", "Waterfront Trail"],
            "food": ["Food Market", "Culinary Tour", "Cooking Class"],
            "landmarks": ["Old Town Square", "Cathedral", "City Tower"],
            "nightlife": ["Jazz Club", "Rooftop Bar", "Theater District"],
            "shopping": ["Main Street Shops", "Artisan Market", "Antique Row"]
        }
        results = attractions.get(category, ["Various local attractions"])
        return f"Found attractions in {destination} ({category}):\n" + "\n".join(f"- {a}" for a in results)
    
    elif tool_name == "check_opening_hours":
        attraction = tool_input.get("attraction", "")
        day = tool_input.get("day_of_week", "")
        # Simulate some closures
        if "museum" in attraction.lower() and day.lower() == "monday":
            return f"⚠️ {attraction} is CLOSED on Mondays"
        elif "market" in attraction.lower() and day.lower() in ["tuesday", "wednesday"]:
            return f"⚠️ {attraction} is only open Thu-Sun"
        else:
            return f"✓ {attraction} is open on {day} (9 AM - 6 PM)"
    
    elif tool_name == "search_restaurants":
        cuisine = tool_input.get("cuisine", "local")
        price = tool_input.get("price_range", "moderate")
        restaurants = {
            "budget": ["Quick Bites Cafe", "Street Food Corner", "Budget Bistro"],
            "moderate": ["Local Favorite", "Cozy Kitchen", "Garden Restaurant"],
            "fine_dining": ["Chef's Table", "Michelin Star", "Elegant Dining"]
        }
        results = restaurants.get(price, restaurants["moderate"])
        return f"Restaurants ({cuisine}, {price}):\n" + "\n".join(f"- {r}" for r in results)
    
    elif tool_name == "estimate_travel_time":
        from_loc = tool_input.get("from_location", "")
        to_loc = tool_input.get("to_location", "")
        mode = tool_input.get("mode", "transit")
        times = {"walking": "25 min", "transit": "15 min", "driving": "10 min"}
        return f"Travel from {from_loc} to {to_loc} by {mode}: approximately {times.get(mode, '20 min')}"
    
    elif tool_name == "save_to_itinerary":
        day = tool_input.get("day", 1)
        time_slot = tool_input.get("time_slot", "")
        activity = tool_input.get("activity", "")
        notes = tool_input.get("notes", "")
        
        # Add to context itinerary
        if "itinerary" not in context:
            context["itinerary"] = {}
        if day not in context["itinerary"]:
            context["itinerary"][day] = {}
        context["itinerary"][day][time_slot] = {"activity": activity, "notes": notes}
        
        return f"✓ Added to Day {day} ({time_slot}): {activity}"
    
    return f"Unknown tool: {tool_name}"


class TripPlanningAgent:
    """An agent that creates trip itineraries with visible planning."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the trip planning agent."""
        self.verbose = verbose
        self.preferences: TripPreferences | None = None
        self.plan_steps: list[PlanStep] = []
        self.day_plans: list[DayPlan] = []
        self.thinking_log: list[str] = []
        self.context: dict = {}
    
    def log(self, message: str, emoji: str = "ℹ️") -> None:
        """Log and optionally display a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {emoji} {message}"
        self.thinking_log.append(log_entry)
        if self.verbose:
            print(log_entry)
    
    def gather_preferences(self, destination: str, duration: int, 
                          interests: list[str], budget: str = "moderate",
                          pace: str = "moderate") -> TripPreferences:
        """
        Gather and validate trip preferences.
        
        Args:
            destination: Where to go
            duration: Number of days
            interests: List of interests
            budget: Budget level
            pace: Trip pace preference
            
        Returns:
            TripPreferences object
        """
        self.log(f"Planning trip to {destination} for {duration} days", "📍")
        self.log(f"Interests: {', '.join(interests)}", "❤️")
        self.log(f"Budget: {budget}, Pace: {pace}", "💰")
        
        self.preferences = TripPreferences(
            destination=destination,
            duration_days=duration,
            interests=interests,
            budget_level=budget,
            pace=pace
        )
        
        self.context["destination"] = destination
        self.context["duration"] = duration
        
        return self.preferences
    
    def create_planning_strategy(self) -> list[PlanStep]:
        """Create the planning strategy for the trip."""
        self.log("Creating planning strategy...", "📋")
        
        strategy_prompt = f"""Create a planning strategy for this trip:

Destination: {self.preferences.destination}
Duration: {self.preferences.duration_days} days
Interests: {', '.join(self.preferences.interests)}
Budget: {self.preferences.budget_level}
Pace: {self.preferences.pace}

Create 4-6 planning steps to build a great itinerary. Consider:
1. Researching top attractions for their interests
2. Checking for any closures or timing constraints
3. Organizing activities logically by location
4. Finding appropriate dining options
5. Building daily schedules

Respond in JSON:
{{
    "strategy": "Brief description of approach",
    "steps": [
        {{"step_number": 1, "action": "Specific planning action"}}
    ]
}}"""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": strategy_prompt}]
        )
        
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        strategy_data = json.loads(text.strip())
        
        self.log(f"Strategy: {strategy_data['strategy']}", "🎯")
        
        self.plan_steps = [
            PlanStep(step_number=s["step_number"], action=s["action"])
            for s in strategy_data["steps"]
        ]
        
        for step in self.plan_steps:
            self.log(f"  {step.step_number}. {step.action}", "  ")
        
        return self.plan_steps
    
    def execute_planning_step(self, step: PlanStep) -> bool:
        """
        Execute a single planning step.
        
        Returns:
            True if successful, False if revision needed
        """
        step.status = "in_progress"
        self.log(f"Step {step.step_number}: {step.action}", "→")
        
        # Build context
        completed_info = "\n".join([
            f"Step {s.step_number}: {s.result}"
            for s in self.plan_steps if s.status == "completed"
        ])
        
        execution_prompt = f"""Execute this planning step for a trip.

Trip details:
- Destination: {self.preferences.destination}
- Duration: {self.preferences.duration_days} days
- Interests: {', '.join(self.preferences.interests)}
- Budget: {self.preferences.budget_level}
- Pace: {self.preferences.pace}

Current step: {step.action}

Previous planning results:
{completed_info if completed_info else "This is the first step."}

Current itinerary progress:
{json.dumps(self.context.get('itinerary', {}), indent=2)}

You have these tools: search_attractions, check_opening_hours, search_restaurants, estimate_travel_time, save_to_itinerary.

Execute this step using the tools as needed. Provide findings and any constraints discovered."""

        messages = [{"role": "user", "content": execution_prompt}]
        
        # Tool use loop
        max_iterations = 5
        revision_needed = False
        
        for _ in range(max_iterations):
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                tools=TRIP_TOOLS,
                messages=messages
            )
            
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        self.log(f"  Using: {block.name}", "🔧")
                        result = execute_trip_tool(block.name, block.input, self.context)
                        
                        # Check for constraints that might need revision
                        if "CLOSED" in result or "only open" in result:
                            self.log(f"  Constraint found: {result}", "⚠️")
                            revision_needed = True
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                # Step complete
                result_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        result_text += block.text
                
                step.result = result_text
                step.status = "completed"
                self.log(f"  Completed", "✓")
                
                return not revision_needed
        
        return False
    
    def revise_plan(self, reason: str) -> None:
        """Revise the plan based on discovered constraints."""
        self.log(f"Revising plan: {reason}", "🔄")
        
        # Mark remaining steps for revision
        for step in self.plan_steps:
            if step.status == "pending":
                step.status = "revised"
        
        # Create new steps
        revision_prompt = f"""The trip plan needs revision due to: {reason}

Current state:
{json.dumps(self.context.get('itinerary', {}), indent=2)}

Completed planning:
{chr(10).join([f"{s.step_number}. {s.action}: {s.result[:100]}..." for s in self.plan_steps if s.status == "completed"])}

Create 2-3 additional planning steps to address the constraint.

Respond in JSON:
{{
    "new_steps": [
        {{"step_number": N, "action": "..."}}
    ],
    "reasoning": "Why these steps"
}}"""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=256,
            messages=[{"role": "user", "content": revision_prompt}]
        )
        
        text = response.content[0].text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        revision_data = json.loads(text.strip())
        
        next_num = len([s for s in self.plan_steps if s.status == "completed"]) + 1
        for i, new_step in enumerate(revision_data.get("new_steps", [])):
            self.plan_steps.append(PlanStep(
                step_number=next_num + i,
                action=new_step["action"]
            ))
            self.log(f"  Added: {new_step['action']}", "  ")
    
    def generate_itinerary(self) -> str:
        """Generate the final day-by-day itinerary."""
        self.log("Generating final itinerary...", "📅")
        
        itinerary_prompt = f"""Create a detailed day-by-day itinerary.

Trip: {self.preferences.duration_days} days in {self.preferences.destination}
Interests: {', '.join(self.preferences.interests)}
Budget: {self.preferences.budget_level}
Pace: {self.preferences.pace}

Planned activities:
{json.dumps(self.context.get('itinerary', {}), indent=2)}

Research findings:
{chr(10).join([f"- {s.result[:150]}..." for s in self.plan_steps if s.status == "completed"])}

Create a complete itinerary with:
- Day theme
- Morning, afternoon, evening activities
- Meal recommendations
- Practical tips

Make it feel like a real, thoughtful travel plan."""

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2048,
            messages=[{"role": "user", "content": itinerary_prompt}]
        )
        
        return response.content[0].text
    
    def plan_trip(self, destination: str, duration: int, interests: list[str],
                  budget: str = "moderate", pace: str = "moderate") -> str:
        """
        Plan a complete trip.
        
        Args:
            destination: Where to go
            duration: Number of days
            interests: List of interests
            budget: Budget level
            pace: Trip pace
            
        Returns:
            Complete itinerary as string
        """
        print("=" * 60)
        print("🗺️  TRIP PLANNING AGENT")
        print("=" * 60)
        print()
        
        # Gather preferences
        self.gather_preferences(destination, duration, interests, budget, pace)
        print()
        
        # Create strategy
        self.create_planning_strategy()
        print()
        
        # Execute planning steps
        self.log("Executing planning steps...", "⚡")
        
        max_steps = 10
        executed = 0
        
        while executed < max_steps:
            # Find next pending step
            next_step = None
            for step in self.plan_steps:
                if step.status == "pending":
                    next_step = step
                    break
            
            if not next_step:
                break
            
            success = self.execute_planning_step(next_step)
            executed += 1
            
            if not success:
                self.revise_plan("Discovered scheduling constraints")
            
            print()
        
        # Generate final itinerary
        itinerary = self.generate_itinerary()
        
        self.log("Itinerary complete!", "✅")
        
        return itinerary
    
    def get_thinking_log(self) -> str:
        """Get the full thinking log."""
        return "\n".join(self.thinking_log)


if __name__ == "__main__":
    agent = TripPlanningAgent(verbose=True)
    
    # Example trip
    itinerary = agent.plan_trip(
        destination="Barcelona, Spain",
        duration=4,
        interests=["art", "food", "architecture", "beaches"],
        budget="moderate",
        pace="relaxed"
    )
    
    print("\n" + "=" * 60)
    print("📋 FINAL ITINERARY")
    print("=" * 60)
    print(itinerary)
    
    print("\n" + "=" * 60)
    print("📝 COMPLETE THINKING LOG")
    print("=" * 60)
    print(agent.get_thinking_log())
