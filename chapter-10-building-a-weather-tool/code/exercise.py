"""
Exercise Solution: Weather tool with 3-day forecast support.

This extends the basic weather tool to include multi-day forecasts.
The tool can return current conditions, a forecast, or both.

Chapter 10: Building a Weather Tool
"""

import os
from dotenv import load_dotenv
import requests
from typing import Any
import anthropic

# Load environment variables
load_dotenv()

# Weather code descriptions
WEATHER_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

# Extended tool definition with forecast_days parameter
WEATHER_TOOL_WITH_FORECAST = {
    "name": "get_weather",
    "description": (
        "Get weather information for a city. Can return current conditions, "
        "a multi-day forecast, or both. Use forecast_days to get upcoming weather. "
        "Use this when users ask about weather now or in the coming days."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name (e.g., 'London', 'New York', 'Tokyo')"
            },
            "units": {
                "type": "string",
                "enum": ["fahrenheit", "celsius"],
                "description": "Temperature units. Defaults to fahrenheit."
            },
            "forecast_days": {
                "type": "integer",
                "minimum": 0,
                "maximum": 7,
                "description": (
                    "Number of days to forecast (0-7). "
                    "0 = current conditions only (default), "
                    "1-7 = include that many days of forecast"
                )
            }
        },
        "required": ["city"]
    }
}


def get_coordinates(city: str) -> dict[str, Any]:
    """Convert city name to coordinates."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    response = requests.get(url, params={"name": city, "count": 1}, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if "results" not in data or len(data["results"]) == 0:
        raise ValueError(f"City not found: {city}")
    
    result = data["results"][0]
    return {
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "name": result["name"],
        "country": result.get("country", "Unknown"),
        "admin1": result.get("admin1", ""),
    }


def get_weather_with_forecast(
    latitude: float,
    longitude: float,
    units: str = "fahrenheit",
    forecast_days: int = 0
) -> dict[str, Any]:
    """
    Fetch weather data including optional forecast.
    
    Args:
        latitude: Geographic latitude
        longitude: Geographic longitude
        units: Temperature units
        forecast_days: Number of days to forecast (0-7)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
        "temperature_unit": units,
    }
    
    # Add daily forecast if requested
    if forecast_days > 0:
        params["daily"] = "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,precipitation_probability_max"
        params["forecast_days"] = min(forecast_days, 7)  # API max is 7
    
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def format_forecast_day(
    date: str,
    temp_min: float,
    temp_max: float,
    weather_code: int,
    precip_sum: float,
    precip_prob: int,
    unit_symbol: str
) -> str:
    """Format a single day's forecast."""
    conditions = WEATHER_CODES.get(weather_code, "Unknown")
    
    # Format the date nicely
    from datetime import datetime
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        day_name = dt.strftime("%A, %b %d")
    except ValueError:
        day_name = date
    
    # Build forecast line
    line = f"  • {day_name}: {conditions}, {temp_min:.0f}-{temp_max:.0f}{unit_symbol}"
    
    # Add precipitation info if significant
    if precip_prob > 20 or precip_sum > 0.5:
        line += f" (💧 {precip_prob}% chance, {precip_sum:.1f}mm)"
    
    return line


def get_weather(
    city: str,
    units: str = "fahrenheit",
    forecast_days: int = 0
) -> str:
    """
    Get weather for a city with optional forecast.
    
    Args:
        city: City name
        units: Temperature units ('fahrenheit' or 'celsius')
        forecast_days: Number of days to forecast (0 = current only)
    
    Returns:
        Formatted weather information string
    """
    if not city or not city.strip():
        return "Error: City name cannot be empty."
    
    if units not in ("fahrenheit", "celsius"):
        units = "fahrenheit"
    
    forecast_days = max(0, min(7, forecast_days))  # Clamp to 0-7
    
    try:
        location = get_coordinates(city.strip())
        weather = get_weather_with_forecast(
            location["latitude"],
            location["longitude"],
            units,
            forecast_days
        )
        
        # Format location string
        location_parts = [location["name"]]
        if location["admin1"]:
            location_parts.append(location["admin1"])
        location_parts.append(location["country"])
        location_str = ", ".join(location_parts)
        
        unit_symbol = "°F" if units == "fahrenheit" else "°C"
        
        # Build response
        lines = [f"Weather for {location_str}:"]
        
        # Current conditions
        if "current" in weather:
            current = weather["current"]
            conditions = WEATHER_CODES.get(current.get("weather_code", -1), "Unknown")
            temp = current.get("temperature_2m", "N/A")
            humidity = current.get("relative_humidity_2m", "N/A")
            wind = current.get("wind_speed_10m", "N/A")
            
            lines.append("")
            lines.append("Current Conditions:")
            lines.append(f"  • {conditions}, {temp}{unit_symbol}")
            lines.append(f"  • Humidity: {humidity}%")
            lines.append(f"  • Wind: {wind} km/h")
        
        # Forecast
        if forecast_days > 0 and "daily" in weather:
            daily = weather["daily"]
            
            lines.append("")
            lines.append(f"{forecast_days}-Day Forecast:")
            
            # Get the number of days available (might be less than requested)
            num_days = len(daily.get("time", []))
            
            for i in range(min(num_days, forecast_days)):
                # Safely get values with defaults
                date = daily["time"][i] if i < len(daily.get("time", [])) else "Unknown"
                temp_max = daily["temperature_2m_max"][i] if i < len(daily.get("temperature_2m_max", [])) else 0
                temp_min = daily["temperature_2m_min"][i] if i < len(daily.get("temperature_2m_min", [])) else 0
                code = daily["weather_code"][i] if i < len(daily.get("weather_code", [])) else 0
                precip = daily["precipitation_sum"][i] if i < len(daily.get("precipitation_sum", [])) else 0
                prob = daily["precipitation_probability_max"][i] if i < len(daily.get("precipitation_probability_max", [])) else 0
                
                lines.append(format_forecast_day(
                    date, temp_min, temp_max, code, precip, prob, unit_symbol
                ))
        
        return "\n".join(lines)
        
    except ValueError as e:
        return f"Error: {str(e)}"
    except requests.Timeout:
        return "Error: Weather service timed out. Please try again."
    except requests.RequestException:
        return "Error: Unable to fetch weather data."
    except Exception as e:
        return f"Error: Unable to get weather information."


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute the weather tool."""
    if tool_name == "get_weather":
        return get_weather(
            city=tool_input.get("city", ""),
            units=tool_input.get("units", "fahrenheit"),
            forecast_days=tool_input.get("forecast_days", 0)
        )
    return f"Error: Unknown tool '{tool_name}'"


def chat(user_message: str) -> str:
    """Send a message to the weather agent with forecast capability."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not configured"
    
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "You are a helpful weather assistant. Use the get_weather tool to answer "
            "weather questions. Set forecast_days based on what the user asks:\n"
            "- 'current weather' or 'right now' -> forecast_days=0\n"
            "- 'tomorrow' or 'next few days' -> forecast_days=3\n"
            "- 'this week' or 'weekly forecast' -> forecast_days=7\n"
            "Be concise and friendly. Use celsius for non-US cities."
        ),
        tools=[WEATHER_TOOL_WITH_FORECAST],
        messages=messages,
    )
    
    while response.stop_reason == "tool_use":
        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
                break
        
        if not tool_use_block:
            break
        
        tool_result = process_tool_call(tool_use_block.name, tool_use_block.input)
        
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result,
            }]
        })
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "You are a helpful weather assistant. Use the get_weather tool to answer "
                "weather questions. Be concise and friendly."
            ),
            tools=[WEATHER_TOOL_WITH_FORECAST],
            messages=messages,
        )
    
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    
    return "I couldn't generate a response."


# Demo and testing
if __name__ == "__main__":
    print("Weather Tool with Forecast - Exercise Solution")
    print("=" * 60)
    
    # Test the raw function first
    print("\n1. Testing raw function - Current conditions:")
    print("-" * 40)
    print(get_weather("London", "celsius", 0))
    
    print("\n2. Testing raw function - 3-day forecast:")
    print("-" * 40)
    print(get_weather("Tokyo", "celsius", 3))
    
    print("\n3. Testing raw function - 7-day forecast:")
    print("-" * 40)
    print(get_weather("New York", "fahrenheit", 7))
    
    # Test with Claude if API key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n4. Testing with Claude agent:")
        print("-" * 40)
        
        queries = [
            "What's the weather like in Paris right now?",
            "What's the forecast for Seattle this week?",
            "Should I pack an umbrella for London tomorrow?",
        ]
        
        for query in queries:
            print(f"\nUser: {query}")
            print(f"Assistant: {chat(query)}")
    else:
        print("\n(Skipping Claude tests - ANTHROPIC_API_KEY not set)")
