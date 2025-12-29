---
chapter: 10
title: "Building a Weather Tool"
part: 2
date: 2025-01-15
draft: false
---

# Chapter 10: Building a Weather Tool

## Introduction

In Chapter 9, you built a calculator tool—a simple function that performs local computation. But the real power of tools emerges when they connect your agent to the outside world. In this chapter, you'll build a weather tool that fetches live data from a real API, transforming your agent from a text processor into something that knows what's happening right now.

This chapter marks an important transition. Calculator tools are useful for learning, but they don't capture the complexity of real-world integrations. Real APIs have rate limits, network timeouts, malformed responses, and a dozen other ways to fail. Learning to handle these gracefully is what separates toy examples from production-ready code.

By the end of this chapter, you'll have a weather tool that Claude can use to answer questions like "What's the weather in Tokyo?" or "Should I bring an umbrella to Seattle today?" More importantly, you'll understand the patterns for building reliable tools that interact with external services.

## Learning Objectives

By the end of this chapter, you will be able to:

- Design tool interfaces that are clear and useful for LLMs
- Integrate with external APIs using proper HTTP request patterns
- Handle API errors gracefully without crashing your agent
- Format API responses so they're useful for LLM reasoning
- Test tool reliability before deploying to production

## Choosing a Weather API

For this chapter, we'll use **Open-Meteo**, a free weather API that requires no API key and no signup. This removes friction so you can focus on learning the patterns. The same patterns apply to any weather API (or any external API).

> **Note:** If you prefer a different weather API like OpenWeatherMap or WeatherAPI, the concepts transfer directly. You'll just need to adjust the URL structure and response parsing.

Open-Meteo provides:
- Free access with no API key required
- Generous rate limits (10,000 requests/day)
- Global coverage with accurate forecasts
- Simple REST API with JSON responses

Let's explore what the API returns before we build our tool.

## Exploring the Weather API

Before building a tool, always explore the API manually. This helps you understand the data structure and anticipate edge cases.

Open-Meteo uses a geocoding API (to convert city names to coordinates) and a forecast API (to get weather data). Here's how they work:

```python
"""
Exploring the Open-Meteo API to understand its structure.

Chapter 10: Building a Weather Tool
"""

import requests

# Step 1: Convert city name to coordinates using geocoding API
geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
geo_response = requests.get(geocoding_url, params={"name": "London", "count": 1})
geo_data = geo_response.json()

print("Geocoding response:")
print(f"  City: {geo_data['results'][0]['name']}")
print(f"  Country: {geo_data['results'][0]['country']}")
print(f"  Latitude: {geo_data['results'][0]['latitude']}")
print(f"  Longitude: {geo_data['results'][0]['longitude']}")

# Step 2: Get weather for those coordinates
latitude = geo_data['results'][0]['latitude']
longitude = geo_data['results'][0]['longitude']

weather_url = "https://api.open-meteo.com/v1/forecast"
weather_response = requests.get(weather_url, params={
    "latitude": latitude,
    "longitude": longitude,
    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
    "temperature_unit": "fahrenheit"
})
weather_data = weather_response.json()

print("\nWeather response:")
print(f"  Temperature: {weather_data['current']['temperature_2m']}°F")
print(f"  Humidity: {weather_data['current']['relative_humidity_2m']}%")
print(f"  Wind Speed: {weather_data['current']['wind_speed_10m']} km/h")
```

Running this shows us the data structure we'll work with. Notice that we need two API calls: one to get coordinates from a city name, and another to get the weather for those coordinates.

## Designing the Tool Interface

Now let's design our tool. Remember from Chapter 8: tool descriptions are prompts for the LLM. They need to be clear about what the tool does and what inputs it needs.

Here's our weather tool definition:

```python
weather_tool = {
    "name": "get_current_weather",
    "description": (
        "Get the current weather conditions for a specified city. "
        "Returns temperature, humidity, wind speed, and a description of conditions. "
        "Use this when the user asks about current weather, temperature, or "
        "whether they need an umbrella/jacket/etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name to get weather for (e.g., 'London', 'New York', 'Tokyo')"
            },
            "units": {
                "type": "string",
                "enum": ["fahrenheit", "celsius"],
                "description": "Temperature units. Defaults to fahrenheit if not specified."
            }
        },
        "required": ["city"]
    }
}
```

Let's analyze this design:

1. **Clear name**: `get_current_weather` tells exactly what it does
2. **Helpful description**: Explains what it returns AND when to use it
3. **Simple parameters**: Just city and optional units
4. **Examples in description**: "London", "New York", "Tokyo" help Claude understand expected format
5. **Optional with default**: Units has a sensible default, reducing required inputs

> **💡 Tip:** When designing tool parameters, prefer fewer required parameters with sensible defaults. This makes the tool easier for the LLM to use correctly.

## Implementing the Weather Function

Now let's implement the actual function that fetches weather data. This is where error handling becomes critical.

```python
"""
Weather tool implementation with comprehensive error handling.

Chapter 10: Building a Weather Tool
"""

import requests
from typing import Any


# Weather code descriptions from Open-Meteo documentation
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def get_coordinates(city: str) -> dict[str, Any]:
    """
    Convert a city name to geographic coordinates.
    
    Args:
        city: Name of the city to look up
        
    Returns:
        Dictionary with latitude, longitude, and resolved city info
        
    Raises:
        ValueError: If the city cannot be found
        requests.RequestException: If the API request fails
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    
    response = requests.get(
        url,
        params={"name": city, "count": 1},
        timeout=10  # Always set timeouts for external APIs
    )
    response.raise_for_status()  # Raises exception for 4xx/5xx status codes
    
    data = response.json()
    
    # Check if we got any results
    if "results" not in data or len(data["results"]) == 0:
        raise ValueError(f"City not found: {city}")
    
    result = data["results"][0]
    return {
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "name": result["name"],
        "country": result.get("country", "Unknown"),
        "admin1": result.get("admin1", ""),  # State/province
    }


def get_weather_data(
    latitude: float,
    longitude: float,
    units: str = "fahrenheit"
) -> dict[str, Any]:
    """
    Fetch current weather data for given coordinates.
    
    Args:
        latitude: Geographic latitude
        longitude: Geographic longitude
        units: Temperature units ('fahrenheit' or 'celsius')
        
    Returns:
        Dictionary with current weather conditions
        
    Raises:
        requests.RequestException: If the API request fails
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    response = requests.get(
        url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
            "temperature_unit": units,
        },
        timeout=10,
    )
    response.raise_for_status()
    
    return response.json()


def get_current_weather(city: str, units: str = "fahrenheit") -> str:
    """
    Get current weather conditions for a city.
    
    This is the main function that the tool calls. It combines geocoding
    and weather lookup, handles errors, and formats the response for the LLM.
    
    Args:
        city: Name of the city
        units: Temperature units ('fahrenheit' or 'celsius')
        
    Returns:
        A formatted string describing current weather conditions,
        or an error message if the lookup fails.
    """
    try:
        # Step 1: Get coordinates for the city
        location = get_coordinates(city)
        
        # Step 2: Get weather for those coordinates
        weather = get_weather_data(
            location["latitude"],
            location["longitude"],
            units
        )
        
        # Step 3: Extract and format the data
        current = weather["current"]
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]
        weather_code = current["weather_code"]
        precipitation = current.get("precipitation", 0)
        
        # Convert weather code to description
        conditions = WEATHER_CODES.get(weather_code, "Unknown conditions")
        
        # Format location string
        location_parts = [location["name"]]
        if location["admin1"]:
            location_parts.append(location["admin1"])
        location_parts.append(location["country"])
        location_str = ", ".join(location_parts)
        
        # Build response string
        unit_symbol = "°F" if units == "fahrenheit" else "°C"
        
        response = (
            f"Current weather in {location_str}:\n"
            f"• Conditions: {conditions}\n"
            f"• Temperature: {temp}{unit_symbol}\n"
            f"• Humidity: {humidity}%\n"
            f"• Wind Speed: {wind_speed} km/h\n"
            f"• Precipitation: {precipitation} mm"
        )
        
        return response
        
    except ValueError as e:
        # City not found
        return f"Error: {str(e)}. Please check the city name and try again."
        
    except requests.Timeout:
        return "Error: Weather service is taking too long to respond. Please try again."
        
    except requests.ConnectionError:
        return "Error: Unable to connect to weather service. Please check your internet connection."
        
    except requests.HTTPError as e:
        return f"Error: Weather service returned an error (HTTP {e.response.status_code})."
        
    except Exception as e:
        # Catch-all for unexpected errors
        return f"Error: Unable to fetch weather data. {str(e)}"
```

Let's break down the key error handling patterns:

### 1. Timeouts

```python
response = requests.get(url, params=params, timeout=10)
```

Always set timeouts. Without them, a slow or unresponsive API can hang your entire agent indefinitely.

### 2. HTTP Status Checking

```python
response.raise_for_status()
```

This raises an exception for 4xx (client error) and 5xx (server error) status codes, allowing you to catch and handle them appropriately.

### 3. Missing Data Validation

```python
if "results" not in data or len(data["results"]) == 0:
    raise ValueError(f"City not found: {city}")
```

Never assume API responses contain the data you expect. Always validate.

### 4. Specific Exception Handling

```python
except requests.Timeout:
    return "Error: Weather service is taking too long..."
except requests.ConnectionError:
    return "Error: Unable to connect..."
except requests.HTTPError as e:
    return f"Error: Weather service returned an error (HTTP {e.response.status_code})."
```

Different errors need different messages. A timeout is different from a connection error is different from a 404.

### 5. Graceful Fallback

```python
except Exception as e:
    return f"Error: Unable to fetch weather data. {str(e)}"
```

Always have a catch-all. Unknown errors shouldn't crash your agent—they should return a helpful message.

> **⚠️ Warning:** Never return raw exception messages to users in production. They can leak sensitive information like API keys or internal paths. Sanitize error messages appropriately.

## Formatting Results for the LLM

Notice how we format the response:

```python
response = (
    f"Current weather in {location_str}:\n"
    f"• Conditions: {conditions}\n"
    f"• Temperature: {temp}{unit_symbol}\n"
    f"• Humidity: {humidity}%\n"
    f"• Wind Speed: {wind_speed} km/h\n"
    f"• Precipitation: {precipitation} mm"
)
```

This format is designed for the LLM to easily extract and reason about:

1. **Clear labels**: "Temperature:", "Humidity:" make values unambiguous
2. **Units included**: "75°F", "60%" leave no room for confusion
3. **Structured layout**: Bullet points separate distinct facts
4. **Location confirmation**: Including the resolved location helps catch misunderstandings

> **💡 Tip:** When formatting tool results, imagine you're writing a note for a colleague who needs to quickly find specific information. Clear labels and consistent structure help the LLM extract exactly what it needs.

## Building the Complete Agent

Now let's integrate our weather tool into a complete agent. This builds on the patterns from Chapter 9:

```python
"""
Complete weather agent that can answer questions about current conditions.

Chapter 10: Building a Weather Tool
"""

import os
import json
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Verify API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize client
client = anthropic.Anthropic()

# Import our weather function (in practice, this would be in a separate module)
# For now, we'll include it here
import requests
from typing import Any

WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle",
    53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain",
    63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow",
    73: "Moderate snow", 75: "Heavy snow", 80: "Slight rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
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


def get_weather_data(latitude: float, longitude: float, units: str = "fahrenheit") -> dict:
    """Fetch weather data for coordinates."""
    url = "https://api.open-meteo.com/v1/forecast"
    response = requests.get(url, params={
        "latitude": latitude, "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
        "temperature_unit": units,
    }, timeout=10)
    response.raise_for_status()
    return response.json()


def get_current_weather(city: str, units: str = "fahrenheit") -> str:
    """Get current weather for a city. Returns formatted string or error message."""
    try:
        location = get_coordinates(city)
        weather = get_weather_data(location["latitude"], location["longitude"], units)
        current = weather["current"]
        
        conditions = WEATHER_CODES.get(current["weather_code"], "Unknown")
        unit_symbol = "°F" if units == "fahrenheit" else "°C"
        
        location_parts = [location["name"]]
        if location["admin1"]:
            location_parts.append(location["admin1"])
        location_parts.append(location["country"])
        
        return (
            f"Current weather in {', '.join(location_parts)}:\n"
            f"• Conditions: {conditions}\n"
            f"• Temperature: {current['temperature_2m']}{unit_symbol}\n"
            f"• Humidity: {current['relative_humidity_2m']}%\n"
            f"• Wind Speed: {current['wind_speed_10m']} km/h\n"
            f"• Precipitation: {current.get('precipitation', 0)} mm"
        )
    except ValueError as e:
        return f"Error: {str(e)}"
    except requests.Timeout:
        return "Error: Weather service timed out. Please try again."
    except requests.RequestException as e:
        return f"Error: Unable to fetch weather data."
    except Exception as e:
        return f"Error: {str(e)}"


# Tool definition
tools = [
    {
        "name": "get_current_weather",
        "description": (
            "Get the current weather conditions for a specified city. "
            "Returns temperature, humidity, wind speed, and conditions. "
            "Use this when the user asks about current weather, temperature, "
            "or whether they need an umbrella/jacket/etc."
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
                }
            },
            "required": ["city"]
        }
    }
]


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return its result."""
    if tool_name == "get_current_weather":
        city = tool_input["city"]
        units = tool_input.get("units", "fahrenheit")
        return get_current_weather(city, units)
    else:
        return f"Error: Unknown tool '{tool_name}'"


def chat_with_weather_agent(user_message: str) -> str:
    """
    Send a message to the weather agent and get a response.
    Handles the complete tool use loop.
    """
    messages = [{"role": "user", "content": user_message}]
    
    # Initial request to Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=(
            "You are a helpful weather assistant. Use the get_current_weather tool "
            "to answer questions about weather conditions. Be concise but friendly. "
            "If the user doesn't specify units, use fahrenheit for US cities and "
            "celsius for other locations."
        ),
        tools=tools,
        messages=messages,
    )
    
    # Handle tool use loop
    while response.stop_reason == "tool_use":
        # Find the tool use block
        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
                break
        
        if not tool_use_block:
            break
            
        # Execute the tool
        tool_result = process_tool_call(
            tool_use_block.name,
            tool_use_block.input
        )
        
        # Add assistant's response and tool result to messages
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": tool_result,
            }]
        })
        
        # Get next response from Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "You are a helpful weather assistant. Use the get_current_weather tool "
                "to answer questions about weather conditions. Be concise but friendly."
            ),
            tools=tools,
            messages=messages,
        )
    
    # Extract final text response
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    
    return "I wasn't able to generate a response."


# Example usage
if __name__ == "__main__":
    # Test queries
    queries = [
        "What's the weather like in Tokyo right now?",
        "Do I need an umbrella in Seattle today?",
        "What's the temperature in Paris in celsius?",
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        print(f"Assistant: {chat_with_weather_agent(query)}")
        print("-" * 50)
```

When you run this, you'll see Claude use the weather tool to answer questions naturally:

```
User: What's the weather like in Tokyo right now?
Assistant: Right now in Tokyo, Japan, it's partly cloudy with a temperature 
of 72°F. The humidity is at 65% with light winds at 12 km/h. Looks like a 
pleasant day—no rain expected!
--------------------------------------------------
```

## Testing Tool Reliability

Before deploying any tool, you should test it thoroughly. Here's a testing approach for the weather tool:

```python
"""
Testing the weather tool for reliability.

Chapter 10: Building a Weather Tool
"""

import time


def test_weather_tool():
    """Run a series of tests on the weather tool."""
    
    test_cases = [
        # (description, city, expected_behavior)
        ("Valid major city", "New York", "should return weather data"),
        ("Valid city with spaces", "Los Angeles", "should handle spaces"),
        ("International city", "東京", "should handle unicode"),
        ("City with accent", "Zürich", "should handle accents"),
        ("Ambiguous city name", "Springfield", "should return first match"),
        ("Invalid city", "Xyzzyville123", "should return error message"),
        ("Empty string", "", "should return error message"),
    ]
    
    print("Testing weather tool reliability\n" + "=" * 50)
    
    passed = 0
    failed = 0
    
    for description, city, expected in test_cases:
        print(f"\nTest: {description}")
        print(f"  Input: '{city}'")
        
        start_time = time.time()
        result = get_current_weather(city) if city else get_current_weather("")
        elapsed = time.time() - start_time
        
        # Check if result matches expected behavior
        is_error = result.startswith("Error:")
        
        if "error" in expected.lower():
            success = is_error
        else:
            success = not is_error and "Current weather in" in result
        
        status = "✓ PASS" if success else "✗ FAIL"
        passed += 1 if success else 0
        failed += 0 if success else 1
        
        print(f"  Result: {result[:80]}...")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Status: {status}")
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0


def test_response_time():
    """Test that the tool responds within acceptable time limits."""
    
    print("\nTesting response times\n" + "=" * 50)
    
    cities = ["London", "Tokyo", "Sydney", "Cairo", "São Paulo"]
    times = []
    
    for city in cities:
        start = time.time()
        get_current_weather(city)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  {city}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    print(f"\n  Average: {avg_time:.2f}s")
    print(f"  Max: {max_time:.2f}s")
    print(f"  Status: {'✓ PASS' if max_time < 15 else '✗ FAIL'} (max < 15s)")


if __name__ == "__main__":
    test_weather_tool()
    test_response_time()
```

Key things to test:

1. **Valid inputs**: Does it work for normal cities?
2. **Edge cases**: Unicode, accents, spaces, ambiguous names
3. **Invalid inputs**: Empty strings, nonsense, missing data
4. **Response times**: Does it respond within acceptable limits?
5. **Error messages**: Are they helpful and not leaking sensitive info?

## Common Pitfalls

### 1. No Timeout on API Calls

**Problem:** Your agent hangs indefinitely when an external API is slow.

```python
# Bad - no timeout
response = requests.get(url)

# Good - always set timeout
response = requests.get(url, timeout=10)
```

### 2. Assuming API Responses Are Well-Formed

**Problem:** Your code crashes when the API returns unexpected data.

```python
# Bad - assumes structure exists
city = data["results"][0]["name"]

# Good - validate first
if "results" in data and len(data["results"]) > 0:
    city = data["results"][0]["name"]
else:
    raise ValueError("City not found")
```

### 3. Returning Raw Error Messages

**Problem:** Error messages leak internal details or confuse the LLM.

```python
# Bad - exposes internal details
except Exception as e:
    return str(e)  # Might contain API keys, paths, etc.

# Good - sanitized message
except Exception as e:
    return "Error: Unable to fetch weather data. Please try again."
```

## Practical Exercise

**Task:** Extend the weather tool to include a 3-day forecast.

**Requirements:**

1. Add a new parameter `forecast_days` (1-7) to the tool definition
2. Modify the function to fetch forecast data when `forecast_days > 0`
3. Format the forecast in a way that's easy for the LLM to summarize
4. Handle the case where the forecast API returns partial data

**Hints:**

- Open-Meteo's forecast API uses `daily` parameter for daily forecasts
- You'll need: `daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum`
- Consider how to format multi-day data clearly

**Solution:** See `code/exercise.py`

## Key Takeaways

- **Real tools need real error handling.** Network requests fail in countless ways—handle them all gracefully.

- **Always set timeouts on external API calls.** An unresponsive API shouldn't freeze your entire agent.

- **Validate API responses before using them.** Never assume the data structure you expect is what you'll receive.

- **Format results for LLM comprehension.** Clear labels, included units, and structured layout help the LLM reason accurately.

- **Test tools thoroughly before deployment.** Edge cases, error conditions, and performance should all be verified.

- **Error messages should be helpful, not revealing.** Never expose internal details, API keys, or stack traces.

## What's Next

In Chapter 11, we'll expand from single tools to multi-tool agents. You'll learn how to give Claude access to multiple tools simultaneously, how it chooses between them, and patterns for organizing growing tool collections. The weather tool you built here will become part of a larger toolkit that includes the calculator from Chapter 9 and new tools we'll create together.
