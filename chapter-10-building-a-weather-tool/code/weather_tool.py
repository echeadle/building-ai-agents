"""
Weather tool implementation with comprehensive error handling.

This module provides a get_current_weather function that fetches live
weather data from the Open-Meteo API. It handles errors gracefully and
formats responses for LLM consumption.

Chapter 10: Building a Weather Tool
"""

import requests
from typing import Any


# Weather code descriptions from Open-Meteo documentation
# https://open-meteo.com/en/docs
WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
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


# Tool definition for Claude
WEATHER_TOOL_DEFINITION = {
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


def get_coordinates(city: str) -> dict[str, Any]:
    """
    Convert a city name to geographic coordinates using Open-Meteo geocoding API.
    
    Args:
        city: Name of the city to look up
        
    Returns:
        Dictionary containing:
        - latitude: Geographic latitude
        - longitude: Geographic longitude
        - name: Resolved city name
        - country: Country name
        - admin1: State/province/region name
        
    Raises:
        ValueError: If the city cannot be found
        requests.Timeout: If the request times out
        requests.ConnectionError: If unable to connect
        requests.HTTPError: If the API returns an error status
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    
    response = requests.get(
        url,
        params={"name": city, "count": 1},
        timeout=10  # Always set timeouts for external APIs
    )
    response.raise_for_status()  # Raises exception for 4xx/5xx status codes
    
    data = response.json()
    
    # Validate response structure
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
    Fetch current weather data for given coordinates from Open-Meteo API.
    
    Args:
        latitude: Geographic latitude
        longitude: Geographic longitude
        units: Temperature units ('fahrenheit' or 'celsius')
        
    Returns:
        Raw API response dictionary containing weather data
        
    Raises:
        requests.Timeout: If the request times out
        requests.ConnectionError: If unable to connect
        requests.HTTPError: If the API returns an error status
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


def format_weather_response(
    location: dict[str, Any],
    weather: dict[str, Any],
    units: str
) -> str:
    """
    Format weather data into a human-readable string suitable for LLM processing.
    
    Args:
        location: Location dictionary from get_coordinates()
        weather: Weather dictionary from get_weather_data()
        units: Temperature units used
        
    Returns:
        Formatted string describing current weather conditions
    """
    current = weather["current"]
    
    # Extract values with safe defaults
    temp = current.get("temperature_2m", "N/A")
    humidity = current.get("relative_humidity_2m", "N/A")
    wind_speed = current.get("wind_speed_10m", "N/A")
    weather_code = current.get("weather_code", -1)
    precipitation = current.get("precipitation", 0)
    
    # Convert weather code to description
    conditions = WEATHER_CODES.get(weather_code, "Unknown conditions")
    
    # Format location string
    location_parts = [location["name"]]
    if location["admin1"]:
        location_parts.append(location["admin1"])
    location_parts.append(location["country"])
    location_str = ", ".join(location_parts)
    
    # Format temperature unit symbol
    unit_symbol = "°F" if units == "fahrenheit" else "°C"
    
    # Build response string
    response = (
        f"Current weather in {location_str}:\n"
        f"• Conditions: {conditions}\n"
        f"• Temperature: {temp}{unit_symbol}\n"
        f"• Humidity: {humidity}%\n"
        f"• Wind Speed: {wind_speed} km/h\n"
        f"• Precipitation: {precipitation} mm"
    )
    
    return response


def get_current_weather(city: str, units: str = "fahrenheit") -> str:
    """
    Get current weather conditions for a city.
    
    This is the main function that the tool calls. It combines geocoding
    and weather lookup, handles all errors gracefully, and formats the
    response for the LLM.
    
    Args:
        city: Name of the city to get weather for
        units: Temperature units ('fahrenheit' or 'celsius'), defaults to fahrenheit
        
    Returns:
        A formatted string describing current weather conditions,
        or an error message if the lookup fails.
        
    Note:
        This function never raises exceptions. All errors are caught and
        returned as helpful error messages. This is important for tool
        functions because we want the LLM to receive and handle errors,
        not crash the agent.
    """
    # Validate inputs
    if not city or not city.strip():
        return "Error: City name cannot be empty. Please provide a city name."
    
    if units not in ("fahrenheit", "celsius"):
        units = "fahrenheit"  # Default to fahrenheit for invalid units
    
    try:
        # Step 1: Get coordinates for the city
        location = get_coordinates(city.strip())
        
        # Step 2: Get weather for those coordinates
        weather = get_weather_data(
            location["latitude"],
            location["longitude"],
            units
        )
        
        # Step 3: Format and return the response
        return format_weather_response(location, weather, units)
        
    except ValueError as e:
        # City not found - provide helpful message
        return f"Error: {str(e)}. Please check the city name and try again."
        
    except requests.Timeout:
        # API took too long to respond
        return "Error: Weather service is taking too long to respond. Please try again in a moment."
        
    except requests.ConnectionError:
        # Unable to reach the API
        return "Error: Unable to connect to weather service. Please check your internet connection."
        
    except requests.HTTPError as e:
        # API returned an error status code
        return f"Error: Weather service returned an error (HTTP {e.response.status_code}). Please try again."
        
    except KeyError as e:
        # Unexpected response structure
        return "Error: Received unexpected data format from weather service."
        
    except Exception as e:
        # Catch-all for unexpected errors
        # Note: In production, you'd log the full exception for debugging
        # but return a sanitized message to the user
        return "Error: Unable to fetch weather data. Please try again."


# Example usage and testing
if __name__ == "__main__":
    # Test various cities
    test_cities = [
        ("London", "celsius"),
        ("New York", "fahrenheit"),
        ("Tokyo", "celsius"),
        ("Sydney", "fahrenheit"),
        ("InvalidCity123", "fahrenheit"),
        ("", "fahrenheit"),
    ]
    
    print("Weather Tool Test Results")
    print("=" * 60)
    
    for city, units in test_cities:
        print(f"\nTesting: '{city}' in {units}")
        print("-" * 40)
        result = get_current_weather(city, units)
        print(result)
