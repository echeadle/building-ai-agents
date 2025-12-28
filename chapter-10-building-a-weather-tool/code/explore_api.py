"""
Exploring the Open-Meteo API to understand its structure.

This script demonstrates how the geocoding and weather APIs work
before we build our tool. Always explore APIs manually first!

Chapter 10: Building a Weather Tool
"""

import requests
import json


def explore_geocoding_api(city: str) -> dict:
    """
    Explore the geocoding API response structure.
    
    The geocoding API converts city names to geographic coordinates.
    """
    print(f"\n{'='*60}")
    print(f"GEOCODING API - Looking up: {city}")
    print("="*60)
    
    url = "https://geocoding-api.open-meteo.com/v1/search"
    
    response = requests.get(
        url,
        params={"name": city, "count": 3},  # Get top 3 results
        timeout=10
    )
    
    data = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Full Response:\n{json.dumps(data, indent=2)}")
    
    if "results" in data and len(data["results"]) > 0:
        first_result = data["results"][0]
        print(f"\nExtracted from first result:")
        print(f"  Name: {first_result.get('name')}")
        print(f"  Country: {first_result.get('country')}")
        print(f"  State/Province: {first_result.get('admin1', 'N/A')}")
        print(f"  Latitude: {first_result.get('latitude')}")
        print(f"  Longitude: {first_result.get('longitude')}")
        print(f"  Timezone: {first_result.get('timezone')}")
        return first_result
    else:
        print("\nNo results found!")
        return {}


def explore_weather_api(latitude: float, longitude: float) -> dict:
    """
    Explore the weather API response structure.
    
    The weather API returns current conditions and forecasts for coordinates.
    """
    print(f"\n{'='*60}")
    print(f"WEATHER API - Coordinates: {latitude}, {longitude}")
    print("="*60)
    
    url = "https://api.open-meteo.com/v1/forecast"
    
    response = requests.get(
        url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "forecast_days": 3,
        },
        timeout=10
    )
    
    data = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Full Response:\n{json.dumps(data, indent=2)}")
    
    if "current" in data:
        current = data["current"]
        print(f"\nExtracted current weather:")
        print(f"  Temperature: {current.get('temperature_2m')}°F")
        print(f"  Humidity: {current.get('relative_humidity_2m')}%")
        print(f"  Wind Speed: {current.get('wind_speed_10m')} km/h")
        print(f"  Weather Code: {current.get('weather_code')}")
        print(f"  Precipitation: {current.get('precipitation')} mm")
    
    if "daily" in data:
        daily = data["daily"]
        print(f"\nExtracted daily forecast:")
        for i in range(len(daily.get("time", []))):
            print(f"  {daily['time'][i]}: "
                  f"{daily['temperature_2m_min'][i]}°F - {daily['temperature_2m_max'][i]}°F, "
                  f"Code: {daily['weather_code'][i]}")
    
    return data


def explore_edge_cases():
    """
    Explore how the API handles edge cases.
    """
    print(f"\n{'='*60}")
    print("EDGE CASES")
    print("="*60)
    
    edge_cases = [
        ("Valid city", "London"),
        ("City with spaces", "New York"),
        ("Unicode city", "東京"),
        ("City with accent", "Zürich"),
        ("Invalid city", "Xyzzyville123"),
        ("Empty string", ""),
    ]
    
    url = "https://geocoding-api.open-meteo.com/v1/search"
    
    for description, city in edge_cases:
        print(f"\n{description}: '{city}'")
        try:
            response = requests.get(
                url,
                params={"name": city, "count": 1},
                timeout=10
            )
            data = response.json()
            
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                print(f"  ✓ Found: {result['name']}, {result.get('country', 'Unknown')}")
            else:
                print(f"  ✗ No results")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    # Explore the geocoding API
    location = explore_geocoding_api("Tokyo")
    
    # If we got coordinates, explore the weather API
    if location:
        explore_weather_api(location["latitude"], location["longitude"])
    
    # Explore edge cases
    explore_edge_cases()
