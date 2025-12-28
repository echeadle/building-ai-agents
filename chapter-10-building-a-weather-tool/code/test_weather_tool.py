"""
Testing the weather tool for reliability.

This module provides comprehensive tests for the weather tool including:
- Valid inputs
- Edge cases (unicode, accents, spaces)
- Invalid inputs
- Error handling
- Response time performance

Chapter 10: Building a Weather Tool
"""

import time
from weather_tool import get_current_weather


def test_valid_cities() -> tuple[int, int]:
    """
    Test that the weather tool works for valid city names.
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("\n" + "=" * 60)
    print("TEST: Valid Cities")
    print("=" * 60)
    
    test_cases = [
        ("London", "celsius"),
        ("New York", "fahrenheit"),
        ("Tokyo", "celsius"),
        ("Sydney", "fahrenheit"),
        ("Paris", "celsius"),
        ("Berlin", "celsius"),
        ("Mumbai", "celsius"),
        ("São Paulo", "celsius"),
    ]
    
    passed = 0
    failed = 0
    
    for city, units in test_cases:
        result = get_current_weather(city, units)
        is_success = "Current weather in" in result
        
        status = "✓ PASS" if is_success else "✗ FAIL"
        passed += 1 if is_success else 0
        failed += 0 if is_success else 1
        
        # Show abbreviated result
        result_preview = result.split('\n')[0] if is_success else result[:60]
        print(f"  {status}: {city} ({units}) -> {result_preview}")
    
    return passed, failed


def test_edge_cases() -> tuple[int, int]:
    """
    Test edge cases like unicode, accents, and spaces.
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    test_cases = [
        # (description, city, should_succeed)
        ("Unicode city (Japanese)", "東京", True),
        ("Unicode city (Chinese)", "北京", True),
        ("City with accent", "Zürich", True),
        ("City with tilde", "São Paulo", True),
        ("City with spaces", "Los Angeles", True),
        ("City with hyphen", "Tel-Aviv", True),
        ("Lowercase city", "london", True),
        ("UPPERCASE city", "PARIS", True),
        ("Mixed case city", "nEw YoRk", True),
        ("City with extra spaces", "  London  ", True),
    ]
    
    passed = 0
    failed = 0
    
    for description, city, should_succeed in test_cases:
        result = get_current_weather(city)
        is_error = result.startswith("Error:")
        actual_success = not is_error
        
        is_pass = actual_success == should_succeed
        status = "✓ PASS" if is_pass else "✗ FAIL"
        passed += 1 if is_pass else 0
        failed += 0 if is_pass else 1
        
        result_preview = result.split('\n')[0][:50] if actual_success else result[:50]
        print(f"  {status}: {description}")
        print(f"         Input: '{city}' -> {result_preview}...")
    
    return passed, failed


def test_invalid_inputs() -> tuple[int, int]:
    """
    Test that invalid inputs return proper error messages.
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("\n" + "=" * 60)
    print("TEST: Invalid Inputs")
    print("=" * 60)
    
    test_cases = [
        # (description, city, expected_to_error)
        ("Empty string", "", True),
        ("Whitespace only", "   ", True),
        ("Nonsense string", "Xyzzyville123ABC", True),
        ("Numbers only", "12345", True),
        ("Special characters", "@#$%^&*()", True),
    ]
    
    passed = 0
    failed = 0
    
    for description, city, expected_error in test_cases:
        result = get_current_weather(city)
        is_error = result.startswith("Error:")
        
        is_pass = is_error == expected_error
        status = "✓ PASS" if is_pass else "✗ FAIL"
        passed += 1 if is_pass else 0
        failed += 0 if is_pass else 1
        
        print(f"  {status}: {description}")
        print(f"         Input: '{city}'")
        print(f"         Result: {result[:60]}...")
    
    return passed, failed


def test_units_parameter() -> tuple[int, int]:
    """
    Test that the units parameter works correctly.
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("\n" + "=" * 60)
    print("TEST: Units Parameter")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test fahrenheit
    result_f = get_current_weather("London", "fahrenheit")
    has_f = "°F" in result_f
    status = "✓ PASS" if has_f else "✗ FAIL"
    passed += 1 if has_f else 0
    failed += 0 if has_f else 1
    print(f"  {status}: Fahrenheit units (should show °F)")
    
    # Test celsius
    result_c = get_current_weather("London", "celsius")
    has_c = "°C" in result_c
    status = "✓ PASS" if has_c else "✗ FAIL"
    passed += 1 if has_c else 0
    failed += 0 if has_c else 1
    print(f"  {status}: Celsius units (should show °C)")
    
    # Test invalid units (should default to fahrenheit)
    result_invalid = get_current_weather("London", "kelvin")
    has_default_f = "°F" in result_invalid
    status = "✓ PASS" if has_default_f else "✗ FAIL"
    passed += 1 if has_default_f else 0
    failed += 0 if has_default_f else 1
    print(f"  {status}: Invalid units (should default to °F)")
    
    return passed, failed


def test_response_times() -> tuple[int, int]:
    """
    Test that responses come back within acceptable time limits.
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("\n" + "=" * 60)
    print("TEST: Response Times")
    print("=" * 60)
    
    cities = ["London", "Tokyo", "Sydney", "New York", "Berlin"]
    max_acceptable_time = 15.0  # seconds
    
    times = []
    passed = 0
    failed = 0
    
    for city in cities:
        start = time.time()
        result = get_current_weather(city)
        elapsed = time.time() - start
        times.append(elapsed)
        
        is_fast = elapsed < max_acceptable_time
        is_success = "Current weather in" in result
        is_pass = is_fast and is_success
        
        status = "✓ PASS" if is_pass else "✗ FAIL"
        passed += 1 if is_pass else 0
        failed += 0 if is_pass else 1
        
        print(f"  {status}: {city} - {elapsed:.2f}s {'(slow!)' if not is_fast else ''}")
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"\n  Summary:")
    print(f"    Average: {avg_time:.2f}s")
    print(f"    Fastest: {min_time:.2f}s")
    print(f"    Slowest: {max_time:.2f}s")
    
    return passed, failed


def test_error_message_format() -> tuple[int, int]:
    """
    Test that error messages are properly formatted and don't leak info.
    
    Returns:
        Tuple of (passed_count, failed_count)
    """
    print("\n" + "=" * 60)
    print("TEST: Error Message Format")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Get an error message
    result = get_current_weather("Xyzzyville123ABC")
    
    # Check that it starts with "Error:"
    starts_with_error = result.startswith("Error:")
    status = "✓ PASS" if starts_with_error else "✗ FAIL"
    passed += 1 if starts_with_error else 0
    failed += 0 if starts_with_error else 1
    print(f"  {status}: Error messages start with 'Error:'")
    
    # Check that it doesn't contain sensitive info
    sensitive_patterns = ["api_key", "API_KEY", "token", "password", "secret", "Traceback"]
    has_sensitive = any(pattern in result for pattern in sensitive_patterns)
    status = "✓ PASS" if not has_sensitive else "✗ FAIL"
    passed += 1 if not has_sensitive else 0
    failed += 0 if not has_sensitive else 1
    print(f"  {status}: Error messages don't leak sensitive info")
    
    # Check that error message is helpful
    is_helpful = "Please" in result or "try again" in result.lower() or "check" in result.lower()
    status = "✓ PASS" if is_helpful else "✗ FAIL"
    passed += 1 if is_helpful else 0
    failed += 0 if is_helpful else 1
    print(f"  {status}: Error messages are helpful")
    
    print(f"\n  Sample error message: {result}")
    
    return passed, failed


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("WEATHER TOOL RELIABILITY TESTS")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run each test suite
    test_suites = [
        ("Valid Cities", test_valid_cities),
        ("Edge Cases", test_edge_cases),
        ("Invalid Inputs", test_invalid_inputs),
        ("Units Parameter", test_units_parameter),
        ("Response Times", test_response_times),
        ("Error Message Format", test_error_message_format),
    ]
    
    results = []
    
    for name, test_fn in test_suites:
        try:
            passed, failed = test_fn()
            results.append((name, passed, failed))
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n  ✗ Test suite '{name}' crashed: {e}")
            results.append((name, 0, 1))
            total_failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed, failed in results:
        total = passed + failed
        pct = (passed / total * 100) if total > 0 else 0
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {name}: {passed}/{total} ({pct:.0f}%)")
    
    print("-" * 60)
    total = total_passed + total_failed
    pct = (total_passed / total * 100) if total > 0 else 0
    print(f"  TOTAL: {total_passed}/{total} tests passed ({pct:.0f}%)")
    
    if total_failed == 0:
        print("\n  🎉 All tests passed!")
    else:
        print(f"\n  ⚠️  {total_failed} test(s) failed")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
