"""
Validate that all required environment variables are configured.

Run this script to check your setup:
    uv run python validate_env.py

Chapter 3: Managing Secrets with python-dotenv
"""

import os
import sys
from dotenv import load_dotenv


def validate_environment() -> bool:
    """
    Check that all required environment variables are present and valid.

    Returns:
        True if all validations pass, False otherwise.
    """
    load_dotenv()

    all_valid = True

    # Check Anthropic API key
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("✗ ANTHROPIC_API_KEY: Missing")
        print("  → Get your key from https://console.anthropic.com/")
        all_valid = False
    elif not api_key.startswith("sk-ant-"):
        print("✗ ANTHROPIC_API_KEY: Invalid format")
        print("  → Anthropic keys start with 'sk-ant-'")
        all_valid = False
    elif len(api_key) < 50:
        print("✗ ANTHROPIC_API_KEY: Seems too short")
        print("  → Check that you copied the full key")
        all_valid = False
    else:
        print(f"✓ ANTHROPIC_API_KEY: Valid (starts with {api_key[:15]}...)")

    # Check for common mistakes
    if api_key and "your-" in api_key.lower():
        print("✗ ANTHROPIC_API_KEY: Contains placeholder text")
        print("  → Replace with your actual API key")
        all_valid = False

    return all_valid


def check_gitignore() -> bool:
    """
    Verify that .env is listed in .gitignore.

    Returns:
        True if .gitignore is properly configured.
    """
    gitignore_path = ".gitignore"

    if not os.path.exists(gitignore_path):
        print("✗ .gitignore: File not found")
        print("  → Create a .gitignore file and add '.env' to it")
        return False

    with open(gitignore_path, "r") as f:
        content = f.read()

    if ".env" in content:
        print("✓ .gitignore: Properly configured")
        return True
    else:
        print("✗ .gitignore: Does not include .env")
        print("  → Add '.env' to your .gitignore file")
        return False


def main() -> int:
    """
    Run all validation checks.

    Returns:
        Exit code: 0 if all checks pass, 1 otherwise.
    """
    print("=" * 50)
    print("Environment Validation")
    print("=" * 50)
    print()

    print("Checking environment variables...")
    env_valid = validate_environment()
    print()

    print("Checking security configuration...")
    gitignore_valid = check_gitignore()
    print()

    print("=" * 50)
    if env_valid and gitignore_valid:
        print("All checks passed! You're ready to proceed.")
        return 0
    else:
        print("Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
