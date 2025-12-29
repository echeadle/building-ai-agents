---
chapter: 3
title: "Managing Secrets with python-dotenv"
part: 1
date: 2025-01-15
draft: false
---

# Chapter 3: Managing Secrets with python-dotenv

## Introduction

In 2019, a developer accidentally committed their AWS credentials to a public GitHub repository. Within hours, cryptocurrency miners had spun up thousands of servers on their account, racking up a $45,000 bill before AWS intervened. This isn't a rare story—it happens every day. GitHub reports scanning billions of commits and finding millions of exposed secrets annually.

When you build AI agents, you'll work with API keys that grant access to powerful (and expensive) services. A leaked Anthropic API key means someone else can run up your bill, access your usage data, and potentially misuse the service under your name. The good news? Preventing this is simple—if you build the right habits from day one.

This chapter establishes a pattern you'll use in every single code file throughout this book: loading secrets from environment variables using **python-dotenv**. It takes five minutes to set up and will save you from potential disaster.

## Learning Objectives

By the end of this chapter, you will be able to:

-   Explain why hardcoding secrets is dangerous and how leaks happen
-   Install python-dotenv using uv
-   Create and structure a `.env` file for your project
-   Load environment variables in Python code
-   Configure `.gitignore` to protect your secrets
-   Validate that secrets are loaded correctly before using them

## Why Secrets Management Matters

Let's be concrete about what can go wrong.

### The Ways Secrets Leak

**1. Committing to version control**

This is the most common leak. You hardcode an API key "just for testing," forget about it, and push to GitHub. Even if you delete it in the next commit, it's still in your git history—and bots are scanning for it.

```python
# DON'T DO THIS - Ever, even "temporarily"
client = anthropic.Anthropic(api_key="sk-ant-api03-XXXXXXX")
```

**2. Sharing code**

You copy your working code to help a colleague or post it on Stack Overflow. The API key goes with it.

**3. Logs and error messages**

Your application crashes and logs the full configuration, including secrets. Those logs end up in a monitoring service, a Slack channel, or a bug report.

**4. Screenshot accidents**

You're writing documentation or recording a tutorial. Your terminal shows the command where you set the API key.

### The Consequences

-   **Financial**: API keys often have usage-based billing. Attackers can run up enormous bills.
-   **Security**: Keys may grant access to sensitive data or the ability to act on your behalf.
-   **Account termination**: Service providers may ban your account for failing to secure credentials.
-   **Professional reputation**: Leaked keys in a work context can have career consequences.

### The Solution: Environment Variables

The fix is straightforward: **never put secrets in your code**. Instead, store them in environment variables that exist only on the machine running the code. The **python-dotenv** library makes this easy by loading variables from a `.env` file that you explicitly exclude from version control.

## Installing python-dotenv with uv

In Chapter 2, you set up your project with uv. Now let's add our first dependency.

Navigate to your project directory and run:

```bash
uv add python-dotenv
```

You should see output similar to:

```
Resolved 1 package in 52ms
Installed 1 package in 12ms
 + python-dotenv==1.0.1
```

Your `pyproject.toml` now includes the dependency:

```toml
[project]
name = "agents-from-scratch"
version = "0.1.0"
description = "Building AI agents from scratch with Python"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.75.0",
    "python-dotenv>=1.0.1",
]
```

> **Note:** uv automatically creates and manages a virtual environment for your project. When you run Python files with `uv run`, it uses this environment with all your installed packages.

## Creating Your .env File

The `.env` file is a simple text file that stores key-value pairs. Create it in your project's root directory (the same folder as `pyproject.toml`).

Create a new file called `.env`:

```bash
touch .env
```

Open it in your editor and add your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

### .env File Structure and Best Practices

Here are the rules for `.env` files:

**Basic syntax:**

```
# This is a comment
VARIABLE_NAME=value

# No spaces around the equals sign
CORRECT=value
WRONG = value

# Quotes are optional but useful for values with spaces
SIMPLE=hello
WITH_SPACES="hello world"

# Multi-line values need quotes
LONG_VALUE="this is a
multi-line value"
```

**Best practices:**

1. **Use descriptive names**: `ANTHROPIC_API_KEY` is clearer than `API_KEY` or `KEY`
2. **Group related variables**: Add comments to organize sections
3. **Don't use quotes unless necessary**: For simple values, skip the quotes
4. **One secret per line**: Keep it readable

Here's what your `.env` file will look like as you progress through this book:

```
# Anthropic API (Chapter 4)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Weather API (Chapter 10)
WEATHER_API_KEY=your-weather-api-key

# Development settings
DEBUG=true
LOG_LEVEL=INFO
```

### Creating a .env.example File

It's helpful to create a template file that shows what variables are needed without revealing actual values. This file _can_ be committed to version control:

```bash
touch .env.example
```

Contents of `.env.example`:

```
# Copy this file to .env and fill in your actual values
# cp .env.example .env

# Required: Get your key from https://console.anthropic.com/
ANTHROPIC_API_KEY=your-api-key-here

# Optional: Weather API for Chapter 10
WEATHER_API_KEY=your-weather-api-key-here
```

This helps collaborators (or future you) know what environment variables your project needs.

## Loading Environment Variables in Python

Now let's write Python code that loads and uses these variables. If you have not created the src directory, create it and then create the config.py file:

```bash
mkdir -p src
touch src/config.py
```

Here's the pattern we'll use throughout this book:

```python
"""
Configuration and secrets loading for the agents project.

This module provides a centralized way to load and validate
environment variables needed by the application.
"""

import os
from dotenv import load_dotenv


def load_config() -> dict[str, str]:
    """
    Load environment variables from .env file and return as a dictionary.

    Returns:
        Dictionary containing configuration values.

    Raises:
        ValueError: If required environment variables are missing.
    """
    # Load variables from .env file into environment
    # This looks for .env in the current directory and parent directories
    load_dotenv()

    # Get the API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")

    # Validate that required variables are present
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. "
            "Please create a .env file with your API key. "
            "See .env.example for the required format."
        )

    return {
        "anthropic_api_key": api_key,
    }


def get_api_key() -> str:
    """
    Convenience function to get just the Anthropic API key.

    Returns:
        The Anthropic API key string.

    Raises:
        ValueError: If the API key is not configured.
    """
    config = load_config()
    return config["anthropic_api_key"]


if __name__ == "__main__":
    # Test that configuration loads correctly
    try:
        config = load_config()
        # Don't print the full key! Just confirm it exists
        key = config["anthropic_api_key"]
        print(f"✓ API key loaded successfully")
        print(f"  Key starts with: {key[:20]}...")
        print(f"  Key length: {len(key)} characters")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
```

### How load_dotenv() Works

The `load_dotenv()` function:

1. Searches for a `.env` file starting in the current directory
2. Reads the file and parses key-value pairs
3. Adds each pair to the process's environment variables
4. Does **not** override variables that already exist in the environment

This last point is important—it means you can override `.env` values by setting real environment variables, which is useful for production deployments.

### Testing Your Configuration

Run the config module to verify everything works:

```bash
uv run python src/config.py
```

If your `.env` file is set up correctly, you'll see:

```
✓ API key loaded successfully
  Key starts with: sk-ant-api03-xxxxxxxx...
  Key length: 108 characters
```

If something's wrong, you'll see a clear error message:

```
✗ Configuration error: ANTHROPIC_API_KEY not found. Please create a .env file with your API key. See .env.example for the required format.
```

## Setting Up .gitignore

This is the critical step that prevents accidental commits. Create or edit `.gitignore` in your project root:

```bash
touch .gitignore
```

Add these lines:

```
# Environment variables and secrets - NEVER COMMIT
.env
.env.local
.env.*.local

# Allow the example file
!.env.example

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
```

**Note:** This is a suggested list of files. Github creates an extensive .gitignore file for python projects. It is one that I often use. Just make sure that .env shows up in the file.

### Verify .gitignore Works

Let's make sure git is ignoring your `.env` file:

```bash
# Initialize git if you haven't already
git init

# Check what git sees
git status
```

You should see `.gitignore` in the list of untracked files, but `.env` should **not** appear. If `.env` does appear, double-check your `.gitignore` syntax.

You can also explicitly check:

```bash
git check-ignore .env
```

If it outputs `.env`, you're protected. If it outputs nothing, something's wrong with your `.gitignore`.

> **Warning:** If you already committed `.env` before setting up `.gitignore`, the damage may be done. Git remembers everything. You'll need to remove it from history using `git filter-branch` or the BFG Repo-Cleaner, and you should **rotate your API key immediately** (generate a new one and delete the old one).

## Validating Secrets Are Loaded

Let's create the file validate_env.py. It is a more comprehensive validation script that you can run anytime to check your setup:

```python
"""
Validate that all required environment variables are configured.

Run this script to check your setup:
    uv run python src/validate_env.py
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
```

Run the validation:

```bash
uv run python src/validate_env.py
```

A successful run looks like:

```
==================================================
Environment Validation
==================================================

Checking environment variables...
✓ ANTHROPIC_API_KEY: Valid (starts with sk-ant-api03-xx...)

Checking security configuration...
✓ .gitignore: Properly configured

==================================================
All checks passed! You're ready to proceed.
```

## The Reusable Pattern

Here's the pattern you'll see at the start of every code file in this book:

```python
"""
[Description of what this file does]

Chapter X: [Chapter Title]
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Rest of your code...
```

This pattern:

1. Loads variables from `.env` immediately
2. Validates critical variables before proceeding
3. Fails fast with a clear error message if something's wrong

💡 **Tip:** You might be tempted to skip validation "just for testing." Don't. The one time you skip it is the time you'll waste an hour debugging why your code doesn't work, only to realize you forgot to set up `.env` in a new directory.

## Common Pitfalls

### 1. The .env File Is in the Wrong Location

`load_dotenv()` searches for `.env` starting from the current working directory. If you run your script from a different directory, it won't find the file.

**Solution:** Always run scripts from your project root, or use an explicit path:

```python
from pathlib import Path
from dotenv import load_dotenv

# Load from a specific location
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
```

### 2. Extra Spaces or Quotes

These look similar but behave differently:

```
# This works
API_KEY=abc123

# This includes the quotes in the value!
API_KEY="abc123"   # Value is: "abc123" (with quotes)

# This includes trailing space
API_KEY=abc123     # Value might include trailing whitespace
```

**Solution:** Use simple `KEY=value` syntax and strip whitespace when reading:

```python
api_key = os.getenv("API_KEY", "").strip()
```

### 3. Forgetting to Call load_dotenv()

If your code works on your machine but fails in tests or for colleagues, check that you're actually calling `load_dotenv()`.

**Solution:** Put `load_dotenv()` at the top of your entry points, before any code that needs environment variables.

## Practical Exercise

**Task:** Set up secrets management for your project and verify it works.

**Requirements:**

1. Create a `.env` file with your Anthropic API key
2. Create a `.env.example` file documenting required variables
3. Create a `.gitignore` that excludes `.env`
4. Write a Python script that loads and validates your configuration
5. Verify that git is ignoring your `.env` file

**Hints:**

-   Get your API key from https://console.anthropic.com/
-   Use the validation script from this chapter as a starting point
-   Test by temporarily removing your `.env` file and confirming you get a clear error

**Solution:** See `code/validate_env.py` in this chapter's directory.

## Key Takeaways

-   **Never hardcode secrets**: Not even temporarily, not even for testing
-   **Use .env files**: Store secrets in `.env` and load with python-dotenv
-   **Always .gitignore your .env**: This is non-negotiable
-   **Validate early**: Check for required variables at startup, not when you need them
-   **Create .env.example**: Help others (and future you) know what variables are needed
-   **Fail fast**: Clear error messages save debugging time

> Security is not optional—build good habits from day one.

## What's Next

With your development environment configured and secrets management in place, you're ready to make your first API call to Claude. In Chapter 4, we'll send a message to the Anthropic API and get a response—the foundation of everything we'll build from there.
