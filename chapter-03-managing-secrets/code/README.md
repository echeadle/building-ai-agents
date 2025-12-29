# Chapter 3: Code Examples

This directory contains the reference code examples for Chapter 3: Managing Secrets with python-dotenv.

**Note:** In the chapter, you'll create your own project (`agents-from-scratch`) and build these files in your `src/` directory. The files here are reference implementations you can compare against.

## Files

### `config.py`

A centralized configuration module that loads and validates environment variables from a `.env` file.

**What it does:**
- Loads environment variables using `python-dotenv`
- Validates that required variables (like `ANTHROPIC_API_KEY`) are present
- Provides convenient accessor functions for configuration values
- Fails fast with clear error messages if configuration is missing

**Usage:**
```python
from config import load_config, get_api_key

# Get all configuration
config = load_config()
api_key = config["anthropic_api_key"]

# Or just get the API key
api_key = get_api_key()
```

**Testing the module:**
```bash
uv run python config.py
```

### `validate_env.py`

A comprehensive validation script that checks your environment setup for common issues.

**What it checks:**
- `ANTHROPIC_API_KEY` is present and valid format
- API key doesn't contain placeholder text
- `.gitignore` file exists and includes `.env`

**Usage:**
```bash
uv run python validate_env.py
```

**Example output:**
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

### `.env.example`

A template file showing the required environment variables for this project.

**What to do:**
1. Copy this file to `.env` in your project root:
   ```bash
   cp .env.example .env
   ```
2. Replace placeholder values with your actual API keys
3. Never commit the `.env` file to version control

**Important:** This `.env.example` file can be committed to git. The actual `.env` file with your secrets should NEVER be committed.

## Security Best Practices

1. **Always use `.env` files** - Never hardcode API keys in your source code
2. **Add `.env` to `.gitignore`** - Prevent accidental commits
3. **Use `.env.example`** - Document what variables are needed without exposing secrets
4. **Validate early** - Use `validate_env.py` to catch configuration issues before they cause problems
5. **Rotate compromised keys** - If you accidentally commit a key, rotate it immediately

## Running the Code

Make sure you've followed the chapter instructions to:

1. Install `python-dotenv` with `uv add python-dotenv`
2. Create a `.env` file with your `ANTHROPIC_API_KEY`
3. Add `.env` to your `.gitignore`

Then run any script using `uv run python <script_name>.py`.

## Common Issues

**"ANTHROPIC_API_KEY not found"**
- Make sure you created a `.env` file (not just `.env.example`)
- Verify the `.env` file is in your project root
- Check that you're running the script from the project root directory

**"Invalid format" error**
- Anthropic API keys start with `sk-ant-`
- Make sure you copied the entire key with no extra spaces

**Scripts can't find `.env` file**
- Ensure `.env` is in the same directory as `pyproject.toml`
- Run scripts from the project root: `uv run python src/config.py`
