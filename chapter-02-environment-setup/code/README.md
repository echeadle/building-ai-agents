# Chapter 2: Code Examples

This directory contains the reference code examples for Chapter 2: Setting Up Your Development Environment.

**Note:** In the chapter, you'll create your own project (`agents-from-scratch`) and build these scripts in your `examples/` directory. The files here are reference implementations you can compare against.

## Files

### `verify_setup.py`

A script that verifies your development environment is correctly configured. It checks:

- Python version is 3.10 or higher
- The `anthropic` package is installed
- The `python-dotenv` package is installed

**Usage:**
```bash
uv run python verify_setup.py
```

### `exercise_solution.py`

The solution to the chapter's practical exercise. This extends `verify_setup.py` with an additional check for the `.env` file.

**What it adds:**
- A `check_env_file()` function that looks for `.env` in the project root
- Uses `pathlib.Path` to locate the project root relative to the script
- Prints a warning (not an error) if the file doesn't exist

**Usage:**
```bash
uv run python exercise_solution.py
```

## Running the Code

Make sure you've followed the chapter instructions to:

1. Install `uv`
2. Initialize your project with `uv init`
3. Install dependencies with `uv add anthropic python-dotenv`

Then run scripts using `uv run python <script_name>.py`.
