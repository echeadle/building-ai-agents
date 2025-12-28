---
chapter: 2
date: 2025-01-15
draft: false
part: 1
title: Setting Up Your Development Environment
---

# Chapter 2: Setting Up Your Development Environment

## Introduction

System Python is not your friend. It varies by OS, it's often outdated,
and on corporate machines you may not be allowed to touch it at all.
Good Python developers don't depend on system Python --- they isolate
their work inside a **project-level virtual environment**. That way your
tools, your interpreter, and your dependencies are consistent
everywhere.

This chapter shows you how to build a clean, reproducible environment
for all the agent code in this book. We'll use **uv**, a modern package
and environment manager that ships with its own Python runtime. That
means:

-   You don't need to install a new version of Python.
-   You don't need admin rights.
-   You won't break your operating system.
-   You'll always run the exact interpreter you intend to run.

By the end of this chapter, you'll have a fully configured project
folder, a reproducible environment, and a reliable way to run your code
without ever touching system Python.

## Learning Objectives

You will learn how to:

-   Use `uv` to create and manage a virtual environment automatically\
-   Run scripts using uv's Python runtime\
-   Verify you're using the correct interpreter (not the system one)\
-   Understand and manage `pyproject.toml`\
-   Install and lock dependencies with `uv`

## Why We Do NOT Use System Python

System Python is for the operating system --- not for development.

Relying on it causes problems:

-   OS updates can silently change your Python version\
-   IT restrictions can block installs\
-   Some distros still ship old Python versions\
-   "It works on my machine" becomes inevitable

Instead, every professional Python project should:

1.  Create an isolated environment\
2.  Use a dedicated interpreter inside that environment\
3.  Never rely on or modify system Python

`uv` takes care of all of this automatically.

## Creating Your Project (uv Handles Everything)

Navigate to your development directory and run:

```bash
uv init agents-from-scratch
```

This creates:

    .
    ├── .git/
    ├── .gitignore
    ├── .python-version
    ├── .venv/
    ├── README.md
    ├── main.py
    └── pyproject.toml

The important part is:

    .venv/

This is your isolated Python environment --- created by uv
automatically.

## Always Run Python Through uv

There are two correct ways to run your code:

### **Option A: uv run (recommended)**

```bash
uv run python script.py
```

This guarantees:

-   You're using uv's managed Python\
-   The environment is activated automatically\
-   Dependencies are loaded correctly

This is the safest, cleanest way to run any script.

### **Option B: Activate the virtual environment**

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows (CMD):

```cmd
.\.venv\Scriptsctivate.bat
```

After activation:

```bash
which python
```

macOS/Linux should show:

    .../agents-from-scratch/.venv/bin/python

Windows should show:

    ...\agents-from-scratch\.venv\Scripts\python.exe

If the path does NOT point into `.venv`, you're not in the right
interpreter.

**Never** run:

    python script.py

unless you've activated `.venv`.

## Understanding `pyproject.toml`

Open the file:

```toml
[project]
name = "agents-from-scratch"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []
```

This defines:

-   Project metadata\
-   Python version requirement (for uv's runtime)\
-   Dependencies list

Whenever you add packages, uv updates this file automatically.

## Installing Your First Package

Install the first library used in this book:

```bash
uv add anthropic
```

`uv` will:

-   Resolve dependencies\
-   Install everything into `.venv`\
-   Update `pyproject.toml`\
-   Generate a reproducible `uv.lock` file

No pip, no virtualenv, no poetry --- uv does all of it.

## Running Code in a Clean Environment

To execute your first script:

```bash
uv run python main.py
```

or, if you activated `.venv`:

```bash
python main.py
```

The key is that you're using the `.venv` interpreter, not system Python.

## Project Directory Structure

Replace the sample `main.py` and build your project structure:

```bash
rm main.py

mkdir -p src/agents/core
mkdir -p src/agents/tools
mkdir -p src/agents/workflows
mkdir -p src/agents/utils
mkdir examples
mkdir tests
```

And create module initialization files:

```bash
touch src/agents/__init__.py
touch src/agents/core/__init__.py
touch src/agents/tools/__init__.py
touch src/agents/workflows/__init__.py
touch src/agents/utils/__init__.py
```

Your tree now looks like:

    agents-from-scratch/
    ├── .venv/
    ├── pyproject.toml
    ├── README.md
    ├── uv.lock
    ├── src/
    │   └── agents/
    │       ├── core/
    │       ├── tools/
    │       ├── utils/
    │       └── workflows/
    ├── examples/
    └── tests/

## Verifying Your Setup

Create `examples/verify_setup.py`:

```python
import sys

def main():
    print("Python executable:", sys.executable)
    print("Version:", sys.version)
    print("✓ Environment is active and working")

if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python examples/verify_setup.py
```

You should see a path that points into `.venv`.

## Common Pitfalls

### 1. Running system Python by accident

Fix:

```bash
which python
```

If it doesn't point to `.venv`, you're using the wrong interpreter.

### 2. Forgetting to use uv run

Fix:

```bash
uv run python script.py
```

### 3. Editing `uv.lock` manually

Don't. Let uv manage it.

## Key Takeaways

-   Never use system Python for development\
-   uv provides its own Python runtime\
-   uv manages virtual environments automatically\
-   Always run your scripts with `uv run python`\
-   Use `which python` to verify the correct interpreter\
-   Your environment is fully reproducible

## What's Next

Next, we'll create a `.env` file to securely store your API keys. This
prepares you for making your first real API call.
