---
chapter: 2
title: "Setting Up Your Development Environment"
part: 1
date: 2025-01-15
draft: false
---

# Chapter 2: Setting Up Your Development Environment

## Introduction

Nothing kills momentum faster than environment problems. You're excited to build your first AI agent, you copy some code from a tutorial, and then... `ModuleNotFoundError`. Or worse, "works on my machine" syndrome when you try to share your code with someone else.

This chapter is about preventing those headaches before they happen. We're going to set up a development environment that's fast, reliable, and reproducible. The tools we install here will serve you throughout this entire book—and for every Python project you build afterward.

In the previous chapter, we explored what AI agents are and why they matter. Now it's time to roll up our sleeves and prepare our workspace. By the end of this chapter, you'll have a fully configured project directory ready for building agents.

## Learning Objectives

By the end of this chapter, you will be able to:

-   Verify that you have Python 3.10 or higher installed
-   Install and use `uv`, a modern Python package manager
-   Create a new project with proper dependency management
-   Understand the structure of `pyproject.toml`
-   Run Python scripts within a managed environment

## Checking Your Python Setup

Before you do anything else, you need to know what Python your system
already has. Some environments (corporate laptops, managed servers,
older Linux distros) won't let you upgrade system Python---so don't
assume you can just install a new version.

First, check what's available:

```bash
python --version
```

or, on systems where python points to Python 2:

```bash
python3 --version
```

If the version is Python 3.10 or newer, you're in good shape. If it's
older---or it's completely locked down---don't worry. We'll handle that
with uv, which ships its own Python runtime.

## If You Need or Want a Newer Python

Every operating system is different, so check what versions your OS
actually supports:

### Linux:

Some distros only ship old versions. You may need to check:

```bash
apt search python3
```

or

```bash
dnf search python3
```

to see what's available.

### macOS:

Homebrew usually stays current, but check first:

```bash
brew search python
```

### Windows:

You can grab installers from python.org, but in many workplaces,
installing software is restricted.

## If You're on a Work Machine

If you work for a company, ask IT/computer support before installing a
new Python version. Many corporate laptops lock down system languages
and interpreters. IT may approve a newer Python---or they may not.

Either way, you're not stuck.

## If You Can't Change Your System Python

(use uv as your runtime)

uv includes its own Python builds and can run your project without
touching the system interpreter. That means you can keep your OS Python
exactly as--is and still run a modern Python toolchain.

You have two options:

### Option A: Run scripts directly with uv

Run Python without ever touching system Python:

```bash
uv run python script.py
```

This guarantees your script runs with uv's managed Python, not whatever
is installed on your system.

### Option B: Activate uv's virtual environment

#### macOS / Linux:

```bash
source .venv/bin/activate
```

#### Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows (CMD):

```cmd
.\.venv\Scriptsctivate.bat
```

Once the environment is active, running python script.py will use uv's
Python.

### Why Python 3.10+?

We require Python 3.10 or higher for several features we'll use throughout this book:

-   **Structural pattern matching** (`match`/`case` statements) — useful for handling different API response types
-   **Improved type hints** — including the `|` union syntax and `TypeAlias`
-   **Better error messages** — Python 3.10+ provides much clearer error tracebacks

## Installing uv

Now for the tool that will make your life significantly easier: **uv**.

`uv` is a modern Python package manager written in Rust. It's a drop-in replacement for `pip`, `pip-tools`, `poetry`, and `virtualenv`—but dramatically faster. We're talking 10-100x faster for most operations.

### Installation

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal and verify it worked:

```bash
uv --version
```

You should see something like:

```
uv 0.5.11
```

### Why uv Over pip or poetry?

You might be wondering why we're using `uv` instead of the built-in `pip` or popular alternatives like `poetry`. Here's why:

| Feature                | pip           | poetry   | uv        |
| ---------------------- | ------------- | -------- | --------- |
| Speed                  | Slow          | Moderate | Very fast |
| Lockfile               | No            | Yes      | Yes       |
| Virtual env management | Separate tool | Built-in | Built-in  |
| Dependency resolution  | Basic         | Good     | Excellent |
| Reproducible builds    | Difficult     | Yes      | Yes       |

The speed difference is not trivial. When you're iterating quickly on agent code and need to install a new package, waiting 30 seconds for pip vs. 2 seconds for uv adds up fast.

But speed isn't the only benefit. `uv` creates a **lockfile** (`uv.lock`) that records the exact versions of every package you install, including all transitive dependencies. This means anyone who clones your project will get exactly the same packages you have—no more "it works on my machine" problems.

> 💡 **Tip:** Even if you've used pip for years, give uv a genuine try. The speed alone will change how you work.

## Creating Your First Project

Let's create the project directory we'll use throughout this book. Navigate to where you keep your coding projects and run:

```bash
uv init agents-from-scratch
```

You'll see output like:

```
Initialized project `agents-from-scratch` at /path/to/agents-from-scratch
```

Now let's explore what was created:

```bash
cd agents-from-scratch
ls -la
```

You should see:

```
.
├── .git/
├── .gitignore
├── .python-version
├── README.md
├── hello.py
└── pyproject.toml
```

Let's break down each file:

-   **`.git/`** — uv automatically initializes a Git repository
-   **`.gitignore`** — Pre-configured to ignore common Python artifacts
-   **`.python-version`** — Specifies which Python version this project uses
-   **`README.md`** — A placeholder README file
-   **`hello.py`** — A sample Python file (we'll replace this)
-   **`pyproject.toml`** — The project configuration file (more on this below)

## Understanding pyproject.toml

Open `pyproject.toml` in your text editor. You'll see something like:

```toml
[project]
name = "agents-from-scratch"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

This file is the heart of your project configuration. Let's understand each section:

### The `[project]` Section

```toml
[project]
name = "agents-from-scratch"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []
```

-   **`name`** — Your project's name (used if you ever publish it as a package)
-   **`version`** — Current version number
-   **`description`** — A short description of what your project does
-   **`readme`** — Points to your README file
-   **`requires-python`** — Minimum Python version required
-   **`dependencies`** — List of packages your project needs (currently empty)

Let's update the description:

```toml
[project]
name = "agents-from-scratch"
version = "0.1.0"
description = "Building AI agents from first principles, without frameworks"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []
```

### The `[build-system]` Section

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

This tells Python how to build your project if you ever want to distribute it as a package. For our purposes, you can leave this as-is.

## Installing Your First Package

Let's install the two packages we'll use throughout this book:

```bash
uv add anthropic python-dotenv
```

You'll see something like:

```
Resolved 9 packages in 156ms
Prepared 9 packages in 412ms
Installed 9 packages in 48ms
 + anthropic==0.42.0
 + anyio==4.8.0
 + certifi==2024.12.14
 + distro==1.9.0
 + h11==0.14.0
 + httpcore==1.0.7
 + httpx==0.28.1
 + python-dotenv==1.0.1
 + sniffio==1.3.1
```

Now check your `pyproject.toml` again:

```toml
[project]
name = "agents-from-scratch"
version = "0.1.0"
description = "Building AI agents from first principles, without frameworks"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.42.0",
    "python-dotenv>=1.0.1",
]
```

Notice that `uv` automatically added the packages to your `dependencies` list. It also created a `uv.lock` file that pins the exact versions of all packages.

> ⚠️ **Warning:** Never edit `uv.lock` manually. This file is managed by uv and ensures reproducible installs.

## Running Python with uv

When you use `uv`, it manages a virtual environment for you automatically. To run a Python script, use:

```bash
uv run python hello.py
```

You should see:

```
Hello from agents-from-scratch!
```

The `uv run` command ensures your script runs with all the correct dependencies installed. You don't need to manually activate a virtual environment.

You can also start an interactive Python session:

```bash
uv run python
```

This drops you into a Python REPL with all your packages available:

```python
>>> import anthropic
>>> anthropic.__version__
'0.42.0'
>>> exit()
```

> 💡 **Tip:** If you prefer the traditional workflow of activating a virtual environment, the environment is located in `.venv/` within your project directory. You can activate it with `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows).

## Setting Up Your Directory Structure

Let's create the directory structure we'll use throughout the book. Replace the sample `hello.py` with our actual project structure:

```bash
# Remove the sample file
rm hello.py

# Create directory structure
mkdir -p src/agents/core
mkdir -p src/agents/tools
mkdir -p src/agents/workflows
mkdir -p src/agents/utils
mkdir examples
mkdir tests
```

Now create the necessary `__init__.py` files to make these proper Python packages:

```bash
touch src/agents/__init__.py
touch src/agents/core/__init__.py
touch src/agents/tools/__init__.py
touch src/agents/workflows/__init__.py
touch src/agents/utils/__init__.py
```

Your structure should now look like:

```
agents-from-scratch/
├── .git/
├── .gitignore
├── .python-version
├── .venv/
├── README.md
├── examples/
├── pyproject.toml
├── src/
│   └── agents/
│       ├── __init__.py
│       ├── core/
│       │   └── __init__.py
│       ├── tools/
│       │   └── __init__.py
│       ├── utils/
│       │   └── __init__.py
│       └── workflows/
│           └── __init__.py
├── tests/
└── uv.lock
```

## Verifying Your Setup

Let's create a simple script to verify everything is working. Create a file called `examples/verify_setup.py`:

```python
"""
Verify that the development environment is correctly configured.

Chapter 2: Setting Up Your Development Environment
"""

import sys


def check_python_version() -> bool:
    """Verify Python version is 3.10 or higher."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 10:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.10 or higher is required")
        return False


def check_anthropic_package() -> bool:
    """Verify anthropic package is installed."""
    try:
        import anthropic
        print(f"Anthropic SDK version: {anthropic.__version__}")
        print("✓ Anthropic package is installed")
        return True
    except ImportError:
        print("✗ Anthropic package is not installed")
        return False


def check_dotenv_package() -> bool:
    """Verify python-dotenv package is installed."""
    try:
        import dotenv
        print("✓ python-dotenv package is installed")
        return True
    except ImportError:
        print("✗ python-dotenv package is not installed")
        return False


def main() -> None:
    """Run all verification checks."""
    print("=" * 50)
    print("Development Environment Verification")
    print("=" * 50)
    print()

    checks = [
        check_python_version,
        check_anthropic_package,
        check_dotenv_package,
    ]

    results = [check() for check in checks]

    print()
    print("=" * 50)

    if all(results):
        print("All checks passed! Your environment is ready.")
        print("=" * 50)
    else:
        print("Some checks failed. Please review the output above.")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

Run it with:

```bash
uv run python examples/verify_setup.py
```

You should see:

```
==================================================
Development Environment Verification
==================================================

Python version: 3.12.1
✓ Python version is compatible
Anthropic SDK version: 0.42.0
✓ Anthropic package is installed
✓ python-dotenv package is installed

==================================================
All checks passed! Your environment is ready.
==================================================
```

If any checks fail, review the earlier sections of this chapter to fix the issue.

## Common Pitfalls

### 1. Forgetting to Use `uv run`

If you run `python script.py` directly instead of `uv run python script.py`, you might use your system Python instead of the project's virtual environment. This leads to "module not found" errors even when packages are installed.

**Fix:** Always use `uv run python` or activate the virtual environment first.

### 2. Editing uv.lock Manually

The lockfile looks like it might be editable, but don't do it. Manual edits can cause dependency resolution failures.

**Fix:** Use `uv add` and `uv remove` to manage packages. The lockfile will update automatically.

### 3. Not Restarting Your Terminal After Installing uv

After installing `uv`, your terminal needs to reload its PATH. If `uv --version` shows "command not found," this is likely the issue.

**Fix:** Close and reopen your terminal, or run `source ~/.bashrc` (or equivalent for your shell).

## Practical Exercise

**Task:** Extend the verification script to also check that a `.env` file exists (we'll create this in the next chapter).

**Requirements:**

-   Add a new function `check_env_file()` that returns `True` if `.env` exists in the project root
-   Print a warning (not an error) if the file doesn't exist, since we haven't created it yet
-   Include the new check in the list of checks

**Hints:**

-   Use `pathlib.Path` to check for file existence
-   The project root is the parent of the `examples/` directory

**Solution:** See `code/exercise_solution.py`

## Key Takeaways

-   **Python 3.10+ is required** for modern features like pattern matching and improved type hints
-   **uv is fast and reliable** — it manages packages, virtual environments, and lockfiles in one tool
-   **pyproject.toml is your project's configuration** — it lists dependencies and project metadata
-   **Always use `uv run`** to execute scripts with the correct dependencies
-   **The lockfile ensures reproducibility** — anyone with your code gets exactly the same packages

## What's Next

Your development environment is ready, but we're not quite done with setup. In Chapter 3, we'll create the `.env` file that securely stores your API key. You'll learn why secrets management matters and how to ensure you never accidentally commit your API key to version control. This is the last setup step before we make our first call to Claude.
