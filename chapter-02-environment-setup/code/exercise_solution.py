"""
Verify that the development environment is correctly configured.
Extended version with .env file check.

Chapter 2: Setting Up Your Development Environment
Exercise Solution
"""

import sys
from pathlib import Path


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


def check_env_file() -> bool:
    """
    Check if .env file exists in the project root.

    Returns True with a warning if not found (since we haven't created it yet).
    This is a soft check - we don't want to fail the verification.
    """
    # Get the project root (two levels up from this script's directory)
    # In your project: project_root/examples/exercise_solution.py
    # In this book: chapter-02-environment-setup/code/exercise_solution.py
    # Either way: script -> [examples or code] -> project_root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    env_file = project_root / ".env"
    
    if env_file.exists():
        print(f"✓ .env file found at {env_file}")
        return True
    else:
        print(f"⚠ .env file not found at {env_file}")
        print("  (This is expected - we'll create it in Chapter 3)")
        # Return True anyway since this is just a warning
        return True


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
        check_env_file,  # Added the new check
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
