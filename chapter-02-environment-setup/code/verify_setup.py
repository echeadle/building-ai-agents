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
