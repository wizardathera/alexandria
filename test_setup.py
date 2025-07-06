#!/usr/bin/env python3
"""
Basic setup verification script for DBC app.

This script checks that the project structure is correct and
imports work properly (when dependencies are installed).
"""

import sys
from pathlib import Path
import os

def check_project_structure():
    """Check that all required directories and files exist."""
    print("🔍 Checking project structure...")
    
    required_dirs = [
        "src", "src/api", "src/utils", "src/rag", "src/mcp", 
        "src/tools", "src/prompts", "tests", "tests/fixtures",
        "data", "data/books", "data/chroma_db", "data/users"
    ]
    
    required_files = [
        "src/__init__.py", "src/main.py", 
        "src/utils/config.py", "src/utils/logger.py", "src/utils/database.py",
        "src/api/__init__.py", "src/api/health.py", "src/api/books.py", "src/api/chat.py",
        "tests/__init__.py", "tests/test_main.py",
        "requirements.txt", ".env.example", ".gitignore",
        "CLAUDE.md", "PLANNING.md", "README.md"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"  ✅ {dir_path}/")
    
    # Check files
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_dirs or missing_files:
        print(f"\n❌ Missing directories: {missing_dirs}")
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is complete!")
    return True

def check_environment():
    """Check environment configuration."""
    print("\n🔍 Checking environment configuration...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_example.exists():
        print("❌ .env.example file missing")
        return False
    
    print("✅ .env.example exists")
    
    if env_file.exists():
        print("✅ .env file exists")
        # Check if it has content
        if env_file.stat().st_size > 0:
            print("✅ .env file has content")
        else:
            print("⚠️  .env file is empty - you may need to copy from .env.example")
    else:
        print("⚠️  .env file missing - copy from .env.example and add your API keys")
    
    return True

def check_python_version():
    """Check Python version compatibility."""
    print("\n🔍 Checking Python version...")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 9):
        print("✅ Python version is compatible (3.9+)")
        return True
    else:
        print("❌ Python version too old. Requires 3.9 or higher")
        return False

def main():
    """Run all setup checks."""
    print("🚀 DBC App Setup Verification")
    print("=" * 40)
    
    checks = [
        check_python_version(),
        check_project_structure(),
        check_environment()
    ]
    
    print("\n" + "=" * 40)
    
    if all(checks):
        print("🎉 Setup verification complete! All checks passed.")
        print("\nNext steps:")
        print("1. Create virtual environment: python3 -m venv venv")
        print("2. Activate environment: source venv/bin/activate")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Copy .env.example to .env and add your OpenAI API key")
        print("5. Run tests: pytest")
        print("6. Start server: uvicorn src.main:app --reload")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)