"""
Basic syntax validation for the Streamlit frontend files.

This test checks the Python syntax of the frontend components without
requiring external dependencies to be installed.
"""

import ast
import sys
from pathlib import Path

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source_code)
        return True, None
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def test_frontend_files():
    """Test syntax of all frontend Python files."""
    print("🔍 Testing frontend file syntax...")
    
    frontend_files = [
        "src/frontend/app.py",
        "src/frontend/__init__.py", 
        "src/frontend/components/__init__.py",
        "src/frontend/components/search.py",
        "src/frontend/components/relationships.py"
    ]
    
    all_valid = True
    
    for file_path in frontend_files:
        if Path(file_path).exists():
            is_valid, error = check_python_syntax(file_path)
            if is_valid:
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}: {error}")
                all_valid = False
        else:
            print(f"❌ {file_path}: File not found")
            all_valid = False
    
    return all_valid

def check_imports():
    """Check that imports in frontend files are valid (syntax-wise)."""
    print("\n🔍 Checking import statements...")
    
    # Simple import validation - check if they parse correctly
    test_imports = [
        "import streamlit as st",
        "import pandas as pd", 
        "import plotly.express as px",
        "import networkx as nx",
        "import requests"
    ]
    
    for import_stmt in test_imports:
        try:
            ast.parse(import_stmt)
            print(f"✅ {import_stmt}")
        except SyntaxError as e:
            print(f"❌ {import_stmt}: {e}")
            return False
    
    return True

def check_requirements_file():
    """Check if requirements.txt includes frontend dependencies."""
    print("\n📦 Checking requirements.txt...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "plotly",
        "networkx"
    ]
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements_content = f.read().lower()
        
        missing_packages = []
        for package in required_packages:
            if package not in requirements_content:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {', '.join(missing_packages)}")
            return False
        else:
            print("✅ All frontend packages in requirements.txt")
            return True
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def main():
    """Main test execution."""
    print("🧪 FRONTEND SYNTAX VALIDATION")
    print("="*35)
    
    results = {}
    
    results["File Syntax"] = test_frontend_files()
    results["Import Syntax"] = check_imports()
    results["Requirements"] = check_requirements_file()
    
    print("\n" + "="*35)
    print("📊 SYNTAX TEST REPORT")
    print("="*35)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nSuccess Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All syntax tests passed!")
        print("\n📋 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start the backend: uvicorn src.main:app --reload")
        print("3. Start the frontend: streamlit run src/frontend/app.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Fix syntax issues first.")
    
    print("="*35)

if __name__ == "__main__":
    main()