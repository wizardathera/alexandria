"""
Test script for the enhanced Streamlit frontend functionality.

This script tests the key components and functionality of the Phase 1.41
enhanced frontend including book management, search, and relationships.
"""

import requests
import json
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_connectivity():
    """Test basic API connectivity."""
    print("🔌 Testing API connectivity...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ API connection successful!")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {e}")
        return False

def test_enhanced_content_endpoints():
    """Test enhanced content API endpoints."""
    print("\n📚 Testing enhanced content endpoints...")
    
    # Test content list endpoint
    try:
        response = requests.get(f"{API_BASE_URL}/api/enhanced/content", timeout=10)
        if response.status_code == 200:
            content_list = response.json()
            print(f"✅ Content list endpoint working - Found {len(content_list)} items")
            return True
        else:
            print(f"❌ Content list endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Content list endpoint error: {e}")
        return False

def test_enhanced_search():
    """Test enhanced search functionality."""
    print("\n🔍 Testing enhanced search...")
    
    search_data = {
        "query": "psychology",
        "n_results": 10,
        "include_relationships": True
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/enhanced/search", json=search_data, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            print(f"✅ Enhanced search working - Found {len(results)} results")
            return True
        else:
            print(f"❌ Enhanced search failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Enhanced search error: {e}")
        return False

def test_frontend_components_import():
    """Test that frontend components can be imported."""
    print("\n🧩 Testing frontend component imports...")
    
    try:
        # Test search component import
        from src.frontend.components.search import get_search_component
        search_component = get_search_component(API_BASE_URL)
        print("✅ Search component imported successfully")
        
        # Test relationships component import
        from src.frontend.components.relationships import get_relationship_component
        relationships_component = get_relationship_component(API_BASE_URL)
        print("✅ Relationships component imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Component import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

def test_streamlit_app_syntax():
    """Test that the Streamlit app has valid syntax."""
    print("\n📄 Testing Streamlit app syntax...")
    
    try:
        # Import the main app module to check for syntax errors
        from src.frontend import app
        print("✅ Streamlit app syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in Streamlit app: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error in Streamlit app: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error in Streamlit app: {e}")
        return False

def test_required_packages():
    """Test that required packages are available."""
    print("\n📦 Testing required packages...")
    
    required_packages = [
        "streamlit",
        "pandas", 
        "plotly",
        "networkx",
        "requests"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required packages available")
        return True

def generate_test_report(test_results):
    """Generate a test report."""
    print("\n" + "="*50)
    print("📊 PHASE 1.41 FRONTEND TEST REPORT")
    print("="*50)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Frontend is ready for use.")
        print("\n🚀 To start the frontend:")
        print("   streamlit run src/frontend/app.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please check the issues above.")
    
    print("="*50)

def main():
    """Main test execution."""
    print("🧪 TESTING ENHANCED STREAMLIT FRONTEND (PHASE 1.41)")
    print("="*55)
    
    test_results = {}
    
    # Run all tests
    test_results["Package Dependencies"] = test_required_packages()
    test_results["Component Imports"] = test_frontend_components_import()
    test_results["Streamlit App Syntax"] = test_streamlit_app_syntax()
    test_results["API Connectivity"] = test_api_connectivity()
    test_results["Enhanced Content API"] = test_enhanced_content_endpoints()
    test_results["Enhanced Search API"] = test_enhanced_search()
    
    # Generate report
    generate_test_report(test_results)

if __name__ == "__main__":
    main()