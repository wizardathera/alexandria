"""
Integration test for frontend components.

This test validates that the frontend components integrate properly
and can be instantiated without external dependencies.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_component_instantiation():
    """Test that components can be instantiated without external API calls."""
    print("🧩 Testing component instantiation...")
    
    try:
        # Mock streamlit and other dependencies
        import unittest.mock as mock
        
        with mock.patch.dict('sys.modules', {
            'streamlit': mock.MagicMock(),
            'pandas': mock.MagicMock(),
            'plotly.express': mock.MagicMock(),
            'plotly.graph_objects': mock.MagicMock(),
            'plotly.subplots': mock.MagicMock(),
            'networkx': mock.MagicMock(),
        }):
            # Test search component
            from src.frontend.components.search import AdvancedSearchComponent
            search_component = AdvancedSearchComponent("http://test-api")
            print("✅ Search component instantiated")
            
            # Test relationships component
            from src.frontend.components.relationships import RelationshipVisualizationComponent
            relationships_component = RelationshipVisualizationComponent("http://test-api")
            print("✅ Relationships component instantiated")
            
            return True
            
    except Exception as e:
        print(f"❌ Component instantiation failed: {e}")
        return False

def test_frontend_structure():
    """Test that the frontend structure is correct."""
    print("\n📁 Testing frontend structure...")
    
    expected_files = [
        "src/frontend/__init__.py",
        "src/frontend/app.py",
        "src/frontend/components/__init__.py",
        "src/frontend/components/search.py",
        "src/frontend/components/relationships.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ Frontend structure is complete")
    return True

def test_documentation_updated():
    """Test that documentation has been updated for Phase 1.41."""
    print("\n📝 Testing documentation updates...")
    
    # Check README.md for frontend references
    try:
        with open("README.md", 'r') as f:
            readme_content = f.read()
        
        if "streamlit run src/frontend/app.py" in readme_content:
            print("✅ README.md updated with frontend instructions")
        else:
            print("❌ README.md missing frontend instructions")
            return False
            
        if "Enhanced Streamlit frontend" in readme_content:
            print("✅ README.md mentions enhanced frontend")
        else:
            print("❌ README.md missing enhanced frontend description")
            return False
            
        return True
        
    except FileNotFoundError:
        print("❌ README.md not found")
        return False

def test_requirements_complete():
    """Test that all frontend requirements are included."""
    print("\n📦 Testing requirements completeness...")
    
    frontend_packages = [
        "streamlit",
        "pandas",
        "plotly", 
        "networkx"
    ]
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read().lower()
        
        missing_packages = []
        for package in frontend_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        else:
            print("✅ All frontend packages included")
            return True
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def main():
    """Main integration test execution."""
    print("🔗 FRONTEND INTEGRATION TESTING")
    print("="*40)
    
    test_results = {}
    
    test_results["Frontend Structure"] = test_frontend_structure()
    test_results["Component Instantiation"] = test_component_instantiation()
    test_results["Documentation Updated"] = test_documentation_updated()
    test_results["Requirements Complete"] = test_requirements_complete()
    
    print("\n" + "="*40)
    print("📊 INTEGRATION TEST REPORT")
    print("="*40)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nSuccess Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All integration tests passed!")
        print("\n✅ Phase 1.41 Frontend Implementation Complete")
        print("\n🚀 Ready for deployment:")
        print("   1. Backend: uvicorn src.main:app --reload")
        print("   2. Frontend: streamlit run src/frontend/app.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed.")
    
    print("="*40)

if __name__ == "__main__":
    main()