#!/usr/bin/env python3
"""
Test script for Alexandria theme system functionality.
This script tests the theme system without requiring full application dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_theme_definitions():
    """Test theme definitions structure."""
    print("🧪 Testing theme definitions...")
    
    # Import theme module directly (avoid __init__.py imports)
    import importlib.util
    spec = importlib.util.spec_from_file_location("themes", "src/frontend/components/themes.py")
    themes_module = importlib.util.module_from_spec(spec)
    
    # Mock streamlit before importing
    import sys
    class MockStreamlit:
        def __init__(self):
            self.session_state = {}
        def warning(self, message):
            print(f"Warning: {message}")
    
    sys.modules['streamlit'] = MockStreamlit()
    
    spec.loader.exec_module(themes_module)
    THEME_DEFINITIONS = themes_module.THEME_DEFINITIONS
    
    # Check theme definitions exist
    assert len(THEME_DEFINITIONS) == 3, f"Expected 3 themes, got {len(THEME_DEFINITIONS)}"
    
    expected_themes = ["Light", "Dark", "Alexandria Classic"]
    for theme_name in expected_themes:
        assert theme_name in THEME_DEFINITIONS, f"Theme '{theme_name}' not found"
        
        theme = THEME_DEFINITIONS[theme_name]
        
        # Check required sections
        assert "name" in theme, f"Theme {theme_name} missing 'name'"
        assert "description" in theme, f"Theme {theme_name} missing 'description'"
        assert "colors" in theme, f"Theme {theme_name} missing 'colors'"
        assert "typography" in theme, f"Theme {theme_name} missing 'typography'"
        assert "spacing" in theme, f"Theme {theme_name} missing 'spacing'"
        
        # Check color definitions
        required_colors = [
            "primary", "secondary", "background", "surface", "text_primary",
            "text_secondary", "accent", "border", "highlight", "error", "success"
        ]
        
        for color in required_colors:
            assert color in theme["colors"], f"Theme {theme_name} missing color '{color}'"
        
        # Check typography definitions
        required_typography = ["font_family", "heading_family", "base_size"]
        
        for typo in required_typography:
            assert typo in theme["typography"], f"Theme {theme_name} missing typography '{typo}'"
    
    print("✅ Theme definitions test passed!")

def test_theme_manager():
    """Test ThemeManager functionality."""
    print("🧪 Testing ThemeManager...")
    
    # Mock streamlit session state
    class MockSessionState:
        def __init__(self):
            self.data = {}
        
        def get(self, key, default=None):
            return self.data.get(key, default)
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __contains__(self, key):
            return key in self.data
    
    # Mock streamlit module
    class MockStreamlit:
        def __init__(self):
            self.session_state = MockSessionState()
        
        def warning(self, message):
            print(f"Warning: {message}")
    
    # Mock streamlit import
    import sys
    sys.modules['streamlit'] = MockStreamlit()
    
    # Import theme module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("themes", "src/frontend/components/themes.py")
    themes_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(themes_module)
    ThemeManager = themes_module.ThemeManager
    
    # Create theme manager instance
    theme_manager = ThemeManager()
    
    # Test that theme manager loads successfully (actual theme may be saved preference)
    assert theme_manager.current_theme in ["Light", "Dark", "Alexandria Classic"], f"Invalid current theme: '{theme_manager.current_theme}'"
    
    # Test available themes
    available_themes = theme_manager.get_available_themes()
    assert len(available_themes) == 3, f"Expected 3 available themes, got {len(available_themes)}"
    
    # Store original theme for restoration
    original_theme = theme_manager.current_theme
    
    # Test theme switching
    success = theme_manager.set_current_theme("Alexandria Classic")
    assert success, "Failed to set Alexandria Classic theme"
    assert theme_manager.current_theme == "Alexandria Classic", f"Expected current theme 'Alexandria Classic', got '{theme_manager.current_theme}'"
    
    # Test invalid theme
    success = theme_manager.set_current_theme("InvalidTheme")
    assert not success, "Should not successfully set invalid theme"
    assert theme_manager.current_theme == "Alexandria Classic", "Theme should remain unchanged after invalid set"
    
    # Test theme definition retrieval
    theme_def = theme_manager.get_current_theme_definition()
    assert theme_def["name"] == "Alexandria Classic", f"Expected theme definition name 'Alexandria Classic', got '{theme_def['name']}'"
    
    # Restore original theme
    theme_manager.set_current_theme(original_theme)
    
    print("✅ ThemeManager test passed!")

def test_css_generation():
    """Test CSS generation functionality."""
    print("🧪 Testing CSS generation...")
    
    # Mock streamlit
    class MockStreamlit:
        def __init__(self):
            self.session_state = {}
        
        def warning(self, message):
            print(f"Warning: {message}")
    
    import sys
    sys.modules['streamlit'] = MockStreamlit()
    
    # Import theme module directly
    import importlib.util
    spec = importlib.util.spec_from_file_location("themes", "src/frontend/components/themes.py")
    themes_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(themes_module)
    ThemeManager = themes_module.ThemeManager
    
    theme_manager = ThemeManager()
    
    # Test CSS generation for each theme
    for theme_name in ["Light", "Dark", "Alexandria Classic"]:
        css_output = theme_manager.generate_css(theme_name)
        
        # Check basic CSS structure
        assert "<style>" in css_output, f"CSS output for {theme_name} missing <style> tag"
        assert "</style>" in css_output, f"CSS output for {theme_name} missing </style> tag"
        assert f"DBC Theme: {theme_name}" in css_output, f"CSS output missing theme identifier for {theme_name}"
        
        # Check CSS variables
        assert "--dbc-primary:" in css_output, f"CSS output for {theme_name} missing primary color variable"
        assert "--dbc-background:" in css_output, f"CSS output for {theme_name} missing background color variable"
        
        # Check basic selectors
        assert ".main .block-container" in css_output, f"CSS output for {theme_name} missing main container selector"
        assert ".stButton > button" in css_output, f"CSS output for {theme_name} missing button selector"
        
        # Check theme-specific enhancements
        if theme_name == "Alexandria Classic":
            assert "📚" in css_output, f"Alexandria Classic theme missing book emoji enhancement"
        elif theme_name == "Dark":
            assert "radial-gradient" in css_output, f"Dark theme missing gradient enhancement"
    
    print("✅ CSS generation test passed!")

def main():
    """Run all theme system tests."""
    print("🚀 Alexandria Theme System Tests")
    print("=" * 50)
    
    try:
        test_theme_definitions()
        test_theme_manager()
        test_css_generation()
        
        print("\n" + "=" * 50)
        print("✅ All theme system tests passed!")
        print("📋 Phase 1.43 Theme System Implementation: COMPLETE")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)