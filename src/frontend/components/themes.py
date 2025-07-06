"""
Theme management system for DBC Streamlit frontend.

This module provides comprehensive theme support with:
- Multiple predefined themes (Light, Dark, Alexandria Classic)
- CSS styling for consistent appearance
- Theme persistence across sessions
- Foundation for Phase 2 advanced theming
"""

import streamlit as st
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

# Theme configuration directory
THEME_CONFIG_DIR = Path.home() / ".dbc" / "themes"
THEME_CONFIG_FILE = THEME_CONFIG_DIR / "user_preferences.json"

# Theme definitions with complete styling
THEME_DEFINITIONS = {
    "Light": {
        "name": "Light",
        "description": "Clean, bright theme for daytime reading",
        "colors": {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e", 
            "background": "#ffffff",
            "surface": "#f8f9fa",
            "text_primary": "#212529",
            "text_secondary": "#6c757d",
            "accent": "#28a745",
            "border": "#dee2e6",
            "highlight": "#fff3cd",
            "error": "#dc3545",
            "success": "#28a745",
            "warning": "#ffc107",
            "info": "#17a2b8"
        },
        "typography": {
            "font_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "heading_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "base_size": "16px",
            "heading_weight": "600",
            "body_weight": "400"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem"
        }
    },
    "Dark": {
        "name": "Dark",
        "description": "Comfortable dark theme for low-light reading",
        "colors": {
            "primary": "#4dabf7",
            "secondary": "#ffd43b",
            "background": "#212529",
            "surface": "#343a40",
            "text_primary": "#f8f9fa",
            "text_secondary": "#adb5bd",
            "accent": "#20c997",
            "border": "#495057",
            "highlight": "#495057",
            "error": "#e74c3c",
            "success": "#2ecc71",
            "warning": "#f39c12",
            "info": "#3498db"
        },
        "typography": {
            "font_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "heading_family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "base_size": "16px",
            "heading_weight": "600",
            "body_weight": "400"
        },
        "spacing": {
            "base": "1rem",
            "small": "0.5rem",
            "large": "2rem"
        }
    },
    "Alexandria Classic": {
        "name": "Alexandria Classic",
        "description": "Elegant library-inspired theme with warm tones",
        "colors": {
            "primary": "#8b4513",
            "secondary": "#daa520",
            "background": "#faf8f3",
            "surface": "#f5f2e8",
            "text_primary": "#3c2415",
            "text_secondary": "#6b4e3d",
            "accent": "#cd853f",
            "border": "#d4c5a9",
            "highlight": "#fff8dc",
            "error": "#a0522d",
            "success": "#556b2f",
            "warning": "#b8860b",
            "info": "#4682b4"
        },
        "typography": {
            "font_family": "Crimson Text, Georgia, serif",
            "heading_family": "Playfair Display, Georgia, serif",
            "base_size": "17px",
            "heading_weight": "700",
            "body_weight": "400"
        },
        "spacing": {
            "base": "1.2rem",
            "small": "0.6rem",
            "large": "2.4rem"
        }
    }
}

class ThemeManager:
    """Manages theme selection, persistence, and application."""
    
    def __init__(self):
        """Initialize theme manager and load user preferences."""
        self.ensure_config_directory()
        self.current_theme = self.load_user_theme_preference()
    
    def ensure_config_directory(self) -> None:
        """Ensure the theme configuration directory exists."""
        THEME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_user_theme_preference(self) -> str:
        """Load user's saved theme preference or return default."""
        try:
            if THEME_CONFIG_FILE.exists():
                with open(THEME_CONFIG_FILE, 'r') as f:
                    preferences = json.load(f)
                    theme_name = preferences.get('selected_theme', 'Light')
                    if theme_name in THEME_DEFINITIONS:
                        return theme_name
        except (json.JSONDecodeError, IOError):
            pass
        return 'Light'  # Default theme
    
    def save_user_theme_preference(self, theme_name: str) -> None:
        """Save user's theme preference to disk."""
        try:
            preferences = {}
            if THEME_CONFIG_FILE.exists():
                with open(THEME_CONFIG_FILE, 'r') as f:
                    preferences = json.load(f)
            
            preferences['selected_theme'] = theme_name
            preferences['last_updated'] = str(st.session_state.get('timestamp', 'unknown'))
            
            with open(THEME_CONFIG_FILE, 'w') as f:
                json.dump(preferences, f, indent=2)
        except IOError:
            st.warning("Could not save theme preference to disk.")
    
    def get_available_themes(self) -> Dict[str, Dict[str, Any]]:
        """Get all available theme definitions."""
        return THEME_DEFINITIONS
    
    def get_theme_definition(self, theme_name: str) -> Optional[Dict[str, Any]]:
        """Get the definition for a specific theme."""
        return THEME_DEFINITIONS.get(theme_name)
    
    def set_current_theme(self, theme_name: str) -> bool:
        """Set the current theme and persist the preference."""
        if theme_name in THEME_DEFINITIONS:
            self.current_theme = theme_name
            self.save_user_theme_preference(theme_name)
            return True
        return False
    
    def get_current_theme_definition(self) -> Dict[str, Any]:
        """Get the current theme's complete definition."""
        return THEME_DEFINITIONS.get(self.current_theme, THEME_DEFINITIONS['Light'])
    
    def generate_css(self, theme_name: Optional[str] = None) -> str:
        """Generate CSS styles for the specified theme."""
        if theme_name is None:
            theme_name = self.current_theme
        
        theme = THEME_DEFINITIONS.get(theme_name, THEME_DEFINITIONS['Light'])
        colors = theme['colors']
        typography = theme['typography']
        spacing = theme['spacing']
        
        css = f"""
        <style>
        /* DBC Theme: {theme['name']} */
        
        /* Root variables for theme colors */
        :root {{
            --dbc-primary: {colors['primary']};
            --dbc-secondary: {colors['secondary']};
            --dbc-background: {colors['background']};
            --dbc-surface: {colors['surface']};
            --dbc-text-primary: {colors['text_primary']};
            --dbc-text-secondary: {colors['text_secondary']};
            --dbc-accent: {colors['accent']};
            --dbc-border: {colors['border']};
            --dbc-highlight: {colors['highlight']};
            --dbc-error: {colors['error']};
            --dbc-success: {colors['success']};
            --dbc-warning: {colors['warning']};
            --dbc-info: {colors['info']};
            
            --dbc-font-family: {typography['font_family']};
            --dbc-heading-family: {typography['heading_family']};
            --dbc-base-size: {typography['base_size']};
            --dbc-heading-weight: {typography['heading_weight']};
            --dbc-body-weight: {typography['body_weight']};
            
            --dbc-spacing-base: {spacing['base']};
            --dbc-spacing-small: {spacing['small']};
            --dbc-spacing-large: {spacing['large']};
        }}
        
        /* Main app background */
        .main .block-container {{
            background-color: var(--dbc-background);
            color: var(--dbc-text-primary);
            font-family: var(--dbc-font-family);
            font-size: var(--dbc-base-size);
            font-weight: var(--dbc-body-weight);
            padding: var(--dbc-spacing-base);
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: var(--dbc-surface);
            border-right: 1px solid var(--dbc-border);
        }}
        
        .css-1d391kg .css-10trblm {{
            color: var(--dbc-text-primary);
        }}
        
        /* Headers and titles */
        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--dbc-heading-family);
            font-weight: var(--dbc-heading-weight);
            color: var(--dbc-text-primary);
            margin-bottom: var(--dbc-spacing-small);
        }}
        
        /* Primary buttons */
        .stButton > button {{
            background-color: var(--dbc-primary);
            color: white;
            border: none;
            border-radius: 6px;
            padding: var(--dbc-spacing-small) var(--dbc-spacing-base);
            font-family: var(--dbc-font-family);
            font-weight: 500;
            transition: background-color 0.2s ease;
        }}
        
        .stButton > button:hover {{
            background-color: var(--dbc-accent);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        /* Secondary/outline buttons */
        .stButton > button[kind="secondary"] {{
            background-color: transparent;
            color: var(--dbc-primary);
            border: 1px solid var(--dbc-primary);
        }}
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {{
            background-color: var(--dbc-background);
            color: var(--dbc-text-primary);
            border: 1px solid var(--dbc-border);
            font-family: var(--dbc-font-family);
        }}
        
        /* Metrics and info boxes */
        .metric-container {{
            background-color: var(--dbc-surface);
            border: 1px solid var(--dbc-border);
            border-radius: 8px;
            padding: var(--dbc-spacing-base);
        }}
        
        /* Chat messages */
        .chat-message {{
            background-color: var(--dbc-surface);
            border-left: 3px solid var(--dbc-primary);
            padding: var(--dbc-spacing-base);
            margin: var(--dbc-spacing-small) 0;
            border-radius: 4px;
        }}
        
        .chat-message.user {{
            border-left-color: var(--dbc-secondary);
            background-color: var(--dbc-highlight);
        }}
        
        /* Tables */
        .dataframe {{
            background-color: var(--dbc-background);
            color: var(--dbc-text-primary);
            border: 1px solid var(--dbc-border);
        }}
        
        .dataframe th {{
            background-color: var(--dbc-surface);
            color: var(--dbc-text-primary);
            font-weight: var(--dbc-heading-weight);
        }}
        
        /* Progress bars */
        .stProgress > div > div > div {{
            background-color: var(--dbc-primary);
        }}
        
        /* Success/warning/error styling */
        .stSuccess {{
            background-color: var(--dbc-success);
            color: white;
        }}
        
        .stWarning {{
            background-color: var(--dbc-warning);
            color: white;
        }}
        
        .stError {{
            background-color: var(--dbc-error);
            color: white;
        }}
        
        .stInfo {{
            background-color: var(--dbc-info);
            color: white;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: var(--dbc-surface);
            color: var(--dbc-text-primary);
            border: 1px solid var(--dbc-border);
        }}
        
        .streamlit-expanderContent {{
            background-color: var(--dbc-background);
            border: 1px solid var(--dbc-border);
            border-top: none;
        }}
        
        /* File uploader */
        .uploadedFile {{
            background-color: var(--dbc-surface);
            border: 1px solid var(--dbc-border);
        }}
        
        /* Code blocks */
        code {{
            background-color: var(--dbc-surface);
            color: var(--dbc-text-secondary);
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        pre {{
            background-color: var(--dbc-surface);
            color: var(--dbc-text-primary);
            border: 1px solid var(--dbc-border);
            border-radius: 6px;
            padding: var(--dbc-spacing-base);
        }}
        
        /* Theme-specific enhancements */
        """
        
        # Add theme-specific enhancements
        if theme_name == "Alexandria Classic":
            css += """
            /* Alexandria Classic theme enhancements */
            .main .block-container {
                background-image: linear-gradient(45deg, transparent 25%, rgba(139,69,19,0.02) 25%, rgba(139,69,19,0.02) 50%, transparent 50%, transparent 75%, rgba(139,69,19,0.02) 75%);
                background-size: 20px 20px;
            }
            
            h1::before {
                content: "📚 ";
            }
            
            .metric-container {
                box-shadow: 0 2px 8px rgba(139,69,19,0.1);
            }
            """
        elif theme_name == "Dark":
            css += """
            /* Dark theme enhancements */
            .main .block-container {
                background-image: radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0);
                background-size: 20px 20px;
            }
            
            .metric-container {
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
            """
        else:  # Light theme
            css += """
            /* Light theme enhancements */
            .metric-container {
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            """
        
        css += """
        </style>
        """
        
        return css

def get_theme_manager() -> ThemeManager:
    """Get or create the global theme manager instance."""
    if 'theme_manager' not in st.session_state:
        st.session_state.theme_manager = ThemeManager()
    return st.session_state.theme_manager

def apply_theme(theme_name: Optional[str] = None) -> None:
    """Apply the specified theme to the current Streamlit app."""
    theme_manager = get_theme_manager()
    
    if theme_name and theme_name != theme_manager.current_theme:
        theme_manager.set_current_theme(theme_name)
    
    # Inject CSS into the page
    css = theme_manager.generate_css()
    st.markdown(css, unsafe_allow_html=True)

def render_theme_selector() -> str:
    """Render theme selector and return selected theme."""
    theme_manager = get_theme_manager()
    available_themes = theme_manager.get_available_themes()
    
    st.subheader("🎨 Theme Selection")
    
    # Create theme selector with descriptions
    theme_options = list(available_themes.keys())
    current_index = theme_options.index(theme_manager.current_theme) if theme_manager.current_theme in theme_options else 0
    
    selected_theme = st.selectbox(
        "Choose Theme",
        theme_options,
        index=current_index,
        format_func=lambda x: f"{x} - {available_themes[x]['description']}",
        key="theme_selector_main"
    )
    
    # Show theme preview
    if selected_theme:
        theme_def = available_themes[selected_theme]
        with st.expander(f"Preview: {selected_theme}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Colors:**")
                colors = theme_def['colors']
                for key, value in colors.items():
                    st.write(f"• {key.replace('_', ' ').title()}: `{value}`")
            
            with col2:
                st.write("**Typography:**")
                typography = theme_def['typography']
                for key, value in typography.items():
                    st.write(f"• {key.replace('_', ' ').title()}: `{value}`")
    
    return selected_theme

def init_theme_system() -> None:
    """Initialize theme system in Streamlit session state."""
    if 'theme_initialized' not in st.session_state:
        theme_manager = get_theme_manager()
        
        # Load user's saved theme preference
        saved_theme = theme_manager.load_user_theme_preference()
        st.session_state.selected_theme = saved_theme
        theme_manager.current_theme = saved_theme
        
        st.session_state.theme_initialized = True

def get_current_theme() -> Dict[str, Any]:
    """Get the current theme definition."""
    theme_manager = get_theme_manager()
    return theme_manager.get_current_theme_definition()

def get_theme_color(color_name: str) -> str:
    """Get a specific color from the current theme."""
    current_theme = get_current_theme()
    return current_theme.get('colors', {}).get(color_name, '#000000')