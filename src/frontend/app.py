"""
Main entry point for Alexandria Streamlit frontend application.

This module provides the main navigation, page routing, and application
initialization for the Alexandria frontend.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path for component imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.frontend.core.state import init_session_state
from src.frontend.core.config import API_BASE_URL
from src.frontend.components.themes import (
    apply_theme,
    get_theme_color
)
from src.frontend.components.modules import (
    render_module_aware_sidebar, 
    render_permission_aware_interface,
    render_module_breadcrumb
)

# Import page modules (we'll create these incrementally)
from src.frontend.pages.analytics import render_analytics_dashboard
from src.frontend.pages.relationships import render_content_relationships
from src.frontend.pages.book_management import render_book_management
from src.frontend.pages.qa_chat import render_qa_chat
from src.frontend.pages.search import render_enhanced_search


def main():
    """Main application entry point."""
    # Page configuration must come first
    st.set_page_config(
        page_title="Dynamic Book Companion",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Apply comprehensive theme system
    apply_theme(st.session_state.selected_theme)
    
    # Additional custom CSS for app-specific styling (themes handle main styling)
    st.markdown(f"""
    <style>
    .main-header {{
        text-align: center;
        padding: 2rem 0;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: {get_theme_color('primary')};
    }}
    
    .feature-box {{
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid {get_theme_color('border')};
        margin: 0.5rem 0;
        background-color: {get_theme_color('surface')};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🌟 Alexandria Platform</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered multi-module platform for reading, learning, and content creation*")
    
    # Render module-aware sidebar and get current selection
    current_selection = render_module_aware_sidebar(API_BASE_URL)
    current_module = current_selection["module"]
    current_page = current_selection["page"]
    
    # Get user permissions and render permission-aware interface
    user_permissions = st.session_state.user_permissions
    features = render_permission_aware_interface(user_permissions, current_module, current_page)
    
    # Render breadcrumb navigation
    render_module_breadcrumb(current_module, current_page)
    
    # Module and page routing (simplified for now)
    if current_module == "library":
        if current_page == "book_management":
            render_book_management()
        elif current_page == "enhanced_search":
            render_enhanced_search()
        elif current_page == "qa_chat":
            render_qa_chat()
        elif current_page == "relationships":
            render_content_relationships()
        elif current_page == "analytics":
            render_analytics_dashboard()
    elif current_module == "lms":
        st.title("🎓 Learning Suite Module")
        st.info("📋 **Coming in Phase 2.0**: Learning Management System features")
    elif current_module == "marketplace":
        st.title("🏪 Marketplace Module")
        st.info("📋 **Coming in Phase 3.0**: Content monetization and community features")
    else:
        st.error(f"Unknown module: {current_module}")
        st.info("Defaulting to Library module...")
        st.title("📖 Book Management")
        st.info("🚧 Book Management page is being refactored. Check back soon!")


if __name__ == "__main__":
    main()