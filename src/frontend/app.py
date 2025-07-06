"""
Alexandria - Streamlit Frontend Application

This is the main Streamlit application providing an enhanced user interface
for the Alexandria platform with support for the enhanced RAG capabilities including
multi-module content, advanced search, and content relationships.
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to Python path for component imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.frontend.components.search import get_search_component
from src.frontend.components.relationships import get_relationship_component
from src.frontend.components.modules import (
    render_module_aware_sidebar, 
    render_permission_aware_interface,
    render_module_breadcrumb,
    get_current_user_permissions,
    create_module_aware_component,
    UserRole
)
from src.frontend.components.themes import (
    init_theme_system,
    apply_theme,
    render_theme_selector,
    get_theme_manager,
    get_current_theme,
    get_theme_color
)
from src.frontend.components.enhanced_chat import (
    EnhancedChatMessage,
    render_enhanced_message,
    render_export_options,
    render_conversation_statistics,
    initialize_enhanced_chat_session,
    clear_enhanced_chat_history,
    add_enhanced_message,
    get_enhanced_chat_history
)
from src.frontend.components.permissions import (
    render_permission_status,
    render_content_visibility_selector,
    apply_permission_filters_to_search,
    render_permission_aware_search_results,
    get_permission_manager,
    clear_permission_cache,
    ContentVisibility
)

# Configuration
API_BASE_URL = "http://localhost:8000"

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'uploaded_books' not in st.session_state:
        st.session_state.uploaded_books = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_book' not in st.session_state:
        st.session_state.current_book = None
    
    # Initialize theme system
    init_theme_system()
    
    # Set selected theme from theme manager
    if 'selected_theme' not in st.session_state:
        theme_manager = get_theme_manager()
        st.session_state.selected_theme = theme_manager.current_theme
    
    # Initialize enhanced chat system
    initialize_enhanced_chat_session()
    # Module-aware navigation state
    if 'current_module' not in st.session_state:
        st.session_state.current_module = "library"
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "book_management"
    if 'user_permissions' not in st.session_state:
        st.session_state.user_permissions = get_current_user_permissions()


def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to the DBC backend."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return {}

def upload_book_file(uploaded_file) -> Dict:
    """Upload a book file to the backend with progress tracking."""
    if not uploaded_file:
        return {}
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Prepare file
        progress_bar.progress(10)
        status_text.text("📁 Preparing file for upload...")
        
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        # Step 2: Upload file
        progress_bar.progress(30)
        status_text.text("⬆️ Uploading file to server...")
        
        response = requests.post(f"{API_BASE_URL}/api/v1/books/upload", files=files)
        response.raise_for_status()
        
        # Step 3: Processing
        progress_bar.progress(60)
        status_text.text("🔄 Processing book content...")
        
        result = response.json()
        
        # Step 4: Extracting text
        progress_bar.progress(80)
        status_text.text("📖 Extracting text and metadata...")
        
        # Step 5: Complete
        progress_bar.progress(100)
        status_text.text("✅ Upload complete!")
        
        return result
        
    except requests.exceptions.RequestException as e:
        progress_bar.progress(0)
        status_text.text("❌ Upload failed!")
        st.error(f"File upload failed: {e}")
        return {}
    finally:
        # Clean up progress indicators after a delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

def get_books_list() -> List[Dict]:
    """Get list of uploaded books from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/books")
        response.raise_for_status()
        return response.json().get("books", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch books: {e}")
        return []

def get_enhanced_content_list() -> List[Dict]:
    """Get enhanced content list with metadata from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/enhanced/content", timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out while loading your library. Please try refreshing the page.")
        return []
    except requests.exceptions.ConnectionError:
        st.error("🔌 Unable to connect to the backend service. Please check if the server is running.")
        if st.button("🔄 Retry Loading Library", key="retry_library_load"):
            st.rerun()
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch your library: {str(e)[:100]}...")
        if st.button("🔄 Retry Loading Library", key="retry_library_load_2"):
            st.rerun()
        return []

def enhanced_search(query: str, module_filter: str = None, content_type_filter: str = None, n_results: int = 10) -> List[Dict]:
    """Perform enhanced search using the enhanced content API."""
    search_data = {
        "query": query,
        "n_results": n_results,
        "include_relationships": True
    }
    
    if module_filter and module_filter != "All":
        search_data["module_filter"] = module_filter.lower()
    
    if content_type_filter and content_type_filter != "All":
        search_data["content_type_filter"] = content_type_filter.lower()
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/enhanced/search", json=search_data)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Enhanced search failed: {e}")
        return []

def get_content_relationships(content_id: str) -> List[Dict]:
    """Get content relationships for a specific content item."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/enhanced/content/{content_id}/relationships")
        response.raise_for_status()
        return response.json().get("relationships", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch relationships: {e}")
        return []

def render_sidebar():
    """Render the sidebar with navigation and theme selection."""
    with st.sidebar:
        st.title("🚀 DBC Navigator")
        
        # Theme Selection using comprehensive theme system
        selected_theme = render_theme_selector()
        
        if selected_theme != st.session_state.selected_theme:
            st.session_state.selected_theme = selected_theme
            theme_manager = get_theme_manager()
            theme_manager.set_current_theme(selected_theme)
            st.rerun()
        
        # Navigation
        st.subheader("📚 Navigation")
        page_options = [
            "📖 Book Management",
            "🔍 Enhanced Search", 
            "💬 Q&A Chat",
            "🔗 Content Relationships",
            "📊 Analytics Dashboard",
            "⚙️ Settings"
        ]
        
        selected_page = st.selectbox("Navigate to:", page_options, key="page_selector")
        
        # System Status
        st.subheader("🔧 System Status")
        try:
            health_response = requests.get(f"{API_BASE_URL}/api/v1/health")
            if health_response.status_code == 200:
                st.success("✅ Backend Connected")
            else:
                st.error("❌ Backend Error")
        except:
            st.error("❌ Backend Offline")
    
    return selected_page

@create_module_aware_component("book_management", "library", ["library"], UserRole.READER)
def render_book_management():
    """Render the enhanced book management interface."""
    st.title("📖 Enhanced Book Management")
    
    # Upload Section
    st.subheader("📤 Upload New Book")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a book file",
            type=['pdf', 'epub', 'doc', 'docx', 'txt', 'html'],
            help="Supported formats: PDF, EPUB, DOC, DOCX, TXT, HTML"
        )
    
    with col2:
        if st.button("🚀 Process Book", type="primary"):
            if uploaded_file:
                # Show file info before processing
                st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
                
                result = upload_book_file(uploaded_file)
                if result:
                    st.success(f"✅ Book '{uploaded_file.name}' uploaded successfully!")
                    
                    # Show processing result details
                    if result.get('book_id'):
                        st.info(f"📚 Book ID: {result['book_id']}")
                    if result.get('processing_status'):
                        st.info(f"🔄 Status: {result['processing_status'].title()}")
                    
                    # Auto-refresh after upload
                    import time
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to upload book")
            else:
                st.warning("Please select a file first")
    
    # Books Library
    st.subheader("📚 Your Book Library")
    
    # Get enhanced content list
    enhanced_content = get_enhanced_content_list()
    
    if enhanced_content:
        # Display as enhanced cards
        for item in enhanced_content:
            # Dynamic status indicators
            status = item.get('processing_status', 'unknown')
            status_icons = {
                'completed': '✅',
                'processing': '🔄',
                'failed': '❌',
                'pending': '⏳',
                'unknown': '❓'
            }
            status_colors = {
                'completed': 'green',
                'processing': 'orange', 
                'failed': 'red',
                'pending': 'blue',
                'unknown': 'gray'
            }
            
            status_icon = status_icons.get(status, '❓')
            status_color = status_colors.get(status, 'gray')
            
            with st.expander(f"{status_icon} {item.get('title', 'Unknown Title')}", expanded=False):
                # Status indicator bar
                status_text = f"Status: {status.title()}"
                if status == 'processing':
                    st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;'>{status_icon} {status_text} - Processing in background...</div>", unsafe_allow_html=True)
                elif status == 'completed':
                    st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;'>{status_icon} {status_text} - Ready for Q&A</div>", unsafe_allow_html=True)
                elif status == 'failed':
                    st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;'>{status_icon} {status_text} - Processing failed</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: {status_color}; color: white; padding: 0.5rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;'>{status_icon} {status_text}</div>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Author:** {item.get('author', 'Unknown')}")
                    st.write(f"**Content Type:** {item.get('content_type', 'book').title()}")
                    st.write(f"**Module:** {item.get('module_type', 'library').title()}")
                    
                    # Enhanced processing info
                    if item.get('created_at'):
                        st.write(f"**Added:** {item['created_at'][:10]}")
                    
                    if item.get('semantic_tags'):
                        tags_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 2px 6px; border-radius: 12px; font-size: 0.8em;'>{tag}</span>" 
                                            for tag in item['semantic_tags'][:5]])
                        st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
                
                with col2:
                    if item.get('metadata'):
                        metadata = item['metadata']
                        st.write("**Metadata:**")
                        st.write(f"📄 Pages: {metadata.get('page_count', 'N/A')}")
                        st.write(f"🌐 Language: {metadata.get('language', 'N/A')}")
                        st.write(f"💾 Size: {metadata.get('file_size_mb', 'N/A')} MB")
                        
                        # Processing metrics
                        if metadata.get('processing_time'):
                            st.write(f"⏱️ Processed in: {metadata['processing_time']:.1f}s")
                        if metadata.get('chunks_created'):
                            st.write(f"📝 Text chunks: {metadata['chunks_created']}")
                
                with col3:
                    st.write("**Actions:**")
                    
                    # Context-aware actions based on status
                    if status == 'completed':
                        if st.button(f"💬 Ask Questions", key=f"chat_{item.get('content_id')}"):
                            st.session_state.current_book = item
                            st.session_state.page_selector = "💬 Q&A Chat"
                            st.rerun()
                        
                        if st.button(f"🔍 Explore", key=f"explore_{item.get('content_id')}"):
                            st.session_state.current_book = item
                            st.info("Book selected for exploration!")
                    
                    elif status == 'processing':
                        st.info("🔄 Processing...")
                        if st.button(f"🔄 Refresh", key=f"refresh_{item.get('content_id')}"):
                            st.rerun()
                    
                    elif status == 'failed':
                        st.error("❌ Failed")
                        if st.button(f"🔄 Retry", key=f"retry_{item.get('content_id')}"):
                            st.warning("Retry functionality coming soon!")
                    
                    else:
                        if st.button(f"🔍 View", key=f"view_{item.get('content_id')}"):
                            st.session_state.current_book = item
                            st.info("Book details loaded!")
                    
                    # Universal delete action
                    if st.button(f"🗑️ Delete", key=f"delete_{item.get('content_id')}"):
                        st.warning("Delete functionality coming soon!")
    else:
        st.info("📚 No books uploaded yet. Upload your first book to get started!")
    
    # Statistics
    if enhanced_content:
        st.subheader("📊 Library Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Books", len(enhanced_content))
        
        with col2:
            processed_count = sum(1 for item in enhanced_content if item.get('processing_status') == 'completed')
            st.metric("Processed", processed_count)
        
        with col3:
            unique_authors = len(set(item.get('author', 'Unknown') for item in enhanced_content))
            st.metric("Authors", unique_authors)
        
        with col4:
            total_tags = sum(len(item.get('semantic_tags', [])) for item in enhanced_content)
            st.metric("Semantic Tags", total_tags)

@create_module_aware_component("enhanced_search", "library", ["library"], UserRole.READER)
def render_enhanced_search():
    """Render the enhanced search interface with permission awareness."""
    st.title("🔍 Permission-Aware Enhanced Search")
    
    # Permission status
    current_user = get_current_user_permissions()
    render_permission_status(current_user, show_details=False)
    
    # Get the search component
    search_component = get_search_component(API_BASE_URL)
    
    # Render the search interface
    search_query, filters, search_button, clear_button = search_component.render_search_interface()
    
    # Add permission info to filters display
    if filters:
        st.info(f"🔐 Search results will be filtered according to your {current_user.role.value} permissions")
    
    # Handle clear button
    if clear_button:
        st.rerun()
    
    # Handle search
    if search_button and search_query:
        with st.spinner("🔍 Searching with enhanced algorithms and permission filtering..."):
            results = search_component.perform_enhanced_search(search_query, filters)
            
            # Apply permission filtering to search results
            if results:
                permission_manager = get_permission_manager()
                filtered_results, permission_stats = render_permission_aware_search_results(results, current_user)
                
                # Update results with permission-filtered version
                results = filtered_results
        
        # Render results
        if results:
            search_component.render_search_results(results, search_query)
        else:
            st.warning("🔒 No accessible results found. This may be due to permission restrictions.")
    
    # Show search suggestions when no active search
    elif not search_query:
        suggestion = search_component.render_search_suggestions()
        if suggestion:
            st.session_state.suggested_query = suggestion
            st.rerun()
    
    # Show search analytics
    with st.expander("📊 Search Analytics", expanded=False):
        search_component.render_search_analytics()

def render_empty_library_state():
    """Render an attractive empty state when no books are available."""
    st.markdown("---")
    
    # Center the empty state content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Empty state illustration
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
            <h2 style="color: #666; margin-bottom: 0.5rem;">Your Library is Empty</h2>
            <p style="color: #888; font-size: 1.1rem; margin-bottom: 2rem;">
                Upload your first book to start asking questions and exploring content with AI.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📤 Upload Your First Book", 
                        type="primary", 
                        use_container_width=True,
                        help="Go to Book Management to upload content"):
                st.session_state.page_selector = "📖 Book Management"
                st.rerun()
        
        with col_b:
            if st.button("🔍 Learn More", 
                        use_container_width=True,
                        help="Learn about Alexandria's features"):
                st.session_state.show_getting_started = True
                st.rerun()
    
    # Getting started guide
    if st.session_state.get('show_getting_started', False):
        st.markdown("---")
        st.subheader("🚀 Getting Started with Alexandria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📖 Supported File Types:**
            - PDF documents
            - EPUB books
            - Word documents (DOC, DOCX)
            - Plain text files (TXT)
            - HTML documents
            """)
        
        with col2:
            st.markdown("""
            **💡 What You Can Do:**
            - Ask questions about your books
            - Get AI-powered answers with source citations
            - Explore relationships between content
            - Export conversations for later reference
            """)
        
        st.info("💡 **Tip:** Start with a PDF or EPUB file for the best experience. Processing typically takes 30-60 seconds depending on file size.")
        
        if st.button("❌ Close Guide"):
            st.session_state.show_getting_started = False
            st.rerun()
    
    # Feature preview
    st.markdown("---")
    st.subheader("🎯 Preview: What Your Chat Will Look Like")
    
    # Mock conversation preview
    with st.chat_message("user"):
        st.write("What are the main themes in this book?")
    
    with st.chat_message("assistant"):
        st.write("Once you upload a book, I'll be able to analyze its content and provide detailed answers like this, complete with:")
        st.write("• **Source citations** with page numbers")
        st.write("• **Confidence scores** for answer reliability")
        st.write("• **Related content** suggestions")
        st.write("• **Conversation export** options")
        
        # Mock confidence indicator
        st.markdown("**Confidence:** 🟢 High (85%)")
        
        # Mock sources
        with st.expander("📚 Sources (Preview)", expanded=False):
            st.write("1. **Your Book Title** - Page 42")
            st.write("2. **Your Book Title** - Page 156")
            st.write("3. **Your Book Title** - Page 203")


def render_error_retry_options(question: str, query_mode: str, selected_content_id: str = None, 
                              max_results: int = 10, include_relationships: bool = True, 
                              module_filter: str = None, current_user=None):
    """Render retry options after an error occurs."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retry Question", use_container_width=True, key=f"retry_{int(datetime.now().timestamp())}"):
            # Set a flag to retry the question
            st.session_state.retry_question = question
            st.session_state.retry_mode = query_mode
            st.session_state.retry_content_id = selected_content_id
            st.session_state.retry_max_results = max_results
            st.session_state.retry_include_relationships = include_relationships
            st.session_state.retry_module_filter = module_filter
            st.rerun()
    
    with col2:
        if st.button("🔧 Check Backend Status", use_container_width=True, key=f"status_{int(datetime.now().timestamp())}"):
            try:
                health_response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=10)
                if health_response.status_code == 200:
                    st.success("✅ Backend service is running")
                else:
                    st.error(f"❌ Backend returned status: {health_response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Cannot reach backend: {str(e)[:50]}...")
    
    with col3:
        if st.button("📚 Go to Library", use_container_width=True, key=f"library_{int(datetime.now().timestamp())}"):
            st.session_state.page_selector = "📖 Book Management"
            st.rerun()
    
    # Auto-retry functionality (if retry flags are set)
    if st.session_state.get('retry_question'):
        st.info("🔄 Retrying your question...")
        # Clear retry flags
        retry_question = st.session_state.retry_question
        st.session_state.retry_question = None
        
        # Simulate re-asking the question by updating the chat input
        st.session_state.retry_in_progress = True
        st.rerun()


def render_qa_chat():
    """Render the enhanced Q&A chat interface with multi-module support and permission awareness."""
    st.title("💬 Enhanced Multi-Module Q&A Chat")
    
    # Permission Status Display
    current_user = get_current_user_permissions()
    
    # Permission status in header
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            render_permission_status(current_user, show_details=False)
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear permission cache", key="clear_permission_cache"):
                clear_permission_cache()
                st.rerun()
    
    # Enhanced Content and Module Selection with Permission Filtering
    with st.spinner("📚 Loading your library..."):
        enhanced_content = get_enhanced_content_list()
    
    if not enhanced_content:
        render_empty_library_state()
        return
    
    # Apply permission filtering to content list
    permission_manager = get_permission_manager()
    accessible_content = permission_manager.filter_accessible_content(current_user, enhanced_content)
    
    if len(accessible_content) < len(enhanced_content):
        filtered_count = len(enhanced_content) - len(accessible_content)
        st.info(f"🔐 {filtered_count} content items filtered by permissions. {len(accessible_content)} items accessible.")
    
    if not accessible_content:
        st.warning("🔒 No accessible content found with your current permissions. Contact an administrator for access.")
        return
    
    # Multi-module query options
    st.subheader("🔍 Permission-Aware Query Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Query mode selection
        query_mode = st.selectbox(
            "Query Mode:",
            options=[
                "Single Content Query",
                "Multi-Module Search", 
                "Cross-Module Discovery",
                "All Content Search"
            ],
            help="Choose how to scope your question"
        )
    
    with col2:
        # Module filter for multi-module queries
        if query_mode != "Single Content Query":
            module_filter = st.selectbox(
                "Module Filter:",
                options=["All Modules", "Library", "LMS", "Marketplace"],
                help="Filter content by module type"
            )
        else:
            module_filter = None
    
    # Content selection for single content queries (using permission-filtered content)
    if query_mode == "Single Content Query":
        content_options = {f"{item.get('title', 'Unknown')} - {item.get('author', 'Unknown')} [{item.get('module_type', 'library').title()}]": item.get('content_id') 
                          for item in accessible_content}
        
        selected_content_title = st.selectbox(
            "Select Content for Q&A:",
            options=list(content_options.keys()),
            help="Choose which content to ask questions about (filtered by your permissions)"
        )
        
        selected_content_id = content_options[selected_content_title]
    else:
        selected_content_id = None
        selected_content_title = f"{query_mode} - {module_filter if module_filter != 'All Modules' else 'All Content'}"
    
    # Advanced query options
    with st.expander("🛠️ Advanced Query Options", expanded=False):
        include_relationships = st.checkbox(
            "Include Content Relationships", 
            value=True,
            help="Include related content in search results"
        )
        
        max_results = st.slider(
            "Maximum Results",
            min_value=3,
            max_value=20,
            value=10,
            help="Maximum number of content pieces to consider"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Minimum confidence for including results"
        )
    
    # Chat Interface Section
    st.subheader("💭 Intelligent Multi-Module Q&A")
    st.caption(f"Current scope: {selected_content_title}")
    
    # Display enhanced chat history
    enhanced_messages = get_enhanced_chat_history()
    
    for message in enhanced_messages:
        render_enhanced_message(message)
    
    # Chat input
    if question := st.chat_input("Ask a question using multi-module search..."):
        # Add user message to enhanced history
        user_message = EnhancedChatMessage(
            message_type='user',
            content=question,
            timestamp=datetime.now()
        )
        add_enhanced_message(user_message)
        
        # Display user message immediately
        render_enhanced_message(user_message)
        
        # Get AI response from enhanced RAG service
        with st.chat_message("assistant"):
            spinner_text = f"🔍 Searching across {query_mode.lower()}..."
            status_placeholder = st.empty()
            
            try:
                with st.spinner(spinner_text):
                    status_placeholder.info("🔄 Connecting to AI service...")
                    
                    # Determine API endpoint and payload based on query mode
                    if query_mode == "Single Content Query":
                        status_placeholder.info("📖 Analyzing your selected content...")
                        # Use traditional chat API for single content
                        response = requests.post(
                            f"{API_BASE_URL}/api/chat/query",
                            json={
                                "question": question,
                                "book_id": selected_content_id,
                                "conversation_id": st.session_state.conversation_id,
                                "context_limit": max_results
                            },
                            timeout=30
                        )
                    else:
                        status_placeholder.info("🔍 Searching across multiple modules...")
                        # Use enhanced multi-module search API with permission filtering
                        search_payload = {
                            "query": question,
                            "n_results": max_results,
                            "include_relationships": include_relationships
                        }
                        
                        # Add module filter if specified
                        if module_filter and module_filter != "All Modules":
                            search_payload["module_filter"] = module_filter.lower()
                        
                        # Apply permission filters to search payload
                        search_payload = apply_permission_filters_to_search(current_user, search_payload)
                        
                        response = requests.post(
                            f"{API_BASE_URL}/api/enhanced/search",
                            json=search_payload,
                            timeout=30
                        )
                
                status_placeholder.info("🧠 Processing AI response...")
                status_placeholder.empty()  # Clear status after success
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Process response based on API type
                    if query_mode == "Single Content Query":
                        # Traditional chat API response
                        st.session_state.conversation_id = result.get('conversation_id')
                        
                        assistant_message = EnhancedChatMessage(
                            message_type='assistant',
                            content=result.get('answer', 'No response received.'),
                            timestamp=datetime.now(),
                            sources=result.get('sources', []),
                            confidence=calculate_confidence_score(result.get('sources', [])),
                            token_usage=result.get('token_usage', {}),
                            message_id=result.get('message_id')
                        )
                    else:
                        # Enhanced search API response with permission filtering
                        search_results = result.get('results', [])
                        
                        # Apply permission-aware filtering to results
                        permission_filtered_results, permission_stats = render_permission_aware_search_results(
                            search_results, current_user
                        )
                        
                        # Filter results by confidence threshold
                        confidence_filtered_results = [
                            r for r in permission_filtered_results 
                            if r.get('similarity_score', 0.0) >= confidence_threshold
                        ]
                        
                        # Generate conversational answer from search results
                        answer_content = generate_answer_from_search_results(
                            confidence_filtered_results, question, query_mode
                        )
                        
                        # Add permission filtering info to answer if results were filtered
                        if permission_stats['filtered_results'] > 0:
                            permission_note = f"\n\n*Note: Search included permission filtering. {permission_stats['accessible_results']} of {permission_stats['total_results']} results were accessible to your {current_user.role.value} role.*"
                            answer_content += permission_note
                        
                        # Convert search results to source format
                        sources = convert_search_results_to_sources(confidence_filtered_results)
                        
                        assistant_message = EnhancedChatMessage(
                            message_type='assistant',
                            content=answer_content,
                            timestamp=datetime.now(),
                            sources=sources,
                            confidence=calculate_confidence_score(sources),
                            token_usage={
                                "search_time_ms": result.get('search_time_ms', 0),
                                "total_results": result.get('total_results', 0),
                                "permission_filtered": permission_stats['filtered_results'],
                                "confidence_filtered": len(permission_filtered_results) - len(confidence_filtered_results),
                                "final_results": len(confidence_filtered_results),
                                "user_role": current_user.role.value
                            },
                            message_id=f"enhanced_search_{int(datetime.now().timestamp())}"
                        )
                    
                    # Add to enhanced history
                    add_enhanced_message(assistant_message)
                    
                    # Render the response (will show immediately due to rerun)
                    st.rerun()
                        
                else:
                    # Enhanced error handling with retry options
                    status_placeholder.empty()
                    error_code = response.status_code
                    
                    if error_code == 404:
                        error_content = "❌ The requested content was not found. This might happen if the book was recently deleted or is still processing."
                    elif error_code == 500:
                        error_content = "🔧 The AI service encountered an internal error. This is usually temporary."
                    elif error_code == 429:
                        error_content = "⏳ Too many requests. Please wait a moment before trying again."
                    else:
                        error_content = f"❌ Unexpected error (Code: {error_code}). Please try again."
                    
                    error_message = EnhancedChatMessage(
                        message_type='assistant',
                        content=error_content,
                        timestamp=datetime.now(),
                        confidence=0.0
                    )
                    add_enhanced_message(error_message)
                    
                    # Show retry options
                    render_error_retry_options(question, query_mode, selected_content_id, max_results, include_relationships, module_filter, current_user)
                        
            except requests.exceptions.Timeout:
                # Timeout error handling
                status_placeholder.empty()
                error_message = EnhancedChatMessage(
                    message_type='assistant',
                    content="⏰ Request timed out. The AI service might be busy. Please try again in a moment.",
                    timestamp=datetime.now(),
                    confidence=0.0
                )
                add_enhanced_message(error_message)
                render_error_retry_options(question, query_mode, selected_content_id, max_results, include_relationships, module_filter, current_user)
                
            except requests.exceptions.ConnectionError:
                # Connection error handling
                status_placeholder.empty()
                error_message = EnhancedChatMessage(
                    message_type='assistant',
                    content="🔌 Unable to connect to the AI service. Please check if the backend server is running and try again.",
                    timestamp=datetime.now(),
                    confidence=0.0
                )
                add_enhanced_message(error_message)
                render_error_retry_options(question, query_mode, selected_content_id, max_results, include_relationships, module_filter, current_user)
                
            except requests.exceptions.RequestException as e:
                # General request error handling
                status_placeholder.empty()
                error_message = EnhancedChatMessage(
                    message_type='assistant',
                    content=f"🔧 Network error occurred: {str(e)[:100]}... Please check your connection and try again.",
                    timestamp=datetime.now(),
                    confidence=0.0
                )
                add_enhanced_message(error_message)
                render_error_retry_options(question, query_mode, selected_content_id, max_results, include_relationships, module_filter, current_user)
    
    # Enhanced Chat Controls
    if enhanced_messages:
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_enhanced_chat_history()
                st.rerun()
        
        with col2:
            if st.button("📊 View Stats", use_container_width=True):
                st.session_state.show_chat_stats = not st.session_state.get('show_chat_stats', False)
                st.rerun()
        
        with col3:
            if st.button("📤 Export Options", use_container_width=True):
                st.session_state.show_export_options = not st.session_state.get('show_export_options', False)
                st.rerun()
        
        with col4:
            message_count = len(enhanced_messages)
            st.metric("Messages", message_count)
        
        # Show statistics if requested
        if st.session_state.get('show_chat_stats', False):
            with st.expander("📊 Conversation Statistics", expanded=True):
                render_conversation_statistics(enhanced_messages)
        
        # Show export options if requested
        if st.session_state.get('show_export_options', False):
            with st.expander("📤 Export Conversation", expanded=True):
                render_export_options(enhanced_messages, selected_content_title)


def calculate_confidence_score(sources: List[Dict]) -> float:
    """Calculate confidence score based on source quality and relevance."""
    if not sources:
        return 0.0
    
    # Simple confidence calculation based on source count and relevance
    total_relevance = sum(source.get('relevance_score', 0.0) for source in sources)
    source_count_factor = min(len(sources) / 3.0, 1.0)  # Cap at 3 sources for max benefit
    avg_relevance = total_relevance / len(sources) if sources else 0.0
    
    # Combine factors
    confidence = (avg_relevance * 0.7) + (source_count_factor * 0.3)
    return min(confidence, 1.0)


def generate_answer_from_search_results(search_results: List[Dict], question: str, query_mode: str) -> str:
    """Generate a conversational answer from multi-module search results."""
    if not search_results:
        return "I couldn't find any relevant information to answer your question. Please try rephrasing or check if the content has been processed."
    
    # Group results by module and content type
    results_by_module = {}
    for result in search_results:
        module = result.get('module_type', 'unknown')
        if module not in results_by_module:
            results_by_module[module] = []
        results_by_module[module].append(result)
    
    # Build answer based on query mode
    answer_parts = []
    
    if query_mode == "Multi-Module Search":
        answer_parts.append(f"Based on my search across multiple modules, here's what I found:")
        
        for module, module_results in results_by_module.items():
            module_name = module.title() if module != 'unknown' else 'Content'
            answer_parts.append(f"\n**From {module_name}:**")
            
            for result in module_results[:3]:  # Limit to top 3 per module
                title = result.get('title', 'Unknown Content')
                score = result.get('similarity_score', 0.0)
                answer_parts.append(f"• *{title}* (relevance: {score:.1%})")
    
    elif query_mode == "Cross-Module Discovery":
        answer_parts.append(f"Here are cross-module connections I discovered for your question:")
        
        # Show relationships across modules
        for result in search_results[:5]:
            title = result.get('title', 'Unknown Content')
            module = result.get('module_type', 'unknown').title()
            score = result.get('similarity_score', 0.0)
            answer_parts.append(f"• **{title}** [{module}] - {score:.1%} relevant")
    
    else:  # All Content Search
        answer_parts.append(f"Searching across all available content, I found {len(search_results)} relevant results:")
        
        for result in search_results[:5]:
            title = result.get('title', 'Unknown Content')
            module = result.get('module_type', 'unknown').title()
            content_type = result.get('content_type', 'unknown').title()
            score = result.get('similarity_score', 0.0)
            answer_parts.append(f"• **{title}** [{module} - {content_type}] ({score:.1%} relevant)")
    
    # Add guidance for follow-up
    if len(search_results) > 5:
        answer_parts.append(f"\n*Found {len(search_results)} total results. Use the source citations below to explore more details.*")
    
    answer_parts.append(f"\nTo get more specific information, you can ask follow-up questions or select a specific piece of content for detailed Q&A.")
    
    return "\n".join(answer_parts)


def convert_search_results_to_sources(search_results: List[Dict]) -> List[Dict]:
    """Convert enhanced search results to source format for citations."""
    sources = []
    
    for result in search_results:
        source = {
            'title': result.get('title', 'Unknown Content'),
            'author': result.get('author', 'Unknown Author'),
            'content_type': result.get('content_type', 'unknown'),
            'module_type': result.get('module_type', 'unknown'),
            'page_number': result.get('source_location', {}).get('page', 'N/A'),
            'chapter': result.get('source_location', {}).get('chapter', ''),
            'section': result.get('source_location', {}).get('section', ''),
            'content': f"Chunk type: {result.get('chunk_type', 'unknown')} | Tags: {', '.join(result.get('semantic_tags', [])[:3])}",
            'relevance_score': result.get('similarity_score', 0.0),
            'relationship_score': result.get('relationship_score', 0.0)
        }
        sources.append(source)
    
    return sources

def render_content_relationships():
    """Render content relationships visualization using the advanced component."""
    # Get the relationships component
    relationships_component = get_relationship_component(API_BASE_URL)
    
    # Render the relationship overview
    relationships_component.render_relationship_overview()

def render_analytics_dashboard():
    """Render analytics and insights dashboard."""
    st.title("📊 Analytics Dashboard")
    
    enhanced_content = get_enhanced_content_list()
    
    if not enhanced_content:
        st.warning("📚 No data available for analytics.")
        return
    
    # Overview metrics
    st.subheader("📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Content", len(enhanced_content))
    
    with col2:
        processed_count = sum(1 for item in enhanced_content if item.get('processing_status') == 'completed')
        st.metric("Processed Items", processed_count)
    
    with col3:
        total_tags = sum(len(item.get('semantic_tags', [])) for item in enhanced_content)
        st.metric("Semantic Tags", total_tags)
    
    with col4:
        unique_authors = len(set(item.get('author', 'Unknown') for item in enhanced_content))
        st.metric("Unique Authors", unique_authors)
    
    # Content type distribution
    st.subheader("📊 Content Distribution")
    
    content_types = [item.get('content_type', 'unknown') for item in enhanced_content]
    type_counts = pd.Series(content_types).value_counts()
    
    if not type_counts.empty:
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Content Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Semantic tags cloud
    st.subheader("🏷️ Popular Tags")
    
    all_tags = []
    for item in enhanced_content:
        all_tags.extend(item.get('semantic_tags', []))
    
    if all_tags:
        tag_counts = pd.Series(all_tags).value_counts().head(20)
        
        fig = px.bar(
            x=tag_counts.index,
            y=tag_counts.values,
            title="Top 20 Semantic Tags"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

def render_settings():
    """Render settings and configuration page with permission management."""
    st.title("⚙️ Settings")
    
    # Permission Management Section
    st.subheader("🔐 Permission Management")
    current_user = get_current_user_permissions()
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_permission_status(current_user, show_details=True)
    
    with col2:
        st.markdown("**Permission Actions:**")
        
        if st.button("🗑️ Clear Permission Cache", use_container_width=True):
            clear_permission_cache()
            st.success("Permission cache cleared!")
        
        if st.button("📊 View Cache Stats", use_container_width=True):
            permission_manager = get_permission_manager()
            cache_stats = permission_manager.get_permission_summary(current_user)
            st.json(cache_stats)
        
        # Future: Phase 2 will add user role management here
        st.info("🔮 **Coming in Phase 2:** User role management, organization settings, and multi-user administration")
    
    st.markdown("---")
    
    # API Configuration
    st.subheader("🔌 API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
        
    with col2:
        if st.button("🔄 Test Connection"):
            try:
                response = requests.get(f"{API_BASE_URL}/api/v1/health")
                if response.status_code == 200:
                    st.success("✅ API connection successful!")
                else:
                    st.error("❌ API connection failed!")
            except:
                st.error("❌ Cannot reach API server!")
    
    # Enhanced Features
    st.subheader("🚀 Enhanced Features")
    
    features = {
        "Enhanced Search": True,
        "Content Relationships": True,
        "Semantic Tagging": True,
        "Multi-Module Support": True,
        "Permission-Aware Search": True
    }
    
    for feature, enabled in features.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{feature}**")
        with col2:
            if enabled:
                st.success("✅ Enabled")
            else:
                st.error("❌ Disabled")
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    system_info = {
        "Frontend": "Streamlit",
        "Backend": "FastAPI",
        "Database": "Enhanced Chroma/Supabase",
        "AI Provider": "OpenAI",
        "Version": "Phase 1.41"
    }
    
    for key, value in system_info.items():
        st.write(f"**{key}:** {value}")


def render_lms_module(page: str, features: Dict[str, bool]):
    """Render LMS module pages (Phase 2 placeholder)."""
    st.title("🎓 Learning Suite Module")
    st.info("📋 **Coming in Phase 2.0**: Learning Management System features")
    
    if page == "course_builder":
        st.subheader("🏗️ Course Builder")
        st.write("Create structured courses from your book content using AI-powered learning path generation.")
    elif page == "learning_paths":
        st.subheader("🛤️ Learning Paths")
        st.write("Design personalized learning progressions with adaptive content delivery.")
    elif page == "assessments":
        st.subheader("📝 Assessments")
        st.write("Create quizzes, tests, and evaluations with automated grading.")
    elif page == "student_analytics":
        st.subheader("📈 Student Analytics")
        st.write("Track student progress and learning outcomes with detailed analytics.")
    
    st.markdown("---")
    st.markdown("**Current Status**: Available in Phase 2.0 with Next.js frontend migration")


def render_marketplace_module(page: str, features: Dict[str, bool]):
    """Render Marketplace module pages (Phase 3 placeholder)."""
    st.title("🏪 Marketplace Module")
    st.info("📋 **Coming in Phase 3.0**: Content monetization and community features")
    
    if page == "content_store":
        st.subheader("🛍️ Content Store")
        st.write("Browse and purchase premium books, courses, and educational content.")
    elif page == "creator_dashboard":
        st.subheader("💼 Creator Dashboard")
        st.write("Manage your content sales, analytics, and revenue streams.")
    elif page == "community":
        st.subheader("👥 Community")
        st.write("Connect with other readers, authors, and educators in the Alexandria community.")
    elif page == "monetization":
        st.subheader("💰 Monetization")
        st.write("Set up pricing, payment processing, and revenue sharing for your content.")
    
    st.markdown("---")
    st.markdown("**Current Status**: Available in Phase 3.0 with full marketplace functionality")


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
    
    # Module and page routing
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
        render_lms_module(current_page, features)
    elif current_module == "marketplace":
        render_marketplace_module(current_page, features)
    else:
        st.error(f"Unknown module: {current_module}")
        st.info("Defaulting to Library module...")
        render_book_management()

if __name__ == "__main__":
    main()