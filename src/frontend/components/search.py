"""
Advanced search components for the DBC Streamlit frontend.

This module provides enhanced search functionality including semantic search,
filters, and advanced query processing using the enhanced RAG backend.
"""

import streamlit as st
import requests
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta


class AdvancedSearchComponent:
    """Advanced search component with enhanced filtering and visualization."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def render_search_filters(self) -> Dict[str, Any]:
        """Render advanced search filters and return filter parameters."""
        st.subheader("🔧 Search Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Content type filter
            content_types = ["All", "Book", "Article", "Document", "Course", "Lesson", "Assessment"]
            content_type = st.selectbox(
                "Content Type",
                content_types,
                help="Filter by type of content"
            )
        
        with col2:
            # Module filter
            modules = ["All", "Library", "LMS", "Marketplace"]
            module = st.selectbox(
                "Module",
                modules,
                help="Filter by platform module"
            )
        
        with col3:
            # Language filter
            languages = ["All", "English", "Spanish", "French", "German"]
            language = st.selectbox(
                "Language",
                languages,
                help="Filter by content language"
            )
        
        # Advanced filters in expander
        with st.expander("🔍 Advanced Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Semantic tags filter
                semantic_tags = st.text_input(
                    "Semantic Tags",
                    placeholder="psychology, learning, memory",
                    help="Search for specific topics (comma-separated)"
                )
                
                # Author filter
                author = st.text_input(
                    "Author",
                    placeholder="Enter author name",
                    help="Filter by author name"
                )
            
            with col2:
                # Reading level filter
                reading_levels = ["All", "Beginner", "Intermediate", "Advanced"]
                reading_level = st.selectbox(
                    "Reading Level",
                    reading_levels,
                    help="Filter by difficulty level"
                )
                
                # Date range filter
                date_range = st.selectbox(
                    "Added Date",
                    ["All Time", "Last Week", "Last Month", "Last Year"],
                    help="Filter by when content was added"
                )
        
        # Search parameters
        col1, col2 = st.columns(2)
        
        with col1:
            max_results = st.slider(
                "Max Results",
                min_value=5,
                max_value=100,
                value=20,
                help="Maximum number of search results"
            )
        
        with col2:
            search_mode = st.selectbox(
                "Search Mode",
                ["Hybrid", "Semantic Only", "Keyword Only", "Graph Traversal"],
                help="Choose search strategy"
            )
        
        return {
            "content_type": content_type if content_type != "All" else None,
            "module": module if module != "All" else None,
            "language": language if language != "All" else None,
            "semantic_tags": [tag.strip() for tag in semantic_tags.split(",") if tag.strip()] if semantic_tags else None,
            "author": author if author else None,
            "reading_level": reading_level if reading_level != "All" else None,
            "date_range": date_range,
            "max_results": max_results,
            "search_mode": search_mode
        }
    
    def render_search_interface(self) -> Tuple[str, Dict[str, Any]]:
        """Render the main search interface."""
        st.title("🔍 Advanced Content Search")
        
        # Search query input
        search_query = st.text_area(
            "Search Query",
            placeholder="Ask a question, search for topics, or describe what you're looking for...",
            help="Use natural language to search across all your content",
            height=100
        )
        
        # Search filters
        filters = self.render_search_filters()
        
        # Search buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear")
        
        with col3:
            if search_query:
                st.write(f"Ready to search: *{search_query[:50]}{'...' if len(search_query) > 50 else ''}*")
        
        return search_query, filters, search_button, clear_button
    
    def perform_enhanced_search(self, query: str, filters: Dict[str, Any]) -> List[Dict]:
        """Perform enhanced search using the backend API."""
        search_data = {
            "query": query,
            "n_results": filters.get("max_results", 20),
            "include_relationships": True
        }
        
        # Add filters
        if filters.get("module"):
            search_data["module_filter"] = filters["module"].lower()
        
        if filters.get("content_type"):
            search_data["content_type_filter"] = filters["content_type"].lower()
        
        # Additional filter parameters (would be handled by enhanced backend)
        if filters.get("semantic_tags"):
            search_data["semantic_tags"] = filters["semantic_tags"]
        
        if filters.get("author"):
            search_data["author_filter"] = filters["author"]
        
        if filters.get("language"):
            search_data["language_filter"] = filters["language"].lower()
        
        if filters.get("reading_level"):
            search_data["reading_level_filter"] = filters["reading_level"].lower()
        
        try:
            response = requests.post(f"{self.api_base_url}/api/enhanced/search", json=search_data)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Search failed: {e}")
            return []
    
    def render_search_results(self, results: List[Dict], query: str):
        """Render enhanced search results with rich metadata."""
        if not results:
            st.info("🔍 No results found. Try adjusting your search terms or filters.")
            return
        
        st.subheader(f"🎯 Search Results ({len(results)} found)")
        
        # Results summary
        self.render_results_summary(results, query)
        
        # Sort options
        col1, col2 = st.columns([1, 3])
        
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Relevance", "Date", "Title", "Author", "Content Type"]
            )
        
        # Sort results
        sorted_results = self.sort_results(results, sort_by)
        
        # Render individual results
        for i, result in enumerate(sorted_results):
            self.render_result_card(result, i)
    
    def render_results_summary(self, results: List[Dict], query: str):
        """Render search results summary with analytics."""
        with st.expander("📊 Search Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_confidence = sum(r.get('confidence_score', 0) for r in results) / len(results)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col2:
                content_types = [r.get('content_type', 'unknown') for r in results]
                unique_types = len(set(content_types))
                st.metric("Content Types", unique_types)
            
            with col3:
                authors = [r.get('author', 'Unknown') for r in results]
                unique_authors = len(set(authors))
                st.metric("Authors", unique_authors)
            
            with col4:
                total_tags = sum(len(r.get('semantic_tags', [])) for r in results)
                st.metric("Total Tags", total_tags)
            
            # Content type distribution
            if content_types:
                type_counts = pd.Series(content_types).value_counts()
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Results by Content Type"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def sort_results(self, results: List[Dict], sort_by: str) -> List[Dict]:
        """Sort search results by specified criteria."""
        if sort_by == "Relevance":
            return sorted(results, key=lambda x: x.get('confidence_score', 0), reverse=True)
        elif sort_by == "Date":
            return sorted(results, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_by == "Title":
            return sorted(results, key=lambda x: x.get('title', '').lower())
        elif sort_by == "Author":
            return sorted(results, key=lambda x: x.get('author', '').lower())
        elif sort_by == "Content Type":
            return sorted(results, key=lambda x: x.get('content_type', '').lower())
        else:
            return results
    
    def render_result_card(self, result: Dict, index: int):
        """Render an individual search result card."""
        # Generate unique, safe keys for buttons
        content_id = result.get('content_id')
        if content_id:
            # Ensure content_id is string and sanitize for button key
            content_id_str = str(content_id).replace(' ', '_').replace('-', '_')
            ask_key = f"ask_{content_id_str}_{index}"
            relations_key = f"relations_{content_id_str}_{index}"
        else:
            # Fallback to index-based keys when content_id is missing
            ask_key = f"ask_result_{index}"
            relations_key = f"relations_result_{index}"
        
        with st.expander(
            f"📄 {result.get('title', 'Unknown Title')}", 
            expanded=index < 3  # Expand first 3 results
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main content info
                st.markdown(f"**📖 Title:** {result.get('title', 'Unknown')}")
                st.markdown(f"**✍️ Author:** {result.get('author', 'Unknown')}")
                st.markdown(f"**📂 Type:** {result.get('content_type', 'unknown').title()}")
                st.markdown(f"**🏗️ Module:** {result.get('module_type', 'unknown').title()}")
                
                # Excerpt
                if result.get('excerpt'):
                    st.markdown("**📝 Relevant Excerpt:**")
                    st.markdown(f"*{result['excerpt']}*")
                
                # Semantic tags
                if result.get('semantic_tags'):
                    st.markdown("**🏷️ Tags:**")
                    tags_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 2px 6px; border-radius: 12px; font-size: 0.8em;'>{tag}</span>" 
                                        for tag in result['semantic_tags']])
                    st.markdown(tags_html, unsafe_allow_html=True)
            
            with col2:
                # Relevance and metadata
                st.markdown("**🎯 Relevance:**")
                confidence = result.get('confidence_score', 0.5)
                st.progress(confidence)
                st.write(f"{confidence:.1%} match")
                
                # Source location
                if result.get('source_location'):
                    loc = result['source_location']
                    st.markdown("**📍 Location:**")
                    if 'page' in loc:
                        st.write(f"📄 Page {loc['page']}")
                    if 'section' in loc:
                        st.write(f"📑 Section: {loc['section']}")
                    if 'chapter' in loc:
                        st.write(f"📚 Chapter: {loc['chapter']}")
                
                # Actions
                st.markdown("**⚡ Actions:**")
                if st.button(f"💬 Ask Questions", key=ask_key):
                    st.session_state.selected_content_for_chat = result
                    st.success("Content selected for Q&A!")
                
                if st.button(f"🔗 View Relations", key=relations_key):
                    st.session_state.selected_content_for_relations = result
                    st.success("View relationships!")
    
    def render_search_suggestions(self, query: str = "") -> List[str]:
        """Render search suggestions based on content and query history."""
        st.subheader("💡 Search Suggestions")
        
        # Sample suggestions - in real implementation, these would come from backend
        suggestions = [
            "What are the main themes in psychology books?",
            "Compare different learning methodologies",
            "Find content about memory and cognition",
            "Show me beginner-friendly programming books",
            "What are the latest trends in artificial intelligence?",
            "Find books by specific authors in my library"
        ]
        
        # Filter suggestions based on current query
        if query:
            filtered_suggestions = [s for s in suggestions if any(word.lower() in s.lower() for word in query.split())]
        else:
            filtered_suggestions = suggestions[:3]
        
        for suggestion in filtered_suggestions:
            if st.button(f"💭 {suggestion}", key=f"suggestion_{suggestion[:20]}"):
                return suggestion
        
        return None
    
    def render_search_analytics(self):
        """Render search analytics and insights."""
        st.subheader("📈 Search Analytics")
        
        # Mock analytics data - in real implementation, this would come from backend
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Searches", "1,234", "+15%")
        
        with col2:
            st.metric("Avg Results", "8.5", "+2.1")
        
        with col3:
            st.metric("Success Rate", "87%", "+5%")
        
        # Search patterns
        search_data = {
            "Date": pd.date_range(start="2025-01-01", periods=30, freq="D"),
            "Searches": [10, 15, 12, 20, 25, 18, 22, 16, 19, 24, 27, 21, 23, 28, 32, 26, 29, 35, 31, 33, 38, 34, 36, 42, 39, 41, 45, 43, 47, 50]
        }
        
        df = pd.DataFrame(search_data)
        
        fig = px.line(df, x="Date", y="Searches", title="Search Activity Over Time")
        st.plotly_chart(fig, use_container_width=True)


def get_search_component(api_base_url: str = "http://localhost:8000") -> AdvancedSearchComponent:
    """Get an instance of the advanced search component."""
    return AdvancedSearchComponent(api_base_url)