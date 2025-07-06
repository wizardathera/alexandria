"""
Enhanced Q&A Chat Interface for DBC Frontend.

This module provides an enhanced Q&A interface with:
- Rich formatting for chat messages
- Source citations with page numbers and context
- Confidence score visualization
- Conversation export functionality
- Improved conversation flow and readability
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import base64

from .themes import get_current_theme, get_theme_color


class EnhancedChatMessage:
    """Enhanced chat message with rich formatting and metadata."""
    
    def __init__(self, message_type: str, content: str, timestamp: datetime = None, 
                 sources: List[Dict] = None, confidence: float = None, 
                 token_usage: Dict = None, message_id: str = None):
        self.message_type = message_type  # 'user' or 'assistant'
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.sources = sources or []
        self.confidence = confidence
        self.token_usage = token_usage or {}
        self.message_id = message_id or f"{message_type}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage."""
        return {
            'type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'sources': self.sources,
            'confidence': self.confidence,
            'token_usage': self.token_usage,
            'message_id': self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedChatMessage':
        """Create message from dictionary."""
        return cls(
            message_type=data.get('type', 'user'),
            content=data.get('content', ''),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            sources=data.get('sources', []),
            confidence=data.get('confidence'),
            token_usage=data.get('token_usage', {}),
            message_id=data.get('message_id')
        )


def render_confidence_score(confidence: float, show_label: bool = True) -> None:
    """Render confidence score visualization."""
    if confidence is None:
        return
    
    # Determine color based on confidence level
    if confidence >= 0.8:
        color = get_theme_color('success')
        label = "High Confidence"
        icon = "🟢"
    elif confidence >= 0.6:
        color = get_theme_color('warning')
        label = "Medium Confidence"
        icon = "🟡"
    else:
        color = get_theme_color('error')
        label = "Low Confidence"
        icon = "🔴"
    
    # Create progress bar for confidence
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(confidence, text=f"{icon} {label} ({confidence:.0%})")
    with col2:
        if show_label:
            st.caption(f"{confidence:.1%}")


def render_source_citations(sources: List[Dict[str, Any]]) -> None:
    """Render source citations with improved formatting."""
    if not sources:
        return
    
    st.markdown("**📚 Sources:**")
    
    with st.expander(f"View {len(sources)} source(s)", expanded=False):
        for i, source in enumerate(sources, 1):
            # Extract source information
            title = source.get('title', 'Unknown Document')
            page = source.get('page_number', 'N/A')
            chapter = source.get('chapter', '')
            section = source.get('section', '')
            content = source.get('content', '')
            relevance = source.get('relevance_score', 0.0)
            
            # Create citation container
            citation_container = st.container()
            with citation_container:
                # Citation header
                header_cols = st.columns([3, 1])
                with header_cols[0]:
                    st.markdown(f"**{i}. {title}**")
                    location_parts = [f"Page {page}" if page != 'N/A' else None,
                                    f"Chapter: {chapter}" if chapter else None,
                                    f"Section: {section}" if section else None]
                    location = " • ".join(filter(None, location_parts))
                    if location:
                        st.caption(f"📍 {location}")
                
                with header_cols[1]:
                    if relevance > 0:
                        st.metric("Relevance", f"{relevance:.0%}", delta=None)
                
                # Citation content
                if content:
                    # Truncate long content with expand option
                    max_length = 200
                    if len(content) > max_length:
                        st.markdown(f"*{content[:max_length]}...*")
                        with st.expander("Read full excerpt"):
                            st.markdown(f"*{content}*")
                    else:
                        st.markdown(f"*{content}*")
                
                if i < len(sources):
                    st.divider()


def render_enhanced_message(message: EnhancedChatMessage) -> None:
    """Render an enhanced chat message with rich formatting."""
    theme = get_current_theme()
    
    if message.message_type == 'user':
        # User message styling
        with st.chat_message("user"):
            st.markdown(f"**You:** {message.content}")
            st.caption(f"🕒 {message.timestamp.strftime('%H:%M:%S')}")
    
    else:  # assistant message
        # Assistant message with enhanced formatting
        with st.chat_message("assistant"):
            # Message header with metadata
            header_cols = st.columns([3, 1])
            with header_cols[0]:
                st.markdown("**Assistant:**")
            with header_cols[1]:
                if message.token_usage:
                    total_tokens = message.token_usage.get('total_tokens', 0)
                    if total_tokens > 0:
                        st.caption(f"🔹 {total_tokens} tokens")
            
            # Main content
            st.markdown(message.content)
            
            # Confidence score
            if message.confidence is not None:
                render_confidence_score(message.confidence)
            
            # Source citations
            if message.sources:
                render_source_citations(message.sources)
            
            # Timestamp
            st.caption(f"🕒 {message.timestamp.strftime('%H:%M:%S')}")


def export_conversation_to_text(messages: List[EnhancedChatMessage], book_title: str = "Unknown Book") -> str:
    """Export conversation to formatted text."""
    lines = [
        f"Alexandria Platform - Chat Export",
        f"Book: {book_title}",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50,
        ""
    ]
    
    for message in messages:
        # Message header
        role = "You" if message.message_type == 'user' else "Assistant"
        timestamp = message.timestamp.strftime('%H:%M:%S')
        lines.append(f"[{timestamp}] {role}:")
        lines.append(message.content)
        
        # Add confidence and sources for assistant messages
        if message.message_type == 'assistant':
            if message.confidence is not None:
                lines.append(f"Confidence: {message.confidence:.1%}")
            
            if message.sources:
                lines.append("\nSources:")
                for i, source in enumerate(message.sources, 1):
                    title = source.get('title', 'Unknown')
                    page = source.get('page_number', 'N/A')
                    lines.append(f"  {i}. {title} (Page {page})")
                    content = source.get('content', '')
                    if content:
                        lines.append(f"     \"{content[:100]}...\"")
        
        lines.append("")  # Empty line between messages
    
    return "\n".join(lines)


def export_conversation_to_json(messages: List[EnhancedChatMessage], book_title: str = "Unknown Book") -> str:
    """Export conversation to JSON format."""
    export_data = {
        "export_info": {
            "book_title": book_title,
            "export_time": datetime.now().isoformat(),
            "message_count": len(messages),
            "alexandria_platform": "Phase 1.44 Enhanced Q&A"
        },
        "conversation": [message.to_dict() for message in messages]
    }
    return json.dumps(export_data, indent=2, default=str)


def render_export_options(messages: List[EnhancedChatMessage], book_title: str = "Unknown Book") -> None:
    """Render conversation export options."""
    if not messages:
        st.info("No conversation to export yet.")
        return
    
    st.subheader("📤 Export Conversation")
    
    export_format = st.selectbox(
        "Choose export format:",
        ["Text (.txt)", "JSON (.json)", "Markdown (.md)"],
        help="Select the format for exporting your conversation"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Preview Export", use_container_width=True):
            st.session_state.show_export_preview = True
    
    with col2:
        # Generate export content
        if export_format == "Text (.txt)":
            content = export_conversation_to_text(messages, book_title)
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            mime_type = "text/plain"
        elif export_format == "JSON (.json)":
            content = export_conversation_to_json(messages, book_title)
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            mime_type = "application/json"
        else:  # Markdown
            content = export_conversation_to_markdown(messages, book_title)
            filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            mime_type = "text/markdown"
        
        st.download_button(
            label="💾 Download",
            data=content,
            file_name=filename,
            mime=mime_type,
            use_container_width=True,
            help=f"Download conversation as {export_format}"
        )
    
    with col3:
        if st.button("📊 Export Stats", use_container_width=True):
            render_conversation_statistics(messages)
    
    # Show preview if requested
    if st.session_state.get('show_export_preview', False):
        with st.expander("📋 Export Preview", expanded=True):
            if export_format == "JSON (.json)":
                st.json(content)
            else:
                st.text(content[:1000] + ("..." if len(content) > 1000 else ""))
        
        if st.button("❌ Close Preview"):
            st.session_state.show_export_preview = False


def export_conversation_to_markdown(messages: List[EnhancedChatMessage], book_title: str = "Unknown Book") -> str:
    """Export conversation to Markdown format."""
    lines = [
        f"# Alexandria Platform - Chat Export",
        f"**Book:** {book_title}  ",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "---",
        ""
    ]
    
    for message in messages:
        timestamp = message.timestamp.strftime('%H:%M:%S')
        
        if message.message_type == 'user':
            lines.append(f"## 👤 You [{timestamp}]")
            lines.append(f"{message.content}")
        else:
            lines.append(f"## 🤖 Assistant [{timestamp}]")
            lines.append(f"{message.content}")
            
            if message.confidence is not None:
                lines.append(f"**Confidence:** {message.confidence:.1%}")
            
            if message.sources:
                lines.append("### 📚 Sources")
                for i, source in enumerate(message.sources, 1):
                    title = source.get('title', 'Unknown')
                    page = source.get('page_number', 'N/A')
                    lines.append(f"{i}. **{title}** (Page {page})")
                    content = source.get('content', '')
                    if content:
                        lines.append(f"   > {content[:200]}...")
        
        lines.append("")
    
    return "\n".join(lines)


def render_conversation_statistics(messages: List[EnhancedChatMessage]) -> None:
    """Render conversation statistics and analytics."""
    if not messages:
        return
    
    # Calculate statistics
    user_messages = [m for m in messages if m.message_type == 'user']
    assistant_messages = [m for m in messages if m.message_type == 'assistant']
    
    total_tokens = sum(m.token_usage.get('total_tokens', 0) for m in assistant_messages)
    avg_confidence = sum(m.confidence for m in assistant_messages if m.confidence) / len([m for m in assistant_messages if m.confidence]) if assistant_messages else 0
    total_sources = sum(len(m.sources) for m in assistant_messages)
    
    # Display statistics
    st.subheader("📊 Conversation Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(messages))
    with col2:
        st.metric("Questions Asked", len(user_messages))
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    with col4:
        st.metric("Sources Cited", total_sources)
    
    # Token usage chart
    if total_tokens > 0:
        st.subheader("🔹 Token Usage")
        token_data = []
        for i, message in enumerate(assistant_messages):
            tokens = message.token_usage.get('total_tokens', 0)
            if tokens > 0:
                token_data.append({'Message': f"Response {i+1}", 'Tokens': tokens})
        
        if token_data:
            df = pd.DataFrame(token_data)
            fig = px.bar(df, x='Message', y='Tokens', title="Token Usage per Response")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Confidence over time
    if len([m for m in assistant_messages if m.confidence]) > 1:
        st.subheader("📈 Confidence Trends")
        confidence_data = []
        for i, message in enumerate(assistant_messages):
            if message.confidence is not None:
                confidence_data.append({
                    'Response': f"Response {i+1}",
                    'Confidence': message.confidence,
                    'Time': message.timestamp
                })
        
        if confidence_data:
            df = pd.DataFrame(confidence_data)
            fig = px.line(df, x='Response', y='Confidence', 
                         title="Response Confidence Over Time",
                         line_shape='spline')
            fig.update_layout(height=300, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)


def initialize_enhanced_chat_session() -> None:
    """Initialize enhanced chat session state."""
    if 'enhanced_chat_history' not in st.session_state:
        st.session_state.enhanced_chat_history = []
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    if 'show_export_preview' not in st.session_state:
        st.session_state.show_export_preview = False


def clear_enhanced_chat_history() -> None:
    """Clear enhanced chat history."""
    st.session_state.enhanced_chat_history = []
    st.session_state.conversation_id = None
    st.session_state.show_export_preview = False


def add_enhanced_message(message: EnhancedChatMessage) -> None:
    """Add enhanced message to chat history."""
    if 'enhanced_chat_history' not in st.session_state:
        st.session_state.enhanced_chat_history = []
    
    st.session_state.enhanced_chat_history.append(message)


def get_enhanced_chat_history() -> List[EnhancedChatMessage]:
    """Get enhanced chat history."""
    if 'enhanced_chat_history' not in st.session_state:
        return []
    return st.session_state.enhanced_chat_history