"""
Content relationship visualization components for the DBC Streamlit frontend.

This module provides visualization and interaction components for exploring
content relationships, semantic connections, and recommendation networks.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import random
import math


class RelationshipVisualizationComponent:
    """Component for visualizing and exploring content relationships."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
    
    def get_content_relationships(self, content_id: str) -> List[Dict]:
        """Get content relationships from the backend API."""
        try:
            response = requests.get(f"{self.api_base_url}/api/enhanced/content/{content_id}/relationships")
            response.raise_for_status()
            return response.json().get("relationships", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch relationships: {e}")
            return []
    
    def get_all_content_for_network(self) -> List[Dict]:
        """Get all content items for network visualization."""
        try:
            response = requests.get(f"{self.api_base_url}/api/enhanced/content")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch content for network: {e}")
            return []
    
    def render_content_selector(self, content_list: List[Dict]) -> Optional[Dict]:
        """Render content selection interface."""
        if not content_list:
            st.warning("📚 No content available for relationship exploration.")
            return None
        
        st.subheader("📖 Select Content to Explore")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            content_options = {
                f"{item.get('title', 'Unknown')} - {item.get('author', 'Unknown')}": item
                for item in content_list
            }
            
            selected_title = st.selectbox(
                "Choose content:",
                options=list(content_options.keys()),
                help="Select which content to explore relationships for"
            )
            
            selected_content = content_options[selected_title]
        
        with col2:
            st.markdown("**Selected Content:**")
            st.write(f"📄 **Type:** {selected_content.get('content_type', 'unknown').title()}")
            st.write(f"🏗️ **Module:** {selected_content.get('module_type', 'unknown').title()}")
            
            if selected_content.get('semantic_tags'):
                st.write(f"🏷️ **Tags:** {len(selected_content['semantic_tags'])}")
        
        return selected_content
    
    def render_relationship_network(self, content_item: Dict, relationships: List[Dict]):
        """Render an interactive network visualization of relationships."""
        st.subheader("🌐 Relationship Network")
        
        if not relationships:
            st.info("🔍 No relationships found for this content yet.")
            return
        
        # Create network graph
        G = nx.Graph()
        
        # Add main node
        main_id = content_item.get('content_id', 'main')
        main_title = content_item.get('title', 'Unknown')
        G.add_node(main_id, 
                  title=main_title,
                  type=content_item.get('content_type', 'unknown'),
                  is_main=True)
        
        # Add related nodes and edges
        for rel in relationships:
            related_id = rel.get('related_content_id', f"rel_{random.randint(1000, 9999)}")
            related_title = rel.get('related_title', 'Unknown')
            
            G.add_node(related_id,
                      title=related_title,
                      type=rel.get('related_content_type', 'unknown'),
                      is_main=False)
            
            G.add_edge(main_id, related_id,
                      weight=rel.get('strength', 0.5),
                      relationship_type=rel.get('relationship_type', 'related'))
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            edge_info.append(f"Relationship: {edge_data.get('relationship_type', 'related')}<br>"
                           f"Strength: {edge_data.get('weight', 0.5):.1%}")
        
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=2, color='rgba(125,125,125,0.5)'),
                                hoverinfo='none',
                                mode='lines',
                                showlegend=False))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_text.append(f"Title: {node_data['title']}<br>"
                           f"Type: {node_data['type'].title()}")
            
            # Color and size based on node type
            if node_data.get('is_main', False):
                node_color.append('red')
                node_size.append(30)
            else:
                node_color.append('lightblue')
                node_size.append(20)
        
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                hovertemplate='%{text}<extra></extra>',
                                text=node_text,
                                textposition="middle center",
                                marker=dict(size=node_size,
                                          color=node_color,
                                          line=dict(width=2, color='black')),
                                showlegend=False))
        
        fig.update_layout(
            title="Content Relationship Network",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Red node = Selected content, Blue nodes = Related content",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_relationship_details(self, relationships: List[Dict]):
        """Render detailed relationship information."""
        st.subheader("🔗 Relationship Details")
        
        if not relationships:
            st.info("🔍 No detailed relationships to display.")
            return
        
        # Group relationships by type
        rel_by_type = {}
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'related')
            if rel_type not in rel_by_type:
                rel_by_type[rel_type] = []
            rel_by_type[rel_type].append(rel)
        
        # Display each relationship type
        for rel_type, rels in rel_by_type.items():
            with st.expander(f"📎 {rel_type.title()} ({len(rels)})", expanded=True):
                for i, rel in enumerate(rels):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**📄 {rel.get('related_title', 'Unknown')}**")
                        if rel.get('explanation'):
                            st.write(f"*{rel['explanation']}*")
                        
                        # Show semantic tags if available
                        if rel.get('related_semantic_tags'):
                            tags_html = " ".join([f"<span style='background-color: #f0f8ff; padding: 2px 6px; border-radius: 10px; font-size: 0.75em;'>{tag}</span>" 
                                                for tag in rel['related_semantic_tags'][:5]])
                            st.markdown(f"**Tags:** {tags_html}", unsafe_allow_html=True)
                    
                    with col2:
                        st.write("**🎯 Strength:**")
                        strength = rel.get('strength', 0.5)
                        st.progress(strength)
                        st.write(f"{strength:.1%}")
                    
                    with col3:
                        st.write("**🔍 Actions:**")
                        if st.button(f"🔍 Explore", key=f"explore_rel_{i}_{rel_type}"):
                            if rel.get('related_content_id'):
                                st.session_state.selected_content_id = rel['related_content_id']
                                st.success("Content selected!")
                        
                        if st.button(f"💬 Discuss", key=f"discuss_rel_{i}_{rel_type}"):
                            st.session_state.discussion_content = rel
                            st.success("Added to discussion!")
                    
                    if i < len(rels) - 1:
                        st.divider()
    
    def render_relationship_analytics(self, relationships: List[Dict]):
        """Render analytics about relationships."""
        st.subheader("📊 Relationship Analytics")
        
        if not relationships:
            st.info("📊 No relationship data for analytics.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Relations", len(relationships))
        
        with col2:
            avg_strength = sum(rel.get('strength', 0) for rel in relationships) / len(relationships)
            st.metric("Avg Strength", f"{avg_strength:.1%}")
        
        with col3:
            rel_types = set(rel.get('relationship_type', 'unknown') for rel in relationships)
            st.metric("Relation Types", len(rel_types))
        
        with col4:
            strong_relations = sum(1 for rel in relationships if rel.get('strength', 0) > 0.7)
            st.metric("Strong Relations", strong_relations)
        
        # Relationship type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            rel_types = [rel.get('relationship_type', 'unknown') for rel in relationships]
            type_counts = pd.Series(rel_types).value_counts()
            
            if not type_counts.empty:
                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Relationship Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strength distribution
            strengths = [rel.get('strength', 0) for rel in relationships]
            
            fig = px.histogram(
                x=strengths,
                nbins=10,
                title="Relationship Strength Distribution",
                labels={'x': 'Strength', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_discovery_suggestions(self, content_item: Dict, relationships: List[Dict]):
        """Render content discovery suggestions based on relationships."""
        st.subheader("💡 Discovery Suggestions")
        
        if not relationships:
            st.info("🔍 Add more content to get personalized discovery suggestions.")
            return
        
        # Generate suggestions based on relationships
        suggestions = []
        
        # Strong relationships suggest exploring deeper
        strong_rels = [rel for rel in relationships if rel.get('strength', 0) > 0.7]
        if strong_rels:
            suggestions.append({
                "title": "🔥 Explore Strong Connections",
                "description": f"You have {len(strong_rels)} strong relationships. These are highly relevant to your current content.",
                "action": "Dive deeper into these connections",
                "priority": "high"
            })
        
        # Different content types suggest cross-disciplinary learning
        content_types = set(rel.get('related_content_type', 'unknown') for rel in relationships)
        if len(content_types) > 1:
            suggestions.append({
                "title": "🌐 Cross-Disciplinary Learning",
                "description": f"Your content connects to {len(content_types)} different types of material.",
                "action": "Explore connections across different content types",
                "priority": "medium"
            })
        
        # Many relationships suggest building a learning path
        if len(relationships) > 5:
            suggestions.append({
                "title": "🛤️ Build Learning Path",
                "description": f"With {len(relationships)} related items, you can create a structured learning journey.",
                "action": "Create a custom learning sequence",
                "priority": "medium"
            })
        
        # Render suggestions
        for suggestion in suggestions:
            with st.container():
                priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                
                st.markdown(f"""
                <div style='padding: 1rem; border-left: 4px solid #1f77b4; background-color: rgba(31, 119, 180, 0.1); margin: 0.5rem 0;'>
                    <h4>{priority_color.get(suggestion['priority'], '🔵')} {suggestion['title']}</h4>
                    <p>{suggestion['description']}</p>
                    <p><strong>Suggested Action:</strong> {suggestion['action']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🚀 Take Action", key=f"action_{suggestion['title'][:10]}"):
                    st.success(f"Action noted: {suggestion['action']}")
    
    def render_relationship_overview(self):
        """Render the main relationship visualization interface."""
        st.title("🔗 Content Relationships Explorer")
        
        # Get all content
        content_list = self.get_all_content_for_network()
        
        if not content_list:
            st.warning("📚 No content available. Please upload some books first.")
            return
        
        # Content selection
        selected_content = self.render_content_selector(content_list)
        
        if not selected_content:
            return
        
        # Get relationships for selected content
        relationships = self.get_content_relationships(selected_content.get('content_id', ''))
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🌐 Network View", "📋 Details", "📊 Analytics", "💡 Discovery"])
        
        with tab1:
            self.render_relationship_network(selected_content, relationships)
        
        with tab2:
            self.render_relationship_details(relationships)
        
        with tab3:
            self.render_relationship_analytics(relationships)
        
        with tab4:
            self.render_discovery_suggestions(selected_content, relationships)


def get_relationship_component(api_base_url: str = "http://localhost:8000") -> RelationshipVisualizationComponent:
    """Get an instance of the relationship visualization component."""
    return RelationshipVisualizationComponent(api_base_url)