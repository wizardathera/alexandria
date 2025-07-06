"""
Frontend components for the Dynamic Book Companion (DBC) Streamlit application.

This package contains reusable UI components that provide enhanced functionality
for the DBC platform including search, visualization, and interaction components.
"""

from .search import AdvancedSearchComponent, get_search_component
from .relationships import RelationshipVisualizationComponent, get_relationship_component

__all__ = [
    "AdvancedSearchComponent", 
    "get_search_component",
    "RelationshipVisualizationComponent",
    "get_relationship_component"
]