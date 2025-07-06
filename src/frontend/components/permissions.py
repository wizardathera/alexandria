"""
Permission-Aware UI Components for Alexandria Platform

This module provides UI components that respect user permissions and content visibility
controls, integrating with the RAG query system and content management.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import requests
from datetime import datetime

from .modules import UserRole, UserPermissions, get_current_user_permissions


class ContentVisibility(Enum):
    """Content visibility levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    ORGANIZATION = "organization"
    RESTRICTED = "restricted"


class PermissionLevel(Enum):
    """Permission levels for content access."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    OWNER = "owner"


@dataclass
class ContentPermissions:
    """Content-specific permission configuration."""
    content_id: str
    visibility: ContentVisibility
    owner_id: str
    organization_id: Optional[str] = None
    allowed_roles: List[UserRole] = None
    allowed_users: List[str] = None
    
    def __post_init__(self):
        if self.allowed_roles is None:
            self.allowed_roles = [UserRole.READER]
        if self.allowed_users is None:
            self.allowed_users = []


class PermissionManager:
    """Manages permission checking and content access control."""
    
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self._permission_cache = {}
        self._cache_timestamp = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    def can_access_content(self, user_permissions: UserPermissions, content_permissions: ContentPermissions) -> bool:
        """Check if user can access specific content."""
        cache_key = f"{user_permissions.user_id}_{content_permissions.content_id}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self._permission_cache[cache_key]
        
        # Calculate permission
        can_access = self._calculate_content_access(user_permissions, content_permissions)
        
        # Cache result
        self._permission_cache[cache_key] = can_access
        self._cache_timestamp[cache_key] = datetime.now()
        
        return can_access
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached permission is still valid."""
        if cache_key not in self._permission_cache:
            return False
        
        cache_time = self._cache_timestamp.get(cache_key)
        if not cache_time:
            return False
        
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl
    
    def _calculate_content_access(self, user_permissions: UserPermissions, content_permissions: ContentPermissions) -> bool:
        """Calculate if user has access to content based on permissions."""
        # Owner always has access
        if user_permissions.user_id == content_permissions.owner_id:
            return True
        
        # Admin always has access
        if user_permissions.role == UserRole.ADMIN:
            return True
        
        # Check visibility level
        if content_permissions.visibility == ContentVisibility.PUBLIC:
            return True
        
        if content_permissions.visibility == ContentVisibility.PRIVATE:
            # Only owner and explicitly allowed users
            return user_permissions.user_id in content_permissions.allowed_users
        
        if content_permissions.visibility == ContentVisibility.ORGANIZATION:
            # Same organization members with appropriate role
            if (user_permissions.organization_id and 
                user_permissions.organization_id == content_permissions.organization_id):
                return user_permissions.role in content_permissions.allowed_roles
        
        if content_permissions.visibility == ContentVisibility.RESTRICTED:
            # Specific role requirements
            return user_permissions.role in content_permissions.allowed_roles
        
        return False
    
    def filter_accessible_content(self, user_permissions: UserPermissions, content_list: List[Dict]) -> List[Dict]:
        """Filter content list to only include items the user can access."""
        accessible_content = []
        
        for content in content_list:
            # Create content permissions from content metadata
            content_permissions = ContentPermissions(
                content_id=content.get('content_id', ''),
                visibility=ContentVisibility(content.get('visibility', 'public')),
                owner_id=content.get('owner_id', 'default_user'),
                organization_id=content.get('organization_id'),
                allowed_roles=[UserRole(role) for role in content.get('allowed_roles', ['reader'])],
                allowed_users=content.get('allowed_users', [])
            )
            
            if self.can_access_content(user_permissions, content_permissions):
                accessible_content.append(content)
        
        return accessible_content
    
    def get_permission_summary(self, user_permissions: UserPermissions) -> Dict[str, Any]:
        """Get a summary of user's permission status."""
        return {
            "user_id": user_permissions.user_id,
            "role": user_permissions.role.value,
            "organization_id": user_permissions.organization_id,
            "enabled_modules": [module.value for module in user_permissions.enabled_modules],
            "cache_size": len(self._permission_cache),
            "permission_level": self._get_permission_level_description(user_permissions.role)
        }
    
    def _get_permission_level_description(self, role: UserRole) -> str:
        """Get human-readable permission level description."""
        descriptions = {
            UserRole.READER: "Can read and search content, ask questions",
            UserRole.EDUCATOR: "Can create courses, manage learning materials",
            UserRole.CREATOR: "Can publish and monetize content",
            UserRole.ADMIN: "Full platform access and user management"
        }
        return descriptions.get(role, "Unknown permission level")
    
    def clear_cache(self) -> None:
        """Clear permission cache."""
        self._permission_cache.clear()
        self._cache_timestamp.clear()


def render_permission_status(user_permissions: UserPermissions, show_details: bool = False) -> None:
    """Render user permission status in the UI."""
    permission_manager = PermissionManager("http://localhost:8000")
    
    # Permission status indicator
    role_colors = {
        UserRole.READER: "🟢",
        UserRole.EDUCATOR: "🟡", 
        UserRole.CREATOR: "🟠",
        UserRole.ADMIN: "🔴"
    }
    
    role_icon = role_colors.get(user_permissions.role, "⚪")
    
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{role_icon} Current Role:** {user_permissions.role.value.title()}")
        
        with col2:
            if st.button("ℹ️ Details", key="permission_details"):
                st.session_state.show_permission_details = not st.session_state.get('show_permission_details', False)
        
        if show_details or st.session_state.get('show_permission_details', False):
            with st.expander("🔐 Permission Details", expanded=True):
                permission_summary = permission_manager.get_permission_summary(user_permissions)
                
                st.markdown(f"**User ID:** `{permission_summary['user_id']}`")
                st.markdown(f"**Permission Level:** {permission_summary['permission_level']}")
                
                if permission_summary['organization_id']:
                    st.markdown(f"**Organization:** {permission_summary['organization_id']}")
                
                st.markdown("**Enabled Modules:**")
                for module in permission_summary['enabled_modules']:
                    st.markdown(f"  • {module.title()}")
                
                if permission_summary['cache_size'] > 0:
                    st.markdown(f"**Cached Permissions:** {permission_summary['cache_size']} items")


def render_content_visibility_selector(current_visibility: ContentVisibility = ContentVisibility.PUBLIC) -> ContentVisibility:
    """Render content visibility selector for content owners."""
    visibility_options = {
        "🌐 Public": ContentVisibility.PUBLIC,
        "🔒 Private": ContentVisibility.PRIVATE,
        "🏢 Organization": ContentVisibility.ORGANIZATION,
        "🚫 Restricted": ContentVisibility.RESTRICTED
    }
    
    visibility_descriptions = {
        ContentVisibility.PUBLIC: "Visible to all users",
        ContentVisibility.PRIVATE: "Only you can access this content",
        ContentVisibility.ORGANIZATION: "Members of your organization can access",
        ContentVisibility.RESTRICTED: "Only users with specific roles can access"
    }
    
    # Find current selection
    current_key = None
    for key, value in visibility_options.items():
        if value == current_visibility:
            current_key = key
            break
    
    selected_key = st.selectbox(
        "Content Visibility:",
        options=list(visibility_options.keys()),
        index=list(visibility_options.keys()).index(current_key) if current_key else 0,
        help="Control who can access this content"
    )
    
    selected_visibility = visibility_options[selected_key]
    
    # Show description
    st.caption(f"ℹ️ {visibility_descriptions[selected_visibility]}")
    
    return selected_visibility


def apply_permission_filters_to_search(user_permissions: UserPermissions, search_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Apply user permission filters to search payload."""
    # Add user context to search payload
    search_payload["user_context"] = {
        "user_id": user_permissions.user_id,
        "role": user_permissions.role.value,
        "organization_id": user_permissions.organization_id,
        "enabled_modules": [module.value for module in user_permissions.enabled_modules]
    }
    
    # Add permission filtering flags
    search_payload["apply_permissions"] = True
    search_payload["respect_visibility"] = True
    
    return search_payload


def render_permission_aware_search_results(search_results: List[Dict], user_permissions: UserPermissions) -> Tuple[List[Dict], Dict[str, int]]:
    """Render search results with permission filtering and statistics."""
    permission_manager = PermissionManager("http://localhost:8000")
    
    # Filter results by permissions
    accessible_results = permission_manager.filter_accessible_content(user_permissions, search_results)
    
    # Calculate statistics
    total_results = len(search_results)
    accessible_count = len(accessible_results)
    filtered_count = total_results - accessible_count
    
    permission_stats = {
        "total_results": total_results,
        "accessible_results": accessible_count,
        "filtered_results": filtered_count,
        "access_rate": (accessible_count / total_results * 100) if total_results > 0 else 0
    }
    
    # Display permission filtering info if results were filtered
    if filtered_count > 0:
        st.info(f"🔐 Permission filter applied: {accessible_count} of {total_results} results accessible to your role ({permission_stats['access_rate']:.1f}%)")
        
        with st.expander("🔍 View Filtering Details"):
            st.markdown(f"**Your Role:** {user_permissions.role.value.title()}")
            st.markdown(f"**Accessible Results:** {accessible_count}")
            st.markdown(f"**Filtered Results:** {filtered_count}")
            
            if filtered_count > 0:
                st.markdown("**Reasons for filtering:**")
                st.markdown("• Content visibility restrictions")
                st.markdown("• Organization membership requirements")
                st.markdown("• Role-based access controls")
    
    return accessible_results, permission_stats


def get_permission_manager() -> PermissionManager:
    """Get global permission manager instance."""
    if 'permission_manager' not in st.session_state:
        st.session_state.permission_manager = PermissionManager("http://localhost:8000")
    return st.session_state.permission_manager


def clear_permission_cache() -> None:
    """Clear permission cache for current session."""
    if 'permission_manager' in st.session_state:
        st.session_state.permission_manager.clear_cache()
        st.success("🗑️ Permission cache cleared successfully!")