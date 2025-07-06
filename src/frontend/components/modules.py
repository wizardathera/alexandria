"""
Module Management Components for Alexandria Platform

This module provides module-aware UI components that support the multi-module
architecture (Library, LMS, Marketplace) with permission-aware interfaces.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
import requests


class ModuleType(Enum):
    """Enumeration of available modules in the Alexandria platform."""
    LIBRARY = "library"
    LMS = "lms"
    MARKETPLACE = "marketplace"


class UserRole(Enum):
    """Enumeration of user roles with different permission levels."""
    READER = "reader"
    EDUCATOR = "educator"
    CREATOR = "creator"
    ADMIN = "admin"


@dataclass
class ModuleConfig:
    """Configuration for a platform module."""
    name: str
    display_name: str
    icon: str
    enabled: bool
    description: str
    required_role: UserRole


@dataclass
class UserPermissions:
    """User permission configuration."""
    user_id: str
    role: UserRole
    organization_id: Optional[str] = None
    enabled_modules: List[ModuleType] = None
    
    def __post_init__(self):
        if self.enabled_modules is None:
            self.enabled_modules = [ModuleType.LIBRARY]  # Default to Library only


class ModuleManager:
    """Manages module configuration and permissions."""
    
    def __init__(self, api_base_url: str):
        self.api_base_url = api_base_url
        self.modules = self._initialize_modules()
        
    def _initialize_modules(self) -> Dict[ModuleType, ModuleConfig]:
        """Initialize module configurations."""
        return {
            ModuleType.LIBRARY: ModuleConfig(
                name="library",
                display_name="Smart Library",
                icon="📚",
                enabled=True,
                description="Personal book management with AI-powered Q&A",
                required_role=UserRole.READER
            ),
            ModuleType.LMS: ModuleConfig(
                name="lms",
                display_name="Learning Suite",
                icon="🎓",
                enabled=False,  # Phase 2
                description="Course creation and learning management",
                required_role=UserRole.EDUCATOR
            ),
            ModuleType.MARKETPLACE: ModuleConfig(
                name="marketplace",
                display_name="Marketplace",
                icon="🏪",
                enabled=False,  # Phase 3
                description="Content monetization and community features",
                required_role=UserRole.CREATOR
            )
        }
    
    def get_available_modules(self, user_permissions: UserPermissions) -> List[ModuleConfig]:
        """Get list of modules available to the user based on permissions."""
        available = []
        for module_type, config in self.modules.items():
            if (config.enabled and 
                module_type in user_permissions.enabled_modules and
                self._has_module_permission(user_permissions, config)):
                available.append(config)
        return available
    
    def _has_module_permission(self, user_permissions: UserPermissions, module_config: ModuleConfig) -> bool:
        """Check if user has permission to access a module."""
        role_hierarchy = {
            UserRole.READER: 1,
            UserRole.EDUCATOR: 2,
            UserRole.CREATOR: 3,
            UserRole.ADMIN: 4
        }
        
        user_level = role_hierarchy.get(user_permissions.role, 0)
        required_level = role_hierarchy.get(module_config.required_role, 1)
        
        return user_level >= required_level
    
    def get_module_navigation_items(self, user_permissions: UserPermissions) -> List[Dict[str, Any]]:
        """Get navigation items for modules the user can access."""
        available_modules = self.get_available_modules(user_permissions)
        
        navigation_items = []
        for module in available_modules:
            navigation_items.append({
                "module": module.name,
                "display_name": f"{module.icon} {module.display_name}",
                "description": module.description,
                "pages": self._get_module_pages(module.name)
            })
        
        return navigation_items
    
    def _get_module_pages(self, module_name: str) -> List[Dict[str, str]]:
        """Get available pages for a specific module."""
        page_configs = {
            "library": [
                {"name": "book_management", "display": "📖 Book Management", "description": "Upload and manage your books"},
                {"name": "enhanced_search", "display": "🔍 Enhanced Search", "description": "Advanced search across your library"},
                {"name": "qa_chat", "display": "💬 Q&A Chat", "description": "Chat with your books using AI"},
                {"name": "relationships", "display": "🔗 Content Relationships", "description": "Explore connections between content"},
                {"name": "analytics", "display": "📊 Reading Analytics", "description": "Track your reading progress and insights"}
            ],
            "lms": [
                {"name": "course_builder", "display": "🏗️ Course Builder", "description": "Create courses from your content"},
                {"name": "learning_paths", "display": "🛤️ Learning Paths", "description": "Design learning progressions"},
                {"name": "assessments", "display": "📝 Assessments", "description": "Create quizzes and tests"},
                {"name": "student_analytics", "display": "📈 Student Analytics", "description": "Track student progress"}
            ],
            "marketplace": [
                {"name": "content_store", "display": "🛍️ Content Store", "description": "Browse and purchase content"},
                {"name": "creator_dashboard", "display": "💼 Creator Dashboard", "description": "Manage your content sales"},
                {"name": "community", "display": "👥 Community", "description": "Connect with other users"},
                {"name": "monetization", "display": "💰 Monetization", "description": "Manage pricing and earnings"}
            ]
        }
        
        return page_configs.get(module_name, [])


def get_current_user_permissions() -> UserPermissions:
    """Get current user permissions. In Phase 1, returns default single-user permissions."""
    # Phase 1: Single-user mode with default permissions
    return UserPermissions(
        user_id="default_user",
        role=UserRole.READER,
        organization_id=None,
        enabled_modules=[ModuleType.LIBRARY]  # Only Library enabled in Phase 1
    )


def render_module_aware_sidebar(api_base_url: str = "http://localhost:8000") -> Dict[str, str]:
    """
    Render a module-aware sidebar navigation.
    
    Returns:
        Dict with 'module' and 'page' keys indicating user selection
    """
    module_manager = ModuleManager(api_base_url)
    user_permissions = get_current_user_permissions()
    
    with st.sidebar:
        st.title("🌟 Alexandria Platform")
        
        # User info (placeholder for Phase 2)
        st.subheader("👤 User Profile")
        st.info(f"**Role:** {user_permissions.role.value.title()}")
        
        # Theme Selection using comprehensive theme system
        from .themes import render_theme_selector, get_theme_manager
        
        selected_theme = render_theme_selector()
        
        if selected_theme != st.session_state.get("selected_theme", "Light"):
            st.session_state.selected_theme = selected_theme
            theme_manager = get_theme_manager()
            theme_manager.set_current_theme(selected_theme)
            st.rerun()
        
        # Module Navigation
        st.subheader("🧭 Modules")
        navigation_items = module_manager.get_module_navigation_items(user_permissions)
        
        # Module selector
        module_options = [(item["module"], item["display_name"]) for item in navigation_items]
        
        if module_options:
            current_module = st.session_state.get("current_module", module_options[0][0])
            
            # Find current module display name
            module_display_names = [name for module, name in module_options]
            current_display = next((name for module, name in module_options if module == current_module), module_display_names[0])
            
            selected_module_display = st.selectbox(
                "Select Module",
                module_display_names,
                index=module_display_names.index(current_display),
                key="module_selector",
                help="Choose which module to use"
            )
            
            # Find the module key for the selected display name
            selected_module = next((module for module, name in module_options if name == selected_module_display), module_options[0][0])
            
            if selected_module != current_module:
                st.session_state.current_module = selected_module
                st.session_state.current_page = None  # Reset page when module changes
                st.rerun()
            
            # Page Navigation within selected module
            st.subheader(f"📋 {selected_module_display}")
            
            # Get pages for selected module
            current_module_nav = next((item for item in navigation_items if item["module"] == selected_module), None)
            
            if current_module_nav:
                page_options = [(page["name"], page["display"]) for page in current_module_nav["pages"]]
                
                current_page = st.session_state.get("current_page", page_options[0][0] if page_options else None)
                
                if page_options:
                    page_display_names = [name for page, name in page_options]
                    current_page_display = next((name for page, name in page_options if page == current_page), page_display_names[0])
                    
                    selected_page_display = st.selectbox(
                        "Select Page",
                        page_display_names,
                        index=page_display_names.index(current_page_display),
                        key="page_selector",
                        help="Choose which page to view"
                    )
                    
                    # Find the page key for the selected display name
                    selected_page = next((page for page, name in page_options if name == selected_page_display), page_options[0][0])
                    
                    if selected_page != current_page:
                        st.session_state.current_page = selected_page
                        st.rerun()
                    
                    # Show page description
                    page_info = next((page for page in current_module_nav["pages"] if page["name"] == selected_page), None)
                    if page_info:
                        st.caption(page_info["description"])
        
        # Phase Information
        st.subheader("📍 Development Phase")
        st.info("**Phase 1.4:** Streamlit Frontend Enhancements")
        
        # Future Modules Preview
        st.subheader("🔮 Coming Soon")
        disabled_modules = [config for module_type, config in module_manager.modules.items() if not config.enabled]
        
        for module in disabled_modules:
            st.caption(f"{module.icon} {module.display_name}")
            st.caption(f"   └ {module.description}")
        
        # System Status
        st.subheader("🔧 System Status")
        try:
            health_response = requests.get(f"{api_base_url}/api/v1/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ Backend Connected")
            else:
                st.error("❌ Backend Error")
        except:
            st.error("❌ Backend Offline")
    
    # Return current selection
    return {
        "module": st.session_state.get("current_module", module_options[0][0] if module_options else "library"),
        "page": st.session_state.get("current_page", "book_management")
    }


def render_permission_aware_interface(user_permissions: UserPermissions, module: str, page: str):
    """
    Render permission-aware interface elements based on user role and module access.
    
    Args:
        user_permissions: Current user's permissions
        module: Current module being accessed
        page: Current page being accessed
    """
    # Permission-based feature flags
    features = {
        "can_upload": user_permissions.role in [UserRole.READER, UserRole.EDUCATOR, UserRole.CREATOR, UserRole.ADMIN],
        "can_create_courses": user_permissions.role in [UserRole.EDUCATOR, UserRole.CREATOR, UserRole.ADMIN],
        "can_monetize": user_permissions.role in [UserRole.CREATOR, UserRole.ADMIN],
        "can_admin": user_permissions.role == UserRole.ADMIN
    }
    
    # Show permission-based alerts or information
    if module == "lms" and not features["can_create_courses"]:
        st.warning("🎓 Course creation requires Educator role or higher. Contact an administrator to upgrade your account.")
    
    if module == "marketplace" and not features["can_monetize"]:
        st.warning("💰 Content monetization requires Creator role or higher. Contact an administrator to upgrade your account.")
    
    # Return feature flags for use in page rendering
    return features


def render_module_breadcrumb(module: str, page: str):
    """Render a breadcrumb navigation showing current module and page."""
    module_manager = ModuleManager("http://localhost:8000")
    
    # Get module config
    module_type = ModuleType(module) if module in [m.value for m in ModuleType] else ModuleType.LIBRARY
    module_config = module_manager.modules.get(module_type)
    
    if module_config:
        module_display = f"{module_config.icon} {module_config.display_name}"
    else:
        module_display = module.title()
    
    # Get page display name
    page_configs = module_manager._get_module_pages(module)
    page_config = next((p for p in page_configs if p["name"] == page), None)
    
    if page_config:
        page_display = page_config["display"]
    else:
        page_display = page.replace("_", " ").title()
    
    # Render breadcrumb
    st.markdown(f"""
    <div style="background-color: rgba(0,0,0,0.05); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <span style="color: #666;">📍 You are here:</span> 
        <strong>{module_display}</strong> → <strong>{page_display}</strong>
    </div>
    """, unsafe_allow_html=True)


def create_module_aware_component(
    component_name: str,
    module: str,
    allowed_modules: List[str] = None,
    permission_required: UserRole = UserRole.READER
):
    """
    Decorator to create module-aware components that respect permissions.
    
    Args:
        component_name: Name of the component for logging/debugging
        module: Module this component belongs to
        allowed_modules: List of modules where this component can be used
        permission_required: Minimum role required to use this component
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            user_permissions = get_current_user_permissions()
            
            # Check module access
            if allowed_modules and module not in allowed_modules:
                st.error(f"Component '{component_name}' is not available in module '{module}'")
                return None
            
            # Check permission level
            role_hierarchy = {
                UserRole.READER: 1,
                UserRole.EDUCATOR: 2,
                UserRole.CREATOR: 3,
                UserRole.ADMIN: 4
            }
            
            user_level = role_hierarchy.get(user_permissions.role, 0)
            required_level = role_hierarchy.get(permission_required, 1)
            
            if user_level < required_level:
                st.warning(f"Component '{component_name}' requires {permission_required.value.title()} role or higher.")
                return None
            
            # Execute the component
            return func(*args, **kwargs)
        
        return wrapper
    return decorator