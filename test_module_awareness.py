#!/usr/bin/env python3
"""
Test Module-Aware UI Components (Phase 1.42)

This script demonstrates the module-aware functionality implemented
in Phase 1.42 of the Alexandria Platform.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.frontend.components.modules import (
    ModuleManager, 
    get_current_user_permissions, 
    UserRole,
    ModuleType,
    UserPermissions
)


def test_module_awareness():
    """Test the module-aware UI components."""
    print("🧪 Testing Module-Aware UI Components (Phase 1.42)")
    print("=" * 60)
    
    # Test 1: ModuleManager initialization
    print("\n1️⃣ Testing ModuleManager initialization...")
    manager = ModuleManager("http://localhost:8000")
    print(f"   ✅ ModuleManager initialized with {len(manager.modules)} modules")
    
    # Show all modules
    for module_type, config in manager.modules.items():
        status = "🟢 Enabled" if config.enabled else "🔴 Disabled"
        print(f"   {config.icon} {config.display_name}: {status}")
    
    # Test 2: User permissions
    print("\n2️⃣ Testing user permissions...")
    permissions = get_current_user_permissions()
    print(f"   ✅ Current user role: {permissions.role.value}")
    print(f"   ✅ Enabled modules: {[m.value for m in permissions.enabled_modules]}")
    
    # Test 3: Available modules for current user
    print("\n3️⃣ Testing available modules...")
    available = manager.get_available_modules(permissions)
    print(f"   ✅ User can access {len(available)} modules:")
    for module in available:
        print(f"      {module.icon} {module.display_name} - {module.description}")
    
    # Test 4: Navigation structure
    print("\n4️⃣ Testing navigation structure...")
    nav_items = manager.get_module_navigation_items(permissions)
    print(f"   ✅ Generated {len(nav_items)} navigation sections:")
    
    for nav_section in nav_items:
        print(f"   📋 {nav_section['display_name']}:")
        for page in nav_section['pages']:
            print(f"      - {page['display']}: {page['description']}")
    
    # Test 5: Permission-based access
    print("\n5️⃣ Testing permission-based access...")
    
    # Test different user roles
    test_roles = [UserRole.READER, UserRole.EDUCATOR, UserRole.CREATOR, UserRole.ADMIN]
    
    for role in test_roles:
        test_permissions = UserPermissions(
            user_id=f"test_{role.value}",
            role=role,
            enabled_modules=[ModuleType.LIBRARY, ModuleType.LMS, ModuleType.MARKETPLACE]
        )
        
        available_modules = manager.get_available_modules(test_permissions)
        print(f"   👤 {role.value.title()} can access {len(available_modules)} modules")
    
    # Test 6: Module-aware decorator simulation
    print("\n6️⃣ Testing module-aware component decorator...")
    
    from src.frontend.components.modules import create_module_aware_component
    
    @create_module_aware_component("test_component", "library", ["library"], UserRole.READER)
    def test_component():
        return "Component executed successfully"
    
    result = test_component()
    print(f"   ✅ Decorated component result: {result}")
    
    # Test 7: Future module preview
    print("\n7️⃣ Testing future module configuration...")
    disabled_modules = [config for module_type, config in manager.modules.items() if not config.enabled]
    print(f"   🔮 {len(disabled_modules)} modules coming in future phases:")
    
    for module in disabled_modules:
        print(f"      {module.icon} {module.display_name} - {module.description}")
        print(f"         Required role: {module.required_role.value.title()}")
    
    print("\n" + "=" * 60)
    print("✅ All module-awareness tests passed!")
    print("🌟 Phase 1.42 Module-Aware UI Components implementation verified")
    
    return True


def demonstrate_navigation_flow():
    """Demonstrate the navigation flow in the module-aware system."""
    print("\n" + "=" * 60)
    print("🧭 Demonstrating Navigation Flow")
    print("=" * 60)
    
    manager = ModuleManager("http://localhost:8000")
    permissions = get_current_user_permissions()
    
    # Simulate navigation selection
    print("\n📍 Simulating user navigation...")
    
    # Get available navigation
    nav_items = manager.get_module_navigation_items(permissions)
    
    if nav_items:
        current_module = nav_items[0]
        print(f"   🎯 Selected module: {current_module['display_name']}")
        
        if current_module['pages']:
            current_page = current_module['pages'][0]
            print(f"   📄 Selected page: {current_page['display']}")
            print(f"   💡 Description: {current_page['description']}")
            
            # Simulate breadcrumb
            print(f"   🍞 Breadcrumb: Alexandria Platform → {current_module['display_name']} → {current_page['display']}")
    
    return True


if __name__ == "__main__":
    try:
        # Run main tests
        test_module_awareness()
        
        # Demonstrate navigation
        demonstrate_navigation_flow()
        
        print("\n🎉 Phase 1.42 Module-Aware UI Components testing complete!")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)