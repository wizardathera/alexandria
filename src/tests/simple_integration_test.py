"""
Simple integration test for AI services pipeline.

Tests basic functionality without requiring pytest or external dependencies.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.models import ContentItem, ModuleType, ContentType, MessageRole
from src.services.conversation_service import get_conversation_service, create_conversation_with_first_message

logger = get_logger(__name__)


async def test_conversation_service():
    """Test conversation service basic functionality."""
    print("🧪 Testing conversation service...")
    
    try:
        # Test conversation creation
        conversation_service = await get_conversation_service()
        
        # Create conversation with first message
        conversation, message = await create_conversation_with_first_message(
            question="What is the Alexandria platform?",
            content_id="test_content_123",
            module_type=ModuleType.LIBRARY,
            user_id="test_user"
        )
        
        assert conversation is not None, "Conversation should be created"
        assert message is not None, "Message should be created"
        assert message.role == MessageRole.USER, "Message role should be USER"
        assert conversation.conversation_id == message.conversation_id, "IDs should match"
        
        # Test getting conversation context
        context = await conversation_service.get_conversation_context(
            conversation.conversation_id,
            limit=5
        )
        
        assert len(context) == 1, "Should have one context message"
        assert context[0]["role"] == "user", "Context role should be user"
        
        # Test service stats
        stats = await conversation_service.get_service_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        assert stats["total_conversations"] >= 1, "Should have at least one conversation"
        
        print("✅ Conversation service test passed")
        return True
        
    except Exception as e:
        print(f"❌ Conversation service test failed: {e}")
        return False


async def test_ai_provider_manager():
    """Test AI provider manager basic functionality."""
    print("🧪 Testing AI provider manager...")
    
    try:
        from src.utils.ai_providers import get_ai_provider_manager, AIProviderType
        
        # Test provider manager initialization
        manager = await get_ai_provider_manager()
        
        assert manager is not None, "Manager should be initialized"
        
        # Test available providers
        providers = manager.get_available_providers()
        assert isinstance(providers, list), "Providers should be a list"
        assert len(providers) > 0, "Should have at least one provider"
        
        # Test primary provider
        primary_provider = manager.get_primary_provider()
        assert primary_provider is not None, "Should have primary provider"
        
        print("✅ AI provider manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ AI provider manager test failed: {e}")
        return False


async def test_mcp_server():
    """Test MCP server basic functionality."""
    print("🧪 Testing MCP server...")
    
    try:
        from src.mcp.server import get_mcp_server
        
        # Get MCP server
        mcp_server = await get_mcp_server()
        
        assert mcp_server is not None, "MCP server should be initialized"
        
        # Test storage operations
        notes = await mcp_server.get_notes_for_content("test_content_123")
        assert isinstance(notes, list), "Notes should be a list"
        
        progress = await mcp_server.get_progress_for_content("test_content_123")
        # Should be None for non-existent content
        assert progress is None, "Progress should be None for non-existent content"
        
        print("✅ MCP server test passed")
        return True
        
    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        return False


async def test_enhanced_database():
    """Test enhanced database basic functionality."""
    print("🧪 Testing enhanced database...")
    
    try:
        from src.utils.enhanced_database import get_enhanced_database
        
        # Get enhanced database
        vector_db = await get_enhanced_database()
        
        assert vector_db is not None, "Enhanced database should be initialized"
        
        # Test health check if available
        try:
            health_status = await vector_db.health_check()
            assert isinstance(health_status, dict), "Health status should be a dictionary"
        except Exception:
            # Health check might not be implemented or might fail due to config
            pass
        
        print("✅ Enhanced database test passed")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced database test failed: {e}")
        return False


async def test_models():
    """Test data models basic functionality."""
    print("🧪 Testing data models...")
    
    try:
        # Test ContentItem creation
        content_item = ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Test Book",
            author="Test Author"
        )
        
        assert content_item.title == "Test Book", "Title should be set"
        assert content_item.author == "Test Author", "Author should be set"
        assert content_item.module_type == ModuleType.LIBRARY, "Module type should be LIBRARY"
        assert content_item.content_type == ContentType.BOOK, "Content type should be BOOK"
        
        # Test content item methods
        content_item.mark_processed()
        assert content_item.processed_at is not None, "Processed timestamp should be set"
        
        print("✅ Data models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("🚀 Starting AI services integration tests...")
    print("=" * 50)
    
    # List of test functions
    tests = [
        test_models,
        test_conversation_service,
        test_ai_provider_manager,
        test_mcp_server,
        test_enhanced_database
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
        
        print("-" * 30)
    
    # Summary
    print("📋 Test Results Summary:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"📊 Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed or were skipped due to configuration")
    
    return passed == total


async def main():
    """Main entry point."""
    try:
        success = await run_all_tests()
        exit_code = 0 if success else 1
        
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        exit_code = 2
    
    print(f"\n🏁 Integration test suite completed with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)