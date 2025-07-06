"""
Simple MCP client test for Alexandria MCP server.

This script tests the basic functionality of the Alexandria MCP server tools.
"""

import asyncio
import json
from typing import Dict, Any

from src.mcp.server import get_mcp_server
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_mcp_tools():
    """Test the MCP server tools."""
    logger.info("Starting MCP tools test...")
    
    try:
        # Get MCP server instance
        server = await get_mcp_server()
        
        # Test 1: Add a note (will fail without real content, but should handle gracefully)
        logger.info("Testing add_note tool...")
        note_result = await server.app.tools["add_note"](
            content_id="test_book_123",
            note_text="This is a test note about the content.",
            note_type="insight",
            tags=["test", "example"],
            section_reference="Chapter 1, Page 5"
        )
        logger.info(f"Add note result: {json.dumps(note_result, indent=2)}")
        
        # Test 2: Fetch resources
        logger.info("Testing fetch_resource tool...")
        resource_result = await server.app.tools["fetch_resource"](
            resource_type="articles",
            query="artificial intelligence",
            limit=3
        )
        logger.info(f"Fetch resource result: {json.dumps(resource_result, indent=2)}")
        
        # Test 3: Update progress (will fail without real content, but should handle gracefully)
        logger.info("Testing update_progress tool...")
        progress_result = await server.app.tools["update_progress"](
            content_id="test_book_123",
            progress_type="reading",
            progress_value=0.25,
            milestone="Completed Chapter 1",
            notes="Good progress so far"
        )
        logger.info(f"Update progress result: {json.dumps(progress_result, indent=2)}")
        
        logger.info("MCP tools test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"MCP tools test failed: {e}")
        return False


async def test_storage_operations():
    """Test the storage operations of the MCP server."""
    logger.info("Testing MCP storage operations...")
    
    try:
        server = await get_mcp_server()
        
        # Test getting notes for content (should return empty list for non-existent content)
        notes = await server.get_notes_for_content("test_book_123")
        logger.info(f"Notes for test content: {len(notes)} notes found")
        
        # Test getting progress for content (should return None for non-existent content)
        progress = await server.get_progress_for_content("test_book_123")
        logger.info(f"Progress for test content: {progress is not None}")
        
        logger.info("Storage operations test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Storage operations test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("Starting Alexandria MCP Server tests...")
    
    # Test tools
    tools_success = await test_mcp_tools()
    
    # Test storage
    storage_success = await test_storage_operations()
    
    if tools_success and storage_success:
        logger.info("All MCP tests passed!")
    else:
        logger.error("Some MCP tests failed!")
    
    return tools_success and storage_success


if __name__ == "__main__":
    asyncio.run(main())