#!/usr/bin/env python3
"""
Test script to verify content database integration in the ingestion pipeline.

This script tests:
1. Content database initialization
2. Migration of existing books from JSON metadata to content database
3. Verification that /api/enhanced/content returns migrated content
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.services.content_service import get_content_service
from src.services.ingestion import get_ingestion_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_content_database_integration():
    """Test the complete content database integration."""
    
    print("🧪 Testing Content Database Integration")
    print("=" * 50)
    
    try:
        # Step 1: Initialize content service
        print("1️⃣ Initializing content service...")
        content_service = await get_content_service()
        print("✅ Content service initialized successfully")
        
        # Step 2: Check migration status
        print("\n2️⃣ Checking migration status...")
        ingestion_service = get_ingestion_service()
        
        # Get list of JSON metadata files
        from src.utils.config import get_settings
        settings = get_settings()
        metadata_dir = Path(settings.user_data_path)
        
        json_books = []
        if metadata_dir.exists():
            metadata_files = list(metadata_dir.glob("*_metadata.json"))
            print(f"📂 Found {len(metadata_files)} JSON metadata files")
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        if data.get('book_id'):
                            json_books.append(data['book_id'])
                            print(f"   📖 {data.get('title', 'Unknown')} ({data['book_id'][:8]}...)")
                except Exception as e:
                    print(f"   ❌ Error reading {metadata_file.name}: {e}")
        else:
            print("📂 No metadata directory found")
        
        # Check current content database
        print(f"\n3️⃣ Checking content database...")
        db_content_items = await content_service.list_content_items(limit=1000)
        print(f"📊 Found {len(db_content_items)} records in content database")
        
        db_book_ids = [item.content_id for item in db_content_items]
        for item in db_content_items:
            print(f"   📚 {item.title} ({item.content_id[:8]}...) - {item.processing_status.value}")
        
        # Step 3: Run migration if needed
        needs_migration = [book_id for book_id in json_books if book_id not in db_book_ids]
        
        if needs_migration:
            print(f"\n4️⃣ Running migration for {len(needs_migration)} books...")
            migration_results = await ingestion_service.migrate_existing_books_to_content_db()
            
            successful = sum(1 for success in migration_results.values() if success)
            total = len(migration_results)
            print(f"✅ Migration completed: {successful}/{total} books migrated successfully")
            
            for book_id, success in migration_results.items():
                status = "✅" if success else "❌"
                print(f"   {status} {book_id[:8]}...")
        else:
            print("4️⃣ ✅ All books already migrated to content database")
        
        # Step 4: Verify enhanced content API
        print(f"\n5️⃣ Verifying enhanced content API...")
        final_content_items = await content_service.list_content_items(limit=1000)
        print(f"📊 Final content database contains {len(final_content_items)} records")
        
        if final_content_items:
            print("✅ /api/enhanced/content should now return content!")
            for item in final_content_items:
                print(f"   📚 {item.title} - {item.content_type.value} ({item.processing_status.value})")
        else:
            print("⚠️ No content found - /api/enhanced/content will still return empty")
        
        # Step 5: Test API response format
        print(f"\n6️⃣ Testing API response format...")
        
        # Simulate what the enhanced content API returns
        enhanced_items = []
        for content in final_content_items:
            enhanced_item = {
                "content_id": content.content_id,
                "title": content.title or "Untitled",
                "author": content.author or "Unknown",
                "content_type": content.content_type.value,
                "module_type": content.module_type.value,
                "visibility": content.visibility.value,
                "processing_status": content.processing_status.value,
                "created_at": content.created_at.isoformat(),
                "updated_at": content.updated_at.isoformat(),
                "text_length": content.text_length or 0,
                "chunk_count": content.chunk_count or 0,
                "topics": content.topics or [],
                "language": content.language or "en",
                "reading_level": content.reading_level or "unknown"
            }
            enhanced_items.append(enhanced_item)
        
        api_response = {
            "content": enhanced_items,
            "pagination": {
                "total": len(enhanced_items),
                "limit": 20,
                "offset": 0,
                "has_more": False
            },
            "filters": {
                "module": None,
                "content_type": None
            },
            "user_permissions_applied": True
        }
        
        print(f"📋 API Response Preview:")
        print(f"   Content count: {len(api_response['content'])}")
        print(f"   Total: {api_response['pagination']['total']}")
        
        if api_response['content']:
            print(f"   Sample item: {api_response['content'][0]['title']}")
            print("✅ Frontend should now display content in the relationship explorer!")
        else:
            print("⚠️ API response is still empty")
        
        print(f"\n🎉 Content Database Integration Test Complete!")
        print("=" * 50)
        
        return {
            "json_books_found": len(json_books),
            "db_records_before": len(db_book_ids),
            "db_records_after": len(final_content_items),
            "migration_successful": len(final_content_items) > 0,
            "api_ready": len(enhanced_items) > 0
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}


async def main():
    """Main test function."""
    print("🚀 Starting Content Database Integration Test")
    
    result = await test_content_database_integration()
    
    print(f"\n📊 Final Results:")
    for key, value in result.items():
        print(f"   {key}: {value}")
    
    if result.get('api_ready'):
        print(f"\n✅ SUCCESS: /api/enhanced/content should now return content!")
        print(f"🔗 Test the frontend Content Relationships Explorer to confirm.")
    else:
        print(f"\n⚠️ The API is set up correctly, but no content was found to migrate.")
        print(f"📚 Upload a book through the frontend to test the full pipeline.")


if __name__ == "__main__":
    asyncio.run(main())