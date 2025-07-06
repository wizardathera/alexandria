#!/usr/bin/env python3
"""
Fix search endpoint functionality by addressing identified issues.
"""

import sys
import asyncio
sys.path.insert(0, 'src')

import time
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.enhanced_database import get_enhanced_database
from src.utils.config import get_settings, DEFAULT_COLLECTION_NAME

logger = get_logger(__name__)

async def fix_search_functionality():
    """Fix the main search functionality issues."""
    
    print("🔧 Fixing Search Endpoint Functionality")
    print("=" * 50)
    
    # Issue 1: Collection data mismatch
    print("\n1️⃣ Analyzing Collection Data")
    print("-" * 30)
    
    enhanced_db = await get_enhanced_database()
    settings = get_settings()
    
    # Test both collections with a known good query
    test_query = "psychology games people play"
    collections_to_test = ["alexandria_books", "dbc_books"]
    
    best_collection = None
    best_avg_similarity = 0.0
    
    for collection_name in collections_to_test:
        try:
            print(f"\n🔍 Testing collection: {collection_name}")
            
            results = await enhanced_db.query_with_permissions(
                collection_name=collection_name,
                query_text=test_query,
                n_results=5
            )
            
            documents = results.get('documents', [])
            distances = results.get('distances', [])
            
            if documents and distances:
                avg_distance = sum(distances) / len(distances)
                avg_similarity = 1.0 - avg_distance
                
                print(f"  📄 Documents: {len(documents)}")
                print(f"  📏 Avg distance: {avg_distance:.4f}")
                print(f"  🎯 Avg similarity: {avg_similarity:.4f}")
                
                # Check content quality
                sample_content = documents[0][:100] if documents else ""
                print(f"  📝 Sample: {sample_content}...")
                
                # Check metadata
                if results.get('metadatas'):
                    sample_meta = results['metadatas'][0]
                    title = sample_meta.get('book_title', 'Unknown')
                    print(f"  📚 Book: {title}")
                
                # Track best performing collection
                if avg_similarity > best_avg_similarity:
                    best_avg_similarity = avg_similarity
                    best_collection = collection_name
                    
            else:
                print(f"  ⚠️  No results found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🏆 Best performing collection: {best_collection} (similarity: {best_avg_similarity:.4f})")
    
    # Issue 2: Update DEFAULT_COLLECTION_NAME if needed
    if best_collection and best_collection != DEFAULT_COLLECTION_NAME:
        print(f"\n2️⃣ Collection Configuration Update Needed")
        print("-" * 40)
        print(f"Current DEFAULT_COLLECTION_NAME: {DEFAULT_COLLECTION_NAME}")
        print(f"Best performing collection: {best_collection}")
        print(f"Recommendation: Update config.py to use '{best_collection}'")
        
        # Show the fix needed
        print("\n📝 Config change needed in src/utils/config.py:")
        print(f'   Change: DEFAULT_COLLECTION_NAME = "{best_collection}"')
    
    # Issue 3: Test enhanced search with correct collection
    print(f"\n3️⃣ Testing Enhanced Search with {best_collection}")
    print("-" * 40)
    
    if best_collection:
        try:
            from src.services.enhanced_embedding_service import get_enhanced_embedding_service
            from src.models import User, UserRole
            
            enhanced_service = await get_enhanced_embedding_service()
            test_user = User(
                user_id="test_user",
                email="test@example.com",
                username="test_user",
                role=UserRole.READER
            )
            
            # Temporarily override collection for testing
            original_collection = DEFAULT_COLLECTION_NAME
            import src.utils.config
            src.utils.config.DEFAULT_COLLECTION_NAME = best_collection
            
            start_time = time.time()
            search_results = await enhanced_service.enhanced_search(
                query=test_query,
                user=test_user,
                n_results=5,
                include_relationships=False  # Disable for testing
            )
            search_time = time.time() - start_time
            
            # Restore original collection name
            src.utils.config.DEFAULT_COLLECTION_NAME = original_collection
            
            documents = search_results.get('documents', [])
            enhanced_results = search_results.get('enhanced_results', [])
            
            print(f"⏱️  Search time: {search_time:.3f}s")
            print(f"📄 Documents: {len(documents)}")
            print(f"🔍 Enhanced results: {len(enhanced_results)}")
            
            if documents:
                distances = search_results.get('distances', [])
                if distances:
                    avg_similarity = 1.0 - (sum(distances) / len(distances))
                    print(f"🎯 Average similarity: {avg_similarity:.4f}")
                
                print(f"✅ Enhanced search working with {best_collection}")
            else:
                print(f"⚠️  Enhanced search still not finding results")
                
        except Exception as e:
            print(f"❌ Enhanced search test failed: {e}")
    
    # Issue 4: Similarity threshold analysis
    print(f"\n4️⃣ Similarity Threshold Analysis")
    print("-" * 40)
    
    if best_collection:
        try:
            # Test with various queries to understand distance patterns
            test_queries = [
                ("psychology", "High relevance expected"),
                ("games people play", "Exact title match expected"),
                ("human behavior", "Medium relevance expected"),
                ("cooking recipes", "Low relevance expected")
            ]
            
            for query, expectation in test_queries:
                results = await enhanced_db.query_with_permissions(
                    collection_name=best_collection,
                    query_text=query,
                    n_results=3
                )
                
                if results.get('distances'):
                    avg_distance = sum(results['distances']) / len(results['distances'])
                    avg_similarity = 1.0 - avg_distance
                    print(f"'{query}': similarity {avg_similarity:.4f} ({expectation})")
            
            # Recommend similarity thresholds
            print(f"\n📊 Recommended Similarity Thresholds:")
            print(f"  🟢 High relevance: > 0.15 (distance < 0.85)")
            print(f"  🟡 Medium relevance: > 0.05 (distance < 0.95)")
            print(f"  🔴 Low relevance: > -0.05 (distance < 1.05)")
            
        except Exception as e:
            print(f"❌ Threshold analysis failed: {e}")
    
    # Issue 5: Performance validation
    print(f"\n5️⃣ Performance Validation")
    print("-" * 30)
    
    if best_collection:
        try:
            # Test search performance
            queries = ["psychology", "human behavior", "social interaction"]
            total_time = 0.0
            
            for query in queries:
                start_time = time.time()
                results = await enhanced_db.query_with_permissions(
                    collection_name=best_collection,
                    query_text=query,
                    n_results=10
                )
                query_time = time.time() - start_time
                total_time += query_time
                
                result_count = len(results.get('documents', []))
                print(f"'{query}': {query_time:.3f}s, {result_count} results")
            
            avg_time = total_time / len(queries)
            print(f"\n⏱️  Average search time: {avg_time:.3f}s")
            print(f"🎯 Performance target (<3s): {'✅ PASS' if avg_time < 3.0 else '❌ FAIL'}")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
    
    # Summary and recommendations
    print(f"\n🎯 Summary and Recommendations")
    print("=" * 50)
    
    if best_collection:
        print(f"✅ Working collection identified: {best_collection}")
        print(f"✅ Average similarity score: {best_avg_similarity:.4f}")
        
        if best_collection != DEFAULT_COLLECTION_NAME:
            print(f"⚠️  ACTION REQUIRED: Update DEFAULT_COLLECTION_NAME to '{best_collection}'")
        
        print(f"✅ Search endpoints should work after configuration update")
        
        # Specific fixes needed
        print(f"\n📋 Specific Fixes Required:")
        print(f"1. Update DEFAULT_COLLECTION_NAME in src/utils/config.py")
        print(f"2. Test enhanced search service")
        print(f"3. Validate API endpoints")
        print(f"4. Update similarity thresholds if needed")
        
    else:
        print(f"❌ No working collection found")
        print(f"🔍 Need to investigate embedding generation issues")

if __name__ == "__main__":
    asyncio.run(fix_search_functionality())