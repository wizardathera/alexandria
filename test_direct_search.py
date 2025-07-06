#!/usr/bin/env python3
"""
Direct test of vector search functionality with existing data.
"""

import sys
import asyncio
sys.path.insert(0, 'src')

import chromadb
from pathlib import Path
from src.utils.config import get_settings

async def test_direct_search():
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    
    print(f"Testing direct search on Chroma database at: {persist_dir}")
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Test both collections
    collections_to_test = ["alexandria_books", "dbc_books"]
    
    for collection_name in collections_to_test:
        print(f"\n🔍 Testing collection: {collection_name}")
        print("-" * 50)
        
        try:
            collection = client.get_collection(collection_name)
            count = collection.count()
            print(f"📊 Documents in collection: {count}")
            
            if count > 0:
                # Test basic query
                test_queries = ["psychology", "human behavior", "games"]
                
                for query in test_queries:
                    print(f"\n🔎 Query: '{query}'")
                    
                    try:
                        results = collection.query(
                            query_texts=[query],
                            n_results=3
                        )
                        
                        result_count = len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0
                        print(f"  📄 Results: {result_count}")
                        
                        if result_count > 0:
                            distances = results['distances'][0]
                            print(f"  📏 Top distances: {[f'{d:.4f}' for d in distances[:3]]}")
                            print(f"  🎯 Top similarities: {[f'{1.0-d:.4f}' for d in distances[:3]]}")
                            
                            # Sample content
                            top_doc = results['documents'][0][0]
                            print(f"  📝 Top result preview: {top_doc[:100]}...")
                            
                            # Sample metadata
                            if results['metadatas'] and results['metadatas'][0]:
                                top_metadata = results['metadatas'][0][0]
                                print(f"  🏷️  Top metadata keys: {list(top_metadata.keys())[:10]}")
                                print(f"  📚 Book ID: {top_metadata.get('book_id', 'Unknown')}")
                                print(f"  📖 Title: {top_metadata.get('book_title', 'Unknown')}")
                        else:
                            print("  ⚠️  No results found")
                            
                    except Exception as e:
                        print(f"  ❌ Query failed: {e}")
                
                # Test similarity search on content
                print(f"\n🔍 Testing similarity on actual content...")
                peek_data = collection.peek(limit=3)
                if peek_data['documents']:
                    sample_content = peek_data['documents'][0]
                    # Use part of actual content for similarity search
                    similarity_query = sample_content[:100]
                    print(f"  🔎 Similarity query: '{similarity_query[:50]}...'")
                    
                    results = collection.query(
                        query_texts=[similarity_query],
                        n_results=5
                    )
                    
                    result_count = len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0
                    print(f"  📄 Similarity results: {result_count}")
                    
                    if result_count > 0:
                        distances = results['distances'][0]
                        print(f"  📏 Similarity distances: {[f'{d:.4f}' for d in distances[:3]]}")
                        print(f"  🎯 Similarity scores: {[f'{1.0-d:.4f}' for d in distances[:3]]}")
                        
                        # Check if first result is exact match (should be distance ~0)
                        if distances[0] < 0.01:
                            print(f"  ✅ Exact match found (distance: {distances[0]:.6f})")
                        else:
                            print(f"  ⚠️  No exact match - closest distance: {distances[0]:.6f}")
            else:
                print(f"  📭 Collection is empty")
                
        except Exception as e:
            print(f"❌ Error testing collection {collection_name}: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_search())