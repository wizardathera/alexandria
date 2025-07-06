#!/usr/bin/env python3
"""
Simple collection rebuild using direct Chroma operations.
"""

import sys
import asyncio
sys.path.insert(0, 'src')

import chromadb
from pathlib import Path
from src.utils.config import get_settings, DEFAULT_COLLECTION_NAME
import uuid
import json
from datetime import datetime

async def simple_rebuild():
    """Simple rebuild using direct Chroma operations."""
    
    print("🔧 Simple Collection Rebuild")
    print("=" * 40)
    
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    print(f"\n1️⃣ Collection Analysis")
    print("-" * 25)
    
    collections = client.list_collections()
    for collection in collections:
        count = collection.count()
        print(f"📚 {collection.name}: {count} documents")
        
        if count > 0 and collection.name == "dbc_books":
            # Test search quality
            results = collection.query(
                query_texts=["psychology human behavior"],
                n_results=3
            )
            
            if results['documents'] and results['documents'][0]:
                avg_distance = sum(results['distances'][0]) / len(results['distances'][0])
                avg_similarity = 1.0 - avg_distance
                print(f"  🎯 Search quality: {avg_similarity:.4f}")
                
                sample_content = results['documents'][0][0]
                print(f"  📝 Sample: {sample_content[:100]}...")
    
    print(f"\n2️⃣ Rebuilding {DEFAULT_COLLECTION_NAME}")
    print("-" * 35)
    
    # Delete existing alexandria_books if it exists
    try:
        client.delete_collection(DEFAULT_COLLECTION_NAME)
        print(f"🗑️  Deleted existing {DEFAULT_COLLECTION_NAME}")
    except:
        print(f"ℹ️  No existing {DEFAULT_COLLECTION_NAME} to delete")
    
    # Get source data from dbc_books
    source_collection = client.get_collection("dbc_books")
    print(f"📚 Getting data from dbc_books...")
    
    all_data = source_collection.get(include=['documents', 'metadatas', 'embeddings'])
    total_docs = len(all_data['documents'])
    print(f"📊 Found {total_docs} documents to migrate")
    
    # Create new alexandria_books collection
    new_collection = client.create_collection(
        DEFAULT_COLLECTION_NAME,
        metadata={
            "created_by": "alexandria_enhanced_app",
            "created_at": datetime.now().isoformat(),
            "migrated_from": "dbc_books",
            "version": "1.3.0"
        }
    )
    print(f"✅ Created new {DEFAULT_COLLECTION_NAME} collection")
    
    # Convert metadata to enhanced format
    print(f"🔄 Converting metadata format...")
    
    enhanced_metadatas = []
    for i, (doc, meta) in enumerate(zip(all_data['documents'], all_data['metadatas'])):
        # Convert to enhanced format
        enhanced_meta = {
            # Core enhanced fields
            "embedding_id": str(uuid.uuid4()),
            "content_id": meta.get('book_id', str(uuid.uuid4())),
            "chunk_index": meta.get('chunk_index', i),
            "module_type": "library",
            "content_type": "book",
            "chunk_type": "paragraph",
            "visibility": "public",
            "creator_id": "migration_user",
            "organization_id": "",
            "language": meta.get('language', 'en'),
            "reading_level": meta.get('reading_level', 'intermediate'),
            "chunk_length": len(doc),
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimension": len(all_data['embeddings'][i]),
            "created_at": datetime.now().isoformat(),
            "importance_score": meta.get('importance_score', 0.5),
            "quality_score": meta.get('coherence_score', 0.5),
            
            # Extract semantic tags
            "semantic_tags": json.dumps(extract_semantic_tags(meta)),
            
            # Extract source location
            "source_location": json.dumps(extract_source_location(meta))
        }
        enhanced_metadatas.append(enhanced_meta)
    
    # Add data in batches
    batch_size = 100
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        
        new_collection.add(
            documents=all_data['documents'][i:batch_end],
            metadatas=enhanced_metadatas[i:batch_end],
            embeddings=all_data['embeddings'][i:batch_end],
            ids=all_data['ids'][i:batch_end]
        )
        
        print(f"  ✅ Added batch {i//batch_size + 1}: docs {i+1}-{batch_end}")
    
    print(f"\n3️⃣ Testing Rebuilt Collection")
    print("-" * 30)
    
    # Test the rebuilt collection
    test_queries = [
        "psychology human behavior",
        "games people play",
        "social interaction"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        results = new_collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if results['documents'] and results['documents'][0]:
            result_count = len(results['documents'][0])
            distances = results['distances'][0]
            avg_similarity = 1.0 - (sum(distances) / len(distances))
            
            print(f"  📄 Results: {result_count}")
            print(f"  🎯 Avg similarity: {avg_similarity:.4f}")
            print(f"  📝 Sample: {results['documents'][0][0][:100]}...")
            
            # Check enhanced metadata
            if results['metadatas'] and results['metadatas'][0]:
                sample_meta = results['metadatas'][0][0]
                print(f"  🏷️  Content ID: {sample_meta.get('content_id', 'N/A')}")
                print(f"  🔖 Module: {sample_meta.get('module_type', 'N/A')}")
                
                # Parse semantic tags
                try:
                    tags = json.loads(sample_meta.get('semantic_tags', '[]'))
                    print(f"  🏷️  Tags: {tags[:5]}")  # Show first 5 tags
                except:
                    print(f"  🏷️  Tags: parsing error")
        else:
            print(f"  ⚠️  No results found")
    
    print(f"\n✅ Collection Rebuild Complete!")
    print(f"📊 Migrated {total_docs} documents to enhanced format")
    print(f"🎯 Collection: {DEFAULT_COLLECTION_NAME}")

def extract_semantic_tags(meta: dict) -> list:
    """Extract semantic tags from original metadata."""
    tags = []
    
    # Extract from various metadata fields
    for field in ['topic_tags', 'concepts', 'entities']:
        if field in meta:
            value = meta[field]
            if isinstance(value, list):
                tags.extend(value)
            elif isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        tags.extend(parsed)
                except:
                    tags.append(value)
    
    # Add book-related tags
    if 'book_title' in meta:
        book_title = meta['book_title'].lower()
        if 'psychology' in book_title or 'psycho' in book_title:
            tags.append('psychology')
        if 'games' in book_title:
            tags.append('games')
        if 'behavior' in book_title or 'behaviour' in book_title:
            tags.append('behavior')
    
    # Add content-based tags
    if 'book_author' in meta:
        author = meta['book_author'].lower()
        if 'berne' in author:
            tags.extend(['transactional_analysis', 'psychology', 'social_psychology'])
    
    # Clean and deduplicate
    tags = [tag.strip().lower() for tag in tags if tag and isinstance(tag, str)]
    return list(set(tags))

def extract_source_location(meta: dict) -> dict:
    """Extract source location from original metadata."""
    location = {}
    
    # Map common fields
    field_mapping = {
        'page': 'page',
        'chapter': 'chapter', 
        'chunk_index': 'chunk_index',
        'source_document_index': 'document_index'
    }
    
    for old_field, new_field in field_mapping.items():
        if old_field in meta:
            location[new_field] = meta[old_field]
    
    return location

if __name__ == "__main__":
    asyncio.run(simple_rebuild())