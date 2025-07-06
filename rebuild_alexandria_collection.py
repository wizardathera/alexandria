#!/usr/bin/env python3
"""
Rebuild the alexandria_books collection with proper enhanced metadata.

This script:
1. Analyzes existing data in both collections
2. Extracts the best content (from dbc_books if it has better data)
3. Rebuilds alexandria_books with enhanced metadata format
4. Tests the rebuilt collection
"""

import sys
import asyncio
sys.path.insert(0, 'src')

import chromadb
from pathlib import Path
from src.utils.config import get_settings, DEFAULT_COLLECTION_NAME
from src.utils.logger import get_logger
from src.utils.enhanced_database import get_enhanced_database
from src.models import EmbeddingMetadata, ModuleType, ContentType, ContentVisibility
import uuid
import json
from datetime import datetime

logger = get_logger(__name__)

async def rebuild_alexandria_collection():
    """Rebuild alexandria_books collection with enhanced metadata."""
    
    print("🔧 Rebuilding Alexandria Books Collection")
    print("=" * 50)
    
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    
    # Direct Chroma client for data migration
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    print(f"\n1️⃣ Analyzing Existing Collections")
    print("-" * 40)
    
    collections = client.list_collections()
    collection_analysis = {}
    
    for collection in collections:
        print(f"\n📚 Collection: {collection.name}")
        count = collection.count()
        print(f"  📊 Documents: {count}")
        
        if count > 0:
            # Sample some data
            sample = collection.peek(limit=3)
            
            # Test search quality with psychology query
            try:
                results = collection.query(
                    query_texts=["psychology human behavior"],
                    n_results=3
                )
                
                if results['documents'] and results['documents'][0]:
                    avg_distance = sum(results['distances'][0]) / len(results['distances'][0])
                    avg_similarity = 1.0 - avg_distance
                    
                    # Check content quality
                    sample_content = results['documents'][0][0]
                    
                    # Check if it mentions psychology/games content
                    content_quality = 0
                    quality_keywords = ['psychology', 'freud', 'berne', 'games', 'behavior', 'social']
                    for keyword in quality_keywords:
                        if keyword.lower() in sample_content.lower():
                            content_quality += 1
                    
                    collection_analysis[collection.name] = {
                        'count': count,
                        'avg_similarity': avg_similarity,
                        'content_quality': content_quality,
                        'sample_content': sample_content[:200],
                        'metadata_keys': list(sample['metadatas'][0].keys()) if sample['metadatas'] else []
                    }
                    
                    print(f"  🎯 Avg similarity: {avg_similarity:.4f}")
                    print(f"  📝 Content quality score: {content_quality}/6")
                    print(f"  📄 Sample: {sample_content[:100]}...")
                    
            except Exception as e:
                print(f"  ❌ Query test failed: {e}")
                collection_analysis[collection.name] = {'count': count, 'error': str(e)}
    
    # Determine the best source collection
    best_collection = None
    best_score = 0
    
    for name, analysis in collection_analysis.items():
        if 'error' not in analysis:
            # Score based on content quality and similarity
            score = analysis.get('content_quality', 0) + (analysis.get('avg_similarity', 0) * 2)
            print(f"\n📊 {name}: Score {score:.2f} (quality: {analysis.get('content_quality', 0)}, similarity: {analysis.get('avg_similarity', 0):.4f})")
            
            if score > best_score:
                best_score = score
                best_collection = name
    
    print(f"\n🏆 Best source collection: {best_collection} (score: {best_score:.2f})")
    
    if not best_collection:
        print("❌ No suitable source collection found!")
        return False
    
    print(f"\n2️⃣ Backing Up Current alexandria_books")
    print("-" * 40)
    
    # Backup current alexandria_books if it exists
    backup_name = f"alexandria_books_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        if DEFAULT_COLLECTION_NAME in [c.name for c in collections]:
            current_collection = client.get_collection(DEFAULT_COLLECTION_NAME)
            current_count = current_collection.count()
            
            if current_count > 0:
                print(f"📦 Backing up {current_count} documents to {backup_name}")
                
                # Get all data from current collection
                all_data = current_collection.get(include=['documents', 'metadatas', 'embeddings'])
                
                # Create backup collection
                backup_collection = client.create_collection(backup_name)
                backup_collection.add(
                    documents=all_data['documents'],
                    metadatas=all_data['metadatas'],
                    embeddings=all_data['embeddings'],
                    ids=all_data['ids']
                )
                print(f"✅ Backup created: {backup_name}")
            
            # Delete current collection
            client.delete_collection(DEFAULT_COLLECTION_NAME)
            print(f"🗑️  Deleted old {DEFAULT_COLLECTION_NAME}")
            
    except Exception as e:
        print(f"⚠️  Backup failed (collection might not exist): {e}")
    
    print(f"\n3️⃣ Migrating Data to Enhanced Format")
    print("-" * 40)
    
    try:
        # Get source collection
        source_collection = client.get_collection(best_collection)
        
        # Get all data from source
        print(f"📚 Reading all data from {best_collection}...")
        all_source_data = source_collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        total_docs = len(all_source_data['documents'])
        print(f"📊 Found {total_docs} documents to migrate")
        
        # Create new alexandria_books collection
        enhanced_db = await get_enhanced_database()
        success = await enhanced_db.create_collection(
            DEFAULT_COLLECTION_NAME,
            {
                "description": "Enhanced Alexandria books collection",
                "version": "1.3.0",
                "migrated_from": best_collection,
                "migrated_at": datetime.now().isoformat()
            }
        )
        
        if not success:
            print("❌ Failed to create enhanced collection")
            return False
        
        print(f"✅ Created enhanced collection: {DEFAULT_COLLECTION_NAME}")
        
        # Convert and migrate data in batches
        batch_size = 50
        migrated_count = 0
        
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = all_source_data['documents'][i:batch_end]
            batch_metas = all_source_data['metadatas'][i:batch_end]
            batch_embeddings = all_source_data['embeddings'][i:batch_end]
            batch_ids = all_source_data['ids'][i:batch_end]
            
            print(f"🔄 Processing batch {i//batch_size + 1}: docs {i+1}-{batch_end}")
            
            # Convert metadata to enhanced format
            enhanced_metadata_list = []
            for j, (doc, meta, embedding, doc_id) in enumerate(zip(batch_docs, batch_metas, batch_embeddings, batch_ids)):
                # Create enhanced metadata
                enhanced_metadata = EmbeddingMetadata(
                    embedding_id=str(uuid.uuid4()),
                    content_id=meta.get('book_id', str(uuid.uuid4())),
                    chunk_index=meta.get('chunk_index', j),
                    module_type=ModuleType.LIBRARY,
                    content_type=ContentType.BOOK,
                    chunk_type=meta.get('content_type', 'paragraph'),
                    visibility=ContentVisibility.PUBLIC,
                    creator_id="migration_user",
                    organization_id=None,
                    semantic_tags=self._extract_semantic_tags_from_metadata(meta),
                    language=meta.get('language', 'en'),
                    reading_level=meta.get('reading_level', 'intermediate'),
                    source_location=self._extract_source_location(meta),
                    text_content=doc,
                    chunk_length=len(doc),
                    embedding_model="text-embedding-ada-002",
                    embedding_dimension=len(embedding),
                    importance_score=meta.get('importance_score', 0.5),
                    quality_score=meta.get('coherence_score', 0.5)
                )
                enhanced_metadata_list.append(enhanced_metadata)
            
            # Add to enhanced collection
            success = await enhanced_db.add_documents_with_metadata(
                collection_name=DEFAULT_COLLECTION_NAME,
                documents=batch_docs,
                embeddings=batch_embeddings,
                embedding_metadata=enhanced_metadata_list,
                ids=batch_ids
            )
            
            if success:
                migrated_count += len(batch_docs)
                print(f"  ✅ Migrated {len(batch_docs)} documents")
            else:
                print(f"  ❌ Failed to migrate batch")
        
        print(f"\n📊 Migration Summary:")
        print(f"  📚 Total documents migrated: {migrated_count}/{total_docs}")
        print(f"  🎯 Success rate: {(migrated_count/total_docs)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger.error(f"Migration error: {e}")
        return False
    
    print(f"\n4️⃣ Testing Rebuilt Collection")
    print("-" * 40)
    
    try:
        # Test enhanced search
        test_queries = [
            "psychology human behavior",
            "games people play",
            "social interaction"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            
            results = await enhanced_db.query_with_permissions(
                collection_name=DEFAULT_COLLECTION_NAME,
                query_text=query,
                n_results=5
            )
            
            documents = results.get('documents', [])
            distances = results.get('distances', [])
            
            if documents and distances:
                avg_similarity = 1.0 - (sum(distances) / len(distances))
                print(f"  📄 Results: {len(documents)}")
                print(f"  🎯 Avg similarity: {avg_similarity:.4f}")
                print(f"  📝 Sample: {documents[0][:100]}...")
            else:
                print(f"  ⚠️  No results found")
        
        print(f"\n✅ Collection rebuild complete!")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def _extract_semantic_tags_from_metadata(self, meta: dict) -> list:
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
    
    # Clean and deduplicate
    tags = [tag.strip().lower() for tag in tags if tag and isinstance(tag, str)]
    return list(set(tags))

def _extract_source_location(self, meta: dict) -> dict:
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
    # Add missing method to global scope for the script
    def _extract_semantic_tags_from_metadata(meta: dict) -> list:
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
        
        # Clean and deduplicate
        tags = [tag.strip().lower() for tag in tags if tag and isinstance(tag, str)]
        return list(set(tags))

    def _extract_source_location(meta: dict) -> dict:
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
    
    # Monkey patch for the script
    rebuild_alexandria_collection._extract_semantic_tags_from_metadata = _extract_semantic_tags_from_metadata
    rebuild_alexandria_collection._extract_source_location = _extract_source_location
    
    asyncio.run(rebuild_alexandria_collection())