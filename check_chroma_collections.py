#!/usr/bin/env python3
"""
Check what collections exist in the Chroma database.
"""

import sys
sys.path.insert(0, 'src')

import chromadb
from pathlib import Path
from src.utils.config import get_settings

def check_chroma_collections():
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_directory)
    
    print(f"Checking Chroma database at: {persist_dir}")
    
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        
        # List all collections
        collections = client.list_collections()
        
        print(f"\nFound {len(collections)} collections:")
        for collection in collections:
            print(f"  - {collection.name}")
            print(f"    Metadata: {collection.metadata}")
            
            # Get count of documents in each collection
            try:
                count = collection.count()
                print(f"    Documents: {count}")
                
                # Sample some data if exists
                if count > 0:
                    sample = collection.peek(limit=3)
                    print(f"    Sample IDs: {sample['ids'][:3] if sample['ids'] else 'None'}")
                    if sample['metadatas']:
                        sample_meta = sample['metadatas'][0]
                        print(f"    Sample metadata keys: {list(sample_meta.keys())}")
                        if 'content_id' in sample_meta:
                            print(f"    Sample content_id: {sample_meta['content_id']}")
                        if 'content_type' in sample_meta:
                            print(f"    Sample content_type: {sample_meta['content_type']}")
                    print()
                    
            except Exception as e:
                print(f"    Error getting count: {e}")
        
        return collections
        
    except Exception as e:
        print(f"Error accessing Chroma database: {e}")
        return []

if __name__ == "__main__":
    check_chroma_collections()