#!/usr/bin/env python3
"""
Diagnostic script for Alexandria app's Chroma database.

This script inspects the Chroma database to help debug enhanced search issues.
It will show collection details, document counts, sample metadata, and embedding information.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

# Add src directory to path to import Alexandria modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.utils.config import get_settings
    from src.utils.logger import get_logger
    settings = get_settings()
    logger = get_logger(__name__)
except ImportError as e:
    print(f"Warning: Could not import Alexandria modules: {e}")
    print("Using default settings...")
    
    class MockSettings:
        def __init__(self):
            self.chroma_persist_directory = "./data/chroma_db"
    
    settings = MockSettings()
    logger = None


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def analyze_embedding_vectors(embeddings: List[List[float]]) -> Dict[str, Any]:
    """Analyze embedding vectors for consistency and quality."""
    if not embeddings:
        return {"error": "No embeddings provided"}
    
    try:
        # Convert to numpy array for analysis
        embed_array = np.array(embeddings)
        
        return {
            "count": len(embeddings),
            "dimensions": embed_array.shape[1] if len(embed_array.shape) > 1 else 0,
            "mean_norm": float(np.mean(np.linalg.norm(embed_array, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embed_array, axis=1))),
            "min_value": float(np.min(embed_array)),
            "max_value": float(np.max(embed_array)),
            "mean_value": float(np.mean(embed_array)),
            "non_zero_ratio": float(np.mean(embed_array != 0))
        }
    except Exception as e:
        return {"error": f"Failed to analyze embeddings: {e}"}


def diagnose_chroma_database():
    """Main diagnostic function."""
    print_header("Alexandria App - Chroma Database Diagnostics")
    
    try:
        # Initialize Chroma client
        persist_dir = Path(settings.chroma_persist_directory)
        print(f"Database path: {persist_dir}")
        print(f"Database exists: {persist_dir.exists()}")
        
        if not persist_dir.exists():
            print("❌ Error: Database directory does not exist!")
            return False
        
        # Connect to Chroma
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=False,
                is_persistent=True
            )
        )
        
        print("✅ Successfully connected to Chroma database")
        
        # List all collections
        print_section("Collections Overview")
        collections = client.list_collections()
        
        if not collections:
            print("❌ No collections found in database!")
            return False
        
        print(f"Found {len(collections)} collection(s):")
        for collection in collections:
            print(f"  - {collection.name}")
            if hasattr(collection, 'metadata') and collection.metadata:
                print(f"    Metadata: {json.dumps(collection.metadata, indent=2)}")
        
        # Analyze each collection
        for collection in collections:
            print_section(f"Collection: {collection.name}")
            
            try:
                # Get collection count
                count = collection.count()
                print(f"Document count: {count}")
                
                if count == 0:
                    print("❌ Collection is empty!")
                    continue
                
                # Get sample documents with all data
                sample_size = min(5, count)
                print(f"\nRetrieving {sample_size} sample documents...")
                
                results = collection.get(
                    limit=sample_size,
                    include=["documents", "metadatas", "embeddings"]
                )
                
                print(f"Retrieved {len(results['ids'])} documents")
                
                # Analyze sample metadata
                print_section("Sample Metadata Analysis")
                if results['metadatas']:
                    sample_metadata = results['metadatas'][0]
                    print("First document metadata:")
                    for key, value in sample_metadata.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:97]}...")
                        else:
                            print(f"  {key}: {value}")
                    
                    # Check for key fields
                    print("\nKey field analysis:")
                    key_fields = [
                        'content_id', 'module_type', 'content_type', 'visibility',
                        'creator_id', 'embedding_model', 'embedding_dimension'
                    ]
                    
                    for field in key_fields:
                        values = [m.get(field, 'MISSING') for m in results['metadatas']]
                        unique_values = set(values)
                        print(f"  {field}: {len(unique_values)} unique values - {list(unique_values)}")
                
                # Analyze embeddings
                print_section("Embedding Analysis")
                if results['embeddings']:
                    embedding_stats = analyze_embedding_vectors(results['embeddings'])
                    print("Embedding statistics:")
                    for key, value in embedding_stats.items():
                        print(f"  {key}: {value}")
                    
                    # Check for embedding model consistency
                    if results['metadatas']:
                        models = [m.get('embedding_model', 'UNKNOWN') for m in results['metadatas']]
                        unique_models = set(models)
                        print(f"\nEmbedding models used: {unique_models}")
                        
                        if len(unique_models) > 1:
                            print("⚠️  Warning: Multiple embedding models detected!")
                            for model in unique_models:
                                count = models.count(model)
                                print(f"    {model}: {count} documents")
                
                # Test similarity search
                print_section("Test Similarity Search")
                if results['documents'] and results['documents'][0]:
                    test_query = "Alexandria"
                    print(f"Testing query: '{test_query}'")
                    
                    try:
                        # Test with different parameters
                        for n_results in [5, 10, 20]:
                            search_results = collection.query(
                                query_texts=[test_query],
                                n_results=n_results,
                                include=["documents", "metadatas", "distances"]
                            )
                            
                            actual_results = len(search_results['documents'][0]) if search_results['documents'] else 0
                            print(f"  n_results={n_results}: Got {actual_results} results")
                            
                            if actual_results > 0:
                                distances = search_results['distances'][0]
                                print(f"    Distance range: {min(distances):.4f} - {max(distances):.4f}")
                                print(f"    Mean distance: {np.mean(distances):.4f}")
                                
                                # Show top 3 results
                                print(f"    Top 3 results:")
                                for i in range(min(3, len(distances))):
                                    doc_preview = search_results['documents'][0][i][:100] + "..." if len(search_results['documents'][0][i]) > 100 else search_results['documents'][0][i]
                                    print(f"      {i+1}. Distance: {distances[i]:.4f} - {doc_preview}")
                        
                        # Test with where clause filtering
                        print(f"\nTesting with visibility filter...")
                        filtered_results = collection.query(
                            query_texts=[test_query],
                            n_results=10,
                            where={"visibility": "public"},
                            include=["documents", "metadatas", "distances"]
                        )
                        actual_filtered = len(filtered_results['documents'][0]) if filtered_results['documents'] else 0
                        print(f"  Filtered results (visibility=public): {actual_filtered}")
                        
                    except Exception as e:
                        print(f"❌ Error during similarity search: {e}")
                
                # Check for common issues
                print_section("Issue Detection")
                
                issues = []
                
                # Check for missing visibility
                if results['metadatas']:
                    visibility_values = [m.get('visibility', 'MISSING') for m in results['metadatas']]
                    if 'MISSING' in visibility_values:
                        issues.append("Some documents missing 'visibility' field")
                
                # Check for consistent embedding dimensions
                if results['embeddings']:
                    dimensions = [len(emb) for emb in results['embeddings']]
                    if len(set(dimensions)) > 1:
                        issues.append(f"Inconsistent embedding dimensions: {set(dimensions)}")
                
                # Check for zero embeddings
                if results['embeddings']:
                    zero_embeddings = [i for i, emb in enumerate(results['embeddings']) if all(v == 0 for v in emb)]
                    if zero_embeddings:
                        issues.append(f"Found {len(zero_embeddings)} zero embeddings")
                
                if issues:
                    print("⚠️  Issues detected:")
                    for issue in issues:
                        print(f"    - {issue}")
                else:
                    print("✅ No obvious issues detected")
                
            except Exception as e:
                print(f"❌ Error analyzing collection {collection.name}: {e}")
                continue
        
        print_section("Diagnostic Summary")
        print("✅ Diagnostic completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Fatal error during diagnostics: {e}")
        return False


if __name__ == "__main__":
    success = diagnose_chroma_database()
    sys.exit(0 if success else 1)