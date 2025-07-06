#!/usr/bin/env python3
"""
Test script for enhanced search functionality in Alexandria app.

This script tests the enhanced search after applying the debugging fixes.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.enhanced_embedding_service import get_enhanced_embedding_service
from src.models import User, UserRole, ModuleType, ContentType
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def test_enhanced_search():
    """Test the enhanced search functionality with debugging."""
    
    print("="*60)
    print("  Testing Enhanced Search Functionality")
    print("="*60)
    
    try:
        # Initialize enhanced embedding service
        print("\n1. Initializing enhanced embedding service...")
        embedding_service = await get_enhanced_embedding_service()
        print("✅ Enhanced embedding service initialized")
        
        # Create test user (anonymous for now)
        print("\n2. Setting up test user...")
        test_user = None  # Anonymous user for testing
        print("✅ Using anonymous user for testing")
        
        # Test queries
        test_queries = [
            "Alexandria",
            "library",
            "books",
            "reading",
            "knowledge"
        ]
        
        print("\n3. Testing search queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: '{query}' ---")
            
            try:
                # Perform enhanced search
                results = await embedding_service.enhanced_search(
                    query=query,
                    user=test_user,
                    n_results=10,
                    include_relationships=False  # Disable relationships for simpler debugging
                )
                
                # Print results
                doc_count = len(results.get('documents', []))
                print(f"✅ Search completed - Found {doc_count} documents")
                
                if doc_count > 0:
                    print(f"   Distance range: {min(results.get('distances', [])):.4f} - {max(results.get('distances', [])):.4f}")
                    
                    # Show first few results
                    for j in range(min(3, doc_count)):
                        doc_preview = results['documents'][j][:100] + "..." if len(results['documents'][j]) > 100 else results['documents'][j]
                        distance = results['distances'][j]
                        similarity = 1.0 - distance
                        print(f"   Result {j+1}: Similarity={similarity:.4f} - {doc_preview}")
                else:
                    print("   ⚠️  No documents found")
                
            except Exception as e:
                print(f"❌ Search failed: {e}")
                logger.error(f"Search failed for query '{query}': {e}")
        
        # Test with filters
        print(f"\n4. Testing search with filters...")
        
        try:
            results = await embedding_service.enhanced_search(
                query="Alexandria",
                user=test_user,
                module_filter=ModuleType.LIBRARY,
                content_type_filter=ContentType.BOOK,
                n_results=10,
                include_relationships=False
            )
            
            doc_count = len(results.get('documents', []))
            print(f"✅ Filtered search completed - Found {doc_count} documents")
            
        except Exception as e:
            print(f"❌ Filtered search failed: {e}")
            logger.error(f"Filtered search failed: {e}")
        
        # Test with relationships
        print(f"\n5. Testing search with relationships...")
        
        try:
            results = await embedding_service.enhanced_search(
                query="Alexandria",
                user=test_user,
                n_results=5,
                include_relationships=True
            )
            
            doc_count = len(results.get('documents', []))
            enhanced_count = len(results.get('enhanced_results', []))
            print(f"✅ Relationship search completed - Found {doc_count} documents, {enhanced_count} enhanced results")
            
        except Exception as e:
            print(f"❌ Relationship search failed: {e}")
            logger.error(f"Relationship search failed: {e}")
        
        print("\n" + "="*60)
        print("  Test Summary")
        print("="*60)
        print("✅ Enhanced search testing completed")
        print("📋 Check the logs above for detailed debugging information")
        print("🔍 If search returns 0 results, run the diagnostic script: python diagnose_chroma.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Fatal error during testing: {e}")
        logger.error(f"Fatal error during testing: {e}")
        return False


def print_usage():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("  How to Use This Test Script")
    print("="*60)
    print("\n1. Run the diagnostic script first:")
    print("   python diagnose_chroma.py")
    print("\n2. Run this test script:")
    print("   python test_enhanced_search.py")
    print("\n3. Check the logs for detailed debugging information")
    print("\n4. Expected behavior:")
    print("   - If the database has documents, search should return results")
    print("   - If search returns 0 results, check:")
    print("     a) Are embeddings created correctly?")
    print("     b) Are query embeddings using the same model?")
    print("     c) Are filters too restrictive?")
    print("     d) Are similarity thresholds too high?")
    print("\n5. Key debug logs to look for:")
    print("   - 'Enhanced search query: <query> with n_results=10'")
    print("   - 'Raw Chroma query returned X results'")
    print("   - 'Distance range: X.XXXX - X.XXXX'")
    print("   - 'Using embedding model: text-embedding-ada-002'")
    print("\n6. If no results are found:")
    print("   - Check visibility field in documents (should be 'public' for anonymous users)")
    print("   - Check if embeddings are zero vectors")
    print("   - Check if query and document embeddings use the same model")
    print("   - Try increasing n_results parameter")
    print("\n" + "="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    success = asyncio.run(test_enhanced_search())
    
    if not success:
        print_usage()
        sys.exit(1)
    else:
        print_usage()
        sys.exit(0)