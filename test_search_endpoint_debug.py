#!/usr/bin/env python3
"""
Test script to debug and verify search endpoint functionality.

This script tests:
1. Embedding generation and persistence
2. Vector similarity search
3. Enhanced search endpoints
4. Performance benchmarks
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
import sys
sys.path.insert(0, 'src')

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.enhanced_database import get_enhanced_database
from src.services.enhanced_embedding_service import get_enhanced_embedding_service
from src.services.content_service import get_content_service
from src.models import User, UserRole, ModuleType, ContentType
from src.utils.embeddings import EmbeddingService

logger = get_logger(__name__)

class SearchEndpointDebugger:
    """Debug and test search endpoint functionality."""
    
    def __init__(self):
        self.settings = get_settings()
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run all search endpoint tests."""
        print("🔍 Starting Search Endpoint Debug Tests")
        print("=" * 60)
        
        # Test 1: Check database connectivity
        await self.test_database_connectivity()
        
        # Test 2: Check existing embeddings
        await self.test_existing_embeddings()
        
        # Test 3: Test basic vector search
        await self.test_basic_vector_search()
        
        # Test 4: Test enhanced search service
        await self.test_enhanced_search_service()
        
        # Test 5: Test API endpoint equivalents
        await self.test_search_api_logic()
        
        # Test 6: Performance benchmarks
        await self.test_performance_benchmarks()
        
        # Test 7: Metadata verification
        await self.test_metadata_completeness()
        
        # Print final summary
        self.print_test_summary()
        
    async def test_database_connectivity(self):
        """Test 1: Database connectivity and initialization."""
        print("\n1️⃣ Testing Database Connectivity")
        print("-" * 40)
        
        try:
            # Test enhanced database
            enhanced_db = await get_enhanced_database()
            print(f"✅ Enhanced database initialized: {type(enhanced_db).__name__}")
            
            # Test embedding service
            embedding_service = EmbeddingService(use_cache=True)
            print(f"✅ Embedding service initialized: {embedding_service.default_model}")
            
            self.test_results['database_connectivity'] = True
            
        except Exception as e:
            print(f"❌ Database connectivity failed: {e}")
            self.test_results['database_connectivity'] = False
            logger.error(f"Database connectivity test failed: {e}")
    
    async def test_existing_embeddings(self):
        """Test 2: Check existing embeddings in the database."""
        print("\n2️⃣ Testing Existing Embeddings")
        print("-" * 40)
        
        try:
            enhanced_db = await get_enhanced_database()
            
            # Get content service to list content
            content_service = await get_content_service()
            default_user = User(
                user_id="test_user",
                email="test@example.com", 
                username="test_user",
                role=UserRole.ADMIN
            )
            
            content_items = await content_service.list_content_items(
                user=default_user,
                limit=10
            )
            
            print(f"📚 Found {len(content_items)} content items")
            
            embeddings_found = 0
            for content in content_items[:3]:  # Check first 3
                embeddings = await enhanced_db.get_content_embeddings(content.content_id)
                print(f"  📄 {content.title}: {len(embeddings)} embeddings")
                embeddings_found += len(embeddings)
                
                if embeddings:
                    sample_embedding = embeddings[0]
                    print(f"    🔸 Sample metadata: {sample_embedding.content_type}, {sample_embedding.module_type}")
                    print(f"    🔸 Text preview: {sample_embedding.text_content[:100]}...")
            
            print(f"\n📊 Total embeddings found: {embeddings_found}")
            self.test_results['existing_embeddings'] = embeddings_found > 0
            self.test_results['embeddings_count'] = embeddings_found
            
        except Exception as e:
            print(f"❌ Existing embeddings test failed: {e}")
            self.test_results['existing_embeddings'] = False
            logger.error(f"Existing embeddings test failed: {e}")
    
    async def test_basic_vector_search(self):
        """Test 3: Basic vector similarity search."""
        print("\n3️⃣ Testing Basic Vector Search")
        print("-" * 40)
        
        test_queries = [
            "What is psychology?",
            "Human behavior and social interactions",
            "Games people play",
            "Reading and learning"
        ]
        
        try:
            enhanced_db = await get_enhanced_database()
            
            for query in test_queries:
                print(f"\n🔎 Query: '{query}'")
                
                start_time = time.time()
                results = await enhanced_db.query_with_permissions(
                    collection_name="alexandria_books",
                    query_text=query,
                    n_results=5
                )
                search_time = time.time() - start_time
                
                result_count = len(results.get('documents', []))
                print(f"  ⏱️  Search time: {search_time:.3f}s")
                print(f"  📄 Results: {result_count}")
                
                if result_count > 0:
                    # Show top result details
                    top_doc = results['documents'][0]
                    top_metadata = results['metadatas'][0] if results.get('metadatas') else {}
                    top_distance = results['distances'][0] if results.get('distances') else 1.0
                    similarity_score = 1.0 - top_distance
                    
                    print(f"  🏆 Top result similarity: {similarity_score:.3f}")
                    print(f"  📝 Content preview: {top_doc[:150]}...")
                    print(f"  🏷️  Metadata: {top_metadata.get('content_type', 'unknown')} from {top_metadata.get('module_type', 'unknown')}")
                else:
                    print("  ⚠️  No results found")
                
                # Store performance data
                if query not in self.test_results:
                    self.test_results[query] = {}
                self.test_results[query]['search_time'] = search_time
                self.test_results[query]['result_count'] = result_count
                
            self.test_results['basic_vector_search'] = True
            
        except Exception as e:
            print(f"❌ Basic vector search failed: {e}")
            self.test_results['basic_vector_search'] = False
            logger.error(f"Basic vector search test failed: {e}")
    
    async def test_enhanced_search_service(self):
        """Test 4: Enhanced search service functionality."""
        print("\n4️⃣ Testing Enhanced Search Service")
        print("-" * 40)
        
        try:
            enhanced_embedding_service = await get_enhanced_embedding_service()
            default_user = User(
                user_id="test_user",
                email="test@example.com",
                username="test_user", 
                role=UserRole.READER
            )
            
            test_query = "psychology and human behavior"
            print(f"🔎 Enhanced search query: '{test_query}'")
            
            start_time = time.time()
            search_results = await enhanced_embedding_service.enhanced_search(
                query=test_query,
                user=default_user,
                n_results=5,
                include_relationships=True
            )
            search_time = time.time() - start_time
            
            print(f"⏱️  Enhanced search time: {search_time:.3f}s")
            
            # Check result structure
            print(f"📊 Search result keys: {list(search_results.keys())}")
            
            documents = search_results.get('documents', [])
            enhanced_results = search_results.get('enhanced_results', [])
            
            print(f"📄 Base documents: {len(documents)}")
            print(f"🔍 Enhanced results: {len(enhanced_results)}")
            
            if enhanced_results:
                print("\n🎯 Enhanced Results Sample:")
                for i, result in enumerate(enhanced_results[:2]):
                    print(f"  {i+1}. Title: {result.get('title', 'Unknown')}")
                    print(f"     Content Type: {result.get('content_type', 'Unknown')}")
                    print(f"     Module: {result.get('module_type', 'Unknown')}")
                    print(f"     Similarity: {result.get('similarity_score', 0):.3f}")
                    print(f"     Relationship: {result.get('relationship_score', 0):.3f}")
                    print(f"     Tags: {result.get('semantic_tags', [])}")
            
            self.test_results['enhanced_search'] = True
            self.test_results['enhanced_search_time'] = search_time
            self.test_results['enhanced_results_count'] = len(enhanced_results)
            
        except Exception as e:
            print(f"❌ Enhanced search service failed: {e}")
            self.test_results['enhanced_search'] = False
            logger.error(f"Enhanced search service test failed: {e}")
    
    async def test_search_api_logic(self):
        """Test 5: API endpoint equivalent functionality."""
        print("\n5️⃣ Testing Search API Logic")
        print("-" * 40)
        
        try:
            # Test enhanced search service
            enhanced_embedding_service = await get_enhanced_embedding_service()
            default_user = User(
                user_id="test_user",
                email="test@example.com",
                username="test_user", 
                role=UserRole.READER
            )
            
            test_query = "psychology and human behavior"
            print(f"🔎 Enhanced search query: '{test_query}'")
            
            start_time = time.time()
            search_results = await enhanced_embedding_service.enhanced_search(
                query=test_query,
                user=default_user,
                n_results=5,
                include_relationships=True
            )
            search_time = time.time() - start_time
            
            print(f"⏱️  Enhanced search time: {search_time:.3f}s")
            
            # Check result structure
            print(f"📊 Search result keys: {list(search_results.keys())}")
            
            documents = search_results.get('documents', [])
            enhanced_results = search_results.get('enhanced_results', [])
            
            print(f"📄 Base documents: {len(documents)}")
            print(f"🔍 Enhanced results: {len(enhanced_results)}")
            
            if enhanced_results:
                print("\n🎯 Enhanced Results Sample:")
                for i, result in enumerate(enhanced_results[:2]):
                    print(f"  {i+1}. Title: {result.get('title', 'Unknown')}")
                    print(f"     Content Type: {result.get('content_type', 'Unknown')}")
                    print(f"     Module: {result.get('module_type', 'Unknown')}")
                    print(f"     Similarity: {result.get('similarity_score', 0):.3f}")
                    print(f"     Relationship: {result.get('relationship_score', 0):.3f}")
                    print(f"     Tags: {result.get('semantic_tags', [])}")
            
            # Test chat/query endpoint logic
            print("🔗 Testing /api/chat/query logic...")
            
            from src.rag.rag_service import get_rag_service
            
            rag_service = await get_rag_service()
            
            test_question = "What are some psychological games people play?"
            print(f"❓ Question: '{test_question}'")
            
            start_time = time.time()
            rag_response = await rag_service.query(
                question=test_question,
                book_id=None,
                context_limit=5
            )
            query_time = time.time() - start_time
            
            print(f"⏱️  RAG query time: {query_time:.3f}s")
            print(f"✅ Answer preview: {rag_response.answer[:200]}...")
            print(f"📚 Sources: {len(rag_response.sources)}")
            print(f"🎯 Confidence: {rag_response.confidence_score:.3f}")
            
            # Test enhanced content API logic
            print("\n🔗 Testing /api/enhanced/search logic...")
            
            enhanced_embedding_service = await get_enhanced_embedding_service()
            default_user = User(
                user_id="test_user",
                email="test@example.com",
                username="test_user",
                role=UserRole.READER
            )
            
            start_time = time.time()
            enhanced_search_results = await enhanced_embedding_service.enhanced_search(
                query=test_question,
                user=default_user,
                module_filter=None,
                content_type_filter=None,
                n_results=10,
                include_relationships=True
            )
            enhanced_time = time.time() - start_time
            
            print(f"⏱️  Enhanced API time: {enhanced_time:.3f}s")
            print(f"📄 Enhanced results: {len(enhanced_search_results.get('enhanced_results', []))}")
            
            self.test_results['api_logic'] = True
            self.test_results['rag_query_time'] = query_time
            self.test_results['enhanced_api_time'] = enhanced_time
            
        except Exception as e:
            print(f"❌ API logic test failed: {e}")
            self.test_results['api_logic'] = False
            logger.error(f"API logic test failed: {e}")
    
    async def test_performance_benchmarks(self):
        """Test 6: Performance benchmarks."""
        print("\n6️⃣ Testing Performance Benchmarks")
        print("-" * 40)
        
        try:
            enhanced_db = await get_enhanced_database()
            
            # Test various query complexities
            queries = [
                "simple",
                "psychology behavior human social interaction",
                "What are the main psychological principles that govern human behavior in social settings and how do they affect interpersonal relationships?",
            ]
            
            benchmark_results = []
            
            for i, query in enumerate(queries):
                print(f"\n🏃 Benchmark {i+1}: {len(query)} characters")
                
                times = []
                for run in range(3):  # 3 runs per query
                    start_time = time.time()
                    results = await enhanced_db.query_with_permissions(
                        collection_name="alexandria_books",
                        query_text=query,
                        n_results=10
                    )
                    search_time = time.time() - start_time
                    times.append(search_time)
                    result_count = len(results.get('documents', []))
                
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                print(f"  ⏱️  Average: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")
                print(f"  📄 Results: {result_count}")
                print(f"  🎯 Target: <3s - {'✅ PASS' if avg_time < 3.0 else '❌ FAIL'}")
                
                benchmark_results.append({
                    'query_length': len(query),
                    'avg_time': avg_time,
                    'result_count': result_count,
                    'passes_target': avg_time < 3.0
                })
            
            # Overall performance assessment
            all_pass = all(b['passes_target'] for b in benchmark_results)
            avg_performance = sum(b['avg_time'] for b in benchmark_results) / len(benchmark_results)
            
            print(f"\n📊 Overall Performance:")
            print(f"  🎯 All queries under 3s: {'✅ YES' if all_pass else '❌ NO'}")
            print(f"  ⏱️  Average response time: {avg_performance:.3f}s")
            
            self.test_results['performance_benchmarks'] = all_pass
            self.test_results['avg_performance'] = avg_performance
            self.test_results['benchmark_details'] = benchmark_results
            
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            self.test_results['performance_benchmarks'] = False
            logger.error(f"Performance benchmark test failed: {e}")
    
    async def test_metadata_completeness(self):
        """Test 7: Metadata completeness and structure."""
        print("\n7️⃣ Testing Metadata Completeness")
        print("-" * 40)
        
        try:
            enhanced_db = await get_enhanced_database()
            
            # Get sample search results
            results = await enhanced_db.query_with_permissions(
                collection_name="alexandria_books",
                query_text="psychology",
                n_results=3
            )
            
            if not results.get('metadatas'):
                print("⚠️  No metadata found in results")
                self.test_results['metadata_completeness'] = False
                return
            
            print("🔍 Checking metadata structure...")
            
            required_fields = [
                'content_id', 'content_type', 'module_type', 'chunk_type',
                'visibility', 'semantic_tags', 'source_location', 'embedding_model'
            ]
            
            optional_fields = [
                'creator_id', 'organization_id', 'language', 'reading_level',
                'importance_score', 'quality_score', 'created_at'
            ]
            
            metadata_sample = results['metadatas'][0]
            print(f"📋 Sample metadata keys: {list(metadata_sample.keys())}")
            
            # Check required fields
            missing_required = []
            for field in required_fields:
                if field not in metadata_sample:
                    missing_required.append(field)
                else:
                    print(f"  ✅ {field}: {metadata_sample[field]}")
            
            # Check optional fields
            present_optional = []
            for field in optional_fields:
                if field in metadata_sample:
                    present_optional.append(field)
                    print(f"  🔸 {field}: {metadata_sample[field]}")
            
            print(f"\n📊 Metadata Assessment:")
            print(f"  ✅ Required fields present: {len(required_fields) - len(missing_required)}/{len(required_fields)}")
            print(f"  🔸 Optional fields present: {len(present_optional)}/{len(optional_fields)}")
            
            if missing_required:
                print(f"  ❌ Missing required: {missing_required}")
            
            # Test semantic tags parsing
            semantic_tags = metadata_sample.get('semantic_tags', '[]')
            try:
                if isinstance(semantic_tags, str):
                    parsed_tags = json.loads(semantic_tags)
                else:
                    parsed_tags = semantic_tags
                print(f"  🏷️  Semantic tags: {parsed_tags}")
            except json.JSONDecodeError:
                print(f"  ⚠️  Invalid semantic tags format: {semantic_tags}")
            
            # Test source location parsing
            source_location = metadata_sample.get('source_location', '{}')
            try:
                if isinstance(source_location, str):
                    parsed_location = json.loads(source_location)
                else:
                    parsed_location = source_location
                print(f"  📍 Source location: {parsed_location}")
            except json.JSONDecodeError:
                print(f"  ⚠️  Invalid source location format: {source_location}")
            
            self.test_results['metadata_completeness'] = len(missing_required) == 0
            self.test_results['metadata_coverage'] = {
                'required_present': len(required_fields) - len(missing_required),
                'required_total': len(required_fields),
                'optional_present': len(present_optional),
                'optional_total': len(optional_fields)
            }
            
        except Exception as e:
            print(f"❌ Metadata completeness test failed: {e}")
            self.test_results['metadata_completeness'] = False
            logger.error(f"Metadata completeness test failed: {e}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🏁 SEARCH ENDPOINT DEBUG SUMMARY")
        print("=" * 60)
        
        # Overall status
        critical_tests = [
            'database_connectivity', 'existing_embeddings', 
            'basic_vector_search', 'enhanced_search'
        ]
        
        critical_passed = sum(1 for test in critical_tests if self.test_results.get(test, False))
        critical_total = len(critical_tests)
        
        print(f"\n📊 Critical Tests: {critical_passed}/{critical_total} passed")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        
        test_labels = {
            'database_connectivity': '🔗 Database Connectivity',
            'existing_embeddings': '📚 Existing Embeddings',
            'basic_vector_search': '🔍 Basic Vector Search', 
            'enhanced_search': '⚡ Enhanced Search',
            'api_logic': '🔗 API Logic',
            'performance_benchmarks': '⏱️  Performance',
            'metadata_completeness': '📋 Metadata'
        }
        
        for test_key, label in test_labels.items():
            status = self.test_results.get(test_key, False)
            print(f"  {label}: {'✅ PASS' if status else '❌ FAIL'}")
        
        # Performance summary
        if 'avg_performance' in self.test_results:
            avg_perf = self.test_results['avg_performance']
            print(f"\n⏱️  Performance Summary:")
            print(f"  Average search time: {avg_perf:.3f}s")
            print(f"  Target (<3s): {'✅ MET' if avg_perf < 3.0 else '❌ MISSED'}")
        
        # Embedding summary
        if 'embeddings_count' in self.test_results:
            print(f"\n📚 Embedding Summary:")
            print(f"  Total embeddings found: {self.test_results['embeddings_count']}")
        
        # Metadata summary
        if 'metadata_coverage' in self.test_results:
            coverage = self.test_results['metadata_coverage']
            print(f"\n📋 Metadata Summary:")
            print(f"  Required fields: {coverage['required_present']}/{coverage['required_total']}")
            print(f"  Optional fields: {coverage['optional_present']}/{coverage['optional_total']}")
        
        # Final verdict
        overall_health = critical_passed >= 3  # At least 3/4 critical tests pass
        print(f"\n🎯 Overall Search Endpoint Health: {'✅ HEALTHY' if overall_health else '❌ NEEDS ATTENTION'}")
        
        # Save results to file
        results_file = Path("search_debug_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")

async def main():
    """Main test runner."""
    debugger = SearchEndpointDebugger()
    await debugger.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())