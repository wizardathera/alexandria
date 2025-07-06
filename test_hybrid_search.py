#!/usr/bin/env python3
"""
Test hybrid search functionality including BM25 and RRF fusion.
"""

import sys
import asyncio
import time
sys.path.insert(0, 'src')

from src.utils.logger import get_logger
from src.utils.hybrid_search import get_hybrid_search_engine, HybridSearchConfig, SearchStrategy

logger = get_logger(__name__)

async def test_hybrid_search():
    """Test hybrid search functionality."""
    
    print("🔀 Testing Hybrid Search Functionality")
    print("=" * 50)
    
    try:
        # Get hybrid search engine
        search_engine = await get_hybrid_search_engine()
        print("✅ Hybrid search engine initialized")
        
        test_query = "psychology games people play"
        
        # Test different search strategies
        strategies_to_test = [
            ("vector", SearchStrategy.VECTOR_ONLY),
            ("bm25", SearchStrategy.BM25_ONLY),
            ("vector_bm25", SearchStrategy.VECTOR_BM25),
            ("auto", SearchStrategy.AUTO)
        ]
        
        print(f"\n🔍 Testing query: '{test_query}'")
        print("-" * 40)
        
        for strategy_name, strategy_enum in strategies_to_test:
            print(f"\n📊 Strategy: {strategy_name}")
            
            try:
                config = HybridSearchConfig(
                    strategy=strategy_enum,
                    fusion_method="rrf",
                    max_results=5,
                    max_results_per_strategy=10
                )
                
                start_time = time.time()
                results = await search_engine.search(
                    query=test_query,
                    config=config,
                    book_id=None
                )
                search_time = time.time() - start_time
                
                print(f"  ⏱️  Search time: {search_time:.3f}s")
                print(f"  📄 Results: {len(results.results)}")
                print(f"  🎯 Strategy used: {results.strategy_used.value}")
                print(f"  🔀 Fusion method: {results.fusion_method}")
                
                if results.results:
                    top_result = results.results[0]
                    print(f"  🏆 Top score: {top_result.final_score:.4f}")
                    print(f"  📝 Preview: {top_result.content[:100]}...")
                    
                    # Show contributing strategies
                    if hasattr(top_result, 'contributing_strategies'):
                        print(f"  🔗 Contributing: {top_result.contributing_strategies}")
                
                # Show timing breakdown
                print(f"  ⏱️  Vector time: {results.vector_time:.3f}s")
                print(f"  ⏱️  BM25 time: {results.bm25_time:.3f}s")
                print(f"  ⏱️  Fusion time: {results.fusion_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ Strategy {strategy_name} failed: {e}")
        
        # Test search suggestions
        print(f"\n🔍 Testing Search Suggestions")
        print("-" * 30)
        
        try:
            suggestions = await search_engine.get_search_suggestions("psyc", 5)
            print(f"Suggestions for 'psyc': {suggestions}")
        except Exception as e:
            print(f"❌ Search suggestions failed: {e}")
        
        # Test engine stats
        print(f"\n📊 Testing Engine Statistics")
        print("-" * 30)
        
        try:
            stats = search_engine.get_engine_stats()
            print(f"Engine stats: {stats}")
        except Exception as e:
            print(f"❌ Engine stats failed: {e}")
            
        print(f"\n✅ Hybrid search testing complete!")
        
    except Exception as e:
        print(f"❌ Hybrid search engine initialization failed: {e}")
        logger.error(f"Hybrid search test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_hybrid_search())