"""
Hybrid Search Engine for DBC Platform.

This module integrates vector search, BM25 keyword search, and graph traversal
into a unified hybrid retrieval system with intelligent result fusion.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import get_logger
from src.utils.database import get_database
from src.utils.embeddings import EmbeddingService
from src.utils.bm25_search import BM25SearchEngine, BM25SearchResult
from src.utils.graph_retrieval import GraphSearchEngine, GraphSearchResult
from src.utils.result_fusion import (
    HybridResultFusion, SearchResult, StrategyResults, FusionResults
)

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Available search strategies."""
    VECTOR_ONLY = "vector"
    BM25_ONLY = "bm25"
    GRAPH_ONLY = "graph"
    VECTOR_BM25 = "vector_bm25"
    VECTOR_GRAPH = "vector_graph"
    BM25_GRAPH = "bm25_graph"
    ALL_STRATEGIES = "all"
    AUTO = "auto"


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    # Strategy selection
    strategy: SearchStrategy = SearchStrategy.AUTO
    fusion_method: str = "rrf"  # 'rrf', 'weighted', 'combsum'
    
    # Result limits
    max_results: int = 10
    max_results_per_strategy: int = 20
    
    # Vector search parameters
    vector_similarity_threshold: float = 0.0
    
    # BM25 parameters
    bm25_strategy: str = "exact"  # 'exact', 'fuzzy', 'ngram', 'synonym'
    bm25_min_score: float = 0.0
    
    # Graph search parameters
    graph_strategy: str = "bfs"  # 'bfs', 'random_walk'
    graph_max_distance: int = 3
    
    # Fusion parameters
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    rrf_k: int = 60
    
    # Performance settings
    timeout_seconds: float = 30.0
    parallel_execution: bool = True


@dataclass
class HybridSearchResult:
    """Result from hybrid search with rich metadata."""
    doc_id: str
    content: str
    final_score: float
    metadata: Dict[str, Any]
    
    # Strategy contributions
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    graph_score: Optional[float] = None
    
    # Fusion information
    contributing_strategies: List[str] = field(default_factory=list)
    fusion_explanation: Dict[str, Any] = field(default_factory=dict)
    original_ranks: Dict[str, int] = field(default_factory=dict)


@dataclass
class HybridSearchResults:
    """Complete results from hybrid search."""
    query: str
    results: List[HybridSearchResult]
    
    # Search metadata
    strategy_used: SearchStrategy
    fusion_method: str
    total_time: float
    
    # Strategy-specific timing
    vector_time: float = 0.0
    bm25_time: float = 0.0
    graph_time: float = 0.0
    fusion_time: float = 0.0
    
    # Strategy-specific counts
    vector_results: int = 0
    bm25_results: int = 0
    graph_results: int = 0
    
    # Performance metrics
    stats: Dict[str, Any] = field(default_factory=dict)


class HybridSearchEngine:
    """
    Unified hybrid search engine combining multiple retrieval strategies.
    
    Orchestrates vector search, BM25 keyword search, and graph traversal
    with intelligent result fusion to provide comprehensive search capabilities.
    """
    
    def __init__(self):
        """Initialize hybrid search engine."""
        self.vector_db = None
        self.embedding_service = EmbeddingService(use_cache=True)
        self.bm25_engine = BM25SearchEngine()
        self.graph_engine = GraphSearchEngine()
        self.fusion_engine = HybridResultFusion()
        
        # Search strategy preferences
        self.strategy_preferences = {
            'short_query': SearchStrategy.BM25_ONLY,          # < 3 words
            'medium_query': SearchStrategy.VECTOR_BM25,       # 3-10 words
            'long_query': SearchStrategy.ALL_STRATEGIES,      # > 10 words
            'conceptual_query': SearchStrategy.VECTOR_GRAPH,  # Contains abstract terms
            'factual_query': SearchStrategy.VECTOR_BM25       # Contains specific facts
        }
        
        logger.info("Hybrid search engine initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize hybrid search engine components.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize vector database
            self.vector_db = await get_database()
            
            logger.info("Hybrid search engine components initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize hybrid search engine: {e}")
            return False
    
    async def index_documents(
        self,
        documents: List[Tuple[str, str, Dict[str, Any]]]
    ):
        """
        Index documents in all search engines.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
        """
        if not documents:
            return
        
        logger.info(f"Indexing {len(documents)} documents in hybrid search engines...")
        
        # Index in parallel
        tasks = []
        
        # BM25 indexing
        tasks.append(self.bm25_engine.index_documents(documents))
        
        # Graph indexing
        tasks.append(self.graph_engine.build_graph_from_documents(documents))
        
        # Wait for all indexing to complete
        await asyncio.gather(*tasks)
        
        logger.info("Document indexing completed for all engines")
    
    async def search(
        self,
        query: str,
        config: HybridSearchConfig = None,
        book_id: Optional[str] = None
    ) -> HybridSearchResults:
        """
        Perform hybrid search across multiple strategies.
        
        Args:
            query: Search query
            config: Search configuration
            book_id: Optional book ID filter
            
        Returns:
            HybridSearchResults: Comprehensive search results
        """
        start_time = time.time()
        config = config or HybridSearchConfig()
        
        if not query or not query.strip():
            return HybridSearchResults(
                query=query,
                results=[],
                strategy_used=config.strategy,
                fusion_method=config.fusion_method,
                total_time=time.time() - start_time
            )
        
        logger.info(f"Performing hybrid search: '{query}' (strategy: {config.strategy.value})")
        
        # Auto-select strategy if needed
        if config.strategy == SearchStrategy.AUTO:
            config.strategy = self._select_search_strategy(query)
            logger.info(f"Auto-selected strategy: {config.strategy.value}")
        
        # Execute search strategies
        strategy_results = await self._execute_search_strategies(query, config, book_id)
        
        # Fuse results
        fusion_start = time.time()
        fusion_results = await self.fusion_engine.fuse_results(
            strategy_results=strategy_results,
            query=query,
            fusion_method=config.fusion_method,
            limit=config.max_results,
            k=config.rrf_k,
            weights=config.fusion_weights
        )
        fusion_time = time.time() - fusion_start
        
        # Convert to hybrid search results
        hybrid_results = self._convert_fusion_results(fusion_results, strategy_results)
        
        # Add timing information
        hybrid_results.fusion_time = fusion_time
        hybrid_results.total_time = time.time() - start_time
        
        # Add strategy-specific timing
        for strategy_result in strategy_results:
            if strategy_result.strategy_name == "vector":
                hybrid_results.vector_time = strategy_result.search_time
                hybrid_results.vector_results = len(strategy_result.results)
            elif strategy_result.strategy_name == "bm25":
                hybrid_results.bm25_time = strategy_result.search_time
                hybrid_results.bm25_results = len(strategy_result.results)
            elif strategy_result.strategy_name == "graph":
                hybrid_results.graph_time = strategy_result.search_time
                hybrid_results.graph_results = len(strategy_result.results)
        
        logger.info(f"Hybrid search completed: {len(hybrid_results.results)} results "
                   f"(total_time: {hybrid_results.total_time:.3f}s)")
        
        return hybrid_results
    
    def _select_search_strategy(self, query: str) -> SearchStrategy:
        """
        Automatically select search strategy based on query characteristics.
        
        Args:
            query: Search query
            
        Returns:
            SearchStrategy: Selected strategy
        """
        words = query.split()
        word_count = len(words)
        
        # Simple heuristics for strategy selection
        if word_count < 3:
            return SearchStrategy.BM25_ONLY
        elif word_count <= 10:
            return SearchStrategy.VECTOR_BM25
        else:
            return SearchStrategy.ALL_STRATEGIES
    
    async def _execute_search_strategies(
        self,
        query: str,
        config: HybridSearchConfig,
        book_id: Optional[str]
    ) -> List[StrategyResults]:
        """Execute the specified search strategies."""
        strategy_tasks = []
        
        # Determine which strategies to execute
        execute_vector = config.strategy in [
            SearchStrategy.VECTOR_ONLY, SearchStrategy.VECTOR_BM25,
            SearchStrategy.VECTOR_GRAPH, SearchStrategy.ALL_STRATEGIES
        ]
        
        execute_bm25 = config.strategy in [
            SearchStrategy.BM25_ONLY, SearchStrategy.VECTOR_BM25,
            SearchStrategy.BM25_GRAPH, SearchStrategy.ALL_STRATEGIES
        ]
        
        execute_graph = config.strategy in [
            SearchStrategy.GRAPH_ONLY, SearchStrategy.VECTOR_GRAPH,
            SearchStrategy.BM25_GRAPH, SearchStrategy.ALL_STRATEGIES
        ]
        
        # Execute strategies
        if config.parallel_execution:
            # Parallel execution
            if execute_vector:
                strategy_tasks.append(self._vector_search(query, config, book_id))
            if execute_bm25:
                strategy_tasks.append(self._bm25_search(query, config))
            if execute_graph:
                strategy_tasks.append(self._graph_search(query, config))
            
            # Wait for all strategies to complete
            if strategy_tasks:
                try:
                    strategy_results = await asyncio.wait_for(
                        asyncio.gather(*strategy_tasks),
                        timeout=config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Search timeout after {config.timeout_seconds}s")
                    strategy_results = []
            else:
                strategy_results = []
        else:
            # Sequential execution
            strategy_results = []
            
            if execute_vector:
                result = await self._vector_search(query, config, book_id)
                strategy_results.append(result)
            
            if execute_bm25:
                result = await self._bm25_search(query, config)
                strategy_results.append(result)
            
            if execute_graph:
                result = await self._graph_search(query, config)
                strategy_results.append(result)
        
        # Filter out None results
        strategy_results = [result for result in strategy_results if result is not None]
        
        return strategy_results
    
    async def _vector_search(
        self,
        query: str,
        config: HybridSearchConfig,
        book_id: Optional[str]
    ) -> Optional[StrategyResults]:
        """Perform vector similarity search."""
        if not self.vector_db:
            logger.warning("Vector database not initialized")
            return None
        
        try:
            start_time = time.time()
            
            # Build metadata filter
            where_clause = {}
            if book_id:
                where_clause["book_id"] = book_id
            
            # Perform vector search
            from src.utils.config import get_settings
            settings = get_settings()
            results = await self.vector_db.query(
                collection_name=settings.chroma_collection_name,
                query_text=query,
                n_results=config.max_results_per_strategy,
                where=where_clause if where_clause else None
            )
            
            search_time = time.time() - start_time
            
            # Convert to SearchResult objects
            search_results = []
            
            if results and "documents" in results:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"],
                    results.get("metadatas", []),
                    results.get("distances", [])
                )):
                    if distance is None or distance <= (1.0 - config.vector_similarity_threshold):
                        score = 1.0 - distance if distance is not None else 0.5
                        
                        search_result = SearchResult(
                            doc_id=metadata.get("doc_id", f"unknown_{i}") if metadata else f"unknown_{i}",
                            score=score,
                            content=doc,
                            metadata=metadata or {},
                            source_strategy="vector",
                            explanation={
                                "similarity_score": score,
                                "distance": distance,
                                "rank": i + 1
                            }
                        )
                        search_results.append(search_result)
            
            return StrategyResults(
                strategy_name="vector",
                results=search_results,
                search_time=search_time,
                total_docs_searched=results.get("total_docs", 0) if results else 0,
                query_info={"similarity_threshold": config.vector_similarity_threshold}
            )
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return None
    
    async def _bm25_search(
        self,
        query: str,
        config: HybridSearchConfig
    ) -> Optional[StrategyResults]:
        """Perform BM25 keyword search."""
        try:
            # Perform BM25 search with fallback strategies
            bm25_results = await self.bm25_engine.search_with_fallback(
                query=query,
                limit=config.max_results_per_strategy,
                strategies=[config.bm25_strategy],
                min_score=config.bm25_min_score
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            for bm25_result in bm25_results.results:
                search_result = SearchResult(
                    doc_id=bm25_result.doc_id,
                    score=bm25_result.score,
                    content=bm25_result.content,
                    metadata=bm25_result.metadata,
                    source_strategy="bm25",
                    explanation={
                        "bm25_score": bm25_result.score,
                        "matched_terms": bm25_result.matched_terms,
                        "strategy": bm25_results.strategy_used
                    }
                )
                search_results.append(search_result)
            
            return StrategyResults(
                strategy_name="bm25",
                results=search_results,
                search_time=bm25_results.search_time,
                total_docs_searched=bm25_results.total_docs_searched,
                query_info={
                    "strategy": bm25_results.strategy_used,
                    "query_tokens": bm25_results.query_tokens,
                    "min_score": config.bm25_min_score
                }
            )
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return None
    
    async def _graph_search(
        self,
        query: str,
        config: HybridSearchConfig
    ) -> Optional[StrategyResults]:
        """Perform graph traversal search."""
        try:
            # TODO: Implement better start node selection based on query
            # For now, use all nodes as potential starting points
            graph_results = await self.graph_engine.search(
                query=query,
                start_nodes=None,  # Will use all nodes
                strategy=config.graph_strategy,
                limit=config.max_results_per_strategy,
                max_distance=config.graph_max_distance
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            for graph_result in graph_results.results:
                search_result = SearchResult(
                    doc_id=graph_result.node_id,
                    score=graph_result.score,
                    content=graph_result.content,
                    metadata=graph_result.metadata,
                    source_strategy="graph",
                    explanation={
                        "graph_score": graph_result.score,
                        "distance": graph_result.distance,
                        "strategy": graph_results.search_strategy
                    }
                )
                search_results.append(search_result)
            
            return StrategyResults(
                strategy_name="graph",
                results=search_results,
                search_time=graph_results.search_time,
                total_docs_searched=graph_results.nodes_visited,
                query_info={
                    "strategy": graph_results.search_strategy,
                    "max_distance": graph_results.max_distance,
                    "start_nodes": len(graph_results.start_nodes)
                }
            )
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return None
    
    def _convert_fusion_results(
        self,
        fusion_results: FusionResults,
        strategy_results: List[StrategyResults]
    ) -> HybridSearchResults:
        """Convert fusion results to hybrid search results."""
        hybrid_results = []
        
        # Create mapping of strategy results for score extraction
        strategy_scores = {}
        for strategy_result in strategy_results:
            strategy_scores[strategy_result.strategy_name] = {
                result.doc_id: result.score for result in strategy_result.results
            }
        
        # Convert each fused result
        for fused_result in fusion_results.results:
            hybrid_result = HybridSearchResult(
                doc_id=fused_result.doc_id,
                content=fused_result.content,
                final_score=fused_result.final_score,
                metadata=fused_result.metadata,
                contributing_strategies=fused_result.contributing_strategies,
                fusion_explanation=fused_result.fusion_explanation,
                original_ranks=fused_result.original_ranks
            )
            
            # Extract strategy-specific scores
            if "vector" in strategy_scores:
                hybrid_result.vector_score = strategy_scores["vector"].get(fused_result.doc_id)
            
            if "bm25" in strategy_scores:
                hybrid_result.bm25_score = strategy_scores["bm25"].get(fused_result.doc_id)
            
            if "graph" in strategy_scores:
                hybrid_result.graph_score = strategy_scores["graph"].get(fused_result.doc_id)
            
            hybrid_results.append(hybrid_result)
        
        return HybridSearchResults(
            query=fusion_results.query,
            results=hybrid_results,
            strategy_used=SearchStrategy.AUTO,  # Will be updated by caller
            fusion_method=fusion_results.fusion_strategy,
            total_time=fusion_results.total_processing_time,
            fusion_time=fusion_results.fusion_time,
            stats=fusion_results.stats
        )
    
    async def get_search_suggestions(
        self,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """
        Get search suggestions based on partial query.
        
        Args:
            partial_query: Partial search query
            limit: Maximum number of suggestions
            
        Returns:
            List[str]: Search suggestions
        """
        suggestions = []
        
        # TODO: Implement sophisticated suggestion logic
        # For now, return simple suggestions based on common patterns
        
        if len(partial_query) >= 2:
            # Simple word completion suggestions
            common_terms = [
                "explain", "summarize", "compare", "analyze",
                "define", "describe", "discuss", "evaluate"
            ]
            
            for term in common_terms:
                if term.startswith(partial_query.lower()):
                    suggestions.append(term)
                
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hybrid search engine."""
        stats = {
            "hybrid_engine": {
                "available_strategies": [s.value for s in SearchStrategy],
                "fusion_methods": ["rrf", "weighted", "combsum"]
            }
        }
        
        # Add stats from individual engines
        try:
            stats["bm25_engine"] = self.bm25_engine.get_engine_stats()
        except Exception as e:
            logger.warning(f"Could not get BM25 stats: {e}")
            stats["bm25_engine"] = {"error": str(e)}
        
        try:
            stats["graph_engine"] = self.graph_engine.get_engine_stats()
        except Exception as e:
            logger.warning(f"Could not get graph stats: {e}")
            stats["graph_engine"] = {"error": str(e)}
        
        try:
            stats["fusion_engine"] = self.fusion_engine.get_fusion_stats()
        except Exception as e:
            logger.warning(f"Could not get fusion stats: {e}")
            stats["fusion_engine"] = {"error": str(e)}
        
        return stats


# Global hybrid search engine instance
_hybrid_search_engine: Optional[HybridSearchEngine] = None


async def get_hybrid_search_engine() -> HybridSearchEngine:
    """
    Get the global hybrid search engine instance.
    
    Returns:
        HybridSearchEngine: Initialized hybrid search engine
    """
    global _hybrid_search_engine
    
    if _hybrid_search_engine is None:
        _hybrid_search_engine = HybridSearchEngine()
        if not await _hybrid_search_engine.initialize():
            raise RuntimeError("Failed to initialize hybrid search engine")
    
    return _hybrid_search_engine