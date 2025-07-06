"""
Performance Comparison Tools for Retrieval Strategies.

This module provides comprehensive benchmarking and evaluation tools for comparing
the performance of different retrieval strategies in the hybrid search system.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import math

from src.utils.logger import get_logger
from src.utils.bm25_search import BM25SearchEngine
from src.utils.graph_retrieval import GraphSearchEngine
from src.utils.result_fusion import HybridResultFusion, StrategyResults, SearchResult
from src.utils.hybrid_search import HybridSearchEngine, HybridSearchConfig, SearchStrategy

logger = get_logger(__name__)


@dataclass
class BenchmarkQuery:
    """Benchmark query with expected results."""
    query: str
    expected_docs: List[str] = field(default_factory=list)  # Expected relevant document IDs
    query_type: str = "general"  # "factual", "conceptual", "comparative", etc.
    difficulty: str = "medium"  # "easy", "medium", "hard"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single retrieval strategy."""
    strategy_name: str
    
    # Speed metrics
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    
    # Quality metrics (require ground truth)
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mean_reciprocal_rank: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    
    # Coverage metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    empty_results: int = 0
    
    # Resource metrics
    average_results_returned: float = 0.0
    max_memory_usage: Optional[float] = None
    
    # Additional statistics
    response_times: List[float] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comparison report between multiple strategies."""
    query_set: str
    strategies_compared: List[str]
    metrics: Dict[str, PerformanceMetrics]
    
    # Overall comparison
    fastest_strategy: str
    most_accurate_strategy: Optional[str] = None
    most_reliable_strategy: str
    
    # Summary statistics
    average_speedup: Dict[str, float] = field(default_factory=dict)  # Relative to baseline
    quality_rankings: Dict[str, int] = field(default_factory=dict)  # 1 = best
    
    # Report metadata
    benchmark_duration: float = 0.0
    total_queries_processed: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class RetrievalEvaluator:
    """
    Evaluation utilities for measuring retrieval quality.
    
    Implements standard information retrieval evaluation metrics.
    """
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            float: Precision@K score
        """
        if not retrieved or k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_retrieved = [doc for doc in top_k if doc in relevant]
        
        return len(relevant_retrieved) / len(top_k)
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            float: Recall@K score
        """
        if not relevant:
            return 0.0
        
        if not retrieved or k <= 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_retrieved = [doc for doc in top_k if doc in relevant]
        
        return len(relevant_retrieved) / len(relevant)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_lists: List[List[str]], relevant_lists: List[List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_lists: List of retrieved document lists for each query
            relevant_lists: List of relevant document lists for each query
            
        Returns:
            float: MRR score
        """
        if not retrieved_lists or not relevant_lists:
            return 0.0
        
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            rr = 0.0
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            float: NDCG@K score
        """
        if not retrieved or not relevant or k <= 0:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc in relevant:
                gain = 1.0  # Binary relevance
                discount = math.log2(i + 2)  # i+2 because rank starts at 1
                dcg += gain / discount
        
        # Calculate IDCG (perfect ranking)
        idcg = 0.0
        for i in range(min(k, len(relevant))):
            gain = 1.0
            discount = math.log2(i + 2)
            idcg += gain / discount
        
        return dcg / idcg if idcg > 0 else 0.0


class StrategyBenchmarker:
    """
    Benchmark individual retrieval strategies.
    
    Measures performance, quality, and reliability of retrieval strategies.
    """
    
    def __init__(self):
        """Initialize strategy benchmarker."""
        self.evaluator = RetrievalEvaluator()
        
    async def benchmark_vector_search(
        self,
        queries: List[BenchmarkQuery],
        vector_db,
        collection_name: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        Benchmark vector similarity search.
        
        Args:
            queries: List of benchmark queries
            vector_db: Vector database instance
            collection_name: Collection name for queries
            
        Returns:
            PerformanceMetrics: Performance metrics
        """
        logger.info(f"Benchmarking vector search with {len(queries)} queries")
        
        # Use default collection name if not provided
        if collection_name is None:
            from src.utils.config import get_settings
            settings = get_settings()
            collection_name = settings.chroma_collection_name
        
        response_times = []
        successful_queries = 0
        failed_queries = 0
        empty_results = 0
        error_messages = []
        total_results = []
        
        # Quality metrics storage
        precision_5_scores = []
        precision_10_scores = []
        recall_5_scores = []
        recall_10_scores = []
        retrieved_lists = []
        relevant_lists = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        
        for query in queries:
            try:
                start_time = time.time()
                
                # Perform vector search
                results = await vector_db.query(
                    collection_name=collection_name,
                    query_text=query.query,
                    n_results=20  # Get more results for evaluation
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if results and "documents" in results and results["documents"]:
                    successful_queries += 1
                    total_results.append(len(results["documents"]))
                    
                    # Extract document IDs for evaluation
                    retrieved_docs = []
                    for metadata in results.get("metadatas", []):
                        if metadata and "doc_id" in metadata:
                            retrieved_docs.append(metadata["doc_id"])
                    
                    # Calculate quality metrics if ground truth available
                    if query.expected_docs:
                        precision_5_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        precision_10_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        recall_5_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        recall_10_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        ndcg_5_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        ndcg_10_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        
                        retrieved_lists.append(retrieved_docs)
                        relevant_lists.append(query.expected_docs)
                else:
                    empty_results += 1
                    total_results.append(0)
                
            except Exception as e:
                failed_queries += 1
                error_messages.append(f"Query '{query.query}': {str(e)}")
                response_times.append(0.0)  # Failed query
                total_results.append(0)
        
        # Calculate summary metrics
        metrics = PerformanceMetrics(
            strategy_name="vector",
            average_response_time=statistics.mean(response_times) if response_times else 0.0,
            median_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0.0,
            total_queries=len(queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            empty_results=empty_results,
            average_results_returned=statistics.mean(total_results) if total_results else 0.0,
            response_times=response_times,
            error_messages=error_messages
        )
        
        # Add quality metrics if available
        if precision_5_scores:
            metrics.precision_at_5 = statistics.mean(precision_5_scores)
            metrics.precision_at_10 = statistics.mean(precision_10_scores)
            metrics.recall_at_5 = statistics.mean(recall_5_scores)
            metrics.recall_at_10 = statistics.mean(recall_10_scores)
            metrics.ndcg_at_5 = statistics.mean(ndcg_5_scores)
            metrics.ndcg_at_10 = statistics.mean(ndcg_10_scores)
            metrics.mean_reciprocal_rank = self.evaluator.mean_reciprocal_rank(
                retrieved_lists, relevant_lists
            )
        
        logger.info(f"Vector search benchmark completed: {successful_queries}/{len(queries)} successful")
        
        return metrics
    
    async def benchmark_bm25_search(
        self,
        queries: List[BenchmarkQuery],
        bm25_engine: BM25SearchEngine
    ) -> PerformanceMetrics:
        """
        Benchmark BM25 keyword search.
        
        Args:
            queries: List of benchmark queries
            bm25_engine: BM25 search engine
            
        Returns:
            PerformanceMetrics: Performance metrics
        """
        logger.info(f"Benchmarking BM25 search with {len(queries)} queries")
        
        response_times = []
        successful_queries = 0
        failed_queries = 0
        empty_results = 0
        error_messages = []
        total_results = []
        
        # Quality metrics storage
        precision_5_scores = []
        precision_10_scores = []
        recall_5_scores = []
        recall_10_scores = []
        retrieved_lists = []
        relevant_lists = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        
        for query in queries:
            try:
                start_time = time.time()
                
                # Perform BM25 search with fallback
                results = await bm25_engine.search_with_fallback(
                    query=query.query,
                    limit=20,
                    strategies=["exact", "fuzzy", "ngram"]
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if results.results:
                    successful_queries += 1
                    total_results.append(len(results.results))
                    
                    # Extract document IDs
                    retrieved_docs = [result.doc_id for result in results.results]
                    
                    # Calculate quality metrics if ground truth available
                    if query.expected_docs:
                        precision_5_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        precision_10_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        recall_5_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        recall_10_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        ndcg_5_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        ndcg_10_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        
                        retrieved_lists.append(retrieved_docs)
                        relevant_lists.append(query.expected_docs)
                else:
                    empty_results += 1
                    total_results.append(0)
                
            except Exception as e:
                failed_queries += 1
                error_messages.append(f"Query '{query.query}': {str(e)}")
                response_times.append(0.0)
                total_results.append(0)
        
        # Calculate summary metrics
        metrics = PerformanceMetrics(
            strategy_name="bm25",
            average_response_time=statistics.mean(response_times) if response_times else 0.0,
            median_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0.0,
            total_queries=len(queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            empty_results=empty_results,
            average_results_returned=statistics.mean(total_results) if total_results else 0.0,
            response_times=response_times,
            error_messages=error_messages
        )
        
        # Add quality metrics if available
        if precision_5_scores:
            metrics.precision_at_5 = statistics.mean(precision_5_scores)
            metrics.precision_at_10 = statistics.mean(precision_10_scores)
            metrics.recall_at_5 = statistics.mean(recall_5_scores)
            metrics.recall_at_10 = statistics.mean(recall_10_scores)
            metrics.ndcg_at_5 = statistics.mean(ndcg_5_scores)
            metrics.ndcg_at_10 = statistics.mean(ndcg_10_scores)
            metrics.mean_reciprocal_rank = self.evaluator.mean_reciprocal_rank(
                retrieved_lists, relevant_lists
            )
        
        logger.info(f"BM25 search benchmark completed: {successful_queries}/{len(queries)} successful")
        
        return metrics
    
    async def benchmark_graph_search(
        self,
        queries: List[BenchmarkQuery],
        graph_engine: GraphSearchEngine
    ) -> PerformanceMetrics:
        """
        Benchmark graph traversal search.
        
        Args:
            queries: List of benchmark queries
            graph_engine: Graph search engine
            
        Returns:
            PerformanceMetrics: Performance metrics
        """
        logger.info(f"Benchmarking graph search with {len(queries)} queries")
        
        response_times = []
        successful_queries = 0
        failed_queries = 0
        empty_results = 0
        error_messages = []
        total_results = []
        
        # Quality metrics storage
        precision_5_scores = []
        precision_10_scores = []
        recall_5_scores = []
        recall_10_scores = []
        retrieved_lists = []
        relevant_lists = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        
        for query in queries:
            try:
                start_time = time.time()
                
                # Perform graph search
                results = await graph_engine.search(
                    query=query.query,
                    strategy="bfs",
                    limit=20
                )
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if results.results:
                    successful_queries += 1
                    total_results.append(len(results.results))
                    
                    # Extract document IDs
                    retrieved_docs = [result.node_id for result in results.results]
                    
                    # Calculate quality metrics if ground truth available
                    if query.expected_docs:
                        precision_5_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        precision_10_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        recall_5_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        recall_10_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        ndcg_5_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        ndcg_10_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        
                        retrieved_lists.append(retrieved_docs)
                        relevant_lists.append(query.expected_docs)
                else:
                    empty_results += 1
                    total_results.append(0)
                
            except Exception as e:
                failed_queries += 1
                error_messages.append(f"Query '{query.query}': {str(e)}")
                response_times.append(0.0)
                total_results.append(0)
        
        # Calculate summary metrics
        metrics = PerformanceMetrics(
            strategy_name="graph",
            average_response_time=statistics.mean(response_times) if response_times else 0.0,
            median_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0.0,
            total_queries=len(queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            empty_results=empty_results,
            average_results_returned=statistics.mean(total_results) if total_results else 0.0,
            response_times=response_times,
            error_messages=error_messages
        )
        
        # Add quality metrics if available
        if precision_5_scores:
            metrics.precision_at_5 = statistics.mean(precision_5_scores)
            metrics.precision_at_10 = statistics.mean(precision_10_scores)
            metrics.recall_at_5 = statistics.mean(recall_5_scores)
            metrics.recall_at_10 = statistics.mean(recall_10_scores)
            metrics.ndcg_at_5 = statistics.mean(ndcg_5_scores)
            metrics.ndcg_at_10 = statistics.mean(ndcg_10_scores)
            metrics.mean_reciprocal_rank = self.evaluator.mean_reciprocal_rank(
                retrieved_lists, relevant_lists
            )
        
        logger.info(f"Graph search benchmark completed: {successful_queries}/{len(queries)} successful")
        
        return metrics
    
    async def benchmark_hybrid_search(
        self,
        queries: List[BenchmarkQuery],
        hybrid_engine: HybridSearchEngine,
        strategy: SearchStrategy = SearchStrategy.ALL_STRATEGIES
    ) -> PerformanceMetrics:
        """
        Benchmark hybrid search engine.
        
        Args:
            queries: List of benchmark queries
            hybrid_engine: Hybrid search engine
            strategy: Search strategy to benchmark
            
        Returns:
            PerformanceMetrics: Performance metrics
        """
        logger.info(f"Benchmarking hybrid search ({strategy.value}) with {len(queries)} queries")
        
        response_times = []
        successful_queries = 0
        failed_queries = 0
        empty_results = 0
        error_messages = []
        total_results = []
        
        # Quality metrics storage
        precision_5_scores = []
        precision_10_scores = []
        recall_5_scores = []
        recall_10_scores = []
        retrieved_lists = []
        relevant_lists = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        
        config = HybridSearchConfig(
            strategy=strategy,
            max_results=20
        )
        
        for query in queries:
            try:
                start_time = time.time()
                
                # Perform hybrid search
                results = await hybrid_engine.search(query.query, config)
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if results.results:
                    successful_queries += 1
                    total_results.append(len(results.results))
                    
                    # Extract document IDs
                    retrieved_docs = [result.doc_id for result in results.results]
                    
                    # Calculate quality metrics if ground truth available
                    if query.expected_docs:
                        precision_5_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        precision_10_scores.append(
                            self.evaluator.precision_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        recall_5_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        recall_10_scores.append(
                            self.evaluator.recall_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        ndcg_5_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 5)
                        )
                        ndcg_10_scores.append(
                            self.evaluator.ndcg_at_k(retrieved_docs, query.expected_docs, 10)
                        )
                        
                        retrieved_lists.append(retrieved_docs)
                        relevant_lists.append(query.expected_docs)
                else:
                    empty_results += 1
                    total_results.append(0)
                
            except Exception as e:
                failed_queries += 1
                error_messages.append(f"Query '{query.query}': {str(e)}")
                response_times.append(0.0)
                total_results.append(0)
        
        # Calculate summary metrics
        metrics = PerformanceMetrics(
            strategy_name=f"hybrid_{strategy.value}",
            average_response_time=statistics.mean(response_times) if response_times else 0.0,
            median_response_time=statistics.median(response_times) if response_times else 0.0,
            p95_response_time=self._percentile(response_times, 95) if response_times else 0.0,
            p99_response_time=self._percentile(response_times, 99) if response_times else 0.0,
            total_queries=len(queries),
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            empty_results=empty_results,
            average_results_returned=statistics.mean(total_results) if total_results else 0.0,
            response_times=response_times,
            error_messages=error_messages
        )
        
        # Add quality metrics if available
        if precision_5_scores:
            metrics.precision_at_5 = statistics.mean(precision_5_scores)
            metrics.precision_at_10 = statistics.mean(precision_10_scores)
            metrics.recall_at_5 = statistics.mean(recall_5_scores)
            metrics.recall_at_10 = statistics.mean(recall_10_scores)
            metrics.ndcg_at_5 = statistics.mean(ndcg_5_scores)
            metrics.ndcg_at_10 = statistics.mean(ndcg_10_scores)
            metrics.mean_reciprocal_rank = self.evaluator.mean_reciprocal_rank(
                retrieved_lists, relevant_lists
            )
        
        logger.info(f"Hybrid search benchmark completed: {successful_queries}/{len(queries)} successful")
        
        return metrics
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value from data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
        
        d0 = sorted_data[int(f)] * (c - k)
        d1 = sorted_data[int(c)] * (k - f)
        
        return d0 + d1


class ComprehensiveBenchmark:
    """
    Comprehensive benchmarking suite for all retrieval strategies.
    
    Provides complete performance comparison and analysis across multiple
    retrieval strategies with detailed reporting.
    """
    
    def __init__(self):
        """Initialize comprehensive benchmark."""
        self.benchmarker = StrategyBenchmarker()
        self.default_queries = self._create_default_query_set()
    
    def _create_default_query_set(self) -> List[BenchmarkQuery]:
        """Create a default set of benchmark queries."""
        return [
            BenchmarkQuery("machine learning algorithms", [], "technical", "medium"),
            BenchmarkQuery("what is artificial intelligence", [], "conceptual", "easy"),
            BenchmarkQuery("deep learning neural networks", [], "technical", "medium"),
            BenchmarkQuery("natural language processing applications", [], "technical", "medium"),
            BenchmarkQuery("computer vision image recognition", [], "technical", "medium"),
            BenchmarkQuery("compare supervised and unsupervised learning", [], "comparative", "hard"),
            BenchmarkQuery("data science methodology", [], "conceptual", "medium"),
            BenchmarkQuery("python programming", [], "technical", "easy"),
            BenchmarkQuery("statistical analysis methods", [], "technical", "medium"),
            BenchmarkQuery("how does reinforcement learning work", [], "conceptual", "hard")
        ]
    
    async def run_comprehensive_benchmark(
        self,
        hybrid_engine: HybridSearchEngine,
        queries: Optional[List[BenchmarkQuery]] = None,
        include_strategies: Optional[List[str]] = None
    ) -> ComparisonReport:
        """
        Run comprehensive benchmark across all strategies.
        
        Args:
            hybrid_engine: Initialized hybrid search engine
            queries: Custom query set (uses default if None)
            include_strategies: Strategies to include (all if None)
            
        Returns:
            ComparisonReport: Comprehensive comparison report
        """
        start_time = time.time()
        queries = queries or self.default_queries
        include_strategies = include_strategies or ["vector", "bm25", "graph", "hybrid"]
        
        logger.info(f"Running comprehensive benchmark with {len(queries)} queries "
                   f"across {len(include_strategies)} strategies")
        
        metrics_dict = {}
        
        # Benchmark each strategy
        if "vector" in include_strategies and hybrid_engine.vector_db:
            try:
                vector_metrics = await self.benchmarker.benchmark_vector_search(
                    queries, hybrid_engine.vector_db
                )
                metrics_dict["vector"] = vector_metrics
            except Exception as e:
                logger.error(f"Vector benchmark failed: {e}")
        
        if "bm25" in include_strategies:
            try:
                bm25_metrics = await self.benchmarker.benchmark_bm25_search(
                    queries, hybrid_engine.bm25_engine
                )
                metrics_dict["bm25"] = bm25_metrics
            except Exception as e:
                logger.error(f"BM25 benchmark failed: {e}")
        
        if "graph" in include_strategies:
            try:
                graph_metrics = await self.benchmarker.benchmark_graph_search(
                    queries, hybrid_engine.graph_engine
                )
                metrics_dict["graph"] = graph_metrics
            except Exception as e:
                logger.error(f"Graph benchmark failed: {e}")
        
        if "hybrid" in include_strategies:
            try:
                hybrid_metrics = await self.benchmarker.benchmark_hybrid_search(
                    queries, hybrid_engine, SearchStrategy.ALL_STRATEGIES
                )
                metrics_dict["hybrid"] = hybrid_metrics
            except Exception as e:
                logger.error(f"Hybrid benchmark failed: {e}")
        
        # Generate comparison report
        benchmark_duration = time.time() - start_time
        report = self._generate_comparison_report(
            metrics_dict, queries, benchmark_duration
        )
        
        logger.info(f"Comprehensive benchmark completed in {benchmark_duration:.2f}s")
        
        return report
    
    def _generate_comparison_report(
        self,
        metrics_dict: Dict[str, PerformanceMetrics],
        queries: List[BenchmarkQuery],
        benchmark_duration: float
    ) -> ComparisonReport:
        """Generate detailed comparison report."""
        if not metrics_dict:
            return ComparisonReport(
                query_set="empty",
                strategies_compared=[],
                metrics={},
                fastest_strategy="none",
                most_reliable_strategy="none",
                benchmark_duration=benchmark_duration,
                total_queries_processed=0
            )
        
        strategies = list(metrics_dict.keys())
        
        # Find fastest strategy (lowest average response time)
        fastest_strategy = min(
            strategies,
            key=lambda s: metrics_dict[s].average_response_time
        )
        
        # Find most reliable strategy (highest success rate)
        most_reliable_strategy = max(
            strategies,
            key=lambda s: metrics_dict[s].successful_queries / max(metrics_dict[s].total_queries, 1)
        )
        
        # Find most accurate strategy (highest precision@5)
        most_accurate_strategy = None
        if any(m.precision_at_5 is not None for m in metrics_dict.values()):
            valid_strategies = [
                s for s in strategies if metrics_dict[s].precision_at_5 is not None
            ]
            if valid_strategies:
                most_accurate_strategy = max(
                    valid_strategies,
                    key=lambda s: metrics_dict[s].precision_at_5
                )
        
        # Calculate relative speedups (compared to slowest)
        slowest_time = max(m.average_response_time for m in metrics_dict.values())
        average_speedup = {}
        for strategy, metrics in metrics_dict.items():
            if metrics.average_response_time > 0:
                speedup = slowest_time / metrics.average_response_time
                average_speedup[strategy] = speedup
            else:
                average_speedup[strategy] = 1.0
        
        # Calculate quality rankings
        quality_rankings = {}
        if most_accurate_strategy:
            # Sort by precision@5 (descending)
            valid_strategies = [
                s for s in strategies if metrics_dict[s].precision_at_5 is not None
            ]
            sorted_strategies = sorted(
                valid_strategies,
                key=lambda s: metrics_dict[s].precision_at_5,
                reverse=True
            )
            quality_rankings = {s: i + 1 for i, s in enumerate(sorted_strategies)}
        
        return ComparisonReport(
            query_set=f"benchmark_{len(queries)}_queries",
            strategies_compared=strategies,
            metrics=metrics_dict,
            fastest_strategy=fastest_strategy,
            most_accurate_strategy=most_accurate_strategy,
            most_reliable_strategy=most_reliable_strategy,
            average_speedup=average_speedup,
            quality_rankings=quality_rankings,
            benchmark_duration=benchmark_duration,
            total_queries_processed=len(queries) * len(strategies)
        )
    
    def generate_report_summary(self, report: ComparisonReport) -> str:
        """
        Generate human-readable summary of comparison report.
        
        Args:
            report: Comparison report
            
        Returns:
            str: Formatted report summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("RETRIEVAL STRATEGY BENCHMARK REPORT")
        summary.append("=" * 60)
        summary.append(f"Query Set: {report.query_set}")
        summary.append(f"Strategies Compared: {', '.join(report.strategies_compared)}")
        summary.append(f"Total Queries Processed: {report.total_queries_processed}")
        summary.append(f"Benchmark Duration: {report.benchmark_duration:.2f}s")
        summary.append("")
        
        # Performance summary
        summary.append("PERFORMANCE SUMMARY")
        summary.append("-" * 30)
        summary.append(f"Fastest Strategy: {report.fastest_strategy}")
        summary.append(f"Most Reliable Strategy: {report.most_reliable_strategy}")
        if report.most_accurate_strategy:
            summary.append(f"Most Accurate Strategy: {report.most_accurate_strategy}")
        summary.append("")
        
        # Detailed metrics
        summary.append("DETAILED METRICS")
        summary.append("-" * 30)
        
        for strategy, metrics in report.metrics.items():
            summary.append(f"\n{strategy.upper()} STRATEGY:")
            summary.append(f"  Average Response Time: {metrics.average_response_time:.3f}s")
            summary.append(f"  Median Response Time: {metrics.median_response_time:.3f}s")
            summary.append(f"  95th Percentile: {metrics.p95_response_time:.3f}s")
            summary.append(f"  Success Rate: {metrics.successful_queries}/{metrics.total_queries} "
                          f"({100 * metrics.successful_queries / max(metrics.total_queries, 1):.1f}%)")
            summary.append(f"  Average Results: {metrics.average_results_returned:.1f}")
            
            if metrics.precision_at_5 is not None:
                summary.append(f"  Precision@5: {metrics.precision_at_5:.3f}")
                summary.append(f"  Precision@10: {metrics.precision_at_10:.3f}")
                summary.append(f"  Recall@5: {metrics.recall_at_5:.3f}")
                summary.append(f"  Recall@10: {metrics.recall_at_10:.3f}")
                summary.append(f"  MRR: {metrics.mean_reciprocal_rank:.3f}")
                summary.append(f"  NDCG@5: {metrics.ndcg_at_5:.3f}")
            
            if metrics.error_messages:
                summary.append(f"  Errors: {len(metrics.error_messages)}")
        
        # Relative performance
        if len(report.strategies_compared) > 1:
            summary.append("\nRELATIVE PERFORMANCE")
            summary.append("-" * 30)
            for strategy, speedup in report.average_speedup.items():
                summary.append(f"  {strategy}: {speedup:.2f}x speedup")
        
        summary.append("")
        summary.append("=" * 60)
        
        return "\n".join(summary)
    
    def export_detailed_results(
        self,
        report: ComparisonReport,
        filepath: str,
        format: str = "json"
    ):
        """
        Export detailed benchmark results to file.
        
        Args:
            report: Comparison report
            filepath: Output file path
            format: Export format ("json" or "csv")
        """
        if format == "json":
            # Convert to JSON-serializable format
            export_data = {
                "query_set": report.query_set,
                "strategies_compared": report.strategies_compared,
                "fastest_strategy": report.fastest_strategy,
                "most_accurate_strategy": report.most_accurate_strategy,
                "most_reliable_strategy": report.most_reliable_strategy,
                "benchmark_duration": report.benchmark_duration,
                "total_queries_processed": report.total_queries_processed,
                "timestamp": report.timestamp.isoformat(),
                "metrics": {}
            }
            
            # Add metrics data
            for strategy, metrics in report.metrics.items():
                export_data["metrics"][strategy] = {
                    "strategy_name": metrics.strategy_name,
                    "average_response_time": metrics.average_response_time,
                    "median_response_time": metrics.median_response_time,
                    "p95_response_time": metrics.p95_response_time,
                    "p99_response_time": metrics.p99_response_time,
                    "total_queries": metrics.total_queries,
                    "successful_queries": metrics.successful_queries,
                    "failed_queries": metrics.failed_queries,
                    "empty_results": metrics.empty_results,
                    "average_results_returned": metrics.average_results_returned,
                    "precision_at_5": metrics.precision_at_5,
                    "precision_at_10": metrics.precision_at_10,
                    "recall_at_5": metrics.recall_at_5,
                    "recall_at_10": metrics.recall_at_10,
                    "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
                    "ndcg_at_5": metrics.ndcg_at_5,
                    "ndcg_at_10": metrics.ndcg_at_10,
                    "response_times": metrics.response_times,
                    "error_count": len(metrics.error_messages)
                }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        logger.info(f"Benchmark results exported to {filepath}")


# Convenience function for quick benchmarking
async def quick_benchmark(
    hybrid_engine: HybridSearchEngine,
    num_queries: int = 10
) -> ComparisonReport:
    """
    Run a quick benchmark with default settings.
    
    Args:
        hybrid_engine: Initialized hybrid search engine
        num_queries: Number of queries to use (limited from default set)
        
    Returns:
        ComparisonReport: Benchmark results
    """
    benchmark = ComprehensiveBenchmark()
    queries = benchmark.default_queries[:num_queries]
    
    return await benchmark.run_comprehensive_benchmark(
        hybrid_engine, queries, ["bm25", "hybrid"]
    )