"""
Result Fusion Algorithms for Hybrid Retrieval Pipeline.

This module implements various algorithms for intelligently combining results
from multiple retrieval strategies (vector search, BM25, graph traversal)
to produce improved overall search results.
"""

import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Generic search result that can come from any retrieval strategy."""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_strategy: str = "unknown"
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class StrategyResults:
    """Results from a single retrieval strategy."""
    strategy_name: str
    results: List[SearchResult]
    search_time: float
    total_docs_searched: int
    query_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    """Result after fusion with combined scoring information."""
    doc_id: str
    final_score: float
    content: str
    metadata: Dict[str, Any]
    contributing_strategies: List[str]
    strategy_scores: Dict[str, float]
    fusion_explanation: Dict[str, Any]
    original_ranks: Dict[str, int]  # Original rank in each strategy


@dataclass
class FusionResults:
    """Complete results from fusion process."""
    query: str
    results: List[FusedResult]
    fusion_strategy: str
    input_strategies: List[str]
    total_processing_time: float
    fusion_time: float
    stats: Dict[str, Any] = field(default_factory=dict)


class FusionStrategy(ABC):
    """Abstract base class for result fusion strategies."""
    
    @abstractmethod
    def fuse_results(
        self,
        strategy_results: List[StrategyResults],
        query: str,
        limit: int = 10
    ) -> FusionResults:
        """
        Fuse results from multiple strategies.
        
        Args:
            strategy_results: Results from different retrieval strategies
            query: Original search query
            limit: Maximum number of results to return
            
        Returns:
            FusionResults: Fused and ranked results
        """
        pass


class ReciprocalRankFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) algorithm implementation.
    
    RRF combines ranked lists by computing:
    score(doc) = Σ(1/(k + rank(doc, list_i))) for all lists containing the document
    
    This approach is effective because it:
    1. Doesn't require score normalization between strategies
    2. Gives higher weight to documents that appear in multiple lists
    3. Emphasizes top-ranked documents while still considering lower ranks
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.
        
        Args:
            k: RRF parameter controlling the weighting curve (typical range: 10-100)
               Lower k gives more weight to high-ranking documents
        """
        self.k = k
        logger.info(f"RRF fusion initialized with k={k}")
    
    def fuse_results(
        self,
        strategy_results: List[StrategyResults],
        query: str,
        limit: int = 10
    ) -> FusionResults:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        Args:
            strategy_results: Results from different retrieval strategies
            query: Original search query
            limit: Maximum number of results to return
            
        Returns:
            FusionResults: RRF-fused results
        """
        start_time = time.time()
        
        if not strategy_results:
            return FusionResults(
                query=query,
                results=[],
                fusion_strategy="rrf",
                input_strategies=[],
                total_processing_time=time.time() - start_time,
                fusion_time=0.0
            )
        
        fusion_start = time.time()
        
        # Track all unique documents and their RRF scores
        document_scores: Dict[str, float] = defaultdict(float)
        document_info: Dict[str, Dict[str, Any]] = {}
        document_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        document_strategies: Dict[str, List[str]] = defaultdict(list)
        document_strategy_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        strategy_names = []
        
        # Process each strategy's results
        for strategy_result in strategy_results:
            strategy_name = strategy_result.strategy_name
            strategy_names.append(strategy_name)
            
            for rank, result in enumerate(strategy_result.results):
                doc_id = result.doc_id
                rank_position = rank + 1  # Convert to 1-based ranking
                
                # Calculate RRF contribution
                rrf_contribution = 1.0 / (self.k + rank_position)
                document_scores[doc_id] += rrf_contribution
                
                # Store document information (use first occurrence)
                if doc_id not in document_info:
                    document_info[doc_id] = {
                        'content': result.content,
                        'metadata': result.metadata.copy()
                    }
                
                # Track which strategies contributed to this document
                if strategy_name not in document_strategies[doc_id]:
                    document_strategies[doc_id].append(strategy_name)
                
                # Store original rank and score from this strategy
                document_ranks[doc_id][strategy_name] = rank_position
                document_strategy_scores[doc_id][strategy_name] = result.score
        
        # Create fused results
        fused_results = []
        
        for doc_id, final_score in document_scores.items():
            if doc_id in document_info:
                fusion_explanation = {
                    'rrf_k_parameter': self.k,
                    'strategy_contributions': {
                        strategy: 1.0 / (self.k + document_ranks[doc_id].get(strategy, float('inf')))
                        for strategy in document_strategies[doc_id]
                    },
                    'total_strategies': len(document_strategies[doc_id]),
                    'coverage_ratio': len(document_strategies[doc_id]) / len(strategy_names)
                }
                
                fused_result = FusedResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    content=document_info[doc_id]['content'],
                    metadata=document_info[doc_id]['metadata'],
                    contributing_strategies=document_strategies[doc_id].copy(),
                    strategy_scores=document_strategy_scores[doc_id].copy(),
                    fusion_explanation=fusion_explanation,
                    original_ranks=document_ranks[doc_id].copy()
                )
                
                fused_results.append(fused_result)
        
        # Sort by final RRF score (descending) and limit results
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = fused_results[:limit]
        
        fusion_time = time.time() - fusion_start
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_fusion_stats(strategy_results, final_results, strategy_names)
        
        logger.info(f"RRF fusion completed: {len(final_results)} results "
                   f"(fusion_time: {fusion_time:.3f}s, total_time: {total_time:.3f}s)")
        
        return FusionResults(
            query=query,
            results=final_results,
            fusion_strategy="rrf",
            input_strategies=strategy_names,
            total_processing_time=total_time,
            fusion_time=fusion_time,
            stats=stats
        )
    
    def _calculate_fusion_stats(
        self,
        strategy_results: List[StrategyResults],
        final_results: List[FusedResult],
        strategy_names: List[str]
    ) -> Dict[str, Any]:
        """Calculate detailed statistics about the fusion process."""
        total_input_results = sum(len(sr.results) for sr in strategy_results)
        
        # Count documents by number of contributing strategies
        coverage_distribution = defaultdict(int)
        for result in final_results:
            num_strategies = len(result.contributing_strategies)
            coverage_distribution[num_strategies] += 1
        
        # Calculate strategy representation in final results
        strategy_representation = defaultdict(int)
        for result in final_results:
            for strategy in result.contributing_strategies:
                strategy_representation[strategy] += 1
        
        return {
            'total_input_results': total_input_results,
            'total_final_results': len(final_results),
            'unique_documents_processed': len(set(
                result.doc_id for sr in strategy_results for result in sr.results
            )),
            'coverage_distribution': dict(coverage_distribution),
            'strategy_representation': dict(strategy_representation),
            'average_strategies_per_result': sum(
                len(result.contributing_strategies) for result in final_results
            ) / len(final_results) if final_results else 0,
            'max_possible_score': len(strategy_names) / self.k if strategy_names else 0
        }


class WeightedScoreFusion(FusionStrategy):
    """
    Weighted score fusion that combines normalized scores from different strategies.
    
    This approach normalizes scores from each strategy and combines them using
    weighted averages, allowing for more fine-grained control over the contribution
    of each retrieval strategy.
    """
    
    def __init__(self, strategy_weights: Dict[str, float] = None, normalization: str = 'min_max'):
        """
        Initialize weighted score fusion.
        
        Args:
            strategy_weights: Weights for each strategy (default: equal weights)
            normalization: Score normalization method ('min_max', 'z_score', 'none')
        """
        self.strategy_weights = strategy_weights or {}
        self.normalization = normalization
        
        logger.info(f"Weighted fusion initialized: weights={strategy_weights}, "
                   f"normalization={normalization}")
    
    def fuse_results(
        self,
        strategy_results: List[StrategyResults],
        query: str,
        limit: int = 10
    ) -> FusionResults:
        """
        Fuse results using weighted score combination.
        
        Args:
            strategy_results: Results from different retrieval strategies
            query: Original search query
            limit: Maximum number of results to return
            
        Returns:
            FusionResults: Weighted-fused results
        """
        start_time = time.time()
        
        if not strategy_results:
            return FusionResults(
                query=query,
                results=[],
                fusion_strategy="weighted",
                input_strategies=[],
                total_processing_time=time.time() - start_time,
                fusion_time=0.0
            )
        
        fusion_start = time.time()
        
        # Normalize scores for each strategy
        normalized_strategy_results = self._normalize_strategy_scores(strategy_results)
        
        # Get strategy names and set default weights
        strategy_names = [sr.strategy_name for sr in strategy_results]
        default_weight = 1.0 / len(strategy_names)
        weights = {name: self.strategy_weights.get(name, default_weight) for name in strategy_names}
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {name: weight / weight_sum for name, weight in weights.items()}
        
        # Collect all unique documents and their weighted scores
        document_scores: Dict[str, float] = defaultdict(float)
        document_info: Dict[str, Dict[str, Any]] = {}
        document_strategies: Dict[str, List[str]] = defaultdict(list)
        document_strategy_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        document_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Process normalized results
        for normalized_results, original_results in zip(normalized_strategy_results, strategy_results):
            strategy_name = normalized_results.strategy_name
            weight = weights[strategy_name]
            
            for rank, (norm_result, orig_result) in enumerate(zip(normalized_results.results, original_results.results)):
                doc_id = norm_result.doc_id
                
                # Add weighted normalized score
                weighted_score = norm_result.score * weight
                document_scores[doc_id] += weighted_score
                
                # Store document information (use first occurrence)
                if doc_id not in document_info:
                    document_info[doc_id] = {
                        'content': norm_result.content,
                        'metadata': norm_result.metadata.copy()
                    }
                
                # Track contributing strategies
                if strategy_name not in document_strategies[doc_id]:
                    document_strategies[doc_id].append(strategy_name)
                
                # Store original scores and ranks
                document_strategy_scores[doc_id][strategy_name] = orig_result.score
                document_ranks[doc_id][strategy_name] = rank + 1
        
        # Create fused results
        fused_results = []
        
        for doc_id, final_score in document_scores.items():
            if doc_id in document_info:
                fusion_explanation = {
                    'normalization_method': self.normalization,
                    'strategy_weights': {
                        strategy: weights[strategy] for strategy in document_strategies[doc_id]
                    },
                    'weighted_contributions': {
                        strategy: document_strategy_scores[doc_id][strategy] * weights.get(strategy, 0)
                        for strategy in document_strategies[doc_id]
                    },
                    'total_strategies': len(document_strategies[doc_id])
                }
                
                fused_result = FusedResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    content=document_info[doc_id]['content'],
                    metadata=document_info[doc_id]['metadata'],
                    contributing_strategies=document_strategies[doc_id].copy(),
                    strategy_scores=document_strategy_scores[doc_id].copy(),
                    fusion_explanation=fusion_explanation,
                    original_ranks=document_ranks[doc_id].copy()
                )
                
                fused_results.append(fused_result)
        
        # Sort by final score and limit
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = fused_results[:limit]
        
        fusion_time = time.time() - fusion_start
        total_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_weighted_fusion_stats(strategy_results, final_results, weights)
        
        logger.info(f"Weighted fusion completed: {len(final_results)} results "
                   f"(fusion_time: {fusion_time:.3f}s, total_time: {total_time:.3f}s)")
        
        return FusionResults(
            query=query,
            results=final_results,
            fusion_strategy="weighted",
            input_strategies=strategy_names,
            total_processing_time=total_time,
            fusion_time=fusion_time,
            stats=stats
        )
    
    def _normalize_strategy_scores(self, strategy_results: List[StrategyResults]) -> List[StrategyResults]:
        """Normalize scores within each strategy's results."""
        normalized_results = []
        
        for strategy_result in strategy_results:
            if not strategy_result.results:
                normalized_results.append(strategy_result)
                continue
            
            scores = [result.score for result in strategy_result.results]
            
            if self.normalization == 'min_max':
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score
                
                if score_range > 0:
                    normalized_scores = [(score - min_score) / score_range for score in scores]
                else:
                    normalized_scores = [1.0] * len(scores)
                    
            elif self.normalization == 'z_score':
                mean_score = sum(scores) / len(scores)
                variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
                std_dev = math.sqrt(variance) if variance > 0 else 1.0
                
                normalized_scores = [(score - mean_score) / std_dev for score in scores]
                # Convert to positive range [0, 1]
                min_norm = min(normalized_scores)
                max_norm = max(normalized_scores)
                norm_range = max_norm - min_norm
                
                if norm_range > 0:
                    normalized_scores = [(score - min_norm) / norm_range for score in normalized_scores]
                else:
                    normalized_scores = [1.0] * len(scores)
                    
            else:  # 'none' - no normalization
                normalized_scores = scores
            
            # Create normalized results
            normalized_strategy_results = []
            for result, norm_score in zip(strategy_result.results, normalized_scores):
                norm_result = SearchResult(
                    doc_id=result.doc_id,
                    score=norm_score,
                    content=result.content,
                    metadata=result.metadata.copy(),
                    source_strategy=result.source_strategy,
                    explanation=result.explanation.copy()
                )
                normalized_strategy_results.append(norm_result)
            
            normalized_strategy_result = StrategyResults(
                strategy_name=strategy_result.strategy_name,
                results=normalized_strategy_results,
                search_time=strategy_result.search_time,
                total_docs_searched=strategy_result.total_docs_searched,
                query_info=strategy_result.query_info.copy()
            )
            
            normalized_results.append(normalized_strategy_result)
        
        return normalized_results
    
    def _calculate_weighted_fusion_stats(
        self,
        strategy_results: List[StrategyResults],
        final_results: List[FusedResult],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate statistics for weighted fusion."""
        total_input_results = sum(len(sr.results) for sr in strategy_results)
        
        # Calculate effective weight utilization
        weight_utilization = defaultdict(float)
        for result in final_results:
            for strategy in result.contributing_strategies:
                weight_utilization[strategy] += weights.get(strategy, 0)
        
        return {
            'total_input_results': total_input_results,
            'total_final_results': len(final_results),
            'strategy_weights': weights.copy(),
            'weight_utilization': dict(weight_utilization),
            'normalization_method': self.normalization,
            'average_final_score': sum(r.final_score for r in final_results) / len(final_results) if final_results else 0
        }


class CombSumFusion(FusionStrategy):
    """
    CombSUM fusion strategy that simply adds scores from different strategies.
    
    This is a simpler alternative that adds raw scores without normalization.
    Works best when all strategies produce scores in similar ranges.
    """
    
    def __init__(self, strategy_weights: Dict[str, float] = None):
        """
        Initialize CombSUM fusion.
        
        Args:
            strategy_weights: Optional weights for each strategy
        """
        self.strategy_weights = strategy_weights or {}
        logger.info(f"CombSUM fusion initialized with weights: {strategy_weights}")
    
    def fuse_results(
        self,
        strategy_results: List[StrategyResults],
        query: str,
        limit: int = 10
    ) -> FusionResults:
        """Fuse results using CombSUM (simple score addition)."""
        start_time = time.time()
        
        if not strategy_results:
            return FusionResults(
                query=query,
                results=[],
                fusion_strategy="combsum",
                input_strategies=[],
                total_processing_time=time.time() - start_time,
                fusion_time=0.0
            )
        
        fusion_start = time.time()
        
        # Get strategy names and weights
        strategy_names = [sr.strategy_name for sr in strategy_results]
        weights = {name: self.strategy_weights.get(name, 1.0) for name in strategy_names}
        
        # Collect documents and sum their scores
        document_scores: Dict[str, float] = defaultdict(float)
        document_info: Dict[str, Dict[str, Any]] = {}
        document_strategies: Dict[str, List[str]] = defaultdict(list)
        document_strategy_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        document_ranks: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        for strategy_result in strategy_results:
            strategy_name = strategy_result.strategy_name
            weight = weights[strategy_name]
            
            for rank, result in enumerate(strategy_result.results):
                doc_id = result.doc_id
                
                # Add weighted score
                document_scores[doc_id] += result.score * weight
                
                # Store document info
                if doc_id not in document_info:
                    document_info[doc_id] = {
                        'content': result.content,
                        'metadata': result.metadata.copy()
                    }
                
                # Track strategies and scores
                if strategy_name not in document_strategies[doc_id]:
                    document_strategies[doc_id].append(strategy_name)
                
                document_strategy_scores[doc_id][strategy_name] = result.score
                document_ranks[doc_id][strategy_name] = rank + 1
        
        # Create and sort results
        fused_results = []
        for doc_id, final_score in document_scores.items():
            if doc_id in document_info:
                fused_result = FusedResult(
                    doc_id=doc_id,
                    final_score=final_score,
                    content=document_info[doc_id]['content'],
                    metadata=document_info[doc_id]['metadata'],
                    contributing_strategies=document_strategies[doc_id].copy(),
                    strategy_scores=document_strategy_scores[doc_id].copy(),
                    fusion_explanation={'method': 'combsum', 'weights': weights},
                    original_ranks=document_ranks[doc_id].copy()
                )
                fused_results.append(fused_result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        final_results = fused_results[:limit]
        
        fusion_time = time.time() - fusion_start
        total_time = time.time() - start_time
        
        return FusionResults(
            query=query,
            results=final_results,
            fusion_strategy="combsum",
            input_strategies=strategy_names,
            total_processing_time=total_time,
            fusion_time=fusion_time,
            stats={'strategy_weights': weights}
        )


class HybridResultFusion:
    """
    Main result fusion engine that can use different fusion strategies.
    
    Provides a unified interface for combining results from multiple retrieval
    strategies with automatic strategy selection and performance optimization.
    """
    
    def __init__(self):
        """Initialize hybrid result fusion engine."""
        self.fusion_strategies = {
            'rrf': ReciprocalRankFusion(),
            'weighted': WeightedScoreFusion(),
            'combsum': CombSumFusion()
        }
        
        # Default strategy preferences based on scenario
        self.strategy_preferences = {
            'balanced': 'rrf',           # Good general-purpose fusion
            'score_based': 'weighted',   # When scores are meaningful and comparable
            'simple': 'combsum',         # Simple addition of scores
            'conservative': 'rrf'        # RRF tends to be more robust
        }
        
        logger.info("Hybrid result fusion engine initialized")
    
    async def fuse_results(
        self,
        strategy_results: List[StrategyResults],
        query: str,
        fusion_method: str = 'auto',
        limit: int = 10,
        **fusion_params
    ) -> FusionResults:
        """
        Fuse results from multiple strategies.
        
        Args:
            strategy_results: Results from different retrieval strategies
            query: Original search query
            fusion_method: Fusion method ('rrf', 'weighted', 'combsum', 'auto')
            limit: Maximum number of results to return
            **fusion_params: Additional parameters for fusion strategies
            
        Returns:
            FusionResults: Fused results
        """
        if not strategy_results:
            logger.warning("No strategy results provided for fusion")
            return FusionResults(
                query=query,
                results=[],
                fusion_strategy="none",
                input_strategies=[],
                total_processing_time=0.0,
                fusion_time=0.0
            )
        
        # Auto-select fusion method if requested
        if fusion_method == 'auto':
            fusion_method = self._select_fusion_method(strategy_results)
        
        # Get fusion strategy
        if fusion_method not in self.fusion_strategies:
            logger.warning(f"Unknown fusion method: {fusion_method}, using RRF")
            fusion_method = 'rrf'
        
        fusion_strategy = self.fusion_strategies[fusion_method]
        
        # Apply fusion parameters if provided
        if fusion_params:
            if hasattr(fusion_strategy, 'k') and 'k' in fusion_params:
                fusion_strategy.k = fusion_params['k']
            if hasattr(fusion_strategy, 'strategy_weights') and 'weights' in fusion_params:
                fusion_strategy.strategy_weights = fusion_params['weights']
        
        # Perform fusion
        logger.info(f"Fusing results using {fusion_method} strategy: "
                   f"{len(strategy_results)} input strategies")
        
        results = fusion_strategy.fuse_results(strategy_results, query, limit)
        
        return results
    
    def _select_fusion_method(self, strategy_results: List[StrategyResults]) -> str:
        """
        Automatically select the best fusion method based on input characteristics.
        
        Args:
            strategy_results: Results from different strategies
            
        Returns:
            str: Selected fusion method
        """
        num_strategies = len(strategy_results)
        total_results = sum(len(sr.results) for sr in strategy_results)
        
        # Simple heuristics for method selection
        if num_strategies == 1:
            return 'rrf'  # RRF handles single strategy gracefully
        
        if num_strategies >= 3 and total_results > 50:
            return 'rrf'  # RRF works well with many strategies and results
        
        if num_strategies == 2:
            return 'weighted'  # Weighted fusion good for pair-wise combination
        
        # Default to RRF for robustness
        return 'rrf'
    
    def evaluate_fusion_quality(
        self,
        fusion_results: FusionResults,
        ground_truth: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the quality of fusion results.
        
        Args:
            fusion_results: Results from fusion process
            ground_truth: Optional list of relevant document IDs
            
        Returns:
            Dict[str, float]: Quality metrics
        """
        metrics = {}
        
        if not fusion_results.results:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Basic metrics
        metrics['total_results'] = len(fusion_results.results)
        metrics['fusion_time'] = fusion_results.fusion_time
        metrics['total_time'] = fusion_results.total_processing_time
        
        # Coverage metrics
        strategy_coverage = defaultdict(int)
        for result in fusion_results.results:
            for strategy in result.contributing_strategies:
                strategy_coverage[strategy] += 1
        
        metrics['strategy_coverage'] = dict(strategy_coverage)
        metrics['avg_strategies_per_result'] = sum(
            len(result.contributing_strategies) for result in fusion_results.results
        ) / len(fusion_results.results)
        
        # Score distribution
        scores = [result.final_score for result in fusion_results.results]
        if scores:
            metrics['score_mean'] = sum(scores) / len(scores)
            metrics['score_std'] = math.sqrt(
                sum((score - metrics['score_mean']) ** 2 for score in scores) / len(scores)
            )
            metrics['score_range'] = max(scores) - min(scores)
        
        # Ground truth evaluation if provided
        if ground_truth:
            retrieved_docs = [result.doc_id for result in fusion_results.results]
            relevant_retrieved = [doc for doc in retrieved_docs if doc in ground_truth]
            
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0
            recall = len(relevant_retrieved) / len(ground_truth) if ground_truth else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        return metrics
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get statistics about available fusion strategies."""
        return {
            'available_strategies': list(self.fusion_strategies.keys()),
            'strategy_preferences': self.strategy_preferences.copy(),
            'default_limits': {
                'max_results': 100,
                'max_strategies': 10
            }
        }