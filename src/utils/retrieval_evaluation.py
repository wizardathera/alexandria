"""
Retrieval confidence scoring and evaluation logging for Phase 1.1.

This module implements comprehensive evaluation of retrieval quality including
confidence scoring, pipeline-level evaluation, and automated benchmarking.
"""

import json
import math
import statistics
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

try:
    from langchain.schema import Document
    import numpy as np
except ImportError:
    Document = dict
    np = None

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for retrieval quality."""
    VERY_LOW = "very_low"     # 0.0 - 0.2
    LOW = "low"               # 0.2 - 0.4  
    MEDIUM = "medium"         # 0.4 - 0.6
    HIGH = "high"             # 0.6 - 0.8
    VERY_HIGH = "very_high"   # 0.8 - 1.0


class RetrievalMetric(Enum):
    """Types of retrieval metrics."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCH = "keyword_match"
    CONTEXTUAL_RELEVANCE = "contextual_relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    FACTUAL_ACCURACY = "factual_accuracy"


@dataclass
class ConfidenceScores:
    """Multi-dimensional confidence scores for retrieval results."""
    
    # Core confidence metrics
    retrieval_confidence: float = 0.5    # Quality of retrieved context
    generation_confidence: float = 0.5   # Model certainty in response
    semantic_confidence: float = 0.5     # Semantic alignment with query
    factual_confidence: float = 0.5      # Factual consistency check
    
    # Additional quality metrics
    completeness_score: float = 0.5      # Coverage of important information
    coherence_score: float = 0.5         # Internal logical consistency
    relevance_score: float = 0.5         # Direct relevance to query
    
    # Aggregate scores
    overall_confidence: float = 0.5      # Combined confidence score
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    
    # Metadata
    calculation_method: str = "weighted_average"
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall confidence score using weighted average."""
        if weights is None:
            weights = {
                "retrieval_confidence": 0.25,
                "generation_confidence": 0.20,
                "semantic_confidence": 0.20,
                "factual_confidence": 0.15,
                "completeness_score": 0.10,
                "coherence_score": 0.05,
                "relevance_score": 0.05
            }
        
        weighted_sum = (
            self.retrieval_confidence * weights.get("retrieval_confidence", 0) +
            self.generation_confidence * weights.get("generation_confidence", 0) +
            self.semantic_confidence * weights.get("semantic_confidence", 0) +
            self.factual_confidence * weights.get("factual_confidence", 0) +
            self.completeness_score * weights.get("completeness_score", 0) +
            self.coherence_score * weights.get("coherence_score", 0) +
            self.relevance_score * weights.get("relevance_score", 0)
        )
        
        self.overall_confidence = max(0.0, min(1.0, weighted_sum))
        self.confidence_level = self._determine_confidence_level(self.overall_confidence)
        return self.overall_confidence
    
    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage."""
        result = asdict(self)
        result["confidence_level"] = self.confidence_level.value
        result["timestamp"] = self.timestamp.isoformat()
        return result


@dataclass
class RetrievalResult:
    """Individual retrieval result with metadata and scoring."""
    
    # Core result data
    document: Document
    similarity_score: float = 0.0
    rank: int = 0
    source: str = "unknown"  # Which retrieval strategy found this
    
    # Quality metrics
    relevance_score: float = 0.0
    importance_score: float = 0.0
    completeness_score: float = 0.0
    
    # Retrieval metadata
    retrieval_method: str = "vector_search"
    retrieval_time: float = 0.0
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence assessment
    confidence_scores: Optional[ConfidenceScores] = None
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite relevance score."""
        if weights is None:
            weights = {
                "similarity_score": 0.4,
                "relevance_score": 0.3,
                "importance_score": 0.2,
                "completeness_score": 0.1
            }
        
        composite = (
            self.similarity_score * weights.get("similarity_score", 0) +
            self.relevance_score * weights.get("relevance_score", 0) +
            self.importance_score * weights.get("importance_score", 0) +
            self.completeness_score * weights.get("completeness_score", 0)
        )
        
        return max(0.0, min(1.0, composite))


@dataclass
class RetrievalEvaluation:
    """Complete evaluation of a retrieval operation."""
    
    # Query information
    query_id: str
    query_text: str
    query_timestamp: datetime = field(default_factory=datetime.now)
    
    # Retrieved results
    results: List[RetrievalResult] = field(default_factory=list)
    total_results: int = 0
    
    # Performance metrics
    retrieval_time: float = 0.0
    processing_time: float = 0.0
    
    # Quality metrics
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # P@1, P@3, P@5
    mean_reciprocal_rank: float = 0.0
    normalized_dcg: float = 0.0
    
    # Confidence assessment
    overall_confidence: ConfidenceScores = field(default_factory=ConfidenceScores)
    
    # Strategy performance
    strategy_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def calculate_precision_at_k(self, relevance_threshold: float = 0.5) -> Dict[int, float]:
        """Calculate precision at different k values."""
        k_values = [1, 3, 5, 10]
        precision_scores = {}
        
        for k in k_values:
            if k > len(self.results):
                k = len(self.results)
            
            if k == 0:
                precision_scores[k] = 0.0
                continue
            
            relevant_count = 0
            for i in range(k):
                if self.results[i].relevance_score >= relevance_threshold:
                    relevant_count += 1
            
            precision_scores[k] = relevant_count / k
        
        self.precision_at_k = precision_scores
        return precision_scores
    
    def calculate_mrr(self, relevance_threshold: float = 0.5) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, result in enumerate(self.results):
            if result.relevance_score >= relevance_threshold:
                self.mean_reciprocal_rank = 1.0 / (i + 1)
                return self.mean_reciprocal_rank
        
        self.mean_reciprocal_rank = 0.0
        return 0.0
    
    def calculate_ndcg(self, k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not self.results:
            return 0.0
        
        # Calculate DCG
        dcg = 0.0
        for i in range(min(k, len(self.results))):
            relevance = self.results[i].relevance_score
            dcg += relevance / math.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted([r.relevance_score for r in self.results], reverse=True)
        idcg = 0.0
        for i in range(min(k, len(ideal_relevances))):
            idcg += ideal_relevances[i] / math.log2(i + 2)
        
        # Calculate NDCG
        if idcg > 0:
            self.normalized_dcg = dcg / idcg
        else:
            self.normalized_dcg = 0.0
        
        return self.normalized_dcg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation to dictionary for logging."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_timestamp": self.query_timestamp.isoformat(),
            "total_results": self.total_results,
            "retrieval_time": self.retrieval_time,
            "processing_time": self.processing_time,
            "precision_at_k": self.precision_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "normalized_dcg": self.normalized_dcg,
            "overall_confidence": self.overall_confidence.to_dict(),
            "strategy_breakdown": self.strategy_breakdown
        }


class ConfidenceCalculator(ABC):
    """Abstract base class for confidence calculation strategies."""
    
    @abstractmethod
    def calculate_confidence(
        self,
        query: str,
        retrieved_docs: List[Document],
        generated_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScores:
        """Calculate confidence scores for retrieval results."""
        pass


class SemanticConfidenceCalculator(ConfidenceCalculator):
    """Calculate confidence based on semantic similarity and relevance."""
    
    def __init__(self):
        """Initialize the semantic confidence calculator."""
        self.keyword_importance_weights = {
            "high": ["important", "critical", "essential", "key", "main"],
            "medium": ["relevant", "useful", "significant", "notable"],
            "low": ["possibly", "might", "perhaps", "sometimes"]
        }
    
    def calculate_confidence(
        self,
        query: str,
        retrieved_docs: List[Document],
        generated_response: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScores:
        """Calculate semantic confidence scores."""
        
        if not retrieved_docs:
            return ConfidenceScores(
                retrieval_confidence=0.0,
                semantic_confidence=0.0,
                overall_confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW
            )
        
        # Calculate retrieval confidence
        retrieval_conf = self._calculate_retrieval_confidence(query, retrieved_docs)
        
        # Calculate semantic confidence
        semantic_conf = self._calculate_semantic_confidence(query, retrieved_docs)
        
        # Calculate generation confidence if response provided
        generation_conf = 0.7  # Default when no response provided
        if generated_response:
            generation_conf = self._calculate_generation_confidence(
                query, retrieved_docs, generated_response
            )
        
        # Calculate factual confidence
        factual_conf = self._calculate_factual_confidence(retrieved_docs, generated_response)
        
        # Calculate completeness and coherence
        completeness = self._calculate_completeness_score(query, retrieved_docs)
        coherence = self._calculate_coherence_score(retrieved_docs)
        relevance = self._calculate_relevance_score(query, retrieved_docs)
        
        # Create confidence scores object
        confidence = ConfidenceScores(
            retrieval_confidence=retrieval_conf,
            generation_confidence=generation_conf,
            semantic_confidence=semantic_conf,
            factual_confidence=factual_conf,
            completeness_score=completeness,
            coherence_score=coherence,
            relevance_score=relevance
        )
        
        # Calculate overall confidence
        confidence.calculate_overall_confidence()
        
        return confidence
    
    def _calculate_retrieval_confidence(self, query: str, docs: List[Document]) -> float:
        """Calculate confidence in retrieved documents."""
        if not docs:
            return 0.0
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Calculate keyword overlap for each document
        overlaps = []
        for doc in docs[:5]:  # Consider top 5 documents
            doc_text = doc.page_content.lower()
            doc_words = set(doc_text.split())
            
            # Calculate keyword overlap
            overlap = len(query_words & doc_words) / max(len(query_words), 1)
            overlaps.append(overlap)
        
        # Calculate average overlap
        avg_overlap = statistics.mean(overlaps) if overlaps else 0.0
        
        # Boost confidence if documents have good metadata scores
        metadata_boost = 0.0
        for doc in docs[:3]:
            importance = doc.metadata.get("importance_score", 0.5)
            metadata_boost += importance
        
        metadata_boost = metadata_boost / min(3, len(docs))
        
        # Combine overlap and metadata boost
        confidence = (avg_overlap * 0.7) + (metadata_boost * 0.3)
        return max(0.0, min(1.0, confidence))
    
    def _calculate_semantic_confidence(self, query: str, docs: List[Document]) -> float:
        """Calculate semantic alignment confidence."""
        if not docs:
            return 0.0
        
        # Simple semantic similarity based on shared concepts
        query_lower = query.lower()
        
        semantic_scores = []
        for doc in docs[:3]:
            doc_text = doc.page_content.lower()
            
            # Check for conceptual overlap
            conceptual_score = 0.0
            
            # Look for important concepts in both query and document
            for level, keywords in self.keyword_importance_weights.items():
                weight = {"high": 1.0, "medium": 0.7, "low": 0.3}[level]
                
                for keyword in keywords:
                    if keyword in query_lower and keyword in doc_text:
                        conceptual_score += weight
            
            # Normalize by document length (favor concise, relevant documents)
            doc_length = len(doc.page_content.split())
            if doc_length > 0:
                conceptual_score = conceptual_score / math.log(doc_length + 1)
            
            semantic_scores.append(min(1.0, conceptual_score))
        
        return statistics.mean(semantic_scores) if semantic_scores else 0.5
    
    def _calculate_generation_confidence(
        self, 
        query: str, 
        docs: List[Document], 
        response: str
    ) -> float:
        """Calculate confidence in generated response."""
        if not response or not docs:
            return 0.5
        
        response_lower = response.lower()
        
        # Check if response uses information from retrieved documents
        doc_overlap = 0.0
        total_content = ""
        
        for doc in docs[:3]:
            total_content += " " + doc.page_content.lower()
        
        # Simple overlap calculation
        response_words = set(response_lower.split())
        doc_words = set(total_content.split())
        
        if response_words:
            overlap = len(response_words & doc_words) / len(response_words)
            doc_overlap = min(1.0, overlap * 2)  # Boost the score
        
        # Check for uncertainty markers in response
        uncertainty_markers = ["might", "could", "possibly", "perhaps", "unclear", "uncertain"]
        uncertainty_penalty = 0.0
        for marker in uncertainty_markers:
            if marker in response_lower:
                uncertainty_penalty += 0.1
        
        confidence = doc_overlap - uncertainty_penalty
        return max(0.0, min(1.0, confidence))
    
    def _calculate_factual_confidence(
        self, 
        docs: List[Document], 
        response: Optional[str] = None
    ) -> float:
        """Calculate factual consistency confidence."""
        # This is a simplified version - could be enhanced with fact-checking models
        if not docs:
            return 0.5
        
        # Check document quality indicators
        quality_score = 0.0
        doc_count = min(3, len(docs))
        
        for doc in docs[:doc_count]:
            # Use metadata quality scores if available
            coherence = doc.metadata.get("coherence_score", 0.5)
            completeness = doc.metadata.get("completeness_score", 0.5)
            importance = doc.metadata.get("importance_score", 0.5)
            
            doc_quality = (coherence + completeness + importance) / 3
            quality_score += doc_quality
        
        if doc_count > 0:
            quality_score = quality_score / doc_count
        
        return max(0.0, min(1.0, quality_score))
    
    def _calculate_completeness_score(self, query: str, docs: List[Document]) -> float:
        """Calculate how completely the documents answer the query."""
        if not docs:
            return 0.0
        
        # Simple heuristic: longer, more detailed responses tend to be more complete
        total_length = sum(len(doc.page_content) for doc in docs[:3])
        
        # Normalize by expected response length (rough heuristic)
        expected_length = len(query.split()) * 50  # Rough estimate
        completeness = min(1.0, total_length / max(expected_length, 100))
        
        # Boost if documents have good completeness metadata
        metadata_completeness = 0.0
        for doc in docs[:3]:
            metadata_completeness += doc.metadata.get("completeness_score", 0.5)
        
        metadata_completeness = metadata_completeness / min(3, len(docs))
        
        # Combine heuristic and metadata
        final_score = (completeness * 0.4) + (metadata_completeness * 0.6)
        return max(0.0, min(1.0, final_score))
    
    def _calculate_coherence_score(self, docs: List[Document]) -> float:
        """Calculate internal coherence of retrieved documents."""
        if not docs:
            return 0.0
        
        # Use metadata coherence scores if available
        coherence_scores = []
        for doc in docs[:3]:
            coherence = doc.metadata.get("coherence_score", 0.5)
            coherence_scores.append(coherence)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_relevance_score(self, query: str, docs: List[Document]) -> float:
        """Calculate overall relevance score."""
        if not docs:
            return 0.0
        
        query_words = set(query.lower().split())
        relevance_scores = []
        
        for doc in docs[:5]:
            doc_words = set(doc.page_content.lower().split())
            
            # Calculate word overlap
            overlap = len(query_words & doc_words) / max(len(query_words), 1)
            
            # Boost with importance score
            importance = doc.metadata.get("importance_score", 0.5)
            combined_score = (overlap * 0.7) + (importance * 0.3)
            
            relevance_scores.append(min(1.0, combined_score))
        
        return statistics.mean(relevance_scores) if relevance_scores else 0.5


class RetrievalEvaluator:
    """Main class for evaluating retrieval operations."""
    
    def __init__(self, confidence_calculator: Optional[ConfidenceCalculator] = None):
        """Initialize the retrieval evaluator."""
        self.confidence_calculator = confidence_calculator or SemanticConfidenceCalculator()
        self.settings = get_settings()
        self.evaluation_history: List[RetrievalEvaluation] = []
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        retrieval_time: float = 0.0,
        generated_response: Optional[str] = None,
        ground_truth: Optional[List[Document]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RetrievalEvaluation:
        """
        Comprehensive evaluation of a retrieval operation.
        
        Args:
            query: The search query
            retrieved_docs: Documents returned by retrieval
            retrieval_time: Time taken for retrieval
            generated_response: Generated response (optional)
            ground_truth: Known relevant documents (optional)
            context: Additional context information
            
        Returns:
            RetrievalEvaluation: Complete evaluation results
        """
        query_id = str(uuid.uuid4())
        
        # Create retrieval results
        results = []
        for i, doc in enumerate(retrieved_docs):
            result = RetrievalResult(
                document=doc,
                similarity_score=doc.metadata.get("similarity_score", 0.0),
                rank=i,
                source=doc.metadata.get("retrieval_source", "unknown"),
                relevance_score=self._calculate_document_relevance(query, doc),
                importance_score=doc.metadata.get("importance_score", 0.5),
                completeness_score=doc.metadata.get("completeness_score", 0.5),
                retrieval_method=doc.metadata.get("retrieval_method", "vector_search"),
                chunk_metadata=doc.metadata
            )
            results.append(result)
        
        # Create evaluation object
        evaluation = RetrievalEvaluation(
            query_id=query_id,
            query_text=query,
            results=results,
            total_results=len(retrieved_docs),
            retrieval_time=retrieval_time
        )
        
        # Calculate quality metrics
        evaluation.calculate_precision_at_k()
        evaluation.calculate_mrr()
        evaluation.calculate_ndcg()
        
        # Calculate confidence scores
        evaluation.overall_confidence = self.confidence_calculator.calculate_confidence(
            query, retrieved_docs, generated_response, context
        )
        
        # Analyze strategy performance
        evaluation.strategy_breakdown = self._analyze_strategy_performance(retrieved_docs)
        
        # Store evaluation
        self.evaluation_history.append(evaluation)
        
        # Log evaluation results
        self._log_evaluation(evaluation)
        
        return evaluation
    
    def _calculate_document_relevance(self, query: str, doc: Document) -> float:
        """Calculate relevance score for a single document."""
        query_words = set(query.lower().split())
        doc_words = set(doc.page_content.lower().split())
        
        # Keyword overlap
        overlap = len(query_words & doc_words) / max(len(query_words), 1)
        
        # Boost with metadata scores
        importance = doc.metadata.get("importance_score", 0.5)
        coherence = doc.metadata.get("coherence_score", 0.5)
        
        # Combine scores
        relevance = (overlap * 0.6) + (importance * 0.3) + (coherence * 0.1)
        return max(0.0, min(1.0, relevance))
    
    def _analyze_strategy_performance(self, docs: List[Document]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by retrieval strategy."""
        strategy_stats = {}
        
        for doc in docs:
            strategy = doc.metadata.get("retrieval_method", "unknown")
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "avg_similarity": 0.0,
                    "avg_importance": 0.0,
                    "similarity_scores": [],
                    "importance_scores": []
                }
            
            stats = strategy_stats[strategy]
            stats["count"] += 1
            
            similarity = doc.metadata.get("similarity_score", 0.0)
            importance = doc.metadata.get("importance_score", 0.5)
            
            stats["similarity_scores"].append(similarity)
            stats["importance_scores"].append(importance)
        
        # Calculate averages
        for strategy, stats in strategy_stats.items():
            if stats["similarity_scores"]:
                stats["avg_similarity"] = statistics.mean(stats["similarity_scores"])
            if stats["importance_scores"]:
                stats["avg_importance"] = statistics.mean(stats["importance_scores"])
            
            # Remove individual scores to save space
            del stats["similarity_scores"]
            del stats["importance_scores"]
        
        return strategy_stats
    
    def _log_evaluation(self, evaluation: RetrievalEvaluation):
        """Log evaluation results for analysis."""
        log_data = {
            "event_type": "retrieval_evaluation",
            "query_id": evaluation.query_id,
            "query_length": len(evaluation.query_text.split()),
            "total_results": evaluation.total_results,
            "retrieval_time": evaluation.retrieval_time,
            "precision_at_1": evaluation.precision_at_k.get(1, 0.0),
            "precision_at_3": evaluation.precision_at_k.get(3, 0.0),
            "precision_at_5": evaluation.precision_at_k.get(5, 0.0),
            "mrr": evaluation.mean_reciprocal_rank,
            "ndcg": evaluation.normalized_dcg,
            "overall_confidence": evaluation.overall_confidence.overall_confidence,
            "confidence_level": evaluation.overall_confidence.confidence_level.value,
            "strategy_count": len(evaluation.strategy_breakdown),
            "timestamp": evaluation.query_timestamp.isoformat()
        }
        
        logger.info("Retrieval evaluation completed", extra=log_data)
    
    def get_performance_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get performance summary of recent evaluations."""
        evaluations = self.evaluation_history
        if last_n:
            evaluations = evaluations[-last_n:]
        
        if not evaluations:
            return {"error": "No evaluations available"}
        
        # Calculate aggregate metrics
        avg_precision_1 = statistics.mean([e.precision_at_k.get(1, 0.0) for e in evaluations])
        avg_precision_3 = statistics.mean([e.precision_at_k.get(3, 0.0) for e in evaluations])
        avg_precision_5 = statistics.mean([e.precision_at_k.get(5, 0.0) for e in evaluations])
        avg_mrr = statistics.mean([e.mean_reciprocal_rank for e in evaluations])
        avg_ndcg = statistics.mean([e.normalized_dcg for e in evaluations])
        avg_confidence = statistics.mean([e.overall_confidence.overall_confidence for e in evaluations])
        avg_retrieval_time = statistics.mean([e.retrieval_time for e in evaluations])
        
        # Confidence level distribution
        confidence_distribution = {}
        for evaluation in evaluations:
            level = evaluation.overall_confidence.confidence_level.value
            confidence_distribution[level] = confidence_distribution.get(level, 0) + 1
        
        return {
            "total_evaluations": len(evaluations),
            "time_range": {
                "start": evaluations[0].query_timestamp.isoformat(),
                "end": evaluations[-1].query_timestamp.isoformat()
            },
            "average_metrics": {
                "precision_at_1": avg_precision_1,
                "precision_at_3": avg_precision_3,
                "precision_at_5": avg_precision_5,
                "mean_reciprocal_rank": avg_mrr,
                "normalized_dcg": avg_ndcg,
                "overall_confidence": avg_confidence,
                "retrieval_time": avg_retrieval_time
            },
            "confidence_distribution": confidence_distribution
        }


# Global evaluator instance
_retrieval_evaluator: Optional[RetrievalEvaluator] = None


def get_retrieval_evaluator() -> RetrievalEvaluator:
    """Get the global retrieval evaluator instance."""
    global _retrieval_evaluator
    
    if _retrieval_evaluator is None:
        _retrieval_evaluator = RetrievalEvaluator()
    
    return _retrieval_evaluator