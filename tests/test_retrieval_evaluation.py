"""
Tests for retrieval evaluation and confidence scoring (Phase 1.1).

This test suite validates the retrieval confidence scoring, evaluation metrics,
and automated benchmarking capabilities.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from datetime import datetime

try:
    from langchain.schema import Document
except ImportError:
    Document = dict

from src.utils.retrieval_evaluation import (
    RetrievalEvaluator,
    SemanticConfidenceCalculator,
    ConfidenceScores,
    ConfidenceLevel,
    RetrievalResult,
    RetrievalEvaluation,
    get_retrieval_evaluator
)


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What are the key concepts in machine learning?"


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents with metadata."""
    docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms. Key concepts include supervised learning, unsupervised learning, and reinforcement learning.",
            metadata={
                "book_id": "ml_book_001",
                "chunk_id": "chunk_001",
                "similarity_score": 0.85,
                "importance_score": 0.9,
                "coherence_score": 0.8,
                "completeness_score": 0.85,
                "retrieval_method": "vector_search",
                "retrieval_source": "enhanced_semantic"
            }
        ),
        Document(
            page_content="Supervised learning uses labeled training data to learn patterns. Common algorithms include linear regression, decision trees, and neural networks.",
            metadata={
                "book_id": "ml_book_001",
                "chunk_id": "chunk_002",
                "similarity_score": 0.75,
                "importance_score": 0.8,
                "coherence_score": 0.9,
                "completeness_score": 0.7,
                "retrieval_method": "vector_search",
                "retrieval_source": "enhanced_semantic"
            }
        ),
        Document(
            page_content="Unsupervised learning finds patterns in data without labeled examples. Clustering and dimensionality reduction are common techniques.",
            metadata={
                "book_id": "ml_book_001",
                "chunk_id": "chunk_003",
                "similarity_score": 0.70,
                "importance_score": 0.75,
                "coherence_score": 0.85,
                "completeness_score": 0.8,
                "retrieval_method": "keyword_search",
                "retrieval_source": "enhanced_semantic"
            }
        )
    ]
    return docs


@pytest.fixture
def sample_response():
    """Sample generated response."""
    return "Machine learning involves several key concepts: supervised learning (using labeled data), unsupervised learning (finding patterns without labels), and reinforcement learning (learning through interaction). These approaches enable algorithms to learn from data and make predictions."


class TestConfidenceScores:
    """Test confidence scoring functionality."""
    
    def test_confidence_scores_creation(self):
        """Test creation of confidence scores object."""
        scores = ConfidenceScores(
            retrieval_confidence=0.8,
            generation_confidence=0.7,
            semantic_confidence=0.9,
            factual_confidence=0.75
        )
        
        assert scores.retrieval_confidence == 0.8
        assert scores.generation_confidence == 0.7
        assert scores.semantic_confidence == 0.9
        assert scores.factual_confidence == 0.75
    
    def test_overall_confidence_calculation(self):
        """Test overall confidence calculation."""
        scores = ConfidenceScores(
            retrieval_confidence=0.8,
            generation_confidence=0.7,
            semantic_confidence=0.9,
            factual_confidence=0.75,
            completeness_score=0.8,
            coherence_score=0.85,
            relevance_score=0.8
        )
        
        overall = scores.calculate_overall_confidence()
        
        # Should be a weighted average
        assert 0.0 <= overall <= 1.0
        assert overall > 0.7  # Should be relatively high given the input scores
        assert scores.overall_confidence == overall
    
    def test_confidence_level_determination(self):
        """Test confidence level classification."""
        # Test very high confidence
        high_scores = ConfidenceScores(
            retrieval_confidence=0.9,
            generation_confidence=0.9,
            semantic_confidence=0.9,
            factual_confidence=0.9
        )
        high_scores.calculate_overall_confidence()
        assert high_scores.confidence_level == ConfidenceLevel.VERY_HIGH
        
        # Test low confidence
        low_scores = ConfidenceScores(
            retrieval_confidence=0.2,
            generation_confidence=0.3,
            semantic_confidence=0.2,
            factual_confidence=0.25
        )
        low_scores.calculate_overall_confidence()
        assert low_scores.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
    
    def test_confidence_scores_serialization(self):
        """Test serialization of confidence scores."""
        scores = ConfidenceScores(
            retrieval_confidence=0.8,
            overall_confidence=0.75,
            confidence_level=ConfidenceLevel.HIGH
        )
        
        scores_dict = scores.to_dict()
        
        assert isinstance(scores_dict, dict)
        assert "retrieval_confidence" in scores_dict
        assert "overall_confidence" in scores_dict
        assert "confidence_level" in scores_dict
        assert scores_dict["confidence_level"] == "high"
        assert "timestamp" in scores_dict


class TestSemanticConfidenceCalculator:
    """Test semantic confidence calculation."""
    
    def test_basic_confidence_calculation(self, sample_query, sample_retrieved_docs):
        """Test basic confidence calculation."""
        calculator = SemanticConfidenceCalculator()
        
        confidence = calculator.calculate_confidence(
            sample_query,
            sample_retrieved_docs
        )
        
        assert isinstance(confidence, ConfidenceScores)
        assert 0.0 <= confidence.retrieval_confidence <= 1.0
        assert 0.0 <= confidence.semantic_confidence <= 1.0
        assert 0.0 <= confidence.overall_confidence <= 1.0
        assert isinstance(confidence.confidence_level, ConfidenceLevel)
    
    def test_confidence_with_response(self, sample_query, sample_retrieved_docs, sample_response):
        """Test confidence calculation with generated response."""
        calculator = SemanticConfidenceCalculator()
        
        confidence = calculator.calculate_confidence(
            sample_query,
            sample_retrieved_docs,
            sample_response
        )
        
        # Should have higher generation confidence with a good response
        assert confidence.generation_confidence > 0.5
        assert confidence.overall_confidence > 0.0
    
    def test_empty_documents_handling(self, sample_query):
        """Test handling of empty document list."""
        calculator = SemanticConfidenceCalculator()
        
        confidence = calculator.calculate_confidence(sample_query, [])
        
        assert confidence.retrieval_confidence == 0.0
        assert confidence.semantic_confidence == 0.0
        assert confidence.confidence_level == ConfidenceLevel.VERY_LOW
    
    def test_retrieval_confidence_calculation(self, sample_query, sample_retrieved_docs):
        """Test retrieval confidence calculation."""
        calculator = SemanticConfidenceCalculator()
        
        retrieval_conf = calculator._calculate_retrieval_confidence(
            sample_query,
            sample_retrieved_docs
        )
        
        assert 0.0 <= retrieval_conf <= 1.0
        # Should be relatively high for good matches
        assert retrieval_conf > 0.3
    
    def test_semantic_confidence_calculation(self, sample_query, sample_retrieved_docs):
        """Test semantic confidence calculation."""
        calculator = SemanticConfidenceCalculator()
        
        semantic_conf = calculator._calculate_semantic_confidence(
            sample_query,
            sample_retrieved_docs
        )
        
        assert 0.0 <= semantic_conf <= 1.0
    
    def test_generation_confidence_calculation(self, sample_query, sample_retrieved_docs, sample_response):
        """Test generation confidence calculation."""
        calculator = SemanticConfidenceCalculator()
        
        generation_conf = calculator._calculate_generation_confidence(
            sample_query,
            sample_retrieved_docs,
            sample_response
        )
        
        assert 0.0 <= generation_conf <= 1.0
        # Should be higher when response uses retrieved content
        assert generation_conf > 0.4


class TestRetrievalResult:
    """Test retrieval result functionality."""
    
    def test_retrieval_result_creation(self, sample_retrieved_docs):
        """Test creation of retrieval result."""
        doc = sample_retrieved_docs[0]
        
        result = RetrievalResult(
            document=doc,
            similarity_score=0.85,
            rank=0,
            source="vector_search",
            relevance_score=0.8,
            importance_score=0.9
        )
        
        assert result.document == doc
        assert result.similarity_score == 0.85
        assert result.rank == 0
        assert result.relevance_score == 0.8
    
    def test_composite_score_calculation(self, sample_retrieved_docs):
        """Test composite score calculation."""
        doc = sample_retrieved_docs[0]
        
        result = RetrievalResult(
            document=doc,
            similarity_score=0.8,
            relevance_score=0.9,
            importance_score=0.85,
            completeness_score=0.75
        )
        
        composite = result.calculate_composite_score()
        
        assert 0.0 <= composite <= 1.0
        # Should be a weighted combination of the scores
        assert composite > 0.7  # Should be relatively high


class TestRetrievalEvaluation:
    """Test retrieval evaluation functionality."""
    
    def test_evaluation_creation(self, sample_query, sample_retrieved_docs):
        """Test creation of retrieval evaluation."""
        results = []
        for i, doc in enumerate(sample_retrieved_docs):
            result = RetrievalResult(
                document=doc,
                similarity_score=doc.metadata.get("similarity_score", 0.0),
                rank=i,
                relevance_score=0.8 - (i * 0.1)  # Decreasing relevance
            )
            results.append(result)
        
        evaluation = RetrievalEvaluation(
            query_id="test_001",
            query_text=sample_query,
            results=results,
            total_results=len(results)
        )
        
        assert evaluation.query_text == sample_query
        assert len(evaluation.results) == 3
        assert evaluation.total_results == 3
    
    def test_precision_at_k_calculation(self, sample_query, sample_retrieved_docs):
        """Test precision@k calculation."""
        results = []
        relevance_scores = [0.8, 0.6, 0.4]  # Decreasing relevance
        
        for i, doc in enumerate(sample_retrieved_docs):
            result = RetrievalResult(
                document=doc,
                rank=i,
                relevance_score=relevance_scores[i]
            )
            results.append(result)
        
        evaluation = RetrievalEvaluation(
            query_id="test_001",
            query_text=sample_query,
            results=results
        )
        
        precision_scores = evaluation.calculate_precision_at_k(relevance_threshold=0.5)
        
        # Should calculate precision for different k values
        assert 1 in precision_scores
        assert 3 in precision_scores
        assert 5 in precision_scores
        
        # P@1 should be 1.0 (first doc is relevant)
        assert precision_scores[1] == 1.0
        # P@3 should be 2/3 (first two docs are relevant)
        assert abs(precision_scores[3] - (2/3)) < 0.01
    
    def test_mrr_calculation(self, sample_query, sample_retrieved_docs):
        """Test Mean Reciprocal Rank calculation."""
        results = []
        relevance_scores = [0.3, 0.8, 0.4]  # First relevant doc at rank 2
        
        for i, doc in enumerate(sample_retrieved_docs):
            result = RetrievalResult(
                document=doc,
                rank=i,
                relevance_score=relevance_scores[i]
            )
            results.append(result)
        
        evaluation = RetrievalEvaluation(
            query_id="test_001",
            query_text=sample_query,
            results=results
        )
        
        mrr = evaluation.calculate_mrr(relevance_threshold=0.5)
        
        # First relevant doc is at rank 2 (index 1), so MRR = 1/2 = 0.5
        assert abs(mrr - 0.5) < 0.01
    
    def test_ndcg_calculation(self, sample_query, sample_retrieved_docs):
        """Test Normalized Discounted Cumulative Gain calculation."""
        results = []
        relevance_scores = [0.9, 0.7, 0.5]
        
        for i, doc in enumerate(sample_retrieved_docs):
            result = RetrievalResult(
                document=doc,
                rank=i,
                relevance_score=relevance_scores[i]
            )
            results.append(result)
        
        evaluation = RetrievalEvaluation(
            query_id="test_001",
            query_text=sample_query,
            results=results
        )
        
        ndcg = evaluation.calculate_ndcg(k=3)
        
        assert 0.0 <= ndcg <= 1.0
        # Should be close to 1.0 since docs are already in optimal order
        assert ndcg > 0.9
    
    def test_evaluation_serialization(self, sample_query, sample_retrieved_docs):
        """Test evaluation serialization."""
        evaluation = RetrievalEvaluation(
            query_id="test_001",
            query_text=sample_query,
            total_results=len(sample_retrieved_docs),
            retrieval_time=0.05
        )
        
        eval_dict = evaluation.to_dict()
        
        assert isinstance(eval_dict, dict)
        assert "query_id" in eval_dict
        assert "query_text" in eval_dict
        assert "total_results" in eval_dict
        assert "retrieval_time" in eval_dict
        assert "precision_at_k" in eval_dict


class TestRetrievalEvaluator:
    """Test main retrieval evaluator functionality."""
    
    def test_evaluator_creation(self):
        """Test creation of retrieval evaluator."""
        evaluator = RetrievalEvaluator()
        
        assert evaluator.confidence_calculator is not None
        assert isinstance(evaluator.evaluation_history, list)
        assert len(evaluator.evaluation_history) == 0
    
    def test_evaluation_execution(self, sample_query, sample_retrieved_docs, sample_response):
        """Test complete evaluation execution."""
        evaluator = RetrievalEvaluator()
        
        evaluation = evaluator.evaluate_retrieval(
            query=sample_query,
            retrieved_docs=sample_retrieved_docs,
            retrieval_time=0.05,
            generated_response=sample_response
        )
        
        assert isinstance(evaluation, RetrievalEvaluation)
        assert evaluation.query_text == sample_query
        assert len(evaluation.results) == len(sample_retrieved_docs)
        assert evaluation.retrieval_time == 0.05
        
        # Should have calculated metrics
        assert len(evaluation.precision_at_k) > 0
        assert evaluation.mean_reciprocal_rank >= 0.0
        assert evaluation.normalized_dcg >= 0.0
        
        # Should have confidence scores
        assert evaluation.overall_confidence.overall_confidence >= 0.0
        assert isinstance(evaluation.overall_confidence.confidence_level, ConfidenceLevel)
        
        # Should be stored in history
        assert len(evaluator.evaluation_history) == 1
    
    def test_document_relevance_calculation(self, sample_query, sample_retrieved_docs):
        """Test document relevance calculation."""
        evaluator = RetrievalEvaluator()
        
        doc = sample_retrieved_docs[0]
        relevance = evaluator._calculate_document_relevance(sample_query, doc)
        
        assert 0.0 <= relevance <= 1.0
        # Should be relatively high for a relevant document
        assert relevance > 0.5
    
    def test_strategy_performance_analysis(self, sample_retrieved_docs):
        """Test strategy performance analysis."""
        evaluator = RetrievalEvaluator()
        
        breakdown = evaluator._analyze_strategy_performance(sample_retrieved_docs)
        
        assert isinstance(breakdown, dict)
        
        # Should have entries for each retrieval method
        assert "vector_search" in breakdown
        
        # Each strategy should have stats
        for strategy, stats in breakdown.items():
            assert "count" in stats
            assert "avg_similarity" in stats
            assert "avg_importance" in stats
            assert stats["count"] > 0
    
    def test_performance_summary(self, sample_query, sample_retrieved_docs):
        """Test performance summary generation."""
        evaluator = RetrievalEvaluator()
        
        # Run a few evaluations
        for i in range(3):
            evaluator.evaluate_retrieval(
                query=f"{sample_query} {i}",
                retrieved_docs=sample_retrieved_docs,
                retrieval_time=0.05 + i * 0.01
            )
        
        summary = evaluator.get_performance_summary()
        
        assert "total_evaluations" in summary
        assert summary["total_evaluations"] == 3
        
        assert "average_metrics" in summary
        metrics = summary["average_metrics"]
        assert "precision_at_1" in metrics
        assert "precision_at_3" in metrics
        assert "mean_reciprocal_rank" in metrics
        assert "overall_confidence" in metrics
        
        assert "confidence_distribution" in summary
    
    def test_empty_evaluations_summary(self):
        """Test performance summary with no evaluations."""
        evaluator = RetrievalEvaluator()
        
        summary = evaluator.get_performance_summary()
        
        assert "error" in summary
        assert "No evaluations available" in summary["error"]


class TestGlobalEvaluator:
    """Test global evaluator instance management."""
    
    def test_get_retrieval_evaluator(self):
        """Test getting global retrieval evaluator instance."""
        evaluator1 = get_retrieval_evaluator()
        evaluator2 = get_retrieval_evaluator()
        
        # Should return the same instance
        assert evaluator1 is evaluator2
        assert isinstance(evaluator1, RetrievalEvaluator)


# Edge cases and error handling tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_query_evaluation(self, sample_retrieved_docs):
        """Test evaluation with empty query."""
        evaluator = RetrievalEvaluator()
        
        evaluation = evaluator.evaluate_retrieval(
            query="",
            retrieved_docs=sample_retrieved_docs
        )
        
        assert isinstance(evaluation, RetrievalEvaluation)
        assert evaluation.query_text == ""
    
    def test_no_retrieved_documents(self):
        """Test evaluation with no retrieved documents."""
        evaluator = RetrievalEvaluator()
        
        evaluation = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_docs=[]
        )
        
        assert evaluation.total_results == 0
        assert len(evaluation.results) == 0
        # Confidence should be very low
        assert evaluation.overall_confidence.confidence_level == ConfidenceLevel.VERY_LOW
    
    def test_documents_without_metadata(self):
        """Test evaluation with documents lacking metadata."""
        docs = [
            Document(
                page_content="Simple document without metadata",
                metadata={}
            )
        ]
        
        evaluator = RetrievalEvaluator()
        
        evaluation = evaluator.evaluate_retrieval(
            query="test query",
            retrieved_docs=docs
        )
        
        # Should handle gracefully with default values
        assert len(evaluation.results) == 1
        result = evaluation.results[0]
        assert result.similarity_score == 0.0  # Default value
        assert result.importance_score == 0.5  # Default value


# Performance tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_document_set_evaluation(self):
        """Test evaluation with large number of documents."""
        # Create 100 test documents
        docs = []
        for i in range(100):
            doc = Document(
                page_content=f"Test document {i} with some content about machine learning and algorithms.",
                metadata={
                    "chunk_id": f"chunk_{i}",
                    "similarity_score": 0.8 - (i * 0.001),  # Decreasing similarity
                    "importance_score": 0.7 + (i % 3) * 0.1,
                    "retrieval_method": "vector_search" if i % 2 == 0 else "keyword_search"
                }
            )
            docs.append(doc)
        
        evaluator = RetrievalEvaluator()
        
        import time
        start_time = time.time()
        
        evaluation = evaluator.evaluate_retrieval(
            query="machine learning algorithms",
            retrieved_docs=docs
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (under 5 seconds)
        assert processing_time < 5.0
        
        # Should handle all documents
        assert evaluation.total_results == 100
        assert len(evaluation.results) == 100
        
        # Should have calculated all metrics
        assert len(evaluation.precision_at_k) > 0
        assert evaluation.mean_reciprocal_rank >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])