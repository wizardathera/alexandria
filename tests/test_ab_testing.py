"""
Tests for A/B testing framework for retrieval quality comparison (Phase 1.1).

This test suite validates the A/B testing capabilities including experiment
creation, traffic allocation, statistical analysis, and automated recommendations.
"""

import pytest
import uuid
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from datetime import datetime, timedelta

try:
    from langchain.schema import Document
except ImportError:
    Document = dict

from src.utils.ab_testing import (
    ABTestingFramework,
    ABTestExperiment,
    TestVariant,
    TestMetrics,
    TestResult,
    TestType,
    TestStatus,
    StatisticalSignificance,
    StatisticalAnalyzer,
    get_ab_testing_framework,
    create_chunking_strategy_test,
    create_retrieval_method_test
)
from src.utils.retrieval_evaluation import RetrievalEvaluation, ConfidenceScores, ConfidenceLevel


@pytest.fixture
def sample_test_variant_a():
    """Sample test variant A configuration."""
    return TestVariant(
        variant_id="variant_a",
        name="Enhanced Semantic Chunking",
        description="Using enhanced semantic chunking with improved metadata",
        chunking_strategy="enhanced_semantic",
        chunking_config={"chunk_size": 1000, "overlap": 200},
        traffic_allocation=0.5
    )


@pytest.fixture
def sample_test_variant_b():
    """Sample test variant B configuration."""
    return TestVariant(
        variant_id="variant_b", 
        name="Recursive Chunking",
        description="Using traditional recursive character chunking",
        chunking_strategy="recursive",
        chunking_config={"chunk_size": 1000, "overlap": 200},
        traffic_allocation=0.5
    )


@pytest.fixture
def sample_test_metrics():
    """Sample test metrics for evaluation."""
    return [
        TestMetrics(
            precision_at_1=0.8,
            precision_at_3=0.75,
            precision_at_5=0.7,
            mean_reciprocal_rank=0.85,
            normalized_dcg=0.8,
            retrieval_time=0.05,
            processing_time=0.1,
            overall_confidence=0.8,
            semantic_confidence=0.85,
            factual_confidence=0.75,
            sample_count=1
        ),
        TestMetrics(
            precision_at_1=0.9,
            precision_at_3=0.8,
            precision_at_5=0.75,
            mean_reciprocal_rank=0.9,
            normalized_dcg=0.85,
            retrieval_time=0.04,
            processing_time=0.09,
            overall_confidence=0.85,
            semantic_confidence=0.9,
            factual_confidence=0.8,
            sample_count=1
        )
    ]


@pytest.fixture
def sample_retrieval_evaluation():
    """Sample retrieval evaluation for testing."""
    confidence = ConfidenceScores(
        retrieval_confidence=0.8,
        generation_confidence=0.75,
        semantic_confidence=0.85,
        factual_confidence=0.8,
        overall_confidence=0.8,
        confidence_level=ConfidenceLevel.HIGH
    )
    
    evaluation = RetrievalEvaluation(
        query_id="test_001",
        query_text="What are the key concepts in machine learning?",
        total_results=3,
        retrieval_time=0.05,
        overall_confidence=confidence
    )
    
    # Set precision metrics
    evaluation.precision_at_k = {1: 0.8, 3: 0.75, 5: 0.7}
    evaluation.mean_reciprocal_rank = 0.85
    evaluation.normalized_dcg = 0.8
    
    return evaluation


class TestTestMetrics:
    """Test TestMetrics functionality."""
    
    def test_metrics_creation(self):
        """Test creation of test metrics object."""
        metrics = TestMetrics(
            precision_at_1=0.8,
            precision_at_3=0.75,
            mean_reciprocal_rank=0.85,
            overall_confidence=0.8
        )
        
        assert metrics.precision_at_1 == 0.8
        assert metrics.precision_at_3 == 0.75
        assert metrics.mean_reciprocal_rank == 0.85
        assert metrics.overall_confidence == 0.8
    
    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        metrics = TestMetrics(
            precision_at_3=0.8,
            mean_reciprocal_rank=0.85,
            normalized_dcg=0.8,
            overall_confidence=0.75,
            retrieval_time=0.05,
            processing_time=0.1
        )
        
        composite = metrics.calculate_composite_score()
        
        assert 0.0 <= composite <= 1.0
        assert composite > 0.5  # Should be relatively high given good metrics
    
    def test_metrics_serialization(self):
        """Test metrics serialization to dictionary."""
        metrics = TestMetrics(
            precision_at_1=0.8,
            overall_confidence=0.75,
            sample_count=10
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert "precision_at_1" in metrics_dict
        assert "overall_confidence" in metrics_dict
        assert "composite_score" in metrics_dict
        assert "timestamp" in metrics_dict


class TestTestVariant:
    """Test TestVariant functionality."""
    
    def test_variant_creation(self, sample_test_variant_a):
        """Test creation of test variant."""
        variant = sample_test_variant_a
        
        assert variant.variant_id == "variant_a"
        assert variant.name == "Enhanced Semantic Chunking"
        assert variant.chunking_strategy == "enhanced_semantic"
        assert variant.traffic_allocation == 0.5
    
    def test_variant_serialization(self, sample_test_variant_a):
        """Test variant serialization."""
        variant_dict = sample_test_variant_a.to_dict()
        
        assert isinstance(variant_dict, dict)
        assert "variant_id" in variant_dict
        assert "chunking_strategy" in variant_dict
        assert "traffic_allocation" in variant_dict


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""
    
    def test_t_test_calculation(self):
        """Test t-test calculation."""
        analyzer = StatisticalAnalyzer()
        
        samples_a = [0.8, 0.85, 0.75, 0.9, 0.82]
        samples_b = [0.7, 0.72, 0.68, 0.75, 0.71]
        
        t_stat, p_value = analyzer.calculate_t_test(samples_a, samples_b)
        
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
        assert t_stat > 0  # samples_a has higher mean
    
    def test_t_test_empty_samples(self):
        """Test t-test with empty samples."""
        analyzer = StatisticalAnalyzer()
        
        t_stat, p_value = analyzer.calculate_t_test([], [0.5, 0.6])
        
        assert t_stat == 0.0
        assert p_value == 1.0
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer()
        
        samples = [0.8, 0.85, 0.75, 0.9, 0.82, 0.78, 0.88]
        
        ci_low, ci_high = analyzer.calculate_confidence_interval(samples)
        
        assert ci_low < ci_high
        # Confidence interval should contain the sample mean
        import statistics
        sample_mean = statistics.mean(samples)
        assert ci_low <= sample_mean <= ci_high
    
    def test_effect_size_calculation(self):
        """Test effect size calculation."""
        analyzer = StatisticalAnalyzer()
        
        samples_a = [0.9, 0.95, 0.85, 0.92, 0.88]
        samples_b = [0.6, 0.65, 0.58, 0.62, 0.61]
        
        effect_size = analyzer.calculate_effect_size(samples_a, samples_b)
        
        assert effect_size > 0
        assert effect_size > 1.0  # Large effect size expected
    
    def test_significance_determination(self):
        """Test statistical significance determination."""
        analyzer = StatisticalAnalyzer()
        
        # Test different p-values
        assert analyzer.determine_significance(0.0005) == StatisticalSignificance.HIGHLY_SIGNIFICANT
        assert analyzer.determine_significance(0.005) == StatisticalSignificance.SIGNIFICANT
        assert analyzer.determine_significance(0.03) == StatisticalSignificance.MARGINALLY_SIGNIFICANT
        assert analyzer.determine_significance(0.1) == StatisticalSignificance.NOT_SIGNIFICANT


class TestABTestExperiment:
    """Test ABTestExperiment functionality."""
    
    def test_experiment_creation(self, sample_test_variant_a, sample_test_variant_b):
        """Test creation of A/B test experiment."""
        experiment = ABTestExperiment(
            test_id="test_001",
            name="Chunking Strategy Comparison",
            description="Comparing enhanced vs recursive chunking",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b,
            target_sample_size=100
        )
        
        assert experiment.test_id == "test_001"
        assert experiment.name == "Chunking Strategy Comparison"
        assert experiment.test_type == TestType.CHUNKING_STRATEGY
        assert experiment.status == TestStatus.PLANNING
        assert experiment.target_sample_size == 100
    
    def test_sample_size_tracking(self, sample_test_variant_a, sample_test_variant_b, sample_test_metrics):
        """Test sample size tracking."""
        experiment = ABTestExperiment(
            test_id="test_001",
            name="Test",
            description="Test", 
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        # Add some samples
        experiment.variant_a_samples.extend(sample_test_metrics)
        experiment.variant_b_samples.append(sample_test_metrics[0])
        
        a_count, b_count = experiment.get_current_sample_size()
        
        assert a_count == 2
        assert b_count == 1
    
    def test_ready_for_analysis(self, sample_test_variant_a, sample_test_variant_b, sample_test_metrics):
        """Test readiness for analysis checking."""
        experiment = ABTestExperiment(
            test_id="test_001",
            name="Test",
            description="Test",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b,
            target_sample_size=40
        )
        
        # Not ready with insufficient samples
        assert not experiment.is_ready_for_analysis()
        
        # Add minimum samples (25% of target = 10 each)
        for _ in range(15):
            experiment.variant_a_samples.append(sample_test_metrics[0])
            experiment.variant_b_samples.append(sample_test_metrics[1])
        
        # Should be ready now
        assert experiment.is_ready_for_analysis()


class TestABTestingFramework:
    """Test main A/B testing framework functionality."""
    
    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = ABTestingFramework()
        
        assert framework.retrieval_evaluator is not None
        assert framework.chunking_benchmark is not None
        assert isinstance(framework.active_experiments, dict)
        assert isinstance(framework.completed_experiments, dict)
        assert len(framework.active_experiments) == 0
    
    def test_experiment_creation(self, sample_test_variant_a, sample_test_variant_b):
        """Test experiment creation."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Test Experiment",
            description="Testing chunking strategies",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b,
            target_sample_size=50
        )
        
        assert isinstance(test_id, str)
        assert test_id in framework.active_experiments
        
        experiment = framework.active_experiments[test_id]
        assert experiment.name == "Test Experiment"
        assert experiment.test_type == TestType.CHUNKING_STRATEGY
        assert experiment.status == TestStatus.PLANNING
    
    def test_experiment_lifecycle(self, sample_test_variant_a, sample_test_variant_b):
        """Test experiment start/stop lifecycle."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Lifecycle Test",
            description="Testing experiment lifecycle",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        # Start experiment
        success = framework.start_experiment(test_id)
        assert success
        
        experiment = framework.active_experiments[test_id]
        assert experiment.status == TestStatus.ACTIVE
        assert experiment.start_time is not None
        
        # Stop experiment
        success = framework.stop_experiment(test_id)
        assert success
        
        assert experiment.status == TestStatus.STOPPED
        assert experiment.end_time is not None
    
    def test_variant_allocation(self, sample_test_variant_a, sample_test_variant_b):
        """Test variant allocation logic."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Allocation Test",
            description="Testing variant allocation",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        framework.start_experiment(test_id)
        
        # Test consistent allocation with user ID
        user_id = "user_123"
        allocation1 = framework.allocate_variant(test_id, user_id)
        allocation2 = framework.allocate_variant(test_id, user_id)
        
        assert allocation1 == allocation2  # Should be consistent
        assert allocation1 in ["variant_a", "variant_b"]
        
        # Test random allocation
        allocations = []
        for _ in range(100):
            allocation = framework.allocate_variant(test_id)
            allocations.append(allocation)
        
        # Should have both variants represented
        assert "variant_a" in allocations
        assert "variant_b" in allocations
    
    def test_evaluation_recording(
        self, 
        sample_test_variant_a, 
        sample_test_variant_b, 
        sample_retrieval_evaluation
    ):
        """Test recording evaluation results."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Recording Test",
            description="Testing evaluation recording",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        framework.start_experiment(test_id)
        
        # Record evaluation for variant A
        success = framework.record_evaluation(
            test_id=test_id,
            variant_id="variant_a",
            query="test query",
            retrieved_docs=[],
            evaluation=sample_retrieval_evaluation,
            processing_time=0.1,
            memory_usage=50.0
        )
        
        assert success
        
        experiment = framework.active_experiments[test_id]
        assert len(experiment.variant_a_samples) == 1
        assert len(experiment.variant_b_samples) == 0
        
        # Check recorded metrics
        metrics = experiment.variant_a_samples[0]
        assert metrics.precision_at_1 == 0.8
        assert metrics.precision_at_3 == 0.75
        assert metrics.processing_time == 0.1
        assert metrics.memory_usage == 50.0
    
    def test_experiment_analysis(
        self,
        sample_test_variant_a,
        sample_test_variant_b,
        sample_retrieval_evaluation
    ):
        """Test experiment statistical analysis."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Analysis Test",
            description="Testing statistical analysis",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b,
            target_sample_size=40
        )
        
        framework.start_experiment(test_id)
        
        # Add samples to both variants (variant A performs better)
        for i in range(15):
            # Variant A - better performance
            eval_a = sample_retrieval_evaluation
            eval_a.precision_at_k = {1: 0.85 + i*0.01, 3: 0.8 + i*0.01, 5: 0.75}
            eval_a.mean_reciprocal_rank = 0.85 + i*0.01
            eval_a.normalized_dcg = 0.8 + i*0.01
            
            framework.record_evaluation(
                test_id=test_id,
                variant_id="variant_a",
                query=f"test query {i}",
                retrieved_docs=[],
                evaluation=eval_a
            )
            
            # Variant B - worse performance  
            eval_b = sample_retrieval_evaluation
            eval_b.precision_at_k = {1: 0.7 - i*0.005, 3: 0.65 - i*0.005, 5: 0.6}
            eval_b.mean_reciprocal_rank = 0.7 - i*0.005
            eval_b.normalized_dcg = 0.65 - i*0.005
            
            framework.record_evaluation(
                test_id=test_id,
                variant_id="variant_b",
                query=f"test query {i}",
                retrieved_docs=[],
                evaluation=eval_b
            )
        
        # Analyze experiment
        result = framework.analyze_experiment(test_id)
        
        assert result is not None
        assert isinstance(result, TestResult)
        assert result.test_id == test_id
        
        # Check statistical metrics
        assert 0.0 <= result.p_value <= 1.0
        assert result.effect_size >= 0.0
        assert isinstance(result.statistical_significance, StatisticalSignificance)
        
        # Should detect variant A as winner (better performance)
        # Note: This depends on having significant difference
        if result.statistical_significance in [
            StatisticalSignificance.SIGNIFICANT, 
            StatisticalSignificance.HIGHLY_SIGNIFICANT
        ]:
            assert result.winning_variant == "variant_a"
    
    def test_experiment_completion(self, sample_test_variant_a, sample_test_variant_b):
        """Test experiment completion."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Completion Test",
            description="Testing experiment completion",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        framework.start_experiment(test_id)
        
        # Complete experiment
        success = framework.complete_experiment(test_id)
        assert success
        
        # Should be moved to completed experiments
        assert test_id not in framework.active_experiments
        assert test_id in framework.completed_experiments
        
        completed_experiment = framework.completed_experiments[test_id]
        assert completed_experiment.status == TestStatus.COMPLETED
        assert completed_experiment.end_time is not None
    
    def test_experiment_status_tracking(self, sample_test_variant_a, sample_test_variant_b):
        """Test experiment status tracking."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Status Test",
            description="Testing status tracking",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        # Get status
        status = framework.get_experiment_status(test_id)
        
        assert status is not None
        assert status["test_id"] == test_id
        assert status["name"] == "Status Test"
        assert status["status"] == "planning"
        assert "sample_sizes" in status
        assert "progress" in status
    
    def test_experiment_listing(self, sample_test_variant_a, sample_test_variant_b):
        """Test experiment listing functionality."""
        framework = ABTestingFramework()
        
        # Create multiple experiments
        test_id1 = framework.create_experiment(
            name="Test 1",
            description="First test",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        test_id2 = framework.create_experiment(
            name="Test 2",
            description="Second test",
            test_type=TestType.RETRIEVAL_METHOD,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        framework.start_experiment(test_id1)
        
        # List all experiments
        experiments = framework.list_experiments()
        assert len(experiments) == 2
        
        # List active experiments only
        active_experiments = framework.list_experiments(TestStatus.ACTIVE)
        assert len(active_experiments) == 1
        assert active_experiments[0]["test_id"] == test_id1


class TestGlobalFramework:
    """Test global framework instance management."""
    
    def test_get_ab_testing_framework(self):
        """Test getting global A/B testing framework instance."""
        framework1 = get_ab_testing_framework()
        framework2 = get_ab_testing_framework()
        
        # Should return the same instance
        assert framework1 is framework2
        assert isinstance(framework1, ABTestingFramework)


class TestHelperFunctions:
    """Test helper functions for common test scenarios."""
    
    def test_create_chunking_strategy_test(self):
        """Test helper function for chunking strategy tests."""
        test_id = create_chunking_strategy_test(
            name="Enhanced vs Recursive",
            strategy_a="enhanced_semantic",
            strategy_b="recursive",
            config_a={"chunk_size": 1200},
            config_b={"chunk_size": 1000}
        )
        
        assert isinstance(test_id, str)
        
        framework = get_ab_testing_framework()
        experiment = framework.active_experiments[test_id]
        
        assert experiment.test_type == TestType.CHUNKING_STRATEGY
        assert experiment.variant_a.chunking_strategy == "enhanced_semantic"
        assert experiment.variant_b.chunking_strategy == "recursive"
        assert experiment.variant_a.chunking_config["chunk_size"] == 1200
        assert experiment.variant_b.chunking_config["chunk_size"] == 1000
    
    def test_create_retrieval_method_test(self):
        """Test helper function for retrieval method tests."""
        test_id = create_retrieval_method_test(
            name="Vector vs Hybrid",
            method_a="vector_search",
            method_b="hybrid_search",
            config_a={"similarity_threshold": 0.7},
            config_b={"similarity_threshold": 0.8}
        )
        
        assert isinstance(test_id, str)
        
        framework = get_ab_testing_framework()
        experiment = framework.active_experiments[test_id]
        
        assert experiment.test_type == TestType.RETRIEVAL_METHOD
        assert experiment.variant_a.retrieval_method == "vector_search"
        assert experiment.variant_b.retrieval_method == "hybrid_search"


# Edge cases and error handling tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_experiment_operations(self):
        """Test operations on invalid experiment IDs."""
        framework = ABTestingFramework()
        
        # Test invalid experiment ID
        assert not framework.start_experiment("invalid_id")
        assert not framework.stop_experiment("invalid_id")
        assert not framework.complete_experiment("invalid_id")
        assert framework.get_experiment_status("invalid_id") is None
        assert framework.analyze_experiment("invalid_id") is None
    
    def test_insufficient_data_analysis(self, sample_test_variant_a, sample_test_variant_b):
        """Test analysis with insufficient data."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Insufficient Data Test",
            description="Testing with insufficient data",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        framework.start_experiment(test_id)
        
        # Try analysis with no data
        result = framework.analyze_experiment(test_id)
        assert result is None
    
    def test_recording_to_inactive_experiment(
        self,
        sample_test_variant_a,
        sample_test_variant_b,
        sample_retrieval_evaluation
    ):
        """Test recording evaluation to inactive experiment."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Inactive Test",
            description="Testing inactive experiment",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b
        )
        
        # Don't start experiment - should fail
        success = framework.record_evaluation(
            test_id=test_id,
            variant_id="variant_a",
            query="test query",
            retrieved_docs=[],
            evaluation=sample_retrieval_evaluation
        )
        
        assert not success


# Performance tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_experiment_handling(
        self,
        sample_test_variant_a,
        sample_test_variant_b,
        sample_retrieval_evaluation
    ):
        """Test handling of experiments with many samples."""
        framework = ABTestingFramework()
        
        test_id = framework.create_experiment(
            name="Large Experiment",
            description="Testing large sample sizes",
            test_type=TestType.CHUNKING_STRATEGY,
            variant_a=sample_test_variant_a,
            variant_b=sample_test_variant_b,
            target_sample_size=200
        )
        
        framework.start_experiment(test_id)
        
        import time
        start_time = time.time()
        
        # Add many samples
        for i in range(100):
            framework.record_evaluation(
                test_id=test_id,
                variant_id="variant_a" if i % 2 == 0 else "variant_b",
                query=f"test query {i}",
                retrieved_docs=[],
                evaluation=sample_retrieval_evaluation
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (under 10 seconds)
        assert processing_time < 10.0
        
        # Should have recorded all samples
        experiment = framework.active_experiments[test_id]
        total_samples = len(experiment.variant_a_samples) + len(experiment.variant_b_samples)
        assert total_samples == 100
        
        # Analysis should still work
        result = framework.analyze_experiment(test_id)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])