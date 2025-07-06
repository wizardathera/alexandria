"""
A/B testing framework for retrieval quality comparison (Phase 1.1).

This module implements comprehensive A/B testing capabilities for comparing
different retrieval strategies, chunking approaches, and configurations with
statistical significance testing and automated optimization.
"""

import json
import math
import statistics
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

try:
    from langchain.schema import Document
    import numpy as np
    from scipy import stats
except ImportError:
    Document = dict
    np = None
    stats = None

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.retrieval_evaluation import RetrievalEvaluator, RetrievalEvaluation, ConfidenceLevel
from src.utils.chunking_benchmark import ChunkingBenchmark, get_chunking_benchmark

logger = get_logger(__name__)


class TestType(Enum):
    """Types of A/B tests that can be performed."""
    CHUNKING_STRATEGY = "chunking_strategy"
    RETRIEVAL_METHOD = "retrieval_method"
    CONFIGURATION = "configuration"
    PIPELINE_COMPARISON = "pipeline_comparison"
    USER_EXPERIENCE = "user_experience"


class TestStatus(Enum):
    """Status of A/B test experiments."""
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ANALYZING = "analyzing"


class StatisticalSignificance(Enum):
    """Statistical significance levels."""
    NOT_SIGNIFICANT = "not_significant"     # p > 0.05
    MARGINALLY_SIGNIFICANT = "marginal"     # 0.01 < p <= 0.05
    SIGNIFICANT = "significant"             # 0.001 < p <= 0.01
    HIGHLY_SIGNIFICANT = "highly_significant"  # p <= 0.001


@dataclass
class TestVariant:
    """Configuration for a single test variant (A or B)."""
    
    variant_id: str
    name: str
    description: str
    
    # Configuration parameters
    chunking_strategy: Optional[str] = None
    chunking_config: Optional[Dict[str, Any]] = None
    retrieval_method: Optional[str] = None
    retrieval_config: Optional[Dict[str, Any]] = None
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    # Test allocation
    traffic_allocation: float = 0.5  # Percentage of traffic (0.0 to 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)


@dataclass
class TestMetrics:
    """Metrics collected for A/B test evaluation."""
    
    # Core retrieval metrics
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mean_reciprocal_rank: float = 0.0
    normalized_dcg: float = 0.0
    
    # Performance metrics
    retrieval_time: float = 0.0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    
    # Quality metrics
    overall_confidence: float = 0.0
    semantic_confidence: float = 0.0
    factual_confidence: float = 0.0
    
    # User experience metrics
    user_satisfaction: Optional[float] = None
    click_through_rate: Optional[float] = None
    session_duration: Optional[float] = None
    
    # Metadata
    sample_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted composite score for comparison."""
        if weights is None:
            weights = {
                "precision_at_3": 0.25,
                "mean_reciprocal_rank": 0.20,
                "normalized_dcg": 0.20,
                "overall_confidence": 0.15,
                "retrieval_time": -0.10,  # Negative weight (lower is better)
                "processing_time": -0.10   # Negative weight (lower is better)
            }
        
        # Normalize time metrics (convert to score where lower time = higher score)
        time_score = 1.0 / (1.0 + self.retrieval_time + self.processing_time)
        
        composite = (
            self.precision_at_3 * weights.get("precision_at_3", 0) +
            self.mean_reciprocal_rank * weights.get("mean_reciprocal_rank", 0) +
            self.normalized_dcg * weights.get("normalized_dcg", 0) +
            self.overall_confidence * weights.get("overall_confidence", 0) +
            time_score * abs(weights.get("retrieval_time", 0) + weights.get("processing_time", 0))
        )
        
        return max(0.0, min(1.0, composite))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["composite_score"] = self.calculate_composite_score()
        return result


@dataclass
class TestResult:
    """Results of an A/B test comparison."""
    
    test_id: str
    variant_a_metrics: TestMetrics
    variant_b_metrics: TestMetrics
    
    # Statistical analysis
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    effect_size: float = 0.0
    statistical_significance: StatisticalSignificance = StatisticalSignificance.NOT_SIGNIFICANT
    
    # Recommendations
    winning_variant: Optional[str] = None
    confidence_in_winner: float = 0.0
    recommendation: str = ""
    
    # Metadata
    test_duration: timedelta = field(default_factory=lambda: timedelta(0))
    total_samples: int = 0
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        result["confidence_interval"] = list(self.confidence_interval)
        result["statistical_significance"] = self.statistical_significance.value
        result["test_duration"] = str(self.test_duration)
        result["analysis_timestamp"] = self.analysis_timestamp.isoformat()
        return result


@dataclass
class ABTestExperiment:
    """Complete A/B test experiment configuration and results."""
    
    # Test identification
    test_id: str
    name: str
    description: str
    test_type: TestType
    
    # Test configuration
    variant_a: TestVariant
    variant_b: TestVariant
    
    # Test parameters
    target_sample_size: int = 100
    significance_level: float = 0.05
    minimum_effect_size: float = 0.1
    
    # Test state
    status: TestStatus = TestStatus.PLANNING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Data collection
    variant_a_samples: List[TestMetrics] = field(default_factory=list)
    variant_b_samples: List[TestMetrics] = field(default_factory=list)
    
    # Results
    current_result: Optional[TestResult] = None
    final_result: Optional[TestResult] = None
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_current_sample_size(self) -> Tuple[int, int]:
        """Get current sample sizes for both variants."""
        return len(self.variant_a_samples), len(self.variant_b_samples)
    
    def is_ready_for_analysis(self) -> bool:
        """Check if test has enough samples for statistical analysis."""
        a_count, b_count = self.get_current_sample_size()
        min_samples_per_variant = max(10, self.target_sample_size // 4)
        return a_count >= min_samples_per_variant and b_count >= min_samples_per_variant
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = asdict(self)
        result["test_type"] = self.test_type.value
        result["status"] = self.status.value
        result["start_time"] = self.start_time.isoformat() if self.start_time else None
        result["end_time"] = self.end_time.isoformat() if self.end_time else None
        result["created_at"] = self.created_at.isoformat()
        return result


class StatisticalAnalyzer:
    """Statistical analysis utilities for A/B testing."""
    
    @staticmethod
    def calculate_t_test(
        samples_a: List[float], 
        samples_b: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate t-test for two independent samples.
        
        Returns:
            Tuple of (t_statistic, p_value)
        """
        if not samples_a or not samples_b:
            return 0.0, 1.0
        
        if len(samples_a) < 2 or len(samples_b) < 2:
            return 0.0, 1.0
        
        # Use scipy if available, otherwise simple approximation
        if stats is not None:
            try:
                t_stat, p_val = stats.ttest_ind(samples_a, samples_b)
                return float(t_stat), float(p_val)
            except Exception as e:
                logger.warning(f"Scipy t-test failed: {e}, using manual calculation")
        
        # Manual t-test calculation
        mean_a = statistics.mean(samples_a)
        mean_b = statistics.mean(samples_b)
        
        var_a = statistics.variance(samples_a) if len(samples_a) > 1 else 0
        var_b = statistics.variance(samples_b) if len(samples_b) > 1 else 0
        
        n_a, n_b = len(samples_a), len(samples_b)
        
        # Pooled standard error
        pooled_se = math.sqrt(var_a / n_a + var_b / n_b)
        
        if pooled_se == 0:
            return 0.0, 1.0
        
        t_stat = (mean_a - mean_b) / pooled_se
        
        # Degrees of freedom (Welch's approximation)
        df = n_a + n_b - 2
        
        # Approximate p-value (simplified)
        p_val = 2 * (1 - StatisticalAnalyzer._t_cdf(abs(t_stat), df))
        
        return t_stat, min(1.0, max(0.0, p_val))
    
    @staticmethod
    def _t_cdf(t: float, df: int) -> float:
        """Approximate t-distribution CDF."""
        # Simple approximation for t-distribution
        if df > 30:
            # For large df, t-distribution approaches normal
            return StatisticalAnalyzer._normal_cdf(t)
        
        # Simplified approximation for small df
        x = t / math.sqrt(df)
        return 0.5 + 0.5 * math.tanh(x)
    
    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    @staticmethod
    def calculate_confidence_interval(
        samples: List[float], 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for sample mean."""
        if len(samples) < 2:
            mean_val = samples[0] if samples else 0.0
            return (mean_val, mean_val)
        
        mean_val = statistics.mean(samples)
        std_err = statistics.stdev(samples) / math.sqrt(len(samples))
        
        # Critical value for given confidence level (approximate)
        alpha = 1 - confidence_level
        z_critical = 1.96  # Approximate for 95% confidence
        
        margin_error = z_critical * std_err
        
        return (mean_val - margin_error, mean_val + margin_error)
    
    @staticmethod
    def calculate_effect_size(samples_a: List[float], samples_b: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not samples_a or not samples_b:
            return 0.0
        
        mean_a = statistics.mean(samples_a)
        mean_b = statistics.mean(samples_b)
        
        if len(samples_a) < 2 or len(samples_b) < 2:
            return abs(mean_a - mean_b)
        
        var_a = statistics.variance(samples_a)
        var_b = statistics.variance(samples_b)
        
        # Pooled standard deviation
        n_a, n_b = len(samples_a), len(samples_b)
        pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return abs(mean_a - mean_b) / pooled_std
    
    @staticmethod
    def determine_significance(p_value: float) -> StatisticalSignificance:
        """Determine statistical significance level."""
        if p_value <= 0.001:
            return StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif p_value <= 0.01:
            return StatisticalSignificance.SIGNIFICANT
        elif p_value <= 0.05:
            return StatisticalSignificance.MARGINALLY_SIGNIFICANT
        else:
            return StatisticalSignificance.NOT_SIGNIFICANT


class ABTestingFramework:
    """
    Main A/B testing framework for retrieval quality comparison.
    
    Provides end-to-end A/B testing capabilities including experiment design,
    traffic allocation, data collection, statistical analysis, and recommendations.
    """
    
    def __init__(self):
        """Initialize the A/B testing framework."""
        self.settings = get_settings()
        self.retrieval_evaluator = RetrievalEvaluator()
        self.chunking_benchmark = get_chunking_benchmark()
        
        # Active experiments
        self.active_experiments: Dict[str, ABTestExperiment] = {}
        self.completed_experiments: Dict[str, ABTestExperiment] = {}
        
        # Statistical analyzer
        self.stats_analyzer = StatisticalAnalyzer()
        
        logger.info("A/B Testing Framework initialized")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        test_type: TestType,
        variant_a: TestVariant,
        variant_b: TestVariant,
        target_sample_size: int = 100,
        significance_level: float = 0.05
    ) -> str:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            test_type: Type of test being performed
            variant_a: Configuration for variant A
            variant_b: Configuration for variant B
            target_sample_size: Target number of samples per variant
            significance_level: Statistical significance threshold
            
        Returns:
            str: Experiment ID
        """
        test_id = str(uuid.uuid4())
        
        experiment = ABTestExperiment(
            test_id=test_id,
            name=name,
            description=description,
            test_type=test_type,
            variant_a=variant_a,
            variant_b=variant_b,
            target_sample_size=target_sample_size,
            significance_level=significance_level
        )
        
        self.active_experiments[test_id] = experiment
        
        logger.info(f"Created A/B test experiment: {name} (ID: {test_id})")
        return test_id
    
    def start_experiment(self, test_id: str) -> bool:
        """Start an A/B test experiment."""
        if test_id not in self.active_experiments:
            logger.error(f"Experiment {test_id} not found")
            return False
        
        experiment = self.active_experiments[test_id]
        experiment.status = TestStatus.ACTIVE
        experiment.start_time = datetime.now()
        
        logger.info(f"Started A/B test experiment: {experiment.name}")
        return True
    
    def stop_experiment(self, test_id: str) -> bool:
        """Stop an A/B test experiment."""
        if test_id not in self.active_experiments:
            logger.error(f"Experiment {test_id} not found")
            return False
        
        experiment = self.active_experiments[test_id]
        experiment.status = TestStatus.STOPPED
        experiment.end_time = datetime.now()
        
        logger.info(f"Stopped A/B test experiment: {experiment.name}")
        return True
    
    def allocate_variant(self, test_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        Allocate a user to a test variant.
        
        Args:
            test_id: Experiment ID
            user_id: Optional user identifier for consistent allocation
            
        Returns:
            str: Variant ID ('variant_a' or 'variant_b') or None if experiment not active
        """
        if test_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[test_id]
        
        if experiment.status != TestStatus.ACTIVE:
            return None
        
        # Consistent allocation based on user ID if provided
        if user_id:
            # Use hash of user_id + test_id for consistent allocation
            hash_input = f"{user_id}_{test_id}"
            hash_value = hash(hash_input) % 100
            allocation_threshold = experiment.variant_a.traffic_allocation * 100
        else:
            # Random allocation
            hash_value = random.randint(0, 99)
            allocation_threshold = experiment.variant_a.traffic_allocation * 100
        
        return "variant_a" if hash_value < allocation_threshold else "variant_b"
    
    def record_evaluation(
        self,
        test_id: str,
        variant_id: str,
        query: str,
        retrieved_docs: List[Document],
        evaluation: RetrievalEvaluation,
        processing_time: float = 0.0,
        memory_usage: float = 0.0
    ) -> bool:
        """
        Record an evaluation result for A/B testing.
        
        Args:
            test_id: Experiment ID
            variant_id: Variant identifier
            query: Search query
            retrieved_docs: Retrieved documents
            evaluation: Retrieval evaluation results
            processing_time: Processing time in seconds
            memory_usage: Memory usage in MB
            
        Returns:
            bool: Success status
        """
        if test_id not in self.active_experiments:
            logger.error(f"Experiment {test_id} not found")
            return False
        
        experiment = self.active_experiments[test_id]
        
        if experiment.status != TestStatus.ACTIVE:
            logger.warning(f"Experiment {test_id} is not active")
            return False
        
        # Create test metrics from evaluation
        metrics = TestMetrics(
            precision_at_1=evaluation.precision_at_k.get(1, 0.0),
            precision_at_3=evaluation.precision_at_k.get(3, 0.0),
            precision_at_5=evaluation.precision_at_k.get(5, 0.0),
            mean_reciprocal_rank=evaluation.mean_reciprocal_rank,
            normalized_dcg=evaluation.normalized_dcg,
            retrieval_time=evaluation.retrieval_time,
            processing_time=processing_time,
            memory_usage=memory_usage,
            overall_confidence=evaluation.overall_confidence.overall_confidence,
            semantic_confidence=evaluation.overall_confidence.semantic_confidence,
            factual_confidence=evaluation.overall_confidence.factual_confidence,
            sample_count=1
        )
        
        # Add to appropriate variant
        if variant_id == "variant_a":
            experiment.variant_a_samples.append(metrics)
        elif variant_id == "variant_b":
            experiment.variant_b_samples.append(metrics)
        else:
            logger.error(f"Invalid variant_id: {variant_id}")
            return False
        
        logger.debug(f"Recorded evaluation for experiment {test_id}, variant {variant_id}")
        
        # Check if we should run analysis
        if experiment.is_ready_for_analysis():
            self._update_current_analysis(experiment)
        
        return True
    
    def analyze_experiment(self, test_id: str) -> Optional[TestResult]:
        """
        Perform statistical analysis of an A/B test experiment.
        
        Args:
            test_id: Experiment ID
            
        Returns:
            TestResult: Analysis results or None if insufficient data
        """
        if test_id not in self.active_experiments:
            experiment = self.completed_experiments.get(test_id)
            if not experiment:
                logger.error(f"Experiment {test_id} not found")
                return None
        else:
            experiment = self.active_experiments[test_id]
        
        if not experiment.is_ready_for_analysis():
            logger.warning(f"Experiment {test_id} has insufficient data for analysis")
            return None
        
        # Calculate aggregate metrics for each variant
        variant_a_metrics = self._calculate_aggregate_metrics(experiment.variant_a_samples)
        variant_b_metrics = self._calculate_aggregate_metrics(experiment.variant_b_samples)
        
        # Extract primary metric values for statistical analysis
        metric_a = [m.calculate_composite_score() for m in experiment.variant_a_samples]
        metric_b = [m.calculate_composite_score() for m in experiment.variant_b_samples]
        
        # Perform statistical analysis
        t_stat, p_value = self.stats_analyzer.calculate_t_test(metric_a, metric_b)
        effect_size = self.stats_analyzer.calculate_effect_size(metric_a, metric_b)
        
        # Calculate confidence intervals
        ci_a = self.stats_analyzer.calculate_confidence_interval(metric_a)
        ci_b = self.stats_analyzer.calculate_confidence_interval(metric_b)
        
        # Determine statistical significance
        significance = self.stats_analyzer.determine_significance(p_value)
        
        # Determine winning variant
        winning_variant = None
        confidence_in_winner = 0.0
        
        if significance in [StatisticalSignificance.SIGNIFICANT, StatisticalSignificance.HIGHLY_SIGNIFICANT]:
            a_mean = statistics.mean(metric_a)
            b_mean = statistics.mean(metric_b)
            
            if a_mean > b_mean:
                winning_variant = "variant_a"
                confidence_in_winner = 1.0 - p_value
            else:
                winning_variant = "variant_b"
                confidence_in_winner = 1.0 - p_value
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            experiment, variant_a_metrics, variant_b_metrics, 
            significance, effect_size, winning_variant
        )
        
        # Create test result
        result = TestResult(
            test_id=test_id,
            variant_a_metrics=variant_a_metrics,
            variant_b_metrics=variant_b_metrics,
            p_value=p_value,
            confidence_interval=ci_b if winning_variant == "variant_b" else ci_a,
            effect_size=effect_size,
            statistical_significance=significance,
            winning_variant=winning_variant,
            confidence_in_winner=confidence_in_winner,
            recommendation=recommendation,
            test_duration=experiment.end_time - experiment.start_time if experiment.end_time and experiment.start_time else timedelta(0),
            total_samples=len(experiment.variant_a_samples) + len(experiment.variant_b_samples)
        )
        
        # Update experiment with result
        experiment.current_result = result
        if experiment.status == TestStatus.COMPLETED:
            experiment.final_result = result
        
        logger.info(f"Analysis completed for experiment {test_id}: {significance.value}, winner: {winning_variant}")
        
        return result
    
    def complete_experiment(self, test_id: str) -> bool:
        """Complete an A/B test experiment and move to completed experiments."""
        if test_id not in self.active_experiments:
            logger.error(f"Experiment {test_id} not found in active experiments")
            return False
        
        experiment = self.active_experiments[test_id]
        experiment.status = TestStatus.COMPLETED
        experiment.end_time = datetime.now()
        
        # Perform final analysis
        final_result = self.analyze_experiment(test_id)
        experiment.final_result = final_result
        
        # Move to completed experiments
        self.completed_experiments[test_id] = experiment
        del self.active_experiments[test_id]
        
        logger.info(f"Completed A/B test experiment: {experiment.name}")
        return True
    
    def _calculate_aggregate_metrics(self, samples: List[TestMetrics]) -> TestMetrics:
        """Calculate aggregate metrics from individual samples."""
        if not samples:
            return TestMetrics()
        
        return TestMetrics(
            precision_at_1=statistics.mean([s.precision_at_1 for s in samples]),
            precision_at_3=statistics.mean([s.precision_at_3 for s in samples]),
            precision_at_5=statistics.mean([s.precision_at_5 for s in samples]),
            mean_reciprocal_rank=statistics.mean([s.mean_reciprocal_rank for s in samples]),
            normalized_dcg=statistics.mean([s.normalized_dcg for s in samples]),
            retrieval_time=statistics.mean([s.retrieval_time for s in samples]),
            processing_time=statistics.mean([s.processing_time for s in samples]),
            memory_usage=statistics.mean([s.memory_usage for s in samples]),
            overall_confidence=statistics.mean([s.overall_confidence for s in samples]),
            semantic_confidence=statistics.mean([s.semantic_confidence for s in samples]),
            factual_confidence=statistics.mean([s.factual_confidence for s in samples]),
            sample_count=len(samples)
        )
    
    def _update_current_analysis(self, experiment: ABTestExperiment):
        """Update current analysis for an active experiment."""
        try:
            result = self.analyze_experiment(experiment.test_id)
            if result:
                experiment.current_result = result
        except Exception as e:
            logger.error(f"Failed to update analysis for experiment {experiment.test_id}: {e}")
    
    def _generate_recommendation(
        self,
        experiment: ABTestExperiment,
        variant_a_metrics: TestMetrics,
        variant_b_metrics: TestMetrics,
        significance: StatisticalSignificance,
        effect_size: float,
        winning_variant: Optional[str]
    ) -> str:
        """Generate actionable recommendations based on test results."""
        
        recommendations = []
        
        if significance == StatisticalSignificance.NOT_SIGNIFICANT:
            recommendations.append("No statistically significant difference detected between variants.")
            recommendations.append("Consider increasing sample size or testing more distinct configurations.")
        
        elif significance == StatisticalSignificance.MARGINALLY_SIGNIFICANT:
            recommendations.append("Marginally significant results detected.")
            recommendations.append("Consider extending the test or implementing with caution.")
        
        else:  # Significant or highly significant
            if winning_variant == "variant_a":
                recommendations.append(f"Variant A ({experiment.variant_a.name}) shows significantly better performance.")
                recommendations.append(f"Recommend implementing Variant A configuration.")
            elif winning_variant == "variant_b":
                recommendations.append(f"Variant B ({experiment.variant_b.name}) shows significantly better performance.")
                recommendations.append(f"Recommend implementing Variant B configuration.")
        
        # Effect size interpretation
        if effect_size < 0.2:
            recommendations.append("Small effect size - practical significance may be limited.")
        elif effect_size < 0.5:
            recommendations.append("Medium effect size - moderately practical improvement.")
        else:
            recommendations.append("Large effect size - substantial practical improvement expected.")
        
        # Performance-specific recommendations
        a_composite = variant_a_metrics.calculate_composite_score()
        b_composite = variant_b_metrics.calculate_composite_score()
        
        if abs(a_composite - b_composite) > 0.1:
            if a_composite > b_composite:
                recommendations.append("Variant A shows superior overall quality metrics.")
            else:
                recommendations.append("Variant B shows superior overall quality metrics.")
        
        # Speed vs. quality trade-offs
        a_speed = 1.0 / (1.0 + variant_a_metrics.retrieval_time + variant_a_metrics.processing_time)
        b_speed = 1.0 / (1.0 + variant_b_metrics.retrieval_time + variant_b_metrics.processing_time)
        
        if abs(a_speed - b_speed) > 0.2:
            if a_speed > b_speed:
                recommendations.append("Variant A provides better performance speed.")
            else:
                recommendations.append("Variant B provides better performance speed.")
        
        return " ".join(recommendations)
    
    def get_experiment_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get current status and progress of an experiment."""
        experiment = self.active_experiments.get(test_id) or self.completed_experiments.get(test_id)
        
        if not experiment:
            return None
        
        a_count, b_count = experiment.get_current_sample_size()
        
        status = {
            "test_id": test_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "test_type": experiment.test_type.value,
            "sample_sizes": {
                "variant_a": a_count,
                "variant_b": b_count,
                "target": experiment.target_sample_size
            },
            "progress": {
                "variant_a": min(1.0, a_count / experiment.target_sample_size),
                "variant_b": min(1.0, b_count / experiment.target_sample_size)
            },
            "ready_for_analysis": experiment.is_ready_for_analysis(),
            "start_time": experiment.start_time.isoformat() if experiment.start_time else None,
            "end_time": experiment.end_time.isoformat() if experiment.end_time else None
        }
        
        if experiment.current_result:
            status["current_analysis"] = experiment.current_result.to_dict()
        
        return status
    
    def list_experiments(self, status_filter: Optional[TestStatus] = None) -> List[Dict[str, Any]]:
        """List all experiments with optional status filtering."""
        all_experiments = {**self.active_experiments, **self.completed_experiments}
        
        experiments = []
        for experiment in all_experiments.values():
            if status_filter is None or experiment.status == status_filter:
                experiments.append({
                    "test_id": experiment.test_id,
                    "name": experiment.name,
                    "status": experiment.status.value,
                    "test_type": experiment.test_type.value,
                    "created_at": experiment.created_at.isoformat(),
                    "sample_count": len(experiment.variant_a_samples) + len(experiment.variant_b_samples)
                })
        
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)
    
    def export_experiment_results(self, test_id: str, filename: Optional[str] = None) -> Optional[str]:
        """Export experiment results to JSON file."""
        experiment = self.active_experiments.get(test_id) or self.completed_experiments.get(test_id)
        
        if not experiment:
            logger.error(f"Experiment {test_id} not found")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ab_test_{experiment.name}_{timestamp}.json"
        
        export_data = {
            "experiment": experiment.to_dict(),
            "variant_a_samples": [sample.to_dict() for sample in experiment.variant_a_samples],
            "variant_b_samples": [sample.to_dict() for sample in experiment.variant_b_samples],
            "analysis_result": experiment.final_result.to_dict() if experiment.final_result else None,
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Experiment results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export experiment results: {e}")
            return None


# Global A/B testing framework instance
_ab_testing_framework: Optional[ABTestingFramework] = None


def get_ab_testing_framework() -> ABTestingFramework:
    """Get the global A/B testing framework instance."""
    global _ab_testing_framework
    
    if _ab_testing_framework is None:
        _ab_testing_framework = ABTestingFramework()
    
    return _ab_testing_framework


# Helper functions for common test scenarios
def create_chunking_strategy_test(
    name: str,
    strategy_a: str,
    strategy_b: str,
    config_a: Optional[Dict[str, Any]] = None,
    config_b: Optional[Dict[str, Any]] = None
) -> str:
    """Create an A/B test comparing two chunking strategies."""
    framework = get_ab_testing_framework()
    
    variant_a = TestVariant(
        variant_id="chunking_a",
        name=f"Chunking Strategy: {strategy_a}",
        description=f"Using {strategy_a} chunking strategy",
        chunking_strategy=strategy_a,
        chunking_config=config_a or {}
    )
    
    variant_b = TestVariant(
        variant_id="chunking_b", 
        name=f"Chunking Strategy: {strategy_b}",
        description=f"Using {strategy_b} chunking strategy",
        chunking_strategy=strategy_b,
        chunking_config=config_b or {}
    )
    
    return framework.create_experiment(
        name=name,
        description=f"Comparing {strategy_a} vs {strategy_b} chunking strategies",
        test_type=TestType.CHUNKING_STRATEGY,
        variant_a=variant_a,
        variant_b=variant_b
    )


def create_retrieval_method_test(
    name: str,
    method_a: str,
    method_b: str,
    config_a: Optional[Dict[str, Any]] = None,
    config_b: Optional[Dict[str, Any]] = None
) -> str:
    """Create an A/B test comparing two retrieval methods."""
    framework = get_ab_testing_framework()
    
    variant_a = TestVariant(
        variant_id="retrieval_a",
        name=f"Retrieval Method: {method_a}",
        description=f"Using {method_a} retrieval method",
        retrieval_method=method_a,
        retrieval_config=config_a or {}
    )
    
    variant_b = TestVariant(
        variant_id="retrieval_b",
        name=f"Retrieval Method: {method_b}",
        description=f"Using {method_b} retrieval method", 
        retrieval_method=method_b,
        retrieval_config=config_b or {}
    )
    
    return framework.create_experiment(
        name=name,
        description=f"Comparing {method_a} vs {method_b} retrieval methods",
        test_type=TestType.RETRIEVAL_METHOD,
        variant_a=variant_a,
        variant_b=variant_b
    )