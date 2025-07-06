"""
Performance Testing and Optimization for Supabase pgvector Migration.

This module provides comprehensive performance testing and optimization tools
for comparing and optimizing vector database performance during migration
from Chroma to Supabase pgvector.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import random
import json

import numpy as np

from src.utils.enhanced_database import EnhancedVectorDatabaseInterface
from src.models import EmbeddingMetadata, User, ModuleType, ContentType
from src.utils.logger import get_logger
from src.utils.config import DEFAULT_COLLECTION_NAME

logger = get_logger(__name__)


class PerformanceMetric(str, Enum):
    """Performance metrics to measure."""
    QUERY_LATENCY = "query_latency"
    THROUGHPUT = "throughput"
    CONCURRENT_QUERIES = "concurrent_queries"
    INDEX_BUILD_TIME = "index_build_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    DISK_IO = "disk_io"
    CONNECTION_OVERHEAD = "connection_overhead"


class TestScenario(str, Enum):
    """Performance test scenarios."""
    SINGLE_QUERY = "single_query"
    BATCH_QUERIES = "batch_queries"
    CONCURRENT_USERS = "concurrent_users"
    LOAD_TEST = "load_test"
    STRESS_TEST = "stress_test"
    SOAK_TEST = "soak_test"


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing."""
    scenario: TestScenario
    duration_seconds: int = 60
    concurrent_users: int = 10
    queries_per_user: int = 100
    query_batch_size: int = 10
    
    # Query parameters
    query_dimensions: int = 1536
    result_limit: int = 5
    similarity_threshold: float = 0.0
    
    # Load test parameters
    ramp_up_seconds: int = 30
    steady_state_seconds: int = 60
    ramp_down_seconds: int = 30
    
    # Optimization parameters
    index_parameters: Dict[str, Any] = field(default_factory=dict)
    connection_pool_size: int = 10
    enable_optimization: bool = True


@dataclass
class PerformanceResult:
    """Result of a single performance measurement."""
    metric: PerformanceMetric
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceTestResult:
    """Complete result of a performance test."""
    test_id: str
    scenario: TestScenario
    database_provider: str
    config: PerformanceTestConfig
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Core metrics
    results: List[PerformanceResult] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    
    # Resource utilization
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate test duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def get_metric_values(self, metric: PerformanceMetric) -> List[float]:
        """Get all values for a specific metric."""
        return [r.value for r in self.results if r.metric == metric]
    
    def get_metric_stats(self, metric: PerformanceMetric) -> Dict[str, float]:
        """Get statistical summary for a metric."""
        values = self.get_metric_values(metric)
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }


class PerformanceTester:
    """
    Comprehensive performance testing tool for vector databases.
    
    Provides various testing scenarios and metrics collection for
    comparing performance between Chroma and Supabase implementations.
    """
    
    def __init__(self):
        """Initialize the performance tester."""
        self._active_tests: Dict[str, PerformanceTestResult] = {}
        self._baseline_results: Dict[str, PerformanceTestResult] = {}
        
    async def run_performance_test(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Optional[Dict[str, Any]] = None
    ) -> PerformanceTestResult:
        """
        Run a comprehensive performance test.
        
        Args:
            db: Database instance to test
            config: Test configuration
            test_data: Optional test data (embeddings, queries)
            
        Returns:
            PerformanceTestResult: Test results
        """
        test_id = f"perf_test_{int(time.time())}"
        
        result = PerformanceTestResult(
            test_id=test_id,
            scenario=config.scenario,
            database_provider=type(db).__name__,
            config=config,
            started_at=datetime.now()
        )
        
        self._active_tests[test_id] = result
        
        logger.info(f"Starting performance test: {test_id} - Scenario: {config.scenario}")
        
        try:
            # Generate test data if not provided
            if test_data is None:
                test_data = await self._generate_test_data(config)
            
            # Run the specific test scenario
            if config.scenario == TestScenario.SINGLE_QUERY:
                await self._test_single_query_performance(db, config, test_data, result)
            elif config.scenario == TestScenario.BATCH_QUERIES:
                await self._test_batch_query_performance(db, config, test_data, result)
            elif config.scenario == TestScenario.CONCURRENT_USERS:
                await self._test_concurrent_user_performance(db, config, test_data, result)
            elif config.scenario == TestScenario.LOAD_TEST:
                await self._test_load_performance(db, config, test_data, result)
            elif config.scenario == TestScenario.STRESS_TEST:
                await self._test_stress_performance(db, config, test_data, result)
            elif config.scenario == TestScenario.SOAK_TEST:
                await self._test_soak_performance(db, config, test_data, result)
            
            # Calculate summary statistics
            self._calculate_summary_stats(result)
            
            result.completed_at = datetime.now()
            logger.info(f"Performance test completed: {test_id} - Duration: {result.duration_seconds:.2f}s")
            
        except Exception as e:
            result.errors.append(f"Test execution failed: {str(e)}")
            logger.error(f"Performance test failed: {test_id} - {e}")
        
        finally:
            self._active_tests.pop(test_id, None)
        
        return result
    
    async def _generate_test_data(self, config: PerformanceTestConfig) -> Dict[str, Any]:
        """Generate test data for performance testing."""
        # Generate random query vectors
        query_vectors = []
        for _ in range(config.queries_per_user * config.concurrent_users):
            vector = np.random.random(config.query_dimensions).tolist()
            query_vectors.append(vector)
        
        # Generate test queries
        test_queries = []
        for i, vector in enumerate(query_vectors):
            test_queries.append({
                "query_id": f"test_query_{i}",
                "embedding": vector,
                "text": f"Test query {i}",
                "n_results": config.result_limit
            })
        
        return {
            "query_vectors": query_vectors,
            "test_queries": test_queries,
            "total_queries": len(test_queries)
        }
    
    async def _test_single_query_performance(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Dict[str, Any],
        result: PerformanceTestResult
    ):
        """Test single query performance."""
        queries = test_data["test_queries"][:config.queries_per_user]
        
        for query in queries:
            start_time = time.time()
            
            try:
                # Execute query
                query_result = await db.query_with_permissions(
                    collection_name=DEFAULT_COLLECTION_NAME,
                    query_text=query["text"],
                    query_embedding=query["embedding"],
                    n_results=query["n_results"]
                )
                
                # Measure latency
                latency_ms = (time.time() - start_time) * 1000
                
                result.results.append(PerformanceResult(
                    metric=PerformanceMetric.QUERY_LATENCY,
                    value=latency_ms,
                    unit="ms",
                    timestamp=datetime.now(),
                    metadata={
                        "query_id": query["query_id"],
                        "results_returned": len(query_result.get("documents", []))
                    }
                ))
                
                # Track success
                if query_result.get("documents"):
                    result.success_rate += 1
                
            except Exception as e:
                result.errors.append(f"Query {query['query_id']} failed: {str(e)}")
        
        # Calculate success rate
        result.success_rate = (result.success_rate / len(queries)) * 100
    
    async def _test_batch_query_performance(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Dict[str, Any],
        result: PerformanceTestResult
    ):
        """Test batch query performance."""
        queries = test_data["test_queries"]
        batch_size = config.query_batch_size
        
        successful_batches = 0
        total_batches = len(queries) // batch_size
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            start_time = time.time()
            
            try:
                # Execute batch queries concurrently
                tasks = []
                for query in batch:
                    task = db.query_with_permissions(
                        collection_name=DEFAULT_COLLECTION_NAME,
                        query_text=query["text"],
                        query_embedding=query["embedding"],
                        n_results=query["n_results"]
                    )
                    tasks.append(task)
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Measure batch latency
                batch_latency_ms = (time.time() - start_time) * 1000
                
                # Check for errors in batch
                batch_errors = sum(1 for r in batch_results if isinstance(r, Exception))
                batch_success_rate = ((len(batch) - batch_errors) / len(batch)) * 100
                
                result.results.append(PerformanceResult(
                    metric=PerformanceMetric.QUERY_LATENCY,
                    value=batch_latency_ms,
                    unit="ms",
                    timestamp=datetime.now(),
                    metadata={
                        "batch_size": len(batch),
                        "batch_index": i // batch_size,
                        "success_rate": batch_success_rate,
                        "errors": batch_errors
                    }
                ))
                
                # Calculate throughput (queries per second)
                throughput = len(batch) / (batch_latency_ms / 1000)
                result.results.append(PerformanceResult(
                    metric=PerformanceMetric.THROUGHPUT,
                    value=throughput,
                    unit="queries/sec",
                    timestamp=datetime.now(),
                    metadata={"batch_index": i // batch_size}
                ))
                
                if batch_success_rate > 90:  # Consider batch successful if >90% queries succeed
                    successful_batches += 1
                
            except Exception as e:
                result.errors.append(f"Batch {i // batch_size} failed: {str(e)}")
        
        # Calculate overall success rate
        result.success_rate = (successful_batches / total_batches) * 100 if total_batches > 0 else 0
    
    async def _test_concurrent_user_performance(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Dict[str, Any],
        result: PerformanceTestResult
    ):
        """Test concurrent user performance."""
        queries_per_user = config.queries_per_user
        concurrent_users = config.concurrent_users
        
        async def simulate_user(user_id: int, user_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Simulate a single user's query pattern."""
            user_results = {
                "user_id": user_id,
                "queries_executed": 0,
                "total_latency": 0,
                "errors": []
            }
            
            for query in user_queries:
                start_time = time.time()
                try:
                    query_result = await db.query_with_permissions(
                        collection_name=DEFAULT_COLLECTION_NAME,
                        query_text=query["text"],
                        query_embedding=query["embedding"],
                        n_results=query["n_results"]
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    user_results["total_latency"] += latency_ms
                    user_results["queries_executed"] += 1
                    
                    # Record individual query latency
                    result.results.append(PerformanceResult(
                        metric=PerformanceMetric.QUERY_LATENCY,
                        value=latency_ms,
                        unit="ms",
                        timestamp=datetime.now(),
                        metadata={
                            "user_id": user_id,
                            "query_id": query["query_id"]
                        }
                    ))
                    
                except Exception as e:
                    user_results["errors"].append(str(e))
            
            return user_results
        
        # Distribute queries among users
        all_queries = test_data["test_queries"]
        user_tasks = []
        
        for user_id in range(concurrent_users):
            start_idx = user_id * queries_per_user
            end_idx = start_idx + queries_per_user
            user_queries = all_queries[start_idx:end_idx]
            
            task = simulate_user(user_id, user_queries)
            user_tasks.append(task)
        
        # Execute all users concurrently
        start_time = time.time()
        user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Analyze concurrent performance
        total_queries = 0
        total_errors = 0
        
        for user_result in user_results:
            if isinstance(user_result, Exception):
                result.errors.append(f"User simulation failed: {str(user_result)}")
                continue
            
            total_queries += user_result["queries_executed"]
            total_errors += len(user_result["errors"])
        
        # Calculate overall throughput
        overall_throughput = total_queries / total_duration
        result.results.append(PerformanceResult(
            metric=PerformanceMetric.THROUGHPUT,
            value=overall_throughput,
            unit="queries/sec",
            timestamp=datetime.now(),
            metadata={
                "concurrent_users": concurrent_users,
                "total_queries": total_queries,
                "total_duration": total_duration
            }
        ))
        
        # Record concurrent query performance
        result.results.append(PerformanceResult(
            metric=PerformanceMetric.CONCURRENT_QUERIES,
            value=concurrent_users,
            unit="users",
            timestamp=datetime.now(),
            metadata={
                "avg_throughput": overall_throughput,
                "success_rate": ((total_queries - total_errors) / total_queries) * 100 if total_queries > 0 else 0
            }
        ))
        
        result.success_rate = ((total_queries - total_errors) / total_queries) * 100 if total_queries > 0 else 0
    
    async def _test_load_performance(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Dict[str, Any],
        result: PerformanceTestResult
    ):
        """Test load performance with ramp-up and steady state."""
        total_duration = config.ramp_up_seconds + config.steady_state_seconds + config.ramp_down_seconds
        
        async def load_generator():
            """Generate load according to the test profile."""
            start_time = time.time()
            queries = test_data["test_queries"]
            query_index = 0
            
            while time.time() - start_time < total_duration:
                current_time = time.time() - start_time
                
                # Determine load level based on phase
                if current_time < config.ramp_up_seconds:
                    # Ramp-up phase
                    load_factor = current_time / config.ramp_up_seconds
                elif current_time < config.ramp_up_seconds + config.steady_state_seconds:
                    # Steady state phase
                    load_factor = 1.0
                else:
                    # Ramp-down phase
                    remaining_time = total_duration - current_time
                    load_factor = remaining_time / config.ramp_down_seconds
                
                # Calculate target concurrent users
                target_users = int(config.concurrent_users * load_factor)
                
                # Execute queries for current load level
                if target_users > 0:
                    query_tasks = []
                    for _ in range(target_users):
                        if query_index < len(queries):
                            query = queries[query_index % len(queries)]
                            query_index += 1
                            
                            task = self._execute_single_query(db, query, current_time)
                            query_tasks.append(task)
                    
                    if query_tasks:
                        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
                        
                        # Process results
                        for query_result in query_results:
                            if isinstance(query_result, dict):
                                result.results.append(PerformanceResult(
                                    metric=PerformanceMetric.QUERY_LATENCY,
                                    value=query_result["latency_ms"],
                                    unit="ms",
                                    timestamp=datetime.now(),
                                    metadata={
                                        "phase": self._get_load_phase(current_time, config),
                                        "load_factor": load_factor,
                                        "target_users": target_users
                                    }
                                ))
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
        
        await load_generator()
        
        # Calculate load test metrics
        phase_results = self._analyze_load_phases(result, config)
        result.summary_stats["load_phases"] = phase_results
    
    async def _execute_single_query(
        self,
        db: EnhancedVectorDatabaseInterface,
        query: Dict[str, Any],
        phase_time: float
    ) -> Dict[str, Any]:
        """Execute a single query and return timing information."""
        start_time = time.time()
        
        try:
            query_result = await db.query_with_permissions(
                collection_name=DEFAULT_COLLECTION_NAME,
                query_text=query["text"],
                query_embedding=query["embedding"],
                n_results=query["n_results"]
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "latency_ms": latency_ms,
                "success": True,
                "results_count": len(query_result.get("documents", [])),
                "phase_time": phase_time
            }
            
        except Exception as e:
            return {
                "latency_ms": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e),
                "phase_time": phase_time
            }
    
    def _get_load_phase(self, current_time: float, config: PerformanceTestConfig) -> str:
        """Determine which load phase we're currently in."""
        if current_time < config.ramp_up_seconds:
            return "ramp_up"
        elif current_time < config.ramp_up_seconds + config.steady_state_seconds:
            return "steady_state"
        else:
            return "ramp_down"
    
    def _analyze_load_phases(
        self,
        result: PerformanceTestResult,
        config: PerformanceTestConfig
    ) -> Dict[str, Any]:
        """Analyze performance across different load phases."""
        phases = {"ramp_up": [], "steady_state": [], "ramp_down": []}
        
        # Group results by phase
        for perf_result in result.results:
            if perf_result.metric == PerformanceMetric.QUERY_LATENCY:
                phase = perf_result.metadata.get("phase", "unknown")
                if phase in phases:
                    phases[phase].append(perf_result.value)
        
        # Calculate statistics for each phase
        phase_stats = {}
        for phase_name, latencies in phases.items():
            if latencies:
                phase_stats[phase_name] = {
                    "count": len(latencies),
                    "avg_latency_ms": statistics.mean(latencies),
                    "p95_latency_ms": np.percentile(latencies, 95),
                    "p99_latency_ms": np.percentile(latencies, 99),
                    "max_latency_ms": max(latencies),
                    "min_latency_ms": min(latencies)
                }
            else:
                phase_stats[phase_name] = {"count": 0}
        
        return phase_stats
    
    async def _test_stress_performance(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Dict[str, Any],
        result: PerformanceTestResult
    ):
        """Test stress performance by gradually increasing load until failure."""
        max_users = config.concurrent_users * 3  # Start with 3x normal load
        current_users = config.concurrent_users
        increment = 5
        
        while current_users <= max_users:
            # Test current load level
            stress_config = PerformanceTestConfig(
                scenario=TestScenario.CONCURRENT_USERS,
                concurrent_users=current_users,
                queries_per_user=20,  # Shorter test for stress testing
                duration_seconds=30
            )
            
            stress_result = await self.run_performance_test(db, stress_config, test_data)
            
            # Record stress test results
            avg_latency = statistics.mean(stress_result.get_metric_values(PerformanceMetric.QUERY_LATENCY))
            
            result.results.append(PerformanceResult(
                metric=PerformanceMetric.CONCURRENT_QUERIES,
                value=current_users,
                unit="users",
                timestamp=datetime.now(),
                metadata={
                    "avg_latency_ms": avg_latency,
                    "success_rate": stress_result.success_rate,
                    "stress_level": current_users / config.concurrent_users
                }
            ))
            
            # Check if system is degrading significantly
            if stress_result.success_rate < 50 or avg_latency > 10000:  # 10 second timeout
                logger.info(f"Stress test limit reached at {current_users} concurrent users")
                break
            
            current_users += increment
    
    async def _test_soak_performance(
        self,
        db: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        test_data: Dict[str, Any],
        result: PerformanceTestResult
    ):
        """Test soak performance over extended duration."""
        duration = config.duration_seconds
        measurement_interval = 60  # Measure every minute
        
        start_time = time.time()
        measurement_count = 0
        
        while time.time() - start_time < duration:
            measurement_start = time.time()
            
            # Run a quick performance measurement
            sample_queries = test_data["test_queries"][:10]  # Small sample
            latencies = []
            
            for query in sample_queries:
                query_start = time.time()
                try:
                    await db.query_with_permissions(
                        collection_name=DEFAULT_COLLECTION_NAME,
                        query_text=query["text"],
                        query_embedding=query["embedding"],
                        n_results=query["n_results"]
                    )
                    latency_ms = (time.time() - query_start) * 1000
                    latencies.append(latency_ms)
                except Exception as e:
                    result.errors.append(f"Soak test query failed: {str(e)}")
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                result.results.append(PerformanceResult(
                    metric=PerformanceMetric.QUERY_LATENCY,
                    value=avg_latency,
                    unit="ms",
                    timestamp=datetime.now(),
                    metadata={
                        "measurement_index": measurement_count,
                        "elapsed_time": time.time() - start_time,
                        "sample_size": len(latencies)
                    }
                ))
            
            measurement_count += 1
            
            # Wait for next measurement interval
            elapsed = time.time() - measurement_start
            sleep_time = max(0, measurement_interval - elapsed)
            await asyncio.sleep(sleep_time)
    
    def _calculate_summary_stats(self, result: PerformanceTestResult):
        """Calculate summary statistics for test results."""
        for metric in PerformanceMetric:
            stats = result.get_metric_stats(metric)
            if stats:
                result.summary_stats[metric.value] = stats
    
    async def compare_database_performance(
        self,
        db1: EnhancedVectorDatabaseInterface,
        db2: EnhancedVectorDatabaseInterface,
        config: PerformanceTestConfig,
        db1_name: str = "Database 1",
        db2_name: str = "Database 2"
    ) -> Dict[str, Any]:
        """
        Compare performance between two databases.
        
        Args:
            db1: First database to compare
            db2: Second database to compare
            config: Test configuration
            db1_name: Name for first database
            db2_name: Name for second database
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        # Generate shared test data
        test_data = await self._generate_test_data(config)
        
        # Run tests on both databases
        result1 = await self.run_performance_test(db1, config, test_data)
        result2 = await self.run_performance_test(db2, config, test_data)
        
        # Compare results
        comparison = {
            "test_config": config.__dict__,
            "databases": {
                db1_name: {
                    "provider": result1.database_provider,
                    "duration": result1.duration_seconds,
                    "success_rate": result1.success_rate,
                    "summary_stats": result1.summary_stats,
                    "errors": len(result1.errors)
                },
                db2_name: {
                    "provider": result2.database_provider,
                    "duration": result2.duration_seconds,
                    "success_rate": result2.success_rate,
                    "summary_stats": result2.summary_stats,
                    "errors": len(result2.errors)
                }
            },
            "performance_comparison": {}
        }
        
        # Compare key metrics
        for metric in [PerformanceMetric.QUERY_LATENCY, PerformanceMetric.THROUGHPUT]:
            if metric.value in result1.summary_stats and metric.value in result2.summary_stats:
                stats1 = result1.summary_stats[metric.value]
                stats2 = result2.summary_stats[metric.value]
                
                # Calculate performance ratio
                if metric == PerformanceMetric.QUERY_LATENCY:
                    # Lower is better for latency
                    ratio = stats2["mean"] / stats1["mean"] if stats1["mean"] > 0 else float('inf')
                    winner = db1_name if stats1["mean"] < stats2["mean"] else db2_name
                else:
                    # Higher is better for throughput
                    ratio = stats1["mean"] / stats2["mean"] if stats2["mean"] > 0 else float('inf')
                    winner = db1_name if stats1["mean"] > stats2["mean"] else db2_name
                
                comparison["performance_comparison"][metric.value] = {
                    f"{db1_name}_mean": stats1["mean"],
                    f"{db2_name}_mean": stats2["mean"],
                    "ratio": ratio,
                    "winner": winner,
                    "improvement_percent": abs((ratio - 1) * 100)
                }
        
        return comparison
    
    def generate_performance_report(
        self,
        result: PerformanceTestResult,
        include_recommendations: bool = True
    ) -> str:
        """Generate a human-readable performance report."""
        report = f"""
Performance Test Report
======================
Test ID: {result.test_id}
Scenario: {result.scenario}
Database Provider: {result.database_provider}
Duration: {result.duration_seconds:.2f} seconds
Success Rate: {result.success_rate:.1f}%

Test Configuration:
- Concurrent Users: {result.config.concurrent_users}
- Queries per User: {result.config.queries_per_user}
- Query Batch Size: {result.config.query_batch_size}
- Result Limit: {result.config.result_limit}

Performance Summary:
"""
        
        # Add metric summaries
        for metric_name, stats in result.summary_stats.items():
            if isinstance(stats, dict) and "mean" in stats:
                report += f"""
{metric_name.replace('_', ' ').title()}:
- Mean: {stats['mean']:.2f}
- Median: {stats['median']:.2f}
- P95: {stats['p95']:.2f}
- P99: {stats['p99']:.2f}
- Min: {stats['min']:.2f}
- Max: {stats['max']:.2f}
- Std Dev: {stats['std_dev']:.2f}
"""
        
        # Add error information
        if result.errors:
            report += f"\nErrors Encountered: {len(result.errors)}\n"
            for error in result.errors[:5]:  # Show first 5 errors
                report += f"- {error}\n"
            if len(result.errors) > 5:
                report += f"... and {len(result.errors) - 5} more errors\n"
        
        # Add recommendations
        if include_recommendations:
            recommendations = self._generate_performance_recommendations(result)
            if recommendations:
                report += "\nPerformance Recommendations:\n"
                for rec in recommendations:
                    report += f"- {rec}\n"
        
        return report
    
    def _generate_performance_recommendations(self, result: PerformanceTestResult) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check query latency
        if PerformanceMetric.QUERY_LATENCY.value in result.summary_stats:
            latency_stats = result.summary_stats[PerformanceMetric.QUERY_LATENCY.value]
            mean_latency = latency_stats["mean"]
            p99_latency = latency_stats["p99"]
            
            if mean_latency > 1000:  # > 1 second
                recommendations.append("High average query latency detected. Consider optimizing indexes or increasing connection pool size.")
            
            if p99_latency > 5000:  # > 5 seconds
                recommendations.append("High P99 latency suggests some queries are very slow. Investigate query patterns and database optimization.")
        
        # Check success rate
        if result.success_rate < 95:
            recommendations.append("Low success rate indicates connection or query failures. Check database configuration and error logs.")
        
        # Check throughput
        if PerformanceMetric.THROUGHPUT.value in result.summary_stats:
            throughput_stats = result.summary_stats[PerformanceMetric.THROUGHPUT.value]
            mean_throughput = throughput_stats["mean"]
            
            if mean_throughput < 10:  # < 10 queries/sec
                recommendations.append("Low throughput detected. Consider optimizing queries, increasing parallelism, or upgrading hardware.")
        
        # Database-specific recommendations
        if "Supabase" in result.database_provider:
            recommendations.append("For Supabase pgvector: Ensure proper HNSW index configuration and consider adjusting connection pool settings.")
        
        if "Chroma" in result.database_provider:
            recommendations.append("For Chroma: Consider adjusting collection settings and ensure sufficient disk space for persistence.")
        
        return recommendations
    
    def export_results(self, result: PerformanceTestResult, file_path: str):
        """Export performance test results to JSON file."""
        export_data = {
            "test_id": result.test_id,
            "scenario": result.scenario,
            "database_provider": result.database_provider,
            "config": result.config.__dict__,
            "started_at": result.started_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "duration_seconds": result.duration_seconds,
            "success_rate": result.success_rate,
            "summary_stats": result.summary_stats,
            "errors": result.errors,
            "total_measurements": len(result.results)
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Performance test results exported to: {file_path}")


# ========================================
# Convenience Functions
# ========================================

async def quick_performance_comparison(
    chroma_db: EnhancedVectorDatabaseInterface,
    supabase_db: EnhancedVectorDatabaseInterface,
    concurrent_users: int = 10,
    queries_per_user: int = 50
) -> Dict[str, Any]:
    """
    Quick performance comparison between Chroma and Supabase.
    
    Args:
        chroma_db: Chroma database instance
        supabase_db: Supabase database instance
        concurrent_users: Number of concurrent users to simulate
        queries_per_user: Number of queries per user
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    tester = PerformanceTester()
    
    config = PerformanceTestConfig(
        scenario=TestScenario.CONCURRENT_USERS,
        concurrent_users=concurrent_users,
        queries_per_user=queries_per_user,
        duration_seconds=60
    )
    
    return await tester.compare_database_performance(
        chroma_db, supabase_db, config, "Chroma", "Supabase"
    )


async def benchmark_migration_performance(
    source_db: EnhancedVectorDatabaseInterface,
    target_db: EnhancedVectorDatabaseInterface
) -> Dict[str, Any]:
    """
    Benchmark performance for migration validation.
    
    Args:
        source_db: Source database
        target_db: Target database
        
    Returns:
        Dict[str, Any]: Benchmark results
    """
    tester = PerformanceTester()
    
    # Test multiple scenarios
    scenarios = [
        (TestScenario.SINGLE_QUERY, {"concurrent_users": 1, "queries_per_user": 100}),
        (TestScenario.CONCURRENT_USERS, {"concurrent_users": 20, "queries_per_user": 50}),
        (TestScenario.BATCH_QUERIES, {"concurrent_users": 5, "query_batch_size": 10})
    ]
    
    benchmark_results = {}
    
    for scenario, params in scenarios:
        config = PerformanceTestConfig(scenario=scenario, **params)
        comparison = await tester.compare_database_performance(
            source_db, target_db, config, "Source", "Target"
        )
        benchmark_results[scenario.value] = comparison
    
    return benchmark_results