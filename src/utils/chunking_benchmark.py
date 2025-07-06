"""
Automated benchmarking system for chunking strategies (Phase 1.1).

This module implements comprehensive benchmarking and comparison of different
chunking strategies including performance metrics, quality assessment, and
automated optimization recommendations.
"""

import time
import json
import statistics
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
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
from src.utils.text_chunking import ChunkingManager, ChunkingConfig
from src.utils.enhanced_chunking import EnhancedSemanticChunker, ChunkingConfig as EnhancedChunkingConfig
from src.utils.retrieval_evaluation import RetrievalEvaluator, get_retrieval_evaluator

logger = get_logger(__name__)


class BenchmarkMetric(Enum):
    """Types of benchmarking metrics."""
    PROCESSING_TIME = "processing_time"
    CHUNK_COUNT = "chunk_count"
    AVERAGE_CHUNK_SIZE = "average_chunk_size"
    CHUNK_SIZE_VARIANCE = "chunk_size_variance"
    METADATA_RICHNESS = "metadata_richness"
    IMPORTANCE_DISTRIBUTION = "importance_distribution"
    COHERENCE_SCORES = "coherence_scores"
    RETRIEVAL_ACCURACY = "retrieval_accuracy"
    MEMORY_USAGE = "memory_usage"


@dataclass
class ChunkingPerformance:
    """Performance metrics for a chunking strategy."""
    
    # Strategy identification
    strategy_name: str
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metrics
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Output metrics
    total_chunks: int = 0
    average_chunk_size: float = 0.0
    chunk_size_std: float = 0.0
    chunk_size_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    average_importance_score: float = 0.0
    average_coherence_score: float = 0.0
    average_completeness_score: float = 0.0
    metadata_richness_score: float = 0.0
    
    # Structure metrics
    detected_headings: int = 0
    detected_sections: int = 0
    hierarchy_levels: int = 0
    
    # Error metrics
    failed_chunks: int = 0
    error_rate: float = 0.0
    
    # Timestamp
    benchmark_timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score combining time and quality."""
        # Normalize processing time (lower is better)
        time_score = max(0.0, 1.0 - (self.processing_time / 10.0))  # Assume 10s is poor
        
        # Quality score (higher is better)
        quality_score = (
            self.average_importance_score * 0.3 +
            self.average_coherence_score * 0.3 +
            self.average_completeness_score * 0.2 +
            self.metadata_richness_score * 0.2
        )
        
        # Combine time and quality (balanced weighting)
        efficiency = (time_score * 0.4) + (quality_score * 0.6)
        return max(0.0, min(1.0, efficiency))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage and analysis."""
        result = asdict(self)
        result["benchmark_timestamp"] = self.benchmark_timestamp.isoformat()
        result["efficiency_score"] = self.calculate_efficiency_score()
        return result


@dataclass
class BenchmarkDataset:
    """Dataset for benchmarking chunking strategies."""
    
    name: str
    description: str
    documents: List[Document] = field(default_factory=list)
    
    # Dataset characteristics
    total_documents: int = 0
    total_length: int = 0
    average_document_length: float = 0.0
    document_types: Dict[str, int] = field(default_factory=dict)
    
    # Expected results (for validation)
    expected_chunk_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # strategy -> (min, max)
    ground_truth_sections: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_document(self, document: Document):
        """Add a document to the benchmark dataset."""
        self.documents.append(document)
        self.total_documents += 1
        self.total_length += len(document.page_content)
        
        # Track document type
        doc_type = document.metadata.get('file_type', 'unknown')
        self.document_types[doc_type] = self.document_types.get(doc_type, 0) + 1
        
        # Recalculate average
        self.average_document_length = self.total_length / self.total_documents
    
    def validate_dataset(self) -> bool:
        """Validate dataset for benchmarking."""
        if not self.documents:
            logger.warning(f"Dataset '{self.name}' has no documents")
            return False
        
        if self.total_length < 1000:
            logger.warning(f"Dataset '{self.name}' is very small ({self.total_length} chars)")
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        doc_lengths = [len(doc.page_content) for doc in self.documents]
        
        return {
            "name": self.name,
            "total_documents": self.total_documents,
            "total_length": self.total_length,
            "average_length": self.average_document_length,
            "min_length": min(doc_lengths) if doc_lengths else 0,
            "max_length": max(doc_lengths) if doc_lengths else 0,
            "std_length": statistics.stdev(doc_lengths) if len(doc_lengths) > 1 else 0,
            "document_types": self.document_types
        }


class ChunkingBenchmark:
    """
    Comprehensive benchmarking system for chunking strategies.
    
    This class provides automated testing and comparison of different
    chunking approaches with detailed performance and quality metrics.
    """
    
    def __init__(self):
        """Initialize the chunking benchmark system."""
        self.settings = get_settings()
        self.chunking_manager = ChunkingManager()
        self.retrieval_evaluator = get_retrieval_evaluator()
        
        # Available strategies to benchmark
        self.strategies = {
            'recursive': self._benchmark_recursive_strategy,
            'semantic': self._benchmark_semantic_strategy,
            'enhanced_semantic': self._benchmark_enhanced_semantic_strategy
        }
        
        # Benchmark results storage
        self.benchmark_results: Dict[str, List[ChunkingPerformance]] = {}
        self.datasets: Dict[str, BenchmarkDataset] = {}
    
    def create_test_dataset(self, name: str, description: str) -> BenchmarkDataset:
        """Create a new test dataset for benchmarking."""
        dataset = BenchmarkDataset(name=name, description=description)
        self.datasets[name] = dataset
        return dataset
    
    def add_sample_datasets(self):
        """Add sample datasets for testing."""
        # Short text dataset
        short_dataset = self.create_test_dataset(
            "short_texts",
            "Short documents for testing basic chunking"
        )
        
        short_docs = [
            Document(
                page_content="This is a short paragraph about machine learning. It covers basic concepts.",
                metadata={"file_type": "txt", "length": "short"}
            ),
            Document(
                page_content="Another brief text about algorithms and data structures. This is also concise.",
                metadata={"file_type": "txt", "length": "short"}
            )
        ]
        
        for doc in short_docs:
            short_dataset.add_document(doc)
        
        # Structured document dataset
        structured_dataset = self.create_test_dataset(
            "structured_docs",
            "Structured documents with headings and sections"
        )
        
        structured_text = """# Chapter 1: Introduction to Data Science

Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.

## 1.1 What is Data Science?

Data science combines multiple fields including statistics, computer science, and domain expertise to analyze and interpret complex data.

### Key Components

The main components of data science include:
- Data collection and cleaning
- Exploratory data analysis
- Statistical modeling
- Machine learning
- Data visualization

## 1.2 Applications

Data science has applications in:
1. Business analytics
2. Healthcare research
3. Financial modeling
4. Social media analysis

# Chapter 2: Statistical Foundations

Understanding statistics is crucial for data science practitioners.

## 2.1 Descriptive Statistics

Descriptive statistics help summarize and describe data characteristics.

## 2.2 Inferential Statistics

Inferential statistics allow us to make predictions and inferences about populations from sample data."""
        
        structured_doc = Document(
            page_content=structured_text,
            metadata={"file_type": "md", "length": "medium", "has_structure": True}
        )
        
        structured_dataset.add_document(structured_doc)
        
        # Long unstructured text dataset
        long_dataset = self.create_test_dataset(
            "long_unstructured",
            "Long documents without clear structure"
        )
        
        long_text = """Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.""" * 5  # Repeat to make it longer
        
        long_doc = Document(
            page_content=long_text,
            metadata={"file_type": "txt", "length": "long", "has_structure": False}
        )
        
        long_dataset.add_document(long_doc)
    
    def benchmark_strategy(
        self,
        strategy_name: str,
        dataset: BenchmarkDataset,
        config: Optional[Dict[str, Any]] = None
    ) -> ChunkingPerformance:
        """
        Benchmark a specific chunking strategy on a dataset.
        
        Args:
            strategy_name: Name of the strategy to benchmark
            dataset: Dataset to test on
            config: Optional strategy configuration
            
        Returns:
            ChunkingPerformance: Performance metrics
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        if not dataset.validate_dataset():
            raise ValueError(f"Invalid dataset: {dataset.name}")
        
        logger.info(f"Benchmarking strategy '{strategy_name}' on dataset '{dataset.name}'")
        
        # Initialize performance tracking
        performance = ChunkingPerformance(
            strategy_name=strategy_name,
            configuration=config or {}
        )
        
        # Track memory usage (simplified)
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure processing time
        start_time = time.time()
        
        try:
            # Run the specific strategy
            benchmark_func = self.strategies[strategy_name]
            results = benchmark_func(dataset, config)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Track memory usage after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # Update performance metrics
            performance.processing_time = processing_time
            performance.memory_usage_mb = max(0, memory_usage)
            
            # Analyze results
            self._analyze_chunking_results(performance, results)
            
            logger.info(f"Benchmarking completed in {processing_time:.3f}s, "
                       f"created {performance.total_chunks} chunks")
            
        except Exception as e:
            logger.error(f"Benchmarking failed for strategy '{strategy_name}': {e}")
            performance.error_rate = 1.0
            performance.failed_chunks = len(dataset.documents)
        
        # Store results
        if strategy_name not in self.benchmark_results:
            self.benchmark_results[strategy_name] = []
        self.benchmark_results[strategy_name].append(performance)
        
        return performance
    
    def benchmark_all_strategies(
        self,
        dataset: BenchmarkDataset,
        configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, ChunkingPerformance]:
        """
        Benchmark all available strategies on a dataset.
        
        Args:
            dataset: Dataset to test on
            configs: Optional configurations for each strategy
            
        Returns:
            Dict mapping strategy names to performance results
        """
        results = {}
        configs = configs or {}
        
        for strategy_name in self.strategies.keys():
            try:
                config = configs.get(strategy_name, {})
                performance = self.benchmark_strategy(strategy_name, dataset, config)
                results[strategy_name] = performance
            except Exception as e:
                logger.error(f"Failed to benchmark strategy '{strategy_name}': {e}")
        
        return results
    
    def compare_strategies(
        self,
        dataset_name: str,
        strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare performance of different strategies.
        
        Args:
            dataset_name: Name of dataset to compare on
            strategies: List of strategies to compare (default: all)
            
        Returns:
            Comparison results with rankings and recommendations
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        strategies = strategies or list(self.strategies.keys())
        
        # Run benchmarks for all strategies
        results = self.benchmark_all_strategies(dataset)
        
        # Filter to requested strategies
        filtered_results = {k: v for k, v in results.items() if k in strategies}
        
        if not filtered_results:
            return {"error": "No valid results for comparison"}
        
        # Calculate rankings
        rankings = self._calculate_strategy_rankings(filtered_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(filtered_results, dataset)
        
        # Create comparison report
        comparison = {
            "dataset": dataset.get_summary(),
            "strategies_compared": list(filtered_results.keys()),
            "performance_metrics": {
                name: perf.to_dict() for name, perf in filtered_results.items()
            },
            "rankings": rankings,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        return comparison
    
    def _benchmark_recursive_strategy(
        self,
        dataset: BenchmarkDataset,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Benchmark recursive chunking strategy."""
        chunking_config = ChunkingConfig(**(config or {}))
        
        all_chunks = []
        for document in dataset.documents:
            chunks = self.chunking_manager.chunk_documents(
                [document], 
                strategy='recursive',
                config=chunking_config
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _benchmark_semantic_strategy(
        self,
        dataset: BenchmarkDataset,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Benchmark semantic chunking strategy."""
        chunking_config = ChunkingConfig(**(config or {}))
        
        all_chunks = []
        for document in dataset.documents:
            chunks = self.chunking_manager.chunk_documents(
                [document],
                strategy='semantic',
                config=chunking_config
            )
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _benchmark_enhanced_semantic_strategy(
        self,
        dataset: BenchmarkDataset,
        config: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Benchmark enhanced semantic chunking strategy."""
        try:
            from src.utils.enhanced_chunking import EnhancedSemanticChunker, ChunkingConfig as EnhancedConfig
            
            enhanced_config = EnhancedConfig(**(config or {}))
            chunker = EnhancedSemanticChunker(enhanced_config)
            
            all_chunks = []
            for document in dataset.documents:
                chunks = chunker.chunk_documents([document], enhanced_config)
                all_chunks.extend(chunks)
            
            return all_chunks
            
        except ImportError:
            logger.warning("Enhanced chunking not available, falling back to semantic")
            return self._benchmark_semantic_strategy(dataset, config)
    
    def _analyze_chunking_results(
        self,
        performance: ChunkingPerformance,
        chunks: List[Document]
    ):
        """Analyze chunking results and update performance metrics."""
        if not chunks:
            return
        
        # Basic metrics
        performance.total_chunks = len(chunks)
        
        # Chunk size analysis
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        performance.average_chunk_size = statistics.mean(chunk_sizes)
        performance.chunk_size_std = statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0
        
        # Size distribution
        size_ranges = {
            "very_small": 0,  # < 200 chars
            "small": 0,       # 200-500 chars
            "medium": 0,      # 500-1000 chars
            "large": 0,       # 1000-2000 chars
            "very_large": 0   # > 2000 chars
        }
        
        for size in chunk_sizes:
            if size < 200:
                size_ranges["very_small"] += 1
            elif size < 500:
                size_ranges["small"] += 1
            elif size < 1000:
                size_ranges["medium"] += 1
            elif size < 2000:
                size_ranges["large"] += 1
            else:
                size_ranges["very_large"] += 1
        
        performance.chunk_size_distribution = size_ranges
        
        # Quality metrics from metadata
        importance_scores = []
        coherence_scores = []
        completeness_scores = []
        metadata_richness = []
        
        structure_metrics = {
            "headings": 0,
            "sections": 0,
            "hierarchy_levels": set()
        }
        
        for chunk in chunks:
            metadata = chunk.metadata
            
            # Extract quality scores
            importance = metadata.get("importance_score")
            if importance is not None:
                importance_scores.append(importance)
            
            coherence = metadata.get("coherence_score")
            if coherence is not None:
                coherence_scores.append(coherence)
            
            completeness = metadata.get("completeness_score")
            if completeness is not None:
                completeness_scores.append(completeness)
            
            # Metadata richness (number of metadata fields)
            metadata_richness.append(len(metadata))
            
            # Structure analysis
            content_type = metadata.get("content_type", "")
            if "heading" in str(content_type).lower():
                structure_metrics["headings"] += 1
            
            hierarchy_level = metadata.get("hierarchy_level")
            if hierarchy_level is not None:
                structure_metrics["hierarchy_levels"].add(hierarchy_level)
        
        # Update performance with averages
        if importance_scores:
            performance.average_importance_score = statistics.mean(importance_scores)
        
        if coherence_scores:
            performance.average_coherence_score = statistics.mean(coherence_scores)
        
        if completeness_scores:
            performance.average_completeness_score = statistics.mean(completeness_scores)
        
        if metadata_richness:
            # Normalize richness score (assume max 20 metadata fields)
            avg_richness = statistics.mean(metadata_richness)
            performance.metadata_richness_score = min(1.0, avg_richness / 20.0)
        
        # Structure metrics
        performance.detected_headings = structure_metrics["headings"]
        performance.hierarchy_levels = len(structure_metrics["hierarchy_levels"])
    
    def _calculate_strategy_rankings(
        self,
        results: Dict[str, ChunkingPerformance]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate rankings for different metrics."""
        if not results:
            return {}
        
        rankings = {}
        
        # Metrics to rank (lower is better for time, higher is better for quality)
        metrics = {
            "processing_time": {"reverse": False, "label": "Processing Speed"},
            "average_chunk_size": {"reverse": False, "label": "Chunk Size Consistency"},
            "average_importance_score": {"reverse": True, "label": "Content Importance"},
            "average_coherence_score": {"reverse": True, "label": "Text Coherence"},
            "metadata_richness_score": {"reverse": True, "label": "Metadata Quality"},
            "efficiency_score": {"reverse": True, "label": "Overall Efficiency"}
        }
        
        for metric, config in metrics.items():
            # Get values for this metric
            values = []
            for strategy, performance in results.items():
                if metric == "efficiency_score":
                    value = performance.calculate_efficiency_score()
                else:
                    value = getattr(performance, metric, 0)
                values.append((strategy, value))
            
            # Sort by metric value
            values.sort(key=lambda x: x[1], reverse=config["reverse"])
            
            # Create ranking
            ranking = []
            for i, (strategy, value) in enumerate(values):
                ranking.append({
                    "rank": i + 1,
                    "strategy": strategy,
                    "value": value,
                    "score": 1.0 - (i / len(values))  # Normalized score
                })
            
            rankings[metric] = {
                "label": config["label"],
                "ranking": ranking
            }
        
        return rankings
    
    def _generate_recommendations(
        self,
        results: Dict[str, ChunkingPerformance],
        dataset: BenchmarkDataset
    ) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if not results:
            return ["No results available for recommendations"]
        
        # Find best overall strategy
        best_efficiency = 0
        best_strategy = None
        
        for strategy, performance in results.items():
            efficiency = performance.calculate_efficiency_score()
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_strategy = strategy
        
        if best_strategy:
            recommendations.append(
                f"Overall best strategy: '{best_strategy}' with {best_efficiency:.3f} efficiency score"
            )
        
        # Speed recommendations
        fastest_time = float('inf')
        fastest_strategy = None
        
        for strategy, performance in results.items():
            if performance.processing_time < fastest_time:
                fastest_time = performance.processing_time
                fastest_strategy = strategy
        
        if fastest_strategy and fastest_strategy != best_strategy:
            recommendations.append(
                f"Fastest processing: '{fastest_strategy}' at {fastest_time:.3f}s"
            )
        
        # Quality recommendations
        highest_quality = 0
        quality_strategy = None
        
        for strategy, performance in results.items():
            quality_score = (
                performance.average_importance_score * 0.4 +
                performance.average_coherence_score * 0.4 +
                performance.metadata_richness_score * 0.2
            )
            
            if quality_score > highest_quality:
                highest_quality = quality_score
                quality_strategy = strategy
        
        if quality_strategy and quality_strategy not in [best_strategy, fastest_strategy]:
            recommendations.append(
                f"Highest quality: '{quality_strategy}' with {highest_quality:.3f} quality score"
            )
        
        # Dataset-specific recommendations
        if dataset.average_document_length > 5000:
            recommendations.append(
                "For long documents, consider enhanced_semantic strategy for better structure analysis"
            )
        elif dataset.average_document_length < 1000:
            recommendations.append(
                "For short documents, recursive strategy may be sufficient and faster"
            )
        
        # Structure-based recommendations
        structured_docs = sum(1 for doc in dataset.documents 
                            if doc.metadata.get("has_structure", False))
        
        if structured_docs > len(dataset.documents) * 0.5:
            recommendations.append(
                "Dataset has structured content - enhanced_semantic strategy recommended"
            )
        
        return recommendations
    
    def export_benchmark_results(self, filename: Optional[str] = None) -> str:
        """Export benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chunking_benchmark_{timestamp}.json"
        
        export_data = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "datasets": {name: dataset.get_summary() for name, dataset in self.datasets.items()},
            "results": {
                strategy: [perf.to_dict() for perf in performances]
                for strategy, performances in self.benchmark_results.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filename}")
        return filename
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}
        
        summary = {
            "total_strategies": len(self.benchmark_results),
            "total_datasets": len(self.datasets),
            "strategies": {}
        }
        
        for strategy, performances in self.benchmark_results.items():
            if not performances:
                continue
            
            # Calculate averages across all runs
            avg_time = statistics.mean([p.processing_time for p in performances])
            avg_chunks = statistics.mean([p.total_chunks for p in performances])
            avg_efficiency = statistics.mean([p.calculate_efficiency_score() for p in performances])
            
            summary["strategies"][strategy] = {
                "runs": len(performances),
                "average_processing_time": avg_time,
                "average_chunks_created": avg_chunks,
                "average_efficiency_score": avg_efficiency,
                "latest_run": performances[-1].benchmark_timestamp.isoformat()
            }
        
        return summary


# Global benchmark instance
_chunking_benchmark: Optional[ChunkingBenchmark] = None


def get_chunking_benchmark() -> ChunkingBenchmark:
    """Get the global chunking benchmark instance."""
    global _chunking_benchmark
    
    if _chunking_benchmark is None:
        _chunking_benchmark = ChunkingBenchmark()
    
    return _chunking_benchmark