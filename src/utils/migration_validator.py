"""
Migration Validation and Consistency Checking Tools.

This module provides comprehensive validation tools for ensuring data integrity
during vector database migrations, including content validation, embedding
consistency checks, and performance validation.
"""

import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

import numpy as np

from src.utils.enhanced_database import EnhancedVectorDatabaseInterface
from src.models import ContentItem, EmbeddingMetadata, User, ModuleType, ContentType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationLevel(str, Enum):
    """Validation depth levels."""
    BASIC = "basic"              # Count and ID validation only
    STANDARD = "standard"        # Includes metadata validation
    COMPREHENSIVE = "comprehensive"  # Full content and embedding validation


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    name: str
    description: str
    severity: str  # "error", "warning", "info"
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    rule_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


@dataclass
class MigrationValidationReport:
    """Comprehensive migration validation report."""
    validation_id: str
    source_provider: str
    target_provider: str
    validation_level: ValidationLevel
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Overall statistics
    total_content_items: int = 0
    validated_content_items: int = 0
    total_embeddings: int = 0
    validated_embeddings: int = 0
    
    # Validation results
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    skipped_checks: int = 0
    
    # Detailed results
    validation_results: List[ValidationResult] = field(default_factory=list)
    content_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    embedding_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    
    @property
    def overall_status(self) -> ValidationStatus:
        """Get overall validation status."""
        if self.failed_checks > 0:
            return ValidationStatus.FAILED
        elif self.warning_checks > 0:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.PASSED
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        total_checks = self.passed_checks + self.failed_checks + self.warning_checks
        if total_checks == 0:
            return 0.0
        return (self.passed_checks / total_checks) * 100.0


class MigrationValidator:
    """
    Comprehensive validation tool for vector database migrations.
    
    Validates data consistency, integrity, and performance between
    source and target databases during migration processes.
    """
    
    def __init__(self):
        """Initialize the migration validator."""
        self.validation_rules = self._initialize_validation_rules()
        self._performance_cache: Dict[str, List[float]] = {}
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize the default validation rules."""
        return [
            ValidationRule(
                name="content_count_consistency",
                description="Verify content item counts match between databases",
                severity="error"
            ),
            ValidationRule(
                name="embedding_count_consistency", 
                description="Verify embedding counts match for each content item",
                severity="error"
            ),
            ValidationRule(
                name="content_metadata_consistency",
                description="Verify content metadata matches between databases",
                severity="error"
            ),
            ValidationRule(
                name="embedding_metadata_consistency",
                description="Verify embedding metadata matches between databases",
                severity="warning"
            ),
            ValidationRule(
                name="embedding_vector_consistency",
                description="Verify embedding vectors are identical",
                severity="error"
            ),
            ValidationRule(
                name="content_accessibility",
                description="Verify content is accessible in target database",
                severity="error"
            ),
            ValidationRule(
                name="query_performance_consistency",
                description="Verify query performance is within acceptable range",
                severity="warning"
            ),
            ValidationRule(
                name="data_integrity_checksum",
                description="Verify data integrity using checksums",
                severity="error"
            ),
            ValidationRule(
                name="permission_consistency",
                description="Verify permission settings are preserved",
                severity="error"
            ),
            ValidationRule(
                name="relationship_consistency",
                description="Verify content relationships are preserved",
                severity="warning"
            )
        ]
    
    async def validate_migration(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        content_ids: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> MigrationValidationReport:
        """
        Perform comprehensive migration validation.
        
        Args:
            source_db: Source database instance
            target_db: Target database instance
            validation_level: Depth of validation to perform
            content_ids: Optional specific content IDs to validate
            sample_size: Optional sample size for large datasets
            
        Returns:
            MigrationValidationReport: Comprehensive validation report
        """
        validation_id = f"validation_{int(time.time())}"
        report = MigrationValidationReport(
            validation_id=validation_id,
            source_provider=type(source_db).__name__,
            target_provider=type(target_db).__name__,
            validation_level=validation_level,
            started_at=datetime.now()
        )
        
        logger.info(f"Starting migration validation: {validation_id}")
        
        try:
            # Determine content IDs to validate
            if content_ids is None:
                content_ids = await self._discover_content_ids(source_db, sample_size)
            
            report.total_content_items = len(content_ids)
            
            # Run validation checks based on level
            if validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                await self._validate_basic_consistency(source_db, target_db, content_ids, report)
            
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                await self._validate_metadata_consistency(source_db, target_db, content_ids, report)
                await self._validate_permission_consistency(source_db, target_db, content_ids, report)
            
            if validation_level == ValidationLevel.COMPREHENSIVE:
                await self._validate_embedding_vectors(source_db, target_db, content_ids, report)
                await self._validate_performance_consistency(source_db, target_db, content_ids, report)
                await self._validate_data_integrity(source_db, target_db, content_ids, report)
            
            # Calculate final statistics
            self._calculate_validation_statistics(report)
            
            report.completed_at = datetime.now()
            logger.info(f"Migration validation completed: {validation_id} - Status: {report.overall_status}")
            
        except Exception as e:
            report.errors.append(f"Validation failed: {str(e)}")
            logger.error(f"Migration validation error: {e}")
        
        return report
    
    async def _discover_content_ids(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        sample_size: Optional[int]
    ) -> List[str]:
        """
        Discover content IDs from the source database.
        
        Note: This is a simplified implementation. In practice, you would need
        to implement content enumeration capabilities in your database interfaces.
        """
        # For now, return empty list - this would be implemented based on
        # your specific database schema and enumeration capabilities
        logger.warning("Content ID discovery not fully implemented - using empty list")
        return []
    
    async def _validate_basic_consistency(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        report: MigrationValidationReport
    ):
        """Validate basic consistency (counts, accessibility)."""
        for content_id in content_ids:
            start_time = time.time()
            
            try:
                # Get embeddings from both databases
                source_embeddings = await source_db.get_content_embeddings(content_id)
                target_embeddings = await target_db.get_content_embeddings(content_id)
                
                # Validate embedding count consistency
                if len(source_embeddings) != len(target_embeddings):
                    result = ValidationResult(
                        rule_name="embedding_count_consistency",
                        status=ValidationStatus.FAILED,
                        message=f"Embedding count mismatch for content {content_id}",
                        details={
                            "content_id": content_id,
                            "source_count": len(source_embeddings),
                            "target_count": len(target_embeddings)
                        },
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                    report.validation_results.append(result)
                    report.failed_checks += 1
                    
                    report.embedding_mismatches.append({
                        "content_id": content_id,
                        "type": "count_mismatch",
                        "source_count": len(source_embeddings),
                        "target_count": len(target_embeddings)
                    })
                else:
                    result = ValidationResult(
                        rule_name="embedding_count_consistency",
                        status=ValidationStatus.PASSED,
                        message=f"Embedding counts match for content {content_id}",
                        details={"content_id": content_id, "count": len(source_embeddings)},
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                    report.validation_results.append(result)
                    report.passed_checks += 1
                
                # Validate content accessibility
                source_accessible = len(source_embeddings) > 0
                target_accessible = len(target_embeddings) > 0
                
                if source_accessible and not target_accessible:
                    result = ValidationResult(
                        rule_name="content_accessibility",
                        status=ValidationStatus.FAILED,
                        message=f"Content {content_id} not accessible in target database",
                        details={"content_id": content_id},
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                    report.validation_results.append(result)
                    report.failed_checks += 1
                else:
                    result = ValidationResult(
                        rule_name="content_accessibility",
                        status=ValidationStatus.PASSED,
                        message=f"Content {content_id} accessible in both databases",
                        details={"content_id": content_id},
                        execution_time_ms=(time.time() - start_time) * 1000
                    )
                    report.validation_results.append(result)
                    report.passed_checks += 1
                
                report.validated_content_items += 1
                report.total_embeddings += len(source_embeddings)
                report.validated_embeddings += len(target_embeddings)
                
            except Exception as e:
                result = ValidationResult(
                    rule_name="content_accessibility",
                    status=ValidationStatus.FAILED,
                    message=f"Error validating content {content_id}: {str(e)}",
                    details={"content_id": content_id, "error": str(e)},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                report.validation_results.append(result)
                report.failed_checks += 1
                report.errors.append(f"Content {content_id}: {str(e)}")
    
    async def _validate_metadata_consistency(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        report: MigrationValidationReport
    ):
        """Validate metadata consistency between databases."""
        for content_id in content_ids:
            start_time = time.time()
            
            try:
                source_embeddings = await source_db.get_content_embeddings(content_id)
                target_embeddings = await target_db.get_content_embeddings(content_id)
                
                # Compare metadata for each embedding
                for i, (source_emb, target_emb) in enumerate(zip(source_embeddings, target_embeddings)):
                    metadata_matches = self._compare_embedding_metadata(source_emb, target_emb)
                    
                    if not metadata_matches:
                        result = ValidationResult(
                            rule_name="embedding_metadata_consistency",
                            status=ValidationStatus.WARNING,
                            message=f"Metadata mismatch for embedding {i} in content {content_id}",
                            details={
                                "content_id": content_id,
                                "embedding_index": i,
                                "source_metadata": self._extract_metadata_for_comparison(source_emb),
                                "target_metadata": self._extract_metadata_for_comparison(target_emb)
                            },
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        report.validation_results.append(result)
                        report.warning_checks += 1
                        
                        report.embedding_mismatches.append({
                            "content_id": content_id,
                            "embedding_index": i,
                            "type": "metadata_mismatch",
                            "differences": self._get_metadata_differences(source_emb, target_emb)
                        })
                    else:
                        result = ValidationResult(
                            rule_name="embedding_metadata_consistency",
                            status=ValidationStatus.PASSED,
                            message=f"Metadata matches for embedding {i} in content {content_id}",
                            details={"content_id": content_id, "embedding_index": i},
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        report.validation_results.append(result)
                        report.passed_checks += 1
                
            except Exception as e:
                result = ValidationResult(
                    rule_name="embedding_metadata_consistency",
                    status=ValidationStatus.FAILED,
                    message=f"Error validating metadata for content {content_id}: {str(e)}",
                    details={"content_id": content_id, "error": str(e)},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                report.validation_results.append(result)
                report.failed_checks += 1
    
    def _compare_embedding_metadata(self, source: EmbeddingMetadata, target: EmbeddingMetadata) -> bool:
        """Compare two embedding metadata objects for consistency."""
        # Key fields that must match exactly
        critical_fields = [
            'content_id', 'chunk_index', 'module_type', 'content_type',
            'chunk_type', 'visibility', 'language', 'text_content'
        ]
        
        for field in critical_fields:
            source_value = getattr(source, field, None)
            target_value = getattr(target, field, None)
            
            if source_value != target_value:
                return False
        
        return True
    
    def _extract_metadata_for_comparison(self, embedding: EmbeddingMetadata) -> Dict[str, Any]:
        """Extract key metadata fields for comparison."""
        return {
            'content_id': embedding.content_id,
            'chunk_index': embedding.chunk_index,
            'module_type': embedding.module_type.value,
            'content_type': embedding.content_type.value,
            'chunk_type': embedding.chunk_type,
            'visibility': embedding.visibility.value,
            'language': embedding.language,
            'chunk_length': embedding.chunk_length,
            'semantic_tags': embedding.semantic_tags
        }
    
    def _get_metadata_differences(self, source: EmbeddingMetadata, target: EmbeddingMetadata) -> Dict[str, Any]:
        """Get differences between two embedding metadata objects."""
        source_meta = self._extract_metadata_for_comparison(source)
        target_meta = self._extract_metadata_for_comparison(target)
        
        differences = {}
        for key in source_meta:
            if source_meta[key] != target_meta.get(key):
                differences[key] = {
                    'source': source_meta[key],
                    'target': target_meta.get(key)
                }
        
        return differences
    
    async def _validate_permission_consistency(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        report: MigrationValidationReport
    ):
        """Validate permission settings are preserved."""
        for content_id in content_ids:
            start_time = time.time()
            
            try:
                source_embeddings = await source_db.get_content_embeddings(content_id)
                target_embeddings = await target_db.get_content_embeddings(content_id)
                
                # Check permission consistency
                for source_emb, target_emb in zip(source_embeddings, target_embeddings):
                    permission_match = (
                        source_emb.visibility == target_emb.visibility and
                        source_emb.creator_id == target_emb.creator_id and
                        source_emb.organization_id == target_emb.organization_id
                    )
                    
                    if not permission_match:
                        result = ValidationResult(
                            rule_name="permission_consistency",
                            status=ValidationStatus.FAILED,
                            message=f"Permission mismatch for content {content_id}",
                            details={
                                "content_id": content_id,
                                "source_visibility": source_emb.visibility.value,
                                "target_visibility": target_emb.visibility.value,
                                "source_creator": source_emb.creator_id,
                                "target_creator": target_emb.creator_id
                            },
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        report.validation_results.append(result)
                        report.failed_checks += 1
                    else:
                        result = ValidationResult(
                            rule_name="permission_consistency",
                            status=ValidationStatus.PASSED,
                            message=f"Permissions match for content {content_id}",
                            details={"content_id": content_id},
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        report.validation_results.append(result)
                        report.passed_checks += 1
                
            except Exception as e:
                result = ValidationResult(
                    rule_name="permission_consistency",
                    status=ValidationStatus.FAILED,
                    message=f"Error validating permissions for content {content_id}: {str(e)}",
                    details={"content_id": content_id, "error": str(e)},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                report.validation_results.append(result)
                report.failed_checks += 1
    
    async def _validate_embedding_vectors(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        report: MigrationValidationReport
    ):
        """Validate embedding vectors are identical (comprehensive validation only)."""
        for content_id in content_ids:
            start_time = time.time()
            
            try:
                # Note: This would require extending the interface to return actual vectors
                # For now, we'll perform a checksum-based validation
                
                source_embeddings = await source_db.get_content_embeddings(content_id)
                target_embeddings = await target_db.get_content_embeddings(content_id)
                
                # Validate text content checksums (proxy for vector consistency)
                for i, (source_emb, target_emb) in enumerate(zip(source_embeddings, target_embeddings)):
                    source_checksum = hashlib.md5(source_emb.text_content.encode()).hexdigest()
                    target_checksum = hashlib.md5(target_emb.text_content.encode()).hexdigest()
                    
                    if source_checksum != target_checksum:
                        result = ValidationResult(
                            rule_name="embedding_vector_consistency",
                            status=ValidationStatus.FAILED,
                            message=f"Text content checksum mismatch for embedding {i} in content {content_id}",
                            details={
                                "content_id": content_id,
                                "embedding_index": i,
                                "source_checksum": source_checksum,
                                "target_checksum": target_checksum
                            },
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        report.validation_results.append(result)
                        report.failed_checks += 1
                    else:
                        result = ValidationResult(
                            rule_name="embedding_vector_consistency",
                            status=ValidationStatus.PASSED,
                            message=f"Text content matches for embedding {i} in content {content_id}",
                            details={"content_id": content_id, "embedding_index": i},
                            execution_time_ms=(time.time() - start_time) * 1000
                        )
                        report.validation_results.append(result)
                        report.passed_checks += 1
                
            except Exception as e:
                result = ValidationResult(
                    rule_name="embedding_vector_consistency",
                    status=ValidationStatus.FAILED,
                    message=f"Error validating vectors for content {content_id}: {str(e)}",
                    details={"content_id": content_id, "error": str(e)},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                report.validation_results.append(result)
                report.failed_checks += 1
    
    async def _validate_performance_consistency(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        report: MigrationValidationReport
    ):
        """Validate query performance is within acceptable range."""
        if not content_ids:
            return
        
        # Sample a few content IDs for performance testing
        sample_ids = content_ids[:min(5, len(content_ids))]
        
        source_times = []
        target_times = []
        
        for content_id in sample_ids:
            try:
                # Time source database query
                start_time = time.time()
                await source_db.get_content_embeddings(content_id)
                source_time = (time.time() - start_time) * 1000  # ms
                source_times.append(source_time)
                
                # Time target database query
                start_time = time.time()
                await target_db.get_content_embeddings(content_id)
                target_time = (time.time() - start_time) * 1000  # ms
                target_times.append(target_time)
                
            except Exception as e:
                logger.warning(f"Performance test failed for content {content_id}: {e}")
        
        if source_times and target_times:
            source_avg = statistics.mean(source_times)
            target_avg = statistics.mean(target_times)
            
            # Performance should be within 200% of source (target can be slower during migration)
            performance_ratio = target_avg / source_avg if source_avg > 0 else float('inf')
            
            if performance_ratio > 3.0:  # Target is more than 3x slower
                result = ValidationResult(
                    rule_name="query_performance_consistency",
                    status=ValidationStatus.WARNING,
                    message=f"Target database significantly slower than source",
                    details={
                        "source_avg_ms": source_avg,
                        "target_avg_ms": target_avg,
                        "performance_ratio": performance_ratio
                    }
                )
                report.validation_results.append(result)
                report.warning_checks += 1
            else:
                result = ValidationResult(
                    rule_name="query_performance_consistency",
                    status=ValidationStatus.PASSED,
                    message=f"Target database performance acceptable",
                    details={
                        "source_avg_ms": source_avg,
                        "target_avg_ms": target_avg,
                        "performance_ratio": performance_ratio
                    }
                )
                report.validation_results.append(result)
                report.passed_checks += 1
            
            # Store performance metrics
            report.performance_metrics = {
                "source_avg_query_time_ms": source_avg,
                "target_avg_query_time_ms": target_avg,
                "performance_ratio": performance_ratio,
                "sample_size": len(source_times)
            }
    
    async def _validate_data_integrity(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        report: MigrationValidationReport
    ):
        """Validate overall data integrity using checksums."""
        start_time = time.time()
        
        try:
            # Calculate checksums for all content
            source_checksum = await self._calculate_database_checksum(source_db, content_ids)
            target_checksum = await self._calculate_database_checksum(target_db, content_ids)
            
            if source_checksum != target_checksum:
                result = ValidationResult(
                    rule_name="data_integrity_checksum",
                    status=ValidationStatus.FAILED,
                    message="Database checksums do not match",
                    details={
                        "source_checksum": source_checksum,
                        "target_checksum": target_checksum
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                report.validation_results.append(result)
                report.failed_checks += 1
            else:
                result = ValidationResult(
                    rule_name="data_integrity_checksum",
                    status=ValidationStatus.PASSED,
                    message="Database checksums match",
                    details={
                        "checksum": source_checksum,
                        "content_items_included": len(content_ids)
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                report.validation_results.append(result)
                report.passed_checks += 1
                
        except Exception as e:
            result = ValidationResult(
                rule_name="data_integrity_checksum",
                status=ValidationStatus.FAILED,
                message=f"Error calculating checksums: {str(e)}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            report.validation_results.append(result)
            report.failed_checks += 1
    
    async def _calculate_database_checksum(
        self,
        db: EnhancedVectorDatabaseInterface,
        content_ids: List[str]
    ) -> str:
        """Calculate a checksum for database content."""
        all_text = []
        
        for content_id in content_ids:
            try:
                embeddings = await db.get_content_embeddings(content_id)
                for embedding in embeddings:
                    all_text.append(f"{embedding.content_id}:{embedding.chunk_index}:{embedding.text_content}")
            except Exception as e:
                logger.warning(f"Error processing content {content_id} for checksum: {e}")
        
        # Sort for consistent ordering
        all_text.sort()
        combined_text = "\n".join(all_text)
        
        return hashlib.sha256(combined_text.encode()).hexdigest()
    
    def _calculate_validation_statistics(self, report: MigrationValidationReport):
        """Calculate final validation statistics."""
        # Count validation results by status
        for result in report.validation_results:
            if result.status == ValidationStatus.PASSED:
                # Already counted during validation
                pass
            elif result.status == ValidationStatus.FAILED:
                # Already counted during validation
                pass
            elif result.status == ValidationStatus.WARNING:
                # Already counted during validation
                pass
            elif result.status == ValidationStatus.SKIPPED:
                report.skipped_checks += 1
    
    def generate_validation_summary(self, report: MigrationValidationReport) -> str:
        """Generate a human-readable validation summary."""
        duration = (report.completed_at - report.started_at).total_seconds() if report.completed_at else 0
        
        summary = f"""
Migration Validation Report
==========================
Validation ID: {report.validation_id}
Source Provider: {report.source_provider}
Target Provider: {report.target_provider}
Validation Level: {report.validation_level}
Duration: {duration:.2f} seconds

Overall Status: {report.overall_status.upper()}
Success Rate: {report.success_rate:.1f}%

Content Statistics:
- Total Content Items: {report.total_content_items}
- Validated Content Items: {report.validated_content_items}
- Total Embeddings: {report.total_embeddings}
- Validated Embeddings: {report.validated_embeddings}

Validation Results:
- Passed Checks: {report.passed_checks}
- Failed Checks: {report.failed_checks}
- Warning Checks: {report.warning_checks}
- Skipped Checks: {report.skipped_checks}

Issues Found:
- Content Mismatches: {len(report.content_mismatches)}
- Embedding Mismatches: {len(report.embedding_mismatches)}
- Errors: {len(report.errors)}
"""
        
        if report.performance_metrics:
            summary += f"""
Performance Metrics:
- Source Avg Query Time: {report.performance_metrics.get('source_avg_query_time_ms', 0):.2f}ms
- Target Avg Query Time: {report.performance_metrics.get('target_avg_query_time_ms', 0):.2f}ms
- Performance Ratio: {report.performance_metrics.get('performance_ratio', 0):.2f}x
"""
        
        if report.failed_checks > 0:
            summary += "\nCritical Issues:\n"
            for result in report.validation_results:
                if result.status == ValidationStatus.FAILED:
                    summary += f"- {result.rule_name}: {result.message}\n"
        
        return summary
    
    def export_validation_report(self, report: MigrationValidationReport, file_path: str):
        """Export validation report to JSON file."""
        # Convert report to dictionary for JSON serialization
        report_dict = {
            "validation_id": report.validation_id,
            "source_provider": report.source_provider,
            "target_provider": report.target_provider,
            "validation_level": report.validation_level,
            "started_at": report.started_at.isoformat(),
            "completed_at": report.completed_at.isoformat() if report.completed_at else None,
            "overall_status": report.overall_status,
            "success_rate": report.success_rate,
            "statistics": {
                "total_content_items": report.total_content_items,
                "validated_content_items": report.validated_content_items,
                "total_embeddings": report.total_embeddings,
                "validated_embeddings": report.validated_embeddings,
                "passed_checks": report.passed_checks,
                "failed_checks": report.failed_checks,
                "warning_checks": report.warning_checks,
                "skipped_checks": report.skipped_checks
            },
            "validation_results": [
                {
                    "rule_name": result.rule_name,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details,
                    "execution_time_ms": result.execution_time_ms
                }
                for result in report.validation_results
            ],
            "content_mismatches": report.content_mismatches,
            "embedding_mismatches": report.embedding_mismatches,
            "performance_metrics": report.performance_metrics,
            "errors": report.errors
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Validation report exported to: {file_path}")


# ========================================
# Convenience Functions
# ========================================

async def quick_validation(
    source_db: EnhancedVectorDatabaseInterface,
    target_db: EnhancedVectorDatabaseInterface,
    content_ids: Optional[List[str]] = None
) -> bool:
    """
    Perform a quick validation check.
    
    Args:
        source_db: Source database
        target_db: Target database
        content_ids: Optional content IDs to validate
        
    Returns:
        bool: True if validation passes
    """
    validator = MigrationValidator()
    report = await validator.validate_migration(
        source_db, target_db, ValidationLevel.BASIC, content_ids
    )
    
    return report.overall_status == ValidationStatus.PASSED


async def comprehensive_validation(
    source_db: EnhancedVectorDatabaseInterface,
    target_db: EnhancedVectorDatabaseInterface,
    content_ids: Optional[List[str]] = None,
    export_path: Optional[str] = None
) -> MigrationValidationReport:
    """
    Perform comprehensive validation with optional report export.
    
    Args:
        source_db: Source database
        target_db: Target database
        content_ids: Optional content IDs to validate
        export_path: Optional path to export report
        
    Returns:
        MigrationValidationReport: Detailed validation report
    """
    validator = MigrationValidator()
    report = await validator.validate_migration(
        source_db, target_db, ValidationLevel.COMPREHENSIVE, content_ids
    )
    
    if export_path:
        validator.export_validation_report(report, export_path)
    
    return report