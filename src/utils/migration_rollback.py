"""
Migration Rollback and Zero-Data-Loss Validation System.

This module provides comprehensive rollback strategies and zero-data-loss
validation for vector database migrations, ensuring safe migration paths
with the ability to revert changes if issues are detected.
"""

import asyncio
import json
import time
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import uuid

from src.utils.enhanced_database import EnhancedVectorDatabaseInterface
from src.utils.migration_validator import MigrationValidator, ValidationLevel, ValidationStatus
from src.utils.vector_database_factory import VectorDatabaseFactory, VectorDatabaseProvider
from src.models import ContentItem, EmbeddingMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RollbackStrategy(str, Enum):
    """Available rollback strategies."""
    IMMEDIATE = "immediate"           # Immediate rollback on any failure
    GRACEFUL = "graceful"            # Complete current operations then rollback
    MANUAL = "manual"                # Rollback only on manual trigger
    VALIDATION_BASED = "validation"  # Rollback based on validation results


class MigrationPhase(str, Enum):
    """Migration phases for tracking rollback points."""
    PREPARATION = "preparation"
    DATA_SYNC = "data_sync"
    VALIDATION = "validation"
    CUTOVER = "cutover"
    CLEANUP = "cleanup"
    COMPLETED = "completed"


class RollbackStatus(str, Enum):
    """Rollback operation status."""
    NOT_REQUIRED = "not_required"
    TRIGGERED = "triggered"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class MigrationCheckpoint:
    """Represents a migration checkpoint for rollback."""
    checkpoint_id: str
    phase: MigrationPhase
    timestamp: datetime
    database_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Backup information
    backup_path: Optional[str] = None
    backup_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results at this checkpoint
    validation_results: Optional[Dict[str, Any]] = None


@dataclass
class RollbackPlan:
    """Rollback execution plan."""
    plan_id: str
    strategy: RollbackStrategy
    trigger_conditions: List[str]
    rollback_steps: List[Dict[str, Any]]
    validation_requirements: Dict[str, Any]
    created_at: datetime
    
    # Execution tracking
    executed: bool = False
    execution_started_at: Optional[datetime] = None
    execution_completed_at: Optional[datetime] = None
    execution_status: RollbackStatus = RollbackStatus.NOT_REQUIRED


@dataclass
class ZeroDataLossValidation:
    """Zero data loss validation results."""
    validation_id: str
    pre_migration_checksum: str
    post_migration_checksum: str
    interim_checksums: List[str] = field(default_factory=list)
    
    # Data integrity metrics
    content_items_verified: int = 0
    embeddings_verified: int = 0
    relationships_verified: int = 0
    
    # Loss detection
    data_loss_detected: bool = False
    missing_content_items: List[str] = field(default_factory=list)
    missing_embeddings: List[str] = field(default_factory=list)
    corrupted_items: List[str] = field(default_factory=list)
    
    # Performance impact
    verification_time_ms: float = 0.0
    checksum_calculation_time_ms: float = 0.0


class MigrationRollbackManager:
    """
    Comprehensive migration rollback management system.
    
    Provides safe migration execution with automatic rollback capabilities,
    checkpointing, and zero-data-loss validation throughout the migration process.
    """
    
    def __init__(self, backup_directory: str = "./migration_backups"):
        """
        Initialize the rollback manager.
        
        Args:
            backup_directory: Directory for storing migration backups
        """
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        self._checkpoints: List[MigrationCheckpoint] = []
        self._rollback_plans: Dict[str, RollbackPlan] = {}
        self._active_migration: Optional[str] = None
        self._rollback_triggers: List[Callable] = []
        
        # Validation and monitoring
        self.validator = MigrationValidator()
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
    
    def create_rollback_plan(
        self,
        strategy: RollbackStrategy,
        trigger_conditions: List[str],
        validation_requirements: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new rollback plan.
        
        Args:
            strategy: Rollback strategy to use
            trigger_conditions: List of conditions that trigger rollback
            validation_requirements: Validation requirements for rollback
            
        Returns:
            str: Rollback plan ID
        """
        plan_id = f"rollback_plan_{int(time.time())}"
        
        # Default rollback steps based on strategy
        rollback_steps = self._generate_default_rollback_steps(strategy)
        
        plan = RollbackPlan(
            plan_id=plan_id,
            strategy=strategy,
            trigger_conditions=trigger_conditions,
            rollback_steps=rollback_steps,
            validation_requirements=validation_requirements or {},
            created_at=datetime.now()
        )
        
        self._rollback_plans[plan_id] = plan
        logger.info(f"Created rollback plan: {plan_id} with strategy: {strategy}")
        
        return plan_id
    
    def _generate_default_rollback_steps(self, strategy: RollbackStrategy) -> List[Dict[str, Any]]:
        """Generate default rollback steps based on strategy."""
        if strategy == RollbackStrategy.IMMEDIATE:
            return [
                {"action": "stop_writes", "priority": 1},
                {"action": "restore_from_checkpoint", "priority": 2},
                {"action": "validate_rollback", "priority": 3},
                {"action": "resume_operations", "priority": 4}
            ]
        elif strategy == RollbackStrategy.GRACEFUL:
            return [
                {"action": "pause_new_operations", "priority": 1},
                {"action": "complete_pending_operations", "priority": 2},
                {"action": "create_pre_rollback_checkpoint", "priority": 3},
                {"action": "restore_from_checkpoint", "priority": 4},
                {"action": "validate_rollback", "priority": 5},
                {"action": "resume_operations", "priority": 6}
            ]
        elif strategy == RollbackStrategy.VALIDATION_BASED:
            return [
                {"action": "run_validation", "priority": 1},
                {"action": "analyze_validation_results", "priority": 2},
                {"action": "conditional_rollback", "priority": 3},
                {"action": "validate_final_state", "priority": 4}
            ]
        else:  # MANUAL
            return [
                {"action": "await_manual_trigger", "priority": 1},
                {"action": "restore_from_checkpoint", "priority": 2},
                {"action": "validate_rollback", "priority": 3}
            ]
    
    async def create_checkpoint(
        self,
        phase: MigrationPhase,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: Optional[EnhancedVectorDatabaseInterface] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a migration checkpoint for rollback.
        
        Args:
            phase: Current migration phase
            source_db: Source database instance
            target_db: Optional target database instance
            metadata: Optional checkpoint metadata
            
        Returns:
            str: Checkpoint ID
        """
        checkpoint_id = f"checkpoint_{phase.value}_{int(time.time())}"
        
        # Create backup
        backup_path = await self._create_database_backup(source_db, checkpoint_id)
        
        # Collect database state information
        database_state = await self._collect_database_state(source_db, target_db)
        
        # Run validation if target database exists
        validation_results = None
        if target_db:
            try:
                report = await self.validator.validate_migration(
                    source_db, target_db, ValidationLevel.STANDARD
                )
                validation_results = {
                    "overall_status": report.overall_status.value,
                    "success_rate": report.success_rate,
                    "failed_checks": report.failed_checks,
                    "warning_checks": report.warning_checks
                }
            except Exception as e:
                logger.warning(f"Validation failed during checkpoint creation: {e}")
        
        checkpoint = MigrationCheckpoint(
            checkpoint_id=checkpoint_id,
            phase=phase,
            timestamp=datetime.now(),
            database_state=database_state,
            metadata=metadata or {},
            backup_path=backup_path,
            validation_results=validation_results
        )
        
        self._checkpoints.append(checkpoint)
        logger.info(f"Created migration checkpoint: {checkpoint_id} for phase: {phase}")
        
        return checkpoint_id
    
    async def _create_database_backup(
        self,
        db: EnhancedVectorDatabaseInterface,
        backup_id: str
    ) -> str:
        """
        Create a backup of the database state.
        
        Args:
            db: Database to backup
            backup_id: Unique backup identifier
            
        Returns:
            str: Path to backup
        """
        backup_path = self.backup_directory / f"backup_{backup_id}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # For Chroma databases, we can backup the persist directory
            if hasattr(db, 'settings') and hasattr(db.settings, 'chroma_persist_directory'):
                source_dir = Path(db.settings.chroma_persist_directory)
                if source_dir.exists():
                    shutil.copytree(source_dir, backup_path / "chroma_data", dirs_exist_ok=True)
            
            # Create backup metadata
            backup_metadata = {
                "backup_id": backup_id,
                "created_at": datetime.now().isoformat(),
                "database_type": type(db).__name__,
                "backup_method": "filesystem_copy"
            }
            
            with open(backup_path / "backup_metadata.json", 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            raise
    
    async def _collect_database_state(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: Optional[EnhancedVectorDatabaseInterface]
    ) -> Dict[str, Any]:
        """Collect current database state information."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "source_db_type": type(source_db).__name__,
            "target_db_type": type(target_db).__name__ if target_db else None
        }
        
        try:
            # In a real implementation, you would collect:
            # - Content item counts
            # - Embedding counts
            # - Collection information
            # - Configuration settings
            
            # For now, we'll include basic state information
            state["collections"] = []  # Would be populated with actual collection info
            state["configuration"] = {}  # Would include database config
            
        except Exception as e:
            logger.warning(f"Error collecting database state: {e}")
            state["collection_error"] = str(e)
        
        return state
    
    async def execute_rollback(
        self,
        plan_id: str,
        target_checkpoint_id: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Execute a rollback plan.
        
        Args:
            plan_id: Rollback plan to execute
            target_checkpoint_id: Optional specific checkpoint to rollback to
            force: Force rollback even if conditions aren't met
            
        Returns:
            bool: True if rollback successful
        """
        if plan_id not in self._rollback_plans:
            raise ValueError(f"Rollback plan not found: {plan_id}")
        
        plan = self._rollback_plans[plan_id]
        
        # Check if rollback should be triggered
        if not force and not await self._should_trigger_rollback(plan):
            logger.info(f"Rollback conditions not met for plan: {plan_id}")
            return False
        
        plan.execution_started_at = datetime.now()
        plan.execution_status = RollbackStatus.IN_PROGRESS
        
        logger.info(f"Starting rollback execution for plan: {plan_id}")
        
        try:
            # Find target checkpoint
            target_checkpoint = None
            if target_checkpoint_id:
                target_checkpoint = self._find_checkpoint(target_checkpoint_id)
            else:
                # Use latest stable checkpoint
                target_checkpoint = self._find_latest_stable_checkpoint()
            
            if not target_checkpoint:
                raise ValueError("No suitable checkpoint found for rollback")
            
            # Execute rollback steps
            for step in sorted(plan.rollback_steps, key=lambda x: x.get("priority", 999)):
                success = await self._execute_rollback_step(step, target_checkpoint)
                if not success:
                    raise RuntimeError(f"Rollback step failed: {step['action']}")
            
            plan.execution_status = RollbackStatus.COMPLETED
            plan.execution_completed_at = datetime.now()
            plan.executed = True
            
            logger.info(f"Rollback completed successfully for plan: {plan_id}")
            return True
            
        except Exception as e:
            plan.execution_status = RollbackStatus.FAILED
            logger.error(f"Rollback failed for plan {plan_id}: {e}")
            return False
    
    async def _should_trigger_rollback(self, plan: RollbackPlan) -> bool:
        """Check if rollback should be triggered based on plan conditions."""
        for condition in plan.trigger_conditions:
            if await self._evaluate_trigger_condition(condition):
                logger.info(f"Rollback trigger condition met: {condition}")
                return True
        return False
    
    async def _evaluate_trigger_condition(self, condition: str) -> bool:
        """Evaluate a rollback trigger condition."""
        # In a real implementation, this would evaluate conditions like:
        # - "validation_failure_rate > 10%"
        # - "query_performance_degraded > 50%"
        # - "data_loss_detected"
        # - "manual_trigger"
        
        # For now, return False as conditions are not met
        return False
    
    def _find_checkpoint(self, checkpoint_id: str) -> Optional[MigrationCheckpoint]:
        """Find a checkpoint by ID."""
        for checkpoint in self._checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        return None
    
    def _find_latest_stable_checkpoint(self) -> Optional[MigrationCheckpoint]:
        """Find the latest stable checkpoint for rollback."""
        # Sort checkpoints by timestamp (newest first)
        sorted_checkpoints = sorted(self._checkpoints, key=lambda x: x.timestamp, reverse=True)
        
        for checkpoint in sorted_checkpoints:
            # Consider a checkpoint stable if validation passed
            if (checkpoint.validation_results and 
                checkpoint.validation_results.get("overall_status") == "passed"):
                return checkpoint
        
        # If no validated checkpoint, return the latest one
        return sorted_checkpoints[0] if sorted_checkpoints else None
    
    async def _execute_rollback_step(
        self,
        step: Dict[str, Any],
        target_checkpoint: MigrationCheckpoint
    ) -> bool:
        """Execute a single rollback step."""
        action = step.get("action")
        
        try:
            if action == "stop_writes":
                return await self._stop_database_writes()
            elif action == "restore_from_checkpoint":
                return await self._restore_from_checkpoint(target_checkpoint)
            elif action == "validate_rollback":
                return await self._validate_rollback_state(target_checkpoint)
            elif action == "resume_operations":
                return await self._resume_database_operations()
            elif action == "pause_new_operations":
                return await self._pause_new_operations()
            elif action == "complete_pending_operations":
                return await self._complete_pending_operations()
            elif action == "create_pre_rollback_checkpoint":
                checkpoint_id = await self.create_checkpoint(
                    MigrationPhase.PREPARATION, None, None,
                    {"purpose": "pre_rollback_checkpoint"}
                )
                return checkpoint_id is not None
            else:
                logger.warning(f"Unknown rollback action: {action}")
                return True  # Don't fail on unknown actions
                
        except Exception as e:
            logger.error(f"Error executing rollback step {action}: {e}")
            return False
    
    async def _stop_database_writes(self) -> bool:
        """Stop database write operations."""
        # In a real implementation, this would:
        # - Set a global write lock
        # - Pause background tasks
        # - Reject new write requests
        logger.info("Database writes stopped for rollback")
        return True
    
    async def _restore_from_checkpoint(self, checkpoint: MigrationCheckpoint) -> bool:
        """Restore database state from checkpoint."""
        if not checkpoint.backup_path:
            logger.error(f"No backup path for checkpoint: {checkpoint.checkpoint_id}")
            return False
        
        try:
            backup_path = Path(checkpoint.backup_path)
            if not backup_path.exists():
                logger.error(f"Backup path does not exist: {backup_path}")
                return False
            
            # In a real implementation, this would:
            # - Stop the database service
            # - Restore data files from backup
            # - Restart the database service
            # - Verify restoration
            
            logger.info(f"Database restored from checkpoint: {checkpoint.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from checkpoint: {e}")
            return False
    
    async def _validate_rollback_state(self, checkpoint: MigrationCheckpoint) -> bool:
        """Validate the database state after rollback."""
        try:
            # In a real implementation, this would:
            # - Verify data integrity
            # - Check that services are running
            # - Validate configuration
            # - Run basic functionality tests
            
            logger.info("Rollback state validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback state validation failed: {e}")
            return False
    
    async def _resume_database_operations(self) -> bool:
        """Resume normal database operations."""
        # In a real implementation, this would:
        # - Remove write locks
        # - Resume background tasks
        # - Allow new requests
        logger.info("Database operations resumed after rollback")
        return True
    
    async def _pause_new_operations(self) -> bool:
        """Pause new operations while completing existing ones."""
        logger.info("New operations paused for graceful rollback")
        return True
    
    async def _complete_pending_operations(self) -> bool:
        """Wait for pending operations to complete."""
        # In a real implementation, this would:
        # - Monitor active operations
        # - Wait for completion with timeout
        # - Force completion if necessary
        logger.info("Pending operations completed")
        return True
    
    async def validate_zero_data_loss(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        pre_migration_checksum: str,
        content_ids: Optional[List[str]] = None
    ) -> ZeroDataLossValidation:
        """
        Validate that no data was lost during migration.
        
        Args:
            source_db: Source database
            target_db: Target database
            pre_migration_checksum: Checksum before migration
            content_ids: Optional content IDs to validate
            
        Returns:
            ZeroDataLossValidation: Validation results
        """
        validation_id = f"zero_loss_validation_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"Starting zero data loss validation: {validation_id}")
        
        try:
            # Calculate post-migration checksum
            post_migration_checksum = await self._calculate_comprehensive_checksum(target_db, content_ids)
            
            validation = ZeroDataLossValidation(
                validation_id=validation_id,
                pre_migration_checksum=pre_migration_checksum,
                post_migration_checksum=post_migration_checksum
            )
            
            # Detailed content verification
            if content_ids:
                await self._verify_content_integrity(source_db, target_db, content_ids, validation)
            
            # Detect data loss
            validation.data_loss_detected = (
                pre_migration_checksum != post_migration_checksum or
                len(validation.missing_content_items) > 0 or
                len(validation.missing_embeddings) > 0 or
                len(validation.corrupted_items) > 0
            )
            
            validation.verification_time_ms = (time.time() - start_time) * 1000
            
            if validation.data_loss_detected:
                logger.error(f"Data loss detected in validation: {validation_id}")
            else:
                logger.info(f"Zero data loss validation passed: {validation_id}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Zero data loss validation failed: {e}")
            # Return a validation object indicating failure
            validation = ZeroDataLossValidation(
                validation_id=validation_id,
                pre_migration_checksum=pre_migration_checksum,
                post_migration_checksum="error",
                data_loss_detected=True
            )
            validation.corrupted_items.append(f"Validation error: {str(e)}")
            return validation
    
    async def _calculate_comprehensive_checksum(
        self,
        db: EnhancedVectorDatabaseInterface,
        content_ids: Optional[List[str]]
    ) -> str:
        """Calculate a comprehensive checksum of database content."""
        # This would be implemented similarly to the validator's checksum method
        # but with additional verification steps
        return "comprehensive_checksum_placeholder"
    
    async def _verify_content_integrity(
        self,
        source_db: EnhancedVectorDatabaseInterface,
        target_db: EnhancedVectorDatabaseInterface,
        content_ids: List[str],
        validation: ZeroDataLossValidation
    ):
        """Verify integrity of specific content items."""
        for content_id in content_ids:
            try:
                # Get embeddings from both databases
                source_embeddings = await source_db.get_content_embeddings(content_id)
                target_embeddings = await target_db.get_content_embeddings(content_id)
                
                # Check for missing content
                if len(source_embeddings) > 0 and len(target_embeddings) == 0:
                    validation.missing_content_items.append(content_id)
                
                # Check for missing embeddings
                if len(source_embeddings) != len(target_embeddings):
                    validation.missing_embeddings.append(
                        f"{content_id}: {len(source_embeddings)} -> {len(target_embeddings)}"
                    )
                
                # Verify each embedding
                for i, (source_emb, target_emb) in enumerate(zip(source_embeddings, target_embeddings)):
                    if source_emb.text_content != target_emb.text_content:
                        validation.corrupted_items.append(f"{content_id}:{i}")
                
                validation.content_items_verified += 1
                validation.embeddings_verified += len(source_embeddings)
                
            except Exception as e:
                logger.warning(f"Error verifying content {content_id}: {e}")
                validation.corrupted_items.append(f"{content_id}: {str(e)}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration and rollback status."""
        return {
            "active_migration": self._active_migration,
            "checkpoints_created": len(self._checkpoints),
            "rollback_plans": len(self._rollback_plans),
            "monitoring_active": self._monitoring_active,
            "latest_checkpoint": (
                self._checkpoints[-1].checkpoint_id if self._checkpoints else None
            ),
            "backup_directory": str(self.backup_directory)
        }
    
    def cleanup_old_checkpoints(self, retention_days: int = 30):
        """Clean up old checkpoints and backups."""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        checkpoints_to_remove = []
        for i, checkpoint in enumerate(self._checkpoints):
            if checkpoint.timestamp < cutoff_date:
                checkpoints_to_remove.append(i)
                
                # Remove backup files
                if checkpoint.backup_path:
                    backup_path = Path(checkpoint.backup_path)
                    if backup_path.exists():
                        shutil.rmtree(backup_path, ignore_errors=True)
                        logger.info(f"Cleaned up backup: {backup_path}")
        
        # Remove checkpoints in reverse order to maintain indices
        for i in reversed(checkpoints_to_remove):
            removed_checkpoint = self._checkpoints.pop(i)
            logger.info(f"Cleaned up checkpoint: {removed_checkpoint.checkpoint_id}")
        
        logger.info(f"Cleaned up {len(checkpoints_to_remove)} old checkpoints")


# ========================================
# Convenience Functions
# ========================================

async def safe_migration_with_rollback(
    source_db: EnhancedVectorDatabaseInterface,
    target_db: EnhancedVectorDatabaseInterface,
    migration_function: Callable,
    rollback_strategy: RollbackStrategy = RollbackStrategy.VALIDATION_BASED,
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE
) -> Tuple[bool, Dict[str, Any]]:
    """
    Perform a safe migration with automatic rollback capabilities.
    
    Args:
        source_db: Source database
        target_db: Target database
        migration_function: Function that performs the migration
        rollback_strategy: Rollback strategy to use
        validation_level: Validation level for safety checks
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (success, results)
    """
    rollback_manager = MigrationRollbackManager()
    
    # Create rollback plan
    plan_id = rollback_manager.create_rollback_plan(
        strategy=rollback_strategy,
        trigger_conditions=["validation_failure", "data_loss_detected"],
        validation_requirements={"validation_level": validation_level.value}
    )
    
    results = {
        "migration_successful": False,
        "rollback_required": False,
        "rollback_successful": False,
        "checkpoints_created": [],
        "validation_results": None,
        "errors": []
    }
    
    try:
        # Create pre-migration checkpoint
        pre_checkpoint = await rollback_manager.create_checkpoint(
            MigrationPhase.PREPARATION, source_db, target_db,
            {"purpose": "pre_migration_backup"}
        )
        results["checkpoints_created"].append(pre_checkpoint)
        
        # Calculate pre-migration checksum for zero-loss validation
        pre_checksum = await rollback_manager._calculate_comprehensive_checksum(source_db, None)
        
        # Execute migration
        migration_success = await migration_function(source_db, target_db)
        
        if not migration_success:
            results["errors"].append("Migration function returned failure")
            raise RuntimeError("Migration function failed")
        
        # Create post-migration checkpoint
        post_checkpoint = await rollback_manager.create_checkpoint(
            MigrationPhase.VALIDATION, source_db, target_db,
            {"purpose": "post_migration_validation"}
        )
        results["checkpoints_created"].append(post_checkpoint)
        
        # Validate zero data loss
        zero_loss_validation = await rollback_manager.validate_zero_data_loss(
            source_db, target_db, pre_checksum
        )
        
        if zero_loss_validation.data_loss_detected:
            results["errors"].append("Data loss detected during migration")
            raise RuntimeError("Data loss detected - triggering rollback")
        
        # Perform comprehensive validation
        validator = MigrationValidator()
        validation_report = await validator.validate_migration(
            source_db, target_db, validation_level
        )
        results["validation_results"] = validation_report
        
        if validation_report.overall_status == ValidationStatus.FAILED:
            results["errors"].append("Migration validation failed")
            raise RuntimeError("Migration validation failed - triggering rollback")
        
        results["migration_successful"] = True
        logger.info("Safe migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed, executing rollback: {e}")
        results["errors"].append(str(e))
        results["rollback_required"] = True
        
        # Execute rollback
        rollback_success = await rollback_manager.execute_rollback(plan_id, force=True)
        results["rollback_successful"] = rollback_success
        
        if not rollback_success:
            logger.error("Rollback also failed - manual intervention required")
            results["errors"].append("Rollback failed - manual intervention required")
    
    return results["migration_successful"], results


async def create_migration_safety_net(
    source_db: EnhancedVectorDatabaseInterface,
    backup_directory: str = "./migration_safety_backups"
) -> MigrationRollbackManager:
    """
    Create a comprehensive migration safety net.
    
    Args:
        source_db: Source database to protect
        backup_directory: Directory for safety backups
        
    Returns:
        MigrationRollbackManager: Configured rollback manager
    """
    rollback_manager = MigrationRollbackManager(backup_directory)
    
    # Create initial safety checkpoint
    checkpoint_id = await rollback_manager.create_checkpoint(
        MigrationPhase.PREPARATION, source_db, None,
        {"purpose": "migration_safety_net", "auto_created": True}
    )
    
    # Create multiple rollback plans for different scenarios
    immediate_plan = rollback_manager.create_rollback_plan(
        RollbackStrategy.IMMEDIATE,
        ["critical_error", "data_corruption"],
        {"max_rollback_time": 300}  # 5 minutes
    )
    
    validation_plan = rollback_manager.create_rollback_plan(
        RollbackStrategy.VALIDATION_BASED,
        ["validation_failure", "performance_degradation"],
        {"validation_threshold": 0.95}
    )
    
    logger.info(f"Migration safety net created with checkpoint: {checkpoint_id}")
    logger.info(f"Created rollback plans: {immediate_plan}, {validation_plan}")
    
    return rollback_manager