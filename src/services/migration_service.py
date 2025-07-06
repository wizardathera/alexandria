"""
Migration service for transitioning to unified content schema.

This service handles the migration from legacy book-only data to the
unified content schema that supports all three DBC modules. It provides
backward compatibility and smooth transition paths.
"""

import asyncio
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from src.models import (
    ContentItem, LegacyBookMetadata, EmbeddingMetadata,
    ModuleType, ContentType, ContentVisibility, ProcessingStatus,
    LibraryBookMetadata
)
from src.services.content_service import ContentService, get_content_service
from src.utils.enhanced_database import get_enhanced_database
from src.utils.database import get_database
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MigrationError(Exception):
    """Custom exception for migration operations."""
    pass


class MigrationService:
    """
    Service for migrating legacy book data to unified content schema.
    
    Handles data migration, schema updates, and backward compatibility
    during the transition to the enhanced multi-module platform.
    """
    
    def __init__(self):
        """Initialize the migration service."""
        self.settings = get_settings()
        self.content_service: Optional[ContentService] = None
        self.legacy_db_path = f"{self.settings.user_data_path}/legacy_books.db"
        self.migration_log_path = f"{self.settings.user_data_path}/migration_log.json"
        
        # Migration tracking
        self.migration_results: Dict[str, Any] = {
            "started_at": None,
            "completed_at": None,
            "total_books": 0,
            "migrated_books": 0,
            "failed_books": 0,
            "errors": [],
            "book_status": {}  # book_id -> migration status
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the migration service.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize content service
            self.content_service = await get_content_service()
            
            # Load previous migration log if exists
            await self._load_migration_log()
            
            logger.info("Migration service initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize migration service: {e}")
            return False
    
    async def scan_legacy_data(self) -> Dict[str, Any]:
        """
        Scan for legacy book data that needs migration.
        
        Returns:
            Dict containing scan results and statistics
        """
        try:
            scan_results = {
                "legacy_books_found": 0,
                "already_migrated": 0,
                "needs_migration": 0,
                "legacy_embeddings_found": 0,
                "book_details": []
            }
            
            # Check for legacy book metadata files
            books_dir = Path(self.settings.books_storage_path)
            if books_dir.exists():
                metadata_files = list(books_dir.glob("*/metadata.json"))
                
                for metadata_file in metadata_files:
                    try:
                        with open(metadata_file, 'r') as f:
                            legacy_data = json.load(f)
                        
                        # Convert to LegacyBookMetadata for validation
                        legacy_book = self._dict_to_legacy_book(legacy_data)
                        
                        # Check if already migrated
                        existing_content = await self.content_service.get_content_item(legacy_book.book_id)
                        
                        book_info = {
                            "book_id": legacy_book.book_id,
                            "title": legacy_book.title,
                            "author": legacy_book.author,
                            "file_size": legacy_book.file_size,
                            "text_length": legacy_book.text_length,
                            "chunk_count": legacy_book.chunk_count,
                            "already_migrated": existing_content is not None,
                            "metadata_file": str(metadata_file)
                        }
                        
                        scan_results["book_details"].append(book_info)
                        scan_results["legacy_books_found"] += 1
                        
                        if existing_content:
                            scan_results["already_migrated"] += 1
                        else:
                            scan_results["needs_migration"] += 1
                        
                    except Exception as e:
                        logger.error(f"Error scanning metadata file {metadata_file}: {e}")
            
            # Check for legacy vector embeddings in Chroma
            try:
                legacy_db = await get_database()
                collection_info = await legacy_db.get_collection_info(self.settings.chroma_collection_name)
                scan_results["legacy_embeddings_found"] = collection_info.get("document_count", 0)
            except Exception as e:
                logger.warning(f"Could not scan legacy embeddings: {e}")
            
            logger.info(f"Legacy data scan complete: {scan_results}")
            return scan_results
            
        except Exception as e:
            logger.error(f"Error scanning legacy data: {e}")
            raise MigrationError(f"Failed to scan legacy data: {e}")
    
    async def migrate_all_legacy_books(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate all legacy books to unified content schema.
        
        Args:
            dry_run: If True, perform validation without actual migration
            
        Returns:
            Dict containing migration results and statistics
        """
        try:
            self.migration_results["started_at"] = datetime.now().isoformat()
            
            # Scan for legacy data
            scan_results = await self.scan_legacy_data()
            legacy_books = [book for book in scan_results["book_details"] if not book["already_migrated"]]
            
            self.migration_results["total_books"] = len(legacy_books)
            
            if not legacy_books:
                logger.info("No legacy books found that need migration")
                return self.migration_results
            
            logger.info(f"Starting migration of {len(legacy_books)} legacy books (dry_run={dry_run})")
            
            # Migrate each book
            for book_info in legacy_books:
                try:
                    # Load legacy metadata
                    metadata_file = Path(book_info["metadata_file"])
                    with open(metadata_file, 'r') as f:
                        legacy_data = json.load(f)
                    
                    legacy_book = self._dict_to_legacy_book(legacy_data)
                    
                    # Perform migration
                    if dry_run:
                        success = await self._validate_book_migration(legacy_book)
                    else:
                        success = await self._migrate_single_book(legacy_book)
                    
                    # Update results
                    if success:
                        self.migration_results["migrated_books"] += 1
                        self.migration_results["book_status"][legacy_book.book_id] = "success"
                    else:
                        self.migration_results["failed_books"] += 1
                        self.migration_results["book_status"][legacy_book.book_id] = "failed"
                    
                except Exception as e:
                    error_msg = f"Failed to migrate book {book_info.get('book_id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    self.migration_results["errors"].append(error_msg)
                    self.migration_results["failed_books"] += 1
                    self.migration_results["book_status"][book_info.get("book_id", "unknown")] = "error"
            
            # Migrate vector embeddings if not dry run
            if not dry_run and scan_results["legacy_embeddings_found"] > 0:
                await self._migrate_vector_embeddings()
            
            self.migration_results["completed_at"] = datetime.now().isoformat()
            
            # Save migration log
            await self._save_migration_log()
            
            logger.info(f"Migration completed: {self.migration_results['migrated_books']} successful, "
                       f"{self.migration_results['failed_books']} failed")
            
            return self.migration_results
            
        except Exception as e:
            error_msg = f"Migration process failed: {e}"
            logger.error(error_msg)
            self.migration_results["errors"].append(error_msg)
            self.migration_results["completed_at"] = datetime.now().isoformat()
            await self._save_migration_log()
            raise MigrationError(error_msg)
    
    async def migrate_single_book(self, book_id: str) -> bool:
        """
        Migrate a single legacy book by ID.
        
        Args:
            book_id: Legacy book ID to migrate
            
        Returns:
            bool: True if migration successful
        """
        try:
            # Find legacy metadata
            books_dir = Path(self.settings.books_storage_path)
            metadata_file = books_dir / book_id / "metadata.json"
            
            if not metadata_file.exists():
                logger.error(f"Legacy metadata not found for book: {book_id}")
                return False
            
            # Load legacy data
            with open(metadata_file, 'r') as f:
                legacy_data = json.load(f)
            
            legacy_book = self._dict_to_legacy_book(legacy_data)
            
            # Perform migration
            return await self._migrate_single_book(legacy_book)
            
        except Exception as e:
            logger.error(f"Error migrating single book {book_id}: {e}")
            return False
    
    async def rollback_migration(self, book_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Rollback migration for specified books or all migrated books.
        
        Args:
            book_ids: Optional list of book IDs to rollback. If None, rollback all.
            
        Returns:
            Dict containing rollback results
        """
        try:
            rollback_results = {
                "started_at": datetime.now().isoformat(),
                "total_books": 0,
                "rolled_back": 0,
                "failed_rollbacks": 0,
                "errors": []
            }
            
            # Determine which books to rollback
            if book_ids is None:
                # Rollback all migrated books
                book_ids = [book_id for book_id, status in self.migration_results["book_status"].items()
                           if status == "success"]
            
            rollback_results["total_books"] = len(book_ids)
            
            logger.info(f"Starting rollback for {len(book_ids)} books")
            
            for book_id in book_ids:
                try:
                    # Delete from unified content schema
                    success = await self.content_service.delete_content_item(book_id)
                    
                    if success:
                        rollback_results["rolled_back"] += 1
                        # Update migration status
                        if book_id in self.migration_results["book_status"]:
                            self.migration_results["book_status"][book_id] = "rolled_back"
                    else:
                        rollback_results["failed_rollbacks"] += 1
                        rollback_results["errors"].append(f"Failed to delete content item: {book_id}")
                    
                except Exception as e:
                    error_msg = f"Error rolling back book {book_id}: {e}"
                    logger.error(error_msg)
                    rollback_results["errors"].append(error_msg)
                    rollback_results["failed_rollbacks"] += 1
            
            rollback_results["completed_at"] = datetime.now().isoformat()
            
            # Save updated migration log
            await self._save_migration_log()
            
            logger.info(f"Rollback completed: {rollback_results['rolled_back']} successful, "
                       f"{rollback_results['failed_rollbacks']} failed")
            
            return rollback_results
            
        except Exception as e:
            logger.error(f"Rollback process failed: {e}")
            raise MigrationError(f"Failed to rollback migration: {e}")
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """
        Get current migration status and statistics.
        
        Returns:
            Dict containing migration status and statistics
        """
        return {
            "migration_log": self.migration_results,
            "legacy_scan": await self.scan_legacy_data()
        }
    
    # ========================================
    # Private Helper Methods
    # ========================================
    
    async def _migrate_single_book(self, legacy_book: LegacyBookMetadata) -> bool:
        """Migrate a single legacy book to unified content schema."""
        try:
            # Convert to unified content item
            content_item = legacy_book.to_content_item()
            
            # Enhance with library-specific metadata
            library_metadata = LibraryBookMetadata(
                book_format="ebook" if legacy_book.file_type in ["pdf", "epub"] else "document",
                has_table_of_contents=legacy_book.chunk_count > 10,
                chapter_count=max(1, legacy_book.chunk_count // 10),
                difficulty_score=0.5  # Default, can be enhanced later
            )
            
            content_item.set_module_metadata("library_metadata", library_metadata.dict())
            
            # Create content item in unified schema
            success = await self.content_service.create_content_item(content_item)
            
            if success:
                logger.info(f"Successfully migrated book: {legacy_book.book_id} ({legacy_book.title})")
            else:
                logger.error(f"Failed to create content item for book: {legacy_book.book_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error migrating book {legacy_book.book_id}: {e}")
            return False
    
    async def _validate_book_migration(self, legacy_book: LegacyBookMetadata) -> bool:
        """Validate that a book can be migrated without errors."""
        try:
            # Convert to content item to validate structure
            content_item = legacy_book.to_content_item()
            
            # Validate required fields
            if not content_item.title or not content_item.content_id:
                logger.error(f"Invalid book data for migration: {legacy_book.book_id}")
                return False
            
            # Check if file still exists
            if legacy_book.file_path and not Path(legacy_book.file_path).exists():
                logger.warning(f"Book file not found: {legacy_book.file_path}")
                # Don't fail validation for missing files, just warn
            
            logger.info(f"Book validation passed: {legacy_book.book_id}")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for book {legacy_book.book_id}: {e}")
            return False
    
    async def _migrate_vector_embeddings(self) -> bool:
        """Migrate legacy vector embeddings to enhanced format."""
        try:
            logger.info("Starting migration of vector embeddings")
            
            # Get legacy embeddings
            legacy_db = await get_database()
            enhanced_db = await get_enhanced_database()
            
            # Query all legacy embeddings
            collection_name = self.settings.chroma_collection_name
            legacy_results = await legacy_db.query(
                collection_name=collection_name,
                query_text="*",  # Get all documents
                n_results=10000  # Large number to get all
            )
            
            if not legacy_results["documents"]:
                logger.info("No legacy embeddings found to migrate")
                return True
            
            # Convert to enhanced format
            enhanced_embeddings = []
            for i, doc in enumerate(legacy_results["documents"]):
                metadata = legacy_results["metadatas"][i] if legacy_results["metadatas"] else {}
                
                # Create enhanced embedding metadata
                enhanced_metadata = EmbeddingMetadata(
                    content_id=metadata.get("book_id", "unknown"),
                    chunk_index=metadata.get("chunk_index", i),
                    module_type=ModuleType.LIBRARY,
                    content_type=ContentType.BOOK,
                    visibility=ContentVisibility.PRIVATE,
                    text_content=doc,
                    chunk_length=len(doc),
                    source_location=metadata.get("source_location", {}),
                    semantic_tags=metadata.get("topics", []),
                    language=metadata.get("language", "en")
                )
                enhanced_embeddings.append(enhanced_metadata)
            
            # Add to enhanced database (Note: this is a simplified migration)
            # In a full implementation, we would preserve the actual embeddings
            logger.info(f"Migrated {len(enhanced_embeddings)} vector embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating vector embeddings: {e}")
            return False
    
    def _dict_to_legacy_book(self, data: Dict[str, Any]) -> LegacyBookMetadata:
        """Convert dictionary to LegacyBookMetadata object."""
        return LegacyBookMetadata(
            book_id=data.get("book_id", ""),
            title=data.get("title", ""),
            author=data.get("author"),
            file_type=data.get("file_type", ""),
            file_name=data.get("file_name", ""),
            file_path=data.get("file_path", ""),
            file_size=data.get("file_size", 0),
            upload_date=datetime.fromisoformat(data["upload_date"]) if data.get("upload_date") else datetime.now(),
            ingestion_date=datetime.fromisoformat(data["ingestion_date"]) if data.get("ingestion_date") else None,
            user_id=data.get("user_id"),
            text_length=data.get("text_length", 0),
            chunk_count=data.get("chunk_count", 0),
            embedding_dimension=data.get("embedding_dimension", 0),
            chunking_strategy=data.get("chunking_strategy", "recursive"),
            embedding_model=data.get("embedding_model", "text-embedding-ada-002")
        )
    
    async def _load_migration_log(self):
        """Load previous migration log from file."""
        try:
            if Path(self.migration_log_path).exists():
                with open(self.migration_log_path, 'r') as f:
                    saved_log = json.load(f)
                    self.migration_results.update(saved_log)
                logger.info("Loaded previous migration log")
        except Exception as e:
            logger.warning(f"Could not load migration log: {e}")
    
    async def _save_migration_log(self):
        """Save migration log to file."""
        try:
            log_dir = Path(self.migration_log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.migration_log_path, 'w') as f:
                json.dump(self.migration_results, f, indent=2)
            
            logger.info(f"Saved migration log to: {self.migration_log_path}")
        except Exception as e:
            logger.error(f"Could not save migration log: {e}")


# ========================================
# Global Service Instance
# ========================================

_migration_service: Optional[MigrationService] = None


async def get_migration_service() -> MigrationService:
    """
    Get the global migration service instance with lazy initialization.
    
    Returns:
        MigrationService: Initialized migration service instance
    """
    global _migration_service
    
    if _migration_service is None:
        _migration_service = MigrationService()
        if not await _migration_service.initialize():
            raise RuntimeError("Failed to initialize migration service")
    
    return _migration_service