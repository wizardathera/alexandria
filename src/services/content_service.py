"""
Content management service for the unified DBC platform.

This service provides comprehensive content management across all three modules:
Smart Library, Learning Suite, and Marketplace. It handles the unified content
schema, permissions, relationships, and migration from legacy book data.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import asdict
import uuid

from src.models import (
    ContentItem, ContentRelationship, EmbeddingMetadata, User,
    ModuleType, ContentType, ContentVisibility, UserRole, ProcessingStatus,
    ContentRelationshipType, LegacyBookMetadata,
    LibraryBookMetadata, LMSCourseMetadata, LMSLessonMetadata, MarketplaceItemMetadata,
    get_content_type_for_module, validate_content_for_module
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContentDatabaseError(Exception):
    """Custom exception for content database operations."""
    pass


class ContentService:
    """
    Unified content management service supporting all DBC modules.
    
    Provides a comprehensive API for managing content items, relationships,
    user permissions, and migration between data models.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the content service.
        
        Args:
            db_path: Optional path to SQLite database file
        """
        self.settings = get_settings()
        self.db_path = db_path or f"{self.settings.user_data_path}/content.db"
        self.connection: Optional[sqlite3.Connection] = None
        
        # In-memory caches for performance
        self._content_cache: Dict[str, ContentItem] = {}
        self._relationship_cache: Dict[str, List[ContentRelationship]] = {}
        self._user_cache: Dict[str, User] = {}
        
        # Cache expiry times
        self._cache_expiry = timedelta(minutes=30)
        self._last_cache_update = datetime.now()
    
    async def initialize(self) -> bool:
        """
        Initialize the content database and create tables.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Ensure database directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            self.connection.row_factory = sqlite3.Row  # Access columns by name
            
            # Enable foreign key constraints
            self.connection.execute("PRAGMA foreign_keys = ON")
            
            # Create all tables
            await self._create_tables()
            
            logger.info(f"Content database initialized at: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize content database: {e}")
            return False
    
    async def _create_tables(self):
        """Create all database tables for the unified content schema."""
        
        # Content items table (unified schema)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS content_items (
                content_id TEXT PRIMARY KEY,
                module_type TEXT NOT NULL,
                content_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                author TEXT,
                file_name TEXT,
                file_path TEXT,
                file_type TEXT,
                file_size INTEGER,
                visibility TEXT NOT NULL DEFAULT 'private',
                created_by TEXT,
                organization_id TEXT,
                processing_status TEXT NOT NULL DEFAULT 'pending',
                text_length INTEGER,
                chunk_count INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                processed_at TEXT,
                parent_content_id TEXT,
                prerequisite_content_ids TEXT,  -- JSON array
                topics TEXT,  -- JSON array
                language TEXT DEFAULT 'en',
                reading_level TEXT,
                module_metadata TEXT,  -- JSON object
                FOREIGN KEY (parent_content_id) REFERENCES content_items (content_id),
                FOREIGN KEY (created_by) REFERENCES users (user_id)
            )
        """)
        
        # Content relationships table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS content_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_content_id TEXT NOT NULL,
                target_content_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.5,
                discovered_by TEXT DEFAULT 'ai',
                human_verified BOOLEAN DEFAULT 0,
                created_at TEXT NOT NULL,
                verified_at TEXT,
                verified_by TEXT,
                context TEXT,
                bidirectional BOOLEAN DEFAULT 0,
                FOREIGN KEY (source_content_id) REFERENCES content_items (content_id),
                FOREIGN KEY (target_content_id) REFERENCES content_items (content_id),
                FOREIGN KEY (verified_by) REFERENCES users (user_id)
            )
        """)
        
        # Enhanced embeddings metadata table
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                embedding_id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                module_type TEXT NOT NULL,
                content_type TEXT NOT NULL,
                chunk_type TEXT DEFAULT 'paragraph',
                visibility TEXT NOT NULL,
                creator_id TEXT,
                organization_id TEXT,
                semantic_tags TEXT,  -- JSON array
                language TEXT DEFAULT 'en',
                reading_level TEXT,
                source_location TEXT,  -- JSON object
                text_content TEXT NOT NULL,
                chunk_length INTEGER NOT NULL,
                embedding_model TEXT DEFAULT 'text-embedding-ada-002',
                embedding_dimension INTEGER DEFAULT 1536,
                created_at TEXT NOT NULL,
                importance_score REAL,
                quality_score REAL,
                FOREIGN KEY (content_id) REFERENCES content_items (content_id),
                FOREIGN KEY (creator_id) REFERENCES users (user_id)
            )
        """)
        
        # Users table for permission management
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                username TEXT,
                role TEXT NOT NULL DEFAULT 'reader',
                organization_id TEXT,
                permissions TEXT,  -- JSON array
                full_name TEXT,
                avatar_url TEXT,
                bio TEXT,
                preferences TEXT,  -- JSON object
                notification_settings TEXT,  -- JSON object
                created_at TEXT NOT NULL,
                last_login TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_verified BOOLEAN DEFAULT 0,
                subscription_tier TEXT
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_content_module_type ON content_items (module_type)",
            "CREATE INDEX IF NOT EXISTS idx_content_type ON content_items (content_type)",
            "CREATE INDEX IF NOT EXISTS idx_content_visibility ON content_items (visibility)",
            "CREATE INDEX IF NOT EXISTS idx_content_creator ON content_items (created_by)",
            "CREATE INDEX IF NOT EXISTS idx_content_org ON content_items (organization_id)",
            "CREATE INDEX IF NOT EXISTS idx_content_status ON content_items (processing_status)",
            
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON content_relationships (source_content_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON content_relationships (target_content_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON content_relationships (relationship_type)",
            
            "CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embedding_metadata (content_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_module ON embedding_metadata (module_type)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_visibility ON embedding_metadata (visibility)",
            
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)",
            "CREATE INDEX IF NOT EXISTS idx_users_org ON users (organization_id)",
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users (role)"
        ]
        
        for index in indexes:
            self.connection.execute(index)
        
        logger.info("Database tables and indexes created successfully")
    
    # ========================================
    # Content Item Management
    # ========================================
    
    async def create_content_item(self, content: ContentItem) -> bool:
        """
        Create a new content item.
        
        Args:
            content: ContentItem to create
            
        Returns:
            bool: True if creation successful
        """
        try:
            # Validate content for module
            if not validate_content_for_module(content):
                raise ContentDatabaseError(
                    f"Content type {content.content_type} not valid for module {content.module_type}"
                )
            
            # Insert content item
            self.connection.execute("""
                INSERT INTO content_items (
                    content_id, module_type, content_type, title, description, author,
                    file_name, file_path, file_type, file_size, visibility, created_by,
                    organization_id, processing_status, text_length, chunk_count,
                    created_at, updated_at, processed_at, parent_content_id,
                    prerequisite_content_ids, topics, language, reading_level, module_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.content_id, content.module_type.value, content.content_type.value,
                content.title, content.description, content.author, content.file_name,
                content.file_path, content.file_type, content.file_size,
                content.visibility.value, content.created_by, content.organization_id,
                content.processing_status.value, content.text_length, content.chunk_count,
                content.created_at.isoformat(), content.updated_at.isoformat(),
                content.processed_at.isoformat() if content.processed_at else None,
                content.parent_content_id, json.dumps(content.prerequisite_content_ids),
                json.dumps(content.topics), content.language, content.reading_level,
                json.dumps(content.module_metadata)
            ))
            
            # Update cache
            self._content_cache[content.content_id] = content
            
            logger.info(f"Created content item: {content.content_id} ({content.title})")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error creating content: {e}")
            raise ContentDatabaseError(f"Content item already exists: {content.content_id}")
        except Exception as e:
            logger.error(f"Error creating content item: {e}")
            return False
    
    async def get_content_item(self, content_id: str, user: Optional[User] = None) -> Optional[ContentItem]:
        """
        Get a content item by ID with permission checking.
        
        Args:
            content_id: Content ID to retrieve
            user: Optional user for permission checking
            
        Returns:
            ContentItem if found and accessible, None otherwise
        """
        try:
            # Check cache first
            if content_id in self._content_cache:
                content = self._content_cache[content_id]
                if user is None or user.can_access_content(content):
                    return content
                else:
                    return None
            
            # Query database
            row = self.connection.execute("""
                SELECT * FROM content_items WHERE content_id = ?
            """, (content_id,)).fetchone()
            
            if not row:
                return None
            
            # Convert to ContentItem
            content = self._row_to_content_item(row)
            
            # Check permissions
            if user and not user.can_access_content(content):
                return None
            
            # Update cache
            self._content_cache[content_id] = content
            
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving content item {content_id}: {e}")
            return None
    
    async def list_content_items(
        self,
        module_type: Optional[ModuleType] = None,
        content_type: Optional[ContentType] = None,
        user: Optional[User] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ContentItem]:
        """
        List content items with filtering and permission checking.
        
        Args:
            module_type: Optional module filter
            content_type: Optional content type filter
            user: Optional user for permission checking
            limit: Maximum number of items to return
            offset: Number of items to skip
            
        Returns:
            List of accessible content items
        """
        try:
            # Build query with filters
            query = "SELECT * FROM content_items WHERE 1=1"
            params = []
            
            if module_type:
                query += " AND module_type = ?"
                params.append(module_type.value)
            
            if content_type:
                query += " AND content_type = ?"
                params.append(content_type.value)
            
            # Add permission filtering
            if user:
                if user.role != UserRole.ADMIN:
                    query += """ AND (
                        visibility = 'public' OR
                        (visibility = 'private' AND created_by = ?) OR
                        (visibility = 'organization' AND organization_id = ?) OR
                        (visibility = 'premium' AND ? IN ('pro', 'enterprise'))
                    )"""
                    params.extend([user.user_id, user.organization_id, user.subscription_tier])
            else:
                # Only public content for anonymous users
                query += " AND visibility = 'public'"
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            rows = self.connection.execute(query, params).fetchall()
            
            content_items = []
            for row in rows:
                content = self._row_to_content_item(row)
                content_items.append(content)
                # Update cache
                self._content_cache[content.content_id] = content
            
            return content_items
            
        except Exception as e:
            logger.error(f"Error listing content items: {e}")
            return []
    
    async def count_content_items(
        self,
        module_type: Optional[ModuleType] = None,
        content_type: Optional[ContentType] = None,
        user: Optional[User] = None
    ) -> int:
        """
        Count content items with filtering and permission checking.
        
        Args:
            module_type: Optional module filter
            content_type: Optional content type filter
            user: Optional user for permission checking
            
        Returns:
            int: Number of accessible content items
        """
        try:
            # Build query
            query = "SELECT COUNT(*) FROM content_items WHERE 1=1"
            params = []
            
            # Apply module filter
            if module_type:
                query += " AND module_type = ?"
                params.append(module_type.value)
            
            # Apply content type filter
            if content_type:
                query += " AND content_type = ?"
                params.append(content_type.value)
            
            # Apply user permissions
            if user:
                if user.role == UserRole.ADMIN:
                    # Admins can see all content
                    pass
                else:
                    # Apply permission filtering
                    query += """ AND (
                        visibility = 'public' OR
                        (visibility = 'private' AND created_by = ?) OR
                        (visibility = 'organization' AND organization_id = ?) OR
                        (visibility = 'premium' AND ? IN ('pro', 'enterprise'))
                    )"""
                    params.extend([user.user_id, user.organization_id, user.subscription_tier])
            else:
                # Only public content for anonymous users
                query += " AND visibility = 'public'"
            
            result = self.connection.execute(query, params).fetchone()
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Error counting content items: {e}")
            return 0
    
    async def update_content_item(self, content: ContentItem) -> bool:
        """
        Update an existing content item.
        
        Args:
            content: Updated content item
            
        Returns:
            bool: True if update successful
        """
        try:
            content.update_timestamp()
            
            self.connection.execute("""
                UPDATE content_items SET
                    title = ?, description = ?, author = ?, file_name = ?, file_path = ?,
                    file_type = ?, file_size = ?, visibility = ?, organization_id = ?,
                    processing_status = ?, text_length = ?, chunk_count = ?, updated_at = ?,
                    processed_at = ?, parent_content_id = ?, prerequisite_content_ids = ?,
                    topics = ?, language = ?, reading_level = ?, module_metadata = ?
                WHERE content_id = ?
            """, (
                content.title, content.description, content.author, content.file_name,
                content.file_path, content.file_type, content.file_size,
                content.visibility.value, content.organization_id,
                content.processing_status.value, content.text_length, content.chunk_count,
                content.updated_at.isoformat(),
                content.processed_at.isoformat() if content.processed_at else None,
                content.parent_content_id, json.dumps(content.prerequisite_content_ids),
                json.dumps(content.topics), content.language, content.reading_level,
                json.dumps(content.module_metadata), content.content_id
            ))
            
            # Update cache
            self._content_cache[content.content_id] = content
            
            logger.info(f"Updated content item: {content.content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating content item {content.content_id}: {e}")
            return False
    
    async def delete_content_item(self, content_id: str, user: Optional[User] = None) -> bool:
        """
        Delete a content item (with permission checking).
        
        Args:
            content_id: Content ID to delete
            user: User attempting deletion (for permission checking)
            
        Returns:
            bool: True if deletion successful
        """
        try:
            # Get content for permission checking
            content = await self.get_content_item(content_id)
            if not content:
                return False
            
            # Check permissions
            if user and user.role != UserRole.ADMIN and content.created_by != user.user_id:
                logger.warning(f"User {user.user_id} denied permission to delete {content_id}")
                return False
            
            # Delete related data first
            self.connection.execute(
                "DELETE FROM content_relationships WHERE source_content_id = ? OR target_content_id = ?",
                (content_id, content_id)
            )
            self.connection.execute(
                "DELETE FROM embedding_metadata WHERE content_id = ?",
                (content_id,)
            )
            
            # Delete content item
            self.connection.execute("DELETE FROM content_items WHERE content_id = ?", (content_id,))
            
            # Remove from cache
            self._content_cache.pop(content_id, None)
            self._relationship_cache.pop(content_id, None)
            
            logger.info(f"Deleted content item: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting content item {content_id}: {e}")
            return False
    
    # ========================================
    # Content Relationship Management
    # ========================================
    
    async def create_relationship(self, relationship: ContentRelationship) -> bool:
        """
        Create a content relationship.
        
        Args:
            relationship: ContentRelationship to create
            
        Returns:
            bool: True if creation successful
        """
        try:
            self.connection.execute("""
                INSERT INTO content_relationships (
                    relationship_id, source_content_id, target_content_id, relationship_type,
                    strength, confidence, discovered_by, human_verified, created_at,
                    verified_at, verified_by, context, bidirectional
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.relationship_id, relationship.source_content_id,
                relationship.target_content_id, relationship.relationship_type.value,
                relationship.strength, relationship.confidence, relationship.discovered_by,
                relationship.human_verified, relationship.created_at.isoformat(),
                relationship.verified_at.isoformat() if relationship.verified_at else None,
                relationship.verified_by, relationship.context, relationship.bidirectional
            ))
            
            # Update cache
            if relationship.source_content_id not in self._relationship_cache:
                self._relationship_cache[relationship.source_content_id] = []
            self._relationship_cache[relationship.source_content_id].append(relationship)
            
            logger.info(f"Created relationship: {relationship.relationship_id}")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Integrity error creating relationship: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    async def get_content_relationships(
        self,
        content_id: str,
        relationship_type: Optional[ContentRelationshipType] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[ContentRelationship]:
        """
        Get relationships for a content item.
        
        Args:
            content_id: Content ID to find relationships for
            relationship_type: Optional type filter
            direction: Direction of relationships to include
            
        Returns:
            List of content relationships
        """
        try:
            # Check cache first
            if content_id in self._relationship_cache:
                cached_relationships = self._relationship_cache[content_id]
                if relationship_type:
                    cached_relationships = [
                        r for r in cached_relationships 
                        if r.relationship_type == relationship_type
                    ]
                return cached_relationships
            
            # Build query
            query = "SELECT * FROM content_relationships WHERE"
            params = []
            
            if direction == "outgoing":
                query += " source_content_id = ?"
                params.append(content_id)
            elif direction == "incoming":
                query += " target_content_id = ?"
                params.append(content_id)
            else:  # both
                query += " (source_content_id = ? OR target_content_id = ?)"
                params.extend([content_id, content_id])
            
            if relationship_type:
                query += " AND relationship_type = ?"
                params.append(relationship_type.value)
            
            rows = self.connection.execute(query, params).fetchall()
            
            relationships = []
            for row in rows:
                relationship = self._row_to_relationship(row)
                relationships.append(relationship)
            
            # Update cache
            self._relationship_cache[content_id] = relationships
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error getting relationships for {content_id}: {e}")
            return []
    
    # ========================================
    # Migration and Legacy Support
    # ========================================
    
    async def migrate_legacy_book(self, legacy_book: LegacyBookMetadata) -> Optional[ContentItem]:
        """
        Migrate a legacy book metadata to the unified content schema.
        
        Args:
            legacy_book: Legacy book metadata to migrate
            
        Returns:
            ContentItem if migration successful, None otherwise
        """
        try:
            # Convert to unified content item
            content_item = legacy_book.to_content_item()
            
            # Check if already exists
            existing = await self.get_content_item(content_item.content_id)
            if existing:
                logger.info(f"Book {content_item.content_id} already migrated")
                return existing
            
            # Create the content item
            if await self.create_content_item(content_item):
                logger.info(f"Migrated legacy book: {content_item.content_id} ({content_item.title})")
                return content_item
            else:
                logger.error(f"Failed to migrate legacy book: {legacy_book.book_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error migrating legacy book {legacy_book.book_id}: {e}")
            return None
    
    async def migrate_all_legacy_books(self, legacy_metadata: List[LegacyBookMetadata]) -> Dict[str, bool]:
        """
        Migrate multiple legacy books to unified content schema.
        
        Args:
            legacy_metadata: List of legacy book metadata
            
        Returns:
            Dict mapping book_id to migration success status
        """
        results = {}
        
        for legacy_book in legacy_metadata:
            try:
                migrated = await self.migrate_legacy_book(legacy_book)
                results[legacy_book.book_id] = migrated is not None
            except Exception as e:
                logger.error(f"Migration failed for {legacy_book.book_id}: {e}")
                results[legacy_book.book_id] = False
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"Migrated {successful}/{len(legacy_metadata)} legacy books")
        
        return results
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _row_to_content_item(self, row: sqlite3.Row) -> ContentItem:
        """Convert database row to ContentItem."""
        return ContentItem(
            content_id=row["content_id"],
            module_type=ModuleType(row["module_type"]),
            content_type=ContentType(row["content_type"]),
            title=row["title"],
            description=row["description"],
            author=row["author"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            visibility=ContentVisibility(row["visibility"]),
            created_by=row["created_by"],
            organization_id=row["organization_id"],
            processing_status=ProcessingStatus(row["processing_status"]),
            text_length=row["text_length"],
            chunk_count=row["chunk_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            processed_at=datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None,
            parent_content_id=row["parent_content_id"],
            prerequisite_content_ids=json.loads(row["prerequisite_content_ids"]) if row["prerequisite_content_ids"] else [],
            topics=json.loads(row["topics"]) if row["topics"] else [],
            language=row["language"],
            reading_level=row["reading_level"],
            module_metadata=json.loads(row["module_metadata"]) if row["module_metadata"] else {}
        )
    
    def _row_to_relationship(self, row: sqlite3.Row) -> ContentRelationship:
        """Convert database row to ContentRelationship."""
        return ContentRelationship(
            relationship_id=row["relationship_id"],
            source_content_id=row["source_content_id"],
            target_content_id=row["target_content_id"],
            relationship_type=ContentRelationshipType(row["relationship_type"]),
            strength=row["strength"],
            confidence=row["confidence"],
            discovered_by=row["discovered_by"],
            human_verified=bool(row["human_verified"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            verified_at=datetime.fromisoformat(row["verified_at"]) if row["verified_at"] else None,
            verified_by=row["verified_by"],
            context=row["context"],
            bidirectional=bool(row["bidirectional"])
        )
    
    async def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Content database connection closed")


# ========================================
# Global Service Instance
# ========================================

_content_service: Optional[ContentService] = None


async def get_content_service() -> ContentService:
    """
    Get the global content service instance with lazy initialization.
    
    Returns:
        ContentService: Initialized content service instance
    """
    global _content_service
    
    if _content_service is None:
        _content_service = ContentService()
        if not await _content_service.initialize():
            raise RuntimeError("Failed to initialize content service")
    
    return _content_service