"""
Supabase pgvector database implementation for DBC platform.

This module provides a production-ready vector database implementation using
Supabase with pgvector extension for the migration from Chroma to production
scale database infrastructure.
"""

import asyncio
import asyncpg
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import asdict

from src.models import (
    ContentItem, EmbeddingMetadata, User, ContentRelationship,
    ModuleType, ContentType, ContentVisibility, UserRole,
    ContentRelationshipType
)
from src.utils.enhanced_database import EnhancedVectorDatabaseInterface
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SupabaseVectorDB(EnhancedVectorDatabaseInterface):
    """
    Supabase pgvector database implementation.
    
    Provides production-scale vector database capabilities with enhanced
    metadata, permission-aware search, and content relationships using
    PostgreSQL with pgvector extension via Supabase.
    """
    
    def __init__(self):
        """Initialize Supabase vector database client."""
        self.settings = get_settings()
        self.pool: Optional[asyncpg.Pool] = None
        self._connection_params = {
            'host': self.settings.supabase_db_host,
            'port': self.settings.supabase_db_port,
            'database': self.settings.supabase_db_name,
            'user': self.settings.supabase_db_user,
            'password': self.settings.supabase_db_password,
            'ssl': 'require',
            'command_timeout': 60,
            'server_settings': {
                'jit': 'off'  # Disable JIT for vector operations
            }
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the Supabase database connection pool.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                min_size=2,
                max_size=10,
                **self._connection_params
            )
            
            # Test connection and ensure extensions
            async with self.pool.acquire() as conn:
                # Enable vector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                
                # Test vector operations
                await conn.execute("SELECT vector_dims(vector '[1,2,3]');")
                
                logger.info("Supabase pgvector database initialized successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize Supabase database: {e}")
            return False
    
    async def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new collection (handled by schema setup).
        
        Args:
            name: Collection name (not used in Supabase implementation)
            metadata: Optional collection metadata
            
        Returns:
            bool: True if creation successful
        """
        # In Supabase implementation, collections are handled by the unified schema
        # This method ensures tables exist and are ready
        try:
            async with self.pool.acquire() as conn:
                # Verify core tables exist
                tables_check = await conn.fetch("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('content_items', 'content_embeddings', 'content_relationships')
                """)
                
                if len(tables_check) < 3:
                    logger.error("Required tables not found. Run migration script first.")
                    return False
                
                logger.info(f"Supabase collection '{name}' ready (unified schema)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to verify Supabase collection {name}: {e}")
            return False
    
    async def add_documents_with_metadata(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        embedding_metadata: List[EmbeddingMetadata],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents with enhanced embedding metadata to Supabase.
        
        Args:
            collection_name: Collection name (used for logging)
            documents: List of document texts
            embeddings: List of embedding vectors
            embedding_metadata: List of enhanced metadata
            ids: Optional list of document IDs
            
        Returns:
            bool: True if addition successful
        """
        if not self.pool:
            logger.error("Database not initialized")
            return False
        
        if len(documents) != len(embeddings) or len(documents) != len(embedding_metadata):
            logger.error("Documents, embeddings, and metadata must have same length")
            return False
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Prepare batch insert data
                    insert_data = []
                    for i, (doc, embedding, metadata) in enumerate(zip(documents, embeddings, embedding_metadata)):
                        embedding_id = ids[i] if ids else str(uuid.uuid4())
                        
                        insert_data.append((
                            uuid.UUID(metadata.embedding_id) if metadata.embedding_id else uuid.UUID(embedding_id),
                            uuid.UUID(metadata.content_id),
                            metadata.chunk_index,
                            metadata.module_type.value,
                            metadata.content_type.value,
                            metadata.chunk_type,
                            metadata.visibility.value,
                            uuid.UUID(metadata.creator_id) if metadata.creator_id else None,
                            uuid.UUID(metadata.organization_id) if metadata.organization_id else None,
                            metadata.semantic_tags,
                            metadata.language,
                            metadata.reading_level,
                            json.dumps(metadata.source_location),
                            doc,  # text_content
                            metadata.chunk_length,
                            embedding,  # vector embedding
                            metadata.embedding_model,
                            metadata.embedding_dimension,
                            metadata.importance_score,
                            metadata.quality_score,
                            metadata.created_at
                        ))
                    
                    # Batch insert embeddings
                    await conn.executemany("""
                        INSERT INTO content_embeddings (
                            embedding_id, content_id, chunk_index,
                            module_type, content_type, chunk_type,
                            visibility, creator_id, organization_id,
                            semantic_tags, language, reading_level,
                            source_location, text_content, chunk_length,
                            embedding, embedding_model, embedding_dimension,
                            importance_score, quality_score, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                        ON CONFLICT (content_id, chunk_index) DO UPDATE SET
                            text_content = EXCLUDED.text_content,
                            embedding = EXCLUDED.embedding,
                            semantic_tags = EXCLUDED.semantic_tags,
                            importance_score = EXCLUDED.importance_score,
                            quality_score = EXCLUDED.quality_score
                    """, insert_data)
                    
                logger.info(f"Added {len(documents)} documents to Supabase collection: {collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add documents to Supabase {collection_name}: {e}")
            return False
    
    async def query_with_permissions(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 5,
        user: Optional[User] = None,
        module_filter: Optional[ModuleType] = None,
        content_type_filter: Optional[ContentType] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query documents with permission and module filtering using pgvector.
        
        Args:
            collection_name: Collection name (for logging)
            query_text: Query text for similarity search
            query_embedding: Pre-computed query embedding vector
            n_results: Number of results to return
            user: Optional user for permission checking
            module_filter: Optional module filter
            content_type_filter: Optional content type filter
            additional_filters: Optional additional metadata filters
            
        Returns:
            Dict[str, Any]: Enhanced query results
        """
        if not self.pool:
            logger.error("Database not initialized")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        if not query_embedding:
            logger.error("Query embedding required for Supabase search")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        try:
            async with self.pool.acquire() as conn:
                # Use the permission-aware search function
                user_id = uuid.UUID(user.user_id) if user and user.user_id else None
                user_role = user.role.value if user else 'reader'
                user_org_id = uuid.UUID(user.organization_id) if user and user.organization_id else None
                user_subscription = user.subscription_tier if user else 'free'
                module_filter_str = module_filter.value if module_filter else None
                content_type_filter_str = content_type_filter.value if content_type_filter else None
                
                results = await conn.fetch("""
                    SELECT * FROM search_embeddings_with_permissions(
                        $1::vector(1536), $2, $3, $4, $5, $6, $7, $8, 0.0
                    )
                """, 
                query_embedding,
                user_id,
                user_role,
                user_org_id,
                user_subscription,
                module_filter_str,
                content_type_filter_str,
                n_results
                )
                
                # Process results
                documents = []
                metadatas = []
                distances = []
                ids = []
                
                for row in results:
                    documents.append(row['text_content'])
                    
                    metadata = {
                        'embedding_id': str(row['embedding_id']),
                        'content_id': str(row['content_id']),
                        'chunk_index': row['chunk_index'],
                        'module_type': row['module_type'],
                        'content_type': row['content_type'],
                        'chunk_type': row['chunk_type'],
                        'semantic_tags': row['semantic_tags'],
                        'source_location': row['source_location'],
                        'similarity_score': float(row['similarity_score'])
                    }
                    metadatas.append(metadata)
                    
                    # Convert similarity to distance (0 = identical, 1 = completely different)
                    distance = 1.0 - float(row['similarity_score'])
                    distances.append(distance)
                    
                    ids.append(str(row['embedding_id']))
                
                logger.info(f"Supabase query returned {len(documents)} results")
                
                return {
                    "documents": documents,
                    "metadatas": metadatas,
                    "distances": distances,
                    "ids": ids
                }
                
        except Exception as e:
            logger.error(f"Failed to perform Supabase query: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    async def get_content_embeddings(
        self,
        content_id: str,
        collection_name: Optional[str] = None
    ) -> List[EmbeddingMetadata]:
        """
        Get all embeddings for a specific content item.
        
        Args:
            content_id: Content ID to retrieve embeddings for
            collection_name: Optional collection name (not used)
            
        Returns:
            List of embedding metadata
        """
        if not self.pool:
            logger.error("Database not initialized")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch("""
                    SELECT 
                        embedding_id, content_id, chunk_index,
                        module_type, content_type, chunk_type,
                        visibility, creator_id, organization_id,
                        semantic_tags, language, reading_level,
                        source_location, text_content, chunk_length,
                        embedding_model, embedding_dimension,
                        importance_score, quality_score, created_at
                    FROM content_embeddings
                    WHERE content_id = $1
                    ORDER BY chunk_index
                """, uuid.UUID(content_id))
                
                embeddings = []
                for row in results:
                    embedding_metadata = EmbeddingMetadata(
                        embedding_id=str(row['embedding_id']),
                        content_id=str(row['content_id']),
                        chunk_index=row['chunk_index'],
                        module_type=ModuleType(row['module_type']),
                        content_type=ContentType(row['content_type']),
                        chunk_type=row['chunk_type'],
                        visibility=ContentVisibility(row['visibility']),
                        creator_id=str(row['creator_id']) if row['creator_id'] else None,
                        organization_id=str(row['organization_id']) if row['organization_id'] else None,
                        semantic_tags=row['semantic_tags'] or [],
                        language=row['language'],
                        reading_level=row['reading_level'],
                        source_location=json.loads(row['source_location']) if row['source_location'] else {},
                        text_content=row['text_content'],
                        chunk_length=row['chunk_length'],
                        embedding_model=row['embedding_model'],
                        embedding_dimension=row['embedding_dimension'],
                        created_at=row['created_at'],
                        importance_score=float(row['importance_score']) if row['importance_score'] else None,
                        quality_score=float(row['quality_score']) if row['quality_score'] else None
                    )
                    embeddings.append(embedding_metadata)
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to get embeddings for content {content_id}: {e}")
            return []
    
    async def delete_content_embeddings(
        self,
        content_id: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete all embeddings for a specific content item.
        
        Args:
            content_id: Content ID to delete embeddings for
            collection_name: Optional collection name (not used)
            
        Returns:
            bool: True if deletion successful
        """
        if not self.pool:
            logger.error("Database not initialized")
            return False
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM content_embeddings 
                    WHERE content_id = $1
                """, uuid.UUID(content_id))
                
                # Extract number of deleted rows from result
                deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                
                logger.info(f"Deleted {deleted_count} embeddings for content: {content_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete embeddings for content {content_id}: {e}")
            return False
    
    async def update_embedding_metadata(
        self,
        embedding_id: str,
        metadata_updates: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Update metadata for a specific embedding.
        
        Args:
            embedding_id: Embedding ID to update
            metadata_updates: Dictionary of metadata updates
            collection_name: Optional collection name (not used)
            
        Returns:
            bool: True if update successful
        """
        if not self.pool:
            logger.error("Database not initialized")
            return False
        
        try:
            async with self.pool.acquire() as conn:
                # Build dynamic update query
                set_clauses = []
                values = []
                param_index = 1
                
                for key, value in metadata_updates.items():
                    if key in ['semantic_tags', 'importance_score', 'quality_score', 'reading_level']:
                        set_clauses.append(f"{key} = ${param_index + 1}")
                        values.append(value)
                        param_index += 1
                
                if not set_clauses:
                    logger.warning("No valid metadata fields to update")
                    return True
                
                # Add embedding_id as first parameter
                values.insert(0, uuid.UUID(embedding_id))
                
                query = f"""
                    UPDATE content_embeddings 
                    SET {', '.join(set_clauses)}
                    WHERE embedding_id = $1
                """
                
                result = await conn.execute(query, *values)
                
                updated_count = int(result.split()[-1]) if result.startswith('UPDATE') else 0
                
                if updated_count > 0:
                    logger.info(f"Updated metadata for embedding: {embedding_id}")
                    return True
                else:
                    logger.warning(f"No embedding found with ID: {embedding_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update metadata for embedding {embedding_id}: {e}")
            return False
    
    async def similarity_search_with_relationships(
        self,
        collection_name: str,
        query_text: str,
        content_relationships: List[ContentRelationship],
        n_results: int = 5,
        relationship_boost: float = 0.1
    ) -> Dict[str, Any]:
        """
        Perform similarity search with relationship-aware scoring.
        
        Args:
            collection_name: Collection to search
            query_text: Query text
            content_relationships: Relationships to consider for boosting
            n_results: Number of results to return
            relationship_boost: Score boost for related content
            
        Returns:
            Dict[str, Any]: Enhanced query results with relationship scoring
        """
        if not self.pool:
            logger.error("Database not initialized")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        try:
            # First, get basic search results
            base_results = await self.query_with_permissions(
                collection_name=collection_name,
                query_text=query_text,
                n_results=n_results * 2  # Get more results for relationship processing
            )
            
            if not base_results["documents"]:
                return base_results
            
            # Get content IDs from results
            content_ids = []
            for metadata in base_results["metadatas"]:
                content_ids.append(metadata.get("content_id"))
            
            # Get relationships for these content items
            async with self.pool.acquire() as conn:
                relationships = await conn.fetch("""
                    SELECT * FROM get_content_relationships($1)
                """, content_ids)
            
            # Build relationship map
            relationship_map = {}
            for rel in relationships:
                source_id = str(rel['source_content_id'])
                if source_id not in relationship_map:
                    relationship_map[source_id] = []
                relationship_map[source_id].append(rel)
                
                if rel['bidirectional']:
                    target_id = str(rel['target_content_id'])
                    if target_id not in relationship_map:
                        relationship_map[target_id] = []
                    relationship_map[target_id].append(rel)
            
            # Process results with relationship scoring
            enhanced_results = []
            for i, metadata in enumerate(base_results["metadatas"]):
                content_id = metadata.get("content_id")
                base_score = 1.0 - base_results["distances"][i]
                
                # Calculate relationship boost
                relationship_score = 0.0
                if content_id in relationship_map:
                    for rel in relationship_map[content_id]:
                        # Type-based boost factors
                        type_boost = {
                            'prerequisite': 0.2,
                            'supplement': 0.15,
                            'sequence': 0.1,
                            'similarity': 0.25,
                            'reference': 0.1,
                            'elaboration': 0.15
                        }.get(rel['relationship_type'], 0.1)
                        
                        relationship_score += float(rel['strength']) * type_boost
                
                final_score = base_score + (relationship_score * relationship_boost)
                
                enhanced_results.append({
                    "document": base_results["documents"][i],
                    "metadata": metadata,
                    "id": base_results["ids"][i],
                    "final_score": final_score,
                    "relationship_score": relationship_score
                })
            
            # Sort by final score and take top results
            enhanced_results.sort(key=lambda x: x["final_score"], reverse=True)
            enhanced_results = enhanced_results[:n_results]
            
            # Convert back to expected format
            return {
                "documents": [r["document"] for r in enhanced_results],
                "metadatas": [r["metadata"] for r in enhanced_results],
                "ids": [r["id"] for r in enhanced_results],
                "distances": [1.0 - r["final_score"] for r in enhanced_results],
                "relationship_scores": [r["relationship_score"] for r in enhanced_results]
            }
            
        except Exception as e:
            logger.error(f"Failed to perform relationship-aware search on Supabase: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    # ========================================
    # Supabase-Specific Methods
    # ========================================
    
    async def add_content_item(self, content_item: ContentItem) -> bool:
        """
        Add a content item to the content_items table.
        
        Args:
            content_item: ContentItem to add
            
        Returns:
            bool: True if addition successful
        """
        if not self.pool:
            logger.error("Database not initialized")
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO content_items (
                        content_id, module_type, content_type, title, description, author,
                        file_name, file_path, file_type, file_size,
                        visibility, created_by, organization_id,
                        processing_status, text_length, chunk_count,
                        parent_content_id, prerequisite_content_ids,
                        topics, language, reading_level, module_metadata,
                        created_at, updated_at, processed_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
                    ON CONFLICT (content_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        processing_status = EXCLUDED.processing_status,
                        text_length = EXCLUDED.text_length,
                        chunk_count = EXCLUDED.chunk_count,
                        processed_at = EXCLUDED.processed_at,
                        updated_at = EXCLUDED.updated_at
                """,
                uuid.UUID(content_item.content_id),
                content_item.module_type.value,
                content_item.content_type.value,
                content_item.title,
                content_item.description,
                content_item.author,
                content_item.file_name,
                content_item.file_path,
                content_item.file_type,
                content_item.file_size,
                content_item.visibility.value,
                uuid.UUID(content_item.created_by) if content_item.created_by else None,
                uuid.UUID(content_item.organization_id) if content_item.organization_id else None,
                content_item.processing_status.value,
                content_item.text_length,
                content_item.chunk_count,
                uuid.UUID(content_item.parent_content_id) if content_item.parent_content_id else None,
                [uuid.UUID(cid) for cid in content_item.prerequisite_content_ids],
                content_item.topics,
                content_item.language,
                content_item.reading_level,
                json.dumps(content_item.module_metadata),
                content_item.created_at,
                content_item.updated_at,
                content_item.processed_at
                )
                
                logger.info(f"Added content item to Supabase: {content_item.content_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add content item to Supabase: {e}")
            return False
    
    async def get_content_item(self, content_id: str) -> Optional[ContentItem]:
        """
        Get a content item by ID.
        
        Args:
            content_id: Content ID to retrieve
            
        Returns:
            ContentItem if found, None otherwise
        """
        if not self.pool:
            logger.error("Database not initialized")
            return None
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM content_items WHERE content_id = $1
                """, uuid.UUID(content_id))
                
                if not row:
                    return None
                
                return ContentItem(
                    content_id=str(row['content_id']),
                    module_type=ModuleType(row['module_type']),
                    content_type=ContentType(row['content_type']),
                    title=row['title'],
                    description=row['description'],
                    author=row['author'],
                    file_name=row['file_name'],
                    file_path=row['file_path'],
                    file_type=row['file_type'],
                    file_size=row['file_size'],
                    visibility=ContentVisibility(row['visibility']),
                    created_by=str(row['created_by']) if row['created_by'] else None,
                    organization_id=str(row['organization_id']) if row['organization_id'] else None,
                    processing_status=row['processing_status'],
                    text_length=row['text_length'],
                    chunk_count=row['chunk_count'],
                    parent_content_id=str(row['parent_content_id']) if row['parent_content_id'] else None,
                    prerequisite_content_ids=[str(cid) for cid in (row['prerequisite_content_ids'] or [])],
                    topics=row['topics'] or [],
                    language=row['language'],
                    reading_level=row['reading_level'],
                    module_metadata=json.loads(row['module_metadata']) if row['module_metadata'] else {},
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    processed_at=row['processed_at']
                )
                
        except Exception as e:
            logger.error(f"Failed to get content item {content_id}: {e}")
            return None
    
    async def delete_content_item(self, content_id: str) -> bool:
        """
        Delete a content item and all its embeddings.
        
        Args:
            content_id: Content ID to delete
            
        Returns:
            bool: True if deletion successful
        """
        if not self.pool:
            logger.error("Database not initialized")
            return False
        
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Delete embeddings first (foreign key constraint)
                    await conn.execute("""
                        DELETE FROM content_embeddings WHERE content_id = $1
                    """, uuid.UUID(content_id))
                    
                    # Delete content item
                    result = await conn.execute("""
                        DELETE FROM content_items WHERE content_id = $1
                    """, uuid.UUID(content_id))
                    
                    deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted content item and embeddings: {content_id}")
                        return True
                    else:
                        logger.warning(f"No content item found with ID: {content_id}")
                        return False
                
        except Exception as e:
            logger.error(f"Failed to delete content item {content_id}: {e}")
            return False
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Supabase database connection pool closed")


# ========================================
# Dual-Write Synchronization Service
# ========================================

class DualWriteVectorDB(EnhancedVectorDatabaseInterface):
    """
    Dual-write vector database for migration between Chroma and Supabase.
    
    Writes to both databases simultaneously to ensure data consistency
    during the migration period. Reads from primary database with fallback
    to secondary for resilience.
    """
    
    def __init__(self, primary_db: EnhancedVectorDatabaseInterface, secondary_db: EnhancedVectorDatabaseInterface):
        """
        Initialize dual-write database wrapper.
        
        Args:
            primary_db: Primary database (reads/writes)
            secondary_db: Secondary database (writes only during sync)
        """
        self.primary_db = primary_db
        self.secondary_db = secondary_db
        self.sync_enabled = True
        self.sync_errors = []
    
    async def initialize(self) -> bool:
        """Initialize both databases."""
        primary_init = await self.primary_db.initialize()
        secondary_init = await self.secondary_db.initialize()
        
        if not primary_init:
            logger.error("Failed to initialize primary database")
            return False
        
        if not secondary_init:
            logger.warning("Failed to initialize secondary database - continuing with primary only")
            self.sync_enabled = False
        
        logger.info(f"Dual-write initialized - sync enabled: {self.sync_enabled}")
        return True
    
    async def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create collection in both databases."""
        primary_result = await self.primary_db.create_collection(name, metadata)
        
        if self.sync_enabled:
            try:
                secondary_result = await self.secondary_db.create_collection(name, metadata)
                if not secondary_result:
                    logger.warning(f"Failed to create collection in secondary DB: {name}")
            except Exception as e:
                logger.warning(f"Error creating collection in secondary DB: {e}")
                self.sync_errors.append(f"create_collection: {e}")
        
        return primary_result
    
    async def add_documents_with_metadata(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        embedding_metadata: List[EmbeddingMetadata],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to both databases."""
        # Write to primary first
        primary_result = await self.primary_db.add_documents_with_metadata(
            collection_name, documents, embeddings, embedding_metadata, ids
        )
        
        if not primary_result:
            logger.error("Failed to write to primary database")
            return False
        
        # Write to secondary if sync enabled
        if self.sync_enabled:
            try:
                secondary_result = await self.secondary_db.add_documents_with_metadata(
                    collection_name, documents, embeddings, embedding_metadata, ids
                )
                if not secondary_result:
                    logger.warning("Failed to sync to secondary database")
                    self.sync_errors.append(f"add_documents sync failed for {len(documents)} docs")
            except Exception as e:
                logger.warning(f"Error syncing to secondary database: {e}")
                self.sync_errors.append(f"add_documents: {e}")
        
        return primary_result
    
    async def query_with_permissions(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 5,
        user: Optional[User] = None,
        module_filter: Optional[ModuleType] = None,
        content_type_filter: Optional[ContentType] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query from primary database with fallback to secondary."""
        try:
            results = await self.primary_db.query_with_permissions(
                collection_name, query_text, query_embedding, n_results,
                user, module_filter, content_type_filter, additional_filters
            )
            
            if results["documents"]:
                return results
            
        except Exception as e:
            logger.error(f"Primary database query failed: {e}")
        
        # Fallback to secondary if enabled
        if self.sync_enabled:
            try:
                logger.info("Falling back to secondary database for query")
                return await self.secondary_db.query_with_permissions(
                    collection_name, query_text, query_embedding, n_results,
                    user, module_filter, content_type_filter, additional_filters
                )
            except Exception as e:
                logger.error(f"Secondary database query also failed: {e}")
        
        return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    async def get_content_embeddings(
        self,
        content_id: str,
        collection_name: Optional[str] = None
    ) -> List[EmbeddingMetadata]:
        """Get embeddings from primary database."""
        return await self.primary_db.get_content_embeddings(content_id, collection_name)
    
    async def delete_content_embeddings(
        self,
        content_id: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete embeddings from both databases."""
        primary_result = await self.primary_db.delete_content_embeddings(content_id, collection_name)
        
        if self.sync_enabled:
            try:
                await self.secondary_db.delete_content_embeddings(content_id, collection_name)
            except Exception as e:
                logger.warning(f"Error deleting from secondary database: {e}")
                self.sync_errors.append(f"delete_embeddings: {e}")
        
        return primary_result
    
    async def update_embedding_metadata(
        self,
        embedding_id: str,
        metadata_updates: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> bool:
        """Update metadata in both databases."""
        primary_result = await self.primary_db.update_embedding_metadata(
            embedding_id, metadata_updates, collection_name
        )
        
        if self.sync_enabled:
            try:
                await self.secondary_db.update_embedding_metadata(
                    embedding_id, metadata_updates, collection_name
                )
            except Exception as e:
                logger.warning(f"Error updating secondary database: {e}")
                self.sync_errors.append(f"update_metadata: {e}")
        
        return primary_result
    
    async def similarity_search_with_relationships(
        self,
        collection_name: str,
        query_text: str,
        content_relationships: List[ContentRelationship],
        n_results: int = 5,
        relationship_boost: float = 0.1
    ) -> Dict[str, Any]:
        """Perform relationship search on primary database."""
        return await self.primary_db.similarity_search_with_relationships(
            collection_name, query_text, content_relationships, n_results, relationship_boost
        )
    
    def get_sync_errors(self) -> List[str]:
        """Get list of synchronization errors."""
        return self.sync_errors.copy()
    
    def clear_sync_errors(self):
        """Clear the synchronization error list."""
        self.sync_errors.clear()
    
    def disable_sync(self):
        """Disable synchronization to secondary database."""
        self.sync_enabled = False
        logger.info("Dual-write synchronization disabled")
    
    def enable_sync(self):
        """Enable synchronization to secondary database."""
        self.sync_enabled = True
        logger.info("Dual-write synchronization enabled")