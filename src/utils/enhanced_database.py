"""
Enhanced vector database utilities for multi-module Alexandria platform.

This module extends the existing vector database abstraction to support
enhanced metadata, permission-aware search, and content relationships
for the unified content schema.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
import json
from pathlib import Path
from datetime import datetime

from src.models import (
    ContentItem, EmbeddingMetadata, User, ContentRelationship,
    ModuleType, ContentType, ContentVisibility, UserRole,
    ContentRelationshipType
)
from src.utils.config import get_settings, DEFAULT_COLLECTION_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedVectorDatabaseInterface(ABC):
    """
    Enhanced abstract interface for vector database operations.
    
    Extends the basic interface with multi-module support, permission-aware
    search, and content relationship capabilities.
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the database connection."""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection with enhanced metadata."""
        pass
    
    @abstractmethod
    async def add_documents_with_metadata(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        embedding_metadata: List[EmbeddingMetadata],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents with enhanced embedding metadata."""
        pass
    
    @abstractmethod
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
        """Query documents with permission and module filtering."""
        pass
    
    @abstractmethod
    async def get_content_embeddings(
        self,
        content_id: str,
        collection_name: Optional[str] = None
    ) -> List[EmbeddingMetadata]:
        """Get all embeddings for a specific content item."""
        pass
    
    @abstractmethod
    async def delete_content_embeddings(
        self,
        content_id: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete all embeddings for a specific content item."""
        pass
    
    @abstractmethod
    async def update_embedding_metadata(
        self,
        embedding_id: str,
        metadata_updates: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> bool:
        """Update metadata for a specific embedding."""
        pass
    
    @abstractmethod
    async def similarity_search_with_relationships(
        self,
        collection_name: str,
        query_text: str,
        content_relationships: List[ContentRelationship],
        n_results: int = 5,
        relationship_boost: float = 0.1
    ) -> Dict[str, Any]:
        """Perform similarity search with relationship-aware scoring."""
        pass


class EnhancedChromaVectorDB(EnhancedVectorDatabaseInterface):
    """
    Enhanced Chroma vector database implementation.
    
    Supports multi-module content, permission-aware search, and
    relationship-based retrieval for the unified Alexandria platform.
    """
    
    def __init__(self):
        """Initialize enhanced Chroma database client."""
        self.settings = get_settings()
        self.client = None
        self._collections = {}
        self._embedding_metadata_cache: Dict[str, List[EmbeddingMetadata]] = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the enhanced Chroma database connection.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Ensure persist directory exists
            persist_dir = Path(self.settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize Chroma client with persistence
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            logger.info(f"Enhanced Chroma database initialized at: {persist_dir}")
            
            # Create enhanced collections if they don't exist
            await self.create_collection(DEFAULT_COLLECTION_NAME, {
                "description": "Enhanced content collection for all Alexandria modules",
                "version": "1.3.0",
                "supports_modules": ["library", "lms", "marketplace"]
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced Chroma database: {e}")
            return False
    
    async def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new enhanced Chroma collection.
        
        Args:
            name: Collection name
            metadata: Optional collection metadata
            
        Returns:
            bool: True if creation successful
        """
        if not self.client:
            logger.error("Database not initialized")
            return False
        
        try:
            # Try to get existing collection first
            try:
                collection = self.client.get_collection(name)
                self._collections[name] = collection
                logger.info(f"Using existing enhanced collection: {name}")
                return True
            except ValueError:
                # Collection doesn't exist, create it
                pass
            
            # Create new collection with enhanced metadata
            collection_metadata = {
                "created_by": "alexandria_enhanced_app",
                "created_at": datetime.now().isoformat(),
                "schema_version": "1.3.0"
            }
            if metadata:
                collection_metadata.update(metadata)
            
            collection = self.client.create_collection(
                name=name,
                metadata=collection_metadata
            )
            
            self._collections[name] = collection
            logger.info(f"Created new enhanced collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create enhanced collection {name}: {e}")
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
        Add documents with enhanced embedding metadata.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            embedding_metadata: List of enhanced metadata
            ids: Optional list of document IDs
            
        Returns:
            bool: True if addition successful
        """
        if not self.client:
            logger.error("Database not initialized")
            return False
        
        if len(documents) != len(embeddings) or len(documents) != len(embedding_metadata):
            logger.error("Documents, embeddings, and metadata must have same length")
            return False
        
        try:
            collection = self._collections.get(collection_name)
            if not collection:
                if not await self.create_collection(collection_name):
                    return False
                collection = self._collections[collection_name]
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Convert enhanced metadata to Chroma format
            chroma_metadatas = []
            for i, metadata in enumerate(embedding_metadata):
                chroma_metadata = {
                    "embedding_id": metadata.embedding_id,
                    "content_id": metadata.content_id,
                    "chunk_index": metadata.chunk_index,
                    "module_type": metadata.module_type.value,
                    "content_type": metadata.content_type.value,
                    "chunk_type": metadata.chunk_type,
                    "visibility": metadata.visibility.value,
                    "creator_id": metadata.creator_id or "",
                    "organization_id": metadata.organization_id or "",
                    "semantic_tags": json.dumps(metadata.semantic_tags),
                    "language": metadata.language,
                    "reading_level": metadata.reading_level or "",
                    "source_location": json.dumps(metadata.source_location),
                    "chunk_length": metadata.chunk_length,
                    "embedding_model": metadata.embedding_model,
                    "embedding_dimension": metadata.embedding_dimension,
                    "created_at": metadata.created_at.isoformat(),
                    "importance_score": metadata.importance_score or 0.0,
                    "quality_score": metadata.quality_score or 0.0
                }
                chroma_metadatas.append(chroma_metadata)
            
            # Add documents with embeddings to collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=chroma_metadatas,
                ids=ids
            )
            
            # Cache metadata for quick access
            for metadata in embedding_metadata:
                content_id = metadata.content_id
                if content_id not in self._embedding_metadata_cache:
                    self._embedding_metadata_cache[content_id] = []
                self._embedding_metadata_cache[content_id].append(metadata)
            
            logger.info(f"Added {len(documents)} enhanced documents to collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add enhanced documents to {collection_name}: {e}")
            return False
    
    async def query_with_permissions(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 10,
        user: Optional[User] = None,
        module_filter: Optional[ModuleType] = None,
        content_type_filter: Optional[ContentType] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query documents with permission and module filtering.
        
        Args:
            collection_name: Name of the collection
            query_text: Query text for similarity search
            query_embedding: Optional pre-computed query embedding
            n_results: Number of results to return
            user: Optional user for permission checking
            module_filter: Optional module filter
            content_type_filter: Optional content type filter
            additional_filters: Optional additional metadata filters
            
        Returns:
            Dict[str, Any]: Enhanced query results
        """
        if not self.client:
            logger.error("Database not initialized")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        try:
            collection = self._collections.get(collection_name)
            if not collection:
                # Try to load existing collection
                try:
                    collection = self.client.get_collection(collection_name)
                    self._collections[collection_name] = collection
                    logger.info(f"Loaded existing collection: {collection_name}")
                except ValueError:
                    logger.error(f"Collection not found: {collection_name}")
                    return {"documents": [], "metadatas": [], "distances": [], "ids": []}
            
            # Log search parameters
            logger.info(f"Enhanced search query: '{query_text}' with n_results={n_results}")
            logger.info(f"User: {user.user_id if user else 'anonymous'}, Module filter: {module_filter}, Content type filter: {content_type_filter}")
            
            # Build permission and module filters
            where_filters = {}
            
            # Permission-based filtering
            if user:
                if user.role != UserRole.ADMIN:
                    # Build visibility filter based on user permissions
                    visibility_conditions = ["public"]
                    
                    # User can see their own private content
                    if user.user_id:
                        where_filters["$or"] = [
                            {"visibility": "public"},
                            {"$and": [{"visibility": "private"}, {"creator_id": user.user_id}]}
                        ]
                        
                        # Add organization content if user belongs to org
                        if user.organization_id:
                            where_filters["$or"].append({
                                "$and": [{"visibility": "organization"}, {"organization_id": user.organization_id}]
                            })
                        
                        # Add premium content if user has subscription
                        if user.subscription_tier in ["pro", "enterprise"]:
                            where_filters["$or"].append({"visibility": "premium"})
                    else:
                        where_filters["visibility"] = "public"
            else:
                # Anonymous users only see public content
                where_filters["visibility"] = "public"
                logger.info("Anonymous user: restricting to public content only")
            
            # Module filtering
            if module_filter:
                where_filters["module_type"] = module_filter.value
            
            # Content type filtering
            if content_type_filter:
                where_filters["content_type"] = content_type_filter.value
            
            # Additional filters
            if additional_filters:
                where_filters.update(additional_filters)
            
            # Log applied filters
            logger.info(f"Applied filters: {where_filters}")
            
            # Perform similarity search with filters
            if query_embedding:
                logger.info(f"Using provided query embedding (dim: {len(query_embedding)})")
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where_filters if where_filters else None
                )
            else:
                logger.info(f"Using query text: '{query_text}'")
                results = collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                    where=where_filters if where_filters else None
                )
            
            # Log raw results
            raw_result_count = len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0
            logger.info(f"Raw Chroma query returned {raw_result_count} results")
            
            if raw_result_count > 0 and results['distances']:
                distances = results['distances'][0]
                logger.info(f"Distance range: {min(distances):.4f} - {max(distances):.4f}")
                logger.info(f"Top 3 similarity scores: {[1.0 - d for d in distances[:3]]}")
                
                # Log document IDs and metadata samples
                for i in range(min(3, raw_result_count)):
                    doc_id = results['ids'][0][i] if results['ids'] else 'unknown'
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    content_id = metadata.get('content_id', 'unknown')
                    content_type = metadata.get('content_type', 'unknown')
                    logger.info(f"  Result {i+1}: ID={doc_id}, Content={content_id}, Type={content_type}, Distance={distances[i]:.4f}")
            
            # Process and enhance results
            enhanced_results = self._process_enhanced_results(results)
            
            logger.info(f"Enhanced query returned {len(enhanced_results['documents'])} results")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to perform enhanced query on {collection_name}: {e}")
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
            collection_name: Optional collection name
            
        Returns:
            List of embedding metadata
        """
        try:
            # Check cache first
            if content_id in self._embedding_metadata_cache:
                return self._embedding_metadata_cache[content_id]
            
            collection_name = collection_name or DEFAULT_COLLECTION_NAME
            collection = self._collections.get(collection_name)
            if not collection:
                return []
            
            # Query for all embeddings with this content_id
            results = collection.get(
                where={"content_id": content_id},
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Convert to EmbeddingMetadata objects
            embeddings = []
            if results["metadatas"]:
                for i, metadata in enumerate(results["metadatas"]):
                    embedding_metadata = self._chroma_metadata_to_embedding_metadata(
                        metadata,
                        results["documents"][i] if results["documents"] else "",
                        results["embeddings"][i] if results["embeddings"] else []
                    )
                    embeddings.append(embedding_metadata)
            
            # Update cache
            self._embedding_metadata_cache[content_id] = embeddings
            
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
            collection_name: Optional collection name
            
        Returns:
            bool: True if deletion successful
        """
        try:
            collection_name = collection_name or DEFAULT_COLLECTION_NAME
            collection = self._collections.get(collection_name)
            if not collection:
                logger.error(f"Collection not found: {collection_name}")
                return False
            
            # Get all IDs for this content
            results = collection.get(
                where={"content_id": content_id},
                include=["documents"]
            )
            
            if results["ids"]:
                # Delete all embeddings for this content
                collection.delete(ids=results["ids"])
                
                # Remove from cache
                self._embedding_metadata_cache.pop(content_id, None)
                
                logger.info(f"Deleted {len(results['ids'])} embeddings for content: {content_id}")
            
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
            collection_name: Optional collection name
            
        Returns:
            bool: True if update successful
        """
        try:
            collection_name = collection_name or DEFAULT_COLLECTION_NAME
            collection = self._collections.get(collection_name)
            if not collection:
                logger.error(f"Collection not found: {collection_name}")
                return False
            
            # Get current metadata
            results = collection.get(
                where={"embedding_id": embedding_id},
                include=["metadatas"]
            )
            
            if not results["metadatas"]:
                logger.error(f"Embedding not found: {embedding_id}")
                return False
            
            # Update metadata
            current_metadata = results["metadatas"][0]
            current_metadata.update(metadata_updates)
            
            # Update in collection (Chroma doesn't support direct metadata updates,
            # so we need to delete and re-add)
            doc_id = results["ids"][0]
            
            # Get full document data
            full_results = collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Delete old entry
            collection.delete(ids=[doc_id])
            
            # Re-add with updated metadata
            collection.add(
                documents=full_results["documents"],
                embeddings=full_results["embeddings"],
                metadatas=[current_metadata],
                ids=[doc_id]
            )
            
            # Clear cache for affected content
            content_id = current_metadata.get("content_id")
            if content_id:
                self._embedding_metadata_cache.pop(content_id, None)
            
            logger.info(f"Updated metadata for embedding: {embedding_id}")
            return True
            
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
        try:
            # First perform regular similarity search
            base_results = await self.query_with_permissions(
                collection_name=collection_name,
                query_text=query_text,
                n_results=n_results * 2  # Get more results for relationship processing
            )
            
            if not base_results["documents"]:
                return base_results
            
            # Build relationship map for quick lookup
            relationship_map = {}
            for rel in content_relationships:
                if rel.source_content_id not in relationship_map:
                    relationship_map[rel.source_content_id] = []
                relationship_map[rel.source_content_id].append(rel)
                
                if rel.bidirectional:
                    if rel.target_content_id not in relationship_map:
                        relationship_map[rel.target_content_id] = []
                    relationship_map[rel.target_content_id].append(rel)
            
            # Process results with relationship scoring
            enhanced_results = []
            for i, metadata in enumerate(base_results["metadatas"]):
                content_id = metadata.get("content_id")
                base_score = 1.0 - base_results["distances"][i]  # Convert distance to similarity
                
                # Apply relationship boost
                relationship_score = 0.0
                if content_id in relationship_map:
                    for rel in relationship_map[content_id]:
                        # Boost based on relationship strength and type
                        type_boost = {
                            ContentRelationshipType.PREREQUISITE: 0.2,
                            ContentRelationshipType.SUPPLEMENT: 0.15,
                            ContentRelationshipType.SEQUENCE: 0.1,
                            ContentRelationshipType.SIMILARITY: 0.25,
                            ContentRelationshipType.REFERENCE: 0.1,
                            ContentRelationshipType.ELABORATION: 0.15
                        }.get(rel.relationship_type, 0.1)
                        
                        relationship_score += rel.strength * type_boost
                
                final_score = base_score + (relationship_score * relationship_boost)
                
                enhanced_results.append({
                    "document": base_results["documents"][i],
                    "metadata": metadata,
                    "id": base_results["ids"][i],
                    "base_score": base_score,
                    "relationship_score": relationship_score,
                    "final_score": final_score,
                    "distance": 1.0 - final_score  # Convert back to distance
                })
            
            # Sort by final score (highest first)
            enhanced_results.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Take top n_results
            enhanced_results = enhanced_results[:n_results]
            
            # Convert back to expected format
            return {
                "documents": [r["document"] for r in enhanced_results],
                "metadatas": [r["metadata"] for r in enhanced_results],
                "ids": [r["id"] for r in enhanced_results],
                "distances": [r["distance"] for r in enhanced_results],
                "relationship_scores": [r["relationship_score"] for r in enhanced_results]
            }
            
        except Exception as e:
            logger.error(f"Failed to perform relationship-aware search: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _process_enhanced_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process Chroma results to enhanced format."""
        if not results["documents"] or not results["documents"][0]:
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }
    
    def _chroma_metadata_to_embedding_metadata(
        self,
        chroma_metadata: Dict[str, Any],
        document: str,
        embedding: List[float]
    ) -> EmbeddingMetadata:
        """Convert Chroma metadata to EmbeddingMetadata object."""
        return EmbeddingMetadata(
            embedding_id=chroma_metadata.get("embedding_id", ""),
            content_id=chroma_metadata.get("content_id", ""),
            chunk_index=chroma_metadata.get("chunk_index", 0),
            module_type=ModuleType(chroma_metadata.get("module_type", "library")),
            content_type=ContentType(chroma_metadata.get("content_type", "book")),
            chunk_type=chroma_metadata.get("chunk_type", "paragraph"),
            visibility=ContentVisibility(chroma_metadata.get("visibility", "private")),
            creator_id=chroma_metadata.get("creator_id") or None,
            organization_id=chroma_metadata.get("organization_id") or None,
            semantic_tags=json.loads(chroma_metadata.get("semantic_tags", "[]")),
            language=chroma_metadata.get("language", "en"),
            reading_level=chroma_metadata.get("reading_level") or None,
            source_location=json.loads(chroma_metadata.get("source_location", "{}")),
            text_content=document,
            chunk_length=chroma_metadata.get("chunk_length", len(document)),
            embedding_model=chroma_metadata.get("embedding_model", "text-embedding-ada-002"),
            embedding_dimension=chroma_metadata.get("embedding_dimension", 1536),
            created_at=datetime.fromisoformat(chroma_metadata.get("created_at", datetime.now().isoformat())),
            importance_score=chroma_metadata.get("importance_score"),
            quality_score=chroma_metadata.get("quality_score")
        )


# ========================================
# Enhanced Database Factory
# ========================================

def get_enhanced_vector_database() -> EnhancedVectorDatabaseInterface:
    """
    Get the configured enhanced vector database implementation.
    
    Returns:
        EnhancedVectorDatabaseInterface: Enhanced database implementation instance
    """
    settings = get_settings()
    
    if settings.vector_db_type == "chroma":
        return EnhancedChromaVectorDB()
    elif settings.vector_db_type == "supabase":
        from src.utils.supabase_vector_db import SupabaseVectorDB
        return SupabaseVectorDB()
    else:
        raise ValueError(f"Unsupported vector database type: {settings.vector_db_type}")


# Global enhanced database instance
_enhanced_vector_db: Optional[EnhancedVectorDatabaseInterface] = None


async def get_enhanced_database() -> EnhancedVectorDatabaseInterface:
    """
    Get the global enhanced database instance with lazy initialization.
    
    Returns:
        EnhancedVectorDatabaseInterface: Initialized enhanced database instance
    """
    global _enhanced_vector_db
    
    if _enhanced_vector_db is None:
        _enhanced_vector_db = get_enhanced_vector_database()
        if not await _enhanced_vector_db.initialize():
            raise RuntimeError("Failed to initialize enhanced vector database")
    
    return _enhanced_vector_db