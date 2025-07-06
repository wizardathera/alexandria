"""
Vector database utilities for the Alexandria application.

This module provides an abstraction layer for vector database operations,
supporting both Chroma (Phase 1) and Supabase (Phase 2) backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
import uuid
from pathlib import Path

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorDatabaseInterface(ABC):
    """
    Abstract interface for vector database operations.
    
    This interface allows easy switching between different vector database
    implementations (Chroma, Supabase, etc.).
    """
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the database connection."""
        pass
    
    @abstractmethod
    async def create_collection(self, name: str) -> bool:
        """Create a new collection."""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to a collection."""
        pass
    
    @abstractmethod
    async def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query documents from a collection."""
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        pass
    
    @abstractmethod
    async def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        pass


class ChromaVectorDB(VectorDatabaseInterface):
    """
    Chroma vector database implementation.
    
    Used for Phase 1 local development and prototyping.
    """
    
    def __init__(self):
        """Initialize Chroma database client."""
        self.settings = get_settings()
        self.client = None
        self._collections = {}
    
    async def initialize(self) -> bool:
        """
        Initialize the Chroma database connection.
        
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
            
            logger.info(f"Chroma database initialized at: {persist_dir}")
            
            # Create default collection if it doesn't exist
            await self.create_collection(self.settings.chroma_collection_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Chroma database: {e}")
            return False
    
    async def create_collection(self, name: str) -> bool:
        """
        Create a new Chroma collection or get existing one.
        
        Args:
            name: Collection name
            
        Returns:
            bool: True if creation/retrieval successful
        """
        if not self.client:
            logger.error("Database not initialized")
            return False
        
        try:
            # Use get_or_create_collection pattern for robust collection handling
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={"created_by": "dbc_app"}
            )
            
            self._collections[name] = collection
            
            # Check if collection was newly created or already existed
            try:
                existing_count = collection.count()
                if existing_count > 0:
                    logger.info(f"Using existing collection '{name}' with {existing_count} documents")
                else:
                    logger.info(f"Created new collection '{name}'")
            except Exception:
                # If count fails, assume it's a new collection
                logger.info(f"Retrieved/created collection '{name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get/create collection {name}: {e}")
            return False
    
    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to a Chroma collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            bool: True if addition successful
        """
        if not self.client:
            logger.error("Database not initialized")
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
            
            # Add documents to collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to {collection_name}: {e}")
            return False
    
    async def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query documents from a Chroma collection.
        
        Args:
            collection_name: Name of the collection
            query_text: Query text for similarity search
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Dict[str, Any]: Query results
        """
        if not self.client:
            logger.error("Database not initialized")
            return {"documents": [], "metadatas": [], "distances": []}
        
        try:
            collection = self._collections.get(collection_name)
            if not collection:
                logger.error(f"Collection not found: {collection_name}")
                return {"documents": [], "metadatas": [], "distances": []}
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            
            logger.info(f"Query returned {len(results['documents'][0])} results")
            
            # Flatten results for easier processing
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
            
        except Exception as e:
            logger.error(f"Failed to query collection {collection_name}: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    async def delete_collection(self, name: str) -> bool:
        """
        Delete a Chroma collection.
        
        Args:
            name: Collection name
            
        Returns:
            bool: True if deletion successful
        """
        if not self.client:
            logger.error("Database not initialized")
            return False
        
        try:
            self.client.delete_collection(name)
            if name in self._collections:
                del self._collections[name]
            
            logger.info(f"Deleted collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {name}: {e}")
            return False
    
    async def get_collection_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a Chroma collection.
        
        Args:
            name: Collection name
            
        Returns:
            Dict[str, Any]: Collection information
        """
        if not self.client:
            logger.error("Database not initialized")
            return {}
        
        try:
            collection = self._collections.get(name)
            if not collection:
                if not await self.create_collection(name):
                    return {}
                collection = self._collections[name]
            
            count = collection.count()
            
            return {
                "name": name,
                "document_count": count,
                "type": "chroma"
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for {name}: {e}")
            return {}


class SupabaseVectorDB(VectorDatabaseInterface):
    """
    Supabase vector database implementation.
    
    Placeholder for Phase 2 cloud deployment with multi-user support.
    """
    
    def __init__(self):
        """Initialize Supabase database client."""
        self.settings = get_settings()
        # TODO: Implement Supabase client initialization
    
    async def initialize(self) -> bool:
        """Initialize Supabase connection."""
        # TODO: Implement Supabase initialization
        logger.info("Supabase implementation not yet available")
        return False
    
    async def create_collection(self, name: str) -> bool:
        """Create Supabase collection."""
        # TODO: Implement Supabase collection creation
        return False
    
    async def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """Add documents to Supabase."""
        # TODO: Implement Supabase document addition
        return False
    
    async def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Query Supabase collection."""
        # TODO: Implement Supabase querying
        return {"documents": [], "metadatas": [], "distances": []}
    
    async def delete_collection(self, name: str) -> bool:
        """Delete Supabase collection."""
        # TODO: Implement Supabase collection deletion
        return False
    
    async def get_collection_info(self, name: str) -> Dict[str, Any]:
        """Get Supabase collection info."""
        # TODO: Implement Supabase collection info
        return {}


def get_vector_database() -> VectorDatabaseInterface:
    """
    Get the configured vector database implementation.
    
    Returns:
        VectorDatabaseInterface: Database implementation instance
    """
    settings = get_settings()
    
    if settings.vector_db_type == "chroma":
        return ChromaVectorDB()
    elif settings.vector_db_type == "supabase":
        return SupabaseVectorDB()
    else:
        raise ValueError(f"Unsupported vector database type: {settings.vector_db_type}")


# Global database instance
_vector_db: Optional[VectorDatabaseInterface] = None


async def get_database() -> VectorDatabaseInterface:
    """
    Get the global database instance with lazy initialization.
    
    Returns:
        VectorDatabaseInterface: Initialized database instance
    """
    global _vector_db
    
    if _vector_db is None:
        _vector_db = get_vector_database()
        if not await _vector_db.initialize():
            raise RuntimeError("Failed to initialize vector database")
    
    return _vector_db