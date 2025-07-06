"""
Book ingestion service for the Alexandria application.

This module orchestrates the complete book ingestion pipeline:
loading documents, chunking text, generating embeddings, and storing
in the vector database with progress tracking.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    from langchain.schema import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str = "", metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from src.utils.document_loaders import DocumentLoaderManager
from src.utils.text_chunking import ChunkingManager, ChunkingConfig
from src.utils.embeddings import EmbeddingService, EmbeddingMetrics
from src.utils.database import get_database
from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)


class IngestionStatus(Enum):
    """Enumeration of ingestion status values."""
    PENDING = "pending"
    LOADING = "loading"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionProgress:
    """Progress tracking for book ingestion."""
    book_id: str
    status: IngestionStatus = IngestionStatus.PENDING
    progress: float = 0.0
    message: str = "Ingestion queued"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Detailed metrics
    file_size: int = 0
    text_length: int = 0
    chunk_count: int = 0
    embedding_metrics: Optional[EmbeddingMetrics] = None
    
    def update_status(self, status: IngestionStatus, progress: float, message: str):
        """Update ingestion status and progress."""
        self.status = status
        self.progress = progress
        self.message = message
        
        if status == IngestionStatus.LOADING and not self.started_at:
            self.started_at = datetime.now()
        elif status == IngestionStatus.COMPLETED:
            self.completed_at = datetime.now()
            self.progress = 1.0
        elif status == IngestionStatus.FAILED:
            self.completed_at = datetime.now()
    
    def set_error(self, error: str):
        """Set error and mark as failed."""
        self.error = error
        self.status = IngestionStatus.FAILED
        self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "book_id": self.book_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metrics": {
                "file_size": self.file_size,
                "text_length": self.text_length,
                "chunk_count": self.chunk_count,
                "embedding_cost": self.embedding_metrics.get_cost_estimate() if self.embedding_metrics else 0
            }
        }


@dataclass
class BookMetadata:
    """Comprehensive book metadata."""
    book_id: str
    title: str
    author: Optional[str] = None
    file_type: str = ""
    file_name: str = ""
    file_path: str = ""
    file_size: int = 0
    upload_date: datetime = field(default_factory=datetime.now)
    ingestion_date: Optional[datetime] = None
    user_id: Optional[str] = None  # For Phase 2 multi-user support
    
    # Content metrics
    text_length: int = 0
    chunk_count: int = 0
    embedding_dimension: int = 0
    
    # Processing metadata
    chunking_strategy: str = "recursive"
    embedding_model: str = "text-embedding-ada-002"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage and API responses."""
        return {
            "book_id": self.book_id,
            "title": self.title,
            "author": self.author,
            "file_type": self.file_type,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "upload_date": self.upload_date.isoformat(),
            "ingestion_date": self.ingestion_date.isoformat() if self.ingestion_date else None,
            "user_id": self.user_id,
            "text_length": self.text_length,
            "chunk_count": self.chunk_count,
            "embedding_dimension": self.embedding_dimension,
            "chunking_strategy": self.chunking_strategy,
            "embedding_model": self.embedding_model
        }


class BookIngestionService:
    """
    Service for ingesting books into the vector database.
    
    Handles the complete pipeline from file loading to vector storage
    with comprehensive progress tracking and error recovery.
    """
    
    def __init__(self):
        """Initialize the book ingestion service."""
        self.settings = get_settings()
        self.document_loader = DocumentLoaderManager()
        self.chunking_manager = ChunkingManager()
        self.embedding_service = EmbeddingService(use_cache=True)
        
        # Progress tracking
        self.active_ingestions: Dict[str, IngestionProgress] = {}
        self.completed_ingestions: Dict[str, IngestionProgress] = {}
        
        # Status callbacks
        self.progress_callbacks: List[Callable[[IngestionProgress], None]] = []
    
    def add_progress_callback(self, callback: Callable[[IngestionProgress], None]):
        """
        Add a callback function for progress updates.
        
        Args:
            callback: Function to call on progress updates
        """
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, progress: IngestionProgress):
        """Notify all registered callbacks of progress update."""
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    async def ingest_book(
        self,
        book_id: str,
        file_path: Path,
        metadata: Optional[BookMetadata] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        embedding_model: Optional[str] = None
    ) -> IngestionProgress:
        """
        Ingest a book file into the vector database.
        
        Args:
            book_id: Unique book identifier
            file_path: Path to the book file
            metadata: Optional book metadata
            chunking_config: Optional chunking configuration
            embedding_model: Optional embedding model name
            
        Returns:
            IngestionProgress: Final ingestion progress
        """
        # Initialize progress tracking
        progress = IngestionProgress(book_id=book_id)
        self.active_ingestions[book_id] = progress
        
        try:
            logger.info(f"Starting ingestion for book {book_id}: {file_path}")
            
            # Step 1: Load document
            progress.update_status(
                IngestionStatus.LOADING, 
                0.1, 
                f"Loading {file_path.suffix} file"
            )
            self._notify_progress(progress)
            
            documents, doc_metadata = await self._load_document(file_path)
            progress.file_size = file_path.stat().st_size
            progress.text_length = sum(len(doc.page_content) for doc in documents)
            
            # Validate extracted text length
            if progress.text_length == 0:
                error_msg = (
                    f"Document {file_path.name} contains no extractable text. "
                    "This may be a scanned document requiring OCR processing."
                )
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            logger.info(f"📄 Document loaded: {len(documents)} chunks, {progress.text_length} characters")
            
            # Log first chunk preview for validation
            if documents and documents[0].page_content:
                preview = documents[0].page_content[:150].replace('\n', ' ').strip()
                logger.info(f"📝 Content preview: '{preview}...'")
            
            # Update or create book metadata
            if not metadata:
                metadata = BookMetadata(
                    book_id=book_id,
                    title=doc_metadata.get('title', file_path.stem),
                    author=doc_metadata.get('author'),
                    file_type=file_path.suffix.lower().lstrip('.'),
                    file_name=file_path.name,
                    file_path=str(file_path),
                    file_size=progress.file_size,
                    text_length=progress.text_length
                )
            
            # Step 2: Chunk documents
            progress.update_status(
                IngestionStatus.CHUNKING,
                0.3,
                f"Chunking text into segments"
            )
            self._notify_progress(progress)
            
            chunked_docs = await self._chunk_documents(
                documents, 
                metadata.file_type,
                chunking_config
            )
            progress.chunk_count = len(chunked_docs)
            metadata.chunk_count = len(chunked_docs)
            
            # Log chunking results
            chunk_sizes = [len(chunk.page_content) for chunk in chunked_docs]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            logger.info(f"📝 Chunking complete: {len(chunked_docs)} chunks created, avg size: {avg_chunk_size:.0f} chars")
            
            # Step 3: Generate embeddings
            progress.update_status(
                IngestionStatus.EMBEDDING,
                0.6,
                f"Generating embeddings for {len(chunked_docs)} chunks"
            )
            self._notify_progress(progress)
            
            embeddings, embedding_metrics = await self._generate_embeddings(
                chunked_docs,
                embedding_model
            )
            progress.embedding_metrics = embedding_metrics
            metadata.embedding_model = embedding_model or "text-embedding-ada-002"
            metadata.embedding_dimension = len(embeddings[0]) if embeddings else 0
            
            # Log embedding results
            total_tokens = embedding_metrics.total_tokens if embedding_metrics else 0
            cost_estimate = embedding_metrics.get_cost_estimate() if embedding_metrics else 0.0
            logger.info(f"🔢 Embeddings generated: {len(embeddings)} vectors, {total_tokens} tokens, ~${cost_estimate:.4f}")
            
            # Step 4: Store in vector database
            progress.update_status(
                IngestionStatus.STORING,
                0.9,
                "Storing in vector database"
            )
            self._notify_progress(progress)
            
            await self._store_in_database(book_id, chunked_docs, embeddings, metadata)
            
            # Step 5: Complete
            metadata.ingestion_date = datetime.now()
            progress.update_status(
                IngestionStatus.COMPLETED,
                1.0,
                f"Successfully ingested {len(chunked_docs)} chunks"
            )
            self._notify_progress(progress)
            
            # Log comprehensive ingestion summary
            duration = (datetime.now() - progress.started_at).total_seconds() if progress.started_at else 0
            logger.info(
                f"🎉 Ingestion completed for '{metadata.title}': "
                f"{len(chunked_docs)} chunks, {progress.text_length} chars, "
                f"{progress.embedding_metrics.total_tokens if progress.embedding_metrics else 0} tokens, "
                f"~${cost_estimate:.4f}, {duration:.1f}s"
            )
            
        except Exception as e:
            error_msg = f"Ingestion failed: {str(e)}"
            logger.error(f"Book ingestion failed for {book_id}: {e}")
            progress.set_error(error_msg)
            self._notify_progress(progress)
        
        finally:
            # Move from active to completed
            if book_id in self.active_ingestions:
                del self.active_ingestions[book_id]
            self.completed_ingestions[book_id] = progress
        
        return progress
    
    async def _load_document(self, file_path: Path) -> tuple[List[Any], Dict[str, Any]]:
        """Load document using appropriate loader."""
        return await asyncio.to_thread(
            self.document_loader.load_document, file_path
        )
    
    async def _chunk_documents(
        self,
        documents: List[Any],
        doc_type: str,
        config: Optional[ChunkingConfig] = None
    ) -> List[Any]:
        """Chunk documents into smaller pieces."""
        # Get recommended strategy
        total_length = sum(len(doc.page_content) for doc in documents)
        strategy = self.chunking_manager.get_recommended_strategy(doc_type, total_length)
        
        return await asyncio.to_thread(
            self.chunking_manager.chunk_documents,
            documents,
            strategy,
            config
        )
    
    async def _generate_embeddings(
        self,
        documents: List[Any],
        model: Optional[str] = None
    ) -> tuple[List[List[float]], EmbeddingMetrics]:
        """Generate embeddings for documents."""
        return await self.embedding_service.embed_documents(documents, model)
    
    def _clean_metadata_for_storage(self, metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata dictionary for ChromaDB storage.
        
        ChromaDB only accepts str, int, float, or bool values in metadata.
        This function removes None values and converts others to acceptable types.
        
        Args:
            metadata_dict: Raw metadata dictionary
            
        Returns:
            Dict[str, Any]: Cleaned metadata dictionary
        """
        cleaned = {}
        removed_keys = []
        
        for key, value in metadata_dict.items():
            if value is None:
                # Remove None values entirely
                removed_keys.append(key)
                continue
            elif isinstance(value, (str, int, float, bool)):
                # Keep accepted types as-is
                cleaned[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to string
                cleaned[key] = str(value)
            else:
                # Convert other types to string
                cleaned[key] = str(value)
        
        if removed_keys:
            logger.debug(f"🧹 Removed None metadata keys: {removed_keys}")
        
        return cleaned
    
    async def _store_in_database(
        self,
        book_id: str,
        documents: List[Any],
        embeddings: List[List[float]],
        metadata: BookMetadata
    ):
        """Store documents and embeddings in vector database."""
        # Note: embeddings parameter is required for API compatibility but not used in current implementation
        _ = embeddings  # Explicitly mark as intentionally unused
        db = await get_database()
        collection_name = self.settings.chroma_collection_name
        
        # Prepare documents for storage
        doc_texts = [doc.page_content for doc in documents]
        doc_metadatas = []
        doc_ids = []
        
        logger.info(f"📦 Preparing {len(documents)} chunks for storage in vector database")
        
        for i, doc in enumerate(documents):
            # Create unique ID for each chunk
            chunk_id = f"{book_id}_chunk_{i}"
            doc_ids.append(chunk_id)
            
            # Prepare metadata with safe values
            chunk_metadata = {
                **doc.metadata,
                "book_id": book_id,
                "book_title": metadata.title or "Unknown Title",
                "book_author": metadata.author or "Unknown Author",
                "chunk_id": chunk_id,
                "chunk_index": i,
                "ingestion_date": metadata.ingestion_date.isoformat() if metadata.ingestion_date else "",
                "file_type": metadata.file_type or "unknown",
                "file_name": metadata.file_name or "unknown",
                "chunk_length": len(doc.page_content)
            }
            
            # Clean metadata to ensure ChromaDB compatibility
            cleaned_metadata = self._clean_metadata_for_storage(chunk_metadata)
            doc_metadatas.append(cleaned_metadata)
        
        logger.info(f"🧹 Metadata cleaned and prepared for {len(doc_metadatas)} chunks")
        
        # Log sample metadata for debugging
        if doc_metadatas:
            sample_metadata = {k: v for k, v in doc_metadatas[0].items() if k != "chunk_id"}
            logger.debug(f"📋 Sample chunk metadata: {sample_metadata}")
        
        # Store in vector database
        success = await db.add_documents(
            collection_name,
            doc_texts,
            doc_metadatas,
            doc_ids
        )
        
        if not success:
            raise RuntimeError("Failed to store documents in vector database")
        
        logger.info(f"✅ Successfully stored {len(documents)} chunks in vector database")
        
        # Also store book metadata separately for easy retrieval
        await self._store_book_metadata(metadata)
    
    async def _store_book_metadata(self, metadata: BookMetadata):
        """Store book metadata for easy retrieval."""
        # For now, store as JSON file in data/users directory
        # In Phase 2, this would go to a proper database
        
        metadata_dir = Path(self.settings.user_data_path)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_dir / f"{metadata.book_id}_metadata.json"
        
        await asyncio.to_thread(
            metadata_file.write_text,
            json.dumps(metadata.to_dict(), indent=2)
        )
    
    def get_ingestion_status(self, book_id: str) -> Optional[IngestionProgress]:
        """
        Get ingestion status for a book.
        
        Args:
            book_id: Book identifier
            
        Returns:
            Optional[IngestionProgress]: Current progress or None if not found
        """
        return (
            self.active_ingestions.get(book_id) or 
            self.completed_ingestions.get(book_id)
        )
    
    def get_active_ingestions(self) -> Dict[str, IngestionProgress]:
        """Get all currently active ingestions."""
        return self.active_ingestions.copy()
    
    def get_completed_ingestions(self) -> Dict[str, IngestionProgress]:
        """Get all completed ingestions."""
        return self.completed_ingestions.copy()
    
    async def cancel_ingestion(self, book_id: str) -> bool:
        """
        Cancel an active ingestion.
        
        Args:
            book_id: Book identifier
            
        Returns:
            bool: True if cancellation was successful
        """
        if book_id in self.active_ingestions:
            progress = self.active_ingestions[book_id]
            progress.set_error("Ingestion cancelled by user")
            self._notify_progress(progress)
            
            # Move to completed
            del self.active_ingestions[book_id]
            self.completed_ingestions[book_id] = progress
            
            logger.info(f"Ingestion cancelled: {book_id}")
            return True
        
        return False
    
    async def delete_book(self, book_id: str) -> bool:
        """
        Delete a book and all its data.
        
        Args:
            book_id: Book identifier
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Remove from vector database
            db = await get_database()
            collection_name = self.settings.chroma_collection_name
            
            # Get all chunk IDs for this book
            results = await db.query(
                collection_name,
                query_text="",  # Empty query to get all
                n_results=10000,  # Large number to get all chunks
                where={"book_id": book_id}
            )
            
            if results.get("ids"):
                # Delete individual chunks (Chroma doesn't support bulk delete by metadata)
                for _ in results["ids"]:
                    # This would need to be implemented in the database interface
                    pass
            
            # Remove metadata file
            metadata_file = Path(self.settings.user_data_path) / f"{book_id}_metadata.json"
            if metadata_file.exists():
                await asyncio.to_thread(metadata_file.unlink)
            
            # Remove from progress tracking
            self.active_ingestions.pop(book_id, None)
            self.completed_ingestions.pop(book_id, None)
            
            logger.info(f"Book deleted: {book_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete book {book_id}: {e}")
            return False


# Global service instance
_ingestion_service: Optional[BookIngestionService] = None


def get_ingestion_service() -> BookIngestionService:
    """
    Get the global ingestion service instance.
    
    Returns:
        BookIngestionService: Service instance
    """
    global _ingestion_service
    
    if _ingestion_service is None:
        _ingestion_service = BookIngestionService()
    
    return _ingestion_service