"""
Book management API endpoints for the DBC application.

This module provides endpoints for uploading, ingesting, and managing books
in various formats (PDF, EPUB, DOC, TXT, HTML).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio
from pathlib import Path
import json

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.services.ingestion import get_ingestion_service, BookMetadata as ServiceBookMetadata

logger = get_logger(__name__)
router = APIRouter()


async def start_ingestion(
    ingestion_service,
    book_id: str,
    file_path: Path,
    metadata: ServiceBookMetadata
):
    """
    Background task to start book ingestion.
    
    Args:
        ingestion_service: Ingestion service instance
        book_id: Book identifier
        file_path: Path to uploaded file
        metadata: Book metadata
    """
    try:
        await ingestion_service.ingest_book(book_id, file_path, metadata)
    except Exception as e:
        logger.error(f"Background ingestion failed for {book_id}: {e}")


class BookMetadata(BaseModel):
    """Book metadata model for API responses."""
    id: str
    title: str
    author: Optional[str] = None
    file_type: str
    file_size: int
    upload_date: datetime
    ingestion_status: str = Field(default="pending")
    user_id: Optional[str] = None  # Prepared for Phase 2 multi-user support
    text_length: int = 0
    chunk_count: int = 0
    ingestion_date: Optional[datetime] = None


class BookUploadResponse(BaseModel):
    """Book upload response model."""
    book_id: str
    message: str
    metadata: BookMetadata


class BookListResponse(BaseModel):
    """Book list response model."""
    books: List[BookMetadata]
    total: int


class IngestionStatus(BaseModel):
    """Ingestion status response model."""
    book_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


@router.post("/books/upload", response_model=BookUploadResponse)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    author: Optional[str] = None
):
    """
    Upload a book file for ingestion.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded book file
        title: Optional book title (extracted if not provided)
        author: Optional book author (extracted if not provided)
        
    Returns:
        BookUploadResponse: Upload confirmation with book metadata
    """
    settings = get_settings()
    logger.info(f"Starting book upload: {file.filename}")
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    supported_formats = settings.get_supported_formats_list()
    
    if file_extension not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}. "
                   f"Supported formats: {', '.join(supported_formats)}"
        )
    
    # Validate file size
    content = await file.read()
    file_size = len(content)
    max_size = settings.get_max_upload_size_bytes()
    
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size} bytes. "
                   f"Maximum size: {settings.max_upload_size_mb} MB"
        )
    
    # Generate unique book ID
    book_id = str(uuid.uuid4())
    
    # Save file to storage
    storage_path = Path(settings.books_storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    file_path = storage_path / f"{book_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Book file saved: {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save book file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    # Create service metadata for ingestion
    service_metadata = ServiceBookMetadata(
        book_id=book_id,
        title=title or Path(file.filename).stem,
        author=author,
        file_type=file_extension,
        file_name=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        upload_date=datetime.now(),
        user_id=None  # Phase 1: single-user
    )
    
    # Start background ingestion task
    ingestion_service = get_ingestion_service()
    background_tasks.add_task(
        start_ingestion, 
        ingestion_service, 
        book_id, 
        file_path, 
        service_metadata
    )
    
    # Create response metadata
    response_metadata = BookMetadata(
        id=book_id,
        title=service_metadata.title,
        author=service_metadata.author,
        file_type=file_extension,
        file_size=file_size,
        upload_date=service_metadata.upload_date,
        ingestion_status="pending",
        user_id=None
    )
    
    logger.info(f"Book uploaded successfully: {book_id}")
    
    return BookUploadResponse(
        book_id=book_id,
        message="Book uploaded successfully. Ingestion started.",
        metadata=response_metadata
    )


@router.get("/books", response_model=BookListResponse)
async def list_books(
    skip: int = 0,
    limit: int = 50,
    user_id: Optional[str] = None  # Prepared for Phase 2
):
    """
    List all uploaded books with optional pagination.
    
    Args:
        skip: Number of books to skip
        limit: Maximum number of books to return
        user_id: User ID filter (Phase 2)
        
    Returns:
        BookListResponse: List of books with metadata
    """
    logger.info(f"Listing books: skip={skip}, limit={limit}")
    
    try:
        # Get all book metadata files
        settings = get_settings()
        metadata_dir = Path(settings.user_data_path)
        
        if not metadata_dir.exists():
            return BookListResponse(books=[], total=0)
        
        books = []
        metadata_files = list(metadata_dir.glob("*_metadata.json"))
        
        # Load metadata from files
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Get ingestion status
                ingestion_service = get_ingestion_service()
                progress = ingestion_service.get_ingestion_status(data['book_id'])
                status = progress.status.value if progress else "unknown"
                
                book_metadata = BookMetadata(
                    id=data['book_id'],
                    title=data['title'],
                    author=data.get('author'),
                    file_type=data['file_type'],
                    file_size=data['file_size'],
                    upload_date=datetime.fromisoformat(data['upload_date']),
                    ingestion_status=status,
                    user_id=data.get('user_id'),
                    text_length=data.get('text_length', 0),
                    chunk_count=data.get('chunk_count', 0),
                    ingestion_date=datetime.fromisoformat(data['ingestion_date']) if data.get('ingestion_date') else None
                )
                books.append(book_metadata)
                
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_file}: {e}")
                continue
        
        # Apply pagination
        total = len(books)
        paginated_books = books[skip:skip + limit]
        
        logger.info(f"Found {total} books, returning {len(paginated_books)}")
        
        return BookListResponse(
            books=paginated_books,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Failed to list books: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve books")


@router.get("/books/{book_id}", response_model=BookMetadata)
async def get_book(book_id: str):
    """
    Get detailed information about a specific book.
    
    Args:
        book_id: Unique book identifier
        
    Returns:
        BookMetadata: Book metadata
    """
    logger.info(f"Getting book details: {book_id}")
    
    try:
        # Load metadata from file
        settings = get_settings()
        metadata_file = Path(settings.user_data_path) / f"{book_id}_metadata.json"
        
        if not metadata_file.exists():
            raise HTTPException(status_code=404, detail="Book not found")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        # Get current ingestion status
        ingestion_service = get_ingestion_service()
        progress = ingestion_service.get_ingestion_status(book_id)
        status = progress.status.value if progress else "unknown"
        
        return BookMetadata(
            id=data['book_id'],
            title=data['title'],
            author=data.get('author'),
            file_type=data['file_type'],
            file_size=data['file_size'],
            upload_date=datetime.fromisoformat(data['upload_date']),
            ingestion_status=status,
            user_id=data.get('user_id'),
            text_length=data.get('text_length', 0),
            chunk_count=data.get('chunk_count', 0),
            ingestion_date=datetime.fromisoformat(data['ingestion_date']) if data.get('ingestion_date') else None
        )
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Book not found")
    except Exception as e:
        logger.error(f"Failed to get book {book_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve book details")


@router.get("/books/{book_id}/status", response_model=IngestionStatus)
async def get_ingestion_status(book_id: str):
    """
    Get the ingestion status of a specific book.
    
    Args:
        book_id: Unique book identifier
        
    Returns:
        IngestionStatus: Current ingestion status
    """
    logger.info(f"Getting ingestion status: {book_id}")
    
    try:
        ingestion_service = get_ingestion_service()
        progress = ingestion_service.get_ingestion_status(book_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Ingestion status not found")
        
        progress_dict = progress.to_dict()
        
        return IngestionStatus(
            book_id=book_id,
            status=progress_dict['status'],
            progress=progress_dict['progress'],
            message=progress_dict['message'],
            started_at=datetime.fromisoformat(progress_dict['started_at']) if progress_dict['started_at'] else None,
            completed_at=datetime.fromisoformat(progress_dict['completed_at']) if progress_dict['completed_at'] else None,
            error=progress_dict['error'],
            metrics=progress_dict['metrics']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ingestion status for {book_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve ingestion status")


@router.delete("/books/{book_id}")
async def delete_book(book_id: str):
    """
    Delete a book and its associated data.
    
    Args:
        book_id: Unique book identifier
        
    Returns:
        Dict: Deletion confirmation
    """
    logger.info(f"Deleting book: {book_id}")
    
    try:
        ingestion_service = get_ingestion_service()
        success = await ingestion_service.delete_book(book_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Book not found")
        
        return {
            "message": "Book deleted successfully",
            "book_id": book_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete book {book_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete book")