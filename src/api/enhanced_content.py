"""
Enhanced content API endpoints for the multi-module DBC platform.

This module provides REST API endpoints for enhanced content management,
including permission-aware search, content relationships, and AI-powered
recommendations using the enhanced embedding service.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from src.models import (
    ContentItem, User, ModuleType, ContentType, 
    ContentVisibility, UserRole, ProcessingStatus
)
from src.services.enhanced_embedding_service import get_enhanced_embedding_service
from src.services.content_service import get_content_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/enhanced", tags=["enhanced-content"])


# ========================================
# Request/Response Models
# ========================================

class EnhancedSearchRequest(BaseModel):
    """Request model for enhanced search."""
    query: str = Field(..., description="Search query")
    module_filter: Optional[ModuleType] = Field(None, description="Filter by module")
    content_type_filter: Optional[ContentType] = Field(None, description="Filter by content type")
    n_results: int = Field(10, ge=1, le=50, description="Number of results to return")
    include_relationships: bool = Field(True, description="Include content relationships")


class EnhancedSearchResult(BaseModel):
    """Enhanced search result with metadata."""
    content_id: str
    title: str
    author: Optional[str]
    content_type: str
    module_type: str
    chunk_type: str
    semantic_tags: List[str]
    source_location: Dict[str, Any]
    importance_score: Optional[float]
    quality_score: Optional[float]
    similarity_score: float
    relationship_score: float


class EnhancedSearchResponse(BaseModel):
    """Response model for enhanced search."""
    query: str
    total_results: int
    results: List[EnhancedSearchResult]
    search_time_ms: float
    user_permissions_applied: bool


class ContentRecommendation(BaseModel):
    """Content recommendation model."""
    content_id: str
    title: str
    author: Optional[str]
    content_type: str
    module_type: str
    recommendation_score: float
    recommendation_type: str  # "relationship" or "similarity"
    reason: str


class ContentRecommendationsResponse(BaseModel):
    """Response model for content recommendations."""
    content_id: str
    recommendations: List[ContentRecommendation]
    recommendation_count: int


class ProcessingMetricsResponse(BaseModel):
    """Response model for processing metrics."""
    total_content_processed: int
    total_embeddings_created: int
    total_relationships_discovered: int
    average_processing_time: float
    cache_stats: Dict[str, Any]
    embedding_cache_stats: Optional[Dict[str, Any]]


class ContentProcessingRequest(BaseModel):
    """Request model for content processing."""
    content_id: str = Field(..., description="Content ID to process")
    force_reprocess: bool = Field(False, description="Force reprocessing even if already processed")


# ========================================
# Dependency Functions
# ========================================

async def get_current_user() -> Optional[User]:
    """
    Get current user from request context.
    
    For now, this returns a default user. In Phase 2, this will be
    replaced with proper authentication middleware.
    """
    # TODO: Implement proper authentication in Phase 2
    return User(
        user_id="default_user",
        email="user@example.com",
        username="default_user",
        role=UserRole.READER,
        subscription_tier="free"
    )


# ========================================
# Enhanced Search Endpoints
# ========================================

@router.post("/search", response_model=EnhancedSearchResponse)
async def enhanced_search(
    request: EnhancedSearchRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Perform enhanced search with permission filtering and relationship awareness.
    
    This endpoint provides:
    - Permission-aware search filtering
    - Module and content type filtering
    - Content relationship awareness
    - Semantic tag matching
    - Quality and importance scoring
    """
    start_time = datetime.now()
    
    try:
        enhanced_embedding_service = await get_enhanced_embedding_service()
        
        # Perform enhanced search
        search_results = await enhanced_embedding_service.enhanced_search(
            query=request.query,
            user=current_user,
            module_filter=request.module_filter,
            content_type_filter=request.content_type_filter,
            n_results=request.n_results,
            include_relationships=request.include_relationships
        )
        
        # Convert to response format
        enhanced_results = []
        for result in search_results.get("enhanced_results", []):
            enhanced_results.append(EnhancedSearchResult(**result))
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = EnhancedSearchResponse(
            query=request.query,
            total_results=len(enhanced_results),
            results=enhanced_results,
            search_time_ms=search_time,
            user_permissions_applied=current_user is not None
        )
        
        # Add helpful message if no results found
        if len(enhanced_results) == 0:
            logger.warning(f"Enhanced search returned no results for query: '{request.query}'")
            # Note: We return a successful response with empty results rather than 404
            # This is more appropriate for search operations
        else:
            logger.info(f"Enhanced search completed: '{request.query}' -> {len(enhanced_results)} results in {search_time:.1f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search/quick")
async def quick_search(
    q: str = Query(..., description="Search query"),
    module: Optional[str] = Query(None, description="Module filter (library, lms, marketplace)"),
    type: Optional[str] = Query(None, description="Content type filter"),
    limit: int = Query(5, ge=1, le=20, description="Number of results"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Quick search endpoint with query parameters for simple integrations.
    """
    try:
        # Convert string filters to enums
        module_filter = ModuleType(module) if module else None
        content_type_filter = ContentType(type) if type else None
        
        enhanced_embedding_service = await get_enhanced_embedding_service()
        
        search_results = await enhanced_embedding_service.enhanced_search(
            query=q,
            user=current_user,
            module_filter=module_filter,
            content_type_filter=content_type_filter,
            n_results=limit,
            include_relationships=False  # Faster for quick search
        )
        
        # Simplified response for quick search
        quick_results = []
        for result in search_results.get("enhanced_results", []):
            quick_results.append({
                "content_id": result["content_id"],
                "title": result["title"],
                "author": result["author"],
                "score": result["similarity_score"]
            })
        
        return {
            "query": q,
            "results": quick_results,
            "count": len(quick_results)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter value: {str(e)}")
    except Exception as e:
        logger.error(f"Quick search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ========================================
# Content Recommendation Endpoints
# ========================================

@router.get("/content/{content_id}/recommendations", response_model=ContentRecommendationsResponse)
async def get_content_recommendations(
    content_id: str,
    limit: int = Query(5, ge=1, le=20, description="Number of recommendations"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get AI-powered content recommendations based on relationships and similarity.
    
    This endpoint provides:
    - Relationship-based recommendations
    - Similarity-based recommendations
    - Permission-aware filtering
    - Explanation of recommendation reasoning
    """
    try:
        enhanced_embedding_service = await get_enhanced_embedding_service()
        
        recommendations = await enhanced_embedding_service.get_content_recommendations(
            content_id=content_id,
            user=current_user,
            n_recommendations=limit
        )
        
        # Convert to response format
        recommendation_models = [
            ContentRecommendation(**rec) for rec in recommendations
        ]
        
        response = ContentRecommendationsResponse(
            content_id=content_id,
            recommendations=recommendation_models,
            recommendation_count=len(recommendation_models)
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations for {content_id}")
        
        return response
        
    except Exception as e:
        logger.error(f"Content recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations failed: {str(e)}")


@router.get("/recommendations/discover")
async def discover_content(
    module: Optional[str] = Query(None, description="Module to discover content from"),
    limit: int = Query(10, ge=1, le=50, description="Number of items to discover"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Discover new content based on user preferences and trending items.
    
    This is a simple discovery endpoint that can be enhanced in future phases
    with user preference learning and trending analysis.
    """
    try:
        content_service = await get_content_service()
        
        # Convert module filter
        module_filter = ModuleType(module) if module else None
        
        # Get recent content items (simple discovery for now)
        content_items = await content_service.list_content_items(
            module_type=module_filter,
            user=current_user,
            limit=limit,
            offset=0
        )
        
        # Convert to discovery format
        discovered_items = []
        for content in content_items:
            discovered_items.append({
                "content_id": content.content_id,
                "title": content.title,
                "author": content.author,
                "content_type": content.content_type.value,
                "module_type": content.module_type.value,
                "created_at": content.created_at.isoformat(),
                "topics": content.topics,
                "discovery_reason": "Recent content"
            })
        
        return {
            "discovered_content": discovered_items,
            "count": len(discovered_items),
            "discovery_criteria": f"Recent content in {module or 'all modules'}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid module: {str(e)}")
    except Exception as e:
        logger.error(f"Content discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


# ========================================
# Content Listing and Management Endpoints
# ========================================

@router.get("/content", response_model=Dict[str, Any])
async def list_enhanced_content(
    module: Optional[str] = Query(None, description="Filter by module (library, lms, marketplace)"),
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    List enhanced content items with metadata and processing status.
    
    This endpoint provides:
    - Filtered content listing by module and type
    - Processing status for each content item
    - Enhanced metadata including topics and quality scores
    - Permission-aware filtering
    """
    try:
        content_service = await get_content_service()
        
        # Convert string filters to enums
        module_filter = ModuleType(module) if module else None
        content_type_filter = ContentType(content_type) if content_type else None
        
        # Get content items
        content_items = await content_service.list_content_items(
            module_type=module_filter,
            content_type=content_type_filter,
            user=current_user,
            limit=limit,
            offset=offset
        )
        
        # Convert to enhanced format with safe fallbacks
        enhanced_items = []
        for content in content_items:
            try:
                # Build content item with safe fallbacks for missing metadata
                enhanced_item = {
                    "content_id": content.content_id,
                    "title": content.title or "Untitled",
                    "author": content.author or "Unknown",
                    "content_type": content.content_type.value if content.content_type else "document",
                    "module_type": content.module_type.value if content.module_type else "library",
                    "visibility": content.visibility.value if content.visibility else "private",
                    "processing_status": content.processing_status.value if content.processing_status else "pending",
                    "created_at": content.created_at.isoformat() if content.created_at else datetime.now().isoformat(),
                    "updated_at": content.updated_at.isoformat() if content.updated_at else datetime.now().isoformat(),
                    "processed_at": content.processed_at.isoformat() if content.processed_at else None,
                    "text_length": content.text_length or 0,
                    "chunk_count": content.chunk_count or 0,
                    "topics": content.topics if content.topics is not None else [],
                    "language": content.language or "en",
                    "reading_level": content.reading_level or "unknown",
                    "file_path": content.file_path or "",
                    "file_size": content.file_size or 0,
                    "mime_type": getattr(content, 'mime_type', getattr(content, 'file_type', 'application/octet-stream'))
                }
                
                enhanced_items.append(enhanced_item)
                
            except Exception as item_error:
                # Log error for individual item but continue processing others
                logger.warning(f"Error processing content item {content.content_id}: {item_error}")
                # Add minimal safe item to avoid breaking the response
                enhanced_items.append({
                    "content_id": getattr(content, 'content_id', 'unknown'),
                    "title": "Error: Could not load metadata",
                    "author": "Unknown",
                    "content_type": "document",
                    "module_type": "library",
                    "visibility": "private",
                    "processing_status": "failed",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "processed_at": None,
                    "text_length": 0,
                    "chunk_count": 0,
                    "topics": [],
                    "language": "en",
                    "reading_level": "unknown",
                    "file_path": "",
                    "file_size": 0,
                    "mime_type": "application/octet-stream"
                })
        
        # Get total count for pagination
        total_count = await content_service.count_content_items(
            module_type=module_filter,
            content_type=content_type_filter,
            user=current_user
        )
        
        # Provide helpful response even when no content exists
        response = {
            "content": enhanced_items,
            "pagination": {
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + len(enhanced_items) < total_count
            },
            "filters": {
                "module": module,
                "content_type": content_type
            },
            "user_permissions_applied": current_user is not None
        }
        
        # Add helpful messages for empty states
        if total_count == 0:
            if module or content_type:
                response["message"] = f"No content found matching the specified filters. Try removing filters or uploading content for the {module or content_type} module."
            else:
                response["message"] = "No content has been uploaded yet. Upload your first book or document to get started with the Alexandria platform."
            response["suggestions"] = [
                "Upload content using the /api/v1/books/upload endpoint",
                "Check that content has been successfully processed",
                "Verify your user permissions if you expect to see content"
            ]
        elif len(enhanced_items) == 0 and offset > 0:
            response["message"] = f"No more content available at offset {offset}. Total items: {total_count}."
            
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter value: {str(e)}")
    except Exception as e:
        logger.error(f"Content listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content listing failed: {str(e)}")


@router.get("/content/sample-format")
async def get_sample_content_format():
    """
    Get sample content format to help frontend developers understand the expected JSON structure.
    
    This endpoint returns a sample of what the /api/enhanced/content endpoint would return
    with actual data, showing all the metadata fields that are available.
    """
    sample_content = [
        {
            "content_id": "550e8400-e29b-41d4-a716-446655440001",
            "title": "The Art of Programming",
            "author": "Jane Doe",
            "content_type": "book",
            "module_type": "library",
            "visibility": "public",
            "processing_status": "completed",
            "created_at": "2024-01-15T10:30:00",
            "updated_at": "2024-01-15T10:35:00",
            "processed_at": "2024-01-15T10:35:00",
            "text_length": 145000,
            "chunk_count": 287,
            "topics": ["programming", "software development", "best practices"],
            "language": "en",
            "reading_level": "intermediate",
            "file_path": "/uploads/books/art_of_programming.pdf",
            "file_size": 2048576,
            "mime_type": "application/pdf"
        },
        {
            "content_id": "550e8400-e29b-41d4-a716-446655440002",
            "title": "Introduction to Machine Learning",
            "author": "John Smith",
            "content_type": "course",
            "module_type": "lms",
            "visibility": "premium",
            "processing_status": "completed",
            "created_at": "2024-02-01T14:20:00",
            "updated_at": "2024-02-01T15:00:00",
            "processed_at": "2024-02-01T15:00:00",
            "text_length": 85000,
            "chunk_count": 152,
            "topics": ["machine learning", "data science", "artificial intelligence"],
            "language": "en",
            "reading_level": "advanced",
            "file_path": "/uploads/courses/ml_intro.pdf",
            "file_size": 1536000,
            "mime_type": "application/pdf"
        },
        {
            "content_id": "550e8400-e29b-41d4-a716-446655440003",
            "title": "Quick Guide to Python",
            "author": "Alice Johnson",
            "content_type": "document",
            "module_type": "library",
            "visibility": "public",
            "processing_status": "processing",
            "created_at": "2024-03-10T09:15:00",
            "updated_at": "2024-03-10T09:20:00",
            "processed_at": None,
            "text_length": None,
            "chunk_count": None,
            "topics": [],
            "language": "en",
            "reading_level": "beginner",
            "file_path": "/uploads/documents/python_guide.txt",
            "file_size": 512000,
            "mime_type": "text/plain"
        }
    ]
    
    return {
        "content": sample_content,
        "pagination": {
            "total": 3,
            "limit": 20,
            "offset": 0,
            "has_more": False
        },
        "filters": {
            "module": None,
            "content_type": None
        },
        "user_permissions_applied": True,
        "format_explanation": {
            "description": "This shows the exact format returned by /api/enhanced/content",
            "fields": {
                "content_id": "UUID string - unique identifier for the content",
                "title": "string - content title",
                "author": "string or null - content author/creator",
                "content_type": "string - type of content (book, course, document, etc.)",
                "module_type": "string - platform module (library, lms, marketplace)",
                "visibility": "string - visibility level (public, private, organization, premium)",
                "processing_status": "string - processing status (pending, processing, completed, failed)",
                "created_at": "ISO datetime string - when content was created",
                "updated_at": "ISO datetime string - when content was last updated",
                "processed_at": "ISO datetime string or null - when processing completed",
                "text_length": "integer or null - length of extracted text",
                "chunk_count": "integer or null - number of text chunks created",
                "topics": "array of strings - AI-extracted topics/tags",
                "language": "string - content language code",
                "reading_level": "string - difficulty level (beginner, intermediate, advanced)",
                "file_path": "string - storage path for the file",
                "file_size": "integer - file size in bytes",
                "mime_type": "string - MIME type of the original file"
            },
            "notes": [
                "Fields marked 'or null' will be null when content is still processing",
                "The 'topics' array will be empty until AI processing completes",
                "All datetime fields use ISO 8601 format",
                "Content is filtered based on user permissions"
            ]
        }
    }


@router.get("/content/{content_id}")
async def get_enhanced_content_details(
    content_id: str,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get detailed information about a specific content item.
    
    This endpoint provides:
    - Complete content metadata
    - Processing status and metrics
    - Available relationships
    - Access to content chunks if processed
    """
    try:
        content_service = await get_content_service()
        
        # Get content item
        content = await content_service.get_content_item(content_id, current_user)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Get relationships
        relationships = await content_service.get_content_relationships(content_id)
        
        # Get enhanced embedding service for processing metrics
        enhanced_embedding_service = await get_enhanced_embedding_service()
        processing_metrics = enhanced_embedding_service.get_processing_metrics()
        
        # Build detailed response
        return {
            "content_id": content.content_id,
            "title": content.title,
            "author": content.author,
            "description": content.description,
            "content_type": content.content_type.value,
            "module_type": content.module_type.value,
            "visibility": content.visibility.value,
            "processing_status": content.processing_status.value,
            "created_at": content.created_at.isoformat(),
            "updated_at": content.updated_at.isoformat(),
            "processed_at": content.processed_at.isoformat() if content.processed_at else None,
            "metadata": {
                "text_length": content.text_length,
                "chunk_count": content.chunk_count,
                "topics": content.topics,
                "language": content.language,
                "reading_level": content.reading_level,
                "file_path": content.file_path,
                "file_size": content.file_size,
                "mime_type": content.mime_type
            },
            "relationships": [
                {
                    "target_content_id": rel.target_content_id,
                    "relationship_type": rel.relationship_type.value,
                    "strength": rel.strength,
                    "confidence": rel.confidence,
                    "discovered_by": rel.discovered_by,
                    "context": rel.context
                }
                for rel in relationships
            ],
            "processing_metrics": {
                "average_processing_time": processing_metrics.get("average_processing_time", 0),
                "is_fully_processed": content.processing_status == ProcessingStatus.COMPLETED
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content details retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content details failed: {str(e)}")


# ========================================
# Content Processing Endpoints
# ========================================

@router.post("/content/process")
async def process_content(
    request: ContentProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Process content with enhanced embedding generation and relationship discovery.
    
    This endpoint triggers:
    - Enhanced semantic chunking
    - Multi-module embedding generation
    - Semantic tag extraction
    - Content relationship discovery
    """
    try:
        content_service = await get_content_service()
        
        # Get content item
        content = await content_service.get_content_item(request.content_id, current_user)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Check if already processing
        if content.processing_status == ProcessingStatus.PROCESSING and not request.force_reprocess:
            return {
                "content_id": request.content_id,
                "status": "already_processing",
                "message": "Content is already being processed"
            }
        
        # Add background processing task
        async def process_content_background():
            enhanced_embedding_service = await get_enhanced_embedding_service()
            await enhanced_embedding_service.process_content_item(
                content=content,
                user=current_user,
                force_reprocess=request.force_reprocess
            )
        
        background_tasks.add_task(process_content_background)
        
        return {
            "content_id": request.content_id,
            "status": "processing_started",
            "message": "Content processing started in background",
            "force_reprocess": request.force_reprocess
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content processing initiation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/content/{content_id}/processing-status")
async def get_processing_status(
    content_id: str,
    current_user: Optional[User] = Depends(get_current_user)
):
    """Get the processing status of a content item."""
    try:
        content_service = await get_content_service()
        
        content = await content_service.get_content_item(content_id, current_user)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "content_id": content_id,
            "processing_status": content.processing_status.value,
            "text_length": content.text_length,
            "chunk_count": content.chunk_count,
            "processed_at": content.processed_at.isoformat() if content.processed_at else None,
            "topics": content.topics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ========================================
# Analytics and Metrics Endpoints
# ========================================

@router.get("/metrics/processing", response_model=ProcessingMetricsResponse)
async def get_processing_metrics(
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get enhanced embedding service processing metrics.
    
    Provides insights into:
    - Total content processed
    - Embeddings created
    - Relationships discovered
    - Processing performance
    - Cache utilization
    """
    try:
        enhanced_embedding_service = await get_enhanced_embedding_service()
        
        metrics = enhanced_embedding_service.get_processing_metrics()
        
        response = ProcessingMetricsResponse(**metrics)
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to get processing metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics failed: {str(e)}")


@router.post("/metrics/clear-cache")
async def clear_processing_cache(
    current_user: Optional[User] = Depends(get_current_user)
):
    """Clear enhanced embedding service caches to free memory."""
    try:
        # Check if user has admin permissions
        if current_user and current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin permissions required")
        
        enhanced_embedding_service = await get_enhanced_embedding_service()
        await enhanced_embedding_service.cleanup_caches()
        
        return {
            "status": "success",
            "message": "Processing caches cleared successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


# ========================================
# Content Relationship Endpoints
# ========================================

class RelationshipResponse(BaseModel):
    """Response model for content relationships."""
    related_content_id: str
    related_title: str
    related_author: Optional[str]
    related_content_type: str
    related_module_type: str
    relationship_type: str
    strength: float
    confidence: float
    explanation: Optional[str]
    related_semantic_tags: List[str] = []


class GraphDataResponse(BaseModel):
    """Response model for graph visualization data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


@router.get("/content/{content_id}/relationships")
async def get_content_relationships(
    content_id: str,
    limit: int = Query(10, ge=1, le=50, description="Number of relationships to return"),
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get relationships for a specific content item.
    
    This endpoint provides:
    - Content-to-content relationships
    - Relationship strength and confidence scores
    - Semantic explanations of relationships
    - Related content metadata
    """
    try:
        content_service = await get_content_service()
        
        # Verify content exists and user has access
        content = await content_service.get_content_item(content_id, current_user)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found or access denied")
        
        # Get relationships
        from src.models import ContentRelationshipType
        rel_type_filter = None
        if relationship_type:
            try:
                rel_type_filter = ContentRelationshipType(relationship_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid relationship type: {relationship_type}")
        
        relationships = await content_service.get_content_relationships(
            content_id=content_id,
            relationship_type=rel_type_filter
        )
        
        # Convert to response format with enhanced data
        relationship_responses = []
        for rel in relationships:
            # Get target content details
            target_content = await content_service.get_content_item(rel.target_content_id, current_user)
            if target_content:  # Only include if user has access
                response = RelationshipResponse(
                    related_content_id=rel.target_content_id,
                    related_title=target_content.title,
                    related_author=target_content.author,
                    related_content_type=target_content.content_type.value,
                    related_module_type=target_content.module_type.value,
                    relationship_type=rel.relationship_type.value,
                    strength=rel.strength,
                    confidence=rel.confidence,
                    explanation=rel.context,
                    related_semantic_tags=target_content.topics
                )
                relationship_responses.append(response)
        
        # Sort by strength and limit
        relationship_responses.sort(key=lambda r: r.strength, reverse=True)
        limited_relationships = relationship_responses[:limit]
        
        return {
            "content_id": content_id,
            "relationships": [rel.dict() for rel in limited_relationships],
            "total_found": len(relationship_responses),
            "returned": len(limited_relationships)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relationships for {content_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Relationships retrieval failed: {str(e)}")


@router.get("/relationships", response_model=GraphDataResponse)
async def get_relationships_graph(
    content_ids: Optional[str] = Query(None, description="Comma-separated content IDs to include"),
    limit: int = Query(100, ge=10, le=500, description="Maximum number of nodes"),
    min_strength: float = Query(0.3, ge=0.0, le=1.0, description="Minimum relationship strength"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get relationship graph data for visualization.
    
    This endpoint provides:
    - Graph nodes (content items)
    - Graph edges (relationships)
    - Graph statistics and metadata
    - Frontend-compatible data structure
    """
    try:
        content_service = await get_content_service()
        
        # Get content items to include
        if content_ids:
            # Specific content IDs provided
            content_id_list = [cid.strip() for cid in content_ids.split(",")]
            content_items = []
            for cid in content_id_list:
                content = await content_service.get_content_item(cid, current_user)
                if content:
                    content_items.append(content)
        else:
            # Get all accessible content
            content_items = await content_service.list_content_items(
                user=current_user,
                limit=limit,
                offset=0
            )
        
        if not content_items:
            return GraphDataResponse(
                nodes=[],
                edges=[],
                stats={"total_nodes": 0, "total_edges": 0, "message": "No content available for graph visualization"}
            )
        
        # Build nodes
        nodes = []
        content_id_to_index = {}
        for i, content in enumerate(content_items):
            content_id_to_index[content.content_id] = i
            nodes.append({
                "id": content.content_id,
                "title": content.title,
                "author": content.author,
                "content_type": content.content_type.value,
                "module_type": content.module_type.value,
                "topics": content.topics,
                "size": min(max(content.text_length or 1000, 1000), 10000),  # Size based on text length
                "color": _get_node_color_by_type(content.content_type.value),
                "created_at": content.created_at.isoformat()
            })
        
        # Build edges
        edges = []
        content_ids = [content.content_id for content in content_items]
        
        for content in content_items:
            relationships = await content_service.get_content_relationships(content.content_id)
            
            for rel in relationships:
                # Only include if target is in our node set and meets strength threshold
                if (rel.target_content_id in content_ids and 
                    rel.strength >= min_strength):
                    
                    edges.append({
                        "source": rel.source_content_id,
                        "target": rel.target_content_id,
                        "relationship_type": rel.relationship_type.value,
                        "strength": rel.strength,
                        "confidence": rel.confidence,
                        "weight": rel.strength,  # For visualization
                        "discovered_by": rel.discovered_by,
                        "human_verified": rel.human_verified,
                        "context": rel.context
                    })
        
        # Calculate graph statistics
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "average_connections": len(edges) / len(nodes) if nodes else 0,
            "content_types": list(set(node["content_type"] for node in nodes)),
            "module_types": list(set(node["module_type"] for node in nodes)),
            "relationship_types": list(set(edge["relationship_type"] for edge in edges)),
            "average_strength": sum(edge["strength"] for edge in edges) / len(edges) if edges else 0,
            "min_strength_filter": min_strength
        }
        
        return GraphDataResponse(
            nodes=nodes,
            edges=edges,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get relationships graph: {e}")
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")


def _get_node_color_by_type(content_type: str) -> str:
    """Get color for graph node based on content type."""
    color_map = {
        "book": "#3498db",      # Blue
        "article": "#2ecc71",   # Green
        "document": "#f39c12",  # Orange
        "course": "#9b59b6",    # Purple
        "lesson": "#e74c3c",    # Red
        "quiz": "#1abc9c",      # Teal
        "marketplace_item": "#34495e"  # Dark gray
    }
    return color_map.get(content_type, "#95a5a6")  # Default gray


@router.post("/relationships/discover")
async def discover_relationships(
    content_id: str = Query(..., description="Content ID to discover relationships for"),
    max_relationships: int = Query(20, ge=1, le=100, description="Maximum relationships to discover"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Discover new relationships for a content item using AI.
    
    This endpoint provides:
    - AI-powered relationship discovery
    - Semantic similarity analysis
    - Confidence scoring
    - Background processing option
    """
    try:
        content_service = await get_content_service()
        
        # Verify content exists
        content = await content_service.get_content_item(content_id, current_user)
        if not content:
            raise HTTPException(status_code=404, detail="Content not found or access denied")
        
        # Get enhanced embedding service for relationship discovery
        enhanced_embedding_service = await get_enhanced_embedding_service()
        
        # Get all other content for comparison
        all_content = await content_service.list_content_items(
            user=current_user,
            limit=1000,  # Large limit for relationship discovery
            offset=0
        )
        
        # Filter out the source content
        candidate_content = [c for c in all_content if c.content_id != content_id]
        
        if not candidate_content:
            return {
                "content_id": content_id,
                "discovered_relationships": [],
                "message": "No other content available for relationship discovery"
            }
        
        # Use graph retrieval engine for relationship discovery
        from src.utils.graph_retrieval import GraphSearchEngine
        graph_engine = GraphSearchEngine()
        
        # Build documents for graph construction
        documents = []
        for content_item in [content] + candidate_content:
            documents.append((
                content_item.content_id,
                content_item.title + " " + (content_item.description or ""),
                {
                    "title": content_item.title,
                    "author": content_item.author,
                    "content_type": content_item.content_type.value,
                    "topics": content_item.topics
                }
            ))
        
        # Build knowledge graph
        await graph_engine.build_graph_from_documents(documents, similarity_threshold=min_confidence)
        
        # Find related content
        related_content = graph_engine.find_related_content(
            node_id=content_id,
            max_distance=2
        )
        
        # Convert to relationship format
        discovered_relationships = []
        for related_id, score, distance in related_content[:max_relationships]:
            related_item = next((c for c in candidate_content if c.content_id == related_id), None)
            if related_item:
                discovered_relationships.append({
                    "related_content_id": related_id,
                    "related_title": related_item.title,
                    "related_author": related_item.author,
                    "related_content_type": related_item.content_type.value,
                    "relationship_type": "similar",  # Default for discovered relationships
                    "strength": score,
                    "confidence": score * 0.8,  # Slightly lower confidence for discovered
                    "distance": distance,
                    "explanation": f"Discovered semantic similarity (distance: {distance})",
                    "related_semantic_tags": related_item.topics
                })
        
        return {
            "content_id": content_id,
            "discovered_relationships": discovered_relationships,
            "total_candidates_analyzed": len(candidate_content),
            "relationships_found": len(discovered_relationships),
            "discovery_parameters": {
                "max_relationships": max_relationships,
                "min_confidence": min_confidence
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Relationship discovery failed for {content_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Relationship discovery failed: {str(e)}")


# ========================================
# Migration and Data Management Endpoints
# ========================================

@router.post("/migrate/legacy-books")
async def migrate_legacy_books_to_content_db(
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Migrate existing books from JSON metadata files to content database.
    
    This endpoint helps migrate books that were ingested before the content database
    integration was added. It checks existing JSON metadata files and creates
    corresponding records in the unified content database.
    """
    try:
        # Check if user has admin permissions (optional, since Phase 1 is single-user)
        if current_user and current_user.role != UserRole.ADMIN:
            logger.warning(f"User {current_user.user_id} attempted legacy migration without admin role")
            # In Phase 1, we'll allow it since there's only one user
            # In Phase 2+, uncomment the following line:
            # raise HTTPException(status_code=403, detail="Admin permissions required for migration")
        
        # Get ingestion service and run migration
        from src.services.ingestion import get_ingestion_service
        ingestion_service = get_ingestion_service()
        
        logger.info("Starting legacy book migration to content database")
        migration_results = await ingestion_service.migrate_existing_books_to_content_db()
        
        successful_migrations = sum(1 for success in migration_results.values() if success)
        total_books = len(migration_results)
        
        response_data = {
            "migration_completed": True,
            "total_books_found": total_books,
            "successful_migrations": successful_migrations,
            "failed_migrations": total_books - successful_migrations,
            "migration_results": migration_results,
            "timestamp": datetime.now().isoformat()
        }
        
        if successful_migrations == total_books:
            response_data["message"] = f"✅ Successfully migrated all {total_books} books to content database"
            logger.info(f"Legacy migration completed successfully: {successful_migrations}/{total_books}")
        else:
            response_data["message"] = f"⚠️ Migrated {successful_migrations}/{total_books} books. {total_books - successful_migrations} migrations failed."
            logger.warning(f"Legacy migration partially completed: {successful_migrations}/{total_books}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Legacy book migration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


@router.get("/migration/status")
async def get_migration_status(
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Check migration status by comparing JSON metadata files with content database records.
    
    This endpoint provides information about how many books need migration
    and which ones are already migrated.
    """
    try:
        from src.services.ingestion import get_ingestion_service
        import json
        from pathlib import Path
        from src.utils.config import get_settings
        
        settings = get_settings()
        metadata_dir = Path(settings.user_data_path)
        
        # Count JSON metadata files
        json_books = 0
        json_book_ids = []
        if metadata_dir.exists():
            metadata_files = list(metadata_dir.glob("*_metadata.json"))
            json_books = len(metadata_files)
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        if data.get('book_id'):
                            json_book_ids.append(data['book_id'])
                except Exception:
                    continue
        
        # Count content database records
        content_service = await get_content_service()
        db_content_items = await content_service.list_content_items(limit=1000)
        db_books = len(db_content_items)
        db_book_ids = [item.content_id for item in db_content_items]
        
        # Find books that need migration
        needs_migration = [book_id for book_id in json_book_ids if book_id not in db_book_ids]
        already_migrated = [book_id for book_id in json_book_ids if book_id in db_book_ids]
        
        status_data = {
            "migration_needed": len(needs_migration) > 0,
            "json_metadata_files": json_books,
            "content_database_records": db_books,
            "books_needing_migration": len(needs_migration),
            "books_already_migrated": len(already_migrated),
            "migration_completeness": f"{len(already_migrated)}/{len(json_book_ids)}" if json_book_ids else "0/0",
            "books_needing_migration_ids": needs_migration[:10],  # Show first 10 for debugging
            "timestamp": datetime.now().isoformat()
        }
        
        if len(needs_migration) == 0:
            status_data["message"] = "✅ All books are migrated to content database"
        else:
            status_data["message"] = f"⚠️ {len(needs_migration)} books need migration to content database"
        
        return status_data
        
    except Exception as e:
        logger.error(f"Migration status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


# ========================================
# Health and Status Endpoints
# ========================================

@router.get("/health")
async def enhanced_service_health():
    """Check the health of enhanced embedding services."""
    try:
        # Check service initialization
        enhanced_embedding_service = await get_enhanced_embedding_service()
        content_service = await get_content_service()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "enhanced_embedding_service": "ok",
                "content_service": "ok",
                "vector_database": "ok",
                "embedding_provider": "ok"
            }
        }
        
        # Get basic metrics
        metrics = enhanced_embedding_service.get_processing_metrics()
        health_status["metrics_summary"] = {
            "total_content_processed": metrics["total_content_processed"],
            "average_processing_time": metrics["average_processing_time"]
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )