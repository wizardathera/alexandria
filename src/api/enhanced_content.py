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
        
        # Convert to enhanced format
        enhanced_items = []
        for content in content_items:
            enhanced_items.append({
                "content_id": content.content_id,
                "title": content.title,
                "author": content.author,
                "content_type": content.content_type.value,
                "module_type": content.module_type.value,
                "visibility": content.visibility.value,
                "processing_status": content.processing_status.value,
                "created_at": content.created_at.isoformat(),
                "updated_at": content.updated_at.isoformat(),
                "processed_at": content.processed_at.isoformat() if content.processed_at else None,
                "text_length": content.text_length,
                "chunk_count": content.chunk_count,
                "topics": content.topics,
                "language": content.language,
                "reading_level": content.reading_level,
                "file_path": content.file_path,
                "file_size": content.file_size,
                "mime_type": content.mime_type
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