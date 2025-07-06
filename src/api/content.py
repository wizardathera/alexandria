"""
Unified content management API endpoints.

This module provides REST API endpoints for managing content across all
DBC modules (Library, LMS, Marketplace) using the unified content schema.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from src.models import (
    ContentItem, ContentRelationship, User,
    ModuleType, ContentType, ContentVisibility, UserRole,
    ContentRelationshipType, ProcessingStatus
)
from src.services.content_service import get_content_service, ContentDatabaseError
from src.services.migration_service import get_migration_service, MigrationError
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/content", tags=["content"])


# ========================================
# Request/Response Models
# ========================================

class ContentItemResponse(ContentItem):
    """Response model for content items with additional metadata."""
    pass


class ContentItemCreateRequest(ContentItem):
    """Request model for creating content items."""
    pass


class ContentItemUpdateRequest(ContentItem):
    """Request model for updating content items."""
    pass


class ContentRelationshipResponse(ContentRelationship):
    """Response model for content relationships."""
    pass


class ContentRelationshipCreateRequest(ContentRelationship):
    """Request model for creating content relationships."""
    pass


class ContentListResponse(BaseModel):
    """Response model for content item lists."""
    items: List[ContentItem]
    total: int
    offset: int
    limit: int
    has_more: bool


class MigrationStatusResponse(BaseModel):
    """Response model for migration status."""
    status: str
    message: str
    details: Optional[dict] = None


# ========================================
# Dependency Functions
# ========================================

async def get_current_user() -> Optional[User]:
    """
    Get current user from authentication context.
    
    For Phase 1, this returns None (single-user mode).
    For Phase 2+, this will extract user from JWT token.
    """
    # TODO: Implement authentication in Phase 2
    # For now, return None for single-user mode
    return None


async def get_authenticated_user() -> User:
    """Get authenticated user (required for protected endpoints)."""
    user = await get_current_user()
    if user is None:
        # For Phase 1, create a default user
        return User(
            user_id="default-user",
            email="user@localhost",
            role=UserRole.ADMIN,  # Admin for Phase 1
            subscription_tier="pro"
        )
    return user


# ========================================
# Content Management Endpoints
# ========================================

@router.post("/items", response_model=ContentItemResponse)
async def create_content_item(
    content_request: ContentItemCreateRequest,
    current_user: User = Depends(get_authenticated_user)
):
    """
    Create a new content item.
    
    Args:
        content_request: Content item data
        current_user: Authenticated user
        
    Returns:
        Created content item
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        content_service = await get_content_service()
        
        # Set creator if not provided
        if not content_request.created_by:
            content_request.created_by = current_user.user_id
        
        # Create content item
        success = await content_service.create_content_item(content_request)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create content item")
        
        # Return created item
        created_item = await content_service.get_content_item(
            content_request.content_id, 
            current_user
        )
        
        if not created_item:
            raise HTTPException(status_code=500, detail="Content item created but not retrievable")
        
        logger.info(f"Created content item: {created_item.content_id} by user {current_user.user_id}")
        return ContentItemResponse(**created_item.dict())
        
    except ContentDatabaseError as e:
        logger.error(f"Database error creating content: {e}")
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating content item: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/items/{content_id}", response_model=ContentItemResponse)
async def get_content_item(
    content_id: str,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get a content item by ID.
    
    Args:
        content_id: Content item ID
        current_user: Optional authenticated user
        
    Returns:
        Content item if found and accessible
        
    Raises:
        HTTPException: If not found or access denied
    """
    try:
        content_service = await get_content_service()
        content_item = await content_service.get_content_item(content_id, current_user)
        
        if not content_item:
            raise HTTPException(status_code=404, detail="Content item not found or access denied")
        
        return ContentItemResponse(**content_item.dict())
        
    except Exception as e:
        logger.error(f"Error retrieving content item {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/items", response_model=ContentListResponse)
async def list_content_items(
    module_type: Optional[ModuleType] = Query(None, description="Filter by module type"),
    content_type: Optional[ContentType] = Query(None, description="Filter by content type"),
    visibility: Optional[ContentVisibility] = Query(None, description="Filter by visibility"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    search_query: Optional[str] = Query(None, description="Search in title/description"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    List content items with filtering and pagination.
    
    Args:
        module_type: Optional module filter
        content_type: Optional content type filter
        visibility: Optional visibility filter
        created_by: Optional creator filter
        search_query: Optional search query
        limit: Number of items to return
        offset: Number of items to skip
        current_user: Optional authenticated user
        
    Returns:
        Paginated list of content items
    """
    try:
        content_service = await get_content_service()
        
        # Get filtered content items
        content_items = await content_service.list_content_items(
            module_type=module_type,
            content_type=content_type,
            user=current_user,
            limit=limit,
            offset=offset
        )
        
        # Apply additional filters (simplified for now)
        filtered_items = content_items
        
        if visibility and current_user and current_user.role == UserRole.ADMIN:
            # Admin can filter by visibility
            filtered_items = [item for item in filtered_items if item.visibility == visibility]
        
        if created_by and current_user and current_user.role == UserRole.ADMIN:
            # Admin can filter by creator
            filtered_items = [item for item in filtered_items if item.created_by == created_by]
        
        if search_query:
            # Simple search in title and description
            search_lower = search_query.lower()
            filtered_items = [
                item for item in filtered_items
                if (search_lower in item.title.lower() or
                    (item.description and search_lower in item.description.lower()))
            ]
        
        # Apply pagination to filtered results
        paginated_items = filtered_items[offset:offset + limit]
        
        return ContentListResponse(
            items=paginated_items,
            total=len(filtered_items),
            offset=offset,
            limit=limit,
            has_more=offset + len(paginated_items) < len(filtered_items)
        )
        
    except Exception as e:
        logger.error(f"Error listing content items: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/items/{content_id}", response_model=ContentItemResponse)
async def update_content_item(
    content_id: str,
    content_update: ContentItemUpdateRequest,
    current_user: User = Depends(get_authenticated_user)
):
    """
    Update a content item.
    
    Args:
        content_id: Content item ID to update
        content_update: Updated content data
        current_user: Authenticated user
        
    Returns:
        Updated content item
        
    Raises:
        HTTPException: If update fails or access denied
    """
    try:
        content_service = await get_content_service()
        
        # Get existing content to check permissions
        existing_content = await content_service.get_content_item(content_id, current_user)
        if not existing_content:
            raise HTTPException(status_code=404, detail="Content item not found or access denied")
        
        # Check if user can modify this content
        if (current_user.role != UserRole.ADMIN and 
            existing_content.created_by != current_user.user_id):
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Ensure content_id matches
        content_update.content_id = content_id
        
        # Update content item
        success = await content_service.update_content_item(content_update)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update content item")
        
        # Return updated item
        updated_item = await content_service.get_content_item(content_id, current_user)
        
        logger.info(f"Updated content item: {content_id} by user {current_user.user_id}")
        return ContentItemResponse(**updated_item.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating content item {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/items/{content_id}")
async def delete_content_item(
    content_id: str,
    current_user: User = Depends(get_authenticated_user)
):
    """
    Delete a content item.
    
    Args:
        content_id: Content item ID to delete
        current_user: Authenticated user
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If deletion fails or access denied
    """
    try:
        content_service = await get_content_service()
        
        # Delete content item with permission checking
        success = await content_service.delete_content_item(content_id, current_user)
        
        if not success:
            raise HTTPException(status_code=404, detail="Content item not found or access denied")
        
        logger.info(f"Deleted content item: {content_id} by user {current_user.user_id}")
        return {"message": "Content item deleted successfully", "content_id": content_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting content item {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ========================================
# Content Relationship Endpoints
# ========================================

@router.post("/relationships", response_model=ContentRelationshipResponse)
async def create_content_relationship(
    relationship_request: ContentRelationshipCreateRequest,
    current_user: User = Depends(get_authenticated_user)
):
    """
    Create a content relationship.
    
    Args:
        relationship_request: Relationship data
        current_user: Authenticated user
        
    Returns:
        Created relationship
        
    Raises:
        HTTPException: If creation fails
    """
    try:
        content_service = await get_content_service()
        
        # Verify both content items exist and are accessible
        source_content = await content_service.get_content_item(
            relationship_request.source_content_id, current_user
        )
        target_content = await content_service.get_content_item(
            relationship_request.target_content_id, current_user
        )
        
        if not source_content or not target_content:
            raise HTTPException(status_code=404, detail="One or both content items not found")
        
        # Create relationship
        success = await content_service.create_relationship(relationship_request)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to create relationship")
        
        logger.info(f"Created relationship: {relationship_request.relationship_id}")
        return ContentRelationshipResponse(**relationship_request.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating content relationship: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/items/{content_id}/relationships", response_model=List[ContentRelationshipResponse])
async def get_content_relationships(
    content_id: str,
    relationship_type: Optional[ContentRelationshipType] = Query(None, description="Filter by relationship type"),
    direction: str = Query("both", pattern="^(outgoing|incoming|both)$", description="Relationship direction"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get relationships for a content item.
    
    Args:
        content_id: Content item ID
        relationship_type: Optional relationship type filter
        direction: Direction filter (outgoing, incoming, both)
        current_user: Optional authenticated user
        
    Returns:
        List of content relationships
        
    Raises:
        HTTPException: If content not found or access denied
    """
    try:
        content_service = await get_content_service()
        
        # Verify content exists and is accessible
        content_item = await content_service.get_content_item(content_id, current_user)
        if not content_item:
            raise HTTPException(status_code=404, detail="Content item not found or access denied")
        
        # Get relationships
        relationships = await content_service.get_content_relationships(
            content_id=content_id,
            relationship_type=relationship_type,
            direction=direction
        )
        
        return [ContentRelationshipResponse(**rel.dict()) for rel in relationships]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting relationships for {content_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ========================================
# Migration Endpoints
# ========================================

@router.get("/migration/status", response_model=MigrationStatusResponse)
async def get_migration_status(
    current_user: User = Depends(get_authenticated_user)
):
    """
    Get migration status and statistics.
    
    Args:
        current_user: Authenticated user (admin only)
        
    Returns:
        Migration status and statistics
        
    Raises:
        HTTPException: If access denied
    """
    try:
        # Only admin can access migration status
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        migration_service = await get_migration_service()
        status = await migration_service.get_migration_status()
        
        return MigrationStatusResponse(status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting migration status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/migration/scan")
async def scan_legacy_data(
    current_user: User = Depends(get_authenticated_user)
):
    """
    Scan for legacy data that needs migration.
    
    Args:
        current_user: Authenticated user (admin only)
        
    Returns:
        Scan results and statistics
        
    Raises:
        HTTPException: If access denied or scan fails
    """
    try:
        # Only admin can perform migration operations
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        migration_service = await get_migration_service()
        scan_results = await migration_service.scan_legacy_data()
        
        logger.info(f"Legacy data scan completed by user {current_user.user_id}")
        return scan_results
        
    except HTTPException:
        raise
    except MigrationError as e:
        logger.error(f"Migration scan error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error scanning legacy data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/migration/migrate")
async def migrate_legacy_data(
    dry_run: bool = Body(False, description="Perform validation without actual migration"),
    current_user: User = Depends(get_authenticated_user)
):
    """
    Migrate legacy data to unified content schema.
    
    Args:
        dry_run: If True, perform validation without actual migration
        current_user: Authenticated user (admin only)
        
    Returns:
        Migration results and statistics
        
    Raises:
        HTTPException: If access denied or migration fails
    """
    try:
        # Only admin can perform migration operations
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        migration_service = await get_migration_service()
        migration_results = await migration_service.migrate_all_legacy_books(dry_run=dry_run)
        
        action = "validated" if dry_run else "migrated"
        logger.info(f"Legacy data {action} by user {current_user.user_id}")
        
        return migration_results
        
    except HTTPException:
        raise
    except MigrationError as e:
        logger.error(f"Migration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error migrating legacy data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/migration/rollback")
async def rollback_migration(
    book_ids: Optional[List[str]] = Body(None, description="Specific book IDs to rollback (all if not provided)"),
    current_user: User = Depends(get_authenticated_user)
):
    """
    Rollback migration for specified books.
    
    Args:
        book_ids: Optional list of book IDs to rollback
        current_user: Authenticated user (admin only)
        
    Returns:
        Rollback results and statistics
        
    Raises:
        HTTPException: If access denied or rollback fails
    """
    try:
        # Only admin can perform migration operations
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        migration_service = await get_migration_service()
        rollback_results = await migration_service.rollback_migration(book_ids)
        
        logger.info(f"Migration rollback performed by user {current_user.user_id}")
        return rollback_results
        
    except HTTPException:
        raise
    except MigrationError as e:
        logger.error(f"Rollback error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error rolling back migration: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ========================================
# Content Statistics Endpoints
# ========================================

@router.get("/stats")
async def get_content_statistics(
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get content statistics across all modules.
    
    Args:
        current_user: Optional authenticated user
        
    Returns:
        Content statistics by module and type
    """
    try:
        content_service = await get_content_service()
        
        stats = {
            "total_content_items": 0,
            "by_module": {},
            "by_content_type": {},
            "by_status": {},
            "recent_activity": []
        }
        
        # Get all accessible content
        all_content = await content_service.list_content_items(
            user=current_user,
            limit=1000  # Large limit to get all content for stats
        )
        
        stats["total_content_items"] = len(all_content)
        
        # Calculate statistics
        for content in all_content:
            # By module
            module = content.module_type.value
            stats["by_module"][module] = stats["by_module"].get(module, 0) + 1
            
            # By content type
            content_type = content.content_type.value
            stats["by_content_type"][content_type] = stats["by_content_type"].get(content_type, 0) + 1
            
            # By status
            status = content.processing_status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        
        # Get recent activity (last 10 items by creation date)
        recent_content = sorted(all_content, key=lambda x: x.created_at, reverse=True)[:10]
        stats["recent_activity"] = [
            {
                "content_id": content.content_id,
                "title": content.title,
                "module_type": content.module_type.value,
                "content_type": content.content_type.value,
                "created_at": content.created_at.isoformat()
            }
            for content in recent_content
        ]
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting content statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")