"""
Health check API endpoints for the DBC application.

This module provides endpoints to check the health and status of the application
and its dependencies.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
from datetime import datetime

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, str]


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""
    status: str
    timestamp: datetime
    version: str
    environment: str
    services: Dict[str, Dict[str, Any]]
    configuration: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Basic health status information
    """
    settings = get_settings()
    
    # Check basic service availability
    services_status = {
        "api": "healthy",
        "vector_db": "healthy",  # Will be updated when vector DB is integrated
        "openai": "unknown"      # Will be updated when OpenAI integration is added
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment=settings.environment,
        services=services_status
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check endpoint with service diagnostics.
    
    Returns:
        DetailedHealthResponse: Detailed health status and configuration
    """
    settings = get_settings()
    
    # Detailed service checks
    services_status = {
        "api": {
            "status": "healthy",
            "details": "FastAPI server running",
            "last_check": datetime.utcnow().isoformat()
        },
        "vector_db": {
            "status": "pending",
            "details": f"Vector DB type: {settings.vector_db_type}",
            "last_check": datetime.utcnow().isoformat()
        },
        "openai": {
            "status": "unknown",
            "details": "API key configured" if settings.openai_api_key else "No API key",
            "last_check": datetime.utcnow().isoformat()
        }
    }
    
    # Configuration summary (without sensitive data)
    configuration = {
        "vector_db_type": settings.vector_db_type,
        "max_upload_size_mb": settings.max_upload_size_mb,
        "supported_formats": settings.get_supported_formats_list(),
        "auth_enabled": settings.auth_enabled,
        "debug": settings.debug
    }
    
    return DetailedHealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment=settings.environment,
        services=services_status,
        configuration=configuration
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check endpoint for deployment health checks.
    
    Returns:
        Dict: Readiness status
    """
    # This will be expanded to check all required services
    # For now, just check if the application can respond
    
    try:
        settings = get_settings()
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Application is ready to serve requests"
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Application is not ready"
        )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check endpoint for deployment health checks.
    
    Returns:
        Dict: Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Application is alive"
    }