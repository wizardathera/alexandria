"""
Alexandria - Main FastAPI Application

This is the entry point for the Alexandria application, providing REST API endpoints
for book ingestion, RAG queries, and MCP tool integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any
import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import get_settings
from src.utils.logger import setup_logger
from src.api.health import router as health_router
from src.api.books import router as books_router
from src.api.chat import router as chat_router
from src.api.content import router as content_router
from src.api.enhanced_content import router as enhanced_content_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger = setup_logger()
    logger.info("Starting Alexandria Application...")
    
    # Initialize services here
    try:
        # Vector database initialization will be added here
        logger.info("Services initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down Alexandria Application...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Alexandria API",
        description="AI-powered platform for interactive book reading and learning",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add root route
    @app.get("/")
    async def root():
        """
        Root endpoint providing API information and links.
        
        Returns:
            Dict: API information and available endpoints
        """
        return {
            "message": "Alexandria API",
            "version": "1.0.0",
            "status": "healthy",
            "documentation": "/docs",
            "endpoints": {
                "health": "/api/v1/health",
                "books": "/api/v1/books",
                "chat": "/api/v1/chat",
                "content": "/api/v1/content",
                "enhanced": "/api/enhanced"
            },
            "description": "AI-powered companion for intelligent book interaction and analysis"
        }
    
    # Include routers
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(books_router, prefix="/api/v1", tags=["books"])
    app.include_router(chat_router, prefix="/api/v1", tags=["chat"])
    app.include_router(content_router)
    app.include_router(enhanced_content_router)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """
        Global exception handler for unhandled errors.
        
        Args:
            request: HTTP request
            exc: Exception that occurred
            
        Returns:
            JSONResponse: Error response
        """
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    """
    Run the application directly with uvicorn.
    For development use only.
    """
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )