"""
Alexandria - Main FastAPI Application Entry Point

This is the entry point for the Alexandria application, providing REST API endpoints
for book ingestion, RAG queries, and MCP tool integration.
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.app_factory import create_app
from src.utils.config import get_settings


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