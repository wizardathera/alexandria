"""
MCP Server implementation for Alexandria application.

This module provides Model Context Protocol (MCP) server functionality
with tools for note-taking, resource fetching, and progress tracking.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.models import InitializationOptions
from mcp.types import (
    EmbeddedResource, 
    Resource, 
    Tool, 
    TextContent,
    JSONSchema,
    INVALID_PARAMS,
    INTERNAL_ERROR
)
from pydantic import BaseModel, Field

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.models import ContentItem, ModuleType, ContentType
from src.services.content_service import get_content_service

logger = get_logger(__name__)


# MCP Tool Models
class AddNoteRequest(BaseModel):
    """Request model for adding notes."""
    content_id: str = Field(..., description="Content ID to associate note with")
    note_text: str = Field(..., description="Note content")
    note_type: str = Field(default="general", description="Type of note")
    tags: List[str] = Field(default_factory=list, description="Note tags")
    section_reference: Optional[str] = Field(None, description="Section/page reference")


class FetchResourceRequest(BaseModel):
    """Request model for fetching resources."""
    resource_type: str = Field(..., description="Type of resource to fetch")
    query: Optional[str] = Field(None, description="Search query")
    content_id: Optional[str] = Field(None, description="Related content ID")
    limit: int = Field(default=10, description="Maximum number of resources")


class UpdateProgressRequest(BaseModel):
    """Request model for updating progress."""
    content_id: str = Field(..., description="Content ID")
    progress_type: str = Field(..., description="Type of progress update")
    progress_value: float = Field(..., description="Progress value (0.0-1.0)")
    milestone: Optional[str] = Field(None, description="Milestone name")
    notes: Optional[str] = Field(None, description="Progress notes")


class AlexandriaMCPServer:
    """
    Alexandria MCP Server implementation.
    
    Provides tools for note-taking, resource fetching, and progress tracking
    integrated with the Alexandria platform's content management system.
    """
    
    def __init__(self):
        """Initialize the MCP server."""
        self.settings = get_settings()
        self.app = FastMCP("Alexandria MCP Server")
        self.notes_storage_path = Path(self.settings.user_data_path) / "notes"
        self.progress_storage_path = Path(self.settings.user_data_path) / "progress"
        
        # Ensure storage directories exist
        self.notes_storage_path.mkdir(parents=True, exist_ok=True)
        self.progress_storage_path.mkdir(parents=True, exist_ok=True)
        
        self._setup_tools()
        logger.info("Alexandria MCP Server initialized")
    
    def _setup_tools(self):
        """Set up MCP tools."""
        
        @self.app.tool()
        async def add_note(
            content_id: str,
            note_text: str,
            note_type: str = "general",
            tags: List[str] = None,
            section_reference: str = None
        ) -> Dict[str, Any]:
            """
            Add a note to a specific content item.
            
            Args:
                content_id: Content ID to associate note with
                note_text: Note content
                note_type: Type of note (general, highlight, question, insight)
                tags: List of tags for organization
                section_reference: Page/section reference
            
            Returns:
                Dict with note information and success status
            """
            try:
                # Validate content exists
                content_service = await get_content_service()
                content = await content_service.get_content_item(content_id)
                
                if not content:
                    return {
                        "success": False,
                        "error": f"Content not found: {content_id}"
                    }
                
                # Create note data
                note_id = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{content_id[:8]}"
                note_data = {
                    "note_id": note_id,
                    "content_id": content_id,
                    "content_title": content.title,
                    "note_text": note_text,
                    "note_type": note_type,
                    "tags": tags or [],
                    "section_reference": section_reference,
                    "created_at": datetime.now().isoformat(),
                    "module_type": content.module_type.value if content.module_type else "library"
                }
                
                # Save note to file
                note_file = self.notes_storage_path / f"{note_id}.json"
                with open(note_file, 'w', encoding='utf-8') as f:
                    json.dump(note_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Note added: {note_id} for content {content_id}")
                
                return {
                    "success": True,
                    "note_id": note_id,
                    "message": f"Note added successfully to '{content.title}'",
                    "note_data": note_data
                }
                
            except Exception as e:
                logger.error(f"Failed to add note: {e}")
                return {
                    "success": False,
                    "error": f"Failed to add note: {str(e)}"
                }
        
        @self.app.tool()
        async def fetch_resource(
            resource_type: str,
            query: str = None,
            content_id: str = None,
            limit: int = 10
        ) -> Dict[str, Any]:
            """
            Fetch related external resources.
            
            Args:
                resource_type: Type of resource (articles, videos, discussions, books)
                query: Search query
                content_id: Related content ID for context
                limit: Maximum number of resources
            
            Returns:
                Dict with fetched resources
            """
            try:
                resources = []
                
                # Get content context if provided
                content_context = None
                if content_id:
                    content_service = await get_content_service()
                    content_context = await content_service.get_content_item(content_id)
                
                # Generate mock resources based on type and query
                # In a real implementation, this would integrate with external APIs
                if resource_type == "articles":
                    resources = await self._fetch_articles(query, content_context, limit)
                elif resource_type == "videos":
                    resources = await self._fetch_videos(query, content_context, limit)
                elif resource_type == "discussions":
                    resources = await self._fetch_discussions(query, content_context, limit)
                elif resource_type == "books":
                    resources = await self._fetch_related_books(query, content_context, limit)
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported resource type: {resource_type}"
                    }
                
                logger.info(f"Fetched {len(resources)} {resource_type} resources")
                
                return {
                    "success": True,
                    "resource_type": resource_type,
                    "query": query,
                    "resources": resources,
                    "content_context": content_context.title if content_context else None
                }
                
            except Exception as e:
                logger.error(f"Failed to fetch resources: {e}")
                return {
                    "success": False,
                    "error": f"Failed to fetch resources: {str(e)}"
                }
        
        @self.app.tool()
        async def update_progress(
            content_id: str,
            progress_type: str,
            progress_value: float,
            milestone: str = None,
            notes: str = None
        ) -> Dict[str, Any]:
            """
            Update reading/learning progress for content.
            
            Args:
                content_id: Content ID
                progress_type: Type of progress (reading, completion, comprehension)
                progress_value: Progress value (0.0-1.0)
                milestone: Milestone name
                notes: Progress notes
            
            Returns:
                Dict with progress update information
            """
            try:
                # Validate content exists
                content_service = await get_content_service()
                content = await content_service.get_content_item(content_id)
                
                if not content:
                    return {
                        "success": False,
                        "error": f"Content not found: {content_id}"
                    }
                
                # Validate progress value
                if not (0.0 <= progress_value <= 1.0):
                    return {
                        "success": False,
                        "error": "Progress value must be between 0.0 and 1.0"
                    }
                
                # Load existing progress or create new
                progress_file = self.progress_storage_path / f"progress_{content_id}.json"
                
                if progress_file.exists():
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                else:
                    progress_data = {
                        "content_id": content_id,
                        "content_title": content.title,
                        "module_type": content.module_type.value if content.module_type else "library",
                        "created_at": datetime.now().isoformat(),
                        "progress_history": []
                    }
                
                # Add new progress entry
                progress_entry = {
                    "progress_type": progress_type,
                    "progress_value": progress_value,
                    "milestone": milestone,
                    "notes": notes,
                    "timestamp": datetime.now().isoformat()
                }
                
                progress_data["progress_history"].append(progress_entry)
                progress_data["last_updated"] = datetime.now().isoformat()
                progress_data[f"current_{progress_type}"] = progress_value
                
                # Save updated progress
                with open(progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Progress updated: {progress_type}={progress_value:.2f} for content {content_id}")
                
                return {
                    "success": True,
                    "message": f"Progress updated for '{content.title}'",
                    "progress_data": progress_entry,
                    "current_progress": {
                        progress_type: progress_value
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to update progress: {e}")
                return {
                    "success": False,
                    "error": f"Failed to update progress: {str(e)}"
                }
    
    async def _fetch_articles(self, query: str, content_context: ContentItem, limit: int) -> List[Dict[str, Any]]:
        """Fetch related articles (mock implementation)."""
        # In a real implementation, this would call external APIs like Wikipedia, Scholar, etc.
        articles = [
            {
                "title": f"Article about {query or 'related topics'}",
                "url": f"https://example.com/article-{i}",
                "summary": f"This article discusses {query or 'topics related to your content'}...",
                "source": "Academic Journal",
                "relevance_score": 0.9 - (i * 0.1),
                "type": "article"
            }
            for i in range(min(limit, 3))
        ]
        return articles
    
    async def _fetch_videos(self, query: str, content_context: ContentItem, limit: int) -> List[Dict[str, Any]]:
        """Fetch related videos (mock implementation)."""
        videos = [
            {
                "title": f"Video: {query or 'Educational Content'}",
                "url": f"https://youtube.com/watch?v=example{i}",
                "summary": f"Educational video about {query or 'your content topics'}",
                "duration": f"{10 + i * 5} minutes",
                "source": "YouTube",
                "relevance_score": 0.85 - (i * 0.1),
                "type": "video"
            }
            for i in range(min(limit, 3))
        ]
        return videos
    
    async def _fetch_discussions(self, query: str, content_context: ContentItem, limit: int) -> List[Dict[str, Any]]:
        """Fetch related discussions (mock implementation)."""
        discussions = [
            {
                "title": f"Discussion: {query or 'Book Discussion'}",
                "url": f"https://reddit.com/r/books/discussion-{i}",
                "summary": f"Community discussion about {query or 'similar content'}",
                "platform": "Reddit",
                "participants": 15 + i * 5,
                "relevance_score": 0.8 - (i * 0.1),
                "type": "discussion"
            }
            for i in range(min(limit, 3))
        ]
        return discussions
    
    async def _fetch_related_books(self, query: str, content_context: ContentItem, limit: int) -> List[Dict[str, Any]]:
        """Fetch related books (mock implementation)."""
        books = [
            {
                "title": f"Related Book: {query or 'Similar Topic'}",
                "author": f"Author {i + 1}",
                "isbn": f"978-0-123456-{i:02d}-0",
                "summary": f"A book exploring {query or 'similar themes'}...",
                "rating": 4.5 - (i * 0.2),
                "relevance_score": 0.9 - (i * 0.15),
                "type": "book"
            }
            for i in range(min(limit, 3))
        ]
        return books
    
    async def get_notes_for_content(self, content_id: str) -> List[Dict[str, Any]]:
        """Get all notes for a specific content item."""
        notes = []
        
        for note_file in self.notes_storage_path.glob(f"*_{content_id[:8]}.json"):
            try:
                with open(note_file, 'r', encoding='utf-8') as f:
                    note_data = json.load(f)
                    if note_data.get('content_id') == content_id:
                        notes.append(note_data)
            except Exception as e:
                logger.warning(f"Failed to load note file {note_file}: {e}")
        
        return sorted(notes, key=lambda x: x.get('created_at', ''), reverse=True)
    
    async def get_progress_for_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get progress data for a specific content item."""
        progress_file = self.progress_storage_path / f"progress_{content_id}.json"
        
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load progress file {progress_file}: {e}")
            return None
    
    def run(self, host: str = None, port: int = None):
        """Run the MCP server."""
        host = host or self.settings.mcp_server_host
        port = port or self.settings.mcp_server_port
        
        logger.info(f"Starting Alexandria MCP Server on {host}:{port}")
        self.app.run(host=host, port=port)


# Global MCP server instance
_mcp_server: Optional[AlexandriaMCPServer] = None


async def get_mcp_server() -> AlexandriaMCPServer:
    """
    Get or create the MCP server instance.
    
    Returns:
        AlexandriaMCPServer: The server instance
    """
    global _mcp_server
    
    if _mcp_server is None:
        _mcp_server = AlexandriaMCPServer()
        logger.info("Alexandria MCP Server singleton created")
    
    return _mcp_server


def main():
    """Main entry point for running the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alexandria MCP Server")
    parser.add_argument("--host", default=None, help="Server host")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    
    args = parser.parse_args()
    
    # Create and run server
    server = AlexandriaMCPServer()
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()