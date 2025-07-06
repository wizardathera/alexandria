"""
Model Context Protocol (MCP) server implementation for Alexandria.

This module provides MCP server functionality with tools for note-taking,
resource fetching, and progress tracking integrated with the Alexandria platform.
"""

from .server import AlexandriaMCPServer, get_mcp_server

__all__ = [
    "AlexandriaMCPServer",
    "get_mcp_server"
]