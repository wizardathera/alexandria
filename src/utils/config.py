"""
Configuration management for the Alexandria application.

This module handles loading and validation of environment variables
and application settings using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
from functools import lru_cache
import os

# Global constants for collection naming
DEFAULT_COLLECTION_NAME = "alexandria_books"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application Configuration
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="info", description="Logging level")
    host: str = Field(default="localhost", description="Application host")
    port: int = Field(default=8000, description="Application port")
    
    # Authentication (Phase 1: disabled, Phase 2: enabled)
    auth_enabled: bool = Field(default=False, description="Enable authentication")
    secret_key: str = Field(default="dev-secret-key", description="Secret key for sessions")
    
    # AI Provider Configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key (optional)")
    
    # Embedding Model Configuration
    embedding_model: str = Field(
        default="text-embedding-ada-002", 
        description="OpenAI embedding model to use"
    )
    llm_model: str = Field(
        default="gpt-3.5-turbo", 
        description="OpenAI language model to use"
    )
    
    # Vector Database Configuration
    vector_db_type: str = Field(default="chroma", description="Vector database type")
    
    # Chroma Configuration (Phase 1)
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", 
        description="Chroma persistence directory"
    )
    chroma_collection_name: str = Field(
        default=DEFAULT_COLLECTION_NAME, 
        description="Chroma collection name"
    )
    
    # Supabase Configuration (Phase 2)
    supabase_url: Optional[str] = Field(default=None, description="Supabase project URL")
    supabase_key: Optional[str] = Field(default=None, description="Supabase anon key")
    
    # Supabase Database Configuration (Phase 2 - for direct PostgreSQL connection)
    supabase_db_host: Optional[str] = Field(default=None, description="Supabase PostgreSQL host")
    supabase_db_port: int = Field(default=5432, description="Supabase PostgreSQL port")
    supabase_db_name: Optional[str] = Field(default=None, description="Supabase database name")
    supabase_db_user: Optional[str] = Field(default=None, description="Supabase database user")
    supabase_db_password: Optional[str] = Field(default=None, description="Supabase database password")
    
    # File Upload Configuration
    max_upload_size_mb: int = Field(default=50, description="Maximum file upload size in MB")
    supported_formats: str = Field(
        default="pdf,epub,doc,docx,txt,html", 
        description="Supported file formats"
    )
    books_storage_path: str = Field(
        default="./data/books", 
        description="Directory for uploaded books"
    )
    user_data_path: str = Field(
        default="./data/users", 
        description="Directory for user data"
    )
    
    # MCP Server Configuration
    mcp_server_port: int = Field(default=8080, description="MCP server port")
    mcp_server_host: str = Field(default="localhost", description="MCP server host")
    mcp_server_name: str = Field(default="alexandria-mcp-server", description="MCP server name")
    
    # Frontend Configuration (Phase 1: Streamlit)
    streamlit_port: int = Field(default=8501, description="Streamlit frontend port")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.lower()
    
    @validator('vector_db_type')
    def validate_vector_db_type(cls, v):
        """Validate vector database type is supported."""
        valid_types = ['chroma', 'supabase']
        if v.lower() not in valid_types:
            raise ValueError(f'Vector DB type must be one of: {valid_types}')
        return v.lower()
    
    @validator('supported_formats')
    def validate_supported_formats(cls, v):
        """Validate supported formats are recognized."""
        valid_formats = ['pdf', 'epub', 'doc', 'docx', 'txt', 'html']
        formats = [f.strip().lower() for f in v.split(',')]
        
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValueError(f'Unsupported format: {fmt}. Valid formats: {valid_formats}')
        
        return v
    
    def get_supported_formats_list(self) -> List[str]:
        """
        Get supported file formats as a list.
        
        Returns:
            List[str]: List of supported file formats
        """
        return [f.strip().lower() for f in self.supported_formats.split(',')]
    
    def get_max_upload_size_bytes(self) -> int:
        """
        Get maximum upload size in bytes.
        
        Returns:
            int: Maximum upload size in bytes
        """
        return self.max_upload_size_mb * 1024 * 1024
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()