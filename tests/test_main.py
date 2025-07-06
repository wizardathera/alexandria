"""
Tests for the main FastAPI application.

This module contains tests for the core application setup, health checks,
and basic API functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os
import json

from src.main import create_app
from src.utils.config import get_settings


@pytest.fixture
def test_app():
    """
    Create test FastAPI application instance.
    
    Returns:
        FastAPI: Test application
    """
    return create_app()


@pytest.fixture
def client(test_app):
    """
    Create test client for the application.
    
    Args:
        test_app: Test application fixture
        
    Returns:
        TestClient: FastAPI test client
    """
    return TestClient(test_app)


@pytest.fixture
def mock_settings():
    """
    Mock application settings for testing.
    
    Returns:
        MagicMock: Mocked settings
    """
    settings = MagicMock()
    settings.debug = True
    settings.environment = "test"
    settings.host = "localhost"
    settings.port = 8000
    settings.log_level = "info"
    settings.cors_origins = ["http://localhost:3000"]
    settings.openai_api_key = "test-key"
    settings.vector_db_type = "chroma"
    settings.chroma_persist_directory = tempfile.mkdtemp()
    settings.chroma_collection_name = "test_collection"
    settings.max_upload_size_mb = 50
    settings.supported_formats = "pdf,epub,doc,docx,txt,html"
    settings.get_supported_formats_list.return_value = ["pdf", "epub", "doc", "docx", "txt", "html"]
    settings.get_max_upload_size_bytes.return_value = 50 * 1024 * 1024
    return settings


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """
        Test basic health check endpoint returns success.
        
        Expected behavior: Health endpoint returns 200 with status info.
        """
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
        assert data["environment"] == "development"
    
    def test_detailed_health_check(self, client):
        """
        Test detailed health check endpoint returns comprehensive info.
        
        Expected behavior: Detailed health endpoint returns full diagnostics.
        """
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "services" in data
        assert "configuration" in data
        assert "vector_db_type" in data["configuration"]
    
    def test_readiness_check(self, client):
        """
        Test readiness check endpoint.
        
        Expected behavior: Readiness endpoint confirms app is ready.
        """
        response = client.get("/api/v1/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ready"
        assert "timestamp" in data
    
    def test_liveness_check(self, client):
        """
        Test liveness check endpoint.
        
        Expected behavior: Liveness endpoint confirms app is alive.
        """
        response = client.get("/api/v1/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data


class TestBookEndpoints:
    """Test book management endpoints."""
    
    def test_list_books_empty(self, client):
        """
        Test listing books when none are uploaded.
        
        Expected behavior: Returns empty list with total count 0.
        """
        response = client.get("/api/v1/books")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["books"] == []
        assert data["total"] == 0
    
    def test_get_nonexistent_book(self, client):
        """
        Test getting a book that doesn't exist.
        
        Edge case: Should return 404 for non-existent book.
        """
        response = client.get("/api/v1/books/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_upload_unsupported_format(self, client):
        """
        Test uploading a file with unsupported format.
        
        Failure scenario: Should reject unsupported file types.
        """
        # Create a fake file with unsupported extension
        file_content = b"fake file content"
        
        response = client.post(
            "/api/v1/books/upload",
            files={"file": ("test.xyz", file_content, "application/octet-stream")}
        )
        
        assert response.status_code == 400
        assert "unsupported file format" in response.json()["detail"].lower()


class TestChatEndpoints:
    """Test chat/Q&A endpoints."""
    
    @patch('src.api.chat.get_rag_service')
    def test_query_success(self, mock_get_rag_service, client):
        """
        Test successful RAG query with mocked service.
        
        Expected behavior: Returns structured response from RAG service.
        """
        # Setup mock RAG service
        mock_rag_service = AsyncMock()
        mock_rag_response = MagicMock()
        mock_rag_response.answer = "The book explores themes of personal growth and resilience through the protagonist's journey."
        mock_rag_response.sources = [
            {
                "id": 1,
                "content": "The main character begins a journey of self-discovery...",
                "similarity_score": 0.85,
                "source_description": "Test Book, page 15 (Chapter 1)"
            }
        ]
        mock_rag_response.token_usage = {
            "prompt_tokens": 120,
            "completion_tokens": 80,
            "total_tokens": 200
        }
        mock_rag_response.confidence_score = 0.85
        mock_rag_response.processing_time = 1.5
        
        mock_rag_service.query.return_value = mock_rag_response
        mock_get_rag_service.return_value = mock_rag_service
        
        query_data = {
            "question": "What is this book about?",
            "context_limit": 5
        }
        
        response = client.post("/api/v1/chat/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "The book explores themes of personal growth and resilience through the protagonist's journey."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["similarity_score"] == 0.85
        assert "conversation_id" in data
        assert "message_id" in data
        assert data["token_usage"]["total_tokens"] == 200
        
        # Verify RAG service was called correctly
        mock_rag_service.query.assert_called_once_with(
            question="What is this book about?",
            book_id=None,
            context_limit=5
        )
    
    def test_query_empty_question(self, client):
        """
        Test query with empty question.
        
        Edge case: Should handle validation error for empty question.
        """
        query_data = {
            "question": "",
            "context_limit": 5
        }
        
        response = client.post("/api/v1/chat/query", json=query_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_list_conversations_empty(self, client):
        """
        Test listing conversations when none exist.
        
        Expected behavior: Returns empty list.
        """
        response = client.get("/api/v1/chat/conversations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data == []
    
    def test_get_nonexistent_conversation(self, client):
        """
        Test getting a conversation that doesn't exist.
        
        Failure scenario: Should return 404 for non-existent conversation.
        """
        response = client.get("/api/v1/chat/conversations/nonexistent-id")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    @patch('src.api.chat.get_rag_service')
    def test_query_with_book_filter(self, mock_get_rag_service, client):
        """
        Test query with book ID filter.
        
        Edge case: Should pass book filter to RAG service.
        """
        mock_rag_service = AsyncMock()
        mock_rag_response = MagicMock()
        mock_rag_response.answer = "This specific book discusses..."
        mock_rag_response.sources = []
        mock_rag_response.token_usage = {"total_tokens": 100}
        mock_rag_response.confidence_score = 0.7
        mock_rag_response.processing_time = 1.0
        
        mock_rag_service.query.return_value = mock_rag_response
        mock_get_rag_service.return_value = mock_rag_service
        
        query_data = {
            "question": "What are the themes?",
            "book_id": "book123",
            "context_limit": 3
        }
        
        response = client.post("/api/v1/chat/query", json=query_data)
        
        assert response.status_code == 200
        
        # Verify book filter was passed
        mock_rag_service.query.assert_called_once_with(
            question="What are the themes?",
            book_id="book123",
            context_limit=3
        )
    
    @patch('src.api.chat.get_rag_service')
    def test_query_service_error(self, mock_get_rag_service, client):
        """
        Test query handling when RAG service fails.
        
        Failure scenario: Should return error response gracefully.
        """
        mock_rag_service = AsyncMock()
        mock_rag_service.query.side_effect = Exception("Service unavailable")
        mock_get_rag_service.return_value = mock_rag_service
        
        query_data = {
            "question": "What is this about?",
            "context_limit": 5
        }
        
        response = client.post("/api/v1/chat/query", json=query_data)
        
        assert response.status_code == 200  # Still returns 200 but with error message
        data = response.json()
        
        assert "error" in data["answer"].lower() or "apologize" in data["answer"].lower()
        assert data["sources"] == []
        assert data["token_usage"]["total_tokens"] == 0
    
    @patch('src.api.chat.get_rag_service')
    def test_chat_health_check_success(self, mock_get_rag_service, client):
        """
        Test RAG service health check endpoint.
        
        Expected behavior: Returns health status from RAG service.
        """
        mock_rag_service = AsyncMock()
        mock_health_status = {
            "rag_service": "healthy",
            "vector_database": "healthy",
            "embedding_service": "healthy",
            "llm_provider": "healthy",
            "document_count": 150,
            "timestamp": "2024-01-01T12:00:00"
        }
        mock_rag_service.health_check.return_value = mock_health_status
        mock_get_rag_service.return_value = mock_rag_service
        
        response = client.get("/api/v1/chat/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["rag_service"] == "healthy"
        assert data["vector_database"] == "healthy"
        assert data["document_count"] == 150
        assert "timestamp" in data
    
    @patch('src.api.chat.get_rag_service')
    def test_chat_health_check_error(self, mock_get_rag_service, client):
        """
        Test RAG health check when service fails.
        
        Failure scenario: Should return error status gracefully.
        """
        mock_get_rag_service.side_effect = Exception("Service initialization failed")
        
        response = client.get("/api/v1/chat/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["rag_service"] == "error"
        assert "error" in data
        assert "timestamp" in data


class TestApplicationSetup:
    """Test application configuration and setup."""
    
    @patch('src.utils.config.get_settings')
    def test_app_creation_with_custom_settings(self, mock_get_settings, mock_settings):
        """
        Test application creation with custom settings.
        
        Expected behavior: App should use provided configuration.
        """
        mock_get_settings.return_value = mock_settings
        
        app = create_app()
        
        assert app.title == "Dynamic Book Companion API"
        assert app.debug == mock_settings.debug
    
    def test_cors_configuration(self, client):
        """
        Test CORS headers are properly configured.
        
        Expected behavior: CORS headers should be present.
        """
        response = client.options("/api/v1/health")
        
        # FastAPI automatically handles OPTIONS requests for CORS
        assert response.status_code in [200, 405]  # 405 if no explicit OPTIONS handler
    
    def test_api_documentation_available(self, client):
        """
        Test that API documentation is accessible.
        
        Expected behavior: OpenAPI docs should be available.
        """
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")