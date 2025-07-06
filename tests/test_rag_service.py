"""
Tests for RAG service functionality.

This module tests the RAG query system including vector search, LLM response generation,
and source citation formatting with proper mocking of external services.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any
import os
import tempfile

from src.rag.rag_service import (
    RAGService, OpenAILLMProvider, QueryContext, RAGResponse,
    get_rag_service
)
from src.utils.embeddings import EmbeddingService


class TestOpenAILLMProvider:
    """Test OpenAI LLM provider functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.openai_api_key = "test-api-key"
        settings.chroma_collection_name = "test_collection"
        return settings
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 70
        
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def llm_provider(self, mock_settings, mock_openai_client):
        """Create LLM provider with mocked client."""
        with patch('src.rag.rag_service.get_settings') as mock_get_settings:
            with patch('src.rag.rag_service.OpenAI') as mock_openai:
                mock_get_settings.return_value = mock_settings
                mock_openai.return_value = mock_openai_client
                provider = OpenAILLMProvider()
                provider.client = mock_openai_client
                return provider
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, llm_provider, mock_openai_client):
        """Test successful response generation."""
        # Expected behavior test
        prompt = "What is the main theme of the book?"
        
        response, token_usage = await llm_provider.generate_response(prompt)
        
        assert response == "Test response"
        assert token_usage["total_tokens"] == 70
        assert token_usage["prompt_tokens"] == 50
        assert token_usage["completion_tokens"] == 20
        
        # Verify API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["messages"][0]["content"] == prompt
        assert call_args[1]["model"] == "gpt-3.5-turbo"
    
    @pytest.mark.asyncio
    async def test_generate_response_with_parameters(self, llm_provider, mock_openai_client):
        """Test response generation with custom parameters."""
        # Edge case test
        prompt = "Test prompt"
        
        await llm_provider.generate_response(
            prompt, max_tokens=500, temperature=0.5, model="gpt-4"
        )
        
        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["max_tokens"] == 500
        assert call_args[1]["temperature"] == 0.5
        assert call_args[1]["model"] == "gpt-4"
    
    @pytest.mark.asyncio
    async def test_generate_response_api_error(self, llm_provider, mock_openai_client):
        """Test handling of API errors."""
        # Failure scenario test
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        with pytest.raises(ValueError, match="Failed to generate response"):
            await llm_provider.generate_response("test prompt")
    
    def test_get_context_limit(self, llm_provider):
        """Test context limit retrieval."""
        # Expected behavior test
        assert llm_provider.get_context_limit("gpt-3.5-turbo") == 16385
        assert llm_provider.get_context_limit("gpt-4") == 8192
        assert llm_provider.get_context_limit() == 16385  # Default
    
    def test_unsupported_model(self, llm_provider, mock_openai_client):
        """Test error handling for unsupported models."""
        # Failure scenario test
        with pytest.raises(ValueError, match="Unsupported model"):
            asyncio.run(llm_provider.generate_response("test", model="unsupported-model"))


class TestRAGService:
    """Test RAG service functionality."""
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create mock vector database."""
        mock_db = AsyncMock()
        mock_db.query.return_value = {
            "documents": [
                "This is the first relevant passage about the main character.",
                "Here is another passage discussing the book's themes."
            ],
            "metadatas": [
                {"book_title": "Test Book", "page": 15, "chapter": "Chapter 1"},
                {"book_title": "Test Book", "page": 32, "chapter": "Chapter 2"}
            ],
            "distances": [0.2, 0.3],
            "ids": ["doc1", "doc2"]
        }
        return mock_db
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock_service = AsyncMock()
        mock_service.embed_query.return_value = [0.1] * 1536  # Mock embedding vector
        return mock_service
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        mock_provider = AsyncMock()
        mock_provider.generate_response.return_value = (
            "Based on the provided context, the main character shows significant growth throughout the story.",
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        return mock_provider
    
    @pytest.fixture
    def mock_settings_rag(self):
        """Create mock settings for RAG service."""
        settings = Mock()
        settings.chroma_collection_name = "test_collection"
        return settings
    
    @pytest.fixture
    def rag_service(self, mock_vector_db, mock_embedding_service, mock_llm_provider, mock_settings_rag):
        """Create RAG service with mocked components."""
        with patch('src.rag.rag_service.get_settings') as mock_get_settings:
            mock_get_settings.return_value = mock_settings_rag
            service = RAGService()
            service.vector_db = mock_vector_db
            service.embedding_service = mock_embedding_service
            service.llm_provider = mock_llm_provider
            return service
    
    @pytest.mark.asyncio
    async def test_query_success(self, rag_service, mock_vector_db, mock_embedding_service, mock_llm_provider):
        """Test successful RAG query processing."""
        # Expected behavior test
        question = "What is the main character's development arc?"
        
        response = await rag_service.query(question)
        
        # Verify response structure
        assert isinstance(response, RAGResponse)
        assert response.answer == "Based on the provided context, the main character shows significant growth throughout the story."
        assert len(response.sources) == 2
        assert response.token_usage["total_tokens"] == 150
        assert response.confidence_score > 0.0
        assert response.processing_time > 0.0
        
        # Verify service calls
        mock_embedding_service.embed_query.assert_called_once_with(question)
        mock_vector_db.query.assert_called_once()
        mock_llm_provider.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_with_book_filter(self, rag_service, mock_vector_db):
        """Test query with book ID filter."""
        # Edge case test
        question = "Tell me about the themes"
        book_id = "book123"
        
        await rag_service.query(question, book_id=book_id)
        
        # Verify book filter was applied
        call_args = mock_vector_db.query.call_args
        assert call_args[1]["where"] == {"book_id": book_id}
    
    @pytest.mark.asyncio
    async def test_query_empty_question(self, rag_service):
        """Test handling of empty questions."""
        # Failure scenario test
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await rag_service.query("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await rag_service.query("   ")
    
    @pytest.mark.asyncio
    async def test_query_no_context_retrieved(self, rag_service, mock_vector_db, mock_llm_provider):
        """Test query when no relevant context is found."""
        # Edge case test
        mock_vector_db.query.return_value = {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": []
        }
        
        response = await rag_service.query("What about non-existent topic?")
        
        # Should still generate response with no-context prompt
        assert response.answer is not None
        assert len(response.sources) == 0
        assert response.confidence_score == 0.1  # Low confidence without context
        
        # Verify LLM was called with no-context prompt
        mock_llm_provider.generate_response.assert_called_once()
        prompt_arg = mock_llm_provider.generate_response.call_args[1]["prompt"]
        assert "don't have access to specific book content" in prompt_arg
    
    @pytest.mark.asyncio
    async def test_query_database_error(self, rag_service, mock_vector_db):
        """Test handling of database errors."""
        # Failure scenario test
        mock_vector_db.query.side_effect = Exception("Database connection failed")
        
        with pytest.raises(ValueError, match="Failed to process query"):
            await rag_service.query("test question")
    
    @pytest.mark.asyncio
    async def test_query_uninitialized_service(self):
        """Test query on uninitialized service."""
        # Failure scenario test
        service = RAGService()
        # Don't initialize service
        
        with pytest.raises(RuntimeError, match="RAG service not initialized"):
            await service.query("test question")
    
    def test_format_sources(self, rag_service):
        """Test source formatting functionality."""
        # Expected behavior test
        context = QueryContext(
            documents=["First document content", "Second document content"],
            metadatas=[
                {"book_title": "Book 1", "page": 10, "chapter": "Introduction"},
                {"book_title": "Book 2", "page": 25}
            ],
            distances=[0.1, 0.3]
        )
        
        sources = rag_service._format_sources(context)
        
        assert len(sources) == 2
        
        # Check first source
        assert sources[0]["id"] == 1
        assert sources[0]["similarity_score"] == 0.9  # 1.0 - 0.1
        assert "Book 1" in sources[0]["source_description"]
        assert "page 10" in sources[0]["source_description"]
        assert "Introduction" in sources[0]["source_description"]
        
        # Check second source
        assert sources[1]["id"] == 2
        assert sources[1]["similarity_score"] == 0.7  # 1.0 - 0.3
        assert "Book 2" in sources[1]["source_description"]
        assert "page 25" in sources[1]["source_description"]
    
    def test_calculate_confidence(self, rag_service):
        """Test confidence score calculation."""
        # Expected behavior test
        context = QueryContext(
            documents=["doc1", "doc2", "doc3"],
            distances=[0.1, 0.2, 0.3],  # Good similarity scores
            total_retrieved=3
        )
        token_usage = {"total_tokens": 150}
        
        confidence = rag_service._calculate_confidence(context, token_usage)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high with good similarity and multiple docs
    
    def test_calculate_confidence_no_context(self, rag_service):
        """Test confidence calculation with no context."""
        # Edge case test
        context = QueryContext(documents=[], total_retrieved=0)
        token_usage = {"total_tokens": 50}
        
        confidence = rag_service._calculate_confidence(context, token_usage)
        
        assert confidence == 0.1  # Low confidence without context
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, rag_service, mock_vector_db):
        """Test successful health check."""
        # Expected behavior test
        mock_vector_db.get_collection_info.return_value = {
            "name": "test_collection",
            "document_count": 100
        }
        
        health = await rag_service.health_check()
        
        assert health["rag_service"] == "healthy"
        assert health["vector_database"] == "healthy"
        assert health["document_count"] == 100
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_health_check_database_error(self, rag_service, mock_vector_db):
        """Test health check with database error."""
        # Failure scenario test
        mock_vector_db.get_collection_info.side_effect = Exception("DB Error")
        
        health = await rag_service.health_check()
        
        assert health["rag_service"] == "error"
        assert "error" in health


class TestRAGServiceIntegration:
    """Integration tests for RAG service."""
    
    @pytest.mark.asyncio
    async def test_get_rag_service_singleton(self):
        """Test RAG service singleton pattern."""
        # Expected behavior test
        with patch('src.rag.rag_service.RAGService') as mock_rag_service_class:
            mock_instance = AsyncMock()
            mock_instance.initialize.return_value = True
            mock_rag_service_class.return_value = mock_instance
            
            # Reset global instance
            import src.rag.rag_service
            src.rag.rag_service._rag_service = None
            
            # First call should create instance
            service1 = await get_rag_service()
            
            # Second call should return same instance
            service2 = await get_rag_service()
            
            assert service1 is service2
            mock_rag_service_class.assert_called_once()
            mock_instance.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_rag_service_initialization_failure(self):
        """Test handling of initialization failure."""
        # Failure scenario test
        with patch('src.rag.rag_service.RAGService') as mock_rag_service_class:
            mock_instance = AsyncMock()
            mock_instance.initialize.return_value = False
            mock_rag_service_class.return_value = mock_instance
            
            # Reset global instance
            import src.rag.rag_service
            src.rag.rag_service._rag_service = None
            
            with pytest.raises(RuntimeError, match="Failed to initialize RAG service"):
                await get_rag_service()


class TestQueryContext:
    """Test QueryContext data class."""
    
    def test_query_context_creation(self):
        """Test QueryContext initialization."""
        # Expected behavior test
        context = QueryContext()
        
        assert context.documents == []
        assert context.metadatas == []
        assert context.distances == []
        assert context.total_retrieved == 0
        assert context.query_embedding_time == 0.0
        assert context.vector_search_time == 0.0
    
    def test_query_context_with_data(self):
        """Test QueryContext with actual data."""
        # Expected behavior test
        context = QueryContext(
            documents=["doc1", "doc2"],
            metadatas=[{"title": "book1"}, {"title": "book2"}],
            distances=[0.1, 0.2],
            total_retrieved=2,
            query_embedding_time=0.5,
            vector_search_time=0.3
        )
        
        assert len(context.documents) == 2
        assert len(context.metadatas) == 2
        assert len(context.distances) == 2
        assert context.total_retrieved == 2
        assert context.query_embedding_time == 0.5
        assert context.vector_search_time == 0.3


class TestRAGResponse:
    """Test RAGResponse data class."""
    
    def test_rag_response_creation(self):
        """Test RAGResponse initialization."""
        # Expected behavior test
        context = QueryContext()
        sources = [{"id": 1, "content": "test"}]
        
        response = RAGResponse(
            answer="Test answer",
            sources=sources,
            context_used=context,
            token_usage={"total_tokens": 100},
            processing_time=1.5,
            confidence_score=0.8
        )
        
        assert response.answer == "Test answer"
        assert response.sources == sources
        assert response.context_used is context
        assert response.token_usage["total_tokens"] == 100
        assert response.processing_time == 1.5
        assert response.confidence_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__])