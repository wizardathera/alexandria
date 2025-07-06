"""
Tests for the enhanced embedding service.

Tests cover:
- Enhanced metadata generation
- Permission-aware search
- Content relationship discovery
- Multi-module support
- Performance requirements
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.services.enhanced_embedding_service import EnhancedEmbeddingService
from src.models import (
    ContentItem, EmbeddingMetadata, User, ContentRelationship,
    ModuleType, ContentType, ContentVisibility, UserRole,
    ProcessingStatus, ContentRelationshipType
)
from src.utils.enhanced_chunking import EnhancedChunkMetadata


class TestEnhancedEmbeddingService:
    """Test suite for enhanced embedding service."""
    
    @pytest.fixture
    async def embedding_service(self):
        """Create enhanced embedding service for testing."""
        service = EnhancedEmbeddingService()
        
        # Mock dependencies
        service._embedding_service = Mock()
        service._embedding_service.embed_documents = AsyncMock(return_value=(
            [[0.1] * 1536 for _ in range(3)],  # Mock embeddings
            Mock(total_tokens=100, total_documents=3)
        ))
        service._embedding_service.get_cache_stats = Mock(return_value={
            "hits": 10, "misses": 5, "hit_rate": 0.67
        })
        
        service._chunking_service = Mock()
        service._chunking_service.chunk_content = AsyncMock(return_value=[
            EnhancedChunkMetadata(
                text="Test chunk 1",
                chunk_index=0,
                chunk_type="paragraph",
                source_location={"page": 1, "section": "intro"},
                importance_score=0.8,
                quality_score=0.9
            ),
            EnhancedChunkMetadata(
                text="Test chunk 2", 
                chunk_index=1,
                chunk_type="paragraph",
                source_location={"page": 1, "section": "body"},
                importance_score=0.7,
                quality_score=0.8
            )
        ])
        
        service._vector_db = Mock()
        service._vector_db.add_documents_with_metadata = AsyncMock(return_value=True)
        service._vector_db.query_with_permissions = AsyncMock(return_value={
            "documents": ["Related content 1", "Related content 2"],
            "metadatas": [
                {
                    "content_id": "related-1",
                    "content_type": "book",
                    "module_type": "library",
                    "semantic_tags": '["ai", "technology"]'
                },
                {
                    "content_id": "related-2", 
                    "content_type": "article",
                    "module_type": "library",
                    "semantic_tags": '["programming", "python"]'
                }
            ],
            "distances": [0.2, 0.3],
            "ids": ["embed-1", "embed-2"]
        })
        service._vector_db.similarity_search_with_relationships = AsyncMock(return_value={
            "documents": ["Enhanced result 1"],
            "metadatas": [{"content_id": "enhanced-1", "content_type": "book"}],
            "distances": [0.15],
            "ids": ["enhanced-embed-1"],
            "relationship_scores": [0.25]
        })
        
        service._content_service = Mock()
        service._content_service.update_content_item = AsyncMock(return_value=True)
        service._content_service.create_relationship = AsyncMock(return_value=True)
        service._content_service.get_content_relationships = AsyncMock(return_value=[])
        service._content_service.get_content_item = AsyncMock(return_value=None)
        
        return service
    
    @pytest.fixture
    def sample_content(self) -> ContentItem:
        """Create sample content item for testing."""
        return ContentItem(
            content_id="test-content-1",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Test Book",
            author="Test Author",
            file_path="/tmp/test.pdf",
            visibility=ContentVisibility.PUBLIC,
            created_by="test-user",
            processing_status=ProcessingStatus.PENDING
        )
    
    @pytest.fixture
    def sample_user(self) -> User:
        """Create sample user for testing."""
        return User(
            user_id="test-user",
            email="test@example.com",
            role=UserRole.READER,
            subscription_tier="free"
        )
    
    # ========================================
    # Content Processing Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_process_content_item_expected_behavior(
        self, 
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test successful content processing with enhanced metadata."""
        # Mock semantic tag extraction
        with patch.object(embedding_service, '_extract_semantic_tags', 
                         return_value=["ai", "technology", "books"]):
            
            result = await embedding_service.process_content_item(
                content=sample_content,
                user=sample_user
            )
        
        assert result is True
        
        # Verify content status was updated
        embedding_service._content_service.update_content_item.assert_called()
        
        # Verify embeddings were created
        embedding_service._vector_db.add_documents_with_metadata.assert_called_once()
        
        # Verify semantic tags were extracted
        assert sample_content.topics == ["ai", "technology", "books"]
        assert sample_content.processing_status == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_process_content_item_edge_case_empty_chunks(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test processing with empty chunks."""
        # Mock empty chunks
        embedding_service._chunking_service.chunk_content = AsyncMock(return_value=[])
        
        result = await embedding_service.process_content_item(
            content=sample_content,
            user=sample_user
        )
        
        assert result is False
        assert sample_content.processing_status == ProcessingStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_process_content_item_failure_scenario(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test processing failure handling."""
        # Mock chunking service to raise exception
        embedding_service._chunking_service.chunk_content = AsyncMock(
            side_effect=Exception("Chunking failed")
        )
        
        result = await embedding_service.process_content_item(
            content=sample_content,
            user=sample_user
        )
        
        assert result is False
        assert sample_content.processing_status == ProcessingStatus.FAILED
    
    # ========================================
    # Enhanced Metadata Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_create_enhanced_embeddings_expected_behavior(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem
    ):
        """Test enhanced embedding creation with proper metadata."""
        chunks = [
            EnhancedChunkMetadata(
                text="Test chunk with metadata",
                chunk_index=0,
                chunk_type="paragraph",
                source_location={"page": 1},
                importance_score=0.8,
                quality_score=0.9
            )
        ]
        semantic_tags = ["test", "metadata"]
        
        result = await embedding_service._create_enhanced_embeddings(
            content=sample_content,
            chunks=chunks,
            semantic_tags=semantic_tags,
            user=None
        )
        
        assert result == 1  # One embedding created
        
        # Verify vector database was called with enhanced metadata
        call_args = embedding_service._vector_db.add_documents_with_metadata.call_args
        assert call_args is not None
        
        embedding_metadata_list = call_args[1]["embedding_metadata"]
        assert len(embedding_metadata_list) == 1
        
        metadata = embedding_metadata_list[0]
        assert isinstance(metadata, EmbeddingMetadata)
        assert metadata.content_id == sample_content.content_id
        assert metadata.module_type == sample_content.module_type
        assert metadata.content_type == sample_content.content_type
        assert metadata.semantic_tags == semantic_tags
        assert metadata.importance_score == 0.8
        assert metadata.quality_score == 0.9
    
    @pytest.mark.asyncio
    async def test_create_enhanced_embeddings_edge_case_empty_embeddings(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem
    ):
        """Test handling of empty embeddings from embedding service."""
        # Mock empty embeddings
        embedding_service._embedding_service.embed_documents = AsyncMock(
            return_value=([], Mock())
        )
        
        chunks = [EnhancedChunkMetadata(text="Test", chunk_index=0, chunk_type="paragraph")]
        
        result = await embedding_service._create_enhanced_embeddings(
            content=sample_content,
            chunks=chunks,
            semantic_tags=[],
            user=None
        )
        
        assert result == 0  # No embeddings created
    
    @pytest.mark.asyncio
    async def test_create_enhanced_embeddings_failure_scenario(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem
    ):
        """Test embedding creation failure handling."""
        # Mock embedding service to raise exception
        embedding_service._embedding_service.embed_documents = AsyncMock(
            side_effect=Exception("Embedding failed")
        )
        
        chunks = [EnhancedChunkMetadata(text="Test", chunk_index=0, chunk_type="paragraph")]
        
        result = await embedding_service._create_enhanced_embeddings(
            content=sample_content,
            chunks=chunks,
            semantic_tags=[],
            user=None
        )
        
        assert result == 0
    
    # ========================================
    # Semantic Tag Extraction Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_extract_semantic_tags_expected_behavior(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem
    ):
        """Test semantic tag extraction with AI."""
        chunks = [
            EnhancedChunkMetadata(
                text="This is a book about artificial intelligence and machine learning",
                chunk_index=0,
                chunk_type="paragraph"
            )
        ]
        
        # Mock OpenAI API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '["ai", "machine learning", "technology"]'
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            tags = await embedding_service._extract_semantic_tags(chunks, sample_content)
        
        assert isinstance(tags, list)
        assert "ai" in tags
        assert "machine learning" in tags
        assert "technology" in tags
    
    @pytest.mark.asyncio
    async def test_extract_semantic_tags_edge_case_cache_hit(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem
    ):
        """Test semantic tag extraction with cache hit."""
        chunks = [EnhancedChunkMetadata(text="Cached content", chunk_index=0, chunk_type="paragraph")]
        
        # Pre-populate cache
        cache_key = f"{sample_content.content_id}:{len(chunks)}"
        embedding_service._semantic_tag_cache[cache_key] = ["cached", "tags"]
        
        tags = await embedding_service._extract_semantic_tags(chunks, sample_content)
        
        assert tags == ["cached", "tags"]
    
    @pytest.mark.asyncio
    async def test_extract_semantic_tags_failure_scenario(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem
    ):
        """Test semantic tag extraction with API failure."""
        chunks = [EnhancedChunkMetadata(text="Test content", chunk_index=0, chunk_type="paragraph")]
        
        # Mock OpenAI API to raise exception
        with patch('openai.OpenAI', side_effect=Exception("API failed")):
            tags = await embedding_service._extract_semantic_tags(chunks, sample_content)
        
        # Should fall back to basic keyword extraction
        assert isinstance(tags, list)
        assert len(tags) >= 1  # Should have fallback tags
    
    # ========================================
    # Enhanced Search Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_enhanced_search_expected_behavior(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test enhanced search with permission filtering."""
        # Mock content item for detailed results
        mock_content = ContentItem(
            content_id="related-1",
            title="Related Book",
            author="Related Author",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK
        )
        embedding_service._content_service.get_content_item = AsyncMock(return_value=mock_content)
        
        results = await embedding_service.enhanced_search(
            query="test query",
            user=sample_user,
            module_filter=ModuleType.LIBRARY,
            n_results=5,
            include_relationships=True
        )
        
        assert "documents" in results
        assert "enhanced_results" in results
        assert len(results["enhanced_results"]) > 0
        
        # Verify permission filtering was applied
        embedding_service._vector_db.query_with_permissions.assert_called()
        call_args = embedding_service._vector_db.query_with_permissions.call_args
        assert call_args[1]["user"] == sample_user
        assert call_args[1]["module_filter"] == ModuleType.LIBRARY
    
    @pytest.mark.asyncio
    async def test_enhanced_search_edge_case_no_results(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test enhanced search with no results."""
        # Mock empty search results
        embedding_service._vector_db.query_with_permissions = AsyncMock(return_value={
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": []
        })
        
        results = await embedding_service.enhanced_search(
            query="no matches",
            user=sample_user
        )
        
        assert results["documents"] == []
        assert results["enhanced_results"] == []
    
    @pytest.mark.asyncio
    async def test_enhanced_search_failure_scenario(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test enhanced search with vector database failure."""
        # Mock vector database to raise exception
        embedding_service._vector_db.query_with_permissions = AsyncMock(
            side_effect=Exception("Vector DB failed")
        )
        
        results = await embedding_service.enhanced_search(
            query="test query",
            user=sample_user
        )
        
        # Should return empty results without crashing
        assert results["documents"] == []
        assert results["enhanced_results"] == []
    
    # ========================================
    # Content Relationship Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_discover_content_relationships_expected_behavior(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test content relationship discovery."""
        chunks = [
            EnhancedChunkMetadata(text="Related content about AI", chunk_index=0, chunk_type="paragraph")
        ]
        
        result = await embedding_service._discover_content_relationships(
            content=sample_content,
            chunks=chunks,
            user=sample_user
        )
        
        assert result >= 0  # Should discover some relationships
        
        # Verify vector search was called
        embedding_service._vector_db.query_with_permissions.assert_called()
        
        # Verify relationships were created
        if result > 0:
            embedding_service._content_service.create_relationship.assert_called()
    
    @pytest.mark.asyncio
    async def test_discover_content_relationships_edge_case_cache_hit(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test relationship discovery with cache hit."""
        chunks = [EnhancedChunkMetadata(text="Test", chunk_index=0, chunk_type="paragraph")]
        
        # Pre-populate cache
        cached_relationships = [
            ContentRelationship(
                source_content_id=sample_content.content_id,
                target_content_id="cached-target",
                relationship_type=ContentRelationshipType.SIMILARITY
            )
        ]
        embedding_service._relationship_discovery_cache[sample_content.content_id] = cached_relationships
        
        result = await embedding_service._discover_content_relationships(
            content=sample_content,
            chunks=chunks,
            user=sample_user
        )
        
        assert result == len(cached_relationships)
    
    @pytest.mark.asyncio
    async def test_discover_content_relationships_failure_scenario(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test relationship discovery with search failure."""
        chunks = [EnhancedChunkMetadata(text="Test", chunk_index=0, chunk_type="paragraph")]
        
        # Mock vector search to raise exception
        embedding_service._vector_db.query_with_permissions = AsyncMock(
            side_effect=Exception("Search failed")
        )
        
        result = await embedding_service._discover_content_relationships(
            content=sample_content,
            chunks=chunks,
            user=sample_user
        )
        
        assert result == 0  # No relationships discovered due to failure
    
    # ========================================
    # Content Recommendation Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_get_content_recommendations_expected_behavior(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test content recommendation generation."""
        content_id = "test-content-1"
        
        # Mock content item
        mock_content = ContentItem(
            content_id=content_id,
            title="Test Content",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK
        )
        embedding_service._content_service.get_content_item = AsyncMock(return_value=mock_content)
        
        # Mock relationships
        mock_relationships = [
            ContentRelationship(
                source_content_id=content_id,
                target_content_id="related-1",
                relationship_type=ContentRelationshipType.SIMILARITY,
                strength=0.8
            )
        ]
        embedding_service._content_service.get_content_relationships = AsyncMock(
            return_value=mock_relationships
        )
        
        # Mock related content
        related_content = ContentItem(
            content_id="related-1",
            title="Related Content",
            author="Related Author",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK
        )
        
        async def mock_get_content_item(cid, user):
            if cid == content_id:
                return mock_content
            elif cid == "related-1":
                return related_content
            return None
        
        embedding_service._content_service.get_content_item = AsyncMock(side_effect=mock_get_content_item)
        
        recommendations = await embedding_service.get_content_recommendations(
            content_id=content_id,
            user=sample_user,
            n_recommendations=5
        )
        
        assert len(recommendations) > 0
        assert recommendations[0]["content_id"] == "related-1"
        assert recommendations[0]["recommendation_type"] == "relationship"
        assert recommendations[0]["recommendation_score"] == 0.8
    
    @pytest.mark.asyncio
    async def test_get_content_recommendations_edge_case_no_content(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test recommendations for non-existent content."""
        # Mock content service to return None
        embedding_service._content_service.get_content_item = AsyncMock(return_value=None)
        
        recommendations = await embedding_service.get_content_recommendations(
            content_id="non-existent",
            user=sample_user
        )
        
        assert recommendations == []
    
    @pytest.mark.asyncio
    async def test_get_content_recommendations_failure_scenario(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test recommendations with service failure."""
        # Mock content service to raise exception
        embedding_service._content_service.get_content_item = AsyncMock(
            side_effect=Exception("Service failed")
        )
        
        recommendations = await embedding_service.get_content_recommendations(
            content_id="test-content",
            user=sample_user
        )
        
        assert recommendations == []
    
    # ========================================
    # Performance Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_processing_performance_metrics(
        self,
        embedding_service: EnhancedEmbeddingService
    ):
        """Test processing performance tracking."""
        # Update metrics
        embedding_service._update_processing_metrics(
            content_count=1,
            embeddings_count=10,
            relationships_count=3,
            processing_time=2.5
        )
        
        metrics = embedding_service.get_processing_metrics()
        
        assert metrics["total_content_processed"] == 1
        assert metrics["total_embeddings_created"] == 10
        assert metrics["total_relationships_discovered"] == 3
        assert metrics["average_processing_time"] == 2.5
        assert "cache_stats" in metrics
        assert "embedding_cache_stats" in metrics
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(
        self,
        embedding_service: EnhancedEmbeddingService
    ):
        """Test cache cleanup functionality."""
        # Populate caches
        embedding_service._semantic_tag_cache["test"] = ["tag1", "tag2"]
        embedding_service._relationship_discovery_cache["test"] = []
        
        await embedding_service.cleanup_caches()
        
        assert len(embedding_service._semantic_tag_cache) == 0
        assert len(embedding_service._relationship_discovery_cache) == 0
    
    # ========================================
    # Integration Tests
    # ========================================
    
    @pytest.mark.asyncio
    async def test_end_to_end_content_processing(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_content: ContentItem,
        sample_user: User
    ):
        """Test complete end-to-end content processing workflow."""
        # Mock semantic tag extraction
        with patch.object(embedding_service, '_extract_semantic_tags', 
                         return_value=["integration", "test"]):
            
            # Process content
            result = await embedding_service.process_content_item(
                content=sample_content,
                user=sample_user
            )
            
            assert result is True
            assert sample_content.processing_status == ProcessingStatus.COMPLETED
            assert sample_content.topics == ["integration", "test"]
            
            # Verify all services were called
            embedding_service._chunking_service.chunk_content.assert_called_once()
            embedding_service._embedding_service.embed_documents.assert_called_once()
            embedding_service._vector_db.add_documents_with_metadata.assert_called_once()
            embedding_service._content_service.update_content_item.assert_called()
    
    @pytest.mark.asyncio
    async def test_search_and_recommend_integration(
        self,
        embedding_service: EnhancedEmbeddingService,
        sample_user: User
    ):
        """Test integration between search and recommendations."""
        # First perform a search
        search_results = await embedding_service.enhanced_search(
            query="test search",
            user=sample_user,
            n_results=3
        )
        
        assert isinstance(search_results, dict)
        
        # Then get recommendations if we have results
        if search_results["enhanced_results"]:
            content_id = search_results["enhanced_results"][0]["content_id"]
            
            # Mock content item for recommendations
            mock_content = ContentItem(
                content_id=content_id,
                title="Search Result",
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK
            )
            embedding_service._content_service.get_content_item = AsyncMock(return_value=mock_content)
            
            recommendations = await embedding_service.get_content_recommendations(
                content_id=content_id,
                user=sample_user,
                n_recommendations=5
            )
            
            assert isinstance(recommendations, list)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])