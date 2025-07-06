"""
Comprehensive tests for the book ingestion pipeline.

This module tests all components of the ingestion system:
document loaders, text chunking, embeddings, and the full pipeline.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import modules under test
from src.utils.document_loaders import (
    DocumentLoaderManager, 
    PDFLoader, 
    EPUBLoader, 
    DOCLoader, 
    TXTLoader, 
    HTMLLoader
)
from src.utils.text_chunking import (
    ChunkingManager, 
    RecursiveTextChunker, 
    SemanticTextChunker,
    ChunkingConfig
)
from src.utils.embeddings import EmbeddingService, EmbeddingMetrics, OpenAIEmbeddingProvider
from src.services.ingestion import BookIngestionService, IngestionStatus, BookMetadata


# Mock Document class for testing
class MockDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


@pytest.fixture
def mock_document():
    """Create a mock document for testing."""
    return MockDocument(
        page_content="This is test content for a mock document.",
        metadata={"source": "test.txt", "file_type": "txt"}
    )


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content for file processing.\n" * 100)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_settings():
    """Mock application settings."""
    settings = Mock()
    settings.openai_api_key = "test-key"
    settings.chroma_collection_name = "test_collection"
    settings.user_data_path = "./test_data"
    settings.books_storage_path = "./test_books"
    return settings


class TestDocumentLoaders:
    """Test document loader functionality."""
    
    def test_txt_loader_can_load(self):
        """
        Test TXT loader correctly identifies text files.
        
        Expected behavior: Should return True for .txt files.
        """
        loader = TXTLoader()
        assert loader.can_load(Path("test.txt"))
        assert not loader.can_load(Path("test.pdf"))
    
    def test_pdf_loader_can_load(self):
        """
        Test PDF loader correctly identifies PDF files.
        
        Expected behavior: Should return True for .pdf files.
        """
        loader = PDFLoader()
        assert loader.can_load(Path("test.pdf"))
        assert not loader.can_load(Path("test.txt"))
    
    def test_epub_loader_can_load(self):
        """
        Test EPUB loader correctly identifies EPUB files.
        
        Edge case: Should handle case-insensitive extensions.
        """
        loader = EPUBLoader()
        assert loader.can_load(Path("test.epub"))
        assert loader.can_load(Path("test.EPUB"))
        assert not loader.can_load(Path("test.pdf"))
    
    def test_doc_loader_can_load(self):
        """
        Test DOC loader correctly identifies Word documents.
        
        Expected behavior: Should handle both .doc and .docx.
        """
        loader = DOCLoader()
        assert loader.can_load(Path("test.doc"))
        assert loader.can_load(Path("test.docx"))
        assert not loader.can_load(Path("test.txt"))
    
    def test_html_loader_can_load(self):
        """
        Test HTML loader correctly identifies HTML files.
        
        Expected behavior: Should handle both .html and .htm.
        """
        loader = HTMLLoader()
        assert loader.can_load(Path("test.html"))
        assert loader.can_load(Path("test.htm"))
        assert not loader.can_load(Path("test.txt"))
    
    def test_document_loader_manager_get_loader(self):
        """
        Test document loader manager returns correct loader.
        
        Expected behavior: Should return appropriate loader for each format.
        """
        manager = DocumentLoaderManager()
        
        txt_loader = manager.get_loader_for_file(Path("test.txt"))
        assert isinstance(txt_loader, TXTLoader)
        
        pdf_loader = manager.get_loader_for_file(Path("test.pdf"))
        assert isinstance(pdf_loader, PDFLoader)
        
        unsupported_loader = manager.get_loader_for_file(Path("test.xyz"))
        assert unsupported_loader is None
    
    def test_document_loader_manager_supported_extensions(self):
        """
        Test document loader manager returns all supported extensions.
        
        Expected behavior: Should include all format extensions.
        """
        manager = DocumentLoaderManager()
        extensions = manager.get_supported_extensions()
        
        expected_extensions = ['.pdf', '.epub', '.doc', '.docx', '.txt', '.html', '.htm']
        for ext in expected_extensions:
            assert ext in extensions
    
    def test_document_loader_unsupported_format(self):
        """
        Test error handling for unsupported file formats.
        
        Failure scenario: Should raise ValueError for unknown formats.
        """
        manager = DocumentLoaderManager()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            manager.load_document(Path("test.xyz"))


class TestTextChunking:
    """Test text chunking functionality."""
    
    def test_chunking_config_defaults(self):
        """
        Test chunking configuration default values.
        
        Expected behavior: Should have sensible defaults.
        """
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.separator == "\n\n"
    
    def test_recursive_chunker_get_optimal_config(self):
        """
        Test recursive chunker returns format-specific configs.
        
        Expected behavior: Different configs for different formats.
        """
        chunker = RecursiveTextChunker()
        
        pdf_config = chunker.get_optimal_config('pdf')
        epub_config = chunker.get_optimal_config('epub')
        txt_config = chunker.get_optimal_config('txt')
        
        # PDF should have larger chunks
        assert pdf_config.chunk_size > txt_config.chunk_size
        # EPUB should have the largest chunks
        assert epub_config.chunk_size >= pdf_config.chunk_size
    
    @patch('src.utils.text_chunking.RecursiveCharacterTextSplitter')
    def test_recursive_chunker_chunk_documents(self, mock_splitter_class):
        """
        Test recursive chunker processes documents correctly.
        
        Expected behavior: Should split documents and add metadata.
        """
        # Setup mock
        mock_splitter = Mock()
        mock_splitter_class.return_value = mock_splitter
        
        # Create mock split results
        split_doc = MockDocument(
            "Chunked content",
            {"source": "test.txt"}
        )
        mock_splitter.split_documents.return_value = [split_doc]
        
        # Test chunking
        chunker = RecursiveTextChunker()
        documents = [MockDocument("Original content", {"file_type": "txt"})]
        
        result = chunker.chunk_documents(documents)
        
        # Verify result
        assert len(result) == 1
        assert result[0].page_content == "Chunked content"
        assert "chunk_index" in result[0].metadata
        assert result[0].metadata["chunking_method"] == "recursive_character"
    
    def test_semantic_chunker_split_by_paragraphs(self):
        """
        Test semantic chunker respects paragraph boundaries.
        
        Edge case: Should handle mixed paragraph and sentence splitting.
        """
        chunker = SemanticTextChunker()
        
        text = "First paragraph.\n\nSecond paragraph with multiple sentences. This is another sentence.\n\nThird paragraph."
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        
        chunks = chunker._semantic_split(text, config)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should be reasonably sized
        for chunk in chunks:
            assert len(chunk) <= config.chunk_size + 50  # Allow some flexibility
    
    def test_chunking_manager_strategy_selection(self):
        """
        Test chunking manager selects appropriate strategy.
        
        Expected behavior: Should choose strategy based on document type and length.
        """
        manager = ChunkingManager()
        
        # Short text document should use recursive
        strategy = manager.get_recommended_strategy("txt", 5000)
        assert strategy == "recursive"
        
        # Long EPUB should use semantic
        strategy = manager.get_recommended_strategy("epub", 50000)
        assert strategy == "semantic"
    
    def test_chunking_manager_estimate_chunk_count(self):
        """
        Test chunk count estimation.
        
        Expected behavior: Should provide reasonable estimates.
        """
        manager = ChunkingManager()
        
        # Test various text lengths
        small_count = manager.estimate_chunk_count(500, 1000, 200)
        large_count = manager.estimate_chunk_count(5000, 1000, 200)
        
        assert small_count == 1
        assert large_count > small_count


class TestEmbeddings:
    """Test embedding service functionality."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        client = Mock()
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = Mock(total_tokens=10)
        
        client.embeddings.create.return_value = mock_response
        return client
    
    @patch('src.utils.embeddings.OpenAI')
    def test_openai_provider_initialization(self, mock_openai_class, mock_settings):
        """
        Test OpenAI provider initializes correctly.
        
        Expected behavior: Should create client with API key.
        """
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            provider = OpenAIEmbeddingProvider()
            assert provider.client == mock_client
    
    @pytest.mark.asyncio
    async def test_openai_provider_generate_embeddings(self, mock_openai_client, mock_settings):
        """
        Test OpenAI provider generates embeddings correctly.
        
        Expected behavior: Should return embeddings and metrics.
        """
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            provider = OpenAIEmbeddingProvider()
            provider.client = mock_openai_client
            
            texts = ["Test text 1", "Test text 2"]
            embeddings, metrics = await provider.generate_embeddings(texts)
            
            assert len(embeddings) == 2
            assert all(len(emb) == 3 for emb in embeddings if emb)  # 3D embeddings from mock
            assert metrics.total_requests > 0
    
    @pytest.mark.asyncio
    async def test_openai_provider_empty_texts(self, mock_openai_client, mock_settings):
        """
        Test OpenAI provider handles empty text list.
        
        Edge case: Should handle empty input gracefully.
        """
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            provider = OpenAIEmbeddingProvider()
            provider.client = mock_openai_client
            
            embeddings, metrics = await provider.generate_embeddings([])
            
            assert embeddings == []
            assert metrics.total_requests == 0
    
    @pytest.mark.asyncio
    async def test_openai_provider_api_error(self, mock_settings):
        """
        Test OpenAI provider handles API errors.
        
        Failure scenario: Should handle API failures gracefully.
        """
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            provider = OpenAIEmbeddingProvider()
            provider.client = mock_client
            
            with pytest.raises(ValueError, match="Embedding generation failed"):
                await provider.generate_embeddings(["Test text"])
    
    def test_embedding_metrics_cost_calculation(self):
        """
        Test embedding metrics calculates costs correctly.
        
        Expected behavior: Should calculate cost based on token usage.
        """
        metrics = EmbeddingMetrics()
        metrics.total_tokens = 1000
        
        cost = metrics.get_cost_estimate("text-embedding-ada-002")
        assert cost == 0.0001  # 1000 tokens * $0.0001 per 1K tokens
    
    @pytest.mark.asyncio
    async def test_embedding_service_embed_documents(self, mock_settings):
        """
        Test embedding service processes documents correctly.
        
        Expected behavior: Should extract text and generate embeddings.
        """
        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.generate_embeddings.return_value = ([[0.1, 0.2]], EmbeddingMetrics())
        
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            with patch('src.utils.embeddings.CachedEmbeddingProvider') as mock_cached:
                mock_cached.return_value = mock_provider
                
                service = EmbeddingService()
                documents = [MockDocument("Test content")]
                
                embeddings, metrics = await service.embed_documents(documents)
                
                assert len(embeddings) == 1
                assert embeddings[0] == [0.1, 0.2]
    
    @pytest.mark.asyncio
    async def test_embedding_service_embed_query(self, mock_settings):
        """
        Test embedding service handles single queries.
        
        Expected behavior: Should embed single query text.
        """
        mock_provider = AsyncMock()
        mock_provider.generate_embeddings.return_value = ([[0.1, 0.2]], EmbeddingMetrics())
        
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            with patch('src.utils.embeddings.CachedEmbeddingProvider') as mock_cached:
                mock_cached.return_value = mock_provider
                
                service = EmbeddingService()
                embedding = await service.embed_query("Test query")
                
                assert embedding == [0.1, 0.2]
    
    @pytest.mark.asyncio
    async def test_embedding_service_empty_query(self, mock_settings):
        """
        Test embedding service rejects empty queries.
        
        Failure scenario: Should raise error for empty query.
        """
        with patch('src.utils.embeddings.get_settings', return_value=mock_settings):
            service = EmbeddingService()
            
            with pytest.raises(ValueError, match="Query cannot be empty"):
                await service.embed_query("")


class TestBookIngestionService:
    """Test complete book ingestion pipeline."""
    
    @pytest.fixture
    def mock_ingestion_service(self, mock_settings):
        """Create mock ingestion service with dependencies."""
        with patch('src.services.ingestion.get_settings', return_value=mock_settings):
            service = BookIngestionService()
            
            # Mock dependencies
            service.document_loader = Mock()
            service.chunking_manager = Mock()
            service.embedding_service = AsyncMock()
            
            return service
    
    @pytest.mark.asyncio
    async def test_ingestion_service_full_pipeline(self, mock_ingestion_service, temp_file):
        """
        Test complete ingestion pipeline.
        
        Expected behavior: Should process file through all stages.
        """
        # Setup mocks
        mock_docs = [MockDocument("Test content", {"file_type": "txt"})]
        mock_metadata = {"title": "Test Book", "author": "Test Author"}
        
        mock_ingestion_service.document_loader.load_document.return_value = (mock_docs, mock_metadata)
        mock_ingestion_service.chunking_manager.chunk_documents.return_value = mock_docs
        mock_ingestion_service.chunking_manager.get_recommended_strategy.return_value = "recursive"
        mock_ingestion_service.embedding_service.embed_documents.return_value = ([[0.1, 0.2]], EmbeddingMetrics())
        
        # Mock database
        mock_db = AsyncMock()
        mock_db.add_documents.return_value = True
        
        with patch('src.services.ingestion.get_database', return_value=mock_db):
            with patch('src.services.ingestion.asyncio.to_thread', side_effect=lambda f, *args: f(*args)):
                # Create metadata
                metadata = BookMetadata(
                    book_id="test-id",
                    title="Test Book",
                    file_type="txt",
                    file_name=temp_file.name,
                    file_path=str(temp_file),
                    file_size=100
                )
                
                # Run ingestion
                progress = await mock_ingestion_service.ingest_book("test-id", temp_file, metadata)
                
                # Verify completion
                assert progress.status == IngestionStatus.COMPLETED
                assert progress.progress == 1.0
                assert progress.error is None
    
    @pytest.mark.asyncio
    async def test_ingestion_service_handles_errors(self, mock_ingestion_service, temp_file):
        """
        Test ingestion service handles errors gracefully.
        
        Failure scenario: Should set error status on failure.
        """
        # Setup mock to raise error
        mock_ingestion_service.document_loader.load_document.side_effect = Exception("Load failed")
        
        with patch('src.services.ingestion.asyncio.to_thread', side_effect=lambda f, *args: f(*args)):
            # Run ingestion
            progress = await mock_ingestion_service.ingest_book("test-id", temp_file)
            
            # Verify error handling
            assert progress.status == IngestionStatus.FAILED
            assert progress.error is not None
            assert "Load failed" in progress.error
    
    def test_ingestion_service_progress_tracking(self, mock_ingestion_service):
        """
        Test ingestion service tracks progress correctly.
        
        Expected behavior: Should update progress through callbacks.
        """
        callback_calls = []
        
        def progress_callback(progress):
            callback_calls.append((progress.status, progress.progress))
        
        mock_ingestion_service.add_progress_callback(progress_callback)
        
        # Simulate progress update
        from src.services.ingestion import IngestionProgress
        progress = IngestionProgress("test-id")
        progress.update_status(IngestionStatus.LOADING, 0.1, "Loading file")
        
        mock_ingestion_service._notify_progress(progress)
        
        # Verify callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0] == (IngestionStatus.LOADING, 0.1)
    
    def test_ingestion_service_get_status(self, mock_ingestion_service):
        """
        Test ingestion service status retrieval.
        
        Expected behavior: Should return current status for book.
        """
        # Add mock progress
        from src.services.ingestion import IngestionProgress
        progress = IngestionProgress("test-id")
        mock_ingestion_service.active_ingestions["test-id"] = progress
        
        # Retrieve status
        retrieved_progress = mock_ingestion_service.get_ingestion_status("test-id")
        
        assert retrieved_progress == progress
        
        # Test non-existent book
        missing_progress = mock_ingestion_service.get_ingestion_status("missing-id")
        assert missing_progress is None
    
    @pytest.mark.asyncio
    async def test_ingestion_service_cancel_ingestion(self, mock_ingestion_service):
        """
        Test ingestion cancellation.
        
        Edge case: Should handle cancellation of active ingestion.
        """
        # Add active ingestion
        from src.services.ingestion import IngestionProgress
        progress = IngestionProgress("test-id")
        mock_ingestion_service.active_ingestions["test-id"] = progress
        
        # Cancel ingestion
        success = await mock_ingestion_service.cancel_ingestion("test-id")
        
        assert success
        assert progress.status == IngestionStatus.FAILED
        assert "cancelled" in progress.error.lower()
        assert "test-id" not in mock_ingestion_service.active_ingestions
        assert "test-id" in mock_ingestion_service.completed_ingestions
    
    @pytest.mark.asyncio
    async def test_ingestion_service_delete_book(self, mock_ingestion_service):
        """
        Test book deletion.
        
        Expected behavior: Should remove all book data.
        """
        # Mock database
        mock_db = AsyncMock()
        mock_db.query.return_value = {"ids": ["chunk1", "chunk2"]}
        
        with patch('src.services.ingestion.get_database', return_value=mock_db):
            with patch('src.services.ingestion.asyncio.to_thread'):
                with patch('src.services.ingestion.Path') as mock_path:
                    mock_file = Mock()
                    mock_file.exists.return_value = True
                    mock_path.return_value = mock_file
                    
                    # Delete book
                    success = await mock_ingestion_service.delete_book("test-id")
                    
                    assert success
                    # Verify cleanup calls
                    mock_file.unlink.assert_called_once()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])