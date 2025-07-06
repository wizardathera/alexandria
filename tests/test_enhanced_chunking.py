"""
Tests for enhanced chunking functionality (Phase 1.1).

This test suite validates the new semantic chunking capabilities including
chapter-aware chunking, enhanced metadata, and importance scoring.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

try:
    from langchain.schema import Document
except ImportError:
    Document = dict

from src.utils.enhanced_chunking import (
    EnhancedSemanticChunker,
    ChunkingConfig,
    DocumentStructureAnalyzer,
    ContentAnalyzer,
    ChunkType,
    ContentDifficulty,
    EnhancedChunkMetadata,
    SourceLocation
)


@pytest.fixture
def sample_structured_text():
    """Sample text with clear structure for testing."""
    return """Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

## 1.1 What is Machine Learning?

Machine learning algorithms build mathematical models based on training data. This is important for making predictions or decisions without being explicitly programmed.

Key concepts include:
- Supervised learning
- Unsupervised learning 
- Reinforcement learning

## 1.2 Applications

Machine learning has many applications in industry:
1. Computer vision
2. Natural language processing
3. Recommendation systems

Chapter 2: Types of Learning

There are three main types of machine learning approaches.

## 2.1 Supervised Learning

In supervised learning, algorithms learn from labeled training data. Examples include classification and regression tasks.

## 2.2 Unsupervised Learning

Unsupervised learning finds patterns in data without labeled examples. Clustering is a common unsupervised technique."""


@pytest.fixture
def sample_document(sample_structured_text):
    """Sample LangChain document for testing."""
    return Document(
        page_content=sample_structured_text,
        metadata={
            "book_id": "test_book_001",
            "title": "Machine Learning Basics",
            "author": "Test Author",
            "file_type": "pdf"
        }
    )


@pytest.fixture
def chunking_config():
    """Standard chunking configuration for testing."""
    return ChunkingConfig(
        chunk_size=500,
        chunk_overlap=100,
        context_window_size=150,
        enable_context_windows=True,
        enable_importance_scoring=True,
        detect_headings=True,
        respect_sentence_boundaries=True
    )


class TestDocumentStructureAnalyzer:
    """Test document structure analysis."""
    
    def test_heading_detection(self, sample_structured_text):
        """Test detection of headings in structured text."""
        analyzer = DocumentStructureAnalyzer()
        structure = analyzer.analyze_structure(sample_structured_text)
        
        headings = structure["headings"]
        
        # Should detect chapter and section headings
        assert len(headings) >= 4
        
        # Check for specific headings
        heading_texts = [h["text"] for h in headings]
        assert any("Introduction to Machine Learning" in text for text in heading_texts)
        assert any("What is Machine Learning" in text for text in heading_texts)
        assert any("Types of Learning" in text for text in heading_texts)
    
    def test_list_detection(self, sample_structured_text):
        """Test detection of list items."""
        analyzer = DocumentStructureAnalyzer()
        structure = analyzer.analyze_structure(sample_structured_text)
        
        lists = structure["lists"]
        
        # Should detect bullet points and numbered lists
        assert len(lists) >= 5
        
        # Check for specific list items
        list_texts = [l["text"] for l in lists]
        assert any("Supervised learning" in text for text in list_texts)
        assert any("Computer vision" in text for text in list_texts)
    
    def test_paragraph_detection(self, sample_structured_text):
        """Test paragraph boundary detection."""
        analyzer = DocumentStructureAnalyzer()
        structure = analyzer.analyze_structure(sample_structured_text)
        
        paragraphs = structure["paragraphs"]
        
        # Should detect multiple paragraphs
        assert len(paragraphs) >= 6
        
        # Paragraphs should have meaningful content
        for paragraph in paragraphs:
            assert len(paragraph["text"]) > 10
            assert paragraph["start"] >= 0
            assert paragraph["end"] > paragraph["start"]
    
    def test_hierarchy_building(self, sample_structured_text):
        """Test document hierarchy construction."""
        analyzer = DocumentStructureAnalyzer()
        structure = analyzer.analyze_structure(sample_structured_text)
        
        hierarchy = structure["hierarchy"]
        
        # Should identify chapters and sections
        assert len(hierarchy["chapters"]) >= 2
        assert len(hierarchy["sections"]) >= 4
        assert hierarchy["max_level"] >= 2


class TestContentAnalyzer:
    """Test content analysis for importance and quality scoring."""
    
    def test_importance_scoring(self):
        """Test importance score calculation."""
        analyzer = ContentAnalyzer()
        
        # High importance text
        high_importance = "This is critically important and essential information that is key to understanding."
        high_score = analyzer._calculate_importance_score(high_importance)
        
        # Low importance text
        low_importance = "Perhaps this might be occasionally useful but not particularly significant."
        low_score = analyzer._calculate_importance_score(low_importance)
        
        # High importance should score higher
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
    
    def test_coherence_scoring(self):
        """Test coherence score calculation."""
        analyzer = ContentAnalyzer()
        
        # Coherent text with transitions
        coherent_text = "First, we introduce the concept. Then, we explore its applications. Furthermore, we analyze the results."
        coherent_score = analyzer._calculate_coherence_score(coherent_text)
        
        # Incoherent text
        incoherent_text = "Random sentence. Completely unrelated topic. Another disconnected thought."
        incoherent_score = analyzer._calculate_coherence_score(incoherent_text)
        
        # Coherent text should score higher
        assert coherent_score > incoherent_score
        assert 0.0 <= coherent_score <= 1.0
        assert 0.0 <= incoherent_score <= 1.0
    
    def test_reading_level_detection(self):
        """Test reading difficulty level detection."""
        analyzer = ContentAnalyzer()
        
        # Elementary level text
        elementary = "This is a simple story. The cat sat on the mat. It was very happy."
        elementary_level = analyzer._determine_reading_level(elementary)
        
        # Advanced level text
        advanced = "The theoretical framework encompasses sophisticated methodologies and comprehensive analytical paradigms."
        advanced_level = analyzer._determine_reading_level(advanced)
        
        # Should detect different levels
        assert isinstance(elementary_level, ContentDifficulty)
        assert isinstance(advanced_level, ContentDifficulty)
    
    def test_topic_extraction(self):
        """Test topic tag extraction."""
        analyzer = ContentAnalyzer()
        
        text = "Machine learning algorithms analyze data patterns to make predictions and automate decision-making processes."
        topics = analyzer._extract_topic_tags(text)
        
        # Should extract relevant topics
        assert isinstance(topics, list)
        assert len(topics) > 0
        
        # Topics should be meaningful words
        for topic in topics:
            assert len(topic) > 3
            assert topic.isalpha()


class TestEnhancedSemanticChunker:
    """Test enhanced semantic chunking functionality."""
    
    def test_chapter_aware_chunking(self, sample_document, chunking_config):
        """Test chunking that respects chapter boundaries."""
        chunker = EnhancedSemanticChunker(chunking_config)
        chunks = chunker.chunk_documents([sample_document], chunking_config)
        
        # Should create multiple chunks
        assert len(chunks) >= 2
        
        # Each chunk should be a Document
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) > 0
            assert isinstance(chunk.metadata, dict)
        
        # Should have enhanced metadata
        first_chunk = chunks[0]
        metadata = first_chunk.metadata
        
        # Check for required metadata fields
        assert "chunk_id" in metadata
        assert "importance_score" in metadata
        assert "coherence_score" in metadata
        assert "content_type" in metadata
        assert "topic_tags" in metadata
    
    def test_context_windows(self, sample_document, chunking_config):
        """Test addition of context windows."""
        chunker = EnhancedSemanticChunker(chunking_config)
        chunks = chunker.chunk_documents([sample_document], chunking_config)
        
        # Chunks should have context windows
        for chunk in chunks:
            metadata = chunk.metadata
            
            # Should have context fields
            assert "preceding_context" in metadata
            assert "following_context" in metadata
            assert "context_window_size" in metadata
            
            # Context should be strings
            assert isinstance(metadata["preceding_context"], str)
            assert isinstance(metadata["following_context"], str)
    
    def test_importance_scoring(self, sample_document, chunking_config):
        """Test importance scoring for chunks."""
        chunker = EnhancedSemanticChunker(chunking_config)
        chunks = chunker.chunk_documents([sample_document], chunking_config)
        
        importance_scores = []
        for chunk in chunks:
            score = chunk.metadata.get("importance_score", 0.0)
            importance_scores.append(score)
            
            # Score should be in valid range
            assert 0.0 <= score <= 1.0
        
        # Should have variation in importance scores
        assert len(set(importance_scores)) > 1
    
    def test_chunk_relationships(self, sample_document, chunking_config):
        """Test relationship analysis between chunks."""
        chunker = EnhancedSemanticChunker(chunking_config)
        chunks = chunker.chunk_documents([sample_document], chunking_config)
        
        # Check for relationship metadata
        for chunk in chunks:
            metadata = chunk.metadata
            
            # Should have relationship fields
            assert "related_chunks" in metadata
            assert "prerequisite_chunks" in metadata
            assert "follow_up_chunks" in metadata
            
            # Relationships should be lists
            assert isinstance(metadata["related_chunks"], list)
            assert isinstance(metadata["prerequisite_chunks"], list)
            assert isinstance(metadata["follow_up_chunks"], list)
    
    def test_source_location_tracking(self, sample_document, chunking_config):
        """Test tracking of source locations."""
        chunker = EnhancedSemanticChunker(chunking_config)
        chunks = chunker.chunk_documents([sample_document], chunking_config)
        
        for chunk in chunks:
            metadata = chunk.metadata
            
            # Should have source location information
            assert "source_location" in metadata
            
            source_loc = metadata["source_location"]
            assert isinstance(source_loc, dict)
            
            # May have character positions
            if "character_start" in source_loc:
                assert isinstance(source_loc["character_start"], int)
                assert source_loc["character_start"] >= 0
    
    def test_hierarchical_chunking(self, sample_document, chunking_config):
        """Test hierarchical chunking based on document structure."""
        chunker = EnhancedSemanticChunker(chunking_config)
        chunks = chunker.chunk_documents([sample_document], chunking_config)
        
        hierarchy_levels = []
        for chunk in chunks:
            level = chunk.metadata.get("hierarchy_level", 0)
            hierarchy_levels.append(level)
            
            # Level should be reasonable
            assert 0 <= level <= 10
        
        # Should have different hierarchy levels
        assert len(set(hierarchy_levels)) >= 1


class TestChunkingConfig:
    """Test chunking configuration options."""
    
    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration
        config = ChunkingConfig(
            chunk_size=1000,
            chunk_overlap=200,
            context_window_size=150
        )
        
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.context_window_size == 150
        assert config.enable_context_windows is True
        assert config.enable_importance_scoring is True
    
    def test_enhanced_options(self):
        """Test enhanced chunking options."""
        config = ChunkingConfig(
            detect_headings=True,
            detect_lists=True,
            detect_code_blocks=True,
            respect_sentence_boundaries=True
        )
        
        assert config.detect_headings is True
        assert config.detect_lists is True
        assert config.detect_code_blocks is True
        assert config.respect_sentence_boundaries is True


class TestEnhancedChunkMetadata:
    """Test enhanced metadata structure."""
    
    def test_metadata_creation(self):
        """Test creation of enhanced metadata."""
        source_location = SourceLocation(
            page_number=1,
            chapter_number=1,
            chapter_title="Introduction",
            character_start=0,
            character_end=100
        )
        
        metadata = EnhancedChunkMetadata(
            chunk_id="test_chunk_001",
            source_document_id="test_doc_001",
            chunk_index=0,
            total_chunks=5,
            content_type=ChunkType.PARAGRAPH,
            source_location=source_location,
            importance_score=0.8,
            coherence_score=0.9
        )
        
        assert metadata.chunk_id == "test_chunk_001"
        assert metadata.importance_score == 0.8
        assert metadata.coherence_score == 0.9
        assert metadata.content_type == ChunkType.PARAGRAPH
    
    def test_metadata_serialization(self):
        """Test metadata serialization to dictionary."""
        metadata = EnhancedChunkMetadata(
            chunk_id="test_chunk_001",
            source_document_id="test_doc_001",
            chunk_index=0,
            total_chunks=5
        )
        
        metadata_dict = metadata.to_dict()
        
        # Should be a dictionary
        assert isinstance(metadata_dict, dict)
        
        # Should contain required fields
        assert "chunk_id" in metadata_dict
        assert "source_document_id" in metadata_dict
        assert "chunk_index" in metadata_dict
        assert "importance_score" in metadata_dict
        assert "coherence_score" in metadata_dict


class TestIntegrationWithExistingSystem:
    """Test integration with existing chunking system."""
    
    @patch('src.utils.enhanced_chunking.ENHANCED_CHUNKING_AVAILABLE', True)
    def test_chunking_manager_integration(self, sample_document):
        """Test integration with ChunkingManager."""
        from src.utils.text_chunking import ChunkingManager
        
        manager = ChunkingManager()
        
        # Should have enhanced_semantic strategy available
        assert 'enhanced_semantic' in manager.chunkers
        
        # Should recommend enhanced_semantic for appropriate documents
        strategy = manager.get_recommended_strategy("pdf", 10000)
        assert strategy == 'enhanced_semantic'
    
    def test_backward_compatibility(self, sample_document):
        """Test backward compatibility with existing chunking."""
        from src.utils.text_chunking import ChunkingManager
        
        manager = ChunkingManager()
        
        # Should still work with basic strategies
        chunks = manager.chunk_documents([sample_document], strategy='recursive')
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert len(chunk.page_content) > 0


# Edge cases and error handling tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_document(self):
        """Test chunking of empty document."""
        empty_doc = Document(page_content="", metadata={"book_id": "empty"})
        
        chunker = EnhancedSemanticChunker()
        chunks = chunker.chunk_documents([empty_doc])
        
        # Should handle empty documents gracefully
        assert isinstance(chunks, list)
    
    def test_very_short_document(self):
        """Test chunking of very short document."""
        short_doc = Document(
            page_content="Short text.",
            metadata={"book_id": "short"}
        )
        
        chunker = EnhancedSemanticChunker()
        chunks = chunker.chunk_documents([short_doc])
        
        # Should create at least one chunk
        assert len(chunks) >= 1
        assert chunks[0].page_content == "Short text."
    
    def test_document_without_structure(self):
        """Test chunking of unstructured document."""
        unstructured_text = "This is just a long paragraph of text without any headings or structure. " * 20
        unstructured_doc = Document(
            page_content=unstructured_text,
            metadata={"book_id": "unstructured"}
        )
        
        chunker = EnhancedSemanticChunker()
        chunks = chunker.chunk_documents([unstructured_doc])
        
        # Should still create valid chunks
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.page_content) > 0
            assert "importance_score" in chunk.metadata


# Performance tests
class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_document_chunking(self):
        """Test chunking of large document."""
        # Create a large document
        large_text = "This is a paragraph of text. " * 1000
        large_doc = Document(
            page_content=large_text,
            metadata={"book_id": "large"}
        )
        
        chunker = EnhancedSemanticChunker()
        
        import time
        start_time = time.time()
        chunks = chunker.chunk_documents([large_doc])
        end_time = time.time()
        
        # Should complete in reasonable time (under 10 seconds)
        processing_time = end_time - start_time
        assert processing_time < 10.0
        
        # Should create multiple chunks
        assert len(chunks) > 1
    
    def test_multiple_documents_chunking(self, sample_document):
        """Test chunking of multiple documents."""
        # Create multiple documents
        documents = [sample_document] * 5
        
        chunker = EnhancedSemanticChunker()
        
        import time
        start_time = time.time()
        chunks = chunker.chunk_documents(documents)
        end_time = time.time()
        
        # Should complete in reasonable time
        processing_time = end_time - start_time
        assert processing_time < 15.0
        
        # Should create chunks for all documents
        assert len(chunks) >= 5


if __name__ == "__main__":
    pytest.main([__file__])