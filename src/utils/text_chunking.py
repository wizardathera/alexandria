"""
Text chunking utilities for the DBC application.

This module provides intelligent text chunking strategies that maintain
context and semantic coherence across different document types.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
from datetime import datetime

try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        MarkdownTextSplitter
    )
    from langchain.schema import Document
except ImportError:
    # Fallback for testing without full installation
    Document = dict
    RecursiveCharacterTextSplitter = object
    TokenTextSplitter = object
    MarkdownTextSplitter = object

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)

# Import enhanced chunking capabilities
try:
    from src.utils.enhanced_chunking import (
        EnhancedSemanticChunker, 
        ChunkingConfig as EnhancedChunkingConfig,
        ChunkType,
        ContentDifficulty
    )
    ENHANCED_CHUNKING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced chunking not available: {e}")
    EnhancedSemanticChunker = None
    EnhancedChunkingConfig = None
    ChunkType = None
    ContentDifficulty = None
    ENHANCED_CHUNKING_AVAILABLE = False


@dataclass
class ChunkingConfig:
    """Configuration for text chunking parameters."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separator: str = "\n\n"
    keep_separator: bool = True
    is_separator_regex: bool = False
    length_function: str = "len"  # "len" or "tiktoken"


class TextChunkerInterface(ABC):
    """
    Abstract interface for text chunking strategies.
    
    This interface ensures consistent chunking behavior across different
    document types while allowing for format-specific optimizations.
    """
    
    @abstractmethod
    def chunk_documents(
        self, 
        documents: List[Document], 
        config: Optional[ChunkingConfig] = None
    ) -> List[Document]:
        """Chunk documents into smaller, semantically coherent pieces."""
        pass
    
    @abstractmethod
    def get_optimal_config(self, document_type: str) -> ChunkingConfig:
        """Get optimal chunking configuration for document type."""
        pass


class RecursiveTextChunker(TextChunkerInterface):
    """
    Recursive character text chunker.
    
    Uses LangChain's RecursiveCharacterTextSplitter for intelligent
    text splitting that preserves semantic structure.
    """
    
    def __init__(self):
        """Initialize the recursive text chunker."""
        self.settings = get_settings()
    
    def chunk_documents(
        self, 
        documents: List[Document], 
        config: Optional[ChunkingConfig] = None
    ) -> List[Document]:
        """
        Chunk documents using recursive character splitting.
        
        Args:
            documents: List of documents to chunk
            config: Optional chunking configuration
            
        Returns:
            List[Document]: List of chunked documents
        """
        if not config:
            # Use default config based on first document's type
            doc_type = documents[0].metadata.get('file_type', 'text') if documents else 'text'
            config = self.get_optimal_config(doc_type)
        
        logger.info(f"Chunking {len(documents)} documents with config: {config}")
        
        try:
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=self._get_separators_for_config(config),
                keep_separator=config.keep_separator,
                is_separator_regex=config.is_separator_regex,
                length_function=len if config.length_function == "len" else self._token_length
            )
            
            # Split documents
            chunked_docs = []
            for i, doc in enumerate(documents):
                chunks = text_splitter.split_documents([doc])
                
                # Add chunk-specific metadata
                for j, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'source_document_index': i,
                        'chunk_size': len(chunk.page_content),
                        'chunking_method': 'recursive_character'
                    })
                
                chunked_docs.extend(chunks)
            
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            raise ValueError(f"Text chunking failed: {e}")
    
    def get_optimal_config(self, document_type: str) -> ChunkingConfig:
        """
        Get optimal chunking configuration for different document types.
        
        Args:
            document_type: Type of document (pdf, epub, txt, etc.)
            
        Returns:
            ChunkingConfig: Optimized configuration for the document type
        """
        configs = {
            'pdf': ChunkingConfig(
                chunk_size=1200,
                chunk_overlap=300,
                separator="\n\n"
            ),
            'epub': ChunkingConfig(
                chunk_size=1500,
                chunk_overlap=250,
                separator="\n\n"
            ),
            'doc': ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n\n"
            ),
            'docx': ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n\n"
            ),
            'txt': ChunkingConfig(
                chunk_size=800,
                chunk_overlap=150,
                separator="\n\n"
            ),
            'html': ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                separator="\n\n"
            )
        }
        
        return configs.get(document_type, ChunkingConfig())
    
    def _get_separators_for_config(self, config: ChunkingConfig) -> List[str]:
        """
        Get separator list for recursive splitting.
        
        Args:
            config: Chunking configuration
            
        Returns:
            List[str]: List of separators in priority order
        """
        # Default separators for recursive splitting
        return [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            " ",     # Word breaks
            ""       # Character breaks
        ]
    
    def _token_length(self, text: str) -> int:
        """
        Calculate text length in tokens (approximate).
        
        Args:
            text: Text to measure
            
        Returns:
            int: Approximate token count
        """
        # Simple approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4


class SemanticTextChunker(TextChunkerInterface):
    """
    Semantic text chunker that attempts to preserve meaning.
    
    This chunker uses sentence boundaries and paragraph structure
    to create more coherent chunks.
    """
    
    def __init__(self):
        """Initialize the semantic text chunker."""
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
    def chunk_documents(
        self, 
        documents: List[Document], 
        config: Optional[ChunkingConfig] = None
    ) -> List[Document]:
        """
        Chunk documents using semantic boundaries.
        
        Args:
            documents: List of documents to chunk
            config: Optional chunking configuration
            
        Returns:
            List[Document]: List of semantically chunked documents
        """
        if not config:
            doc_type = documents[0].metadata.get('file_type', 'text') if documents else 'text'
            config = self.get_optimal_config(doc_type)
        
        logger.info(f"Semantic chunking {len(documents)} documents")
        
        chunked_docs = []
        for i, doc in enumerate(documents):
            chunks = self._semantic_split(doc.page_content, config)
            
            for j, chunk_text in enumerate(chunks):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_index': j,
                        'total_chunks': len(chunks),
                        'source_document_index': i,
                        'chunk_size': len(chunk_text),
                        'chunking_method': 'semantic'
                    }
                )
                chunked_docs.append(chunk_doc)
        
        logger.info(f"Created {len(chunked_docs)} semantic chunks")
        return chunked_docs
    
    def get_optimal_config(self, document_type: str) -> ChunkingConfig:
        """Get optimal semantic chunking configuration."""
        # Semantic chunking uses slightly larger chunks to preserve context
        configs = {
            'pdf': ChunkingConfig(chunk_size=1500, chunk_overlap=300),
            'epub': ChunkingConfig(chunk_size=2000, chunk_overlap=400),
            'doc': ChunkingConfig(chunk_size=1200, chunk_overlap=250),
            'docx': ChunkingConfig(chunk_size=1200, chunk_overlap=250),
            'txt': ChunkingConfig(chunk_size=1000, chunk_overlap=200),
            'html': ChunkingConfig(chunk_size=1200, chunk_overlap=250)
        }
        
        return configs.get(document_type, ChunkingConfig(chunk_size=1200, chunk_overlap=250))
    
    def _semantic_split(self, text: str, config: ChunkingConfig) -> List[str]:
        """
        Split text using semantic boundaries.
        
        Args:
            text: Text to split
            config: Chunking configuration
            
        Returns:
            List[str]: List of semantically coherent chunks
        """
        # First, split by paragraphs
        paragraphs = self.paragraph_breaks.split(text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    current_chunk = self._get_overlap_text(current_chunk, config.chunk_overlap)
                
                # If paragraph itself is too long, split by sentences
                if len(paragraph) > config.chunk_size:
                    sentences = self._split_by_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > config.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = self._get_overlap_text(current_chunk, config.chunk_overlap)
                        current_chunk += " " + sentence if current_chunk else sentence
                else:
                    current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a good breaking point near the overlap size
        overlap_text = text[-overlap_size:]
        
        # Find the last sentence boundary in the overlap
        sentences = self._split_by_sentences(overlap_text)
        if len(sentences) > 1:
            return sentences[-1]
        
        return overlap_text


class ChunkingManager:
    """
    Manager for text chunking operations.
    
    Provides a unified interface for different chunking strategies
    and automatic strategy selection based on document type.
    Enhanced with Phase 1.1 semantic chunking capabilities.
    """
    
    def __init__(self):
        """Initialize the chunking manager."""
        self.chunkers = {
            'recursive': RecursiveTextChunker(),
            'semantic': SemanticTextChunker()
        }
        
        # Add enhanced chunking if available
        if ENHANCED_CHUNKING_AVAILABLE:
            self.chunkers['enhanced_semantic'] = EnhancedSemanticChunker()
            self.default_strategy = 'enhanced_semantic'
            logger.info("Enhanced semantic chunking enabled (Phase 1.1)")
        else:
            self.default_strategy = 'recursive'
            logger.info("Using basic chunking strategies")
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: str = None,
        config: Optional[ChunkingConfig] = None
    ) -> List[Document]:
        """
        Chunk documents using the specified strategy.
        
        Args:
            documents: Documents to chunk
            strategy: Chunking strategy ('recursive' or 'semantic')
            config: Optional chunking configuration
            
        Returns:
            List[Document]: Chunked documents
        """
        if not documents:
            return []
        
        strategy = strategy or self.default_strategy
        
        if strategy not in self.chunkers:
            logger.warning(f"Unknown chunking strategy: {strategy}. Using default: {self.default_strategy}")
            strategy = self.default_strategy
        
        chunker = self.chunkers[strategy]
        
        try:
            chunked_docs = chunker.chunk_documents(documents, config)
            
            # Add global metadata
            for doc in chunked_docs:
                doc.metadata['chunking_strategy'] = strategy
                doc.metadata['chunking_timestamp'] = str(datetime.now())
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Chunking failed with strategy {strategy}: {e}")
            
            # Fallback to simple chunking
            if strategy != 'recursive':
                logger.info("Falling back to recursive chunking")
                return self.chunkers['recursive'].chunk_documents(documents, config)
            
            raise
    
    def get_recommended_strategy(self, document_type: str, document_length: int) -> str:
        """
        Get recommended chunking strategy based on document characteristics.
        Enhanced for Phase 1.1 with improved strategy selection.
        
        Args:
            document_type: Type of document
            document_length: Length of document in characters
            
        Returns:
            str: Recommended chunking strategy
        """
        # If enhanced chunking is available, use it for most documents
        if ENHANCED_CHUNKING_AVAILABLE:
            # Enhanced semantic chunking is best for structured documents
            if document_type in ['epub', 'pdf', 'docx'] and document_length > 2000:
                return 'enhanced_semantic'
            # Also good for medium-length documents of any type
            elif document_length > 1000:
                return 'enhanced_semantic'
            # Even for shorter documents, enhanced chunking provides better metadata
            elif document_length > 200:
                return 'enhanced_semantic'
            # Fall back to semantic for very short documents
            else:
                return 'semantic'
        else:
            # Fallback to original logic if enhanced chunking unavailable
            if document_type in ['epub', 'pdf'] and document_length > 10000:
                return 'semantic'
            return 'recursive'
    
    def estimate_chunk_count(
        self, 
        text_length: int, 
        chunk_size: int = 1000, 
        overlap: int = 200
    ) -> int:
        """
        Estimate the number of chunks for a given text length.
        
        Args:
            text_length: Length of text in characters
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            int: Estimated number of chunks
        """
        if text_length <= chunk_size:
            return 1
        
        effective_chunk_size = chunk_size - overlap
        return max(1, (text_length - overlap) // effective_chunk_size + 1)