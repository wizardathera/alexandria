"""
Document loaders for various file formats in the DBC application.

This module provides loaders for PDF, EPUB, DOC/DOCX, TXT, and HTML files,
with consistent text extraction and metadata handling across all formats.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Protocol
from pathlib import Path
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)

# Define Document protocol for type safety
class DocumentProtocol(Protocol):
    """Protocol for Document-like objects."""
    page_content: str
    metadata: Dict[str, Any]

# Import document processing libraries with detailed error handling
# Try Document class first (needed by all loaders)
try:
    from langchain.schema import Document
    DocumentClass = Document
    logger.info("Successfully imported LangChain Document class")
except ImportError as e:
    logger.warning(f"LangChain Document class not available: {e}")
    
    class DocumentClass:
        def __init__(self, page_content: str = "", metadata: Dict[str, Any] = None):
            self.page_content = page_content
            self.metadata = metadata or {}
    
    logger.info("Using fallback Document class implementation")

# Try to import each loader separately for better error reporting
try:
    from langchain_community.document_loaders import UnstructuredFileLoader  # type: ignore
    logger.info("Successfully imported UnstructuredFileLoader from langchain_community")
except ImportError as e:
    try:
        from langchain.document_loaders import UnstructuredFileLoader
        logger.info("Successfully imported UnstructuredFileLoader from langchain")
    except ImportError as e2:
        logger.error(f"UnstructuredFileLoader not available: {e2}")
        
        class UnstructuredFileLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
            def load(self):
                raise ImportError(
                    "UnstructuredFileLoader requires langchain and unstructured packages. "
                    "Install with: pip install langchain unstructured"
                )

try:
    from langchain_community.document_loaders import TextLoader  # type: ignore
    logger.info("Successfully imported TextLoader from langchain_community")
except ImportError as e:
    try:
        from langchain.document_loaders import TextLoader
        logger.info("Successfully imported TextLoader from langchain")
    except ImportError as e2:
        logger.error(f"TextLoader not available: {e2}")
        
        class TextLoader:
            def __init__(self, file_path: str, encoding: str = 'utf-8'):
                self.file_path = file_path
                self.encoding = encoding
            def load(self):
                raise ImportError(
                    "TextLoader requires langchain package. "
                    "Install with: pip install langchain"
                )

try:
    from langchain_community.document_loaders import UnstructuredHTMLLoader  # type: ignore
    logger.info("Successfully imported UnstructuredHTMLLoader from langchain_community")
except ImportError as e:
    try:
        from langchain.document_loaders import UnstructuredHTMLLoader
        logger.info("Successfully imported UnstructuredHTMLLoader from langchain")
    except ImportError as e2:
        logger.error(f"UnstructuredHTMLLoader not available: {e2}")
        
        class UnstructuredHTMLLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
            def load(self):
                raise ImportError(
                    "UnstructuredHTMLLoader requires langchain and unstructured packages. "
                    "Install with: pip install langchain unstructured"
                )

try:
    from langchain_community.document_loaders import UnstructuredEPubLoader  # type: ignore
    logger.info("Successfully imported UnstructuredEPubLoader from langchain_community")
except ImportError as e:
    try:
        from langchain.document_loaders import UnstructuredEPubLoader
        logger.info("Successfully imported UnstructuredEPubLoader from langchain")
    except ImportError as e2:
        logger.error(f"UnstructuredEPubLoader not available: {e2}")
        
        class UnstructuredEPubLoader:
            def __init__(self, file_path: str):
                self.file_path = file_path
            def load(self):
                raise ImportError(
                    "UnstructuredEPubLoader requires langchain and unstructured packages. "
                    "Install with: pip install langchain unstructured"
                )


def validate_dependencies() -> Dict[str, bool]:
    """
    Validate that all optional dependencies are available.
    
    Returns:
        Dict[str, bool]: Status of each dependency
    """
    dependencies = {
        'langchain': False,
        'langchain_community': False,
        'unstructured': False
    }
    
    # Test langchain core
    try:
        import langchain  # noqa: F401
        dependencies['langchain'] = True
    except ImportError:
        pass
    
    # Test langchain_community
    try:
        import langchain_community  # type: ignore  # noqa: F401
        dependencies['langchain_community'] = True
    except ImportError:
        pass
    
    # Test unstructured
    try:
        import unstructured  # noqa: F401
        dependencies['unstructured'] = True
    except ImportError:
        pass
    
    return dependencies


def get_dependency_status() -> str:
    """
    Get a human-readable status of all dependencies.
    
    Returns:
        str: Formatted dependency status
    """
    deps = validate_dependencies()
    
    status_lines = ["Document Loader Dependencies:"]
    for dep, available in deps.items():
        status = "✅ Available" if available else "❌ Missing"
        status_lines.append(f"  {dep}: {status}")
    
    missing = [dep for dep, available in deps.items() if not available]
    if missing:
        status_lines.append("\nTo install missing dependencies:")
        status_lines.append(f"  pip install {' '.join(missing)}")
    
    return "\n".join(status_lines)


class DocumentLoaderInterface(ABC):
    """
    Abstract interface for document loaders.
    
    This interface ensures consistent behavior across all file format loaders.
    """
    
    @abstractmethod
    def can_load(self, file_path: Path) -> bool:
        """Check if this loader can handle the given file."""
        pass
    
    @abstractmethod
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """Load document and extract metadata."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        pass


class PDFLoader(DocumentLoaderInterface):
    """
    PDF document loader using UnstructuredFileLoader.
    
    Handles PDF files with text extraction and metadata preservation.
    """
    
    def can_load(self, file_path: Path) -> bool:
        """
        Check if file is a PDF.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if file is a PDF
        """
        return file_path.suffix.lower() == '.pdf'
    
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """
        Load PDF document and extract text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
            
        Raises:
            ValueError: If PDF contains no extractable text (likely scanned images)
        """
        logger.info(f"Loading PDF: {file_path}")
        
        try:
            # TODO: Replace deprecated UnstructuredFileLoader with UnstructuredLoader from langchain-unstructured
            # Use UnstructuredFileLoader for PDF processing
            loader = UnstructuredFileLoader(str(file_path))
            documents = loader.load()
            
            # Validate text extraction
            total_text_length = sum(len(doc.page_content.strip()) for doc in documents)
            
            if total_text_length == 0:
                error_msg = (
                    f"No text extracted from PDF '{file_path.name}'. "
                    "This PDF likely contains scanned images without a text layer. "
                    "OCR processing may be required to extract text from image-based PDFs."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log first chunk preview for debugging
            if documents and documents[0].page_content:
                preview_text = documents[0].page_content[:200].replace('\n', ' ').strip()
                logger.info(f"✅ PDF text extraction successful - Preview: '{preview_text}...'")
            
            # Extract metadata
            metadata = self._extract_pdf_metadata(file_path, documents)
            
            # Enhance documents with file-specific metadata
            for doc in documents:
                doc.metadata.update({
                    'file_type': 'pdf',
                    'file_path': str(file_path),
                    'loader_type': 'unstructured'
                })
            
            logger.info(f"✅ Successfully loaded PDF: {len(documents)} chunks, {total_text_length} characters total")
            return documents, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load PDF {file_path}: {e}")
            raise ValueError(f"Unable to load PDF file: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions for PDF loader."""
        return ['.pdf']
    
    def _extract_pdf_metadata(self, file_path: Path, documents: List[DocumentProtocol]) -> Dict[str, Any]:
        """
        Extract metadata from PDF file and documents.
        
        Args:
            file_path: Path to the PDF file
            documents: Loaded document chunks
            
        Returns:
            Dict[str, Any]: Extracted metadata
        """
        # Basic file metadata
        stat = file_path.stat()
        
        # Combine all document text for analysis
        full_text = '\n'.join([doc.page_content for doc in documents])
        
        return {
            'title': file_path.stem,  # Use filename as title initially
            'author': None,  # PDF metadata extraction would go here
            'file_size': stat.st_size,
            'file_name': file_path.name,
            'text_length': len(full_text),
            'chunk_count': len(documents),
            'created_date': datetime.fromtimestamp(stat.st_ctime),
            'modified_date': datetime.fromtimestamp(stat.st_mtime)
        }


class EPUBLoader(DocumentLoaderInterface):
    """
    EPUB document loader using UnstructuredEPubLoader.
    
    Handles EPUB e-book files with chapter-aware processing.
    """
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is an EPUB."""
        return file_path.suffix.lower() == '.epub'
    
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """
        Load EPUB document and extract text with metadata.
        
        Args:
            file_path: Path to the EPUB file
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
        """
        logger.info(f"Loading EPUB: {file_path}")
        
        try:
            # Use specialized EPUB loader
            loader = UnstructuredEPubLoader(str(file_path))
            documents = loader.load()
            
            # Extract metadata
            metadata = self._extract_epub_metadata(file_path, documents)
            
            # Enhance documents with file-specific metadata
            for doc in documents:
                doc.metadata.update({
                    'file_type': 'epub',
                    'file_path': str(file_path),
                    'loader_type': 'unstructured_epub'
                })
            
            logger.info(f"Successfully loaded EPUB: {len(documents)} chapters/sections")
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Failed to load EPUB {file_path}: {e}")
            raise ValueError(f"Unable to load EPUB file: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions for EPUB loader."""
        return ['.epub']
    
    def _extract_epub_metadata(self, file_path: Path, documents: List[DocumentProtocol]) -> Dict[str, Any]:
        """Extract metadata from EPUB file."""
        stat = file_path.stat()
        full_text = '\n'.join([doc.page_content for doc in documents])
        
        return {
            'title': file_path.stem,
            'author': None,  # EPUB metadata extraction would go here
            'file_size': stat.st_size,
            'file_name': file_path.name,
            'text_length': len(full_text),
            'chunk_count': len(documents),
            'created_date': datetime.fromtimestamp(stat.st_ctime),
            'modified_date': datetime.fromtimestamp(stat.st_mtime)
        }


class DOCLoader(DocumentLoaderInterface):
    """
    DOC/DOCX document loader using UnstructuredFileLoader.
    
    Handles Microsoft Word documents with formatting preservation.
    """
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a DOC or DOCX."""
        return file_path.suffix.lower() in ['.doc', '.docx']
    
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """
        Load DOC/DOCX document and extract text with metadata.
        
        Args:
            file_path: Path to the DOC/DOCX file
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
        """
        logger.info(f"Loading DOC/DOCX: {file_path}")
        
        try:
            # TODO: Replace deprecated UnstructuredFileLoader with UnstructuredLoader from langchain-unstructured
            # Use UnstructuredFileLoader for Word document processing
            loader = UnstructuredFileLoader(str(file_path))
            documents = loader.load()
            
            # Extract metadata
            metadata = self._extract_doc_metadata(file_path, documents)
            
            # Enhance documents with file-specific metadata
            for doc in documents:
                doc.metadata.update({
                    'file_type': file_path.suffix.lower().lstrip('.'),
                    'file_path': str(file_path),
                    'loader_type': 'unstructured'
                })
            
            logger.info(f"Successfully loaded DOC/DOCX: {len(documents)} sections")
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Failed to load DOC/DOCX {file_path}: {e}")
            raise ValueError(f"Unable to load Word document: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions for DOC loader."""
        return ['.doc', '.docx']
    
    def _extract_doc_metadata(self, file_path: Path, documents: List[DocumentProtocol]) -> Dict[str, Any]:
        """Extract metadata from DOC/DOCX file."""
        stat = file_path.stat()
        full_text = '\n'.join([doc.page_content for doc in documents])
        
        return {
            'title': file_path.stem,
            'author': None,  # Word document metadata extraction would go here
            'file_size': stat.st_size,
            'file_name': file_path.name,
            'text_length': len(full_text),
            'chunk_count': len(documents),
            'created_date': datetime.fromtimestamp(stat.st_ctime),
            'modified_date': datetime.fromtimestamp(stat.st_mtime)
        }


class TXTLoader(DocumentLoaderInterface):
    """
    Plain text document loader using TextLoader.
    
    Handles plain text files with encoding detection.
    """
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is a plain text file."""
        return file_path.suffix.lower() == '.txt'
    
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """
        Load plain text document.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
        """
        logger.info(f"Loading TXT: {file_path}")
        
        try:
            # Use TextLoader for plain text processing
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
            
            # Extract metadata
            metadata = self._extract_txt_metadata(file_path, documents)
            
            # Enhance documents with file-specific metadata
            for doc in documents:
                doc.metadata.update({
                    'file_type': 'txt',
                    'file_path': str(file_path),
                    'loader_type': 'text'
                })
            
            logger.info(f"Successfully loaded TXT: {len(documents)} documents")
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path}: {e}")
            raise ValueError(f"Unable to load text file: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions for TXT loader."""
        return ['.txt']
    
    def _extract_txt_metadata(self, file_path: Path, documents: List[DocumentProtocol]) -> Dict[str, Any]:
        """Extract metadata from text file."""
        stat = file_path.stat()
        full_text = '\n'.join([doc.page_content for doc in documents])
        
        return {
            'title': file_path.stem,
            'author': None,
            'file_size': stat.st_size,
            'file_name': file_path.name,
            'text_length': len(full_text),
            'chunk_count': len(documents),
            'created_date': datetime.fromtimestamp(stat.st_ctime),
            'modified_date': datetime.fromtimestamp(stat.st_mtime)
        }


class HTMLLoader(DocumentLoaderInterface):
    """
    HTML document loader using UnstructuredHTMLLoader.
    
    Handles HTML files with tag-aware text extraction.
    """
    
    def can_load(self, file_path: Path) -> bool:
        """Check if file is an HTML file."""
        return file_path.suffix.lower() in ['.html', '.htm']
    
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """
        Load HTML document and extract text with metadata.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
        """
        logger.info(f"Loading HTML: {file_path}")
        
        try:
            # Use specialized HTML loader
            loader = UnstructuredHTMLLoader(str(file_path))
            documents = loader.load()
            
            # Extract metadata
            metadata = self._extract_html_metadata(file_path, documents)
            
            # Enhance documents with file-specific metadata
            for doc in documents:
                doc.metadata.update({
                    'file_type': 'html',
                    'file_path': str(file_path),
                    'loader_type': 'unstructured_html'
                })
            
            logger.info(f"Successfully loaded HTML: {len(documents)} sections")
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Failed to load HTML {file_path}: {e}")
            raise ValueError(f"Unable to load HTML file: {e}")
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported extensions for HTML loader."""
        return ['.html', '.htm']
    
    def _extract_html_metadata(self, file_path: Path, documents: List[DocumentProtocol]) -> Dict[str, Any]:
        """Extract metadata from HTML file."""
        stat = file_path.stat()
        full_text = '\n'.join([doc.page_content for doc in documents])
        
        return {
            'title': file_path.stem,
            'author': None,  # HTML meta tag extraction would go here
            'file_size': stat.st_size,
            'file_name': file_path.name,
            'text_length': len(full_text),
            'chunk_count': len(documents),
            'created_date': datetime.fromtimestamp(stat.st_ctime),
            'modified_date': datetime.fromtimestamp(stat.st_mtime)
        }


class DocumentLoaderManager:
    """
    Manager class for all document loaders.
    
    Provides a unified interface for loading any supported file format.
    """
    
    def __init__(self):
        """Initialize the document loader manager."""
        self.loaders = [
            PDFLoader(),
            EPUBLoader(),
            DOCLoader(),
            TXTLoader(),
            HTMLLoader()
        ]
        self.settings = get_settings()
    
    def get_loader_for_file(self, file_path: Path) -> Optional[DocumentLoaderInterface]:
        """
        Get the appropriate loader for a given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Optional[DocumentLoaderInterface]: Loader that can handle the file
        """
        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader
        return None
    
    def is_supported_format(self, file_path: Path) -> bool:
        """
        Check if the file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if format is supported
        """
        return self.get_loader_for_file(file_path) is not None
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions.
        
        Returns:
            List[str]: List of supported extensions
        """
        extensions = []
        for loader in self.loaders:
            extensions.extend(loader.get_supported_extensions())
        return extensions
    
    def load_document(self, file_path: Path) -> Tuple[List[DocumentProtocol], Dict[str, Any]]:
        """
        Load a document using the appropriate loader.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
            
        Raises:
            ValueError: If file format is not supported
        """
        loader = self.get_loader_for_file(file_path)
        if not loader:
            supported = ', '.join(self.get_supported_extensions())
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {supported}"
            )
        
        return loader.load_document(file_path)