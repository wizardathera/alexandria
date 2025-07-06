"""
OCR processing utilities for scanned PDF documents.

This module provides OCR capabilities for PDFs that contain only images
without extractable text layers, using pytesseract and PDF2image.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import asyncio
from datetime import datetime

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    convert_from_path = None
    Image = None
    OCR_AVAILABLE = False

try:
    from langchain.schema import Document
except ImportError:
    Document = dict

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)


class OCRProcessor:
    """
    OCR processor for extracting text from scanned PDF documents.
    
    Uses pytesseract and pdf2image to convert PDF pages to images
    and extract text using Optical Character Recognition.
    """
    
    def __init__(self):
        """Initialize OCR processor."""
        self.settings = get_settings()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if OCR dependencies are available."""
        if not OCR_AVAILABLE:
            raise ImportError(
                "OCR dependencies not installed. Install with: "
                "pip install pytesseract pdf2image pillow"
            )
        
        # Check if tesseract is installed on system
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "Tesseract OCR not found. Install with: "
                "sudo apt-get install tesseract-ocr (Ubuntu/Debian) or "
                "brew install tesseract (macOS)"
            )
    
    def is_available(self) -> bool:
        """
        Check if OCR processing is available.
        
        Returns:
            bool: True if OCR can be performed
        """
        try:
            self._check_dependencies()
            return True
        except (ImportError, RuntimeError):
            return False
    
    async def process_pdf_with_ocr(
        self,
        pdf_path: Path,
        language: str = "eng",
        dpi: int = 200,
        max_pages: Optional[int] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Process a scanned PDF using OCR to extract text.
        
        Args:
            pdf_path: Path to the PDF file
            language: OCR language (default: 'eng' for English)
            dpi: DPI for PDF to image conversion (higher = better quality/slower)
            max_pages: Maximum number of pages to process (None = all pages)
            
        Returns:
            Tuple[List[Document], Dict[str, Any]]: Documents and metadata
            
        Raises:
            ValueError: If OCR processing fails
            RuntimeError: If dependencies are missing
        """
        if not self.is_available():
            raise RuntimeError("OCR processing not available - missing dependencies")
        
        logger.info(f"🔍 Starting OCR processing for: {pdf_path.name}")
        start_time = datetime.now()
        
        try:
            # Convert PDF to images
            logger.info(f"📄 Converting PDF to images (DPI: {dpi})")
            images = await asyncio.to_thread(
                convert_from_path, 
                str(pdf_path), 
                dpi=dpi,
                first_page=1,
                last_page=max_pages
            )
            
            if not images:
                raise ValueError("No pages found in PDF or PDF is corrupted")
            
            logger.info(f"🖼️ Converted {len(images)} pages to images")
            
            # Process each page with OCR
            documents = []
            total_text_length = 0
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"🔍 Processing page {page_num}/{len(images)} with OCR...")
                
                # Extract text from image
                page_text = await asyncio.to_thread(
                    pytesseract.image_to_string,
                    image,
                    lang=language,
                    config='--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
                )
                
                # Clean and validate extracted text
                page_text = page_text.strip()
                if len(page_text) < 10:  # Skip pages with minimal text
                    logger.warning(f"⚠️ Page {page_num} contains minimal text ({len(page_text)} chars) - skipping")
                    continue
                
                total_text_length += len(page_text)
                
                # Create document for this page
                doc = Document(
                    page_content=page_text,
                    metadata={
                        'page': page_num,
                        'source': str(pdf_path),
                        'processing_method': 'ocr',
                        'ocr_language': language,
                        'ocr_dpi': dpi,
                        'page_text_length': len(page_text)
                    }
                )
                documents.append(doc)
                
                # Log progress
                if page_num % 5 == 0 or page_num == len(images):
                    logger.info(f"📝 Processed {page_num} pages, extracted {total_text_length} characters so far")
            
            # Validate overall extraction
            if not documents:
                raise ValueError(
                    "OCR failed to extract any readable text from PDF. "
                    "The document may be too low quality, corrupted, or not contain text."
                )
            
            if total_text_length < 100:
                logger.warning(
                    f"⚠️ OCR extracted minimal text ({total_text_length} chars). "
                    "Consider using higher DPI or different OCR settings."
                )
            
            # Create metadata
            duration = (datetime.now() - start_time).total_seconds()
            metadata = {
                'title': pdf_path.stem,
                'source': str(pdf_path),
                'processing_method': 'ocr',
                'ocr_language': language,
                'ocr_dpi': dpi,
                'total_pages': len(images),
                'processed_pages': len(documents),
                'total_text_length': total_text_length,
                'processing_duration_seconds': duration,
                'text': ' '.join(doc.page_content for doc in documents)
            }
            
            # Log first chunk preview
            if documents and documents[0].page_content:
                preview = documents[0].page_content[:200].replace('\n', ' ').strip()
                logger.info(f"📝 OCR text preview: '{preview}...'")
            
            logger.info(
                f"✅ OCR processing completed: {len(documents)} pages, "
                f"{total_text_length} characters, {duration:.1f}s"
            )
            
            return documents, metadata
            
        except Exception as e:
            logger.error(f"❌ OCR processing failed for {pdf_path}: {e}")
            raise ValueError(f"OCR processing failed: {e}")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported OCR languages.
        
        Returns:
            List[str]: Available language codes
        """
        if not self.is_available():
            return []
        
        try:
            langs = pytesseract.get_languages(config='')
            return langs
        except Exception:
            return ['eng']  # Default to English if detection fails
    
    def estimate_processing_time(
        self,
        pdf_path: Path,
        dpi: int = 200
    ) -> Dict[str, Any]:
        """
        Estimate OCR processing time based on PDF size and settings.
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI setting for conversion
            
        Returns:
            Dict[str, Any]: Time estimates and recommendations
        """
        try:
            # Get basic PDF info without full conversion
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            
            # Rough estimates based on file size and DPI
            # These are approximations and can vary significantly
            base_time_per_mb = 30  # seconds per MB at 200 DPI
            dpi_multiplier = (dpi / 200) ** 2  # Processing time scales roughly with DPI squared
            
            estimated_time = file_size_mb * base_time_per_mb * dpi_multiplier
            
            # Provide recommendations
            recommendations = []
            if estimated_time > 300:  # > 5 minutes
                recommendations.append("Consider using lower DPI (150) for faster processing")
            if file_size_mb > 50:
                recommendations.append("Large file - consider processing in smaller batches")
            if dpi > 300:
                recommendations.append("Very high DPI may not improve accuracy significantly")
            
            return {
                'file_size_mb': round(file_size_mb, 2),
                'estimated_time_seconds': round(estimated_time),
                'estimated_time_minutes': round(estimated_time / 60, 1),
                'dpi': dpi,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate processing time: {e}")
            return {
                'error': str(e),
                'estimated_time_seconds': None
            }


def get_ocr_processor() -> Optional[OCRProcessor]:
    """
    Get OCR processor instance if available.
    
    Returns:
        Optional[OCRProcessor]: OCR processor or None if not available
    """
    try:
        return OCRProcessor()
    except (ImportError, RuntimeError) as e:
        logger.warning(f"OCR processor not available: {e}")
        return None


def check_ocr_availability() -> Dict[str, Any]:
    """
    Check OCR availability and provide setup instructions.
    
    Returns:
        Dict[str, Any]: Availability status and setup instructions
    """
    status = {
        'available': False,
        'python_packages': {},
        'system_dependencies': {},
        'setup_instructions': []
    }
    
    # Check Python packages
    try:
        import pytesseract
        status['python_packages']['pytesseract'] = {
            'installed': True,
            'version': getattr(pytesseract, '__version__', 'unknown')
        }
    except ImportError:
        status['python_packages']['pytesseract'] = {'installed': False}
        status['setup_instructions'].append("pip install pytesseract")
    
    try:
        import pdf2image
        status['python_packages']['pdf2image'] = {
            'installed': True,
            'version': getattr(pdf2image, '__version__', 'unknown')
        }
    except ImportError:
        status['python_packages']['pdf2image'] = {'installed': False}
        status['setup_instructions'].append("pip install pdf2image")
    
    try:
        from PIL import Image
        status['python_packages']['pillow'] = {
            'installed': True,
            'version': getattr(Image, '__version__', 'unknown')
        }
    except ImportError:
        status['python_packages']['pillow'] = {'installed': False}
        status['setup_instructions'].append("pip install pillow")
    
    # Check system dependencies
    try:
        if pytesseract:
            version = pytesseract.get_tesseract_version()
            status['system_dependencies']['tesseract'] = {
                'installed': True,
                'version': str(version)
            }
    except:
        status['system_dependencies']['tesseract'] = {'installed': False}
        status['setup_instructions'].extend([
            "sudo apt-get install tesseract-ocr (Ubuntu/Debian)",
            "brew install tesseract (macOS)",
            "Download from: https://github.com/UB-Mannheim/tesseract/wiki (Windows)"
        ])
    
    # Check overall availability
    all_python_packages = all(
        pkg['installed'] for pkg in status['python_packages'].values()
    )
    all_system_deps = all(
        dep['installed'] for dep in status['system_dependencies'].values()
    )
    
    status['available'] = all_python_packages and all_system_deps
    
    return status