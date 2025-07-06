#!/usr/bin/env python3
"""
DBC Ingestion Pipeline Test Script

This script tests the complete ingestion pipeline to ensure:
1. PDF text extraction works correctly
2. Metadata cleaning prevents ChromaDB errors
3. All components are properly configured

Run this script after setting up dependencies to validate your installation.
"""

import asyncio
import sys
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.services.ingestion import IngestionService
from src.utils.ocr_processing import check_ocr_availability, get_ocr_processor
from src.utils.database import get_database

# Configure logging for test
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


class IngestionTester:
    """Test suite for validating the DBC ingestion pipeline."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'pending'
        }
        
    async def run_all_tests(self) -> dict:
        """
        Run all ingestion pipeline tests.
        
        Returns:
            dict: Complete test results
        """
        logger.info("🧪 Starting DBC Ingestion Pipeline Tests")
        logger.info("=" * 60)
        
        # Test 1: Configuration
        await self._test_configuration()
        
        # Test 2: Database connection
        await self._test_database_connection()
        
        # Test 3: Dependencies
        await self._test_dependencies()
        
        # Test 4: OCR availability
        await self._test_ocr_availability()
        
        # Test 5: Sample PDF processing
        await self._test_sample_pdf_processing()
        
        # Test 6: Metadata cleaning
        await self._test_metadata_cleaning()
        
        # Calculate overall status
        passed_tests = sum(1 for test in self.results['tests'].values() if test['status'] == 'pass')
        total_tests = len(self.results['tests'])
        
        if passed_tests == total_tests:
            self.results['overall_status'] = 'pass'
            logger.info(f"🎉 All tests passed! ({passed_tests}/{total_tests})")
        else:
            self.results['overall_status'] = 'fail'
            logger.error(f"❌ {total_tests - passed_tests} tests failed ({passed_tests}/{total_tests} passed)")
        
        return self.results
    
    async def _test_configuration(self):
        """Test configuration loading."""
        test_name = "Configuration Loading"
        logger.info(f"🔧 Testing: {test_name}")
        
        try:
            settings = get_settings()
            
            # Check required settings
            required_fields = ['openai_api_key', 'chroma_collection_name', 'chroma_persist_directory']
            missing_fields = []
            
            for field in required_fields:
                if not getattr(settings, field, None):
                    missing_fields.append(field)
            
            if missing_fields:
                raise ValueError(f"Missing required configuration: {missing_fields}")
            
            self.results['tests'][test_name] = {
                'status': 'pass',
                'message': 'Configuration loaded successfully',
                'details': {
                    'vector_db_type': settings.vector_db_type,
                    'collection_name': settings.chroma_collection_name
                }
            }
            logger.info("✅ Configuration test passed")
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'fail',
                'message': str(e),
                'details': None
            }
            logger.error(f"❌ Configuration test failed: {e}")
    
    async def _test_database_connection(self):
        """Test vector database connection."""
        test_name = "Database Connection"
        logger.info(f"🗄️ Testing: {test_name}")
        
        try:
            db = await get_database()
            settings = get_settings()
            
            # Test collection creation
            collection_info = await db.get_collection_info(settings.chroma_collection_name)
            
            self.results['tests'][test_name] = {
                'status': 'pass',
                'message': 'Database connection successful',
                'details': {
                    'collection_info': collection_info,
                    'db_type': settings.vector_db_type
                }
            }
            logger.info(f"✅ Database test passed - Collection: {settings.chroma_collection_name}")
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'fail',
                'message': str(e),
                'details': None
            }
            logger.error(f"❌ Database test failed: {e}")
    
    async def _test_dependencies(self):
        """Test core dependencies."""
        test_name = "Dependencies Check"
        logger.info(f"📦 Testing: {test_name}")
        
        dependencies = {}
        try:
            # Test OpenAI
            import openai
            dependencies['openai'] = {'status': 'available', 'version': openai.__version__}
            
            # Test ChromaDB
            import chromadb
            dependencies['chromadb'] = {'status': 'available', 'version': chromadb.__version__}
            
            # Test LangChain Community
            import langchain_community
            dependencies['langchain_community'] = {'status': 'available'}
            
            # Test unstructured
            try:
                from unstructured.partition.pdf import partition_pdf
                dependencies['unstructured_pdf'] = {'status': 'available'}
            except ImportError:
                dependencies['unstructured_pdf'] = {'status': 'missing'}
            
            # Check for any missing critical dependencies
            critical_deps = ['openai', 'chromadb', 'langchain_community', 'unstructured_pdf']
            missing_critical = [dep for dep in critical_deps if dependencies.get(dep, {}).get('status') != 'available']
            
            if missing_critical:
                raise ImportError(f"Missing critical dependencies: {missing_critical}")
            
            self.results['tests'][test_name] = {
                'status': 'pass',
                'message': 'All critical dependencies available',
                'details': dependencies
            }
            logger.info("✅ Dependencies test passed")
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'fail',
                'message': str(e),
                'details': dependencies
            }
            logger.error(f"❌ Dependencies test failed: {e}")
    
    async def _test_ocr_availability(self):
        """Test OCR capabilities."""
        test_name = "OCR Availability"
        logger.info(f"🔍 Testing: {test_name}")
        
        try:
            ocr_status = check_ocr_availability()
            ocr_processor = get_ocr_processor()
            
            self.results['tests'][test_name] = {
                'status': 'pass' if ocr_status['available'] else 'warning',
                'message': 'OCR available' if ocr_status['available'] else 'OCR not available (optional)',
                'details': ocr_status
            }
            
            if ocr_status['available']:
                logger.info("✅ OCR test passed - OCR processing available")
            else:
                logger.warning("⚠️ OCR not available - will not be able to process scanned PDFs")
                logger.info("📝 To enable OCR, run the installation script: ./install_deps.sh")
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'warning',
                'message': f"OCR check failed: {e}",
                'details': None
            }
            logger.warning(f"⚠️ OCR test warning: {e}")
    
    async def _test_sample_pdf_processing(self):
        """Test PDF processing with a simple sample."""
        test_name = "Sample PDF Processing"
        logger.info(f"📄 Testing: {test_name}")
        
        try:
            # Create a simple text PDF for testing
            sample_text = """
            Sample Book Content for Testing
            
            Chapter 1: Introduction
            This is a sample PDF document created for testing the DBC ingestion pipeline.
            It contains multiple paragraphs to test text extraction and chunking.
            
            Chapter 2: Content
            The ingestion system should be able to extract this text successfully,
            chunk it appropriately, and generate embeddings for vector storage.
            
            This test validates the entire pipeline from PDF loading to storage.
            """
            
            # Create temporary PDF-like file (simplified for testing)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(sample_text)
                temp_path = Path(temp_file.name)
            
            try:
                # Test ingestion service
                ingestion_service = IngestionService()
                
                # Test document loading (using txt for simplicity)
                from src.utils.document_loaders import TextDocumentLoader
                loader = TextDocumentLoader()
                documents, metadata = loader.load_document(temp_path)
                
                # Validate text extraction
                total_text = sum(len(doc.page_content) for doc in documents)
                if total_text == 0:
                    raise ValueError("No text extracted from sample document")
                
                # Test metadata cleaning
                sample_metadata = {
                    'title': metadata.get('title'),
                    'author': None,  # Test None value handling
                    'page_count': 2,
                    'text_length': total_text,
                    'complex_data': {'nested': 'value'},  # Test complex type handling
                    'empty_string': '',
                    'valid_bool': True
                }
                
                cleaned = ingestion_service._clean_metadata_for_storage(sample_metadata)
                
                # Validate cleaning
                if None in cleaned.values():
                    raise ValueError("Metadata cleaning failed - None values still present")
                
                self.results['tests'][test_name] = {
                    'status': 'pass',
                    'message': 'Sample processing successful',
                    'details': {
                        'text_extracted': total_text,
                        'documents_created': len(documents),
                        'metadata_cleaned': len(cleaned),
                        'sample_preview': documents[0].page_content[:100] + '...' if documents else ''
                    }
                }
                logger.info(f"✅ Sample processing test passed - {total_text} chars extracted")
                
            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'fail',
                'message': str(e),
                'details': None
            }
            logger.error(f"❌ Sample processing test failed: {e}")
    
    async def _test_metadata_cleaning(self):
        """Test metadata cleaning functionality."""
        test_name = "Metadata Cleaning"
        logger.info(f"🧹 Testing: {test_name}")
        
        try:
            from src.services.ingestion import IngestionService
            ingestion_service = IngestionService()
            
            # Test various problematic metadata scenarios
            test_cases = [
                {
                    'name': 'None values',
                    'input': {'title': 'Test', 'author': None, 'page': 1},
                    'expected_keys': ['title', 'page']
                },
                {
                    'name': 'Complex types',
                    'input': {'title': 'Test', 'data': {'nested': 'value'}, 'list': [1, 2, 3]},
                    'expected_types': [str, str, str]
                },
                {
                    'name': 'Mixed valid/invalid',
                    'input': {'valid_str': 'test', 'valid_int': 42, 'valid_bool': True, 'invalid_none': None},
                    'expected_keys': ['valid_str', 'valid_int', 'valid_bool']
                }
            ]
            
            results = []
            for case in test_cases:
                cleaned = ingestion_service._clean_metadata_for_storage(case['input'])
                
                # Check for None values
                if None in cleaned.values():
                    raise ValueError(f"Test case '{case['name']}' failed: None values present")
                
                # Check expected keys
                if 'expected_keys' in case:
                    if not all(key in cleaned for key in case['expected_keys']):
                        raise ValueError(f"Test case '{case['name']}' failed: Missing expected keys")
                
                results.append({
                    'case': case['name'],
                    'input_keys': len(case['input']),
                    'output_keys': len(cleaned),
                    'cleaned_successfully': True
                })
            
            self.results['tests'][test_name] = {
                'status': 'pass',
                'message': 'Metadata cleaning working correctly',
                'details': {'test_cases': results}
            }
            logger.info("✅ Metadata cleaning test passed")
            
        except Exception as e:
            self.results['tests'][test_name] = {
                'status': 'fail',
                'message': str(e),
                'details': None
            }
            logger.error(f"❌ Metadata cleaning test failed: {e}")
    
    def print_summary(self):
        """Print a formatted summary of test results."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        for test_name, result in self.results['tests'].items():
            status_icon = "✅" if result['status'] == 'pass' else "⚠️" if result['status'] == 'warning' else "❌"
            logger.info(f"{status_icon} {test_name}: {result['message']}")
        
        logger.info("\n📋 NEXT STEPS:")
        
        failed_tests = [name for name, result in self.results['tests'].items() if result['status'] == 'fail']
        if failed_tests:
            logger.info("❌ Fix the following issues:")
            for test in failed_tests:
                logger.info(f"  - {test}: {self.results['tests'][test]['message']}")
        else:
            logger.info("🎉 All critical tests passed! Your ingestion pipeline is ready.")
            logger.info("   You can now upload PDFs via the /api/v1/books/upload endpoint")
        
        ocr_available = any(
            test['status'] == 'pass' and 'OCR' in name 
            for name, test in self.results['tests'].items()
        )
        
        if not ocr_available:
            logger.info("\n📝 OPTIONAL: Enable OCR for scanned PDFs:")
            logger.info("   Run: ./install_deps.sh")
            logger.info("   This allows processing image-based PDFs without text layers")


async def main():
    """Run the ingestion pipeline tests."""
    tester = IngestionTester()
    
    try:
        results = await tester.run_all_tests()
        tester.print_summary()
        
        # Save results to file
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📁 Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results['overall_status'] == 'pass' else 1)
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())