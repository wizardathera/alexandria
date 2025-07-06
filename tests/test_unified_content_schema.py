"""
Comprehensive tests for the unified content schema and multi-module support.

This test suite validates the unified content models, content service,
enhanced database operations, and migration functionality.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.models import (
    ContentItem, ContentRelationship, EmbeddingMetadata, User,
    ModuleType, ContentType, ContentVisibility, UserRole, ProcessingStatus,
    ContentRelationshipType, LegacyBookMetadata,
    LibraryBookMetadata, LMSCourseMetadata, LMSLessonMetadata, MarketplaceItemMetadata,
    get_content_type_for_module, validate_content_for_module
)
from src.services.content_service import ContentService, ContentDatabaseError
from src.services.migration_service import MigrationService, MigrationError
from src.utils.enhanced_database import EnhancedChromaVectorDB


class TestUnifiedContentModels:
    """Test the unified content models and validation."""
    
    def test_content_item_creation(self):
        """Test creating content items for different modules."""
        # Test library book
        book = ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Test Book",
            author="Test Author",
            visibility=ContentVisibility.PUBLIC
        )
        
        assert book.module_type == ModuleType.LIBRARY
        assert book.content_type == ContentType.BOOK
        assert book.title == "Test Book"
        assert book.visibility == ContentVisibility.PUBLIC
        assert book.processing_status == ProcessingStatus.PENDING
        
        # Test LMS course
        course = ContentItem(
            module_type=ModuleType.LMS,
            content_type=ContentType.COURSE,
            title="Python Basics",
            description="Learn Python programming",
            visibility=ContentVisibility.ORGANIZATION
        )
        
        assert course.module_type == ModuleType.LMS
        assert course.content_type == ContentType.COURSE
        assert course.visibility == ContentVisibility.ORGANIZATION
    
    def test_content_item_validation(self):
        """Test content item validation for module compatibility."""
        # Valid combinations
        assert validate_content_for_module(ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Test"
        ))
        
        assert validate_content_for_module(ContentItem(
            module_type=ModuleType.LMS,
            content_type=ContentType.LESSON,
            title="Test"
        ))
        
        # Invalid combinations would require model validation in Pydantic
        # but the function checks compatibility
        
    def test_content_relationships(self):
        """Test content relationship creation and management."""
        relationship = ContentRelationship(
            source_content_id="book-1",
            target_content_id="book-2",
            relationship_type=ContentRelationshipType.PREREQUISITE,
            strength=0.8,
            confidence=0.9
        )
        
        assert relationship.source_content_id == "book-1"
        assert relationship.target_content_id == "book-2"
        assert relationship.relationship_type == ContentRelationshipType.PREREQUISITE
        assert relationship.strength == 0.8
        assert relationship.confidence == 0.9
        assert not relationship.human_verified
        
        # Test human verification
        relationship.verify_by_human("user-123")
        assert relationship.human_verified
        assert relationship.verified_by == "user-123"
        assert relationship.verified_at is not None
    
    def test_embedding_metadata(self):
        """Test enhanced embedding metadata."""
        metadata = EmbeddingMetadata(
            content_id="test-content",
            chunk_index=0,
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            visibility=ContentVisibility.PUBLIC,
            text_content="This is a test chunk",
            chunk_length=20,
            semantic_tags=["programming", "python"]
        )
        
        assert metadata.content_id == "test-content"
        assert metadata.module_type == ModuleType.LIBRARY
        assert metadata.content_type == ContentType.BOOK
        assert metadata.visibility == ContentVisibility.PUBLIC
        assert len(metadata.semantic_tags) == 2
        
        # Test adding semantic tags
        metadata.add_semantic_tag("beginner")
        assert "beginner" in metadata.semantic_tags
        assert len(metadata.semantic_tags) == 3
        
        # Test duplicate tag handling
        metadata.add_semantic_tag("python")
        assert len(metadata.semantic_tags) == 3  # Should not duplicate
    
    def test_user_permissions(self):
        """Test user permission checking."""
        user = User(
            email="test@example.com",
            role=UserRole.READER,
            organization_id="org-123"
        )
        
        # Test public content access
        public_content = ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Public Book",
            visibility=ContentVisibility.PUBLIC
        )
        assert user.can_access_content(public_content)
        
        # Test private content access (not owner)
        private_content = ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Private Book",
            visibility=ContentVisibility.PRIVATE,
            created_by="other-user"
        )
        assert not user.can_access_content(private_content)
        
        # Test private content access (owner)
        private_content.created_by = user.user_id
        assert user.can_access_content(private_content)
        
        # Test organization content access
        org_content = ContentItem(
            module_type=ModuleType.LMS,
            content_type=ContentType.COURSE,
            title="Org Course",
            visibility=ContentVisibility.ORGANIZATION,
            organization_id="org-123"
        )
        assert user.can_access_content(org_content)
        
        # Test premium content access (no subscription)
        premium_content = ContentItem(
            module_type=ModuleType.MARKETPLACE,
            content_type=ContentType.PREMIUM_COURSE,
            title="Premium Course",
            visibility=ContentVisibility.PREMIUM
        )
        assert not user.can_access_content(premium_content)
        
        # Test premium content access (with subscription)
        user.subscription_tier = "pro"
        assert user.can_access_content(premium_content)
    
    def test_legacy_book_migration(self):
        """Test legacy book metadata conversion."""
        legacy_book = LegacyBookMetadata(
            book_id="legacy-book-1",
            title="Legacy Book",
            author="Legacy Author",
            file_type="pdf",
            file_name="legacy.pdf",
            file_path="/path/to/legacy.pdf",
            file_size=1024000,
            text_length=50000,
            chunk_count=150
        )
        
        # Convert to unified content item
        content_item = legacy_book.to_content_item()
        
        assert content_item.content_id == "legacy-book-1"
        assert content_item.module_type == ModuleType.LIBRARY
        assert content_item.content_type == ContentType.BOOK
        assert content_item.title == "Legacy Book"
        assert content_item.author == "Legacy Author"
        assert content_item.file_type == "pdf"
        assert content_item.file_size == 1024000
        assert content_item.text_length == 50000
        assert content_item.chunk_count == 150
    
    def test_module_specific_metadata(self):
        """Test module-specific metadata extensions."""
        # Library book metadata
        book_metadata = LibraryBookMetadata(
            isbn="978-0123456789",
            publisher="Test Publisher",
            genre="Programming",
            page_count=300,
            book_format="ebook",
            has_table_of_contents=True,
            chapter_count=12
        )
        
        assert book_metadata.isbn == "978-0123456789"
        assert book_metadata.genre == "Programming"
        assert book_metadata.has_table_of_contents
        
        # LMS course metadata
        course_metadata = LMSCourseMetadata(
            course_duration=480,  # 8 hours
            skill_level="intermediate",
            learning_objectives=["Learn Python", "Build projects"],
            lesson_count=24,
            has_certificate=True,
            price=99.99
        )
        
        assert course_metadata.course_duration == 480
        assert course_metadata.skill_level == "intermediate"
        assert len(course_metadata.learning_objectives) == 2
        assert course_metadata.has_certificate
        
        # Marketplace item metadata
        marketplace_metadata = MarketplaceItemMetadata(
            price=49.99,
            license_type="commercial",
            purchase_count=150,
            rating=4.7,
            revenue_share=0.85
        )
        
        assert marketplace_metadata.price == 49.99
        assert marketplace_metadata.license_type == "commercial"
        assert marketplace_metadata.revenue_share == 0.85


class TestContentService:
    """Test the unified content service."""
    
    @pytest.fixture
    async def content_service(self):
        """Create a test content service with temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            service = ContentService(db_path=tmp_file.name)
            await service.initialize()
            yield service
            await service.close()
            Path(tmp_file.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_content_item(self):
        """Create a sample content item for testing."""
        return ContentItem(
            content_id="test-book-1",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Test Book",
            author="Test Author",
            description="A test book for unit testing",
            file_name="test.pdf",
            file_type="pdf",
            file_size=1024000,
            visibility=ContentVisibility.PUBLIC,
            created_by="user-1",
            text_length=10000,
            chunk_count=25
        )
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            user_id="test-user-1",
            email="test@example.com",
            role=UserRole.READER,
            organization_id="test-org"
        )
    
    async def test_create_content_item(self, content_service, sample_content_item):
        """Test creating a content item."""
        success = await content_service.create_content_item(sample_content_item)
        assert success
        
        # Verify it was created
        retrieved = await content_service.get_content_item(sample_content_item.content_id)
        assert retrieved is not None
        assert retrieved.title == sample_content_item.title
        assert retrieved.author == sample_content_item.author
        assert retrieved.module_type == sample_content_item.module_type
    
    async def test_get_content_item_with_permissions(self, content_service, sample_content_item, sample_user):
        """Test retrieving content with permission checking."""
        # Create private content
        sample_content_item.visibility = ContentVisibility.PRIVATE
        sample_content_item.created_by = "other-user"
        
        await content_service.create_content_item(sample_content_item)
        
        # Should not be accessible to different user
        retrieved = await content_service.get_content_item(sample_content_item.content_id, sample_user)
        assert retrieved is None
        
        # Should be accessible to owner
        sample_user.user_id = "other-user"
        retrieved = await content_service.get_content_item(sample_content_item.content_id, sample_user)
        assert retrieved is not None
    
    async def test_list_content_items_with_filters(self, content_service, sample_user):
        """Test listing content with module and permission filters."""
        # Create multiple content items
        library_book = ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Library Book",
            visibility=ContentVisibility.PUBLIC
        )
        
        lms_course = ContentItem(
            module_type=ModuleType.LMS,
            content_type=ContentType.COURSE,
            title="LMS Course",
            visibility=ContentVisibility.PUBLIC
        )
        
        private_book = ContentItem(
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Private Book",
            visibility=ContentVisibility.PRIVATE,
            created_by="other-user"
        )
        
        # Create all items
        await content_service.create_content_item(library_book)
        await content_service.create_content_item(lms_course)
        await content_service.create_content_item(private_book)
        
        # Test module filtering
        library_items = await content_service.list_content_items(
            module_type=ModuleType.LIBRARY,
            user=sample_user
        )
        assert len(library_items) == 1  # Only public library book
        assert library_items[0].title == "Library Book"
        
        # Test content type filtering
        book_items = await content_service.list_content_items(
            content_type=ContentType.BOOK,
            user=sample_user
        )
        assert len(book_items) == 1  # Only public book
        
        # Test no filter (all accessible content)
        all_items = await content_service.list_content_items(user=sample_user)
        assert len(all_items) == 2  # Library book and LMS course
    
    async def test_update_content_item(self, content_service, sample_content_item):
        """Test updating content items."""
        # Create initial item
        await content_service.create_content_item(sample_content_item)
        
        # Update the item
        sample_content_item.title = "Updated Title"
        sample_content_item.description = "Updated description"
        sample_content_item.mark_processed()
        
        success = await content_service.update_content_item(sample_content_item)
        assert success
        
        # Verify updates
        retrieved = await content_service.get_content_item(sample_content_item.content_id)
        assert retrieved.title == "Updated Title"
        assert retrieved.description == "Updated description"
        assert retrieved.processing_status == ProcessingStatus.COMPLETED
        assert retrieved.processed_at is not None
    
    async def test_delete_content_item(self, content_service, sample_content_item, sample_user):
        """Test deleting content items with permission checking."""
        # Create item
        await content_service.create_content_item(sample_content_item)
        
        # Test deletion by non-owner (should fail)
        sample_user.user_id = "different-user"
        success = await content_service.delete_content_item(sample_content_item.content_id, sample_user)
        assert not success
        
        # Test deletion by owner (should succeed)
        sample_user.user_id = sample_content_item.created_by
        success = await content_service.delete_content_item(sample_content_item.content_id, sample_user)
        assert success
        
        # Verify deletion
        retrieved = await content_service.get_content_item(sample_content_item.content_id)
        assert retrieved is None
    
    async def test_content_relationships(self, content_service):
        """Test content relationship management."""
        # Create test content items
        book1 = ContentItem(
            content_id="book-1",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Beginner Book",
            visibility=ContentVisibility.PUBLIC
        )
        
        book2 = ContentItem(
            content_id="book-2",
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title="Advanced Book",
            visibility=ContentVisibility.PUBLIC
        )
        
        await content_service.create_content_item(book1)
        await content_service.create_content_item(book2)
        
        # Create relationship
        relationship = ContentRelationship(
            source_content_id="book-1",
            target_content_id="book-2",
            relationship_type=ContentRelationshipType.PREREQUISITE,
            strength=0.9,
            confidence=0.8
        )
        
        success = await content_service.create_relationship(relationship)
        assert success
        
        # Retrieve relationships
        relationships = await content_service.get_content_relationships("book-1")
        assert len(relationships) == 1
        assert relationships[0].target_content_id == "book-2"
        assert relationships[0].relationship_type == ContentRelationshipType.PREREQUISITE
        
        # Test relationship filtering
        prereq_relationships = await content_service.get_content_relationships(
            "book-1",
            relationship_type=ContentRelationshipType.PREREQUISITE
        )
        assert len(prereq_relationships) == 1


class TestMigrationService:
    """Test the migration service for legacy data."""
    
    @pytest.fixture
    async def migration_service(self):
        """Create a test migration service."""
        with patch('src.services.migration_service.get_content_service') as mock_get_service:
            mock_content_service = AsyncMock()
            mock_get_service.return_value = mock_content_service
            
            service = MigrationService()
            await service.initialize()
            yield service, mock_content_service
    
    @pytest.fixture
    def sample_legacy_book(self):
        """Create a sample legacy book for testing."""
        return LegacyBookMetadata(
            book_id="legacy-1",
            title="Legacy Test Book",
            author="Legacy Author",
            file_type="pdf",
            file_name="legacy.pdf",
            file_path="/tmp/legacy.pdf",
            file_size=2048000,
            upload_date=datetime.now() - timedelta(days=30),
            ingestion_date=datetime.now() - timedelta(days=29),
            text_length=25000,
            chunk_count=75,
            embedding_dimension=1536
        )
    
    async def test_migrate_single_book(self, migration_service, sample_legacy_book):
        """Test migrating a single legacy book."""
        service, mock_content_service = migration_service
        mock_content_service.create_content_item.return_value = True
        
        # Mock legacy metadata file
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open_with_data(sample_legacy_book)):
            
            success = await service.migrate_single_book(sample_legacy_book.book_id)
            assert success
            
            # Verify content service was called
            mock_content_service.create_content_item.assert_called_once()
    
    async def test_scan_legacy_data(self, migration_service):
        """Test scanning for legacy data."""
        service, mock_content_service = migration_service
        
        # Mock finding legacy metadata files
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.glob', return_value=[Path("/tmp/book1/metadata.json")]), \
             patch('builtins.open', mock_open_with_legacy_data()):
            
            mock_content_service.get_content_item.return_value = None  # Not yet migrated
            
            results = await service.scan_legacy_data()
            
            assert results["legacy_books_found"] == 1
            assert results["needs_migration"] == 1
            assert results["already_migrated"] == 0
    
    async def test_migration_validation(self, migration_service, sample_legacy_book):
        """Test migration validation (dry run)."""
        service, _ = migration_service
        
        with patch.object(service, 'scan_legacy_data') as mock_scan:
            mock_scan.return_value = {
                "book_details": [{
                    "book_id": sample_legacy_book.book_id,
                    "title": sample_legacy_book.title,
                    "already_migrated": False,
                    "metadata_file": "/tmp/metadata.json"
                }]
            }
            
            with patch('builtins.open', mock_open_with_data(sample_legacy_book)):
                results = await service.migrate_all_legacy_books(dry_run=True)
                
                assert results["total_books"] == 1
                assert results["migrated_books"] == 1  # Validation passed
                assert results["failed_books"] == 0


class TestEnhancedVectorDatabase:
    """Test the enhanced vector database with multi-module support."""
    
    @pytest.fixture
    async def enhanced_db(self):
        """Create a test enhanced vector database."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('src.utils.config.get_settings') as mock_settings:
                mock_settings.return_value.chroma_persist_directory = tmp_dir
                
                db = EnhancedChromaVectorDB()
                await db.initialize()
                yield db
    
    @pytest.fixture
    def sample_embedding_metadata(self):
        """Create sample embedding metadata."""
        return [
            EmbeddingMetadata(
                content_id="test-content-1",
                chunk_index=0,
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                visibility=ContentVisibility.PUBLIC,
                text_content="This is the first chunk of content.",
                chunk_length=35,
                semantic_tags=["programming", "beginner"],
                creator_id="user-1"
            ),
            EmbeddingMetadata(
                content_id="test-content-1",
                chunk_index=1,
                module_type=ModuleType.LIBRARY,
                content_type=ContentType.BOOK,
                visibility=ContentVisibility.PUBLIC,
                text_content="This is the second chunk of content.",
                chunk_length=36,
                semantic_tags=["programming", "intermediate"],
                creator_id="user-1"
            )
        ]
    
    async def test_add_documents_with_metadata(self, enhanced_db, sample_embedding_metadata):
        """Test adding documents with enhanced metadata."""
        documents = [meta.text_content for meta in sample_embedding_metadata]
        embeddings = [[0.1] * 1536 for _ in documents]  # Mock embeddings
        
        success = await enhanced_db.add_documents_with_metadata(
            collection_name="test_collection",
            documents=documents,
            embeddings=embeddings,
            embedding_metadata=sample_embedding_metadata
        )
        
        assert success
    
    async def test_query_with_permissions(self, enhanced_db, sample_embedding_metadata):
        """Test permission-aware querying."""
        # Add test documents
        documents = [meta.text_content for meta in sample_embedding_metadata]
        embeddings = [[0.1] * 1536 for _ in documents]
        
        await enhanced_db.add_documents_with_metadata(
            collection_name="test_collection",
            documents=documents,
            embeddings=embeddings,
            embedding_metadata=sample_embedding_metadata
        )
        
        # Test query with user permissions
        user = User(
            user_id="user-1",
            email="test@example.com",
            role=UserRole.READER
        )
        
        results = await enhanced_db.query_with_permissions(
            collection_name="test_collection",
            query_text="programming content",
            user=user,
            module_filter=ModuleType.LIBRARY
        )
        
        assert len(results["documents"]) > 0
        
        # Verify metadata is properly returned
        for metadata in results["metadatas"]:
            assert metadata["module_type"] == "library"
            assert metadata["visibility"] == "public"
    
    async def test_get_content_embeddings(self, enhanced_db, sample_embedding_metadata):
        """Test retrieving embeddings for specific content."""
        # Add test documents
        documents = [meta.text_content for meta in sample_embedding_metadata]
        embeddings = [[0.1] * 1536 for _ in documents]
        
        await enhanced_db.add_documents_with_metadata(
            collection_name="test_collection",
            documents=documents,
            embeddings=embeddings,
            embedding_metadata=sample_embedding_metadata
        )
        
        # Retrieve embeddings for content
        content_embeddings = await enhanced_db.get_content_embeddings("test-content-1")
        
        assert len(content_embeddings) == 2
        assert all(emb.content_id == "test-content-1" for emb in content_embeddings)
    
    async def test_delete_content_embeddings(self, enhanced_db, sample_embedding_metadata):
        """Test deleting embeddings for specific content."""
        # Add test documents
        documents = [meta.text_content for meta in sample_embedding_metadata]
        embeddings = [[0.1] * 1536 for _ in documents]
        
        await enhanced_db.add_documents_with_metadata(
            collection_name="test_collection",
            documents=documents,
            embeddings=embeddings,
            embedding_metadata=sample_embedding_metadata
        )
        
        # Delete embeddings
        success = await enhanced_db.delete_content_embeddings("test-content-1")
        assert success
        
        # Verify deletion
        remaining_embeddings = await enhanced_db.get_content_embeddings("test-content-1")
        assert len(remaining_embeddings) == 0


# ========================================
# Helper Functions for Testing
# ========================================

def mock_open_with_data(legacy_book: LegacyBookMetadata):
    """Create a mock open function that returns legacy book data."""
    from unittest.mock import mock_open
    
    book_data = {
        "book_id": legacy_book.book_id,
        "title": legacy_book.title,
        "author": legacy_book.author,
        "file_type": legacy_book.file_type,
        "file_name": legacy_book.file_name,
        "file_path": legacy_book.file_path,
        "file_size": legacy_book.file_size,
        "upload_date": legacy_book.upload_date.isoformat(),
        "ingestion_date": legacy_book.ingestion_date.isoformat() if legacy_book.ingestion_date else None,
        "text_length": legacy_book.text_length,
        "chunk_count": legacy_book.chunk_count,
        "embedding_dimension": legacy_book.embedding_dimension
    }
    
    return mock_open(read_data=json.dumps(book_data))


def mock_open_with_legacy_data():
    """Create a mock open function with sample legacy data."""
    from unittest.mock import mock_open
    
    legacy_data = {
        "book_id": "test-legacy-book",
        "title": "Test Legacy Book",
        "author": "Test Author",
        "file_type": "pdf",
        "file_name": "test.pdf",
        "file_path": "/tmp/test.pdf",
        "file_size": 1024000,
        "upload_date": datetime.now().isoformat(),
        "text_length": 10000,
        "chunk_count": 50
    }
    
    return mock_open(read_data=json.dumps(legacy_data))


# ========================================
# Integration Tests
# ========================================

class TestUnifiedContentIntegration:
    """Integration tests for the unified content system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_content_lifecycle(self):
        """Test complete content lifecycle from creation to deletion."""
        # This would be a comprehensive integration test
        # combining content service, enhanced database, and migration
        pass
    
    @pytest.mark.asyncio
    async def test_cross_module_relationships(self):
        """Test relationships between content from different modules."""
        # Test book -> course relationships
        # Test course -> marketplace item relationships
        pass
    
    @pytest.mark.asyncio
    async def test_permission_inheritance(self):
        """Test permission inheritance in content hierarchies."""
        # Test parent-child permission relationships
        # Test organization-level permissions
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])