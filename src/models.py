"""
Unified data models for the DBC application.

This module defines the core data models that support all three platform modules:
Smart Library, Learning Suite, and Marketplace. The unified content schema
enables seamless integration and migration across phases.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field, validator
import uuid


# ========================================
# Enums for Multi-Module Support
# ========================================

class ModuleType(str, Enum):
    """Platform modules supported by the DBC application."""
    LIBRARY = "library"
    LMS = "lms" 
    MARKETPLACE = "marketplace"


class ContentType(str, Enum):
    """Content types supported across all modules."""
    # Library content types
    BOOK = "book"
    ARTICLE = "article"
    DOCUMENT = "document"
    
    # LMS content types  
    COURSE = "course"
    LESSON = "lesson"
    ASSESSMENT = "assessment"
    QUIZ = "quiz"
    ASSIGNMENT = "assignment"
    
    # Marketplace content types
    MARKETPLACE_ITEM = "marketplace_item"
    PREMIUM_COURSE = "premium_course"
    DIGITAL_PRODUCT = "digital_product"


class ContentVisibility(str, Enum):
    """Content visibility levels for permission control."""
    PUBLIC = "public"           # Visible to all users
    PRIVATE = "private"         # Visible only to creator
    ORGANIZATION = "organization"  # Visible to organization members
    PREMIUM = "premium"         # Requires payment/subscription


class UserRole(str, Enum):
    """User roles for permission management."""
    READER = "reader"           # Basic user, can read content
    EDUCATOR = "educator"       # Can create courses and assessments
    CREATOR = "creator"         # Can publish to marketplace
    ADMIN = "admin"             # Full platform access


class ProcessingStatus(str, Enum):
    """Processing status for content items."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


# ========================================
# Unified Content Models
# ========================================

class ContentItem(BaseModel):
    """
    Unified content item supporting all module types.
    
    This is the core content model that supports books (Library),
    courses/lessons (LMS), and marketplace items (Marketplace).
    """
    
    # Core identifiers
    content_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    module_type: ModuleType = Field(..., description="Platform module this content belongs to")
    content_type: ContentType = Field(..., description="Specific type of content")
    
    # Basic metadata
    title: str = Field(..., description="Content title")
    description: Optional[str] = Field(None, description="Content description")
    author: Optional[str] = Field(None, description="Content author/creator")
    
    # File information (for file-based content)
    file_name: Optional[str] = Field(None, description="Original file name")
    file_path: Optional[str] = Field(None, description="Storage path for file")
    file_type: Optional[str] = Field(None, description="File extension/type")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    
    # Permission and visibility
    visibility: ContentVisibility = Field(default=ContentVisibility.PRIVATE)
    created_by: Optional[str] = Field(None, description="Creator user ID")
    organization_id: Optional[str] = Field(None, description="Organization ID for multi-tenant")
    
    # Processing metadata
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    text_length: Optional[int] = Field(None, description="Length of extracted text")
    chunk_count: Optional[int] = Field(None, description="Number of text chunks")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processed_at: Optional[datetime] = Field(None, description="When processing completed")
    
    # Module-specific metadata (flexible JSON field)
    module_metadata: Dict[str, Any] = Field(default_factory=dict, description="Module-specific data")
    
    # Content relationships
    parent_content_id: Optional[str] = Field(None, description="Parent content ID (for lessons in courses)")
    prerequisite_content_ids: List[str] = Field(default_factory=list, description="Required prerequisite content")
    
    # Semantic metadata
    topics: List[str] = Field(default_factory=list, description="AI-extracted topics")
    language: str = Field(default="en", description="Content language")
    reading_level: Optional[str] = Field(None, description="Reading difficulty level")
    
    @validator('content_id')
    def validate_content_id(cls, v):
        """Ensure content_id is a valid UUID string."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("content_id must be a valid UUID")
        return v
    
    @validator('module_metadata')
    def validate_module_metadata(cls, v, values):
        """Validate module-specific metadata based on module type."""
        module_type = values.get('module_type')
        content_type = values.get('content_type')
        
        # Module-specific validation logic can be added here
        if module_type == ModuleType.LMS and content_type == ContentType.COURSE:
            # Validate course-specific metadata
            pass
        elif module_type == ModuleType.MARKETPLACE:
            # Validate marketplace-specific metadata
            pass
        
        return v
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
    
    def mark_processed(self):
        """Mark content as successfully processed."""
        self.processing_status = ProcessingStatus.COMPLETED
        self.processed_at = datetime.now()
        self.update_timestamp()
    
    def get_module_metadata_for_type(self, metadata_type: str) -> Any:
        """Get specific metadata for the module type."""
        return self.module_metadata.get(metadata_type)
    
    def set_module_metadata(self, metadata_type: str, value: Any):
        """Set module-specific metadata."""
        self.module_metadata[metadata_type] = value
        self.update_timestamp()


# ========================================
# Module-Specific Content Extensions
# ========================================

class LibraryBookMetadata(BaseModel):
    """Extended metadata for library books."""
    isbn: Optional[str] = None
    publisher: Optional[str] = None
    publication_date: Optional[datetime] = None
    genre: Optional[str] = None
    page_count: Optional[int] = None
    book_format: Optional[str] = None  # "print", "ebook", "audiobook"
    
    # Reading metrics
    average_reading_time: Optional[int] = None  # minutes
    difficulty_score: Optional[float] = None  # 0.0-1.0
    
    # Content structure
    has_table_of_contents: bool = False
    chapter_count: Optional[int] = None
    has_index: bool = False


class LMSCourseMetadata(BaseModel):
    """Extended metadata for LMS courses."""
    course_duration: Optional[int] = None  # minutes
    skill_level: Optional[str] = None  # "beginner", "intermediate", "advanced"
    learning_objectives: List[str] = Field(default_factory=list)
    assessment_count: Optional[int] = None
    lesson_count: Optional[int] = None
    
    # Course structure
    is_self_paced: bool = True
    has_certificate: bool = False
    completion_rate: Optional[float] = None  # 0.0-1.0
    
    # Pricing (for marketplace integration)
    price: Optional[float] = None
    currency: str = "USD"


class LMSLessonMetadata(BaseModel):
    """Extended metadata for LMS lessons."""
    lesson_order: int = 0
    estimated_duration: Optional[int] = None  # minutes
    lesson_type: str = "content"  # "content", "video", "interactive", "assessment"
    
    # Learning tracking
    completion_criteria: Dict[str, Any] = Field(default_factory=dict)
    interactive_elements: List[str] = Field(default_factory=list)


class MarketplaceItemMetadata(BaseModel):
    """Extended metadata for marketplace items."""
    price: float = 0.0
    currency: str = "USD"
    license_type: str = "standard"  # "standard", "extended", "commercial"
    
    # Sales metadata
    purchase_count: int = 0
    rating: Optional[float] = None  # 0.0-5.0
    review_count: int = 0
    
    # Creator monetization
    revenue_share: float = 0.85  # 85% to creator, 15% platform fee
    royalty_model: str = "percentage"  # "percentage", "fixed"


# ========================================
# Content Relationships
# ========================================

class ContentRelationshipType(str, Enum):
    """Types of relationships between content items."""
    PREREQUISITE = "prerequisite"      # Content A is required before B
    SUPPLEMENT = "supplement"          # Content A supplements B
    SEQUENCE = "sequence"              # Content A comes before B in sequence
    ALTERNATIVE = "alternative"        # Content A is alternative to B
    REFERENCE = "reference"            # Content A references B
    SIMILARITY = "similarity"          # Content A is similar to B
    CONTRADICTION = "contradiction"    # Content A contradicts B
    ELABORATION = "elaboration"        # Content A elaborates on B


class ContentRelationship(BaseModel):
    """Relationship between two content items."""
    relationship_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_content_id: str = Field(..., description="Source content ID")
    target_content_id: str = Field(..., description="Target content ID") 
    relationship_type: ContentRelationshipType = Field(..., description="Type of relationship")
    
    # Relationship metadata
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Relationship strength 0.0-1.0")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="AI confidence in relationship")
    discovered_by: str = Field(default="ai", description="How relationship was discovered")
    human_verified: bool = Field(default=False, description="Whether human verified")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    verified_at: Optional[datetime] = Field(None, description="When human verified")
    verified_by: Optional[str] = Field(None, description="User who verified")
    
    # Context metadata
    context: Optional[str] = Field(None, description="Context where relationship applies")
    bidirectional: bool = Field(default=False, description="Whether relationship works both ways")
    
    def verify_by_human(self, user_id: str):
        """Mark relationship as human-verified."""
        self.human_verified = True
        self.verified_at = datetime.now()
        self.verified_by = user_id
    
    def update_strength(self, new_strength: float):
        """Update relationship strength based on user interactions."""
        self.strength = max(0.0, min(1.0, new_strength))


# ========================================
# Enhanced Vector Embeddings
# ========================================

class EmbeddingMetadata(BaseModel):
    """Enhanced metadata for vector embeddings supporting multi-module content."""
    
    # Core embedding info
    embedding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str = Field(..., description="Associated content ID")
    chunk_index: int = Field(..., description="Chunk index within content")
    
    # Module and content awareness
    module_type: ModuleType = Field(..., description="Platform module")
    content_type: ContentType = Field(..., description="Content type")
    chunk_type: str = Field(default="paragraph", description="Type of text chunk")
    
    # Permission metadata
    visibility: ContentVisibility = Field(..., description="Content visibility")
    creator_id: Optional[str] = Field(None, description="Content creator/owner")
    organization_id: Optional[str] = Field(None, description="Organization for multi-tenant")
    
    # Semantic metadata
    semantic_tags: List[str] = Field(default_factory=list, description="AI-extracted topics")
    language: str = Field(default="en", description="Content language")
    reading_level: Optional[str] = Field(None, description="Reading difficulty level")
    
    # Source location metadata
    source_location: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Location within source (page, chapter, section, etc.)"
    )
    
    # Text chunk details
    text_content: str = Field(..., description="The actual text chunk")
    chunk_length: int = Field(..., description="Length of text chunk")
    
    # Processing metadata
    embedding_model: str = Field(default="text-embedding-ada-002", description="Model used for embedding")
    embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Importance and quality scores
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content importance")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Content quality")
    
    def add_semantic_tag(self, tag: str):
        """Add a semantic tag if not already present."""
        if tag not in self.semantic_tags:
            self.semantic_tags.append(tag)
    
    def update_importance_score(self, score: float):
        """Update importance score with validation."""
        self.importance_score = max(0.0, min(1.0, score))
    
    def get_permission_filter(self) -> Dict[str, Any]:
        """Get permission filter for vector search."""
        return {
            "visibility": self.visibility.value,
            "creator_id": self.creator_id,
            "organization_id": self.organization_id
        }


# ========================================
# Conversation History Models
# ========================================

class ConversationStatus(str, Enum):
    """Status of a conversation."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MessageRole(str, Enum):
    """Role of a message in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Conversation(BaseModel):
    """Conversation model for chat history."""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = Field(None, description="Conversation title")
    
    # Content context
    content_id: Optional[str] = Field(None, description="Associated content ID")
    module_type: Optional[ModuleType] = Field(None, description="Module context")
    
    # User and permission context
    user_id: Optional[str] = Field(None, description="User who owns this conversation")
    organization_id: Optional[str] = Field(None, description="Organization context")
    
    # Conversation metadata
    status: ConversationStatus = Field(default=ConversationStatus.ACTIVE)
    message_count: int = Field(default=0, description="Number of messages in conversation")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_message_at: Optional[datetime] = Field(None, description="Timestamp of last message")
    
    # Settings and preferences
    settings: Dict[str, Any] = Field(default_factory=dict, description="Conversation-specific settings")
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
    
    def add_message(self):
        """Increment message count and update timestamps."""
        self.message_count += 1
        self.last_message_at = datetime.now()
        self.update_timestamp()
    
    def generate_title_from_first_message(self, first_message: str, max_length: int = 50) -> str:
        """Generate conversation title from first user message."""
        if not first_message:
            return "New Conversation"
        
        # Clean and truncate the message
        title = first_message.strip()
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        return title


class ChatMessage(BaseModel):
    """Individual chat message model."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(..., description="Parent conversation ID")
    
    # Message content
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    
    # Context metadata
    content_id: Optional[str] = Field(None, description="Associated content ID if relevant")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="RAG sources for assistant messages")
    
    # Processing metadata
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage for this message")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    confidence_score: Optional[float] = Field(None, description="Confidence score for assistant responses")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Message status and quality
    is_edited: bool = Field(default=False, description="Whether message was edited")
    is_deleted: bool = Field(default=False, description="Whether message was deleted")
    quality_rating: Optional[int] = Field(None, ge=1, le=5, description="User quality rating 1-5")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    
    def mark_edited(self):
        """Mark message as edited."""
        self.is_edited = True
        self.metadata["edited_at"] = datetime.now().isoformat()
    
    def add_source(self, source: Dict[str, Any]):
        """Add a source to the message."""
        if source not in self.sources:
            self.sources.append(source)
    
    def set_quality_rating(self, rating: int):
        """Set user quality rating for the message."""
        if 1 <= rating <= 5:
            self.quality_rating = rating
            self.metadata["rated_at"] = datetime.now().isoformat()


class ConversationHistory(BaseModel):
    """Complete conversation history including all messages."""
    conversation: Conversation
    messages: List[ChatMessage] = Field(default_factory=list)
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages from the conversation."""
        return sorted(self.messages, key=lambda m: m.created_at, reverse=True)[:limit]
    
    def get_context_messages(self, limit: int = 5) -> List[Dict[str, str]]:
        """Get recent messages formatted for LLM context."""
        recent_messages = self.get_recent_messages(limit)
        context = []
        
        for message in reversed(recent_messages):  # Reverse to get chronological order
            if not message.is_deleted:
                context.append({
                    "role": message.role.value,
                    "content": message.content
                })
        
        return context
    
    def get_message_count_by_role(self) -> Dict[str, int]:
        """Get count of messages by role."""
        counts = {}
        for message in self.messages:
            if not message.is_deleted:
                role = message.role.value
                counts[role] = counts.get(role, 0) + 1
        return counts


# ========================================
# User and Permission Models
# ========================================

class User(BaseModel):
    """User model for multi-module platform."""
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str = Field(..., description="User email address")
    username: Optional[str] = Field(None, description="Display username")
    
    # Role and permissions
    role: UserRole = Field(default=UserRole.READER)
    organization_id: Optional[str] = Field(None, description="Organization membership")
    permissions: List[str] = Field(default_factory=list, description="Specific permissions")
    
    # Profile information
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    
    # Settings and preferences
    preferences: Dict[str, Any] = Field(default_factory=dict)
    notification_settings: Dict[str, bool] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    
    # Account status
    is_active: bool = True
    is_verified: bool = False
    subscription_tier: Optional[str] = None  # "free", "pro", "enterprise"
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or self.role == UserRole.ADMIN
    
    def can_access_content(self, content: ContentItem) -> bool:
        """Check if user can access specific content."""
        # Public content accessible to all
        if content.visibility == ContentVisibility.PUBLIC:
            return True
        
        # Private content only accessible to creator
        if content.visibility == ContentVisibility.PRIVATE:
            return content.created_by == self.user_id
        
        # Organization content accessible to org members
        if content.visibility == ContentVisibility.ORGANIZATION:
            return (self.organization_id == content.organization_id and 
                   self.organization_id is not None)
        
        # Premium content requires subscription or payment
        if content.visibility == ContentVisibility.PREMIUM:
            return self.subscription_tier in ["pro", "enterprise"]
        
        return False


# ========================================
# Migration and Compatibility Models
# ========================================

@dataclass
class LegacyBookMetadata:
    """Legacy book metadata for backward compatibility."""
    book_id: str
    title: str
    author: Optional[str] = None
    file_type: str = ""
    file_name: str = ""
    file_path: str = ""
    file_size: int = 0
    upload_date: datetime = field(default_factory=datetime.now)
    ingestion_date: Optional[datetime] = None
    user_id: Optional[str] = None
    text_length: int = 0
    chunk_count: int = 0
    embedding_dimension: int = 0
    chunking_strategy: str = "recursive"
    embedding_model: str = "text-embedding-ada-002"
    
    def to_content_item(self) -> ContentItem:
        """Convert legacy book metadata to unified content item."""
        return ContentItem(
            content_id=self.book_id,
            module_type=ModuleType.LIBRARY,
            content_type=ContentType.BOOK,
            title=self.title,
            author=self.author,
            file_name=self.file_name,
            file_path=self.file_path,
            file_type=self.file_type,
            file_size=self.file_size,
            created_by=self.user_id,
            processing_status=ProcessingStatus.COMPLETED if self.ingestion_date else ProcessingStatus.PENDING,
            text_length=self.text_length,
            chunk_count=self.chunk_count,
            created_at=self.upload_date,
            processed_at=self.ingestion_date,
            module_metadata={
                "embedding_dimension": self.embedding_dimension,
                "chunking_strategy": self.chunking_strategy,
                "embedding_model": self.embedding_model,
                "library_metadata": LibraryBookMetadata().dict()
            }
        )


# ========================================
# Database Schema Utilities
# ========================================

def get_content_type_for_module(module: ModuleType) -> List[ContentType]:
    """Get valid content types for a specific module."""
    content_types = {
        ModuleType.LIBRARY: [ContentType.BOOK, ContentType.ARTICLE, ContentType.DOCUMENT],
        ModuleType.LMS: [ContentType.COURSE, ContentType.LESSON, ContentType.ASSESSMENT, 
                        ContentType.QUIZ, ContentType.ASSIGNMENT],
        ModuleType.MARKETPLACE: [ContentType.MARKETPLACE_ITEM, ContentType.PREMIUM_COURSE, 
                               ContentType.DIGITAL_PRODUCT]
    }
    return content_types.get(module, [])


def validate_content_for_module(content: ContentItem) -> bool:
    """Validate that content type is appropriate for module."""
    valid_types = get_content_type_for_module(content.module_type)
    return content.content_type in valid_types


# ========================================
# Export All Models
# ========================================

__all__ = [
    # Enums
    "ModuleType", "ContentType", "ContentVisibility", "UserRole", "ProcessingStatus",
    "ContentRelationshipType", "ConversationStatus", "MessageRole",
    
    # Core Models
    "ContentItem", "ContentRelationship", "EmbeddingMetadata", "User",
    
    # Conversation Models
    "Conversation", "ChatMessage", "ConversationHistory",
    
    # Module-Specific Extensions
    "LibraryBookMetadata", "LMSCourseMetadata", "LMSLessonMetadata", "MarketplaceItemMetadata",
    
    # Legacy Compatibility
    "LegacyBookMetadata",
    
    # Utilities
    "get_content_type_for_module", "validate_content_for_module"
]