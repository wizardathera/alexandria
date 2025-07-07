# RAG_GUIDELINES.md

## 🧠 RAG Database Development Guidelines

### Critical Architectural Principles

#### 1. Multi-Module Foundation First
- **Always design for all three modules**: Library, LMS, Marketplace
- **Unified content schema**: Use `content_items` as universal content container
- **Module-aware embeddings**: Include content_type, module, and permissions in all vector operations
- **Cross-module relationships**: Design content relationships from the start

#### 2. Permission-Aware RAG Operations
- **Every query must respect user permissions**: Filter by visibility and access rights
- **Role-based content access**: Reader, Educator, Creator, Admin roles with different access levels
- **Organization boundaries**: Multi-tenant isolation for Phase 2+ preparation
- **Permission caching**: Optimize performance while maintaining security

#### 3. Migration-Ready Architecture
- **Abstract vector operations**: Use VectorDatabaseInterface for easy Chroma → Supabase migration
- **Dual-write capability**: Design for gradual migration with data validation
- **Schema versioning**: Support backward compatibility during transitions
- **Performance benchmarking**: Maintain query response times during migrations

## RAG Implementation Standards

### Vector Embedding Requirements
```python
# Required metadata structure for all embeddings
embedding_metadata = {
    "content_id": str,           # UUID reference to content_items
    "content_type": str,         # "book", "course", "lesson", "marketplace_item"
    "module_name": str,          # "library", "lms", "marketplace"
    "chunk_type": str,           # "paragraph", "heading", "summary", "question"
    "visibility": str,           # "public", "private", "organization"
    "creator_id": str,           # Content creator/owner
    "organization_id": str,      # For multi-tenant isolation
    "semantic_tags": List[str],  # AI-extracted topics and categories
    "language": str,             # "en", "es", etc.
    "reading_level": str,        # "beginner", "intermediate", "advanced"
    "source_location": Dict      # Page, chapter, section, timestamp
}
```

### Search Query Standards
- **Multi-dimensional filtering**: Always include user permissions, content type, and module filters
- **Hybrid search approach**: Combine vector similarity with keyword matching for better relevance
- **Context-aware retrieval**: Use conversation history and user preferences for personalization
- **Performance requirements**: Maintain <3 second response times with enhanced metadata

### Content Relationship Mapping
- **AI-powered discovery**: Use embeddings to find related content automatically
- **Human curation override**: Allow content editors to verify and adjust AI relationships
- **Cross-module relationships**: Enable books → courses, courses → marketplace connections
- **Relationship strength scoring**: Use confidence scores for recommendation ranking

## 🤖 Agentic RAG Implementation

The RAG system must be **agentic**, not just simple retrieval:

### Core Components
1. **Query Understanding** - parse and interpret user intent
2. **Tool Selection** - choose appropriate tools for the query
3. **Information Synthesis** - combine multiple sources
4. **Response Generation** - provide clear, contextual answers

### Required Capabilities
- Understand complex, multi-part questions
- Choose between different information sources
- Summarize and explain outputs in plain language
- Handle follow-up questions and context

## Database Schema Evolution Rules

### Schema Change Management
- **Backward Compatibility**: New schema must support existing book data without migration
- **Incremental Adoption**: New features can be gradually enabled without breaking existing functionality
- **Migration Scripts**: All schema changes must include validated migration and rollback scripts
- **Data Validation**: Comprehensive testing with representative data before production deployment

### Content Type Extensibility
```python
# Content type hierarchy for future expansion
CONTENT_TYPES = {
    "library": ["book", "article", "document"],
    "lms": ["course", "lesson", "assessment", "quiz", "assignment"],
    "marketplace": ["marketplace_item", "premium_course", "digital_product"]
}
```

## AI and LLM Integration Guidelines

### Provider Abstraction
- **Multi-provider support**: Design for OpenAI + Anthropic + local models
- **Consistent interfaces**: Abstract LLM calls behind service interfaces
- **Cost monitoring**: Track token usage and API costs across all providers
- **Quality assurance**: A/B testing framework for comparing provider performance

### Response Generation Standards
- **Source citation**: Always include content source and confidence scores
- **Context management**: Maintain conversation history and context switching
- **Module-aware responses**: Tailor responses based on content module and user role
- **Quality metrics**: Monitor response accuracy, relevance, and user satisfaction

## Performance Optimization Guidelines

### Caching Strategy
```python
# Multi-level caching hierarchy
class RAGCachingStrategy:
    # L1: Query result caching (Redis) - 5 minutes TTL
    # L2: Embedding caching (Local/Redis) - 1 hour TTL
    # L3: Content metadata caching (Redis) - 24 hours TTL
    # L4: User permission caching (Redis) - 15 minutes TTL
    # L5: Relationship caching (Redis) - 24 hours TTL
```

### Database Optimization
- **Index Strategy**: Multi-dimensional indexes for complex queries
- **Query Optimization**: Efficient joins and filtering for cross-module searches
- **Connection Pooling**: Managed database connections for concurrent users
- **Performance Monitoring**: Real-time query performance tracking and alerting