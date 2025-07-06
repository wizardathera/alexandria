**This document is synchronized with PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md as of 2025-07-05.**

# 📊 Data Model & Storage Architecture

## Overview

This document details the data storage architecture for the Alexandria platform, covering database schemas, vector storage strategies, and content management systems.

For additional architecture details, see:
- [Architecture Overview](ARCHITECTURE_OVERVIEW.md)
- [Frontend Architecture](ARCHITECTURE_FRONTEND.md)
- [Backend Architecture](ARCHITECTURE_BACKEND.md)
- [AI Services & RAG](ARCHITECTURE_AI_SERVICES.md)

## 🗄️ Database Schema Overview

### Core Entities

#### Users & Organizations
```sql
-- Core user management
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user', -- 'user', 'educator', 'creator', 'admin'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan_type VARCHAR(50) DEFAULT 'free', -- 'free', 'premium', 'enterprise'
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    org_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    role VARCHAR(50) DEFAULT 'member', -- 'member', 'admin', 'owner'
    permissions JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Authentication & sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    preferences_json JSONB DEFAULT '{}',
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Content Management
```sql
-- Universal content items table
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    content_type VARCHAR(50) NOT NULL, -- 'book', 'course', 'lesson', 'assessment', 'marketplace_item'
    module_name VARCHAR(20) NOT NULL, -- 'library', 'lms', 'marketplace'
    creator_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    
    -- Content metadata
    description TEXT,
    metadata_json JSONB DEFAULT '{}',
    tags VARCHAR(255)[],
    
    -- Access control
    visibility VARCHAR(20) DEFAULT 'private', -- 'public', 'private', 'organization', 'marketplace'
    access_permissions JSONB DEFAULT '{}',
    
    -- File storage
    file_path VARCHAR(1000),
    file_size INTEGER,
    file_format VARCHAR(50),
    
    -- Status and lifecycle
    status VARCHAR(50) DEFAULT 'draft', -- 'draft', 'published', 'archived', 'deleted'
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Library-specific content
CREATE TABLE books (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    isbn VARCHAR(20),
    author VARCHAR(500),
    publisher VARCHAR(255),
    publication_date DATE,
    page_count INTEGER,
    genre VARCHAR(100),
    reading_level VARCHAR(50), -- 'beginner', 'intermediate', 'advanced'
    language VARCHAR(10) DEFAULT 'en'
);

-- LMS-specific content
CREATE TABLE courses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    course_code VARCHAR(50),
    difficulty_level VARCHAR(50),
    estimated_duration INTEGER, -- minutes
    prerequisites JSONB DEFAULT '[]',
    learning_objectives JSONB DEFAULT '[]'
);

CREATE TABLE lessons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    course_id UUID REFERENCES courses(id) ON DELETE CASCADE,
    order_index INTEGER NOT NULL,
    lesson_type VARCHAR(50), -- 'text', 'video', 'interactive', 'assessment'
    duration INTEGER, -- minutes
    prerequisites JSONB DEFAULT '[]'
);

CREATE TABLE assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES lessons(id) ON DELETE CASCADE,
    questions_json JSONB NOT NULL,
    passing_score DECIMAL(5,2) DEFAULT 70.00,
    time_limit INTEGER, -- minutes, NULL for unlimited
    max_attempts INTEGER DEFAULT 3
);

-- Marketplace-specific content
CREATE TABLE marketplace_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    price DECIMAL(10,2),
    currency VARCHAR(3) DEFAULT 'USD',
    license_type VARCHAR(50), -- 'personal', 'educational', 'commercial'
    revenue_share DECIMAL(5,2) DEFAULT 70.00, -- Creator's share percentage
    featured_until TIMESTAMP NULL,
    total_sales INTEGER DEFAULT 0,
    total_revenue DECIMAL(12,2) DEFAULT 0.00
);
```

#### User Interactions and Progress
```sql
-- User libraries and collections
CREATE TABLE user_libraries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    progress DECIMAL(5,2) DEFAULT 0.00, -- Percentage completed
    current_position JSONB, -- Page, chapter, timestamp, etc.
    notes TEXT,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    added_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Student progress tracking (LMS)
CREATE TABLE student_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    course_id UUID REFERENCES courses(id) ON DELETE CASCADE,
    lesson_id UUID REFERENCES lessons(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'not_started', -- 'not_started', 'in_progress', 'completed', 'failed'
    score DECIMAL(5,2),
    time_spent INTEGER, -- minutes
    completed_at TIMESTAMP NULL,
    started_at TIMESTAMP DEFAULT NOW()
);

-- User queries and conversations
CREATE TABLE user_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id),
    query_text TEXT NOT NULL,
    response TEXT,
    context_used JSONB, -- Retrieved context and sources
    response_quality DECIMAL(3,2), -- User feedback on response quality
    processing_time INTEGER, -- milliseconds
    created_at TIMESTAMP DEFAULT NOW()
);

-- User notes and annotations
CREATE TABLE user_notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    note_text TEXT NOT NULL,
    note_type VARCHAR(50) DEFAULT 'general', -- 'highlight', 'comment', 'question', 'general'
    location_data JSONB, -- Position in content (page, paragraph, etc.)
    is_public BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User activity tracking
CREATE TABLE user_activity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    action_type VARCHAR(50) NOT NULL, -- 'login', 'upload', 'query', 'purchase', etc.
    resource_id UUID, -- ID of affected resource
    resource_type VARCHAR(50), -- Type of affected resource
    metadata_json JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### System Analytics and Metrics
```sql
-- System-wide usage metrics
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    metadata_json JSONB DEFAULT '{}',
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Error logging and monitoring
CREATE TABLE error_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    request_data JSONB,
    context_data JSONB,
    severity VARCHAR(20) DEFAULT 'error', -- 'debug', 'info', 'warning', 'error', 'critical'
    resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🔍 Vector Database Architecture

### Enhanced Vector Database Strategy

#### Phase 1: Enhanced Chroma (Local Development)
```python
# Enhanced local vector store with multi-module support
collection = chroma_client.create_collection(
    name="content_embeddings",
    embedding_function=openai_embedding_function,
    metadata={
        "description": "Multi-module content embeddings",
        "supports": ["books", "courses", "lessons", "marketplace_items"]
    }
)

# Enhanced metadata structure for future compatibility
enhanced_metadata = {
    "content_id": str,
    "content_type": str,  # "book", "course", "lesson", "marketplace_item"
    "module": str,        # "library", "lms", "marketplace"
    "chunk_type": str,    # "paragraph", "heading", "summary", "question"
    "visibility": str,    # "public", "private", "organization"
    "language": str,
    "reading_level": str,
    "semantic_tags": List[str],
    "user_permissions": List[str],
    "source_location": Dict[str, Any]
}
```

#### Phase 2+: Supabase pgvector (Production)
```sql
-- Enhanced vector storage supporting all content types
CREATE TABLE content_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embedding dimension
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) NOT NULL, -- 'paragraph', 'heading', 'summary', 'question', 'answer'
    
    -- Module and content type information
    content_type VARCHAR(50) NOT NULL, -- 'book', 'course', 'lesson', 'assessment', 'marketplace_item'
    module_name VARCHAR(20) NOT NULL,  -- 'library', 'lms', 'marketplace'
    
    -- Source location and context
    source_location JSONB, -- page, chapter, section, timestamp, etc.
    
    -- Enhanced metadata for search and recommendations
    module_metadata JSONB DEFAULT '{}', -- Module-specific metadata
    semantic_tags JSONB DEFAULT '[]',   -- AI-extracted tags and categories
    
    -- Search optimization fields
    text_length INTEGER NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    reading_level VARCHAR(20),
    difficulty_level VARCHAR(20),
    
    -- Access control and permissions
    visibility VARCHAR(20) DEFAULT 'private', -- 'public', 'private', 'organization', 'marketplace'
    access_permissions JSONB DEFAULT '{}', -- Detailed permission structure
    creator_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Multi-dimensional indexes for optimal query performance
CREATE INDEX idx_content_embeddings_module_type ON content_embeddings(module_name, content_type);
CREATE INDEX idx_content_embeddings_visibility ON content_embeddings(visibility);
CREATE INDEX idx_content_embeddings_creator ON content_embeddings(creator_id);
CREATE INDEX idx_content_embeddings_org ON content_embeddings(organization_id);
CREATE INDEX idx_content_embeddings_language ON content_embeddings(language);
CREATE INDEX idx_content_embeddings_difficulty ON content_embeddings(difficulty_level);
CREATE INDEX idx_content_embeddings_tags ON content_embeddings USING GIN(semantic_tags);
CREATE INDEX idx_content_embeddings_metadata ON content_embeddings USING GIN(module_metadata);

-- Vector similarity search indexes (multiple distance metrics)
CREATE INDEX idx_content_embeddings_vector_cosine ON content_embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_content_embeddings_vector_l2 ON content_embeddings 
    USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
CREATE INDEX idx_content_embeddings_vector_ip ON content_embeddings 
    USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);

-- Composite indexes for complex queries
CREATE INDEX idx_content_complex_search ON content_embeddings(
    visibility, module_name, content_type, language, difficulty_level
);
```

#### Content Relationships for Recommendations
```sql
-- Enhanced content relationships for cross-module recommendations
CREATE TABLE content_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    target_content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- 'prerequisite', 'related', 'sequence', 'alternative', 'follows'
    relationship_strength DECIMAL(3,2) NOT NULL CHECK (relationship_strength >= 0 AND relationship_strength <= 1),
    relationship_context JSONB DEFAULT '{}', -- Additional context (difficulty progression, topic overlap, etc.)
    
    -- AI-generated vs human-curated
    created_by_ai BOOLEAN DEFAULT true,
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    human_verified BOOLEAN DEFAULT false,
    
    -- Cross-module relationship support
    cross_module BOOLEAN DEFAULT false,
    source_module VARCHAR(20),
    target_module VARCHAR(20),
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Prevent duplicate relationships
    UNIQUE(source_content_id, target_content_id, relationship_type)
);

-- Indexes for relationship queries
CREATE INDEX idx_content_relationships_source ON content_relationships(source_content_id, relationship_strength DESC);
CREATE INDEX idx_content_relationships_target ON content_relationships(target_content_id, relationship_strength DESC);
CREATE INDEX idx_content_relationships_type ON content_relationships(relationship_type, relationship_strength DESC);
CREATE INDEX idx_content_relationships_cross_module ON content_relationships(cross_module, source_module, target_module);
```

#### User Interaction Tracking for Personalization
```sql
-- Enhanced user interactions for better recommendations
CREATE TABLE user_content_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL, -- 'view', 'query', 'bookmark', 'rate', 'complete', 'share'
    
    -- Interaction details
    interaction_data JSONB DEFAULT '{}', -- Query text, rating, progress percentage, time spent, etc.
    interaction_quality DECIMAL(3,2), -- Quality score (e.g., time spent, completion rate)
    
    -- Context information
    session_id VARCHAR(100),
    device_type VARCHAR(20),
    access_method VARCHAR(20), -- 'web', 'mobile', 'api'
    
    -- Timing and duration
    interaction_duration INTEGER, -- seconds
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for user behavior analysis
CREATE INDEX idx_user_interactions_user_time ON user_content_interactions(user_id, created_at DESC);
CREATE INDEX idx_user_interactions_content ON user_content_interactions(content_id, interaction_type);
CREATE INDEX idx_user_interactions_quality ON user_content_interactions(interaction_quality DESC, created_at DESC);
```

## 💳 E-commerce & Payment Schema

### Payment and Purchase Management
```sql
-- Payment and purchasing tables
CREATE TABLE purchases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    stripe_payment_intent_id VARCHAR(255) UNIQUE,
    total_amount DECIMAL(10,2) NOT NULL CHECK (total_amount >= 0),
    currency VARCHAR(3) DEFAULT 'USD',
    tax_amount DECIMAL(10,2) DEFAULT 0.00 CHECK (tax_amount >= 0),
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'completed', 'failed', 'refunded'
    billing_address JSONB,
    purchased_at TIMESTAMP DEFAULT NOW(),
    refunded_at TIMESTAMP NULL
);

CREATE TABLE purchase_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    purchase_id UUID REFERENCES purchases(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    item_type VARCHAR(50) NOT NULL, -- 'book', 'course', 'subscription'
    price DECIMAL(10,2) NOT NULL CHECK (price >= 0),
    license_type VARCHAR(50) DEFAULT 'personal', -- 'personal', 'educational', 'commercial'
    access_expires_at TIMESTAMP NULL -- NULL for permanent access
);

CREATE TABLE user_content_access (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    purchase_id UUID REFERENCES purchases(id) ON DELETE SET NULL,
    access_type VARCHAR(50) NOT NULL, -- 'purchased', 'trial', 'subscription'
    access_granted_at TIMESTAMP DEFAULT NOW(),
    access_expires_at TIMESTAMP NULL,
    last_accessed_at TIMESTAMP NULL,
    
    -- Prevent duplicate access records
    UNIQUE(user_id, content_id, access_type)
);

-- Revenue tracking and analytics
CREATE TABLE revenue_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    content_id UUID REFERENCES content_items(id),
    creator_id UUID REFERENCES users(id),
    gross_revenue DECIMAL(12,2) DEFAULT 0.00,
    net_revenue DECIMAL(12,2) DEFAULT 0.00,
    platform_fee DECIMAL(12,2) DEFAULT 0.00,
    creator_share DECIMAL(12,2) DEFAULT 0.00,
    transaction_count INTEGER DEFAULT 0,
    refund_count INTEGER DEFAULT 0,
    refund_amount DECIMAL(12,2) DEFAULT 0.00,
    
    UNIQUE(date, content_id)
);
```

## 🔍 Discovery & Recommendation Schema

### AI-Powered Recommendation System
```sql
-- User behavior and preference tracking
CREATE TABLE user_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) NOT NULL, -- 'view', 'like', 'share', 'read', 'purchase'
    interaction_data JSONB DEFAULT '{}', -- Duration, completion rate, rating, etc.
    context_data JSONB DEFAULT '{}', -- Device, time, referrer, mood, etc.
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    preference_type VARCHAR(50) NOT NULL, -- 'genre', 'author', 'difficulty', 'length'
    preference_value VARCHAR(255) NOT NULL,
    preference_strength DECIMAL(3,2) NOT NULL CHECK (preference_strength >= 0 AND preference_strength <= 1),
    learned_from VARCHAR(50) DEFAULT 'implicit', -- 'explicit', 'implicit', 'inferred'
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(user_id, preference_type, preference_value)
);

CREATE TABLE content_similarities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_a_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    content_b_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    similarity_type VARCHAR(50) NOT NULL, -- 'thematic', 'stylistic', 'difficulty', 'audience'
    similarity_score DECIMAL(3,2) NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    calculation_method VARCHAR(50) NOT NULL, -- 'embedding', 'metadata', 'collaborative'
    calculated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(content_a_id, content_b_id, similarity_type)
);

-- Recommendation tracking and optimization
CREATE TABLE recommendation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_context JSONB DEFAULT '{}', -- Request context and parameters
    algorithm_used VARCHAR(50) NOT NULL,
    recommendations_shown JSONB NOT NULL, -- Array of recommended content IDs
    user_feedback JSONB DEFAULT '{}', -- Clicks, dismissals, ratings
    session_quality_score DECIMAL(3,2), -- Post-session quality assessment
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🎨 Theme System Schema

### Theme Management and Customization
```sql
-- Theme management and user preferences
CREATE TABLE user_themes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    theme_name VARCHAR(100) NOT NULL,
    base_theme VARCHAR(50) NOT NULL, -- Reference to ReadingEnvironment enum
    customizations JSONB DEFAULT '{}', -- Color, typography, layout overrides
    is_custom BOOLEAN DEFAULT false,
    is_public BOOLEAN DEFAULT false, -- For community sharing
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE theme_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    primary_theme_id UUID REFERENCES user_themes(id),
    dark_mode_preference VARCHAR(10) DEFAULT 'auto', -- 'auto', 'light', 'dark'
    device_preferences JSONB DEFAULT '{}', -- Different themes per device type
    contextual_preferences JSONB DEFAULT '{}', -- Time-based, genre-based themes
    auto_adapt_enabled BOOLEAN DEFAULT true,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE community_themes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    creator_id UUID REFERENCES users(id) ON DELETE CASCADE,
    theme_data JSONB NOT NULL, -- Complete theme configuration
    name VARCHAR(100) NOT NULL,
    description TEXT,
    tags VARCHAR(255)[] DEFAULT '{}', -- For discovery and categorization
    downloads_count INTEGER DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0.0 CHECK (rating_average >= 0 AND rating_average <= 5),
    rating_count INTEGER DEFAULT 0,
    is_featured BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Theme usage analytics
CREATE TABLE theme_usage_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    theme_id UUID, -- Can reference user_themes or community_themes
    usage_duration INTEGER NOT NULL, -- Minutes spent using theme
    reading_session_id UUID, -- Link to reading session
    device_type VARCHAR(50),
    context_data JSONB DEFAULT '{}', -- Time of day, book genre, etc.
    user_satisfaction_rating INTEGER CHECK (user_satisfaction_rating >= 1 AND user_satisfaction_rating <= 5),
    recorded_at TIMESTAMP DEFAULT NOW()
);
```

## 🔐 Row-Level Security (RLS) Policies

### User Data Protection
```sql
-- Enable RLS on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_libraries ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_queries ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY user_isolation_policy ON users
    FOR ALL USING (auth.uid() = id);

CREATE POLICY user_preferences_policy ON user_preferences
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY user_libraries_policy ON user_libraries
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY user_notes_policy ON user_notes
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY user_queries_policy ON user_queries
    FOR ALL USING (auth.uid() = user_id);

-- Content visibility policies
ALTER TABLE content_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY content_visibility_policy ON content_items
    FOR SELECT USING (
        visibility = 'public' OR
        (visibility = 'private' AND creator_id = auth.uid()) OR
        (visibility = 'organization' AND organization_id IN (
            SELECT org_id FROM user_organizations WHERE user_id = auth.uid()
        ))
    );
```

## 💾 Backup and Disaster Recovery

### Backup Strategy
```sql
-- Automated daily backups
CREATE OR REPLACE FUNCTION create_daily_backup()
RETURNS void AS $$
BEGIN
    -- Create backup tables with timestamp
    EXECUTE format('CREATE TABLE users_backup_%s AS SELECT * FROM users', 
                   to_char(now(), 'YYYY_MM_DD'));
    EXECUTE format('CREATE TABLE content_items_backup_%s AS SELECT * FROM content_items', 
                   to_char(now(), 'YYYY_MM_DD'));
    -- Add other critical tables...
END;
$$ LANGUAGE plpgsql;

-- Schedule daily backups
SELECT cron.schedule('daily-backup', '0 2 * * *', 'SELECT create_daily_backup();');
```

### Data Retention Policies
```sql
-- Clean up old analytics data (older than 2 years)
CREATE OR REPLACE FUNCTION cleanup_old_analytics()
RETURNS void AS $$
BEGIN
    DELETE FROM usage_metrics WHERE date < NOW() - INTERVAL '2 years';
    DELETE FROM theme_usage_analytics WHERE recorded_at < NOW() - INTERVAL '2 years';
    DELETE FROM user_content_interactions WHERE created_at < NOW() - INTERVAL '2 years';
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly cleanup
SELECT cron.schedule('monthly-cleanup', '0 3 1 * *', 'SELECT cleanup_old_analytics();');
```

## 📈 Performance Optimization

### Database Indexing Strategy
```sql
-- Composite indexes for common query patterns
CREATE INDEX idx_content_items_module_visibility ON content_items(module_name, visibility, created_at DESC);
CREATE INDEX idx_user_libraries_progress ON user_libraries(user_id, progress DESC, last_accessed DESC);
CREATE INDEX idx_user_queries_content_time ON user_queries(content_id, created_at DESC);

-- Partial indexes for frequently queried subsets
CREATE INDEX idx_published_content ON content_items(created_at DESC) WHERE status = 'published';
CREATE INDEX idx_public_content ON content_items(title) WHERE visibility = 'public';

-- Text search indexes
CREATE INDEX idx_content_items_fts ON content_items USING gin(to_tsvector('english', title || ' ' || description));
CREATE INDEX idx_books_fts ON books USING gin(to_tsvector('english', author || ' ' || genre));
```

### Query Optimization
```sql
-- Materialized views for expensive aggregate queries
CREATE MATERIALIZED VIEW content_stats AS
SELECT 
    content_type,
    module_name,
    COUNT(*) as total_count,
    AVG(CASE WHEN ul.rating IS NOT NULL THEN ul.rating END) as avg_rating,
    COUNT(CASE WHEN ul.rating IS NOT NULL THEN 1 END) as rating_count
FROM content_items ci
LEFT JOIN user_libraries ul ON ci.id = ul.content_id
WHERE ci.status = 'published'
GROUP BY content_type, module_name;

-- Refresh materialized views nightly
SELECT cron.schedule('refresh-stats', '0 1 * * *', 'REFRESH MATERIALIZED VIEW content_stats;');
```

## 🔗 Related Documentation

- [Backend Architecture →](ARCHITECTURE_BACKEND.md)
- [Frontend Architecture →](ARCHITECTURE_FRONTEND.md)
- [AI Services & RAG →](ARCHITECTURE_AI_SERVICES.md)
- [Architecture Overview →](ARCHITECTURE_OVERVIEW.md)