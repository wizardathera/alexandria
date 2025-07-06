-- DBC Platform - Supabase pgvector Schema Migration
-- Phase 1.35: Migration-Ready Architecture for Chroma → Supabase
-- 
-- This schema supports the unified content model across all DBC modules:
-- - Smart Library (books, articles, documents)
-- - Learning Suite (courses, lessons, assessments)
-- - Marketplace (premium content, digital products)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ========================================
-- Core Tables
-- ========================================

-- Users table for multi-module platform
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    
    -- Role and permissions
    role VARCHAR(20) DEFAULT 'reader' CHECK (role IN ('reader', 'educator', 'creator', 'admin')),
    organization_id UUID,
    permissions TEXT[], -- Array of specific permissions
    
    -- Profile information
    full_name VARCHAR(255),
    avatar_url TEXT,
    bio TEXT,
    
    -- Settings and preferences (JSONB for flexibility)
    preferences JSONB DEFAULT '{}',
    notification_settings JSONB DEFAULT '{}',
    
    -- Account status
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    subscription_tier VARCHAR(20) DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Organizations table for multi-tenant support
CREATE TABLE IF NOT EXISTS organizations (
    organization_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    
    -- Organization metadata
    description TEXT,
    website_url TEXT,
    logo_url TEXT,
    
    -- Settings
    settings JSONB DEFAULT '{}',
    billing_settings JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Status
    is_active BOOLEAN DEFAULT true
);

-- Unified content items table supporting all modules
CREATE TABLE IF NOT EXISTS content_items (
    content_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Module classification
    module_type VARCHAR(20) NOT NULL CHECK (module_type IN ('library', 'lms', 'marketplace')),
    content_type VARCHAR(30) NOT NULL CHECK (content_type IN (
        'book', 'article', 'document',                    -- Library types
        'course', 'lesson', 'assessment', 'quiz', 'assignment',  -- LMS types
        'marketplace_item', 'premium_course', 'digital_product'  -- Marketplace types
    )),
    
    -- Basic metadata
    title VARCHAR(500) NOT NULL,
    description TEXT,
    author VARCHAR(255),
    
    -- File information (for file-based content)
    file_name VARCHAR(255),
    file_path TEXT,
    file_type VARCHAR(10),
    file_size BIGINT,
    
    -- Permission and visibility
    visibility VARCHAR(20) DEFAULT 'private' CHECK (visibility IN ('public', 'private', 'organization', 'premium')),
    created_by UUID REFERENCES users(user_id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(organization_id) ON DELETE SET NULL,
    
    -- Processing metadata
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'archived')),
    text_length INTEGER,
    chunk_count INTEGER,
    
    -- Content relationships
    parent_content_id UUID REFERENCES content_items(content_id) ON DELETE CASCADE,
    prerequisite_content_ids UUID[], -- Array of content IDs
    
    -- Semantic metadata
    topics TEXT[], -- AI-extracted topics
    language VARCHAR(10) DEFAULT 'en',
    reading_level VARCHAR(20),
    
    -- Module-specific metadata (flexible JSONB)
    module_metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Content relationships table for AI-powered connections
CREATE TABLE IF NOT EXISTS content_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_content_id UUID NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    target_content_id UUID NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    
    -- Relationship metadata
    relationship_type VARCHAR(30) NOT NULL CHECK (relationship_type IN (
        'prerequisite', 'supplement', 'sequence', 'alternative', 
        'reference', 'similarity', 'contradiction', 'elaboration'
    )),
    strength DECIMAL(3,2) DEFAULT 0.5 CHECK (strength >= 0.0 AND strength <= 1.0),
    confidence DECIMAL(3,2) DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Discovery metadata
    discovered_by VARCHAR(20) DEFAULT 'ai' CHECK (discovered_by IN ('ai', 'human', 'system')),
    human_verified BOOLEAN DEFAULT false,
    context TEXT,
    bidirectional BOOLEAN DEFAULT false,
    
    -- Verification tracking
    verified_at TIMESTAMP WITH TIME ZONE,
    verified_by UUID REFERENCES users(user_id) ON DELETE SET NULL,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Prevent self-references and duplicates
    CONSTRAINT no_self_reference CHECK (source_content_id != target_content_id),
    CONSTRAINT unique_relationship UNIQUE (source_content_id, target_content_id, relationship_type)
);

-- ========================================
-- Vector Embeddings Table
-- ========================================

-- Enhanced embeddings table with multi-module support
CREATE TABLE IF NOT EXISTS content_embeddings (
    embedding_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    
    -- Module and content awareness
    module_type VARCHAR(20) NOT NULL,
    content_type VARCHAR(30) NOT NULL,
    chunk_type VARCHAR(30) DEFAULT 'paragraph',
    
    -- Permission metadata (denormalized for query performance)
    visibility VARCHAR(20) NOT NULL,
    creator_id UUID,
    organization_id UUID,
    
    -- Semantic metadata
    semantic_tags TEXT[], -- AI-extracted topics and categories
    language VARCHAR(10) DEFAULT 'en',
    reading_level VARCHAR(20),
    
    -- Source location metadata (JSONB for flexibility)
    source_location JSONB DEFAULT '{}', -- page, chapter, section, timestamp, etc.
    
    -- Text and embedding data
    text_content TEXT NOT NULL,
    chunk_length INTEGER NOT NULL,
    embedding vector(1536) NOT NULL, -- OpenAI ada-002 dimension
    
    -- Processing metadata
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
    embedding_dimension INTEGER DEFAULT 1536,
    
    -- Quality scores
    importance_score DECIMAL(3,2) CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    quality_score DECIMAL(3,2) CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique chunks per content
    CONSTRAINT unique_content_chunk UNIQUE (content_id, chunk_index)
);

-- ========================================
-- Supporting Tables
-- ========================================

-- Chat conversations table for persistent history
CREATE TABLE IF NOT EXISTS chat_conversations (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    content_id UUID REFERENCES content_items(content_id) ON DELETE CASCADE,
    
    -- Conversation metadata
    title VARCHAR(255),
    module_type VARCHAR(20) NOT NULL,
    conversation_type VARCHAR(30) DEFAULT 'qa' CHECK (conversation_type IN ('qa', 'discovery', 'analysis', 'comparison')),
    
    -- Conversation data
    messages JSONB NOT NULL DEFAULT '[]', -- Array of message objects
    summary TEXT,
    tags TEXT[],
    
    -- Settings
    is_shared BOOLEAN DEFAULT false,
    share_settings JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_message_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User progress tracking for reading and learning
CREATE TABLE IF NOT EXISTS user_progress (
    progress_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content_id UUID NOT NULL REFERENCES content_items(content_id) ON DELETE CASCADE,
    
    -- Progress data
    progress_percentage DECIMAL(5,2) DEFAULT 0.0 CHECK (progress_percentage >= 0.0 AND progress_percentage <= 100.0),
    last_position JSONB DEFAULT '{}', -- page, chapter, timestamp, etc.
    completion_status VARCHAR(20) DEFAULT 'not_started' CHECK (completion_status IN ('not_started', 'in_progress', 'completed', 'abandoned')),
    
    -- Time tracking
    total_time_spent INTEGER DEFAULT 0, -- seconds
    reading_sessions JSONB DEFAULT '[]', -- Array of session objects
    
    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint
    CONSTRAINT unique_user_content_progress UNIQUE (user_id, content_id)
);

-- ========================================
-- Indexes for Performance
-- ========================================

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_organization ON users(organization_id);
CREATE INDEX IF NOT EXISTS idx_users_subscription ON users(subscription_tier);

-- Organizations table indexes
CREATE INDEX IF NOT EXISTS idx_organizations_slug ON organizations(slug);
CREATE INDEX IF NOT EXISTS idx_organizations_active ON organizations(is_active);

-- Content items indexes
CREATE INDEX IF NOT EXISTS idx_content_module_type ON content_items(module_type);
CREATE INDEX IF NOT EXISTS idx_content_type ON content_items(content_type);
CREATE INDEX IF NOT EXISTS idx_content_visibility ON content_items(visibility);
CREATE INDEX IF NOT EXISTS idx_content_creator ON content_items(created_by);
CREATE INDEX IF NOT EXISTS idx_content_organization ON content_items(organization_id);
CREATE INDEX IF NOT EXISTS idx_content_status ON content_items(processing_status);
CREATE INDEX IF NOT EXISTS idx_content_parent ON content_items(parent_content_id);
CREATE INDEX IF NOT EXISTS idx_content_created_at ON content_items(created_at);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_content_module_visibility ON content_items(module_type, visibility);
CREATE INDEX IF NOT EXISTS idx_content_type_status ON content_items(content_type, processing_status);
CREATE INDEX IF NOT EXISTS idx_content_creator_module ON content_items(created_by, module_type);

-- Content relationships indexes
CREATE INDEX IF NOT EXISTS idx_relationships_source ON content_relationships(source_content_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON content_relationships(target_content_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON content_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_strength ON content_relationships(strength);
CREATE INDEX IF NOT EXISTS idx_relationships_verified ON content_relationships(human_verified);

-- Vector embeddings indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_content ON content_embeddings(content_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_module ON content_embeddings(module_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_visibility ON content_embeddings(visibility);
CREATE INDEX IF NOT EXISTS idx_embeddings_creator ON content_embeddings(creator_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_organization ON content_embeddings(organization_id);

-- Composite indexes for permission-aware search
CREATE INDEX IF NOT EXISTS idx_embeddings_module_visibility ON content_embeddings(module_type, visibility);
CREATE INDEX IF NOT EXISTS idx_embeddings_visibility_creator ON content_embeddings(visibility, creator_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_organization_visibility ON content_embeddings(organization_id, visibility);

-- Vector similarity search index (HNSW for performance)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_hnsw ON content_embeddings 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Additional vector indexes for different distance metrics
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_l2 ON content_embeddings 
USING hnsw (embedding vector_l2_ops) 
WITH (m = 16, ef_construction = 64);

-- Chat conversations indexes
CREATE INDEX IF NOT EXISTS idx_conversations_user ON chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_content ON chat_conversations(content_id);
CREATE INDEX IF NOT EXISTS idx_conversations_module ON chat_conversations(module_type);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON chat_conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversations_shared ON chat_conversations(is_shared);

-- User progress indexes
CREATE INDEX IF NOT EXISTS idx_progress_user ON user_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_progress_content ON user_progress(content_id);
CREATE INDEX IF NOT EXISTS idx_progress_status ON user_progress(completion_status);
CREATE INDEX IF NOT EXISTS idx_progress_last_accessed ON user_progress(last_accessed_at);

-- ========================================
-- Functions and Triggers
-- ========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_items_updated_at BEFORE UPDATE ON content_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_relationships_updated_at BEFORE UPDATE ON content_relationships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chat_conversations_updated_at BEFORE UPDATE ON chat_conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_progress_updated_at BEFORE UPDATE ON user_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for permission-aware embedding search
CREATE OR REPLACE FUNCTION search_embeddings_with_permissions(
    query_embedding vector(1536),
    user_id_param UUID DEFAULT NULL,
    user_role_param VARCHAR(20) DEFAULT 'reader',
    user_org_id_param UUID DEFAULT NULL,
    user_subscription_param VARCHAR(20) DEFAULT 'free',
    module_filter_param VARCHAR(20) DEFAULT NULL,
    content_type_filter_param VARCHAR(30) DEFAULT NULL,
    limit_param INTEGER DEFAULT 5,
    similarity_threshold DECIMAL DEFAULT 0.0
)
RETURNS TABLE (
    embedding_id UUID,
    content_id UUID,
    chunk_index INTEGER,
    text_content TEXT,
    similarity_score DECIMAL,
    module_type VARCHAR(20),
    content_type VARCHAR(30),
    chunk_type VARCHAR(30),
    semantic_tags TEXT[],
    source_location JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.embedding_id,
        e.content_id,
        e.chunk_index,
        e.text_content,
        (1 - (e.embedding <=> query_embedding))::DECIMAL as similarity_score,
        e.module_type,
        e.content_type,
        e.chunk_type,
        e.semantic_tags,
        e.source_location
    FROM content_embeddings e
    WHERE 
        -- Permission filtering
        (user_role_param = 'admin' OR
         e.visibility = 'public' OR
         (e.visibility = 'private' AND e.creator_id = user_id_param) OR
         (e.visibility = 'organization' AND e.organization_id = user_org_id_param AND user_org_id_param IS NOT NULL) OR
         (e.visibility = 'premium' AND user_subscription_param IN ('pro', 'enterprise')))
        
        -- Module filtering
        AND (module_filter_param IS NULL OR e.module_type = module_filter_param)
        
        -- Content type filtering  
        AND (content_type_filter_param IS NULL OR e.content_type = content_type_filter_param)
        
        -- Similarity threshold
        AND (1 - (e.embedding <=> query_embedding)) >= similarity_threshold
    
    ORDER BY e.embedding <=> query_embedding
    LIMIT limit_param;
END;
$$ LANGUAGE plpgsql;

-- Function to get content relationships for recommendation boosting
CREATE OR REPLACE FUNCTION get_content_relationships(content_ids UUID[])
RETURNS TABLE (
    source_content_id UUID,
    target_content_id UUID,
    relationship_type VARCHAR(30),
    strength DECIMAL,
    confidence DECIMAL,
    bidirectional BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        r.source_content_id,
        r.target_content_id,
        r.relationship_type,
        r.strength,
        r.confidence,
        r.bidirectional
    FROM content_relationships r
    WHERE 
        r.source_content_id = ANY(content_ids) OR
        (r.bidirectional = true AND r.target_content_id = ANY(content_ids));
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- Row Level Security (RLS) Setup
-- ========================================

-- Enable RLS on sensitive tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data (except admins)
CREATE POLICY users_self_policy ON users
    FOR ALL USING (
        user_id = current_setting('app.current_user_id')::UUID OR
        current_setting('app.current_user_role', true) = 'admin'
    );

-- Content visibility policies
CREATE POLICY content_visibility_policy ON content_items
    FOR SELECT USING (
        visibility = 'public' OR
        created_by = current_setting('app.current_user_id')::UUID OR
        (visibility = 'organization' AND organization_id = current_setting('app.current_user_org_id', true)::UUID) OR
        current_setting('app.current_user_role', true) = 'admin'
    );

-- Embedding visibility follows content visibility
CREATE POLICY embeddings_visibility_policy ON content_embeddings
    FOR SELECT USING (
        visibility = 'public' OR
        creator_id = current_setting('app.current_user_id')::UUID OR
        (visibility = 'organization' AND organization_id = current_setting('app.current_user_org_id', true)::UUID) OR
        current_setting('app.current_user_role', true) = 'admin'
    );

-- Chat conversations privacy
CREATE POLICY chat_conversations_policy ON chat_conversations
    FOR ALL USING (
        user_id = current_setting('app.current_user_id')::UUID OR
        is_shared = true OR
        current_setting('app.current_user_role', true) = 'admin'
    );

-- User progress privacy
CREATE POLICY user_progress_policy ON user_progress
    FOR ALL USING (
        user_id = current_setting('app.current_user_id')::UUID OR
        current_setting('app.current_user_role', true) = 'admin'
    );

-- ========================================
-- Migration Validation Views
-- ========================================

-- View to validate schema compatibility
CREATE OR REPLACE VIEW migration_validation AS
SELECT 
    'content_items' as table_name,
    COUNT(*) as record_count,
    COUNT(DISTINCT module_type) as module_types,
    COUNT(DISTINCT content_type) as content_types,
    MIN(created_at) as earliest_record,
    MAX(created_at) as latest_record
FROM content_items
UNION ALL
SELECT 
    'content_embeddings' as table_name,
    COUNT(*) as record_count,
    COUNT(DISTINCT module_type) as module_types,
    COUNT(DISTINCT content_type) as content_types,
    MIN(created_at) as earliest_record,
    MAX(created_at) as latest_record
FROM content_embeddings
UNION ALL
SELECT 
    'content_relationships' as table_name,
    COUNT(*) as record_count,
    COUNT(DISTINCT relationship_type) as module_types,
    0 as content_types,
    MIN(created_at) as earliest_record,
    MAX(created_at) as latest_record
FROM content_relationships;

-- Performance monitoring view
CREATE OR REPLACE VIEW performance_metrics AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_tup_read = 0 THEN 0
        ELSE (idx_tup_fetch::float / idx_tup_read::float * 100)::decimal(5,2)
    END as index_hit_rate
FROM pg_stat_user_indexes 
WHERE schemaname = current_schema()
ORDER BY tablename, indexname;

-- ========================================
-- Comments for Documentation
-- ========================================

COMMENT ON TABLE users IS 'Multi-module user accounts with role-based permissions';
COMMENT ON TABLE organizations IS 'Multi-tenant organization support';
COMMENT ON TABLE content_items IS 'Unified content storage for Library, LMS, and Marketplace modules';
COMMENT ON TABLE content_relationships IS 'AI-powered content relationships for recommendations';
COMMENT ON TABLE content_embeddings IS 'Vector embeddings with enhanced metadata for multi-module search';
COMMENT ON TABLE chat_conversations IS 'Persistent chat history with conversation threading';
COMMENT ON TABLE user_progress IS 'Reading and learning progress tracking across all content';

COMMENT ON FUNCTION search_embeddings_with_permissions IS 'Permission-aware vector similarity search with module filtering';
COMMENT ON FUNCTION get_content_relationships IS 'Retrieve content relationships for recommendation boosting';

-- Migration completion marker
INSERT INTO content_items (content_id, module_type, content_type, title, description, visibility)
VALUES (
    'migration-marker-v1.35',
    'library',
    'document',
    'DBC Migration Marker v1.35',
    'Schema migration completed for Phase 1.35 - Migration-Ready Architecture',
    'public'
) ON CONFLICT (content_id) DO NOTHING;