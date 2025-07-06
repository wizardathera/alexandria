**This document is synchronized with PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md as of 2025-07-05.**

# 🔧 Alexandria Platform - Technical Specifications

## 📋 Overview and Purpose

This document serves as the authoritative technical specification for the **Alexandria Platform**, an AI-powered reading and learning ecosystem that combines Smart Library, Learning Suite, and Marketplace capabilities. Alexandria transforms traditional reading into an interactive, intelligent experience while enabling content creators to monetize their knowledge.

The platform serves three distinct user segments:
- **Individual Readers**: Free Smart Library with intelligent RAG-powered Q&A
- **Educators & Businesses**: Subscription-based Learning Suite for structured education
- **Content Creators**: Marketplace for monetizing educational content

This specification is maintained in sync with the following documents:
- [`PRODUCT_REQUIREMENTS.md`](../PRODUCT_REQUIREMENTS.md) - Product requirements and user stories
- [`ARCHITECTURE_OVERVIEW.md`](./ARCHITECTURE_OVERVIEW.md) - System architecture and design decisions
- [`SECURITY_PRIVACY_PLAN.md`](../SECURITY_PRIVACY_PLAN.md) - Security and privacy requirements
- [`ROADMAP_OVERVIEW.md`](./ROADMAP_OVERVIEW.md) - Strategic roadmap overview and objectives
- [`PLANNING_OVERVIEW.md`](./PLANNING_OVERVIEW.md) - Strategic planning and current phase focus

**⚠️ IMPORTANT**: Any changes to the referenced documents automatically trigger an update to this specification to maintain technical consistency.

## 🎯 MVP Scope Matrix

### Phase 1.0: Core Smart Library (Current Phase)
| Component | Status | Description | Dependencies |
|-----------|--------|-------------|--------------|
| **Streamlit Frontend** | ✅ Complete | Basic upload, Q&A, dashboard | Python, Streamlit |
| **Book Ingestion Pipeline** | ✅ Complete | PDF, EPUB, DOC, TXT, HTML support | LangChain, Chroma |
| **RAG System** | ✅ Complete | Vector search with citations | OpenAI, Embeddings |
| **Enhanced Chunking** | ✅ Complete | Semantic chunking with metadata | Custom algorithms |
| **Hybrid Retrieval** | ✅ Complete | Vector + BM25 + graph traversal | RRF fusion |
| **Multi-Module Schema** | ✅ Complete | Unified content model | Enhanced DB design |
| **MCP Server** | ✅ Complete | AddNoteTool implementation | FastMCP |
| **Docker Environment** | ✅ Complete | Containerized deployment | Docker Compose |

### Phase 2.0: Production Frontend & Main Library
| Component | Status | Description | Dependencies |
|-----------|--------|-------------|--------------|
| **Next.js Frontend** | 📋 Planned | Professional UI/UX with TypeScript | Next.js 14, Tailwind |
| **User Authentication** | 📋 Planned | Account management and sessions | Supabase Auth |
| **Main Library Catalog** | 📋 Planned | Public domain book browsing | Curated collection |
| **Book Purchasing** | 📋 Planned | E-commerce with DRM controls | Stripe integration |
| **Theme System** | 📋 Planned | Customizable UI aesthetics | CSS variables |
| **Persistent Chat** | 📋 Planned | Cross-session conversation history | PostgreSQL |
| **Supabase Migration** | 📋 Planned | Production vector database | Supabase pgvector |
| **Hypatia Assistant** | 📋 Planned | Branded conversational AI | Custom prompts |

### Phase 3.0: Community & Marketplace
| Component | Status | Description | Dependencies |
|-----------|--------|-------------|--------------|
| **Social Features** | 🔮 Future | Sharing, following, discussions | Community framework |
| **Marketplace** | 🔮 Future | Content monetization platform | Payment processing |
| **Advanced Personalization** | 🔮 Future | AI-driven recommendations | User behavior ML |
| **Mobile Optimization** | 🔮 Future | Responsive design enhancements | PWA capabilities |

### Phase 4.0: Desktop Application
| Component | Status | Description | Dependencies |
|-----------|--------|-------------|--------------|
| **Electron App** | 🔮 Future | Cross-platform desktop version | Electron + Next.js |
| **Offline Capabilities** | 🔮 Future | Local storage and sync | IndexedDB |
| **Native Integrations** | 🔮 Future | OS-specific features | Platform APIs |

## 🗄️ Data Model (DDL Summary)

### Core Content Schema
```sql
-- Unified content model supporting all modules
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_type VARCHAR(50) NOT NULL, -- 'book', 'course', 'lesson', 'marketplace_item'
    module_name VARCHAR(20) NOT NULL, -- 'library', 'lms', 'marketplace'
    title VARCHAR(255) NOT NULL,
    description TEXT,
    creator_id UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    visibility VARCHAR(20) DEFAULT 'public', -- 'public', 'private', 'organization'
    status VARCHAR(20) DEFAULT 'draft', -- 'draft', 'published', 'archived'
    metadata JSONB, -- Flexible metadata storage
    content_data JSONB, -- Type-specific content data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector embeddings with enhanced metadata
CREATE TABLE vector_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    embedding VECTOR(1536), -- OpenAI embedding dimension
    chunk_text TEXT NOT NULL,
    chunk_type VARCHAR(50), -- 'paragraph', 'heading', 'summary', 'question'
    vector_namespace VARCHAR(100) NOT NULL, -- Module-specific namespace
    semantic_tags TEXT[], -- AI-extracted topics
    source_location JSONB, -- Page, chapter, section reference
    language VARCHAR(10) DEFAULT 'en',
    reading_level VARCHAR(20), -- 'beginner', 'intermediate', 'advanced'
    importance_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enhanced user management
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(20) DEFAULT 'reader', -- 'reader', 'educator', 'creator', 'admin'
    subscription_tier VARCHAR(20) DEFAULT 'free', -- 'free', 'pro', 'enterprise'
    preferences JSONB, -- User preferences and settings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content relationships for recommendations
CREATE TABLE content_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    target_content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- 'prerequisite', 'related', 'sequel'
    strength_score FLOAT DEFAULT 0.5,
    created_by VARCHAR(20) DEFAULT 'ai', -- 'ai', 'human', 'community'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chat history for persistent conversations
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    session_title VARCHAR(255),
    messages JSONB NOT NULL, -- Array of message objects
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Progress tracking
CREATE TABLE user_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    progress_type VARCHAR(50) NOT NULL, -- 'reading', 'course', 'assessment'
    completion_percentage FLOAT DEFAULT 0,
    time_spent_minutes INTEGER DEFAULT 0,
    last_position JSONB, -- Current reading position
    achievements JSONB, -- Milestones and badges
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Organizations for multi-tenancy
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(20) DEFAULT 'basic',
    settings JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Marketplace transactions (Phase 3)
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    content_item_id UUID REFERENCES content_items(id),
    transaction_type VARCHAR(20) NOT NULL, -- 'purchase', 'subscription', 'refund'
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    stripe_payment_intent_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Indexes for Performance
```sql
-- Vector search optimization
CREATE INDEX idx_vector_embeddings_content_item ON vector_embeddings(content_item_id);
CREATE INDEX idx_vector_embeddings_namespace ON vector_embeddings(vector_namespace);
CREATE INDEX idx_vector_embeddings_semantic_tags ON vector_embeddings USING GIN(semantic_tags);

-- Content search optimization
CREATE INDEX idx_content_items_module_type ON content_items(module_name, content_type);
CREATE INDEX idx_content_items_creator ON content_items(creator_id);
CREATE INDEX idx_content_items_visibility ON content_items(visibility, status);

-- User and session optimization
CREATE INDEX idx_chat_sessions_user_content ON chat_sessions(user_id, content_item_id);
CREATE INDEX idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX idx_content_relationships_source ON content_relationships(source_content_id);
```

## 🔌 API Endpoints

### Core Library API (`/api/library/`)
```yaml
Book Management:
  POST /api/library/books/upload:
    description: Upload and process book files
    request: multipart/form-data (file, metadata)
    response: { book_id: UUID, status: "processing" }
  
  GET /api/library/books:
    description: List user's book collection
    response: [{ id: UUID, title: string, status: string, metadata: object }]
  
  GET /api/library/books/{book_id}:
    description: Get specific book details
    response: { id: UUID, title: string, content: string, metadata: object }
  
  DELETE /api/library/books/{book_id}:
    description: Remove book from library
    response: { success: boolean }

Query and Search:
  POST /api/library/chat:
    description: RAG-powered Q&A with chat history
    request: { query: string, book_id?: UUID, session_id?: UUID }
    response: { answer: string, sources: array, confidence: number }
  
  GET /api/library/chat/{session_id}:
    description: Retrieve chat session history
    response: { messages: array, book_context: object }
  
  POST /api/library/search:
    description: Hybrid search across user's library
    request: { query: string, filters: object }
    response: { results: array, total: number }
```

### User Management API (`/api/users/`)
```yaml
Authentication:
  POST /api/users/register:
    description: Create new user account
    request: { email: string, password: string, full_name?: string }
    response: { user: object, session: object }
  
  POST /api/users/login:
    description: Authenticate user
    request: { email: string, password: string }
    response: { user: object, session: object }
  
  POST /api/users/logout:
    description: End user session
    response: { success: boolean }

Profile Management:
  GET /api/users/profile:
    description: Get user profile and preferences
    response: { user: object, preferences: object }
  
  PUT /api/users/profile:
    description: Update user profile
    request: { full_name?: string, preferences?: object }
    response: { user: object }
  
  GET /api/users/progress:
    description: Get reading progress and analytics
    response: { progress: array, statistics: object }
```

### Main Library API (`/api/catalog/`) - Phase 2
```yaml
Public Catalog:
  GET /api/catalog/books:
    description: Browse public domain book collection
    query: { genre?: string, author?: string, page?: number }
    response: { books: array, pagination: object }
  
  GET /api/catalog/books/{book_id}:
    description: Get book details and preview
    response: { book: object, preview: string }
  
  POST /api/catalog/books/{book_id}/add:
    description: Add book to personal library
    response: { success: boolean, book_id: UUID }

Purchasing:
  POST /api/catalog/books/{book_id}/purchase:
    description: Purchase premium book
    request: { payment_method: string }
    response: { payment_intent: string, book_access: object }
  
  GET /api/catalog/purchases:
    description: List user's purchased books
    response: { purchases: array }
```

### Learning Suite API (`/api/lms/`) - Phase 2
```yaml
Course Management:
  POST /api/lms/courses:
    description: Create new course
    request: { title: string, description: string, content_items: array }
    response: { course: object }
  
  GET /api/lms/courses:
    description: List available courses
    response: { courses: array }
  
  GET /api/lms/courses/{course_id}:
    description: Get course details and lessons
    response: { course: object, lessons: array }
  
  POST /api/lms/courses/{course_id}/enroll:
    description: Enroll in course
    response: { enrollment: object }

Progress Tracking:
  GET /api/lms/progress/{course_id}:
    description: Get course progress
    response: { progress: object, next_lesson: object }
  
  POST /api/lms/progress/{lesson_id}/complete:
    description: Mark lesson as completed
    response: { progress: object, achievement?: object }
```

### Hypatia Assistant API (`/api/hypatia/`) - Phase 2
```yaml
Conversational AI:
  POST /api/hypatia/chat:
    description: Interact with Hypatia assistant
    request: { message: string, context?: object }
    response: { response: string, suggestions?: array }
  
  GET /api/hypatia/onboarding:
    description: Get onboarding guidance
    response: { steps: array, current_step: number }
  
  POST /api/hypatia/feedback:
    description: Submit interaction feedback
    request: { rating: number, comment?: string }
    response: { success: boolean }
```

### Marketplace API (`/api/marketplace/`) - Phase 3
```yaml
Content Monetization:
  POST /api/marketplace/items:
    description: List content for sale
    request: { content_item_id: UUID, price: number, description: string }
    response: { marketplace_item: object }
  
  GET /api/marketplace/items:
    description: Browse marketplace content
    response: { items: array, categories: array }
  
  POST /api/marketplace/items/{item_id}/purchase:
    description: Purchase marketplace content
    response: { payment_intent: string }
```

### System APIs (`/api/system/`)
```yaml
Health and Monitoring:
  GET /api/system/health:
    description: System health check
    response: { status: "healthy", services: object }
  
  GET /api/system/metrics:
    description: Performance metrics (admin only)
    response: { performance: object, usage: object }
  
  GET /api/system/version:
    description: Application version information
    response: { version: string, build: string }
```

## 🚀 Non-Functional Targets

### Performance Requirements
```yaml
Response Times:
  API Endpoints: <500ms (95th percentile)
  RAG Queries: <3 seconds (average)
  Book Upload Processing: <30 seconds per MB
  Search Results: <2 seconds
  Page Load Times: <2 seconds (initial), <1 second (navigation)

Scalability:
  Concurrent Users: 100+ (Phase 2), 1000+ (Phase 3)
  Database Connections: 50+ concurrent
  Vector Search: 1000+ queries/minute
  File Storage: 100GB+ per user (premium)
  Book Processing: 10+ books/hour per user

Reliability:
  Uptime: 99.9% availability
  Data Durability: 99.99% (no data loss)
  Backup Recovery: <4 hours RTO, <1 hour RPO
  Error Rate: <0.1% for API calls
```

### Security & Privacy
```yaml
Authentication:
  Session Management: JWT with refresh tokens
  Password Security: bcrypt hashing, complexity requirements
  API Security: Rate limiting, CORS, input validation
  Data Encryption: AES-256 at rest, TLS 1.3 in transit

Privacy:
  GDPR Compliance: Data portability, right to deletion
  Data Minimization: Only collect necessary information
  Consent Management: Explicit consent for data processing
  Anonymization: Remove PII from analytics data

Access Control:
  Role-Based Permissions: Reader, Educator, Creator, Admin
  Resource Isolation: Multi-tenant data segregation
  API Authentication: Bearer token validation
  Audit Logging: All access attempts logged
```

### Quality Attributes
```yaml
Usability:
  Accessibility: WCAG 2.1 AA compliance
  Mobile Responsiveness: Support for 375px+ screens
  Internationalization: English (primary), Spanish/French (Phase 3)
  Error Handling: User-friendly error messages

Maintainability:
  Code Coverage: >80% backend, >70% frontend
  Documentation: API docs, architecture diagrams
  Testing: Unit, integration, E2E test suites
  Monitoring: Real-time performance and error tracking

Compatibility:
  Browsers: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
  Mobile: iOS 14+, Android 10+
  File Formats: PDF, EPUB, DOC/DOCX, TXT, HTML
  Integrations: Stripe, Supabase, OpenAI APIs
```

## 📋 Phase-2 (Education Add-On) Delta

### Frontend Evolution: Next.js + Electron
The frontend architecture evolves through distinct phases:

**Phase 1.0**: Streamlit (Current)
- Rapid prototyping and validation
- Single-user local deployment
- Basic file upload and Q&A interface

**Phase 2.0**: Next.js Production Frontend
- Professional TypeScript/React application
- Multi-user authentication and session management
- Responsive design with Tailwind CSS
- Progressive Web App (PWA) capabilities
- Server-side rendering for SEO optimization

**Phase 4.0**: Electron Desktop Application
- Cross-platform native desktop experience
- Offline-first capabilities with local storage
- Native OS integrations (file system, notifications)
- Enhanced performance for large document libraries

### Key Technical Additions for Phase 2

#### User Authentication & Multi-Tenancy
```typescript
// User role hierarchy
enum UserRole {
  READER = 'reader',
  EDUCATOR = 'educator', 
  CREATOR = 'creator',
  ADMIN = 'admin'
}

// Subscription tiers
enum SubscriptionTier {
  FREE = 'free',
  PRO = 'pro',
  ENTERPRISE = 'enterprise'
}
```

#### Enhanced RAG System
- **Cross-module search**: Simultaneous search across Library, LMS, and Marketplace content
- **Permission-aware retrieval**: Filter results based on user access rights
- **Graph RAG integration**: Semantic relationship traversal for complex queries
- **Conversation persistence**: Long-term chat history with searchable transcripts

#### Main Library Features
- **Public domain catalog**: Curated collection of freely available books
- **Premium purchasing**: Stripe-integrated e-commerce for current authors
- **DRM controls**: Digital rights management for purchased content
- **Content recommendations**: AI-powered discovery based on reading history

#### Hypatia Assistant Integration
The branded conversational AI assistant provides:
- **Onboarding guidance**: Step-by-step platform introduction
- **Feature discovery**: Contextual help and feature explanations
- **Book recommendations**: Personalized suggestions based on interests
- **Learning path guidance**: Educational content sequencing

### Database Migration Strategy

#### Vector Store Evolution
**Phase 1.0**: Chroma (Local Development)
```python
# Local vector database for prototyping
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("alexandria_books")
```

**Phase 2.0+**: Supabase pgvector (Production)
```sql
-- Production-ready vector storage
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE vector_embeddings (
    id UUID PRIMARY KEY,
    embedding VECTOR(1536),
    content_item_id UUID,
    -- Enhanced metadata for multi-module support
    vector_namespace VARCHAR(100),
    module_name VARCHAR(20),
    user_permissions JSONB
);
```

#### Migration Approach
1. **Dual-write phase**: Write to both Chroma and Supabase during transition
2. **Validation phase**: Compare query results between systems
3. **Cutover phase**: Switch reads to Supabase, maintain Chroma backup
4. **Cleanup phase**: Remove Chroma dependencies after validation

### Marketplace & DRM Integration

#### Content Monetization
```typescript
interface MarketplaceItem {
  id: string;
  content_item_id: string;
  price_cents: number;
  license_type: 'single_user' | 'multi_user' | 'enterprise';
  drm_enabled: boolean;
  preview_percentage: number; // 0-100
}
```

#### Digital Rights Management
- **Watermarking**: Embed user information in purchased content
- **Access control**: Time-limited and device-limited access
- **Download limits**: Restrict offline access duration
- **Sharing prevention**: Detect and prevent unauthorized distribution

### Performance Optimization for Phase 2

#### Caching Strategy
```yaml
Multi-Level Caching:
  L1_Browser: Service Worker, 5 minutes TTL
  L2_CDN: CloudFlare, 1 hour TTL  
  L3_Application: Redis, 15 minutes TTL
  L4_Database: PostgreSQL query cache, 1 hour TTL
```

#### Search Performance
- **Semantic indexing**: Pre-computed embeddings for common queries
- **Query optimization**: Intelligent query rewriting and expansion
- **Result caching**: Cache frequent search results with invalidation
- **Batch processing**: Background embedding generation for new content

---

## 🔄 Document Synchronization Policy

This technical specification is automatically updated when changes are made to:
- **PRODUCT_REQUIREMENTS.md**: User stories and feature requirements
- **ARCHITECTURE_OVERVIEW.md**: High-level system design and technical decisions
- **ARCHITECTURE_BACKEND.md**: Backend architecture and services
- **ARCHITECTURE_FRONTEND.md**: Frontend architecture and components
- **ARCHITECTURE_DATA_MODEL.md**: Data models and database design
- **ARCHITECTURE_AI_SERVICES.md**: AI/ML services and RAG implementation
- **ROADMAP_OVERVIEW.md**: Timeline and milestone adjustments
- **PLANNING_OVERVIEW.md**: Strategic focus and phase priorities
- **SECURITY_PRIVACY_PLAN.md**: Security and compliance requirements

**Maintenance Schedule**:
- **Weekly**: Review for consistency with active development
- **Phase Completion**: Comprehensive update after each major phase
- **Architecture Changes**: Immediate update for structural modifications
- **API Changes**: Real-time synchronization with endpoint modifications

## 📊 Success Metrics

### Technical Performance KPIs
- **API Response Time**: <500ms (95th percentile)
- **RAG Query Speed**: <3 seconds average
- **System Uptime**: 99.9% availability
- **Code Coverage**: >80% backend, >70% frontend
- **Security Scans**: Zero high/critical vulnerabilities

### User Experience KPIs
- **Page Load Speed**: <2 seconds initial, <1 second navigation
- **Mobile Performance**: Lighthouse score >90
- **Accessibility**: WCAG 2.1 AA compliance
- **Error Rate**: <0.1% for user actions
- **User Satisfaction**: >4.5/5 rating for core features

### Business Impact KPIs
- **User Retention**: >70% monthly active users
- **Feature Adoption**: >60% adoption for new features
- **Conversion Rate**: >15% free-to-paid conversion (Phase 2)
- **Revenue Growth**: >25% MRR growth (Phase 3)
- **Content Quality**: >4.0/5 average content rating

---

*This technical specification is maintained in sync with the Alexandria Platform development roadmap. Last updated: 2025-07-04*

**Version**: 1.0.0  
**Status**: Active Development (Phase 1.4)  
**Next Review**: Phase 1.5 Completion