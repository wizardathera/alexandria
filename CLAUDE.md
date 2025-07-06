# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📘 Project: Alexandria Platform

**Strategic Vision**: A comprehensive AI-powered platform combining Smart Library, Learning Suite, and Marketplace capabilities to serve individual readers, educators, and content creators.

**Core Modules**:
1. **Smart Library** - Personal book management with advanced RAG for Q&A
2. **Learning Suite** - Course creation and learning management system  
3. **Marketplace** - Content monetization and community features

**Target Users**: 
- **Phase 1**: Individual readers and learners (free Smart Library)
- **Phase 2**: Educators and businesses (paid Learning Suite subscriptions)
- **Phase 3**: Content creators and authors (Marketplace revenue sharing)

**Business Model**: Freemium Smart Library → Subscription LMS → Transaction-based Marketplace

## 🎯 Core Development Principles

### Strategic Development Approach
- **Modular Architecture**: Design each module (Library, LMS, Marketplace) with clear boundaries
- **Phase-Appropriate Technology**: Choose technology that supports current phase while enabling future growth  
- **User-First Development**: Always prioritize user experience over technical complexity
- **Revenue-Aware Design**: Build features that support the business model progression
- **Migration-Friendly Code**: Write code that facilitates gradual technology migrations

### Module Development Priority
1. **Phase 1 Focus**: Smart Library with robust RAG capabilities
2. **Phase 2 Focus**: Learning Suite with course creation and management
3. **Phase 3 Focus**: Marketplace with monetization and community features

### Communication Style
- **Be very thorough** in all explanations
- **Use clear, plain language** - avoid technical jargon
- **Confirm steps before assuming or deleting anything**
- **Always ask clarifying questions** if uncertain about requirements
- Provide step-by-step instructions for running or testing code
- Update documentation after every significant change

### Code Quality Standards
- **Never exceed 500 lines per file** - split modules when necessary
- **Never hallucinate** libraries, functions, or file paths
- Always read `docs/PLANNING.md` and the relevant `docs/TASK_*.md` files before coding
- **PEP8 compliance** with type hints for all functions
- **Google-style docstrings** required for every function
- **Include detailed comments** explaining non-obvious code
- **Modular design** - keep API calls and business logic separate
- **Never hardcode credentials** - use environment variables only
- Confirm task completion only after:
  - All relevant tests pass
  - All files are properly documented
  - The feature works as described

### Testing Requirements (Mandatory)
Every feature must include exactly these test types:
1. **Expected behavior test** - normal use case
2. **Edge case test** - boundary conditions  
3. **Failure scenario test** - error handling

**Mock Strategy**:
- **Always mock external services** (OpenAI, Supabase, Chroma)
- Store mock data in `/tests/fixtures/`
- Never make real API calls in tests

## 🛠️ Tech Stack & Architecture

**Backend:**
- Python (main language)
- LangChain for AI orchestration
- **Vector Database**: Chroma (prototype) → Supabase with pgvector (production)
- FastAPI for REST APIs
- Pydantic for data validation
- SQLAlchemy/SQLModel for ORM (if needed)

**Frontend:**
- **Phase 1**: Streamlit (rapid prototyping and local testing)
- **Phase 2**: Next.js (production with user accounts and professional UI)

**AI Provider:**
- **Primary**: OpenAI APIs (Chat Completion, Embeddings)
- **Future**: Support for Anthropic or local models

**Authentication:**
- **Phase 1**: Single-user (no auth required)
- **Phase 2**: Supabase Auth or NextAuth.js for multi-user support

**Deployment:**
- Docker & docker-compose
- FastMCP for MCP server implementation

**AI Components:**
- Agentic RAG (not simple retrieval)
- Role-play simulation capabilities
- Progress tracking and milestone management

## 📂 Required File Structure

```
alexandria-app/
├── docs/PLANNING_OVERVIEW.md     # Strategic planning and current development phases
├── docs/PLANNING_PHASES.md       # Detailed phase descriptions and deliverables
├── docs/PLANNING_TASKS_BREAKDOWN.md # Task breakdowns and priorities
├── docs/PLANNING_DEPENDENCIES.md # Dependencies and sequencing
├── docs/PLANNING_NOTES_HISTORY.md # Historical context and archived sections
├── docs/ROADMAP_OVERVIEW.md      # Strategic roadmap overview and objectives
├── docs/ROADMAP_PHASES.md        # Detailed phase descriptions and deliverables  
├── docs/ROADMAP_FEATURES.md      # Feature-level plans and priorities
├── docs/ROADMAP_TIMELINES.md     # Time-based planning and schedules
├── docs/ROADMAP_NOTES_HISTORY.md # Historical context and archived sections  
├── docs/ARCHITECTURE_OVERVIEW.md  # Technical architecture and design decisions
├── docs/ARCHITECTURE_BACKEND.md  # Backend architecture and services
├── docs/ARCHITECTURE_FRONTEND.md # Frontend architecture and components
├── docs/ARCHITECTURE_DATA_MODEL.md # Data models and database design
├── docs/ARCHITECTURE_AI_SERVICES.md # AI/ML services and RAG implementation
├── docs/TASK.md                 # Main task tracking file
├── docs/TASK_BACKEND.md         # Backend development tasks
├── docs/TASK_FRONTEND.md        # Frontend development tasks
├── docs/TASK_INFRASTRUCTURE.md  # Infrastructure and deployment tasks
├── docs/TASK_SECURITY_COMPLIANCE.md # Security and compliance tasks
├── docs/TASK_PRODUCT_FEATURES.md # Product feature development tasks
├── docs/TASK_MISC.md            # Miscellaneous tasks
├── docs/FUTURE_FEATURES.md      # Future feature backlog
├── docs/PRODUCT_REQUIREMENTS.md # Product requirements and specifications
├── docs/TECHNICAL_SPECIFICATIONS.md # Technical specifications
├── docs/SECURITY_PRIVACY_PLAN.md # Security and privacy planning
├── docs/DEPLOYMENT_GUIDE.md     # Deployment and operations guide
├── README.md                # Installation and usage instructions
├── CLAUDE.md                # This file - development guidance
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-service orchestration
├── /src/
│   ├── main.py             # Application entry point with modular routing
│   ├── /library/           # Smart Library module (Phase 1)
│   ├── /lms/               # Learning Suite module (Phase 2) 
│   ├── /marketplace/       # Marketplace module (Phase 3)
│   ├── /shared/            # Shared services (auth, payments, etc.)
│   ├── /rag/               # Agentic RAG components
│   ├── /mcp/               # MCP server implementation
│   ├── /prompts/           # Prompt templates
│   ├── /tools/             # Modular tool implementations
│   ├── /utils/             # Helper functions
│   └── /api/               # FastAPI endpoints with module namespaces
└── /tests/
    ├── /library/           # Smart Library tests
    ├── /lms/               # Learning Suite tests  
    ├── /marketplace/       # Marketplace tests
    ├── /shared/            # Shared services tests
    └── /fixtures/          # Test data and mocks
```

## 📚 Documentation Structure

**Strategic Planning Documents**:
- **docs/PLANNING_OVERVIEW.md** - Overall project strategy and current phase focus
- **docs/PLANNING_PHASES.md** - Detailed phase descriptions and deliverables
- **docs/PLANNING_TASKS_BREAKDOWN.md** - Task breakdowns and priorities
- **docs/PLANNING_DEPENDENCIES.md** - Dependencies and sequencing
- **docs/PLANNING_NOTES_HISTORY.md** - Historical context and archived sections
- **docs/ROADMAP_OVERVIEW.md** - Strategic roadmap overview and objectives
- **docs/ROADMAP_PHASES.md** - Detailed phase descriptions and deliverables
- **docs/ROADMAP_FEATURES.md** - Feature-level plans and priorities
- **docs/ROADMAP_TIMELINES.md** - Time-based planning and schedules
- **docs/ROADMAP_NOTES_HISTORY.md** - Historical context and archived sections
- **docs/ARCHITECTURE_OVERVIEW.md** - High-level system architecture and design decisions
- **docs/ARCHITECTURE_BACKEND.md** - Backend architecture and services
- **docs/ARCHITECTURE_FRONTEND.md** - Frontend architecture and components
- **docs/ARCHITECTURE_DATA_MODEL.md** - Data models and database design
- **docs/ARCHITECTURE_AI_SERVICES.md** - AI/ML services and RAG implementation

**Development Documents**:
- **CLAUDE.md** - This file with development guidance and standards  
- **README.md** - User-facing installation and usage instructions
- **docs/TASK.md** - Main task tracking file
- **docs/TASK_BACKEND.md** - Backend development tasks
- **docs/TASK_FRONTEND.md** - Frontend development tasks
- **docs/TASK_INFRASTRUCTURE.md** - Infrastructure and deployment tasks
- **docs/TASK_SECURITY_COMPLIANCE.md** - Security and compliance tasks
- **docs/TASK_PRODUCT_FEATURES.md** - Product feature development tasks
- **docs/TASK_MISC.md** - Miscellaneous tasks

## 📋 Development Commands

### Setup & Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Code Quality
```bash
# Format code
black src/ tests/

# Type checking (if mypy is used)
mypy src/

# Linting (if flake8 is used)
flake8 src/ tests/
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_main.py

# Run with coverage
pytest --cov=src tests/
```

### Development Server
```bash
# Start FastAPI server
uvicorn src.main:app --reload

# Start Streamlit frontend (if using Streamlit)
streamlit run src/frontend/app.py

# Start MCP server
python src/mcp/server.py
```

### Docker Operations
```bash
# Build and run with docker-compose
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🔧 Code Standards & Conventions

### Python Style
- **PEP8 compliance** - use `black` for formatting
- **Type hints required** for all function parameters and returns
- **Google-style docstrings** for every function:

```python
def example_function(param1: str, param2: int) -> dict:
    """
    Brief description of what the function does.
    
    Args:
        param1 (str): Description of first parameter.
        param2 (int): Description of second parameter.
    
    Returns:
        dict: Description of return value.
    
    Raises:
        ValueError: When param2 is negative.
    """
    pass
```

### Data Validation
- Use **Pydantic models** for all data structures
- Validate input data at API boundaries
- Include proper error handling and user-friendly error messages

### Testing Requirements
Every feature must include:
1. **Expected behavior test** - normal use case
2. **Edge case test** - boundary conditions
3. **Failure scenario test** - error handling

```python
# Example test structure
def test_feature_success():
    """Test normal operation."""
    pass

def test_feature_edge_case():
    """Test boundary conditions."""
    pass

def test_feature_failure():
    """Test error handling."""
    pass
```

### Mocking Strategy
- Mock external services (OpenAI API, Supabase, Chroma)
- Use `pytest-mock` or `unittest.mock`
- Store mock data in `/tests/fixtures/`

## 📚 Book Content & File Format Support

### Supported File Formats
The system must accept uploads in these formats with specific loaders:

1. **PDF** - Use `UnstructuredFileLoader`
2. **EPUB** - Use `EPubLoader` 
3. **DOC/DOCX** - Use `UnstructuredFileLoader`
4. **TXT** - Use `TextLoader`
5. **HTML** - Use `UnstructuredFileLoader` or `BSHTMLLoader`

### File Processing Guidelines
- **Modular Design**: Each file type has its own loader module
- **Future Expansion**: Design to easily add Markdown, MOBI, etc.
- **Error Handling**: Graceful failure for unsupported formats
- **Chunking Strategy**: Consistent text chunking across all formats
- **Metadata Preservation**: Extract and store file metadata (title, author, etc.)

## 🔄 Migration Strategy Guidelines

### Frontend Migration (Streamlit → Next.js)
**Current Phase**: Streamlit for rapid prototyping
**Future Phase**: Next.js for production

**Guidelines**:
- Keep all frontend logic in **modular functions** for easy reuse
- Document which components need Next.js equivalents:
  - File upload handlers
  - Display components  
  - User interaction patterns
- Separate business logic from UI logic
- Use consistent data structures between phases

### Database Migration (Chroma → Supabase)
**Current Phase**: Chroma for local development
**Future Phase**: Supabase with pgvector for production

**Guidelines**:
- **Abstract vector operations** behind interface classes
- Design consistent schema that works for both systems
- Document all migration steps and schema considerations
- Test migration path with sample data
- Plan for zero-downtime migration strategy

### Authentication Evolution (Single-user → Multi-user)
**Current Phase**: No authentication (single-user)
**Future Phase**: Full user management

**Guidelines**:
- **Always include `user_id`** fields in data models (set to default for Phase 1)
- Design APIs to accept user context (even if ignored initially)
- Plan data isolation and security boundaries early
- Use Supabase Auth or NextAuth.js for production authentication

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

## 🔌 MCP Server Requirements

### Required Tools
Implement these tools using FastMCP:

1. **AddNoteTool**
   - Save user notes and reflections
   - Associate with specific book sections
   - Support categorization and tagging

2. **FetchResourceTool**
   - Retrieve related external resources
   - Support multiple resource types (articles, videos, discussions)
   - Cache frequently accessed resources

3. **UpdateProgressTool**
   - Track reading progress and milestones
   - Update learning achievements
   - Generate progress reports

### Configuration
- Load environment variables securely
- Provide clear examples for local and Docker deployment
- Include proper error handling and logging

## 🔌 API Integration & Provider Management

### OpenAI Integration (Primary)
**Current Provider**: OpenAI for all LLM and embedding tasks

**Required APIs**:
- **Chat Completion**: GPT-3.5/GPT-4 for question answering and conversations
- **Embeddings**: text-embedding-ada-002 for vector generation
- **Future**: GPT-4 Vision for image/document analysis

**Implementation Guidelines**:
- **Modular API calls**: Each API operation in separate function
- **Provider abstraction**: Design interface to support future providers
- **Error handling**: Proper retry logic and rate limiting
- **Cost tracking**: Log token usage for monitoring
- **Security**: Store API keys in environment variables only

### Future Provider Support
**Planned Additions**:
- **Anthropic**: Claude models as alternative to GPT
- **Local Models**: Ollama or similar for offline operation
- **Other Providers**: Cohere, Hugging Face, etc.

**Design Requirements**:
- Provider interface/abstract base class
- Consistent response format across providers
- Configuration-based provider selection
- Graceful fallback between providers

## 🚀 Deployment & Environment

### Environment Variables (.env)
```bash
# Required API keys
OPENAI_API_KEY=your_openai_key_here

# Phase 2: Supabase configuration (for production migration)
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Application configuration
DEBUG=true
LOG_LEVEL=info
VECTOR_DB_TYPE=chroma  # Phase 1: chroma, Phase 2+: supabase
AUTH_ENABLED=false     # Phase 1: false, Phase 2+: true

# Module configuration (enable/disable modules by phase)
LIBRARY_MODULE_ENABLED=true      # Phase 1+: Always enabled
LMS_MODULE_ENABLED=false         # Phase 2+: Learning Suite
MARKETPLACE_MODULE_ENABLED=false # Phase 3+: Marketplace

# File upload configuration
MAX_UPLOAD_SIZE_MB=50
SUPPORTED_FORMATS=pdf,epub,doc,docx,txt,html

# Payment configuration (Phase 3+)
STRIPE_PUBLIC_KEY=your_stripe_public_key_here
STRIPE_SECRET_KEY=your_stripe_secret_key_here
STRIPE_WEBHOOK_SECRET=your_stripe_webhook_secret_here

# MCP Server configuration
MCP_SERVER_PORT=8080
MCP_SERVER_HOST=localhost
```

### Docker Configuration
- Ensure all dependencies are in `requirements.txt`
- Use multi-stage builds for optimization
- Include health checks for services
- Provide clear instructions for both development and production

## 📝 Documentation Requirements

### Always Update After Changes
- `README.md` - installation and usage instructions
- `docs/PLANNING_OVERVIEW.md` - strategic planning and current phase focus
- `docs/ROADMAP_OVERVIEW.md` - strategic roadmap overview
- `docs/ARCHITECTURE_OVERVIEW.md` - high-level technical architecture decisions
- `docs/ARCHITECTURE_BACKEND.md` - backend architecture and services
- `docs/ARCHITECTURE_FRONTEND.md` - frontend architecture and components
- `docs/ARCHITECTURE_DATA_MODEL.md` - data models and database design
- `docs/ARCHITECTURE_AI_SERVICES.md` - AI/ML services and RAG implementation
- `docs/TASK_*.md` - current tasks organized by category
- Function docstrings - keep them current with code changes

**NOTE**: See the **Mandatory Documentation Synchronization Policy** in the Critical Development Rules section for complete enforcement requirements.

### User-Friendly Instructions
- Provide step-by-step setup instructions
- Include troubleshooting common issues
- Explain what each component does in plain language
- Give examples of how to use the application

## ⚠️ Critical Development Rules

### Before Any Coding Session
1. **Read** `docs/PLANNING_OVERVIEW.md`, `docs/ROADMAP_OVERVIEW.md`, and relevant `docs/TASK_*.md` files first
2. **Understand current phase** and module priorities (Library → LMS → Marketplace)
3. **Review RAG database strategy** - understand multi-module architecture implications
4. **Ask clarifying questions** if requirements are unclear
5. **Confirm approach** before making significant changes
6. **Design with future phases in mind** - modular, migration-friendly code

### File Management
- **Never exceed 500 lines** in any single file
- **Split modules** when files become too large
- **Maintain consistent** file structure as specified above

### Quality Assurance
- **Test everything** - no untested code in main branch
- **Document everything** - especially for non-technical users
- **Validate assumptions** - don't guess at requirements

### Completion Criteria
Mark tasks complete only when:
- ✅ All tests pass
- ✅ Code is properly documented
- ✅ Feature works as specified
- ✅ Documentation is updated
- ✅ RAG performance metrics are validated
- ✅ Migration path is tested (if applicable)

### 📋 Task & Phase Numbering Convention

**CRITICAL RULE**: When outlining or generating tasks in ANY project documentation (docs/PLANNING.md, docs/TASK_*.md, docs/ROADMAP_PHASES.md, docs/ARCHITECTURE_*.md, etc.), you MUST always use the **structured task & phase numbering convention** defined below to ensure clear sequential order that can be easily followed and referenced.

#### **🎯 Task & Phase Numbering Convention**

**1️⃣ Phase Levels**

Each major phase is labeled with an integer followed by .0.

**Example:**
- `1.0` — Phase 1: Foundational Setup
- `2.0` — Phase 2: Production Frontend

**2️⃣ Subphases**

Subphases within a phase are labeled by appending .1, .2, .3, etc. to the phase number.

**Example:**
- `1.1` — Subphase 1: Environment & Configs
- `1.2` — Subphase 2: Backend APIs

**3️⃣ Individual Tasks**

Individual tasks within a subphase are labeled by appending a second decimal point with sequential numbering.

**Example:**
- `1.11` — Initialize repository
- `1.12` — Configure .env files
- `1.21` — Build authentication routes

**4️⃣ Constraints**

- Subphases should not contain an excessive number of major tasks. Prefer creating additional subphases if more than ~10 tasks are needed.
- Avoid lettered or inconsistent numbering formats (1A, 1.1.a, etc.).
- Use this convention consistently in:
  - PLANNING.md
  - TASK_*.md (all task files)
  - ROADMAP_PHASES.md
  - FUTURE_FEATURES.md
  - docs/ARCHITECTURE_OVERVIEW.md
  - docs/ARCHITECTURE_BACKEND.md
  - docs/ARCHITECTURE_FRONTEND.md
  - docs/ARCHITECTURE_DATA_MODEL.md
  - docs/ARCHITECTURE_AI_SERVICES.md

#### **Application Requirements**:
- **New Task Creation**: All newly created tasks must follow this phase/subphase/task numbering structure
- **Task Reordering**: When reordering or updating existing tasks, renumber to maintain clean sequence
- **Cross-References**: When referencing tasks, use the numeric identifier (e.g., "Depends on Task 1.21")
- **Phase Integration**: Task numbers should follow the phase.subphase.task pattern consistently

#### **Documentation Consistency**:
This structured numbering rule applies to:
- Task lists in docs/PLANNING_OVERVIEW.md
- Task definitions in docs/TASK_*.md files  
- Milestone breakdowns in docs/ROADMAP_PHASES.md
- Implementation steps in docs/ARCHITECTURE_OVERVIEW.md and related architecture documents
- Any other project documentation containing task sequences

**ENFORCEMENT**: 
- **Verify** that all current and future tasks are reorganized to match this structure
- **Update** existing entries to align with this convention
- **Enforce** this numbering scheme in all further edits and expansions
- **Reject** additions or edits that deviate from this numbering format unless explicitly instructed otherwise

All future task outputs and documentation will follow this phase/subphase/task numbering convention without exception. Any existing documentation with non-conforming task structures should be updated to comply when modified.

### 📘 Documentation Synchronization Policy (Updated 2025-07-05)

#### Synchronization Baseline:
As of 2025-07-05, all active development, reference, and supporting documents have been fully synchronized. This includes:

**✅ PLANNING Files**
- PLANNING_OVERVIEW.md
- PLANNING_PHASES.md
- PLANNING_DEPENDENCIES.md
- PLANNING_TASKS_BREAKDOWN.md

**✅ TASK Files**
- TASK_BACKEND.md
- TASK_FRONTEND.md
- TASK_INFRASTRUCTURE.md
- TASK_PRODUCT_FEATURES.md
- TASK_SECURITY_COMPLIANCE.md

**✅ ROADMAP Files**
- ROADMAP_OVERVIEW.md
- ROADMAP_PHASES.md
- ROADMAP_FEATURES.md
- ROADMAP_TIMELINES.md

**✅ ARCHITECTURE Files**
- ARCHITECTURE_OVERVIEW.md
- ARCHITECTURE_BACKEND.md
- ARCHITECTURE_FRONTEND.md
- ARCHITECTURE_DATA_MODEL.md
- ARCHITECTURE_AI_SERVICES.md

**✅ Other Reference Files**
- SECURITY_PRIVACY_PLAN.md
- TECHNICAL_SPECIFICATIONS.md

**Note:**
PLANNING.md and ROADMAP.md are deprecated and should never be referenced again. All references must use PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md.

#### 🎯 Automatic Synchronization Instructions
Any time a development phase, milestone, or major update is completed:

✅ Immediately update and synchronize all of the above files.

✅ Recalculate and record all task statuses, completion percentages, and phase progress.

✅ Ensure all cross-references are correct and point only to canonical files.

✅ Update the synchronization metadata header in each file to reflect the new date.

**Synchronization Header Template:**
```
**This document is synchronized with PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md as of [latest date].**
```

✅ Confirm no residual references remain to deprecated files (PLANNING.md, ROADMAP.md) after any edits.

✅ If any ambiguity exists, flag the document for manual review before considering the update complete.

#### ⚠️ Enforcement Requirement
No phase or task shall be marked complete until:

- All synchronized documents are updated.
- All broken or outdated references are removed.
- All cross-references between planning, roadmap, task, and architecture documents are validated.

#### ✅ Documentation Health Acknowledgment
This policy certifies that, as of 2025-07-05:

- All documents listed here are fully synchronized.
- No further legacy references exist.
- All numbering, progress, and phase details are accurate.

#### Documentation Synchronization Checklist:
Before marking any development phase as complete, verify:
- [ ] docs/PLANNING_OVERVIEW.md reflects current phase status and completed deliverables
- [ ] docs/TASK_*.md files show accurate task completion status and updated progress metrics in each category
- [ ] docs/ARCHITECTURE_OVERVIEW.md and related architecture documents record any architectural changes or decisions made
- [ ] docs/ROADMAP_PHASES.md timeline and milestones are current with actual progress
- [ ] Any future features or requirements discovered are documented appropriately
- [ ] docs/FUTURE_FEATURES.md cleanup completed per Feature Lifecycle Management policy

#### 🔄 Feature Lifecycle Management Policy

**CRITICAL RULE**: When ANY feature listed in FUTURE_FEATURES.md moves into active development or is completed, it MUST be removed from FUTURE_FEATURES.md and properly tracked in the active documentation system.

**Feature Transition Requirements**:
1. **Upon Moving to Active Development**:
   - **REMOVE** the feature from docs/FUTURE_FEATURES.md completely
   - **ADD** the feature to docs/ROADMAP_PHASES.md with appropriate phase assignment
   - **ADD** the feature to docs/PLANNING.md under the relevant phase deliverables
   - **CREATE** specific tasks in appropriate docs/TASK_*.md files for implementation tracking

2. **Upon Feature Completion**:
   - **VERIFY** removal from docs/FUTURE_FEATURES.md (if not already done)
   - **UPDATE** docs/ROADMAP_PHASES.md to mark the feature as completed
   - **UPDATE** docs/PLANNING_OVERVIEW.md to check off the completed deliverable
   - **UPDATE** appropriate docs/TASK_*.md files to mark all related tasks as completed
   - **UPDATE** docs/ARCHITECTURE_OVERVIEW.md and related architecture documents with any architectural changes or decisions

3. **Cross-Reference Validation**:
   - **NO DUPLICATES**: Ensure no feature exists in both FUTURE_FEATURES.md and active development docs
   - **NO ORPHANS**: Every active development item must have corresponding entries in ROADMAP_PHASES.md, PLANNING_OVERVIEW.md, and appropriate TASK_*.md files
   - **NO STALE ENTRIES**: Remove outdated or superseded future feature ideas

**docs/FUTURE_FEATURES.md Integrity Rule**: This file must ONLY contain pending, unimplemented ideas that are not yet in active development. Any feature that has been scheduled, started, or completed must be removed.

**VIOLATION CONSEQUENCE**: Any development phase where documentation is not synchronized is considered INCOMPLETE and must be returned to in-progress status until all documentation is properly updated.

## 🧠 RAG Database Development Guidelines

### Critical Architectural Principles

#### **1. Multi-Module Foundation First**
- **Always design for all three modules**: Library, LMS, Marketplace
- **Unified content schema**: Use `content_items` as universal content container
- **Module-aware embeddings**: Include content_type, module, and permissions in all vector operations
- **Cross-module relationships**: Design content relationships from the start

#### **2. Permission-Aware RAG Operations**
- **Every query must respect user permissions**: Filter by visibility and access rights
- **Role-based content access**: Reader, Educator, Creator, Admin roles with different access levels
- **Organization boundaries**: Multi-tenant isolation for Phase 2+ preparation
- **Permission caching**: Optimize performance while maintaining security

#### **3. Migration-Ready Architecture**
- **Abstract vector operations**: Use VectorDatabaseInterface for easy Chroma → Supabase migration
- **Dual-write capability**: Design for gradual migration with data validation
- **Schema versioning**: Support backward compatibility during transitions
- **Performance benchmarking**: Maintain query response times during migrations

### RAG Implementation Standards

#### **Vector Embedding Requirements**
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

#### **Search Query Standards**
- **Multi-dimensional filtering**: Always include user permissions, content type, and module filters
- **Hybrid search approach**: Combine vector similarity with keyword matching for better relevance
- **Context-aware retrieval**: Use conversation history and user preferences for personalization
- **Performance requirements**: Maintain <3 second response times with enhanced metadata

#### **Content Relationship Mapping**
- **AI-powered discovery**: Use embeddings to find related content automatically
- **Human curation override**: Allow content editors to verify and adjust AI relationships
- **Cross-module relationships**: Enable books → courses, courses → marketplace connections
- **Relationship strength scoring**: Use confidence scores for recommendation ranking

### Testing Requirements for RAG Components

#### **Required Test Categories**
1. **Vector Search Tests**:
   - Similarity search accuracy
   - Permission filtering correctness
   - Performance under load
   - Cross-module search capabilities

2. **Content Relationship Tests**:
   - AI relationship discovery accuracy
   - Cross-module relationship validation
   - Human curation override functionality
   - Recommendation quality metrics

3. **Migration Tests**:
   - Data integrity during Chroma → Supabase migration
   - Performance equivalence testing
   - Rollback scenario validation
   - Schema backward compatibility

#### **Performance Benchmarks**
- **Query Response Time**: <3 seconds for 95% of queries
- **Search Relevance**: >85% user satisfaction with search results
- **Recommendation Quality**: >80% relevance for AI-suggested relationships
- **Concurrent Users**: Support 100+ simultaneous queries in Phase 2

### Database Schema Evolution Rules

#### **Schema Change Management**
- **Backward Compatibility**: New schema must support existing book data without migration
- **Incremental Adoption**: New features can be gradually enabled without breaking existing functionality
- **Migration Scripts**: All schema changes must include validated migration and rollback scripts
- **Data Validation**: Comprehensive testing with representative data before production deployment

#### **Content Type Extensibility**
```python
# Content type hierarchy for future expansion
CONTENT_TYPES = {
    "library": ["book", "article", "document"],
    "lms": ["course", "lesson", "assessment", "quiz", "assignment"],
    "marketplace": ["marketplace_item", "premium_course", "digital_product"]
}
```

### AI and LLM Integration Guidelines

#### **Provider Abstraction**
- **Multi-provider support**: Design for OpenAI + Anthropic + local models
- **Consistent interfaces**: Abstract LLM calls behind service interfaces
- **Cost monitoring**: Track token usage and API costs across all providers
- **Quality assurance**: A/B testing framework for comparing provider performance

#### **Response Generation Standards**
- **Source citation**: Always include content source and confidence scores
- **Context management**: Maintain conversation history and context switching
- **Module-aware responses**: Tailor responses based on content module and user role
- **Quality metrics**: Monitor response accuracy, relevance, and user satisfaction

### Performance Optimization Guidelines

#### **Caching Strategy**
```python
# Multi-level caching hierarchy
class RAGCachingStrategy:
    # L1: Query result caching (Redis) - 5 minutes TTL
    # L2: Embedding caching (Local/Redis) - 1 hour TTL
    # L3: Content metadata caching (Redis) - 24 hours TTL
    # L4: User permission caching (Redis) - 15 minutes TTL
    # L5: Relationship caching (Redis) - 24 hours TTL
```

#### **Database Optimization**
- **Index Strategy**: Multi-dimensional indexes for complex queries
- **Query Optimization**: Efficient joins and filtering for cross-module searches
- **Connection Pooling**: Managed database connections for concurrent users
- **Performance Monitoring**: Real-time query performance tracking and alerting

### Development Workflow for RAG Features

#### **Feature Development Process**
1. **Design Review**: Confirm multi-module implications and migration impact
2. **Interface Design**: Create abstract interfaces before implementation
3. **Test-Driven Development**: Write tests for expected behavior, edge cases, and failures
4. **Performance Testing**: Validate response times and concurrent user support
5. **Migration Testing**: Verify compatibility with both Chroma and Supabase
6. **Documentation Update**: Update architecture and API documentation

#### **Code Review Checklist**
- ✅ Supports all three modules (Library, LMS, Marketplace)
- ✅ Includes proper permission filtering
- ✅ Maintains performance requirements
- ✅ Follows migration-ready patterns
- ✅ Includes comprehensive test coverage
- ✅ Documents multi-module implications
- ✅ Validates with representative data

## 🎯 Success Metrics

The Alexandria app is successful when:
- Non-technical users can easily install and use it
- The RAG system provides accurate, helpful responses
- Role-play scenarios are engaging and educational
- Progress tracking motivates continued learning
- The codebase is maintainable and well-tested

## 📦 Dependency Maintenance Policy

**Effective Date: 2025-07-05**

### Policy Requirements

#### **1. Dependency Synchronization**
- **After any development work** or environment changes, `pip freeze` must be run to capture all installed packages
- **The requirements.txt file** must be regenerated and committed to version control
- **If any new libraries** were introduced but not installed via pip, they must be manually added to requirements.txt

#### **2. Environment Validation**
Before deployment or sharing the project, the environment must be validated by:
```bash
# Create clean virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from requirements.txt
pip install -r requirements.txt

# Verify application startup
python src/main.py
streamlit run src/frontend/app.py
```

#### **3. Documentation Requirements**
- **Any changes to dependencies** must be documented in the CHANGELOG.md under the "Dependencies" section
- **The project must not rely** on undeclared packages to avoid ModuleNotFoundError at runtime
- **Version pinning strategy**: Pin exact versions for critical dependencies, use ranges for utility packages

#### **4. Dependency Categories**
Structure requirements.txt with clear categories:
```bash
# Core Application Dependencies
# AI & ML Libraries  
# Web Framework & API
# File Processing & Document Loaders
# Frontend & UI Components
# Database & Storage
# Development & Testing
# Code Quality & Formatting
# Security & Environment
# Utilities & Helpers
```

#### **5. Regular Maintenance**
- **Monthly dependency review**: Check for security updates and version compatibility
- **Quarterly major updates**: Evaluate major version upgrades and deprecations
- **Security scanning**: Run `pip audit` or equivalent tools to identify vulnerabilities
- **Clean unused dependencies**: Remove packages that are no longer needed

#### **6. Development Workflow Integration**
- **Before committing**: Always run `pip freeze > requirements.txt` if packages were added
- **After merging**: Validate that all team members can install dependencies successfully
- **CI/CD pipeline**: Include dependency validation in automated testing

### Goals
- **Prevent runtime errors** caused by missing libraries
- **Keep the environment reproducible** and consistent across development, testing, and production
- **Ensure smooth onboarding** for collaborators and deployment processes
- **Maintain security** by tracking and updating vulnerable dependencies

### Enforcement
- **All pull requests** must include updated requirements.txt if dependencies changed
- **Deployment scripts** must validate environment before proceeding
- **No exceptions**: Missing dependencies discovered in production constitute a critical bug

## 🆘 Getting Help

If you need clarification on any aspect of this project:
1. Ask specific questions about requirements
2. Propose alternative approaches when appropriate
3. Explain trade-offs clearly
4. Always prioritize user experience and code maintainability