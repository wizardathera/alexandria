# ARCHITECTURE_OVERVIEW.md

## 🛠️ Tech Stack & Architecture

### Backend
- Python (main language)
- LangChain for AI orchestration
- **Vector Database**: Chroma (prototype) → Supabase with pgvector (production)
- FastAPI for REST APIs
- Pydantic for data validation
- SQLAlchemy/SQLModel for ORM (if needed)

### Frontend
- **Phase 1**: Streamlit (rapid prototyping and local testing)
- **Phase 2**: Next.js (production with user accounts and professional UI)

### AI Provider
- **Primary**: OpenAI APIs (Chat Completion, Embeddings)
- **Future**: Support for Anthropic or local models

### Authentication
- **Phase 1**: Single-user (no auth required)
- **Phase 2**: Supabase Auth or NextAuth.js for multi-user support

### Deployment
- Docker & docker-compose
- FastMCP for MCP server implementation

### AI Components
- Agentic RAG (not simple retrieval)
- Hypatia conversational assistant (Phase 2+)
- Progress tracking and milestone management
- Multi-tenant semantic search and recommendations

## Migration Strategy Guidelines

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