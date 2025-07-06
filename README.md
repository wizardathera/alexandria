# Alexandria 📚

An AI-powered comprehensive platform combining Smart Library, Learning Suite, and Marketplace capabilities to serve individual readers, educators, and content creators. Alexandria uses advanced retrieval-augmented generation (RAG), role-play simulation, and supportive tools to create personalized learning experiences.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd Alexandria_app
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

5. **Run the application:**
   ```bash
   # Start the FastAPI backend
   uvicorn src.main:app --reload

   # In another terminal, start Streamlit frontend
   streamlit run src/frontend/app.py
   ```

6. **Access the application:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health
   - Streamlit UI: http://localhost:8501

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_main.py -v
```

## 📁 Project Structure

```
alexandria-app/
├── src/
│   ├── main.py              # FastAPI application entry point
│   ├── api/                 # API endpoints
│   │   ├── health.py        # Health check endpoints
│   │   ├── books.py         # Book management endpoints
│   │   └── chat.py          # Chat/Q&A endpoints
│   ├── utils/               # Utility modules
│   │   ├── config.py        # Configuration management
│   │   ├── database.py      # Vector database abstraction
│   │   └── logger.py        # Logging setup
│   ├── rag/                 # RAG components
│   ├── mcp/                 # MCP server
│   ├── tools/               # Modular tools
│   └── prompts/             # Prompt templates
├── tests/                   # Test suite
├── data/                    # Data storage
│   ├── books/               # Uploaded books
│   ├── chroma_db/           # Vector database
│   └── users/               # User data
└── docs/                    # **Complete project documentation**
```

## ⚙️ Configuration

Key environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Vector Database (Phase 1: Chroma)
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# File Upload
MAX_UPLOAD_SIZE_MB=50
SUPPORTED_FORMATS=pdf,epub,doc,docx,txt,html

# Application
DEBUG=true
LOG_LEVEL=info
```

## 📋 Current Status

### ✅ Completed (Phase 1 Foundation)
- [x] Project structure and documentation
- [x] FastAPI application setup
- [x] Configuration management
- [x] Logging system
- [x] Health check endpoints
- [x] Vector database abstraction (Chroma)
- [x] Basic API endpoints (placeholder)
- [x] Test suite foundation

### ✅ Completed (Phase 1.41 - Enhanced Frontend)
- [x] Enhanced Streamlit frontend with multi-module support
- [x] Advanced book management interface with metadata display
- [x] Enhanced search functionality with filtering
- [x] Content relationships visualization
- [x] Analytics dashboard with insights
- [x] Theme selection system (Light, Dark, DBC Classic)
- [x] Q&A chat interface
- [x] Integration with enhanced RAG backend

### 🔄 In Progress
- [ ] MCP server with tools
- [ ] Advanced content relationship discovery

### 📅 Planned (Phase 2)
- [ ] Enhanced RAG with agent capabilities
- [ ] Role-play scenarios
- [ ] Progress tracking
- [ ] Next.js frontend
- [ ] Multi-user support with authentication
- [ ] Supabase migration

## 🛠️ Development

### Migration Tools (Phase 1.35)

The migration architecture implemented in Phase 1.35 provides production-ready tools for database migrations:

```bash
# Run content migration from legacy book schema to unified schema
python scripts/run_migration.py migrate

# Validate migration results
python scripts/run_migration.py validate

# Test Supabase connection readiness
python scripts/run_migration.py test-supabase

# Run full migration suite
python scripts/run_migration.py all

# Enable verbose logging
python scripts/run_migration.py migrate --verbose
```

**Migration Features:**
- ✅ Dual-write capability (Chroma + Supabase)
- ✅ Zero-downtime migration strategy
- ✅ Data validation and consistency checking
- ✅ Complete rollback capabilities
- ✅ Performance testing and optimization

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Docker (Coming Soon)

```bash
# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

## 📚 Supported File Formats

- **PDF** - Adobe Portable Document Format
- **EPUB** - Electronic Publication format
- **DOC/DOCX** - Microsoft Word documents
- **TXT** - Plain text files
- **HTML** - Web pages and HTML documents

## 🔧 API Endpoints

### Health Checks
- `GET /api/v1/health` - Basic health status
- `GET /api/v1/health/detailed` - Detailed diagnostics
- `GET /api/v1/health/ready` - Readiness probe
- `GET /api/v1/health/live` - Liveness probe

### Books (Placeholder)
- `POST /api/v1/books/upload` - Upload book files
- `GET /api/v1/books` - List uploaded books
- `GET /api/v1/books/{id}` - Get book details
- `DELETE /api/v1/books/{id}` - Delete book

### Chat/Q&A (Placeholder)
- `POST /api/v1/chat/query` - Ask questions about books
- `GET /api/v1/chat/conversations` - List conversations
- `GET /api/v1/chat/conversations/{id}` - Get conversation history

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **API key errors**: Check `.env` file has correct OpenAI API key
3. **Database errors**: Ensure `data/chroma_db` directory exists and is writable
4. **Port conflicts**: Change ports in `.env` if 8000 or 8501 are in use

### Getting Help

For issues and questions:
1. Check the logs in `logs/alexandria.log`
2. Review the API documentation at `/docs`
3. Run the test suite to verify setup
4. Check environment variable configuration

## 📚 Documentation

Complete project documentation is located in the `/docs` folder:

- **Strategic Planning**: [`/docs/PLANNING_OVERVIEW.md`](./docs/PLANNING_OVERVIEW.md) - Project strategy and high-level goals
- **Development Roadmap**: [`/docs/ROADMAP_OVERVIEW.md`](./docs/ROADMAP_OVERVIEW.md) - Strategic roadmap and timeline overview
- **Task Tracking**: [`/docs/TASK_*.md`](./docs/) - Development tasks organized by category (Frontend, Backend, Infrastructure, Security, Features, Misc)
- **Technical Architecture**: [`/docs/ARCHITECTURE_OVERVIEW.md`](./docs/ARCHITECTURE_OVERVIEW.md) - System architecture and design
- **Future Features**: [`/docs/FUTURE_FEATURES.md`](./docs/FUTURE_FEATURES.md) - Planned enhancements including Classic Literature Comprehension Engine
- **Development Guidelines**: [`CLAUDE.md`](./CLAUDE.md) - Development standards and requirements
- **Deployment & DevOps**: [`/docs/DEPLOYMENT_GUIDE.md`](./docs/DEPLOYMENT_GUIDE.md) - CI/CD, deployment, and operational procedures

For a complete documentation index, see [`/docs/README.md`](./docs/README.md).

---

*Alexandria Platform is in active development. Current focus is on Phase 1 Smart Library implementation.*