# DEVELOPMENT_GUIDELINES.md

## Communication Style & Collaboration

### How Claude Should Respond
- **Be very thorough** in all explanations
- **Use clear, plain language** - avoid technical jargon
- **Confirm steps before assuming or deleting anything**
- **Always ask clarifying questions** if uncertain about requirements
- Provide step-by-step instructions for running or testing code
- Update documentation after every significant change

### When NOT to Use Claude
- **Security-critical paths** where human review is essential
- **High hallucination risk scenarios** (complex configuration files, sensitive data handling)
- **Production deployment decisions** requiring business judgment
- **User permission and access control** design
- **Financial transactions or billing logic**

### Critical Development Rules

#### Before Any Coding Session
1. **Read** `docs/PLANNING.md`, `docs/ROADMAP.md`, and `docs/TASKS.md` files first
2. **Understand current phase** - Phase 2.0 Multi-User Platform + Learning Suite Foundation
3. **Review multi-tenant architecture** - understand user isolation and permission requirements
4. **Consider Hypatia integration** - conversational AI assistant scope and limitations
5. **Ask clarifying questions** if requirements are unclear
6. **Confirm approach** before making significant changes
7. **Design with scalability in mind** - multi-tenant, multi-user architecture

#### Quality Assurance
- **Test everything** - no untested code in main branch
- **Document everything** - especially for non-technical users
- **Validate assumptions** - don't guess at requirements

#### Completion Criteria
Mark tasks complete only when:
- ✅ All tests pass
- ✅ Code is properly documented
- ✅ Feature works as specified
- ✅ Documentation is updated
- ✅ RAG performance metrics are validated
- ✅ Migration path is tested (if applicable)

## Code Standards & Conventions

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

## Development Commands

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