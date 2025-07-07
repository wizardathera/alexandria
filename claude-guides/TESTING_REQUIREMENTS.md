# TESTING_REQUIREMENTS.md

## Testing Requirements (Mandatory)

Every feature must include exactly these test types:
1. **Expected behavior test** - normal use case
2. **Edge case test** - boundary conditions  
3. **Failure scenario test** - error handling

### Test Structure Example
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

## Mock Strategy

**Mock Strategy**:
- **Always mock external services** (OpenAI, Supabase, Chroma)
- Store mock data in `/tests/fixtures/`
- Never make real API calls in tests

### Mocking Guidelines
- Mock external services (OpenAI API, Supabase, Chroma)
- Use `pytest-mock` or `unittest.mock`
- Store mock data in `/tests/fixtures/`

## Test Organization

### Required Test Categories for RAG Components

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

## Performance Benchmarks

- **Query Response Time**: <3 seconds for 95% of queries
- **Search Relevance**: >85% user satisfaction with search results
- **Recommendation Quality**: >80% relevance for AI-suggested relationships
- **Concurrent Users**: Support 100+ simultaneous queries in Phase 2

## Development Workflow for RAG Features

### Feature Development Process
1. **Design Review**: Confirm multi-module implications and migration impact
2. **Interface Design**: Create abstract interfaces before implementation
3. **Test-Driven Development**: Write tests for expected behavior, edge cases, and failures
4. **Performance Testing**: Validate response times and concurrent user support
5. **Migration Testing**: Verify compatibility with both Chroma and Supabase
6. **Documentation Update**: Update architecture and API documentation

### Code Review Checklist
- ✅ Supports all three modules (Library, LMS, Marketplace)
- ✅ Includes proper permission filtering
- ✅ Maintains performance requirements
- ✅ Follows migration-ready patterns
- ✅ Includes comprehensive test coverage
- ✅ Documents multi-module implications
- ✅ Validates with representative data

## Dependency Maintenance Policy

**Effective Date: 2025-07-05**

### Policy Requirements

#### 1. Dependency Synchronization
- **After any development work** or environment changes, `pip freeze` must be run to capture all installed packages
- **The requirements.txt file** must be regenerated and committed to version control
- **If any new libraries** were introduced but not installed via pip, they must be manually added to requirements.txt

#### 2. Environment Validation
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

#### 3. Documentation Requirements
- **Any changes to dependencies** must be documented in the CHANGELOG.md under the "Dependencies" section
- **The project must not rely** on undeclared packages to avoid ModuleNotFoundError at runtime
- **Version pinning strategy**: Pin exact versions for critical dependencies, use ranges for utility packages

#### 4. Development Workflow Integration
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