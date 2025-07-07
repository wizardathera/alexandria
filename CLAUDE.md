# CLAUDE.md

This file provides core coding discipline rules and guidance to Claude Code (claude.ai/code) when working on the Alexandria platform.

## 📘 Mission

Alexandria is a comprehensive AI-powered platform combining Smart Library, Learning Suite, and Marketplace capabilities. The current focus is Phase 1: Smart Library with robust RAG capabilities for individual readers and learners.

## ⚠️ CRITICAL CODING RULES

### File Management (NON-NEGOTIABLE)
- **Never exceed 500 lines** in any single Python file - split modules when necessary
- **Test files MUST be in `/tests/` folder** - never leave test files in root directory
- **Modular design** - keep API calls and business logic separate
- **Never hardcode credentials** - use environment variables only

### Code Quality Standards
- **Never hallucinate** libraries, functions, or file paths
- Always read `docs/PLANNING.md`, `docs/ROADMAP.md`, and `docs/TASKS.md` before coding
- **PEP8 compliance** with type hints for all functions
- **Google-style docstrings** required for every function
- **Include detailed comments** explaining non-obvious code

### Testing Requirements (MANDATORY)
Every feature must include exactly these test types:
1. **Expected behavior test** - normal use case
2. **Edge case test** - boundary conditions  
3. **Failure scenario test** - error handling

**Mock Strategy**:
- **Always mock external services** (OpenAI, Supabase, Chroma)
- Store mock data in `/tests/fixtures/`
- Never make real API calls in tests

### Streamlit Rules
- **Only one `st.set_page_config()` call** per application - must be in main entry point
- **No duplicate page configurations** across modules
- **Modular functions** for easy migration to Next.js in Phase 2

### When NOT to Use Claude
- **Security-critical paths** where human review is essential
- **High hallucination risk scenarios** (complex configuration files, sensitive data handling)
- **Production deployment decisions** requiring business judgment
- **User permission and access control** design
- **Financial transactions or billing logic**

### Task Completion Criteria
Mark tasks complete only when:
- ✅ All tests pass
- ✅ Code is properly documented
- ✅ Feature works as specified
- ✅ Documentation is updated
- ✅ All test files are in `/tests/` folder

## 🗂️ Navigation Index

For detailed guidance on specific topics, refer to these modular files:

| **Topic** | **File** | **Contents** |
|-----------|----------|--------------|
| **Platform Vision** | [claude-guides/PROJECT_OVERVIEW.md](claude-guides/PROJECT_OVERVIEW.md) | Platform vision, modules, user roles, monetization roadmap |
| **Development Standards** | [claude-guides/DEVELOPMENT_GUIDELINES.md](claude-guides/DEVELOPMENT_GUIDELINES.md) | Collaboration assumptions, response tone, Claude's constraints |
| **Technical Architecture** | [claude-guides/ARCHITECTURE_OVERVIEW.md](claude-guides/ARCHITECTURE_OVERVIEW.md) | Technical stack, RAG architecture, backend/frontend models |
| **Testing Standards** | [claude-guides/TESTING_REQUIREMENTS.md](claude-guides/TESTING_REQUIREMENTS.md) | Test coverage, formats, expectations for Claude's test outputs |
| **Hypatia Assistant** | [claude-guides/HYPATIA_GUIDELINES.md](claude-guides/HYPATIA_GUIDELINES.md) | Rules for Hypatia's functionality, tone, memory, limits |
| **RAG Implementation** | [claude-guides/RAG_GUIDELINES.md](claude-guides/RAG_GUIDELINES.md) | Embedding best practices, chunking, filters, hallucination protection |
| **File Organization** | [claude-guides/FILE_STRUCTURE.md](claude-guides/FILE_STRUCTURE.md) | Folder rules, documentation sync, task and phase formatting |

## 🆘 Getting Help

If you need clarification on any aspect of this project:
1. Ask specific questions about requirements
2. Propose alternative approaches when appropriate
3. Explain trade-offs clearly
4. Always prioritize user experience and code maintainability