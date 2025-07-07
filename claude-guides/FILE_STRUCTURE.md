# FILE_STRUCTURE.md

## 📂 Required File Structure

```
alexandria-app/
├── docs/
│   ├── PLANNING.md              # High-level planning overview of phases and milestones
│   ├── ROADMAP.md               # Business milestones and feature progression (not detailed tasks)
│   ├── ARCHITECTURE.md          # Combined technical architecture and design decisions
│   ├── TASKS.md                 # Current, active Epics and Stories only
│   ├── TECHNICAL_SPECIFICATIONS.md # Technical stack and implementation details
│   ├── PRODUCT_REQUIREMENTS.md  # Complete product requirements and specifications
│   ├── DEPLOYMENT_GUIDE.md      # Setup, deployment, and operations guide
│   └── SECURITY_PRIVACY_PLAN.md # Security and compliance considerations
├── claude-guides/               # Modular Claude guidance files
│   ├── PROJECT_OVERVIEW.md      # Platform vision and modules
│   ├── DEVELOPMENT_GUIDELINES.md # Collaboration and code standards
│   ├── ARCHITECTURE_OVERVIEW.md # Technical stack details
│   ├── TESTING_REQUIREMENTS.md  # Test coverage and requirements
│   ├── HYPATIA_GUIDELINES.md    # Hypatia assistant rules
│   ├── RAG_GUIDELINES.md        # RAG best practices
│   └── FILE_STRUCTURE.md        # This file - folder rules
├── README.md                    # Installation and usage instructions
├── CLAUDE.md                    # Core development guidance (trimmed)
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml          # Multi-service orchestration
├── /src/
│   ├── main.py                  # Application entry point with modular routing
│   ├── /library/                # Smart Library module (Phase 1)
│   ├── /lms/                    # Learning Suite module (Phase 2) 
│   ├── /marketplace/            # Marketplace module (Phase 3)
│   ├── /shared/                 # Shared services (auth, payments, etc.)
│   ├── /rag/                    # Agentic RAG components
│   ├── /mcp/                    # MCP server implementation
│   ├── /prompts/                # Prompt templates
│   ├── /tools/                  # Modular tool implementations
│   ├── /utils/                  # Helper functions
│   └── /api/                    # FastAPI endpoints with module namespaces
└── /tests/
    ├── /library/                # Smart Library tests
    ├── /lms/                    # Learning Suite tests  
    ├── /marketplace/            # Marketplace tests
    ├── /shared/                 # Shared services tests
    └── /fixtures/               # Test data and mocks
```

## 📚 Documentation Structure

**New Streamlined System** (Effective 2025-07-05):

### Core Documentation Files
- **docs/PLANNING.md** - High-level planning overview of phases and milestones
- **docs/ROADMAP.md** - Business milestones and feature progression (not detailed tasks)
- **docs/ARCHITECTURE.md** - Combined technical architecture and design decisions
- **docs/TASKS.md** - Current, active Epics and Stories only
- **docs/TECHNICAL_SPECIFICATIONS.md** - Technical stack and implementation details
- **docs/PRODUCT_REQUIREMENTS.md** - Complete product requirements and specifications
- **docs/DEPLOYMENT_GUIDE.md** - Setup, deployment, and operations guide
- **docs/SECURITY_PRIVACY_PLAN.md** - Security and compliance considerations

### Documentation Rules
- **All tasks are now tracked only in `TASKS.md`**
- **Completed Epics and Stories are removed from `TASKS.md` and archived via Git history**
- **No duplication of tasks or subtasks in planning or roadmap documents**
- **`PLANNING.md` contains only phase summaries and references `TASKS.md`**
- **`ROADMAP.md` contains only business milestones, not implementation tasks**
- **`ARCHITECTURE.md` is the sole technical architecture reference**
- **Claude must never reintroduce deprecated documentation files**

### Deprecated Files (Never Reference Again)
- All PLANNING_* files (PLANNING_PHASES.md, PLANNING_TASKS_BREAKDOWN.md, etc.)
- All ROADMAP_* files (ROADMAP_PHASES.md, ROADMAP_TIMELINES.md, etc.)
- All TASK_* files (TASK_BACKEND.md, TASK_FRONTEND.md, etc.)
- All ARCHITECTURE_* files (ARCHITECTURE_OVERVIEW.md, ARCHITECTURE_BACKEND.md, etc.)
- FUTURE_FEATURES.md
- ARCHITECTURE_ARCHIVE_2024.md

## Task Management Workflow

### Task Lifecycle
1. **New tasks** are added to `TASKS.md` with Epic, Story, Labels, Priority, and Status
2. **Active work** is tracked with Status updates (To Do → In Progress → Done)
3. **Completed tasks** are immediately removed from `TASKS.md` (archived in Git history)
4. **Cross-references** use only the current active task numbers in `TASKS.md`

### Epic and Story Structure
- **Epic**: Large feature area (e.g., "Frontend Migration", "Backend Stability")
- **Story**: Specific deliverable within an epic (e.g., "Fix Search Endpoint Functionality")
- **Subtasks**: Implementation steps listed as bullets under each story
- **Labels**: Technology tags (frontend, backend, ai, etc.)
- **Priority**: High/Medium/Low based on impact and dependencies

### Documentation Synchronization
- **No manual synchronization required** - single source of truth in each file
- **No cross-file duplication** - each file has distinct purpose
- **Git history** serves as archive for completed work
- **`TASKS.md`** is the only file requiring frequent updates

## 📋 Task & Phase Numbering Convention

**CRITICAL RULE**: When outlining or generating tasks in ANY project documentation, you MUST always use the **structured task & phase numbering convention** defined below to ensure clear sequential order that can be easily followed and referenced.

### Task & Phase Numbering Convention

#### 1️⃣ Phase Levels
Each major phase is labeled with an integer followed by .0.

**Example:**
- `1.0` — Phase 1: Foundational Setup
- `2.0` — Phase 2: Production Frontend

#### 2️⃣ Subphases
Subphases within a phase are labeled by appending .1, .2, .3, etc. to the phase number.

**Example:**
- `1.1` — Subphase 1: Environment & Configs
- `1.2` — Subphase 2: Backend APIs

#### 3️⃣ Individual Tasks
Individual tasks within a subphase are labeled by appending a second decimal point with sequential numbering.

**Example:**
- `1.11` — Initialize repository
- `1.12` — Configure .env files
- `1.21` — Build authentication routes

#### 4️⃣ Constraints
- Subphases should not contain an excessive number of major tasks. Prefer creating additional subphases if more than ~10 tasks are needed.
- Avoid lettered or inconsistent numbering formats (1A, 1.1.a, etc.).
- Use this convention consistently in all documentation

### Application Requirements
- **New Task Creation**: All newly created tasks must follow this phase/subphase/task numbering structure
- **Task Reordering**: When reordering or updating existing tasks, renumber to maintain clean sequence
- **Cross-References**: When referencing tasks, use the numeric identifier (e.g., "Depends on Task 1.21")
- **Phase Integration**: Task numbers should follow the phase.subphase.task pattern consistently

### Documentation Consistency
This structured numbering rule applies to:
- Task lists in docs/PLANNING.md
- Task definitions in docs/TASKS.md files  
- Milestone breakdowns in docs/ROADMAP.md
- Implementation steps in docs/ARCHITECTURE.md
- Any other project documentation containing task sequences

## Always Update After Changes
- `README.md` - installation and usage instructions
- `docs/PLANNING.md` - strategic planning and current phase focus
- `docs/ROADMAP.md` - strategic roadmap overview
- `docs/ARCHITECTURE.md` - high-level technical architecture decisions
- `docs/TASKS.md` - current tasks organized by status
- Function docstrings - keep them current with code changes

### User-Friendly Instructions
- Provide step-by-step setup instructions
- Include troubleshooting common issues
- Explain what each component does in plain language
- Give examples of how to use the application