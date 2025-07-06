**This document is synchronized with PLANNING_OVERVIEW.md and ROADMAP_OVERVIEW.md as of 2025-07-05.**

# Alexandria — Deployment & DevOps Guide

Version: 1.0  
Date: 2025-07-05  
Owner: Andrew Folkler

This guide describes how code moves from a developer's laptop to production, how we keep it running, and how we recover if something goes wrong.  
Aligned with the current Technical Spec, Planning, and Roadmap documentation.

---

## Contents

1. Repository & Branching Strategy  
2. Local-Dev Environment  
3. Environments & Infrastructure Map  
4. CI / CD Pipelines  
5. Secrets Management  
6. Database Migration Workflow  
7. Monitoring, Alerting & Logging  
8. Backup & Restore Procedures  
9. Rollback & Hot-Fix Playbook  
10. Scaling & Cost Controls  
11. Disaster Recovery & Chaos Simulations  
12. Change-Management & Release Calendar  

---

## 1 · Repository & Branching Strategy

| Branch       | Purpose                    | Rules                                        |
|--------------|----------------------------|----------------------------------------------|
| main         | Production-ready code      | Protected: PR + 2 reviews + green CI        |
| develop      | Integration branch (staging)| Protected: 1 review + green CI             |
| feature/*    | Short-lived features/fixes | Rebase onto develop before PR               |
| hotfix/*     | Urgent prod patches        | Branch from main; PR back to main & develop |

---

## 2 · Local-Dev Environment

**Prerequisites:**

- python 3.11+ (FastAPI services & RAG pipeline)
- streamlit latest (Frontend Phase 1)
- docker compose v2
- supabase cli latest (Phase 2+ migration)
- chroma for local vector storage

**Quick-Start:**

```bash
git clone git@github.com:alexandria-org/alexandria-platform.git
cd alexandria-platform
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
streamlit run src/frontend/app.py
```

**Test Suite:**

```bash
pytest                                  # unit + integration
pytest tests/test_ingestion.py         # content ingestion tests
pytest tests/test_rag_service.py       # RAG functionality tests
bandit -r src                          # static security scan
```

---

## 3 · Environments & Infrastructure Map

| Layer         | Local              | CI Preview          | Staging             | Production           |
|---------------|--------------------|--------------------|---------------------|---------------------|
| Front-end     | localhost:8501     | Streamlit Cloud    | staging.alexandria.ai | app.alexandria.ai   |
| API           | localhost:8000     | FastAPI Test       | Same w/ STAGING=1   | PROD=1 flag         |
| DB            | Chroma local       | Chroma test        | Supabase staging    | Supabase prod       |
| Vector Store  | Chroma sqlite      | Chroma memory      | pgvector staging    | pgvector prod       |
| Auth          | None (Phase 1)     | —                  | Supabase Auth       | Supabase Auth       |
| OpenAI        | Dev key            | Org dev key        | Org staging key     | Org prod key        |

**Infra Diagram:** `docs/architecture/alexandria_infra_2025-07.png`

---

## 4 · CI / CD Pipelines

| Workflow File                      | Trigger           | Steps                                        |
|------------------------------------|-------------------|----------------------------------------------|
| `.github/workflows/ci.yml`        | PR push           | Lint, unit tests, safety scan, integration  |
| `.github/workflows/staging-deploy.yml` | Merge to develop | Build, deploy to staging, smoke tests      |
| `.github/workflows/prod-deploy.yml`   | Manual action    | Tag release, build, deploy, DB migrations   |

**Manual Gates:**

- Staging QA sign-off
- Security Lead approves dependency scan
- PM checks launch checklist

---

## 5 · Secrets Management

| Context   | Storage                          |
|-----------|----------------------------------|
| Local     | `.env` (gitignored)             |
| CI        | GitHub Actions Encrypted Secrets|
| Staging   | Environment Variables           |
| Production| Secure Environment Variables    |

**Rotation:** Every 90 days by DevOps.

**Required Secrets:**
- `OPENAI_API_KEY`
- `SUPABASE_URL` (Phase 2+)
- `SUPABASE_KEY` (Phase 2+)

---

## 6 · Database Migration Workflow

**Phase 1 (Chroma):**
1. Update vector schema in `src/utils/enhanced_database.py`
2. Run local tests with `pytest tests/test_enhanced_embedding_service.py`
3. Deploy with code changes

**Phase 2+ (Supabase):**
1. Write migration script (`src/migrations/*.sql`)
2. Test locally: `python scripts/run_migration.py`
3. PR includes migration file; reviewer checks backward compatibility
4. On staging deploy: auto-migrate
5. Prod deploy: guarded migrate with auto-rollback on failure

**Rollback Strategy:**
- Phase 1: Rebuild Chroma database from source documents
- Phase 2+: Postgres transactional DDL, abort build if fail

---

## 7 · Monitoring, Alerting & Logging

| Service           | Purpose                    | Alert Channel  |
|-------------------|----------------------------|----------------|
| Application Logs  | FastAPI/Streamlit errors   | File rotation  |
| Health Checks     | `/health` endpoint         | Uptime monitor |
| RAG Performance   | Query response times       | Log analysis   |
| Token Usage       | OpenAI API consumption     | Daily reports  |

**Error Budget:** 2% monthly; breach = launch freeze + post-mortem.

**Log Files:**
- `logs/alexandria.log` - Application events
- `logs/performance.log` - Query metrics

---

## 8 · Backup & Restore

| Asset              | Method                    | Frequency    | Retention   |
|--------------------|---------------------------|--------------|-------------|
| Chroma Vector DB   | Export embeddings to JSON| Daily        | 7 days      |
| Book PDFs          | File system backup        | Daily        | 30 days     |
| Configuration      | Git repository backup     | On push      | Indefinite  |
| Supabase DB        | PITR + nightly snapshot   | Continuous   | 30 days     |

**Restore Drill:** Quarterly to isolated staging.

---

## 9 · Rollback & Hot-Fix Playbook

1. **Detect issue** (alert/smoke test)
2. **Triage severity** (P0/P1)
3. **Immediate action:**
   ```bash
   # Rollback to previous working version
   git revert <commit_hash>
   # Redeploy immediately
   ```
4. **Post-Rollback:**
   - Disable feature flag if needed
   - Open hotfix branch
   - Fix, test, merge, redeploy

**MTTR Goal:** <30 min for P0 outage.

---

## 10 · Scaling & Cost Controls

| Layer          | Scaling                          | Cost Cap        |
|----------------|----------------------------------|-----------------|
| Streamlit      | Single instance (Phase 1)       | $50/mo          |
| FastAPI        | Auto-scale; 10 concurrent max    | $100/mo alert   |
| Chroma         | Local storage limit 10GB         | Free            |
| Supabase       | Auto-scale; 1k RPS limit         | $200/mo         |
| OpenAI tokens  | Hard cap $500/mo; rate limiting  | $400/mo alert   |

---

## 11 · Disaster Recovery & Chaos Simulations

| Scenario                | Planned Response                                      |
|-------------------------|-------------------------------------------------------|
| Vector DB corruption    | Rebuild from source documents (2-4h RTO)            |
| API key compromise      | Rotate keys, update all environments (<1h)          |
| Supabase outage         | Graceful degradation to read-only mode              |
| Token flooding          | Per-user rate limiting; circuit-breaker returns 429 |

**Quarterly Chaos Day:** Simulate vector DB rebuild and API key rotation.

---

## 12 · Change Management & Release Calendar

| Release Type   | Cadence          | Owner            |
|----------------|------------------|------------------|
| Patch          | ad-hoc           | Tech Lead        |
| Minor Feature  | bi-weekly        | PM               |
| Major Phase    | As scheduled     | Steering Team    |

**Launch Checklist:** 
- All tests green
- Security scan clean
- Performance benchmarks passed
- Documentation updated
- Rollback plan confirmed

---

## Alexandria-Specific Considerations

### Module Deployment Strategy

**Phase 1:** Smart Library only
- Streamlit frontend
- FastAPI backend
- Chroma vector storage
- Single-user mode

**Phase 2:** + Learning Suite
- Next.js frontend migration
- Supabase migration
- Multi-user authentication
- Course management features

**Phase 3:** + Marketplace
- Payment processing
- Content monetization
- Community features

### RAG System Deployment

**Performance Requirements:**
- Query response time: <3 seconds
- Embedding generation: <5 seconds per document
- Concurrent users: 10 (Phase 1), 100+ (Phase 2+)

**Quality Assurance:**
- A/B testing framework for RAG improvements
- Content relevance scoring
- User feedback integration

### Migration Procedures

**Chroma → Supabase Vector Migration:**
1. Export all embeddings from Chroma
2. Set up parallel Supabase instance
3. Dual-write to both systems
4. Validate data consistency
5. Switch read traffic to Supabase
6. Deprecate Chroma

**Streamlit → Next.js Migration:**
1. Build Next.js components
2. Parallel deployment
3. Feature flag rollout
4. User migration
5. Streamlit deprecation

---

## Document History

| Version | Date       | Author         | Notes                                    |
|---------|------------|----------------|------------------------------------------|
| 1.0     | 2025-07-05 | Andrew Folkler | Initial deployment guide for Alexandria  |