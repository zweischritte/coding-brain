# Phase 5 Continuation: API Route Wiring

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**PRD**: `docs/PRD-PHASE5-API-ROUTE-WIRING.md`
**Template**: `docs/CONTINUATION-PROMPT-TEMPLATE.md` (use when creating next phase prompt)
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/SYSTEM-CONTEXT.md` for system overview (if unfamiliar with the codebase)
2. Read `docs/PRD-PHASE5-API-ROUTE-WIRING.md` for detailed feature specs and test cases
3. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress and daily log
4. Check Section 4 (Task Tracker) below for current status
5. Check the Daily Log "Resume Point" column for exactly where to continue

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update task status tables for completed work
   - Add entry to Daily Log with date, work completed, and notes
2. **UPDATE Section 4 (Task Tracker)** with completed/remaining work
3. **CREATE next continuation prompt** following this same format
4. Commit all files together:

```bash
git add docs/CONTINUATION-PROMPT-PHASE5-API-ROUTES.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "docs: update Phase 5 session progress"
```

---

## 1. Phase 5 Context

**Prerequisite**: Phase 4 Multi-tenant Stores COMPLETE (125 tests)

Phase 4 built these stores that are **not yet exposed via API**:
- `ScopedMemoryStore` - PostgreSQL with RLS
- `PostgresFeedbackStore` - Feedback events with retention queries
- `PostgresExperimentStore` - A/B experiments with status history
- `ValkeyEpisodicStore` - Session-scoped ephemeral memory
- `TenantQdrantStore` - Vector embeddings with tenant filtering
- `TenantOpenSearchStore` - Hybrid search with tenant alias routing

**Goal**: Wire these stores to REST API routes with proper auth, validation, and tests.

**Blocker to Fix First**: Phase 1 MCP auth incomplete (SSE auth + tool scopes)

**Target**: ~100 new tests, reaching ~3,304 total tests

---

## 2. Command Reference

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Run router tests only
docker compose exec codingbrain-mcp pytest tests/routers/ -v

# Run security tests
docker compose exec codingbrain-mcp pytest tests/security/ -v

# Run specific test file
docker compose exec codingbrain-mcp pytest tests/routers/test_feedback_router.py -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Check pre-existing failures
docker compose exec codingbrain-mcp pytest tests/security/test_router_auth.py -v
```

---

## 3. Architecture Patterns

### Router Pattern (existing style)

```python
from fastapi import APIRouter, Depends, HTTPException
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

@router.post("")
async def create_feedback(
    feedback: FeedbackCreate,
    principal: Principal = Depends(require_scopes(Scope.FEEDBACK_WRITE)),
    db: Session = Depends(get_db)
):
    """Create feedback event."""
    store = PostgresFeedbackStore(db)
    return store.append(feedback, principal.user_id, principal.org_id)
```

### Store Injection Pattern

```python
from app.stores.feedback_store import PostgresFeedbackStore

def get_feedback_store(db: Session = Depends(get_db)) -> PostgresFeedbackStore:
    """Dependency for feedback store."""
    return PostgresFeedbackStore(db)

@router.get("/metrics")
async def get_metrics(
    principal: Principal = Depends(require_scopes(Scope.FEEDBACK_READ)),
    store: PostgresFeedbackStore = Depends(get_feedback_store)
):
    return store.get_aggregate_metrics(principal.org_id)
```

### Tenant Isolation Pattern

```python
# ALWAYS filter by user from principal - never trust client input
user = db.query(User).filter(User.user_id == principal.user_id).first()
if not user:
    raise HTTPException(status_code=404, detail="User not found")

# Filter all queries by user.id for tenant isolation
query = db.query(Resource).filter(Resource.user_id == user.id)
```

---

## 4. Task Tracker

### Feature Progress

| # | Feature | Status | Tests Written | Tests Passing | Commit |
|---|---------|--------|---------------|---------------|--------|
| 0 | Fix pre-existing test failures | NOT STARTED | 0 | 0 | - |
| 1 | MCP SSE Authentication | NOT STARTED | 0 | 0 | - |
| 2 | Add New Scopes | NOT STARTED | 0 | 0 | - |
| 3 | Feedback Router (4 endpoints) | NOT STARTED | 0 | 0 | - |
| 4 | Experiments Router (7 endpoints) | NOT STARTED | 0 | 0 | - |
| 5 | Search Router (3 endpoints) | NOT STARTED | 0 | 0 | - |
| 6 | Register routers in main.py | NOT STARTED | 0 | 0 | - |

### Current Task Details

**Next Task**: Feature 0 - Fix pre-existing test failures

**Subtasks**:
1. [ ] Run `test_router_auth.py` to see exact failure messages
2. [ ] Fix `test_delete_memory_requires_auth` - Wrong endpoint signature
3. [ ] Fix `test_create_app_requires_auth` - No POST /apps endpoint
4. [ ] Fix `test_export_requires_auth` - Needs investigation
5. [ ] Fix `test_memories_delete_requires_scope` - Wrong endpoint signature
6. [ ] Verify all 4 tests pass
7. [ ] Commit fix

**After Feature 0**: Proceed to Feature 1 (MCP SSE Auth) or Feature 2 (Add Scopes)

---

## 5. Feature Specifications Summary

See `docs/PRD-PHASE5-API-ROUTE-WIRING.md` for full details.

### Feature 0: Fix Pre-existing Test Failures (BLOCKING)

4 tests failing in `test_router_auth.py`:
- `test_delete_memory_requires_auth`
- `test_create_app_requires_auth`
- `test_export_requires_auth`
- `test_memories_delete_requires_scope`

### Feature 1: MCP SSE Authentication (BLOCKING)

Add JWT auth to `/mcp` and `/concepts` SSE endpoints.

### Feature 2: Add New Scopes

```python
FEEDBACK_READ = "feedback:read"
FEEDBACK_WRITE = "feedback:write"
EXPERIMENTS_READ = "experiments:read"
EXPERIMENTS_WRITE = "experiments:write"
SEARCH_READ = "search:read"
```

### Feature 3: Feedback Router

| Method | Path | Scope |
|--------|------|-------|
| POST | `/api/v1/feedback` | FEEDBACK_WRITE |
| GET | `/api/v1/feedback` | FEEDBACK_READ |
| GET | `/api/v1/feedback/metrics` | FEEDBACK_READ |
| GET | `/api/v1/feedback/by-tool` | FEEDBACK_READ |

### Feature 4: Experiments Router

| Method | Path | Scope |
|--------|------|-------|
| POST | `/api/v1/experiments` | EXPERIMENTS_WRITE |
| GET | `/api/v1/experiments` | EXPERIMENTS_READ |
| GET | `/api/v1/experiments/{id}` | EXPERIMENTS_READ |
| PUT | `/api/v1/experiments/{id}/status` | EXPERIMENTS_WRITE |
| POST | `/api/v1/experiments/{id}/assign` | EXPERIMENTS_WRITE |
| GET | `/api/v1/experiments/{id}/assignment` | EXPERIMENTS_READ |
| GET | `/api/v1/experiments/{id}/history` | EXPERIMENTS_READ |

### Feature 5: Search Router

| Method | Path | Scope |
|--------|------|-------|
| POST | `/api/v1/search` | SEARCH_READ |
| POST | `/api/v1/search/lexical` | SEARCH_READ |
| POST | `/api/v1/search/semantic` | SEARCH_READ |

---

## 6. Known Issues

1. **MCP auth tests hang** - `test_mcp_auth.py` takes too long; blocked
2. **Pre-existing test failures** in `test_router_auth.py` (4 tests)
3. **Pydantic deprecation** - `app/schemas.py:54` uses V1 @validator

---

## 7. Key Files

**Stores (use these in routers):**
- `openmemory/api/app/stores/feedback_store.py` - PostgresFeedbackStore
- `openmemory/api/app/stores/experiment_store.py` - PostgresExperimentStore
- `openmemory/api/app/stores/opensearch_store.py` - TenantOpenSearchStore

**Existing routers (reference for patterns):**
- `openmemory/api/app/routers/memories.py` - Memory CRUD with Principal auth
- `openmemory/api/app/routers/apps.py` - App management with scopes

**Security (required for auth):**
- `openmemory/api/app/security/dependencies.py` - require_scopes, get_current_principal
- `openmemory/api/app/security/types.py` - Principal, Scope enum

**MCP (needs auth fix):**
- `openmemory/api/app/routers/mcp_server.py` - SSE endpoints need auth

**Tests (to fix/add):**
- `openmemory/api/tests/security/test_router_auth.py` - Fix 4 failing tests
- `openmemory/api/tests/routers/test_feedback_router.py` - NEW
- `openmemory/api/tests/routers/test_experiments_router.py` - NEW
- `openmemory/api/tests/routers/test_search_router.py` - NEW

---

## 8. Test Estimates

| Router | Estimated Tests |
|--------|-----------------|
| Fix pre-existing failures | 4 (fix) |
| MCP auth fix | 5 |
| New scopes | 4 |
| Feedback router | 25-30 |
| Experiments router | 30-35 |
| Search router | 20-25 |
| **Total Phase 5** | **90-110** |

**Current**: 3,204 tests
**Target**: 3,204 + 100 = ~3,304 tests

---

## 9. Success Criteria

Phase 5 is complete when:

1. [ ] Pre-existing test_router_auth.py failures fixed (4 tests)
2. [ ] MCP SSE endpoints have auth (Phase 1 blocker resolved)
3. [ ] MCP tools have permission checks
4. [ ] All new scopes added to Scope enum (5 scopes)
5. [ ] Feedback router implemented with 4 endpoints (25+ tests)
6. [ ] Experiments router implemented with 7 endpoints (30+ tests)
7. [ ] Search router implemented with 3 endpoints (20+ tests)
8. [ ] All routers registered in main.py
9. [ ] Zero regression (all 3,204+ existing tests pass)

---

## 10. Session End Checklist

Before ending session, complete these tasks:

1. [ ] Update test counts in `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
2. [ ] Update Feature Progress table in Section 4 above
3. [ ] Add entry to Daily Log with work completed
4. [ ] Update "Next Task" in Section 4 with remaining work
5. [ ] Create next continuation prompt (if phase complete or major milestone)
6. [ ] Commit changes:

```bash
git add docs/CONTINUATION-PROMPT-PHASE5-API-ROUTES.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "$(cat <<'EOF'
docs: update Phase 5 session progress

Session: YYYY-MM-DD
- [Brief summary of work completed]
- [Test count if changed]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## 11. Last Session Summary (2025-12-27)

**Phase 5 PRD Complete**: Created comprehensive PRD with:
- 10 success criteria
- 6 features with detailed specs
- ~100 test specifications
- Architecture patterns from codebase exploration

**Exploration Results** (4 parallel sub-agents):
- Router patterns: Principal dependency, require_scopes, tenant isolation via user.id
- Security: Scope enum, JWT validation, DPoP support
- Stores: PostgresFeedbackStore, PostgresExperimentStore, TenantOpenSearchStore APIs
- Tests: JWT mocking, TestClient patterns, scope enforcement tests

**Current State**: 3,204 tests, ready to start Feature 0 (fix test failures)

**Resume Point**: Run `test_router_auth.py` to investigate 4 failing tests
