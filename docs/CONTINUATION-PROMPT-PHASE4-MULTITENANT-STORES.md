# Phase 4 Continuation: Multi-tenant Data Plane Stores

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**PRD**: `docs/PRD-PHASE4-MULTITENANT-STORES.md` (detailed specs, test cases, scratchpad)
**Template**: `docs/CONTINUATION-PROMPT-TEMPLATE.md` (use when creating next phase prompt)
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/SYSTEM-CONTEXT.md` for system overview (if unfamiliar with the codebase)
1. Read `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md` for overall plan
1. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress and daily log
1. Check Section 4 (Next Tasks) below for priorities
1. Check the Daily Log "Resume Point" column for exactly where to continue

### At Session End - MANDATORY

1. **UPDATE `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:**
   - Update test counts in Summary section
   - Update task status tables for completed work
   - Add entry to Daily Log with date, work completed, and notes
1. Update Section 4 (Next Tasks) with remaining work
1. Commit BOTH files together:

```bash
git add docs/CONTINUATION-PROMPT-PHASE4-MULTITENANT-STORES.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "docs: update session progress and implementation tracker"
```

---

## 1. Current Gaps (per Implementation Plan)

**Phase 1 Gap**: MCP auth incomplete (SSE auth + tool scopes pending)
**Phase 3**: COMPLETE (tenant isolation + migration verification utilities)

---

## 2. Command Reference

```bash
# Run store tests
docker compose exec codingbrain-mcp pytest tests/stores/ -v

# Run security tests (includes tenant isolation)
docker compose exec codingbrain-mcp pytest tests/security/ -v

# Run migrations
docker compose exec codingbrain-mcp alembic upgrade head

# Check migration status
docker compose exec codingbrain-mcp alembic current

# Generate new migration
docker compose exec codingbrain-mcp alembic revision --autogenerate -m "description"

# PostgreSQL shell (for RLS testing)
docker compose exec postgres psql -U postgres -d openmemory
```

---

## 3. Architecture Patterns

### RLS Policy Pattern (PostgreSQL)

Per implementation plan, RLS uses session variable for tenant isolation:

```sql
-- Enable RLS on table
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;

-- Create policy using session variable
CREATE POLICY tenant_isolation ON memories
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Set context before queries (in Python)
session.execute(text("SET app.current_user_id = :user_id"), {"user_id": str(user_id)})
```

### Store ABC Pattern

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

T = TypeVar('T')

class BaseStore(ABC, Generic[T]):
    """Abstract base for all stores with tenant isolation."""

    @abstractmethod
    def get(self, id: UUID) -> T | None:
        """Get by ID (RLS enforces tenant scope)."""
        ...

    @abstractmethod
    def list(self, **filters) -> list[T]:
        """List all (RLS enforces tenant scope)."""
        ...

    @abstractmethod
    def create(self, entity: T) -> T:
        """Create new entity."""
        ...

    @abstractmethod
    def update(self, entity: T) -> T:
        """Update entity (RLS enforces ownership)."""
        ...

    @abstractmethod
    def delete(self, id: UUID) -> bool:
        """Delete entity (RLS enforces ownership)."""
        ...
```

### Session Context Injection

```python
from contextlib import contextmanager

@contextmanager
def tenant_session(db: Session, user_id: UUID):
    """Set RLS context for tenant isolation."""
    db.execute(text("SET app.current_user_id = :uid"), {"uid": str(user_id)})
    try:
        yield db
    finally:
        db.execute(text("RESET app.current_user_id"))
```

---

## 4. Next Tasks

**See Progress file Phase 4 table for detailed status.**

### COMPLETED: Priority 1 - ScopedMemoryStore (PostgreSQL + RLS)

1. [x] Write TDD tests for RLS enforcement (`tests/stores/test_memory_store.py`) - 16 tests
2. [x] Create Alembic migration for RLS policies on `memories` table - `add_rls_policies.py`
3. [x] Implement `tenant_session` context manager in `database.py` - 7 tests
4. [x] Implement `ScopedMemoryStore` class - `app/stores/memory_store.py`
5. [x] Add contract tests for store interface - 2 tests

### COMPLETED: Priority 2 - Other PostgreSQL Stores

1. [x] FeedbackStore with retention queries - 17 tests (`app/stores/feedback_store.py`)
2. [x] ExperimentStore with status history - 18 tests (`app/stores/experiment_store.py`)

### Priority 3: External Stores (NEXT)

1. [ ] EpisodicMemoryStore (Valkey) with TTL
2. [ ] Neo4j stores with org_id constraints
3. [ ] Qdrant EmbeddingStore with tenant payload index
4. [ ] OpenSearch tenant alias strategy

### Lower Priority

- Phase 1 MCP auth (blocked on SSE architecture)
- Technical debt items (see Progress file)

---

## 5. Known Issues

1. **MCP auth tests hang** - test_mcp_auth.py tests take too long; blocked

2. **Pre-existing test failures** in test_router_auth.py:
   - `test_delete_memory_requires_auth` - Wrong endpoint signature
   - `test_create_app_requires_auth` - No POST /apps endpoint
   - `test_export_requires_auth` - Needs investigation
   - `test_memories_delete_requires_scope` - Wrong endpoint signature

3. **Pydantic deprecation** - `app/schemas.py:54` uses V1 @validator

---

## 6. Last Session Summary (2025-12-27)

**Completed**: Phase 4 Features 3 & 4 - FeedbackStore and ExperimentStore

- Implemented `PostgresFeedbackStore` in `app/stores/feedback_store.py`
  - Append-only storage for FeedbackEvent
  - Query by user, org, query_id, experiment_id
  - Retention query support (30-day default)
  - Aggregate metrics (acceptance rate, outcome distribution, by-tool)
  - RRF weight optimization queries
- Implemented `PostgresExperimentStore` in `app/stores/experiment_store.py`
  - Full CRUD for A/B Experiment entities
  - Status history tracking with audit trail
  - Variant assignment persistence
  - Status transition timestamps (start_time, end_time)
  - Org-scoped tenant isolation
- Both stores:
  - Self-contained domain types (avoid circular imports from feedback module)
  - SQLAlchemy models with proper indexes
  - Implements ABC interface for testing
  - Tenant isolation via org_id/user_id
- Added root `conftest.py` for proper Python path setup in tests
- TDD tests: 35 new tests
  - `tests/stores/test_feedback_store.py` - 17 tests
  - `tests/stores/test_experiment_store.py` - 18 tests

**Commit**: `e1cc11ea` - feat(stores): add FeedbackStore and ExperimentStore with TDD tests

**Result**: 3,139 total tests (+35 from this session, 60 total store tests)

---

## 7. Commit Template

```bash
git add docs/CONTINUATION-PROMPT-PHASE4-MULTITENANT-STORES.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "$(cat <<'EOF'
docs: update session progress

Session: YYYY-MM-DD
- [Brief summary]
- [Test count if changed]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## 8. Key Files for Phase 4

**PRD with detailed specs:**

- `docs/PRD-PHASE4-MULTITENANT-STORES.md` - Success criteria, test specs, feature details, Agent Scratchpad

**Existing (read first):**
- `openmemory/api/app/models.py` - Memory, User, App models
- `openmemory/api/app/database.py` - DB connection + `tenant_session()` context manager
- `openmemory/api/app/routers/memories.py` - Current memory access patterns
- `openmemory/api/tests/security/test_tenant_isolation.py` - Existing tenant tests (19 tests)
- `openmemory/api/app/alembic/utils.py` - MigrationVerifier, BatchMigrator, RollbackManager

**Created in This Phase:**

- `openmemory/api/app/stores/__init__.py` - Store package exports
- `openmemory/api/app/stores/base.py` - BaseStore ABC interface
- `openmemory/api/app/stores/memory_store.py` - ScopedMemoryStore implementation
- `openmemory/api/app/stores/feedback_store.py` - FeedbackStore with retention queries (17 tests)
- `openmemory/api/app/stores/experiment_store.py` - ExperimentStore with status history (18 tests)
- `openmemory/api/conftest.py` - Root conftest for Python path setup
- `openmemory/api/tests/stores/conftest.py` - Shared test fixtures
- `openmemory/api/tests/stores/test_tenant_session.py` - 7 tests
- `openmemory/api/tests/stores/test_memory_store.py` - 16 tests
- `openmemory/api/tests/stores/test_feedback_store.py` - 17 tests
- `openmemory/api/tests/stores/test_experiment_store.py` - 18 tests
- `openmemory/api/tests/integration/test_rls_enforcement.py` - 13 tests (PostgreSQL only)
- `openmemory/api/alembic/versions/add_rls_policies.py` - RLS migration

**To Create (Next - External Stores):**

- `openmemory/api/app/stores/episodic_store.py` - EpisodicMemoryStore (Valkey with TTL)
- `openmemory/api/tests/stores/test_episodic_store.py` - EpisodicStore tests
- Neo4j stores (SymbolStore, Registry, DependencyGraph)
- Qdrant EmbeddingStore with tenant payload index
- OpenSearch tenant alias implementation
