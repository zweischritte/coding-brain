# Phase 3 Continuation: PostgreSQL Migration

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
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
git add docs/CONTINUATION-PROMPT-PHASE3-POSTGRES-MIGRATION.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
git commit -m "docs: update session progress and implementation tracker"
```

---

## 1. Current Gaps (per Implementation Plan)

**Phase 1 Gap**: MCP auth incomplete (SSE auth + tool scopes pending)
**Phase 3**: ✅ COMPLETE (tenant isolation + migration verification utilities)

---

## 2. Command Reference

```bash
# Security tests
docker compose exec codingbrain-mcp pytest tests/security/ -v

# Specific test file
docker compose exec codingbrain-mcp pytest tests/security/test_tenant_isolation.py -v

# Run migrations (when ready)
docker compose exec codingbrain-mcp alembic upgrade head

# Check migration status
docker compose exec codingbrain-mcp alembic current
```

---

## 3. Architecture Patterns

### Tenant Isolation Pattern

All routers must follow this pattern:

```python
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope

def get_user_from_principal(db: Session, principal: Principal) -> User:
    user = db.query(User).filter(User.user_id == principal.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/")
async def list_items(
    principal: Principal = Depends(require_scopes(Scope.ITEMS_READ)),
    db: Session = Depends(get_db)
):
    user = get_user_from_principal(db, principal)
    query = db.query(Item).filter(Item.user_id == user.id)  # ALWAYS filter
    ...
```

### Cross-Tenant Access Behavior

- Return **empty results** (not 403) when data doesn't exist for user
- Return **404** when specific item doesn't exist or user doesn't own it
- Never return 403 as it leaks information about data existence

---

## 4. Next Tasks

**See Progress file task tables for detailed status. Priorities below.**

### Priority 1: Phase 1 Gap - MCP Auth (Blocked)

See Progress file Phase 1 table - "MCP tool permission checks" is blocked pending SSE auth architecture.

**When unblocked:**

1. Add auth to MCP SSE endpoints
2. Add tool-level permission checks in MCP server
3. Run test_mcp_auth.py to verify

### Priority 2: Phase 3 - ✅ COMPLETE

All Alembic migration tasks completed:

- ✅ Migration scaffolding (pre-existing - 4 migrations)
- ✅ Migration verification utilities (MigrationVerifier, BackupValidator)
- ✅ Data migration tooling (BatchMigrator with progress callbacks)
- ✅ Rollback procedures (RollbackManager with savepoints)
- ✅ env.py hooks for pre/post migration verification (ALEMBIC_VERIFY_MIGRATIONS=true)
- ✅ 33 TDD tests in test_migration_verification.py

### Priority 3: Phase 4 - Multi-tenant Data Plane Stores (Next)

See Progress file Phase 4 table - all tasks Not Started.

### Technical Debt (Lower Priority)

- Extract shared `get_user_from_principal()` helper to reduce duplication
- Address pre-existing test failures in test_router_auth.py (see Blocking Issues Log)
- Fix Pydantic V1 @validator deprecation warning in `app/schemas.py:54`

---

## 5. Known Issues

1. **Pre-existing test failures in test_router_auth.py:**
   - `test_delete_memory_requires_auth` - Tests wrong endpoint signature
   - `test_create_app_requires_auth` - No POST /apps endpoint exists
   - `test_export_requires_auth` - Needs investigation
   - `test_memories_delete_requires_scope` - Tests wrong endpoint signature

2. **MCP auth tests hang** - test_mcp_auth.py tests take too long/hang

3. **Pydantic deprecation** - `app/schemas.py:54` uses V1 @validator

---

## 6. Last Session Summary (2025-12-27)

**Completed**: Phase 3c - Migration Verification Utilities

- Created `app/alembic/utils.py` with:
  - `MigrationVerifier` - Row count and checksum verification
  - `BackupValidator` - Pre-migration backup validation
  - `BatchMigrator` - Large data migration in batches
  - `RollbackManager` - Safe rollback with savepoints
- Updated `alembic/env.py` with pre/post migration hooks
- Added 33 TDD tests in `tests/infrastructure/test_migration_verification.py`

**Result**: 3,079 total tests (33 new migration verification tests)

---

## 7. Commit Template

```bash
git add docs/CONTINUATION-PROMPT-PHASE3-POSTGRES-MIGRATION.md docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md
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
