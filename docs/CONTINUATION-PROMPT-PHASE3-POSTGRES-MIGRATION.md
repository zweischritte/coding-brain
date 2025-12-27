# Phase 3 Continuation: PostgreSQL Migration

**Plan**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## SESSION WORKFLOW

### At Session Start

1. Read `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md` for overall plan
1. Read `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` for current progress and daily log
1. Check Section 4 (Next Tasks) below for what to work on
1. Continue from where the last session left off

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
**Phase 3 Gap**: Alembic migrations NOT started (plan requires migration scaffolding, verification, rollback)

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

### Phase 1 Gap - MCP Auth

- [ ] Add auth to MCP SSE endpoints
- [ ] Add tool-level permission checks in MCP server
- [ ] Run test_mcp_auth.py to verify

### Phase 3 Gap - Alembic Migrations

- [ ] Set up Alembic migration scaffolding
- [ ] Create initial migration from current models
- [ ] Document pre-migration backup procedure
- [ ] Add row count/checksum verification
- [ ] Document rollback procedure
- [ ] Test migration on fresh PostgreSQL

### Technical Debt

- [ ] Extract shared `get_user_from_principal()` helper to reduce duplication
- [ ] Address pre-existing test failures in test_router_auth.py
- [ ] Fix Pydantic V1 @validator deprecation warning in `app/schemas.py:54`

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

**Completed**: Router tenant isolation audit

- graph.py (12 endpoints) - ✅ SECURE
- entities.py (12 endpoints) - ✅ SECURE
- stats.py (1 endpoint) - 🔧 FIXED `App.owner == user` → `App.owner_id == user.id`
- backup.py (2 endpoints) - ✅ SECURE

**Result**: 131 security tests passing, 19 tenant isolation tests

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
