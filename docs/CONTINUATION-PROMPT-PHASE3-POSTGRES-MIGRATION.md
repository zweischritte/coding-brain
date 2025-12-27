# Phase 3 Continuation: PostgreSQL Migration

**Purpose**: Continue Phase 3 - PostgreSQL Migration for multi-user support.
**Usage**: Paste this entire prompt to resume implementation exactly where interrupted.
**Development Style**: STRICT TDD - Write failing tests first, then implement. Use subagents for exploration.

---

## 1. Current State Summary

| Component | Status | Tests |
|-----------|--------|-------|
| Security Core Types | COMPLETE | 32 |
| JWT Validation | COMPLETE | 24 |
| DPoP RFC 9449 | COMPLETE | 16 |
| Security Headers | COMPLETE | 27 |
| Router Auth (all routers) | COMPLETE | 25 |
| MCP Server Auth | COMPLETE | 13+ |
| Pydantic Settings | COMPLETE | 13 |
| **Tenant Isolation Tests** | **COMPLETE** | **16** |
| **Apps Router Tenant Fix** | **COMPLETE** | - |
| **Database PostgreSQL Support** | **COMPLETE** | - |

**Phase 3a Progress**: 128+ security tests passing. Apps router now filters by owner_id. Database.py supports PostgreSQL with connection pooling.

---

## 2. Completed Work Registry - DO NOT REDO

### Phase 0b: Security Module
- `openmemory/api/app/security/types.py` - Principal, TokenClaims, Scope, errors
- `openmemory/api/app/security/jwt.py` - JWT validation (now uses Settings)
- `openmemory/api/app/security/dpop.py` - DPoP RFC 9449 with Valkey replay cache
- `openmemory/api/app/security/dependencies.py` - get_current_principal(), require_scopes()
- `openmemory/api/app/security/middleware.py` - Security headers
- `openmemory/api/app/security/exception_handlers.py` - 401/403 formatting

### Phase 1: Router Auth - ALL COMPLETE
- All routers require JWT authentication
- All MCP tools check scopes
- User ID comes from JWT token, not URL parameters

### Phase 2: Configuration and Secrets - COMPLETE
- `openmemory/api/app/settings/settings.py` - Pydantic Settings with validation
- `openmemory/api/app/settings/__init__.py` - Module exports
- `openmemory/api/main.py` - Lifespan handler for startup validation
- `openmemory/.env.example` - Complete configuration template
- `docs/SECRET-ROTATION.md` - Rotation procedures

### Phase 3a: Tenant Isolation (2025-12-27) - COMPLETE
- `openmemory/api/tests/security/test_tenant_isolation.py` - 16 tests for multi-tenant isolation
- `openmemory/api/app/routers/apps.py` - Fixed: all endpoints now filter by owner_id
  - Added `get_user_from_principal()` helper
  - Added `get_app_or_404_for_user()` helper with ownership check
  - `list_apps()` - filters by `App.owner_id == user.id`
  - `get_app_details()` - verifies ownership before returning
  - `list_app_memories()` - verifies app ownership + filters by user_id
  - `list_app_accessed_memories()` - verifies app ownership + filters by user_id
  - `update_app_details()` - verifies ownership before updating
- `openmemory/api/app/database.py` - Updated for PostgreSQL support
  - Uses Settings.database_url (PostgreSQL from Pydantic Settings)
  - PostgreSQL connection pooling with QueuePool
  - SQLite fallback only for development
- `openmemory/api/app/routers/memories.py` - Fixed SQLite-specific json_extract
  - Replaced `func.json_extract(Memory.metadata_, '$.vault')` with `Memory.vault`
  - Replaced `func.json_extract(Memory.metadata_, '$.layer')` with `Memory.layer`
  - Uses indexed columns for database-agnostic filtering

### Infrastructure
- `openmemory/docker-compose.yml` - Project name `codingbrain`, all containers prefixed
- `openmemory/.env` - Complete local dev environment
- `openmemory/api/requirements.txt` - Includes python-jose[cryptography], pydantic-settings

---

## 3. Next Task: Verify Remaining Routers (Phase 3b)

### Goal
Verify all remaining routers properly filter by user_id from JWT principal.

### Routers to Audit

| Router | File | Status |
|--------|------|--------|
| memories | `app/routers/memories.py` | ✅ Uses principal.user_id |
| apps | `app/routers/apps.py` | ✅ Fixed - uses owner_id |
| graph | `app/routers/graph.py` | ⚠️ NEEDS AUDIT |
| entities | `app/routers/entities.py` | ⚠️ NEEDS AUDIT |
| stats | `app/routers/stats.py` | ⚠️ NEEDS AUDIT |
| backup | `app/routers/backup.py` | ⚠️ NEEDS AUDIT |

### STEP 1: Audit Graph Router

**Use a subagent** to explore:

```
Use Task tool with subagent_type=Explore to:
1. Read openmemory/api/app/routers/graph.py
2. Check if queries filter by principal.user_id
3. Identify any endpoints that might leak cross-tenant data
```

### STEP 2: Write Tests for Graph Router Tenant Isolation

Add tests to `test_tenant_isolation.py`:

```python
class TestGraphTenantIsolation:
    """Tests for tenant isolation in the graph router."""

    def test_graph_stats_filtered_by_user(self, client, mock_jwt_config, user_a_headers):
        """Graph stats should only include the user's data."""
        ...

    def test_cannot_access_other_user_graph_entities(self, client, mock_jwt_config, user_a_headers):
        """User cannot see entities from other users' memories."""
        ...
```

### STEP 3: Fix Graph Router (if needed)

Apply the same pattern as apps.py:
1. Get user from principal using `get_user_from_principal()`
2. Filter all queries by `user_id`
3. Verify ownership before returning data

### STEP 4: Repeat for entities, stats, backup routers

---

## 4. TDD Workflow - MANDATORY

1. **RED**: Run tests, confirm they fail
2. **GREEN**: Write minimal code to pass tests
3. **REFACTOR**: Clean up while keeping tests green

**NEVER skip the RED phase.**

---

## 5. Exit Gates for Phase 3

| Metric | Threshold |
|--------|-----------|
| test_tenant_isolation.py | All tests pass |
| All routers audited | Filter by user_id |
| Cross-tenant queries | Return empty, not 403 |
| All security tests | 128+ passing |

---

## 6. Command Reference

```bash
# Tenant isolation tests
docker compose exec codingbrain-mcp pytest tests/security/test_tenant_isolation.py -v

# All security tests (subset that runs quickly)
docker compose exec codingbrain-mcp pytest tests/security/test_tenant_isolation.py tests/security/test_types.py tests/security/test_jwt_validation.py tests/security/test_config.py tests/security/test_dpop.py tests/security/test_security_headers.py -v

# Check a specific router
docker compose exec codingbrain-mcp python -c "from app.routers.graph import router; print('OK')"

# Run migrations (when ready for PostgreSQL)
docker compose exec codingbrain-mcp alembic upgrade head

# Check migration status
docker compose exec codingbrain-mcp alembic current
```

---

## 7. Phase 3a Completion Summary (2025-12-27)

**What was implemented:**
- Tenant isolation tests (16 tests in test_tenant_isolation.py)
- Apps router security fix (all endpoints filter by owner_id)
- Database.py PostgreSQL support (connection pooling, Settings integration)
- Memories router fix (replaced SQLite json_extract with indexed columns)

**Security fixes applied:**
- `list_apps()` - Now filters by `App.owner_id == user.id`
- `get_app_details()` - Now verifies ownership
- `list_app_memories()` - Now verifies app ownership + user_id filter
- `list_app_accessed_memories()` - Now verifies app ownership + user_id filter
- `update_app_details()` - Now verifies ownership before update

**Files created/modified:**
- `openmemory/api/tests/security/test_tenant_isolation.py` - NEW (16 tests)
- `openmemory/api/app/routers/apps.py` - MODIFIED (tenant isolation)
- `openmemory/api/app/database.py` - MODIFIED (PostgreSQL support)
- `openmemory/api/app/routers/memories.py` - MODIFIED (json_extract fix)

**Test count:** 128+ security tests passing

---

## 8. Known Issues / Technical Debt

1. **Pre-existing test failures in test_router_auth.py:**
   - `test_delete_memory_requires_auth` - Tests wrong endpoint signature
   - `test_create_app_requires_auth` - No POST /apps endpoint exists
   - `test_export_requires_auth` - Needs investigation
   - `test_memories_delete_requires_scope` - Tests wrong endpoint signature

2. **MCP auth tests hang** - The test_mcp_auth.py tests take too long/hang

3. **Pydantic deprecation warning** - `app/schemas.py:54` uses V1 @validator

---

## 9. Architecture Decisions

### Tenant Isolation Pattern

All routers should follow this pattern:

```python
from app.security.dependencies import require_scopes
from app.security.types import Principal, Scope

def get_user_from_principal(db: Session, principal: Principal) -> User:
    """Get the User record for the authenticated principal."""
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

    # ALWAYS filter by user.id
    query = db.query(Item).filter(Item.user_id == user.id)
    ...
```

### Cross-Tenant Access Behavior

- Return **empty results** (not 403) when data doesn't exist for user
- Return **404** when specific item doesn't exist or user doesn't own it
- Never return 403 as it leaks information about data existence

---

## 10. Next Session Checklist

- [ ] Audit graph.py router for tenant isolation
- [ ] Audit entities.py router for tenant isolation
- [ ] Audit stats.py router for tenant isolation
- [ ] Audit backup.py router for tenant isolation
- [ ] Add tests for any found issues
- [ ] Fix any routers missing tenant filtering
- [ ] Verify PostgreSQL Alembic migrations are current
- [ ] Run full security test suite
