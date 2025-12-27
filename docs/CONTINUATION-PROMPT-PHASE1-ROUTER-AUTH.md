# Phase 1 Continuation: MCP Server Authentication

**Purpose**: Continue Phase 1 Security Enforcement Baseline - MCP Server Auth.
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
| Router Auth (all routers) | COMPLETE | tests pending verification |
| MCP Server Auth | NOT STARTED | tests written, impl pending |

**All Routers Converted**: memories.py, apps.py, graph.py, entities.py, stats.py, backup.py

---

## 2. Completed Work Registry - DO NOT REDO

### Security Module (Phase 0b)
- `openmemory/api/app/security/types.py` - Principal, TokenClaims, Scope, errors
- `openmemory/api/app/security/jwt.py` - JWT validation
- `openmemory/api/app/security/dpop.py` - DPoP RFC 9449 with Valkey replay cache
- `openmemory/api/app/security/dependencies.py` - get_current_principal(), require_scopes()
- `openmemory/api/app/security/middleware.py` - Security headers
- `openmemory/api/app/security/exception_handlers.py` - 401/403 formatting

### Router Auth - ALL COMPLETE
- `openmemory/api/app/routers/memories.py` - 15+ endpoints, MEMORIES_READ/WRITE/DELETE
- `openmemory/api/app/routers/apps.py` - 5 endpoints, APPS_READ/WRITE
- `openmemory/api/app/routers/graph.py` - 12 endpoints, GRAPH_READ
- `openmemory/api/app/routers/entities.py` - 14 endpoints, ENTITIES_READ/WRITE
- `openmemory/api/app/routers/stats.py` - 1 endpoint, STATS_READ
- `openmemory/api/app/routers/backup.py` - 2 endpoints, BACKUP_READ/WRITE

---

## 3. Next Task: MCP Server Authentication

### STEP 1: Run Router Auth Tests (TDD Verification)

**Use a subagent** to verify router tests:

```
Use Task tool with subagent_type=Explore to:
1. Read openmemory/api/tests/security/test_router_auth.py
2. Understand what tests are checking
3. Report any potential issues
```

Then run tests:
```bash
cd /Users/grischadallmer/git/coding-brain/openmemory
docker compose exec api pytest tests/security/test_router_auth.py -v --tb=short
```

**DO NOT proceed to Step 2 until router tests pass or are fixed.**

---

### STEP 2: Understand MCP Server Structure

**Use a subagent** to explore:

```
Use Task tool with subagent_type=Explore to:
1. Read openmemory/api/app/mcp_server.py
2. Identify SSE endpoints that accept user_id params
3. Identify all tool handlers
4. Report the authentication surface
```

---

### STEP 3: Run MCP Tests (TDD Red Phase)

```bash
docker compose exec api pytest tests/security/test_mcp_auth.py -v --tb=short
```

**Confirm tests fail.** This is the RED phase.

---

### STEP 4: Implement MCP SSE Auth

Modify `openmemory/api/app/mcp_server.py`:

1. Add imports:
```python
from app.security.jwt import validate_jwt
from app.security.types import Principal, AuthenticationError, Scope
```

2. SSE endpoint changes:
   - Remove `user_id` from URL path
   - Extract JWT from Authorization header
   - Validate and create Principal
   - Set context vars from Principal
   - Return 401 if no valid JWT

---

### STEP 5: Implement Tool Scope Checks

For each MCP tool:
- `add_memories`: Check MEMORIES_WRITE
- `search_memories`: Check MEMORIES_READ
- `delete_memory`: Check MEMORIES_DELETE
- `delete_all_memories`: Check MEMORIES_DELETE
- `update_memory`: Check MEMORIES_WRITE
- `list_memories`: Check MEMORIES_READ

---

### STEP 6: Run MCP Tests (TDD Green Phase)

```bash
docker compose exec api pytest tests/security/test_mcp_auth.py -v
```

**All tests should pass.**

---

### STEP 7: Run All Security Tests

```bash
docker compose exec api pytest tests/security/ -v
```

---

### STEP 8: Update Progress Tracker

Add to `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`:

```markdown
| 2025-12-XX | Phase 1 complete | MCP server SSE endpoints require JWT; all tools check scopes; test_router_auth.py and test_mcp_auth.py all passing |
```

---

### STEP 9: Commit and Write Next Continuation Prompt

Commit with message:
```
feat(security): add JWT auth to MCP server SSE and tool scope checks
```

Write next continuation prompt for Phase 2 (Configuration and Secrets).

---

## 4. TDD Workflow - MANDATORY

1. **RED**: Run tests, confirm they fail
2. **GREEN**: Write minimal code to pass tests
3. **REFACTOR**: Clean up while keeping tests green

**NEVER skip the RED phase.** If tests already pass, verify the feature works.

---

## 5. Subagent Usage - RECOMMENDED

Use subagents liberally:

```
# Explore structure
Use Task tool with subagent_type=Explore to understand mcp_server.py

# Read test file
Use Task tool with subagent_type=Explore to analyze test_mcp_auth.py

# Run tests in background
Use Bash with run_in_background=true for long test runs
```

---

## 6. Exit Gates for Phase 1

| Metric | Threshold |
|--------|-----------|
| test_router_auth.py | All tests pass |
| test_mcp_auth.py | All tests pass |
| No user_id params | 0 remaining anywhere |
| MCP SSE auth | JWT required |
| MCP tool scopes | All tools check scopes |

---

## 7. Command Reference

```bash
# Router auth tests
docker compose exec api pytest tests/security/test_router_auth.py -v

# MCP auth tests
docker compose exec api pytest tests/security/test_mcp_auth.py -v

# All security tests
docker compose exec api pytest tests/security/ -v

# Syntax check
docker compose exec api python -c "from app.mcp_server import mcp"
```
