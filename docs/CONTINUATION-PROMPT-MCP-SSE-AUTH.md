# Continuation Prompt: MCP SSE Session-Bound Authentication

Use this prompt to implement the MCP SSE session binding security fixes.

---

## Role
You are a senior backend/security engineer implementing MCP SSE authentication hardening to close critical security gaps.

## Objective
Implement Phase 0 (demo-ready) security fixes from `docs/PRD-MCP-SSE-SESSION-BINDING.md`:
1. Close the auth bypass in `_check_tool_scope`
2. Bind SSE `session_id` to authenticated principal
3. Validate session binding on every POST

## Guardrails
- Keep changes minimal and low-risk.
- No protocol changes (no JWT in body, no MCP extension, no WebSocket migration).
- Do NOT touch `openmemory/api/app/axis_guidance_server.py` (out of scope).
- Preserve backward compatibility for clients sending `Authorization` header on GET and POST.
- Avoid logging sensitive data (Authorization headers, session_id, request bodies).

## Required Reading (in order)
1. `docs/PRD-MCP-SSE-SESSION-BINDING.md` — Full PRD with success criteria, test specs, implementation details
2. `docs/PRD-MCP-SSE-AUTH-PLAN-V2.md` — Original security analysis
3. `docs/ADR-MCP-SSE-AUTH.md` — Architecture decision record
4. `openmemory/api/app/mcp_server.py` — SSE handlers (lines 3037-3137), `_check_tool_scope` (lines 215-237)
5. `openmemory/api/app/security/jwt.py` — `validate_jwt`, `validate_iat_not_future`
6. `openmemory/api/app/security/dependencies.py` — `get_current_principal` pattern
7. `openmemory/api/app/security/types.py` — `Principal`, `TokenClaims`, `Scope`
8. `openmemory/api/app/stores/episodic_store.py` — TTL and key patterns for reference
9. `.venv/lib/python3.14/site-packages/mcp/server/sse.py` — MCP SSE transport (`connect_sse`, `session_id` generation)

## Phase 0 Implementation Tasks

### Task 1: Remove Auth Bypass (CRITICAL)
**File**: `openmemory/api/app/mcp_server.py` lines 227-228

```python
# BEFORE
principal = principal_var.get(None)
if not principal:
    return None  # Backwards compatibility - no auth check

# AFTER
principal = principal_var.get(None)
if not principal:
    return json.dumps({
        "error": "Authentication required",
        "code": "MISSING_AUTH",
    })
```

### Task 2: Create Session Binding Store
**New file**: `openmemory/api/app/security/session_binding.py`

Implement `MemorySessionBindingStore` with:
- `create(session_id, user_id, org_id, dpop_thumbprint?, ttl?)` → `SessionBinding`
- `get(session_id)` → `SessionBinding | None` (returns None if expired)
- `validate(session_id, user_id, org_id, dpop_thumbprint?)` → `bool`
- `delete(session_id)` → `bool`
- Thread-safe with `threading.Lock`
- Default TTL: 30 minutes (configurable via `MCP_SESSION_TTL_SECONDS` env)

### Task 3: Create SSE Transport Wrapper
**New file**: `openmemory/api/app/mcp/sse_transport.py`

Implement `SessionAwareSseTransport(SseServerTransport)` that:
- Captures `session_id` from `_read_stream_writers` after `connect_sse()`
- Exposes `current_session_id` property
- Passes through all other functionality unchanged

### Task 4: Add IAT Validation to POST Handler
**File**: `openmemory/api/app/mcp_server.py` in `handle_post_message()`

```python
from app.security.jwt import validate_iat_not_future

principal = _extract_principal_from_request(request)
validate_iat_not_future(principal.claims.iat)  # Raises AuthenticationError
```

### Task 5: Integrate Session Binding
**File**: `openmemory/api/app/mcp_server.py`

**In GET handler** (`handle_sse`):
```python
async with sse.connect_sse(...) as (read_stream, write_stream):
    session_id = sse.current_session_id
    if session_id:
        get_session_binding_store().create(
            session_id=session_id,
            user_id=principal.user_id,
            org_id=principal.org_id,
        )
    # ... existing code ...
finally:
    if session_id:
        get_session_binding_store().delete(session_id)
```

**In POST handler** (`handle_post_message`):
```python
# Extract session_id from query params
session_id_str = request.query_params.get("session_id")
if not session_id_str:
    return JSONResponse(status_code=400, content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"})

try:
    session_id = UUID(session_id_str)
except ValueError:
    return JSONResponse(status_code=400, content={"error": "Invalid session_id", "code": "INVALID_SESSION_ID"})

# Validate binding
if not get_session_binding_store().validate(session_id, principal.user_id, principal.org_id):
    return JSONResponse(status_code=403, content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"})
```

## Phase 1 Tasks (if time allows)
1. Add Valkey-backed session binding store with TTL
2. Add config: `MCP_SESSION_STORE=memory|valkey`
3. Write tests in `tests/security/test_mcp_session_binding.py`
4. Write tests in `tests/stores/test_session_binding_store.py`

## Success Criteria Checklist
- [ ] SC-001: Missing Authorization on POST → 401
- [ ] SC-002: Mismatched JWT vs session → 403
- [ ] SC-003: Tool call without principal → error (no execution)
- [ ] SC-004: Session binding created on GET with TTL
- [ ] SC-005: Session binding validated on POST
- [ ] SC-006: Local Docker works with memory store
- [ ] SC-007: Future iat tokens rejected (>30s skew)
- [ ] SC-008: session_id captured from SSE transport
- [ ] SC-009: Existing clients continue working
- [ ] SC-010: No sensitive data logged

## Test Commands
```bash
cd openmemory/api

# Run existing security tests (regression)
pytest tests/security/ -v

# Run new session binding tests (after implementation)
pytest tests/security/test_mcp_session_binding.py -v
pytest tests/stores/test_session_binding_store.py -v

# Run all tests
pytest -v
```

## Deliverables
- Code changes implementing Phase 0
- Short summary with file references
- Test results (or manual verification steps)
- Suggested next steps

## Output Format
```
## Change Summary
- [file]: [description of change]

## Files Modified/Created
- openmemory/api/app/mcp_server.py (modified)
- openmemory/api/app/security/session_binding.py (new)
- openmemory/api/app/mcp/sse_transport.py (new)

## Tests Run
[paste test output]

## Next Steps
1. ...
2. ...
```

---

## Template Variables
- **REPO_ROOT**: `/Users/grischadallmer/git/coding-brain`
- **TARGET_BRANCH**: `coding-brain`
- **ENV_MODE**: `local` (memory store) | `company` (Valkey store)
- **MCP_SESSION_TTL_SECONDS**: `1800` (30 minutes default)
