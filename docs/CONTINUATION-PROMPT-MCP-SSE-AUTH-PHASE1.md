# Continuation Prompt: MCP SSE Session Binding - Phase 1

Use this prompt to continue with Phase 1 enhancements after Phase 0 is complete.

---

## Role
You are a senior backend/security engineer continuing MCP SSE authentication hardening with production-ready enhancements.

## Context
Phase 0 is COMPLETE (commit e91e0cf7). All core security fixes are implemented:
- Auth bypass closed in `_check_tool_scope()` - fails closed
- IAT validation added to POST handler
- Memory-based session binding store with 30-minute TTL
- SessionAwareSseTransport captures session_id
- Session binding on GET, validation on POST
- 82 tests passing

## Objective
Implement Phase 1 production enhancements:
1. Valkey-backed session binding store for multi-worker deployments
2. Configurable store selection (`MCP_SESSION_STORE=memory|valkey`)
3. DPoP validation on POST requests
4. Session cleanup background task
5. E2E tests with real SSE connections

## Guardrails
- Keep changes backward compatible with Phase 0
- Memory store remains default for local development
- Valkey store optional, used when configured
- No breaking changes to existing MCP clients
- Preserve all 82 passing tests

## Required Reading (in order)
1. `docs/PRD-MCP-SSE-AUTH-IMPLEMENTATION.md` - Implementation PRD with scratchpad
2. `openmemory/api/app/security/session_binding.py` - Memory store implementation
3. `openmemory/api/app/mcp/sse_transport.py` - Transport wrapper
4. `openmemory/api/app/mcp_server.py` - Handler integration (lines 3067-3100, 3113-3165)
5. `openmemory/api/app/stores/episodic_store.py` - Valkey patterns reference
6. `openmemory/api/app/security/dpop.py` - DPoP validation reference

## Phase 1 Implementation Tasks

### Task 1: Valkey Session Binding Store
**New file**: `openmemory/api/app/security/valkey_session_binding.py`

Implement `ValkeySessionBindingStore` with:
- Same interface as `MemorySessionBindingStore`
- Key format: `mcp:session:{session_id}`
- Use `setex` for automatic TTL expiration
- Serialize/deserialize `SessionBinding` to JSON
- Handle connection errors gracefully (fail closed)

### Task 2: Store Factory
**Modify**: `openmemory/api/app/security/session_binding.py`

Update `get_session_binding_store()` to:
- Check `MCP_SESSION_STORE` env var (default: "memory")
- Return `ValkeySessionBindingStore` when "valkey"
- Return `MemorySessionBindingStore` when "memory"
- Cache singleton per store type

### Task 3: DPoP Validation on POST
**Modify**: `openmemory/api/app/mcp_server.py`

In `handle_post_message()`:
- If principal has `dpop_thumbprint`, require DPoP header
- Validate DPoP proof using existing `DPoPValidator`
- Verify htm="POST" and htu matches request URI
- Return 401 if DPoP required but missing/invalid

### Task 4: Session Cleanup Background Task
**New file**: `openmemory/api/app/tasks/session_cleanup.py`

Implement periodic cleanup:
- Run every 5 minutes (configurable)
- Call `store.cleanup_expired()` for memory store
- For Valkey, TTL handles cleanup automatically
- Log cleanup stats

### Task 5: E2E Tests
**New file**: `openmemory/api/tests/integration/test_mcp_sse_e2e.py`

Write tests with real SSE connections:
- Full flow: GET → capture session_id → POST with session_id
- Session hijacking attempt (different user POST)
- Session expiry handling
- DPoP binding and validation

## Success Criteria Checklist
- [ ] SC-011: Valkey store persists bindings with TTL
- [ ] SC-012: Store selection via env var works
- [ ] SC-013: DPoP validation on POST when bound
- [ ] SC-014: Background cleanup runs without errors
- [ ] SC-015: E2E tests cover full session lifecycle
- [ ] SC-016: Existing 82 tests still pass
- [ ] SC-017: Memory store remains default

## Test Commands
```bash
cd openmemory/api

# Run all Phase 0 + Phase 1 tests
pytest tests/security/ tests/stores/test_session_binding_store.py tests/mcp/ -v

# Run new Valkey store tests
pytest tests/stores/test_valkey_session_binding_store.py -v

# Run E2E tests (requires running server)
pytest tests/integration/test_mcp_sse_e2e.py -v

# Run all tests
pytest -v
```

## Deliverables
- Code changes implementing Phase 1
- Short summary with file references
- Test results (all tests passing)
- Updated scratchpad in PRD

## Output Format
```
## Change Summary
- [file]: [description of change]

## Files Modified/Created
- openmemory/api/app/security/valkey_session_binding.py (new)
- openmemory/api/app/security/session_binding.py (modified)
- openmemory/api/app/mcp_server.py (modified)
- openmemory/api/app/tasks/session_cleanup.py (new)
- openmemory/api/tests/stores/test_valkey_session_binding_store.py (new)
- openmemory/api/tests/integration/test_mcp_sse_e2e.py (new)

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
- **MCP_SESSION_STORE**: `memory` | `valkey`
