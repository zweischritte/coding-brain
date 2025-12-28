# PRD: MCP SSE Session-Bound Authentication

**Status**: Ready for Implementation
**Owner**: Security Architecture
**Phase**: 0 (Demo-Ready)
**Goal**: Bind MCP SSE session_id to authenticated principal, close auth bypass gaps

---

## Executive Summary

This PRD addresses critical security vulnerabilities in the MCP SSE authentication flow:
1. **Auth bypass** in `_check_tool_scope()` when `principal_var` is unset
2. **Session hijacking** via unbound session_id
3. **Missing iat validation** in MCP SSE handlers

Phase 0 implements the minimal fixes required for a secure, demo-ready system.

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Define Success Criteria

All measurable acceptance criteria for Phase 0:

1. [x] **SC-001**: Missing Authorization header on POST returns 401 Unauthorized
2. [x] **SC-002**: POST with mismatched JWT (different user_id) vs session binding returns 403 Forbidden
3. [x] **SC-003**: Tool calls without `principal_var` set return auth error (no tool execution)
4. [x] **SC-004**: Session binding store creates binding on GET with TTL (default 30 min)
5. [x] **SC-005**: Session binding store validates binding on POST before tool execution
6. [x] **SC-006**: Local Docker works with in-memory session binding store
7. [x] **SC-007**: Future-dated iat tokens are rejected (clock skew > 30 seconds)
8. [x] **SC-008**: Session_id is captured from SSE transport on GET
9. [x] **SC-009**: Existing MCP clients continue working (Authorization header on GET/POST)
10. [x] **SC-010**: No sensitive data logged (Authorization, session_id, request bodies)

### 1.2 Define Edge Cases

Edge cases that must be handled:

1. **Clock skew**: Tokens with iat up to 30 seconds in the future are accepted
2. **Session expiry**: POST to expired session returns 403 (not 401)
3. **Concurrent POSTs**: Multiple simultaneous POSTs to same session all succeed if bound
4. **Session cleanup**: SSE disconnect triggers binding cleanup (Phase 1)
5. **Store unavailable**: If session store fails, reject request (fail closed)
6. **DPoP mismatch**: If DPoP thumbprint was bound, POST must have matching DPoP (Phase 1)
7. **Missing session_id**: POST without session_id query param returns 400
8. **Invalid session_id format**: Non-UUID session_id returns 400
9. **HAS_SECURITY=False**: In production, reject MCP requests; in dev, allow with env flag

### 1.3 Design Test Suite Structure

```
openmemory/api/tests/
├── security/
│   ├── test_mcp_session_binding.py      # NEW: Session binding tests
│   ├── test_mcp_auth_bypass.py          # NEW: Auth bypass regression tests
│   ├── test_mcp_iat_validation.py       # NEW: IAT validation tests
│   └── test_mcp_auth.py                 # EXISTING: Extended for new flows
├── stores/
│   └── test_session_binding_store.py    # NEW: Binding store unit tests
└── integration/
    └── test_mcp_sse_e2e.py              # NEW: Full SSE flow integration
```

### 1.4 Write Test Specifications

| Test ID | Feature | Test Type | Test Description | Expected Outcome |
|---------|---------|-----------|------------------|------------------|
| SB-001 | Session Binding | Unit | Create binding with valid principal | Binding stored with TTL |
| SB-002 | Session Binding | Unit | Validate matching principal | Returns True |
| SB-003 | Session Binding | Unit | Validate mismatched user_id | Returns False |
| SB-004 | Session Binding | Unit | Validate mismatched org_id | Returns False |
| SB-005 | Session Binding | Unit | Get expired binding | Returns None |
| SB-006 | Session Binding | Unit | Delete binding | Binding removed |
| AB-001 | Auth Bypass | Unit | _check_tool_scope with no principal | Returns error JSON |
| AB-002 | Auth Bypass | Unit | _check_tool_scope with valid scope | Returns None |
| AB-003 | Auth Bypass | Unit | _check_tool_scope with missing scope | Returns error JSON |
| AB-004 | Auth Bypass | Integration | Tool call without JWT | Returns 401 |
| AB-005 | Auth Bypass | Integration | Tool call with invalid JWT | Returns 401 |
| IAT-001 | IAT Validation | Unit | Token iat = now - 5 seconds | Accepted |
| IAT-002 | IAT Validation | Unit | Token iat = now + 29 seconds | Accepted (within skew) |
| IAT-003 | IAT Validation | Unit | Token iat = now + 60 seconds | Rejected |
| SSE-001 | SSE Transport | Unit | connect_sse exposes session_id | session_id captured |
| SSE-002 | SSE Transport | Integration | GET creates binding | Binding exists in store |
| SSE-003 | SSE Transport | Integration | POST validates binding | Request accepted |
| SSE-004 | SSE Transport | Integration | POST with wrong user | 403 Forbidden |
| SSE-005 | SSE Transport | Integration | POST with expired session | 403 Forbidden |
| SSE-006 | SSE Transport | Integration | POST without session_id | 400 Bad Request |
| E2E-001 | Full Flow | E2E | Connect SSE, send tool call | Tool executes |
| E2E-002 | Full Flow | E2E | Tool requires scope, user lacks scope | Tool rejected |
| E2E-003 | Full Flow | E2E | Cross-tenant tool call | 403 Forbidden |

---

## PHASE 2: Feature Specifications

### Feature 1: Remove Auth Bypass in `_check_tool_scope`

**Description**: Change `_check_tool_scope()` to fail closed when `principal_var` is not set.

**Dependencies**: None

**Test Cases**:
- [x] Unit: `_check_tool_scope()` returns error when principal is None
- [x] Unit: `_check_tool_scope()` returns None when principal has scope
- [x] Integration: MCP tool without auth returns error

**Implementation Approach**:
```python
# BEFORE (mcp_server.py:227-228)
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

**Git Commit Message**: `fix(mcp): fail closed when principal_var missing in tool scope check`

---

### Feature 2: Add IAT Validation to MCP SSE Handlers

**Description**: Call `validate_iat_not_future()` in MCP SSE POST handler to reject future-dated tokens.

**Dependencies**: `security/jwt.py:validate_iat_not_future`

**Test Cases**:
- [x] Unit: Token with past iat accepted
- [x] Unit: Token with future iat (within 30s) accepted
- [x] Unit: Token with future iat (>30s) rejected
- [x] Integration: POST with future-dated token returns 401

**Implementation Approach**:
```python
# In handle_post_message() after extracting principal
from app.security.jwt import validate_iat_not_future

principal = _extract_principal_from_request(request)
validate_iat_not_future(principal.claims.iat)  # Raises AuthenticationError if invalid
```

**Git Commit Message**: `feat(mcp): add iat validation to SSE POST handler`

---

### Feature 3: Session Binding Store

**Description**: Implement memory-based session binding store with TTL support.

**Dependencies**: None (Valkey support in Phase 1)

**Test Cases**:
- [x] Unit: Create binding stores all fields
- [x] Unit: Get binding returns correct data
- [x] Unit: Validate binding matches principal
- [x] Unit: Expired binding returns None
- [x] Unit: Delete binding removes entry
- [x] Integration: Binding survives concurrent access

**Implementation Approach**:

New file: `openmemory/api/app/security/session_binding.py`

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import UUID
import threading

@dataclass
class SessionBinding:
    """Binding between SSE session and authenticated principal."""
    session_id: UUID
    user_id: str
    org_id: str
    issued_at: datetime
    expires_at: datetime
    dpop_thumbprint: Optional[str] = None

class MemorySessionBindingStore:
    """Thread-safe in-memory session binding store with TTL."""

    def __init__(self, default_ttl_seconds: int = 1800):
        self._bindings: Dict[UUID, SessionBinding] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def create(self, session_id: UUID, user_id: str, org_id: str,
               dpop_thumbprint: Optional[str] = None,
               ttl_seconds: Optional[int] = None) -> SessionBinding:
        """Create a new session binding."""
        ttl = ttl_seconds or self._default_ttl
        now = datetime.now(timezone.utc)
        binding = SessionBinding(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl),
            dpop_thumbprint=dpop_thumbprint,
        )
        with self._lock:
            self._bindings[session_id] = binding
        return binding

    def get(self, session_id: UUID) -> Optional[SessionBinding]:
        """Get binding if exists and not expired."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding and binding.expires_at > datetime.now(timezone.utc):
                return binding
            elif binding:
                del self._bindings[session_id]  # Cleanup expired
            return None

    def validate(self, session_id: UUID, user_id: str, org_id: str,
                 dpop_thumbprint: Optional[str] = None) -> bool:
        """Validate that session binding matches principal."""
        binding = self.get(session_id)
        if not binding:
            return False
        if binding.user_id != user_id or binding.org_id != org_id:
            return False
        if binding.dpop_thumbprint and dpop_thumbprint != binding.dpop_thumbprint:
            return False
        return True

    def delete(self, session_id: UUID) -> bool:
        """Delete a session binding."""
        with self._lock:
            if session_id in self._bindings:
                del self._bindings[session_id]
                return True
            return False
```

**Git Commit Message**: `feat(security): add memory-based session binding store with TTL`

---

### Feature 4: SSE Transport Wrapper for Session ID Capture

**Description**: Create a minimal wrapper around `SseServerTransport` that captures the generated `session_id` during `connect_sse()`.

**Dependencies**: `mcp.server.sse.SseServerTransport`

**Test Cases**:
- [x] Unit: Wrapper captures session_id on connect
- [x] Unit: Wrapper passes through all other functionality
- [x] Integration: session_id available after connect_sse()

**Implementation Approach**:

New file: `openmemory/api/app/mcp/sse_transport.py`

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Tuple, Optional
from uuid import UUID
from mcp.server.sse import SseServerTransport
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

class SessionAwareSseTransport(SseServerTransport):
    """SSE transport wrapper that exposes session_id for binding."""

    def __init__(self, endpoint: str):
        super().__init__(endpoint)
        self._current_session_id: Optional[UUID] = None

    @property
    def current_session_id(self) -> Optional[UUID]:
        """Get the session_id from the most recent connect_sse call."""
        return self._current_session_id

    @asynccontextmanager
    async def connect_sse(
        self, scope, receive, send
    ) -> AsyncGenerator[Tuple[MemoryObjectReceiveStream, MemoryObjectSendStream], None]:
        """Wrap connect_sse to capture session_id before yielding."""

        # Call parent's connect_sse
        async with super().connect_sse(scope, receive, send) as streams:
            # The session_id is stored in _read_stream_writers dict
            # Find the most recently added session_id
            if self._read_stream_writers:
                self._current_session_id = list(self._read_stream_writers.keys())[-1]

            yield streams

            # Clear after connection ends
            self._current_session_id = None
```

**Alternative approach** (if accessing _read_stream_writers is fragile):

```python
import uuid
from uuid import UUID

class SessionAwareSseTransport(SseServerTransport):
    """SSE transport with session_id capture via patching."""

    def __init__(self, endpoint: str):
        super().__init__(endpoint)
        self._captured_session_ids: dict = {}  # scope_id -> session_id

    @asynccontextmanager
    async def connect_sse(self, scope, receive, send):
        # Generate session_id ourselves and inject it
        # This requires understanding the parent implementation

        # Simpler: Store session_id in request scope for retrieval
        scope_id = id(scope)

        async with super().connect_sse(scope, receive, send) as streams:
            # After connection, find session from internal state
            # The library stores it in _read_stream_writers
            for sid in self._read_stream_writers:
                if sid not in self._captured_session_ids.values():
                    self._captured_session_ids[scope_id] = sid
                    break

            yield streams

            # Cleanup
            self._captured_session_ids.pop(scope_id, None)

    def get_session_id(self, scope) -> Optional[UUID]:
        return self._captured_session_ids.get(id(scope))
```

**Git Commit Message**: `feat(mcp): add SSE transport wrapper to capture session_id`

---

### Feature 5: Bind Session on GET, Validate on POST

**Description**: Integrate session binding into MCP SSE handlers.

**Dependencies**: Features 3, 4

**Test Cases**:
- [x] Integration: GET creates session binding
- [x] Integration: POST with valid binding succeeds
- [x] Integration: POST with mismatched binding returns 403
- [x] Integration: POST with expired binding returns 403
- [x] Integration: POST without session_id returns 400

**Implementation Approach**:

Modify `openmemory/api/app/mcp_server.py`:

```python
from app.security.session_binding import MemorySessionBindingStore, get_session_binding_store
from app.mcp.sse_transport import SessionAwareSseTransport
from uuid import UUID

# Replace sse initialization
sse = SessionAwareSseTransport("/mcp/messages/")

# Global session binding store (singleton)
_session_binding_store: Optional[MemorySessionBindingStore] = None

def get_session_binding_store() -> MemorySessionBindingStore:
    global _session_binding_store
    if _session_binding_store is None:
        ttl = int(os.environ.get("MCP_SESSION_TTL_SECONDS", "1800"))
        _session_binding_store = MemorySessionBindingStore(default_ttl_seconds=ttl)
    return _session_binding_store

# Modified GET handler
@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    principal = _extract_principal_from_request(request)

    # ... existing context var setup ...

    try:
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
            # Capture and bind session_id
            session_id = sse.current_session_id
            if session_id:
                store = get_session_binding_store()
                store.create(
                    session_id=session_id,
                    user_id=principal.user_id,
                    org_id=principal.org_id,
                    dpop_thumbprint=getattr(principal, 'dpop_thumbprint', None),
                )

            await mcp._mcp_server.run(read_stream, write_stream, ...)
    finally:
        # Cleanup binding on disconnect
        if session_id:
            get_session_binding_store().delete(session_id)
        # ... existing cleanup ...

# Modified POST handler
async def handle_post_message(request: Request):
    # 1. Extract and validate session_id from query params
    session_id_str = request.query_params.get("session_id")
    if not session_id_str:
        return JSONResponse(status_code=400, content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"})

    try:
        session_id = UUID(session_id_str)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid session_id format", "code": "INVALID_SESSION_ID"})

    # 2. Authenticate the request
    try:
        principal = _extract_principal_from_request(request)
        validate_iat_not_future(principal.claims.iat)  # NEW: iat validation
    except AuthenticationError as e:
        return JSONResponse(status_code=401, content={"error": e.message, "code": e.code}, headers={"WWW-Authenticate": 'Bearer realm="mcp"'})

    # 3. Validate session binding
    store = get_session_binding_store()
    if not store.validate(session_id, principal.user_id, principal.org_id):
        return JSONResponse(status_code=403, content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"})

    # 4. Set context variables and process
    # ... existing code ...
```

**Git Commit Message**: `feat(mcp): integrate session binding into SSE GET/POST handlers`

---

## PHASE 3: Development Protocol

### The Recursive Testing Loop

Execute this loop for EVERY feature:

```
1. WRITE TESTS FIRST
   └── Create failing tests that define expected behavior

2. IMPLEMENT FEATURE
   └── Write minimum code to pass tests

3. RUN ALL TESTS
   ├── Unit tests for new feature
   ├── Integration tests
   └── ALL previous tests (regression)

4. ON PASS:
   ├── git add -A
   ├── git commit -m "feat(scope): description"
   └── Update Agent Scratchpad below

5. ON FAIL:
   ├── Spawn general-purpose agent: "Debug why [test] is failing"
   ├── If complex: Spawn Explore agent to find similar working patterns
   ├── DO NOT proceed until green
   └── Return to step 3

6. REGRESSION VERIFICATION
   ├── After each new feature
   ├── Verify all past features still work
   └── If regression found: fix before continuing

7. REPEAT for next feature
```

### Git Checkpoint Protocol

```bash
# After each passing feature
git add -A
git commit -m "type(scope): description"

# Conventional commit types:
# feat:     New feature
# fix:      Bug fix
# test:     Adding tests
# refactor: Code refactoring
# docs:     Documentation

# Tag milestone
git tag -a v0.1.0-mcp-session-binding -m "Phase 0: MCP SSE Session Binding"
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-28
**Current Phase**: Phase 0 - PRD Creation
**Last Action**: Created comprehensive PRD with test specifications

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | Remove auth bypass | [ ] | [ ] | [ ] | |
| 2 | Add iat validation | [ ] | [ ] | [ ] | |
| 3 | Session binding store | [ ] | [ ] | [ ] | |
| 4 | SSE transport wrapper | [ ] | [ ] | [ ] | |
| 5 | Integrate binding | [ ] | [ ] | [ ] | |

### Decisions Made

1. **Decision**: Use in-memory session binding store for Phase 0
   - **Rationale**: Simpler implementation, suitable for single-process demo
   - **Alternatives Considered**: Valkey store (deferred to Phase 1 for multi-worker support)

2. **Decision**: Capture session_id by accessing `_read_stream_writers` dict
   - **Rationale**: Minimal library modification, avoids patching UUID generation
   - **Alternatives Considered**: Subclassing with custom UUID injection (more complex)

3. **Decision**: 30-minute default session TTL
   - **Rationale**: Matches typical SSE session duration, aligns with existing patterns
   - **Alternatives Considered**: 1 hour (too long for security), 5 minutes (too short for UX)

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Test patterns | pytest + httpx, security tests in tests/security/, 247+ test methods |
| Explore | MCP server auth | `_check_tool_scope` bypass at line 227-228, principal_var pattern |
| Explore | SSE transport | session_id generated in connect_sse, stored in _read_stream_writers |

### Known Issues & Blockers

- [ ] Issue: SSE transport session_id capture relies on internal `_read_stream_writers` dict
  - Status: Acceptable for Phase 0, may need revision if library changes
  - Attempted solutions: N/A - using documented internal structure

### Notes for Next Session

> Continue from here in the next session:

- [ ] Create test file: `tests/security/test_mcp_session_binding.py`
- [ ] Create test file: `tests/stores/test_session_binding_store.py`
- [ ] Implement Feature 1: Remove auth bypass
- [ ] Implement Feature 3: Session binding store
- [ ] Implement Feature 4: SSE transport wrapper
- [ ] Implement Feature 2: IAT validation
- [ ] Implement Feature 5: Integration

### Test Results Log

```
[No tests run yet - PRD phase]
```

### Recent Git History

```
7eaf38f5 feat(phase4.5): complete GDPR compliance module with SAR export and cascading delete
703dbca7 docs: add Phase 4.5 GDPR Compliance continuation prompt
b98c962b feat(phase7): complete Deployment, DR, and Hardening
```

---

## Execution Checklist

When executing this PRD, follow this order:

- [x] 1. Read Agent Scratchpad for prior context
- [x] 2. **Spawn parallel Explore agents** to understand:
  - Existing codebase structure
  - Test patterns and conventions
  - Related modules and dependencies
- [x] 3. Review/complete success criteria (Phase 1)
- [x] 4. Design test suite structure (Phase 1)
- [ ] 5. Write feature specifications (Phase 2)
- [ ] 6. For each feature:
  - [ ] Write tests first
  - [ ] Implement feature
  - [ ] Run unit/integration tests
  - [ ] Commit on green
  - [ ] Run regression tests
  - [ ] Update scratchpad
- [ ] 7. Tag milestone when complete

---

## Quick Reference Commands

```bash
# Run tests
cd openmemory/api
pytest tests/security/test_mcp_session_binding.py -v
pytest tests/stores/test_session_binding_store.py -v
pytest tests/security/ -v  # All security tests

# Git workflow
git status
git add -A
git commit -m "feat(mcp): description"
git log --oneline -5

# Tag milestone
git tag -a v0.1.0-mcp-session-binding -m "Phase 0: MCP SSE Session Binding"
```

---

## Key File References

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| MCP SSE handlers | [mcp_server.py](openmemory/api/app/mcp_server.py) | 3037-3137 | GET/POST endpoints |
| Tool scope check | [mcp_server.py](openmemory/api/app/mcp_server.py) | 215-237 | Auth bypass point |
| JWT validation | [jwt.py](openmemory/api/app/security/jwt.py) | 79-203 | Token validation |
| Principal types | [types.py](openmemory/api/app/security/types.py) | 33-166 | Security types |
| Episodic store | [episodic_store.py](openmemory/api/app/stores/episodic_store.py) | 48-403 | TTL patterns |
| MCP SSE lib | .venv/.../mcp/server/sse.py | 144-161 | session_id generation |

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
