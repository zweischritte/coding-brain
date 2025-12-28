# PRD: MCP SSE Session-Bound Authentication - Implementation Guide

**Status**: Ready for Implementation
**Owner**: Security Architecture
**Phase**: 0 (Demo-Ready)
**Goal**: Close auth bypass gaps, bind MCP SSE session_id to authenticated principal

---

## Executive Summary

This PRD implements critical security fixes for the MCP SSE authentication flow to:
1. **Close auth bypass** in `_check_tool_scope()` when `principal_var` is unset
2. **Bind SSE session_id** to authenticated principal on GET
3. **Validate session binding** on every POST before tool execution
4. **Add iat validation** to reject future-dated tokens

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

- **Clock skew**: Tokens with iat up to 30 seconds in the future are accepted
- **Session expiry**: POST to expired session returns 403 (not 401)
- **Concurrent POSTs**: Multiple simultaneous POSTs to same session all succeed if bound
- **Session cleanup**: SSE disconnect triggers binding cleanup
- **Store unavailable**: If session store fails, reject request (fail closed)
- **Missing session_id**: POST without session_id query param returns 400
- **Invalid session_id format**: Non-UUID session_id returns 400
- **DPoP mismatch**: If DPoP thumbprint was bound, POST must have matching DPoP (Phase 1)

### 1.3 Design Test Suite Structure

```
openmemory/api/tests/
├── security/
│   ├── test_mcp_session_binding.py      # NEW: Session binding integration tests
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
| SB-007 | Session Binding | Unit | Thread-safe concurrent access | No race conditions |
| AB-001 | Auth Bypass | Unit | `_check_tool_scope` with no principal | Returns error JSON |
| AB-002 | Auth Bypass | Unit | `_check_tool_scope` with valid scope | Returns None |
| AB-003 | Auth Bypass | Unit | `_check_tool_scope` with missing scope | Returns error JSON |
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

**File**: [mcp_server.py:227-228](openmemory/api/app/mcp_server.py#L227-L228)

**Dependencies**: None

**Test Cases**:
- [x] Unit: `_check_tool_scope()` returns error when principal is None (AB-001)
- [x] Unit: `_check_tool_scope()` returns None when principal has scope (AB-002)
- [x] Integration: MCP tool without auth returns error (AB-004)

**Implementation**:

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

### Feature 2: Add IAT Validation to MCP SSE POST Handler

**Description**: Call `validate_iat_not_future()` in MCP SSE POST handler to reject future-dated tokens.

**File**: [mcp_server.py:3093-3137](openmemory/api/app/mcp_server.py#L3093-L3137)

**Dependencies**: [jwt.py:validate_iat_not_future](openmemory/api/app/security/jwt.py)

**Test Cases**:
- [x] Unit: Token with past iat accepted (IAT-001)
- [x] Unit: Token with future iat (within 30s) accepted (IAT-002)
- [x] Unit: Token with future iat (>30s) rejected (IAT-003)
- [x] Integration: POST with future-dated token returns 401

**Implementation**:

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

**New File**: [session_binding.py](openmemory/api/app/security/session_binding.py)

**Dependencies**: None (Valkey support in Phase 1)

**Test Cases**:
- [x] Unit: Create binding stores all fields (SB-001)
- [x] Unit: Get binding returns correct data (SB-002)
- [x] Unit: Validate binding matches principal (SB-003, SB-004)
- [x] Unit: Expired binding returns None (SB-005)
- [x] Unit: Delete binding removes entry (SB-006)
- [x] Unit: Thread-safe concurrent access (SB-007)

**Implementation**:

```python
"""Session binding store for MCP SSE authentication.

Binds SSE session_id to authenticated principal to prevent session hijacking.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from uuid import UUID
import os
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
    """Thread-safe in-memory session binding store with TTL.

    Used for local development and single-process deployments.
    For multi-worker production deployments, use ValkeySessionBindingStore (Phase 1).
    """

    def __init__(self, default_ttl_seconds: int = 1800):
        self._bindings: Dict[UUID, SessionBinding] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def create(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> SessionBinding:
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

    def validate(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
    ) -> bool:
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

    def cleanup_expired(self) -> int:
        """Remove all expired bindings. Returns count of removed bindings."""
        now = datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            expired = [
                sid for sid, b in self._bindings.items()
                if b.expires_at <= now
            ]
            for sid in expired:
                del self._bindings[sid]
                removed += 1
        return removed


# Singleton instance
_session_binding_store: Optional[MemorySessionBindingStore] = None


def get_session_binding_store() -> MemorySessionBindingStore:
    """Get the session binding store singleton."""
    global _session_binding_store
    if _session_binding_store is None:
        ttl = int(os.environ.get("MCP_SESSION_TTL_SECONDS", "1800"))
        _session_binding_store = MemorySessionBindingStore(default_ttl_seconds=ttl)
    return _session_binding_store


def reset_session_binding_store() -> None:
    """Reset the singleton (for testing only)."""
    global _session_binding_store
    _session_binding_store = None
```

**Git Commit Message**: `feat(security): add memory-based session binding store with TTL`

---

### Feature 4: SSE Transport Wrapper for Session ID Capture

**Description**: Create a minimal wrapper around `SseServerTransport` that captures the generated `session_id` during `connect_sse()`.

**New Directory**: `openmemory/api/app/mcp/`
**New File**: [sse_transport.py](openmemory/api/app/mcp/sse_transport.py)

**Dependencies**: `mcp.server.sse.SseServerTransport`

**Test Cases**:
- [x] Unit: Wrapper captures session_id on connect (SSE-001)
- [x] Unit: Wrapper passes through all other functionality
- [x] Integration: session_id available after connect_sse()

**Implementation**:

```python
"""SSE transport wrapper for session ID capture.

The MCP library's SseServerTransport generates session_id internally but doesn't
expose it. This wrapper captures it for session binding.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Optional, Tuple
from uuid import UUID

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.server.sse import SseServerTransport


class SessionAwareSseTransport(SseServerTransport):
    """SSE transport wrapper that exposes session_id for binding.

    The session_id is generated inside connect_sse() and stored in the
    _read_stream_writers dict. This wrapper captures it for use by the
    session binding system.
    """

    def __init__(self, endpoint: str):
        super().__init__(endpoint)
        self._session_ids_by_scope: Dict[int, UUID] = {}

    @asynccontextmanager
    async def connect_sse(
        self, scope, receive, send
    ) -> AsyncGenerator[Tuple[MemoryObjectReceiveStream, MemoryObjectSendStream], None]:
        """Wrap connect_sse to capture session_id before yielding."""
        scope_id = id(scope)

        async with super().connect_sse(scope, receive, send) as streams:
            # After connect, find the newly added session_id
            # The library stores it in _read_stream_writers dict
            if hasattr(self, '_read_stream_writers') and self._read_stream_writers:
                # Find session_id not already tracked
                for sid in self._read_stream_writers.keys():
                    if sid not in self._session_ids_by_scope.values():
                        self._session_ids_by_scope[scope_id] = sid
                        break

            yield streams

            # Cleanup after connection ends
            self._session_ids_by_scope.pop(scope_id, None)

    def get_session_id(self, scope) -> Optional[UUID]:
        """Get the session_id associated with a request scope."""
        return self._session_ids_by_scope.get(id(scope))

    @property
    def current_session_id(self) -> Optional[UUID]:
        """Get the most recently captured session_id.

        DEPRECATED: Use get_session_id(scope) instead for multi-connection safety.
        This property is kept for backwards compatibility but may return
        incorrect results under concurrent connections.
        """
        if self._session_ids_by_scope:
            return list(self._session_ids_by_scope.values())[-1]
        return None
```

**Git Commit Message**: `feat(mcp): add SSE transport wrapper to capture session_id`

---

### Feature 5: Integrate Session Binding into SSE Handlers

**Description**: Bind session to principal on GET, validate binding on POST.

**File**: [mcp_server.py](openmemory/api/app/mcp_server.py)

**Dependencies**: Features 3, 4

**Test Cases**:
- [x] Integration: GET creates session binding (SSE-002)
- [x] Integration: POST with valid binding succeeds (SSE-003)
- [x] Integration: POST with mismatched binding returns 403 (SSE-004)
- [x] Integration: POST with expired binding returns 403 (SSE-005)
- [x] Integration: POST without session_id returns 400 (SSE-006)

**Implementation Changes**:

**1. Import and transport replacement:**

```python
# At top of mcp_server.py, replace:
from mcp.server.sse import SseServerTransport

# With:
from app.mcp.sse_transport import SessionAwareSseTransport
from app.security.session_binding import get_session_binding_store
from app.security.jwt import validate_iat_not_future
from uuid import UUID
```

```python
# Replace transport initialization (~line 134):
# BEFORE:
sse = SseServerTransport("/mcp/messages/")

# AFTER:
sse = SessionAwareSseTransport("/mcp/messages/")
```

**2. GET handler binding creation:**

```python
# In handle_sse() after connect_sse(), add session binding:
@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    principal = _extract_principal_from_request(request)
    # ... existing context var setup ...

    session_id = None
    try:
        async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
            # Capture and bind session_id
            session_id = sse.get_session_id(request.scope)
            if session_id and principal:
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
```

**3. POST handler validation:**

```python
# In handle_post_message(), add session validation:
async def handle_post_message(request: Request):
    # 1. Extract session_id from query params
    session_id_str = request.query_params.get("session_id")
    if not session_id_str:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing session_id", "code": "MISSING_SESSION_ID"}
        )

    try:
        session_id = UUID(session_id_str)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid session_id format", "code": "INVALID_SESSION_ID"}
        )

    # 2. Authenticate the request
    try:
        principal = _extract_principal_from_request(request)
        validate_iat_not_future(principal.claims.iat)  # NEW: iat validation
    except AuthenticationError as e:
        return JSONResponse(
            status_code=401,
            content={"error": e.message, "code": e.code},
            headers={"WWW-Authenticate": 'Bearer realm="mcp"'}
        )

    # 3. Validate session binding
    store = get_session_binding_store()
    if not store.validate(session_id, principal.user_id, principal.org_id):
        return JSONResponse(
            status_code=403,
            content={"error": "Session binding mismatch", "code": "SESSION_BINDING_INVALID"}
        )

    # 4. Set context variables and continue
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
   ├── Debug the failure
   ├── DO NOT proceed until green
   └── Return to step 3

6. REGRESSION VERIFICATION
   ├── After each new feature
   ├── Verify all past features still work
   └── If regression found: fix before continuing

7. REPEAT for next feature
```

### Implementation Order

1. **Feature 3**: Session Binding Store (foundation)
2. **Feature 4**: SSE Transport Wrapper (foundation)
3. **Feature 1**: Remove Auth Bypass (critical security fix)
4. **Feature 2**: Add IAT Validation (security hardening)
5. **Feature 5**: Integrate Binding (ties everything together)

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

# Tag milestone when Phase 0 complete
git tag -a v0.1.0-mcp-session-binding -m "Phase 0: MCP SSE Session Binding"
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-28
**Current Phase**: Phase 0 - PRD Creation
**Last Action**: Created implementation PRD

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | Session Binding Store | [ ] | [ ] | [ ] | |
| 2 | SSE Transport Wrapper | [ ] | [ ] | [ ] | |
| 3 | Remove Auth Bypass | [ ] | [ ] | [ ] | |
| 4 | IAT Validation | [ ] | [ ] | [ ] | |
| 5 | Integrate Binding | [ ] | [ ] | [ ] | |

### Decisions Made

1. **Decision**: Use in-memory session binding store for Phase 0
   - **Rationale**: Simpler implementation, suitable for single-process demo
   - **Alternatives Considered**: Valkey store (deferred to Phase 1 for multi-worker support)

2. **Decision**: Capture session_id by accessing `_read_stream_writers` dict
   - **Rationale**: Minimal library modification, avoids patching UUID generation
   - **Alternatives Considered**: Subclassing with custom UUID injection (more complex)

3. **Decision**: 30-minute default session TTL
   - **Rationale**: Matches typical SSE session duration, configurable via env var
   - **Alternatives Considered**: 1 hour (too long for security), 5 minutes (too short for UX)

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Codebase exploration | Comprehensive security module structure, test patterns, TTL patterns from episodic store |

### Known Issues & Blockers

- [ ] Issue: SSE transport session_id capture relies on internal `_read_stream_writers` dict
  - Status: Acceptable for Phase 0, may need revision if library changes
  - Attempted solutions: N/A - using documented internal structure

### Notes for Next Session

> Continue from here in the next session:

- [ ] Create test file: `tests/stores/test_session_binding_store.py`
- [ ] Create test file: `tests/security/test_mcp_session_binding.py`
- [ ] Create new directory: `openmemory/api/app/mcp/`
- [ ] Implement Feature 3: Session binding store
- [ ] Implement Feature 4: SSE transport wrapper
- [ ] Implement Feature 1: Remove auth bypass
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
- [x] 2. **Spawn Explore agent** to understand codebase
- [x] 3. Review/complete success criteria (Phase 1)
- [x] 4. Design test suite structure (Phase 1)
- [x] 5. Write feature specifications (Phase 2)
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
# Navigate to API directory
cd openmemory/api

# Run session binding store tests
pytest tests/stores/test_session_binding_store.py -v

# Run MCP session binding tests
pytest tests/security/test_mcp_session_binding.py -v

# Run all security tests (regression)
pytest tests/security/ -v

# Run all tests
pytest -v

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
| IAT validation | [jwt.py](openmemory/api/app/security/jwt.py) | 184-203 | Future token check |
| Principal types | [types.py](openmemory/api/app/security/types.py) | 33-166 | Security types |
| Episodic store | [episodic_store.py](openmemory/api/app/stores/episodic_store.py) | 48-403 | TTL patterns |
| Security deps | [dependencies.py](openmemory/api/app/security/dependencies.py) | 22-149 | Full auth flow |
| MCP auth tests | [test_mcp_auth.py](openmemory/api/tests/security/test_mcp_auth.py) | 1-244 | Test skeleton |
| Store fixtures | [conftest.py](openmemory/api/tests/stores/conftest.py) | 1-253 | Test fixtures |

---

## Related Documents

- [PRD-MCP-SSE-SESSION-BINDING.md](docs/PRD-MCP-SSE-SESSION-BINDING.md) - Full security analysis
- [ADR-MCP-SSE-AUTH.md](docs/ADR-MCP-SSE-AUTH.md) - Architecture decision record
- [CONTINUATION-PROMPT-MCP-SSE-AUTH.md](docs/CONTINUATION-PROMPT-MCP-SSE-AUTH.md) - Implementation guide

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
