# ADR: MCP SSE Authentication Architecture

**Status**: Proposed
**Date**: 2025-12-28
**Deciders**: TBD
**Technical Story**: Phase 1 MCP auth blocker - SSE transport authentication gap

## Context and Problem Statement

The MCP (Model Context Protocol) server uses SSE (Server-Sent Events) for client communication. There is an authentication gap where:

1. Initial SSE GET connection authenticates via JWT in Authorization header
2. Subsequent POST messages are separate HTTP requests where Python context variables are lost
3. Tool-level auth checks may be bypassed due to a "backwards compatibility" fallback

This creates a security vulnerability where unauthenticated tool invocations could execute.

## Current State

### Authentication Flow Today

**Phase 1: SSE Connection (GET)**
- Client calls `GET /mcp/{client_name}/sse/{user_id}`
- Server extracts JWT from `Authorization: Bearer <token>` header
- Validates JWT via `validate_jwt()` (checks signature, expiration, scopes, org_id, jti)
- Sets context variables: `user_id_var`, `org_id_var`, `client_name_var`, `principal_var`
- Opens SSE connection via `sse.connect_sse()` to stream responses

**Phase 2: Tool Invocations (POST)**
- Client sends `POST /mcp/{client_name}/sse/{user_id}/messages/`
- Each POST is a **separate HTTP request** with its own request scope
- Server re-authenticates by extracting JWT from `Authorization` header again
- Sets context variables again
- SSE library's `handle_post_message()` processes the MCP message
- Tools read context vars to determine user_id/org_id, call `_check_tool_scope()`

### The Security Gap

Location: `openmemory/api/app/mcp_server.py` lines 226-229

```python
def _check_tool_scope(required_scope: str) -> str | None:
    # ...
    principal = principal_var.get(None)
    if not principal:
        return None  # Backwards compatibility - no auth check
    # ...
```

**Problems:**
1. **Context Loss**: Between SSE connection and POST message, HTTP request context is lost
2. **No Session State**: POST messages are isolated HTTP requests with no server-side session
3. **Backwards Compatibility Fallback**: If `principal_var.get(None)` returns None, auth is bypassed
4. **Context Var Isolation**: If POST arrives without setting principal_var, tools execute with zero auth
5. **No Replay Prevention**: POST messages lack proof they came from the authenticated session

## Constraints

### Protocol Constraints (MCP/SSE)
- MCP spec doesn't define session persistence for SSE transport
- MCP messages are JSON-RPC 2.0 without standard auth fields
- MCP protocol assumes auth at transport layer, not per-message

### Library Constraints
- Python `contextvars`: Task-scoped, don't persist across HTTP boundaries
- `mcp.server.sse.SseServerTransport`: Transport-level only, doesn't validate auth on POST
- FastAPI: Each request/task gets fresh context vars

### Client Compatibility
- IDE Plugins (VSCode): Use `fetch` API, send JWT in `Authorization` header on GET and POST
- CLI Tools: Similar pattern (GET to open SSE, POST to send messages)
- MCP Clients: No standard way to send session tokens in POST body without extension

## Decision Drivers

1. **Security**: Close the auth gap completely
2. **Minimal Breaking Changes**: Existing clients should continue working
3. **Implementation Effort**: Prefer low-effort solutions
4. **Statelessness**: Avoid server-side session state if possible
5. **Protocol Alignment**: Stay within MCP spec where possible

## Considered Options

### Option A: Session-Based Auth (Valkey/Redis)

**Approach:**
- On SSE GET: Create session in Valkey with JWT claims, key: `mcp_session:{session_id}` (TTL 24h)
- Return `Set-Cookie: mcp_session={session_id}` (httponly, secure)
- On POST: Extract session from cookie, validate session exists and matches JWT claims

**Pros:**
- Stateful, no JWT on every POST (lighter payloads)
- Integrates with existing Valkey store patterns
- Session TTL prevents replay (auto-cleanup)
- Client-compatible for browsers

**Cons:**
- Breaks CLI/headless clients (can't handle cookies)
- IDE plugins need changes (most don't handle Set-Cookie)
- Adds session management complexity
- **Effort: High**

### Option B: Per-Message JWT in Body

**Approach:**
- Client includes JWT in every POST message body: `{"jsonrpc": "2.0", "method": "...", "_auth": {"token": "..."}}`
- Server extracts `_auth.token`, validates it, sets context vars
- Falls back to Authorization header if body field missing (backwards compat)

**Pros:**
- No state on server (stateless)
- Works with all clients without changes initially
- Low client migration effort
- Backwards compatible
- Clean separation: auth travels with each message

**Cons:**
- Repeats JWT on every POST (higher bandwidth, minimal)
- JWT visibility in request body logs
- No replay protection without JTI tracking
- **Effort: Low-Medium**

### Option C: MCP Protocol Extension

**Approach:**
- Define custom MCP `authenticate` tool that returns a session token
- Client calls it after SSE connection
- Client includes token in subsequent tool calls as parameter

**Pros:**
- Stays within MCP abstraction
- Decouples auth from transport
- Could support multiple auth schemes

**Cons:**
- Modifies MCP protocol semantics
- Breaking for existing clients
- Tool implementations must check for token parameter
- **Effort: Very High**

### Option D: WebSocket Migration

**Approach:**
- Replace SSE with WebSocket
- Keep connection open, send all tool calls on same WebSocket frame
- Auth happens once at upgrade; context persists for connection lifetime

**Pros:**
- Natural solution for persistent authenticated sessions
- Context vars stay alive for entire connection
- No replay/session concerns

**Cons:**
- Massive breaking change for all MCP clients
- MCP library's SseServerTransport becomes unused
- IDE plugins, CLI tools all need rewrites
- **Effort: Extreme**

### Option E: Query Parameter Session ID

**Approach:**
- On SSE GET: Return session ID in response header
- Client includes session ID in POST: `POST /mcp/messages?session={session_id}`
- Server validates: session exists in Valkey, principal matches, TTL valid

**Pros:**
- Works with all clients (no cookie handling, just URL param)
- Stateful but minimal overhead
- Backward compatible (session_id optional)
- Can combine with JWT header validation

**Cons:**
- Session ID visible in logs/URLs
- Client must parse response and include in POSTs
- Requires client implementation update
- **Effort: Medium**

## Decision

**Recommended: Option B (Per-Message JWT in Body)**

### Rationale

1. **Immediate Impact**: Closes the backwards-compatibility fallback gap without server-side state
2. **Client Flexibility**: Works with existing CLI, IDE plugins, and custom clients without changes initially
3. **Non-Breaking**: Authorization header fallback allows gradual client migration
4. **Minimal Effort**: Focused code change (body parsing + validation) vs. infrastructure changes
5. **Aligns with Existing Pattern**: Uses same `validate_jwt()` and `Principal` flow as REST routers
6. **Stateless**: No Valkey session keys, no cookies, no protocol changes

### Alternative

**Option E (Query Parameter Session ID)** as fallback if JWT-per-message feels too verbose:
- Use session ID as lighter alternative to full JWT on POST
- Session only stored if client doesn't send JWT header

## Implementation Plan

### Step 1: Remove Backwards Compatibility Fallback (CRITICAL - Emergency Fix)

```python
# mcp_server.py line 226-229
# BEFORE:
principal = principal_var.get(None)
if not principal:
    return None  # Backwards compatibility - no auth check

# AFTER:
principal = principal_var.get(None)
if not principal:
    return json.dumps({
        "error": "Authentication required",
        "code": "MISSING_AUTH",
    })
```

### Step 2: Update Message Handler for Body Auth

```python
async def handle_post_message(request: Request):
    # Try to extract auth from body first
    body = await request.json()
    auth_token = body.get("_auth", {}).get("token")

    if auth_token:
        claims = validate_jwt(auth_token)
        principal = Principal(user_id=claims.sub, org_id=claims.org_id, claims=claims)
    else:
        # Fall back to Authorization header
        principal = _extract_principal_from_request(request)

    # Set context var
    principal_var.set(principal)
    # ... continue with message handling
```

### Step 3: SSE Response Hint (Optional)

```python
@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    # ... auth ...
    response = await sse.connect_sse(...)
    response.headers["X-Auth-Scheme"] = "jwt-per-message"
    return response
```

### Step 4: Client Documentation

- Document: send `{"_auth": {"token": "..."}}` in POST body
- Provide migration guide for existing clients
- Support header-only auth for 2-3 releases (with deprecation warning)

### Step 5: Testing (~25-30 tests)

| Test Category | Tests |
|---------------|-------|
| POST with JWT in body | 5 |
| POST without any auth | 3 |
| POST with invalid JWT | 4 |
| Scope enforcement | 6 |
| Backward compat (header only) | 4 |
| Integration (full SSE flow) | 5 |

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Clients don't implement JWT-in-body | Medium | Provide reference implementation; keep header fallback |
| JWT exposure in request body logs | Low | Document secure logging; mark `_auth` as sensitive |
| Replay attacks on POST | Medium | Use JTI tracking (already in JWT validation); enforce short expiry |
| Header vs. body JWT conflicts | Low | Document precedence: body wins; test both |
| Backwards compat fallback exploited | **High** | **Remove fallback immediately (Step 1)** |

## Consequences

### Positive
- Closes critical security gap
- Minimal client migration required
- No new infrastructure dependencies
- Aligns with existing security patterns

### Negative
- Slightly larger POST payloads (JWT included)
- Clients eventually need to update (but not immediately)

### Neutral
- Two auth paths to maintain temporarily (body + header)

## Comparison Matrix

| Criteria | Option A | Option B | Option C | Option D | Option E |
|----------|----------|----------|----------|----------|----------|
| Implementation Effort | High | **Low** | Very High | Extreme | Medium |
| Client Breaking Changes | Yes | **No** | Yes | Extreme | Minor |
| Stateful Server | Yes | **No** | Yes | No | Light |
| Works with CLI/IDE | Limited | **Full** | Partial | Partial | Full |
| Security Posture | Strong | **Strong** | Medium | Strong | Strong |
| MCP Spec Aligned | Yes | **Yes** | No | Partial | Yes |

## Second Perspective Analysis

A second independent analysis was conducted that raised important counterpoints:

### Key Findings

1. **POST Handler Already Authenticates**: The current `handle_post_message()` (lines 3093-3137) DOES extract principal via `_extract_principal_from_request()` and returns 401 on failure. The context var IS being set properly.

2. **Backwards Compatibility Fallback is Rarely Triggered**: In normal operation, principal_var is set by POST handler before tools execute. The fallback at line 227 would only trigger if:
   - MCP's `sse.handle_post_message()` spawns child tasks where context vars don't propagate
   - Exception handling creates early returns before principal_var is set
   - Task context isolation issues occur

3. **Missing Issues Not in Original ADR**:
   - **No Active JTI Replay Prevention**: `validate_jwt()` extracts JTI but doesn't check Valkey cache to prevent replay
   - **DPoP Binding Gap**: POST requests don't validate DPoP proofs, weakening proof-of-possession for tool invocations
   - **Token Logging Risk**: Option B puts JWT in request body, which often appears in logs (RFC 6750 Section 5 violation)

### Critique of Option B

| Concern | Severity | Details |
|---------|----------|---------|
| JWT in request body logs | **High** | Violates RFC 6750 best practices; requires log sanitization |
| DPoP not addressed | **Medium** | If token is DPoP-bound, POST needs DPoP header with htm/htu |
| JTI replay prevention missing | **Medium** | validate_jwt() doesn't check JTI cache |

### Alternative Recommendation: Hybrid Approach

The second perspective recommends **Option E + DPoP validation** instead of Option B:

**Immediate (Emergency):**
1. Remove backwards compatibility fallback (same as original Step 1)
2. Add JTI replay prevention to `validate_jwt()`
3. Verify context var propagation through `sse.handle_post_message()` call chain

**Short-term:**
1. Implement Option E (Query Parameter Session ID):
   - On GET: Return `X-Session-ID` header
   - On POST: Require `?session={session_id}` in URL
   - Validate session exists in Valkey, matches principal, TTL valid
   - Leverage existing Valkey patterns (DPoP JTI cache)

2. Add DPoP validation to POST requests:
   - Require `DPoP` header if token is DPoP-bound
   - Validate htm="POST" and htu matches request URI

**Comparison:**

| Criteria | Option B (Body JWT) | Option E + DPoP |
|----------|---------------------|-----------------|
| Token in logs | YES (risk) | NO |
| DPoP support | NO | YES |
| Replay prevention | Weak | Strong (existing cache) |
| Client breaking change | High | Minimal |
| Stateless | YES | NO (but uses existing Valkey) |

### Conclusion

The second perspective agrees on the urgency of removing the backwards-compatibility fallback but recommends Option E over Option B due to:
- Avoiding token exposure in logs
- Leveraging existing Valkey infrastructure
- Properly addressing DPoP binding
- Lower client migration burden

**Final Recommendation**: Consider Option E as primary with DPoP validation, rather than Option B. Both perspectives agree Step 1 (remove fallback) is critical and should be done immediately.

---

## References

- MCP Specification: https://modelcontextprotocol.io/
- OAuth 2.0 Bearer Token Usage: RFC 6750
- DPoP (Demonstrating Proof of Possession): RFC 9449
- Existing security module: `openmemory/api/app/security/`
- MCP server implementation: `openmemory/api/app/mcp_server.py`
