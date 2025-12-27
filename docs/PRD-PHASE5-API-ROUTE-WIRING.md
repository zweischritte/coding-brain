# PRD: Phase 5 - API Route Wiring

**Plan Reference**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress Tracker**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Continuation Prompt**: `docs/CONTINUATION-PROMPT-PHASE5-API-ROUTES.md`
**Style**: STRICT TDD - Write failing tests first, then implement.

---

## Executive Summary

Phase 5 wires the Phase 4 multi-tenant stores to REST API routes with proper authentication, authorization, and validation. Before adding new routes, we must fix the Phase 1 MCP auth blocker and pre-existing test failures.

**Prerequisite**: Phase 4 Multi-tenant Stores COMPLETE (125 tests, 3,204 total)

**Goal**: Expose `PostgresFeedbackStore`, `PostgresExperimentStore`, and `TenantOpenSearchStore` via REST API routes with JWT auth and scope enforcement.

**Target**: Add ~100 new tests (90-110 estimated), reaching ~3,304 total tests.

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Define Success Criteria

All measurable acceptance criteria for Phase 5:

1. [x] Phase 1 MCP SSE endpoints have authentication (blocker resolved)
2. [ ] MCP tools have permission checks (tool-level scopes)
3. [ ] Pre-existing `test_router_auth.py` failures fixed (4 tests)
4. [ ] New scopes added: `FEEDBACK_READ`, `FEEDBACK_WRITE`, `EXPERIMENTS_READ`, `EXPERIMENTS_WRITE`, `SEARCH_READ`
5. [ ] Feedback router: 4 endpoints with 25+ tests
6. [ ] Experiments router: 7 endpoints with 30+ tests
7. [ ] Search router: 3 endpoints with 20+ tests
8. [ ] All routers registered in `main.py`
9. [ ] Zero regression (all 3,204+ existing tests pass)
10. [ ] All new endpoints return proper 401/403 for auth failures

### 1.2 Define Edge Cases

Edge cases that must be handled:

**Authentication & Authorization:**
- Missing Authorization header → 401
- Invalid/expired JWT → 401
- Valid JWT but missing required scope → 403
- Cross-tenant access attempt → 404 (not 403, to prevent enumeration)

**Feedback Router:**
- Feedback for non-existent query_id → 400 or accept (TBD)
- Feedback metrics with no data → Return zero counts, not error
- Metrics by-tool with no matching tools → Return empty list

**Experiments Router:**
- Status transition to invalid state → 400
- Assign user to non-existent experiment → 404
- Assign user to non-RUNNING experiment → 400
- Get assignment for unassigned user → 404 or null response
- List experiments when org has none → Empty list, not error

**Search Router:**
- Search with empty query → 400
- Search when OpenSearch unavailable → 503
- Hybrid search without query_vector → 400

### 1.3 Design Test Suite Structure

```
openmemory/api/tests/
├── routers/
│   ├── test_feedback_router.py      # NEW: 25-30 tests
│   ├── test_experiments_router.py   # NEW: 30-35 tests
│   └── test_search_router.py        # NEW: 20-25 tests
├── security/
│   ├── test_router_auth.py          # FIX: 4 failing tests
│   ├── test_mcp_auth.py             # FIX: MCP SSE auth tests
│   └── test_new_scopes.py           # NEW: 10-15 tests for new scopes
└── integration/
    └── test_store_router_integration.py  # NEW: Store-to-router integration
```

### 1.4 Write Test Specifications

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| **MCP Auth Fix** | | | |
| SSE auth | Unit | `test_mcp_sse_requires_auth` | 401 without token |
| SSE auth | Unit | `test_mcp_sse_accepts_valid_token` | 200 with valid JWT |
| Tool scopes | Unit | `test_mcp_tool_requires_scope` | 403 without scope |
| **Feedback Router** | | | |
| POST /feedback | Unit | `test_create_feedback_requires_auth` | 401 without token |
| POST /feedback | Unit | `test_create_feedback_requires_scope` | 403 without FEEDBACK_WRITE |
| POST /feedback | Unit | `test_create_feedback_success` | 201 with valid data |
| POST /feedback | Unit | `test_create_feedback_validates_outcome` | 400 for invalid outcome |
| GET /feedback | Unit | `test_query_feedback_requires_auth` | 401 without token |
| GET /feedback | Unit | `test_query_feedback_filters_by_org` | Only returns org's data |
| GET /feedback/metrics | Unit | `test_metrics_aggregates_correctly` | Returns acceptance_rate, counts |
| GET /feedback/by-tool | Unit | `test_metrics_by_tool_groups_correctly` | Returns tool-grouped metrics |
| **Experiments Router** | | | |
| POST /experiments | Unit | `test_create_experiment_requires_auth` | 401 without token |
| POST /experiments | Unit | `test_create_experiment_success` | 201 with experiment_id |
| GET /experiments | Unit | `test_list_experiments_filters_by_org` | Only returns org's experiments |
| GET /experiments/{id} | Unit | `test_get_experiment_not_found` | 404 for non-existent |
| PUT /experiments/{id}/status | Unit | `test_update_status_requires_write_scope` | 403 without EXPERIMENTS_WRITE |
| PUT /experiments/{id}/status | Unit | `test_update_status_records_history` | Status history updated |
| POST /experiments/{id}/assign | Unit | `test_assign_variant_sticky` | Same user gets same variant |
| GET /experiments/{id}/assignment | Unit | `test_get_assignment_not_found` | 404 for unassigned user |
| GET /experiments/{id}/history | Unit | `test_status_history_ordered` | Returns chronological history |
| **Search Router** | | | |
| POST /search | Unit | `test_search_requires_auth` | 401 without token |
| POST /search | Unit | `test_hybrid_search_filters_by_org` | Only returns org's documents |
| POST /search/lexical | Unit | `test_lexical_search_success` | Returns matching documents |
| POST /search/semantic | Unit | `test_semantic_requires_vector` | 400 without query_vector |
| **Scope Tests** | | | |
| Scope enum | Unit | `test_new_scopes_exist` | All 5 new scopes in enum |
| Scope enforcement | Unit | `test_feedback_scope_enforcement` | Routes check correct scopes |

---

## PHASE 2: Feature Specifications

### Feature 0: Fix Pre-existing Test Failures (PRIORITY 0)

**Description**: Fix 4 failing tests in `test_router_auth.py` before adding new routes.

**Dependencies**: None (blocking other work)

**Failing Tests**:
1. `test_delete_memory_requires_auth` - Wrong endpoint signature
2. `test_create_app_requires_auth` - No POST /apps endpoint
3. `test_export_requires_auth` - Needs investigation
4. `test_memories_delete_requires_scope` - Wrong endpoint signature

**Test Cases** (already exist, need fixing):
- [ ] Unit test: `test_delete_memory_requires_auth` - Investigate actual DELETE endpoint path
- [ ] Unit test: `test_create_app_requires_auth` - Verify POST /apps exists or update test
- [ ] Unit test: `test_export_requires_auth` - Verify /backup/export endpoint
- [ ] Unit test: `test_memories_delete_requires_scope` - Align with actual endpoint

**Implementation Approach**:
1. Run `test_router_auth.py` to see exact failures
2. Compare test expectations vs actual router endpoints
3. Either fix router or fix test based on intended behavior
4. Ensure all 4 tests pass

**Git Commit Message**: `fix(security): resolve pre-existing router auth test failures`

---

### Feature 1: MCP SSE Authentication (PRIORITY 0 - BLOCKING)

**Description**: Add JWT authentication to MCP SSE endpoints (`/mcp`, `/concepts`).

**Dependencies**: Security module (Phase 1 complete)

**Affected Files**:
- `openmemory/api/app/routers/mcp_server.py`
- `openmemory/api/tests/security/test_mcp_auth.py`

**Test Cases** (write these first):
- [ ] Unit test: `test_mcp_sse_endpoint_requires_auth` - 401 without Authorization header
- [ ] Unit test: `test_mcp_sse_rejects_invalid_token` - 401 with malformed/expired JWT
- [ ] Unit test: `test_mcp_sse_accepts_valid_token` - 200 with valid JWT
- [ ] Unit test: `test_concepts_endpoint_requires_auth` - 401 for /concepts
- [ ] Unit test: `test_mcp_tool_requires_scope` - 403 when token lacks required scope

**Implementation Approach**:
1. Add `require_scopes()` dependency to SSE endpoints
2. Add tool-level scope checks in MCP tool handlers
3. Return proper 401/403 responses

**Git Commit Message**: `feat(mcp): add JWT authentication to SSE endpoints`

---

### Feature 2: Add New Scopes (PRIORITY 4)

**Description**: Add new OAuth 2.0 scopes for feedback, experiments, and search resources.

**Dependencies**: Security types module exists

**Affected Files**:
- `openmemory/api/app/security/types.py`

**New Scopes**:
```python
class Scope(str, Enum):
    # ... existing scopes ...

    # Feedback operations
    FEEDBACK_READ = "feedback:read"
    FEEDBACK_WRITE = "feedback:write"

    # Experiment operations
    EXPERIMENTS_READ = "experiments:read"
    EXPERIMENTS_WRITE = "experiments:write"

    # Search operations (uses existing MEMORIES_READ for now)
    SEARCH_READ = "search:read"
```

**Test Cases** (write these first):
- [ ] Unit test: `test_feedback_scopes_exist` - FEEDBACK_READ and FEEDBACK_WRITE in enum
- [ ] Unit test: `test_experiments_scopes_exist` - EXPERIMENTS_READ and EXPERIMENTS_WRITE in enum
- [ ] Unit test: `test_search_scope_exists` - SEARCH_READ in enum
- [ ] Unit test: `test_scope_values_correct` - Scope values match pattern `resource:action`

**Implementation Approach**:
1. Add new enum members to `Scope` class
2. Follow existing naming convention (`RESOURCE_ACTION = "resource:action"`)

**Git Commit Message**: `feat(security): add feedback, experiments, and search scopes`

---

### Feature 3: Feedback Router (PRIORITY 1)

**Description**: REST API for feedback events with append, query, and metrics endpoints.

**Dependencies**:
- `PostgresFeedbackStore` (Phase 4 complete)
- New scopes (Feature 2)

**Affected Files**:
- `openmemory/api/app/routers/feedback.py` (NEW)
- `openmemory/api/app/routers/__init__.py` (export)
- `openmemory/api/main.py` (register)
- `openmemory/api/tests/routers/test_feedback_router.py` (NEW)

**Endpoints**:

| Method | Path | Scope | Description |
|--------|------|-------|-------------|
| POST | `/api/v1/feedback` | FEEDBACK_WRITE | Append feedback event |
| GET | `/api/v1/feedback` | FEEDBACK_READ | Query feedback by filters |
| GET | `/api/v1/feedback/metrics` | FEEDBACK_READ | Aggregate acceptance rate |
| GET | `/api/v1/feedback/by-tool` | FEEDBACK_READ | Metrics grouped by tool |

**Pydantic Schemas**:

```python
class FeedbackCreate(BaseModel):
    query_id: str
    memory_id: Optional[str] = None
    outcome: str = Field(..., pattern="^(accepted|rejected|modified|ignored)$")
    tool_name: Optional[str] = None
    experiment_id: Optional[str] = None
    decision_time_ms: Optional[int] = None
    metadata: dict = Field(default_factory=dict)

class FeedbackResponse(BaseModel):
    event_id: str
    query_id: str
    outcome: str
    user_id: str
    org_id: str
    tool_name: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class FeedbackMetrics(BaseModel):
    total_count: int
    accepted_count: int
    rejected_count: int
    modified_count: int
    ignored_count: int
    acceptance_rate: float  # accepted / total

class ToolMetrics(BaseModel):
    tool_name: str
    total_count: int
    acceptance_rate: float
```

**Test Cases** (25-30 tests):
- [ ] Unit: `test_create_feedback_requires_auth` - 401 without token
- [ ] Unit: `test_create_feedback_requires_write_scope` - 403 without FEEDBACK_WRITE
- [ ] Unit: `test_create_feedback_success` - 201 with valid data
- [ ] Unit: `test_create_feedback_validates_outcome` - 400 for invalid outcome
- [ ] Unit: `test_create_feedback_injects_user_org` - user_id/org_id from principal
- [ ] Unit: `test_query_feedback_requires_auth` - 401 without token
- [ ] Unit: `test_query_feedback_requires_read_scope` - 403 without FEEDBACK_READ
- [ ] Unit: `test_query_feedback_filters_by_org` - Only returns org's events
- [ ] Unit: `test_query_feedback_with_limit` - Respects limit param
- [ ] Unit: `test_query_feedback_with_since_until` - Time range filtering
- [ ] Unit: `test_query_feedback_by_query_id` - Filter by query_id
- [ ] Unit: `test_metrics_requires_auth` - 401 without token
- [ ] Unit: `test_metrics_returns_aggregates` - acceptance_rate, counts
- [ ] Unit: `test_metrics_empty_org` - Returns zeros, not error
- [ ] Unit: `test_metrics_with_time_range` - Filters by since/until
- [ ] Unit: `test_by_tool_requires_auth` - 401 without token
- [ ] Unit: `test_by_tool_groups_correctly` - Returns per-tool metrics
- [ ] Unit: `test_by_tool_empty_org` - Returns empty list
- [ ] Integration: `test_create_and_query_feedback` - Full round-trip
- [ ] Integration: `test_feedback_tenant_isolation` - Org A can't see Org B's data

**Implementation Approach**:
1. Create Pydantic schemas in `app/routers/feedback.py`
2. Create dependency for `PostgresFeedbackStore`
3. Implement 4 endpoints with scope requirements
4. Register router in `main.py`

**Git Commit Message**: `feat(api): add feedback router with 4 endpoints`

---

### Feature 4: Experiments Router (PRIORITY 2)

**Description**: REST API for A/B experiments with CRUD, status management, and variant assignment.

**Dependencies**:
- `PostgresExperimentStore` (Phase 4 complete)
- New scopes (Feature 2)

**Affected Files**:
- `openmemory/api/app/routers/experiments.py` (NEW)
- `openmemory/api/app/routers/__init__.py` (export)
- `openmemory/api/main.py` (register)
- `openmemory/api/tests/routers/test_experiments_router.py` (NEW)

**Endpoints**:

| Method | Path | Scope | Description |
|--------|------|-------|-------------|
| POST | `/api/v1/experiments` | EXPERIMENTS_WRITE | Create experiment |
| GET | `/api/v1/experiments` | EXPERIMENTS_READ | List org's experiments |
| GET | `/api/v1/experiments/{id}` | EXPERIMENTS_READ | Get experiment details |
| PUT | `/api/v1/experiments/{id}/status` | EXPERIMENTS_WRITE | Update status |
| POST | `/api/v1/experiments/{id}/assign` | EXPERIMENTS_WRITE | Assign user to variant |
| GET | `/api/v1/experiments/{id}/assignment` | EXPERIMENTS_READ | Get user's assignment |
| GET | `/api/v1/experiments/{id}/history` | EXPERIMENTS_READ | Get status history |

**Pydantic Schemas**:

```python
class VariantCreate(BaseModel):
    name: str
    weight: float = Field(..., ge=0.0, le=1.0)
    description: str = ""
    config: dict = Field(default_factory=dict)

class ExperimentCreate(BaseModel):
    name: str
    description: str = ""
    variants: List[VariantCreate] = Field(..., min_length=2)
    traffic_percentage: float = Field(1.0, ge=0.0, le=1.0)

class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    description: str
    status: str
    variants: List[dict]
    traffic_percentage: float
    created_at: datetime
    start_time: Optional[datetime]
    end_time: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(draft|running|paused|completed|rolled_back)$")
    reason: Optional[str] = None

class AssignmentResponse(BaseModel):
    experiment_id: str
    variant_id: str
    variant_name: str
    assigned_at: datetime
    variant_config: dict

class StatusHistoryEntry(BaseModel):
    status: str
    changed_at: datetime
    reason: Optional[str]
```

**Test Cases** (30-35 tests):
- [ ] Unit: `test_create_experiment_requires_auth` - 401 without token
- [ ] Unit: `test_create_experiment_requires_write_scope` - 403 without scope
- [ ] Unit: `test_create_experiment_success` - 201 with experiment_id
- [ ] Unit: `test_create_experiment_validates_variants` - 400 for < 2 variants
- [ ] Unit: `test_create_experiment_validates_weights` - 400 for invalid weights
- [ ] Unit: `test_list_experiments_requires_auth` - 401 without token
- [ ] Unit: `test_list_experiments_filters_by_org` - Only returns org's experiments
- [ ] Unit: `test_list_experiments_filter_by_status` - Status filter works
- [ ] Unit: `test_list_experiments_empty_org` - Returns empty list
- [ ] Unit: `test_get_experiment_requires_auth` - 401 without token
- [ ] Unit: `test_get_experiment_not_found` - 404 for non-existent
- [ ] Unit: `test_get_experiment_cross_tenant` - 404 for other org's experiment
- [ ] Unit: `test_get_experiment_success` - Returns full experiment details
- [ ] Unit: `test_update_status_requires_auth` - 401 without token
- [ ] Unit: `test_update_status_requires_write_scope` - 403 without scope
- [ ] Unit: `test_update_status_not_found` - 404 for non-existent
- [ ] Unit: `test_update_status_success` - Status updated
- [ ] Unit: `test_update_status_records_reason` - Reason stored in history
- [ ] Unit: `test_update_status_sets_start_time` - start_time set when RUNNING
- [ ] Unit: `test_update_status_sets_end_time` - end_time set when COMPLETED
- [ ] Unit: `test_assign_requires_auth` - 401 without token
- [ ] Unit: `test_assign_requires_write_scope` - 403 without scope
- [ ] Unit: `test_assign_not_found` - 404 for non-existent experiment
- [ ] Unit: `test_assign_not_running` - 400 for non-RUNNING experiment
- [ ] Unit: `test_assign_success` - Returns assignment details
- [ ] Unit: `test_assign_sticky` - Same user gets same variant on re-assign
- [ ] Unit: `test_get_assignment_requires_auth` - 401 without token
- [ ] Unit: `test_get_assignment_not_found` - 404 for unassigned user
- [ ] Unit: `test_get_assignment_success` - Returns assignment
- [ ] Unit: `test_history_requires_auth` - 401 without token
- [ ] Unit: `test_history_not_found` - 404 for non-existent experiment
- [ ] Unit: `test_history_ordered` - Returns chronological history
- [ ] Integration: `test_full_experiment_lifecycle` - Create → Run → Assign → Complete

**Implementation Approach**:
1. Create Pydantic schemas
2. Create dependency for `PostgresExperimentStore`
3. Implement 7 endpoints with scope requirements
4. Handle status transitions with validation
5. Register router in `main.py`

**Git Commit Message**: `feat(api): add experiments router with 7 endpoints`

---

### Feature 5: Search Router (PRIORITY 3)

**Description**: REST API for hybrid search using OpenSearch (lexical + vector).

**Dependencies**:
- `TenantOpenSearchStore` (Phase 4 complete)
- New scopes (Feature 2)

**Affected Files**:
- `openmemory/api/app/routers/search.py` (NEW)
- `openmemory/api/app/routers/__init__.py` (export)
- `openmemory/api/main.py` (register)
- `openmemory/api/tests/routers/test_search_router.py` (NEW)

**Endpoints**:

| Method | Path | Scope | Description |
|--------|------|-------|-------------|
| POST | `/api/v1/search` | SEARCH_READ | Hybrid search (lexical + vector) |
| POST | `/api/v1/search/lexical` | SEARCH_READ | Lexical-only search |
| POST | `/api/v1/search/semantic` | SEARCH_READ | Semantic-only search |

**Pydantic Schemas**:

```python
class SearchFilters(BaseModel):
    vault: Optional[str] = None
    layer: Optional[str] = None
    app_id: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)
    filters: Optional[SearchFilters] = None
    include_vector: bool = False  # For hybrid/semantic search

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    query_vector: List[float] = Field(..., min_length=1)  # Required for semantic
    limit: int = Field(10, ge=1, le=100)
    filters: Optional[SearchFilters] = None

class SearchResult(BaseModel):
    memory_id: str
    score: float
    content: str
    highlights: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    took_ms: int
```

**Test Cases** (20-25 tests):
- [ ] Unit: `test_search_requires_auth` - 401 without token
- [ ] Unit: `test_search_requires_scope` - 403 without SEARCH_READ
- [ ] Unit: `test_search_validates_query` - 400 for empty query
- [ ] Unit: `test_search_validates_limit` - 400 for limit > 100
- [ ] Unit: `test_search_filters_by_org` - Only returns org's documents
- [ ] Unit: `test_search_with_vault_filter` - Vault filter applied
- [ ] Unit: `test_search_with_layer_filter` - Layer filter applied
- [ ] Unit: `test_search_returns_results` - Results with score, content
- [ ] Unit: `test_search_returns_took_ms` - Timing included
- [ ] Unit: `test_lexical_requires_auth` - 401 without token
- [ ] Unit: `test_lexical_search_success` - Returns matching documents
- [ ] Unit: `test_lexical_empty_results` - Returns empty list, not error
- [ ] Unit: `test_semantic_requires_auth` - 401 without token
- [ ] Unit: `test_semantic_requires_vector` - 400 without query_vector
- [ ] Unit: `test_semantic_validates_vector_dims` - 400 for wrong dimensions
- [ ] Unit: `test_semantic_search_success` - Returns matching documents
- [ ] Unit: `test_search_opensearch_unavailable` - 503 when service down
- [ ] Integration: `test_search_tenant_isolation` - Org A can't search Org B's docs
- [ ] Integration: `test_hybrid_vs_lexical` - Hybrid returns better results

**Implementation Approach**:
1. Create Pydantic schemas
2. Create dependency for `TenantOpenSearchStore` with org_id from principal
3. Implement 3 search endpoints
4. Handle OpenSearch connection errors gracefully
5. Register router in `main.py`

**Git Commit Message**: `feat(api): add search router with hybrid/lexical/semantic endpoints`

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
   ├── docker compose exec codingbrain-mcp pytest tests/routers/test_{feature}_router.py -v
   ├── docker compose exec codingbrain-mcp pytest tests/security/ -v
   └── docker compose exec codingbrain-mcp pytest tests/ -v (full regression)

4. ON PASS:
   ├── git add -A
   ├── git commit -m "feat(scope): description"
   └── Update Agent Scratchpad below

5. ON FAIL:
   ├── Investigate failure
   ├── DO NOT proceed until green
   └── Return to step 3

6. REGRESSION VERIFICATION
   ├── Run full test suite after each feature
   └── Verify 3,204+ tests still pass

7. REPEAT for next feature
```

### Git Checkpoint Protocol

```bash
# After each passing feature
git add -A
git commit -m "type(scope): description"

# Feature commits:
# feat(security): add feedback, experiments, and search scopes
# feat(api): add feedback router with 4 endpoints
# feat(api): add experiments router with 7 endpoints
# feat(api): add search router with hybrid/lexical/semantic endpoints
# fix(security): resolve pre-existing router auth test failures
# feat(mcp): add JWT authentication to SSE endpoints
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-27
**Current Phase**: Phase 5 - API Route Wiring
**Last Action**: Created PRD document

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 0 | Fix pre-existing test failures | [ ] | [ ] | [ ] | |
| 1 | MCP SSE Authentication | [ ] | [ ] | [ ] | |
| 2 | Add New Scopes | [ ] | [ ] | [ ] | |
| 3 | Feedback Router | [ ] | [ ] | [ ] | |
| 4 | Experiments Router | [ ] | [ ] | [ ] | |
| 5 | Search Router | [ ] | [ ] | [ ] | |

### Decisions Made

1. **Decision**: Start with fixing pre-existing test failures before adding new features
   - **Rationale**: Clean baseline ensures we're not masking regressions
   - **Alternatives Considered**: Skip and fix later (rejected - too risky)

2. **Decision**: Use `SEARCH_READ` scope for all search endpoints
   - **Rationale**: Search is read-only; no write operations on search
   - **Alternatives Considered**: Reuse `MEMORIES_READ` (rejected - search may expand beyond memories)

### Attempted Approaches

| Approach | Outcome | Notes |
|----------|---------|-------|
| | | |

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Existing router patterns | Principal dependency, require_scopes, get_db, tenant isolation via user.id filter |
| Explore | Security dependencies | Principal dataclass with user_id/org_id, Scope enum, require_scopes factory |
| Explore | Store implementations | PostgresFeedbackStore/ExperimentStore use Session, TenantOpenSearchStore needs org_id |
| Explore | Router test patterns | JWT mocking, TestClient, scope enforcement tests, tenant isolation checks |

### Known Issues & Blockers

- [x] Issue: Pre-existing test failures in `test_router_auth.py` (4 tests)
  - Status: Identified - need investigation
  - Attempted solutions: None yet

- [x] Issue: MCP auth tests hang
  - Status: Identified - SSE auth architecture needed
  - Attempted solutions: None yet

### Notes for Next Session

> Continue from here in the next session:

- [ ] Run `test_router_auth.py` to see exact failure messages
- [ ] Investigate MCP SSE endpoint structure in `mcp_server.py`
- [ ] Add new scopes to `types.py` (quick win)
- [ ] Start with Feedback Router tests (highest value)

### Test Results Log

```
[Run tests and paste output here]
```

### Browser Test Results

N/A - Backend API only, no browser testing needed for Phase 5.

### Recent Git History

```
[Run `git log --oneline -5` and paste here]
```

---

## Execution Checklist

When executing this PRD, follow this order:

- [x] 1. Read Agent Scratchpad for prior context
- [x] 2. Spawn parallel Explore agents to understand codebase (DONE)
- [x] 3. Review/complete success criteria (Phase 1)
- [x] 4. Design test suite structure (Phase 1)
- [x] 5. Write feature specifications (Phase 2)
- [ ] 6. For each feature (in priority order):
  - [ ] **Feature 0**: Fix pre-existing test failures
    - [ ] Run failing tests
    - [ ] Investigate and fix
    - [ ] Verify all pass
    - [ ] Commit
  - [ ] **Feature 1**: MCP SSE Authentication
    - [ ] Write failing tests
    - [ ] Implement auth
    - [ ] Run tests
    - [ ] Commit
  - [ ] **Feature 2**: Add New Scopes
    - [ ] Write scope tests
    - [ ] Add scopes to enum
    - [ ] Run tests
    - [ ] Commit
  - [ ] **Feature 3**: Feedback Router
    - [ ] Write 25+ tests
    - [ ] Implement router
    - [ ] Register in main.py
    - [ ] Run tests + regression
    - [ ] Commit
  - [ ] **Feature 4**: Experiments Router
    - [ ] Write 30+ tests
    - [ ] Implement router
    - [ ] Register in main.py
    - [ ] Run tests + regression
    - [ ] Commit
  - [ ] **Feature 5**: Search Router
    - [ ] Write 20+ tests
    - [ ] Implement router
    - [ ] Register in main.py
    - [ ] Run tests + regression
    - [ ] Commit
- [ ] 7. Final regression test (all 3,300+ tests)
- [ ] 8. Tag milestone: `git tag -a v0.5.0 -m "Phase 5: API Route Wiring"`

---

## Quick Reference Commands

```bash
# Run all tests
docker compose exec codingbrain-mcp pytest tests/ -v

# Run router tests only
docker compose exec codingbrain-mcp pytest tests/routers/ -v

# Run security tests
docker compose exec codingbrain-mcp pytest tests/security/ -v

# Run specific test file
docker compose exec codingbrain-mcp pytest tests/routers/test_feedback_router.py -v

# Run with coverage
docker compose exec codingbrain-mcp pytest tests/ --cov=app --cov-report=term-missing

# Git workflow
git status
git add -A
git commit -m "feat(scope): description"
git log --oneline -5
```

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
