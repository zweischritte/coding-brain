# PRD: Phase 4 - Multi-tenant Data Plane Stores

**Plan Reference**: `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md`
**Progress Tracker**: `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md`
**Continuation Prompt**: `docs/CONTINUATION-PROMPT-PHASE4-MULTITENANT-STORES.md`
**Style**: STRICT TDD - Write failing tests first, then implement

---

## PHASE 1: Success Criteria & Test Design

### 1.1 Success Criteria

All criteria must pass before Phase 4 is considered complete:

1. [ ] **RLS Enforcement**: PostgreSQL Row Level Security policies prevent cross-tenant data access at the database level
2. [ ] **ScopedMemoryStore**: Full CRUD operations with RLS protection, passing 20+ tests
3. [ ] **FeedbackStore**: Persistent feedback events with retention queries, passing 15+ tests
4. [ ] **ExperimentStore**: Experiment/variant/assignment persistence with status history, passing 15+ tests
5. [ ] **EpisodicMemoryStore**: Valkey-backed session context with TTL and recency decay, passing 10+ tests
6. [ ] **Neo4j Stores**: CODE_* graph stores with org_id constraints, passing 15+ tests
7. [ ] **Qdrant EmbeddingStore**: Per-model collections with tenant payload index, passing 10+ tests
8. [ ] **OpenSearch Alias Strategy**: Tenant-aware index aliases documented and tested, passing 10+ tests
9. [ ] **Contract Tests**: ABC interface compliance verified for all store implementations, passing 20+ tests
10. [ ] **Zero Regression**: All 3,079 existing tests continue to pass

### 1.2 Edge Cases

Document edge cases that must be handled:

**RLS Edge Cases**:
- Session variable not set (must fail closed, not open)
- Malformed UUID in session variable
- Concurrent connections with different tenant contexts
- Connection pooling with stale tenant context
- RLS policy bypass attempts via raw SQL

**Store Edge Cases**:
- Empty result sets (return empty list, not error)
- Duplicate key insertions (handle gracefully)
- Optimistic locking conflicts on updates
- Large batch operations exceeding memory
- Network timeouts to external stores (Valkey, Neo4j, Qdrant, OpenSearch)
- Store unavailability (circuit breaker behavior)

**Tenant Isolation Edge Cases**:
- User with no org_id in JWT (reject at auth layer)
- Cross-tenant ID guessing attacks (return 404, not 403)
- Tenant migration/merge scenarios
- Orphaned data from deleted tenants

### 1.3 Test Suite Structure

```
openmemory/api/tests/
├── stores/                              # NEW: Store unit tests
│   ├── __init__.py
│   ├── conftest.py                      # Shared fixtures for stores
│   ├── test_memory_store.py             # ScopedMemoryStore tests
│   ├── test_feedback_store.py           # FeedbackStore tests
│   ├── test_experiment_store.py         # ExperimentStore tests
│   ├── test_episodic_store.py           # EpisodicMemoryStore tests
│   ├── test_neo4j_stores.py             # Neo4j store tests
│   ├── test_qdrant_store.py             # EmbeddingStore tests
│   ├── test_opensearch_store.py         # OpenSearch store tests
│   └── test_store_contracts.py          # ABC contract tests
├── integration/                          # NEW: Integration tests
│   ├── __init__.py
│   ├── test_rls_enforcement.py          # RLS policy integration tests
│   └── test_tenant_isolation_e2e.py     # End-to-end tenant isolation
└── security/
    └── test_tenant_isolation.py         # Existing (19 tests)
```

### 1.4 Test Specifications

#### Priority 1: ScopedMemoryStore + RLS

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| RLS Policy Creation | Integration | Verify RLS policy exists on memories table | Policy `tenant_isolation` applied |
| RLS Blocks Unset Context | Integration | Query without session var returns empty | Zero rows, not error |
| RLS Filters by User | Integration | Query with user_id var filters results | Only user's memories returned |
| RLS Prevents Cross-Tenant | Integration | User A cannot see User B's memories via RLS | Zero rows returned |
| tenant_session Context | Unit | Context manager sets and resets session var | Var set on enter, reset on exit |
| tenant_session Error Handling | Unit | Context resets var even on exception | RESET called in finally block |
| Store.get() | Unit | Get memory by ID with RLS | Returns memory or None |
| Store.list() | Unit | List memories with filters | Returns filtered list |
| Store.create() | Unit | Create new memory | Returns created memory with ID |
| Store.update() | Unit | Update existing memory | Returns updated memory |
| Store.delete() | Unit | Soft delete memory | Sets state to deleted |
| Store Cross-Tenant Get | Unit | Get other user's memory by ID | Returns None (RLS blocks) |
| Store Cross-Tenant Update | Unit | Update other user's memory | Raises error or no-op |
| Batch Operations | Unit | Create/update multiple memories | All succeed atomically |
| Connection Pool Context | Integration | Pooled connections get correct tenant | Each connection isolated |

#### Priority 2: Other PostgreSQL Stores

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| FeedbackStore.append() | Unit | Append feedback event | Event persisted |
| FeedbackStore.query_by_user() | Unit | Query events by user/org | Returns user's events |
| FeedbackStore.retention_query() | Unit | Query with time window | Returns events in window |
| FeedbackStore.aggregate() | Unit | Calculate acceptance rate | Returns correct percentage |
| ExperimentStore.create() | Unit | Create experiment with variants | Experiment and variants created |
| ExperimentStore.assign_user() | Unit | Assign user to variant | Assignment recorded |
| ExperimentStore.get_assignment() | Unit | Get user's assignment | Returns variant or None |
| ExperimentStore.status_history() | Unit | Track status changes | History preserved |

#### Priority 3: External Stores

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| EpisodicStore.set() | Unit | Store session context with TTL | Data stored with expiry |
| EpisodicStore.get() | Unit | Retrieve session context | Returns data or None |
| EpisodicStore.decay() | Unit | Apply recency decay | Older items deprioritized |
| Neo4j.create_node() | Unit | Create CODE_* node with org_id | Node created with constraint |
| Neo4j.query_with_tenant() | Unit | Query enforces org_id filter | Only tenant's nodes returned |
| Qdrant.upsert() | Unit | Store embedding with tenant payload | Point stored with org_id |
| Qdrant.search() | Unit | Search with tenant filter | Only tenant's points returned |
| OpenSearch.alias_strategy() | Unit | Create tenant alias | Alias points to tenant index |
| OpenSearch.search_with_alias() | Unit | Search via alias | Only tenant's docs returned |

#### Priority 4: Contract Tests

| Feature | Test Type | Test Description | Expected Outcome |
|---------|-----------|------------------|------------------|
| MemoryStore ABC Compliance | Contract | All ABC methods implemented | No abstract methods remaining |
| FeedbackStore ABC Compliance | Contract | All ABC methods implemented | Interface contract satisfied |
| ExperimentStore ABC Compliance | Contract | All ABC methods implemented | Interface contract satisfied |
| Store Type Hints | Contract | Return types match ABC | Type checker passes |
| Store Error Handling | Contract | Exceptions match ABC spec | Expected exceptions raised |

---

## PHASE 2: Feature Specifications

### Feature 1: PostgreSQL RLS Infrastructure

**Description**: Enable Row Level Security on PostgreSQL tables with session-based tenant context injection.

**Dependencies**:
- PostgreSQL 16+ with RLS support (Phase 0.5 complete)
- Alembic migrations (Phase 3 complete)
- Principal extraction with user_id (Phase 1 complete)

**Test Cases** (write these first):
- [ ] Unit: `tenant_session()` context manager sets `app.current_user_id` session variable
- [ ] Unit: `tenant_session()` resets variable on normal exit
- [ ] Unit: `tenant_session()` resets variable on exception
- [ ] Integration: RLS policy on `memories` table blocks queries without session var
- [ ] Integration: RLS policy returns only rows matching session var
- [ ] Integration: Connection pool correctly isolates tenant contexts

**Implementation Approach**:
1. Create Alembic migration to enable RLS on `memories`, `apps`, `feedback_events` tables
2. Create RLS policies using `current_setting('app.current_user_id')::uuid`
3. Implement `tenant_session()` context manager in `database.py`
4. Update `get_db()` dependency to optionally inject tenant context

**Files to Create/Modify**:
- `openmemory/api/alembic/versions/xxx_add_rls_policies.py` - RLS migration
- `openmemory/api/app/database.py` - Add `tenant_session()` context manager
- `openmemory/api/tests/stores/conftest.py` - Test fixtures
- `openmemory/api/tests/integration/test_rls_enforcement.py` - RLS tests

**Git Commit Message**: `feat(rls): add PostgreSQL Row Level Security for tenant isolation`

---

### Feature 2: BaseStore ABC and ScopedMemoryStore

**Description**: Define abstract base class for all stores and implement ScopedMemoryStore with RLS.

**Dependencies**:
- Feature 1: RLS Infrastructure
- Existing Memory model in `models.py`

**Test Cases** (write these first):
- [ ] Unit: `ScopedMemoryStore.get(id)` returns memory or None
- [ ] Unit: `ScopedMemoryStore.list()` returns list of memories
- [ ] Unit: `ScopedMemoryStore.list(state=active)` filters by state
- [ ] Unit: `ScopedMemoryStore.list(app_id=x)` filters by app
- [ ] Unit: `ScopedMemoryStore.create(memory)` persists and returns with ID
- [ ] Unit: `ScopedMemoryStore.update(memory)` updates and returns
- [ ] Unit: `ScopedMemoryStore.delete(id)` soft-deletes (sets state)
- [ ] Unit: Cross-tenant get returns None (RLS blocks)
- [ ] Unit: Cross-tenant list returns empty list (RLS blocks)
- [ ] Contract: ScopedMemoryStore implements all BaseStore methods

**Implementation Approach**:
1. Define `BaseStore[T]` ABC in `stores/base.py`
2. Implement `ScopedMemoryStore` extending `BaseStore[Memory]`
3. Use `tenant_session()` for all database operations
4. Return None/empty for cross-tenant attempts (RLS enforcement)

**Files to Create**:
- `openmemory/api/app/stores/__init__.py`
- `openmemory/api/app/stores/base.py` - BaseStore ABC
- `openmemory/api/app/stores/memory_store.py` - ScopedMemoryStore
- `openmemory/api/tests/stores/test_memory_store.py` - Tests

**Git Commit Message**: `feat(stores): add BaseStore ABC and ScopedMemoryStore with RLS`

---

### Feature 3: FeedbackStore (PostgreSQL)

**Description**: Persistent feedback event storage with retention and aggregation queries.

**Dependencies**:
- Feature 1: RLS Infrastructure
- Feature 2: BaseStore ABC

**Test Cases** (write these first):
- [ ] Unit: `FeedbackStore.append(event)` persists event
- [ ] Unit: `FeedbackStore.query_by_user(user_id, org_id)` returns user's events
- [ ] Unit: `FeedbackStore.query_by_time_range(start, end)` filters by time
- [ ] Unit: `FeedbackStore.aggregate_acceptance_rate(org_id)` calculates rate
- [ ] Unit: `FeedbackStore.aggregate_by_query_type(org_id)` groups metrics
- [ ] Unit: Retention query excludes events older than threshold
- [ ] Unit: Cross-tenant query returns empty (RLS blocks)
- [ ] Contract: FeedbackStore implements required interface

**Implementation Approach**:
1. Create `FeedbackEvent` SQLAlchemy model with RLS-ready schema
2. Create Alembic migration for `feedback_events` table with RLS
3. Implement `FeedbackStore` with aggregation methods
4. Port logic from existing `InMemoryFeedbackStore`

**Files to Create**:
- `openmemory/api/app/stores/feedback_store.py`
- `openmemory/api/tests/stores/test_feedback_store.py`
- `openmemory/api/alembic/versions/xxx_add_feedback_events.py`

**Git Commit Message**: `feat(stores): add persistent FeedbackStore with retention queries`

---

### Feature 4: ExperimentStore (PostgreSQL)

**Description**: Persistent experiment, variant, and assignment storage with status history.

**Dependencies**:
- Feature 1: RLS Infrastructure
- Feature 2: BaseStore ABC

**Test Cases** (write these first):
- [ ] Unit: `ExperimentStore.create_experiment(exp)` creates with variants
- [ ] Unit: `ExperimentStore.get_experiment(id)` returns experiment or None
- [ ] Unit: `ExperimentStore.list_experiments(org_id)` returns org's experiments
- [ ] Unit: `ExperimentStore.assign_user(exp_id, user_id)` assigns to variant
- [ ] Unit: `ExperimentStore.get_assignment(exp_id, user_id)` returns assignment
- [ ] Unit: `ExperimentStore.update_status(exp_id, status)` updates with history
- [ ] Unit: `ExperimentStore.get_status_history(exp_id)` returns history
- [ ] Unit: Assignment is deterministic (same user, same variant)
- [ ] Contract: ExperimentStore implements required interface

**Implementation Approach**:
1. Create `Experiment`, `Variant`, `Assignment`, `StatusHistory` models
2. Create Alembic migration with RLS policies
3. Implement `ExperimentStore` with deterministic assignment logic
4. Port logic from existing `InMemoryExperimentStore`

**Files to Create**:
- `openmemory/api/app/stores/experiment_store.py`
- `openmemory/api/tests/stores/test_experiment_store.py`
- `openmemory/api/alembic/versions/xxx_add_experiments.py`

**Git Commit Message**: `feat(stores): add persistent ExperimentStore with status history`

---

### Feature 5: EpisodicMemoryStore (Valkey)

**Description**: Session-scoped ephemeral memory with TTL and recency decay in Valkey.

**Dependencies**:
- Valkey 8+ (Phase 0.5 complete)
- Principal with user_id and session_id

**Test Cases** (write these first):
- [ ] Unit: `EpisodicStore.set(key, value, ttl)` stores with expiry
- [ ] Unit: `EpisodicStore.get(key)` returns value or None
- [ ] Unit: `EpisodicStore.get(expired_key)` returns None
- [ ] Unit: `EpisodicStore.list_session(session_id)` returns session items
- [ ] Unit: `EpisodicStore.decay(factor)` applies recency weighting
- [ ] Unit: Keys are scoped by user_id (tenant isolation)
- [ ] Unit: Connection failure raises appropriate error
- [ ] Contract: EpisodicStore implements required interface

**Implementation Approach**:
1. Define `EpisodicMemoryStore` interface
2. Implement using Valkey client with TTL support
3. Use key prefix `episodic:{user_id}:{session_id}:` for isolation
4. Implement decay via sorted sets with timestamps

**Files to Create**:
- `openmemory/api/app/stores/episodic_store.py`
- `openmemory/api/tests/stores/test_episodic_store.py`

**Git Commit Message**: `feat(stores): add EpisodicMemoryStore with Valkey TTL`

---

### Feature 6: Neo4j Stores (CODE_* Graph)

**Description**: Neo4j stores for code graph with org_id constraints on all queries.

**Dependencies**:
- Neo4j (existing)
- Principal with org_id

**Test Cases** (write these first):
- [ ] Unit: `SymbolStore.create(symbol)` creates node with org_id
- [ ] Unit: `SymbolStore.find(name, org_id)` returns matching symbols
- [ ] Unit: `SymbolStore.find(name, other_org)` returns empty (isolation)
- [ ] Unit: `RegistryStore.register(repo)` creates CODE_REPO node
- [ ] Unit: `DependencyStore.add_dependency(from, to)` creates edge
- [ ] Unit: `DependencyStore.get_dependencies(node, org_id)` returns deps
- [ ] Unit: All queries include `WHERE org_id = $org_id` clause
- [ ] Contract: Neo4j stores implement required interfaces

**Implementation Approach**:
1. Define ABC interfaces for Symbol, Registry, Dependency stores
2. Implement with Cypher wrappers that enforce org_id filtering
3. Use parameterized queries to prevent injection
4. Validate org_id before all operations

**Files to Create**:
- `openmemory/api/app/stores/neo4j/__init__.py`
- `openmemory/api/app/stores/neo4j/symbol_store.py`
- `openmemory/api/app/stores/neo4j/registry_store.py`
- `openmemory/api/app/stores/neo4j/dependency_store.py`
- `openmemory/api/tests/stores/test_neo4j_stores.py`

**Git Commit Message**: `feat(stores): add Neo4j stores with org_id tenant constraints`

---

### Feature 7: Qdrant EmbeddingStore

**Description**: Vector embeddings with per-model collections and tenant payload filtering.

**Dependencies**:
- Qdrant (Phase 0.5 complete)
- Principal with org_id

**Test Cases** (write these first):
- [ ] Unit: `EmbeddingStore.upsert(id, vector, payload)` stores with org_id
- [ ] Unit: `EmbeddingStore.search(query, org_id, limit)` returns matches
- [ ] Unit: `EmbeddingStore.search(query, other_org)` returns empty (isolation)
- [ ] Unit: `EmbeddingStore.delete(id, org_id)` removes point
- [ ] Unit: Payload index exists for org_id field
- [ ] Unit: Per-model collection naming convention enforced
- [ ] Contract: EmbeddingStore implements required interface

**Implementation Approach**:
1. Define `EmbeddingStore` ABC interface
2. Implement with Qdrant client, adding org_id to all payloads
3. Create payload index on org_id field for efficient filtering
4. Use collection naming: `embeddings_{model_name}`

**Files to Create**:
- `openmemory/api/app/stores/qdrant_store.py`
- `openmemory/api/tests/stores/test_qdrant_store.py`

**Git Commit Message**: `feat(stores): add Qdrant EmbeddingStore with tenant payload index`

---

### Feature 8: OpenSearch Tenant Alias Strategy

**Description**: Tenant-aware index aliases for OpenSearch with optional dedicated indices.

**Dependencies**:
- OpenSearch (existing)
- Principal with org_id

**Test Cases** (write these first):
- [ ] Unit: `SearchStore.create_tenant_alias(org_id)` creates alias
- [ ] Unit: `SearchStore.search(query, org_id)` uses tenant alias
- [ ] Unit: `SearchStore.search(query, other_org)` isolated (different alias)
- [ ] Unit: Large tenant gets dedicated index
- [ ] Unit: Alias points to correct underlying index
- [ ] Unit: Cross-alias search returns empty
- [ ] Contract: SearchStore implements required interface

**Implementation Approach**:
1. Define `SearchStore` ABC interface
2. Implement with OpenSearch client using alias-based routing
3. Create alias: `search_tenant_{org_id}` pointing to shared or dedicated index
4. Document when to use dedicated indices (compliance, scale)

**Files to Create**:
- `openmemory/api/app/stores/opensearch_store.py`
- `openmemory/api/tests/stores/test_opensearch_store.py`
- `docs/OPENSEARCH-TENANT-STRATEGY.md` - Architecture documentation

**Git Commit Message**: `feat(stores): add OpenSearch tenant alias strategy`

---

### Feature 9: Store Contract Test Suite

**Description**: Contract tests verifying all store implementations satisfy ABC interfaces.

**Dependencies**:
- All store implementations (Features 2-8)

**Test Cases** (write these first):
- [ ] Contract: All stores have required methods
- [ ] Contract: Return types match interface definitions
- [ ] Contract: Exceptions are of expected types
- [ ] Contract: None returns for missing entities (not exceptions)
- [ ] Contract: Empty lists for no-match queries (not None)
- [ ] Parametrized: Run same tests against all store implementations

**Implementation Approach**:
1. Create parametrized test suite that runs against all store types
2. Use ABC inspection to verify method signatures
3. Create test base class for common store behaviors
4. Document contract expectations in docstrings

**Files to Create**:
- `openmemory/api/tests/stores/test_store_contracts.py`

**Git Commit Message**: `test(stores): add contract test suite for store ABC compliance`

---

## PHASE 3: Development Protocol

### The Recursive Testing Loop

Execute this loop for EVERY feature:

```
1. WRITE TESTS FIRST
   └── Create failing tests in tests/stores/ or tests/integration/

2. RUN TESTS (expect failures)
   └── docker compose exec codingbrain-mcp pytest tests/stores/test_X.py -v

3. IMPLEMENT FEATURE
   └── Write minimum code to pass tests

4. RUN ALL TESTS
   ├── docker compose exec codingbrain-mcp pytest tests/stores/ -v
   ├── docker compose exec codingbrain-mcp pytest tests/security/ -v
   └── docker compose exec codingbrain-mcp pytest tests/ -v (full regression)

5. ON PASS:
   ├── git add -A
   ├── git commit -m "feat(stores): [description]"
   └── Update Agent Scratchpad below

6. ON FAIL:
   ├── Debug failure
   ├── DO NOT proceed until green
   └── Return to step 3

7. REGRESSION VERIFICATION
   ├── Run full test suite (3,079+ tests)
   └── If regression: fix before continuing
```

### Git Checkpoint Protocol

```bash
# After each passing feature
git add -A
git commit -m "feat(stores): [description]"

# Tag milestones
git tag -a v0.4.0-stores -m "Phase 4: Multi-tenant stores complete"

# Before risky changes (RLS migration)
git stash
# or create branch
git checkout -b experiment/rls-policies
```

---

## PHASE 4: Agent Scratchpad

### Current Session Context

**Date Started**: 2025-12-27
**Current Phase**: Phase 4 - Multi-tenant Data Plane Stores
**Last Action**: Implemented RLS Infrastructure and ScopedMemoryStore with 25 TDD tests

### Implementation Progress Tracker

| # | Feature | Tests Written | Tests Passing | Committed | Commit Hash |
|---|---------|---------------|---------------|-----------|-------------|
| 1 | RLS Infrastructure | [x] 7+13 | [x] 7 (13 skipped) | [ ] | pending |
| 2 | BaseStore + ScopedMemoryStore | [x] 16+2 | [x] 18 | [ ] | pending |
| 3 | FeedbackStore | [ ] | [ ] | [ ] | |
| 4 | ExperimentStore | [ ] | [ ] | [ ] | |
| 5 | EpisodicMemoryStore | [ ] | [ ] | [ ] | |
| 6 | Neo4j Stores | [ ] | [ ] | [ ] | |
| 7 | Qdrant EmbeddingStore | [ ] | [ ] | [ ] | |
| 8 | OpenSearch Alias | [ ] | [ ] | [ ] | |
| 9 | Contract Tests | [x] 2 | [x] 2 | [ ] | pending |

### Decisions Made

1. **Decision**: Use `user_id` for RLS (not `org_id`) initially
   - **Rationale**: Current schema uses `user_id` on Memory/App tables; org_id not yet in database
   - **Alternatives Considered**: Add org_id migration first, but that's a larger change

2. **Decision**: Soft delete via state enum instead of hard delete
   - **Rationale**: Existing Memory model uses `MemoryState.deleted` pattern
   - **Alternatives Considered**: Hard delete with audit log

3. **Decision**: Mock external stores in unit tests
   - **Rationale**: Tests should be fast and not require running services
   - **Alternatives Considered**: Docker-based integration tests (for later)

### Attempted Approaches

| Approach | Outcome | Notes |
|----------|---------|-------|
| | | |

### Sub-Agent Results Log

| Agent Type | Query | Key Findings |
|------------|-------|--------------|
| Explore | Test structure patterns | No conftest.py at root; use local fixtures; mock DB pattern |
| Explore | Model and database patterns | User-based tenancy via user_id; no formal store layer yet |
| Explore | RLS/tenant isolation | App-level isolation complete; RLS policies not yet created |

### Known Issues & Blockers

- [ ] Issue: MCP auth tests hang
  - Status: Blocked (from Phase 1)
  - Attempted solutions: Deferred to later

- [ ] Issue: Pre-existing test failures in test_router_auth.py
  - Status: Open (4 tests fail)
  - Attempted solutions: Wrong endpoint signatures; needs investigation

### Notes for Next Session

> Continue from here in the next session:

- [x] Create `tests/stores/conftest.py` with shared fixtures
- [x] Write failing tests for `tenant_session()` context manager
- [x] Write failing tests for RLS policy enforcement
- [x] Implement RLS migration and `tenant_session()`
- [x] Implement ScopedMemoryStore
- [ ] Implement FeedbackStore with retention queries
- [ ] Implement ExperimentStore with status history
- [ ] Run RLS integration tests on PostgreSQL

### Test Results Log

```
tests/stores/ - 25 tests (23 passed, 2 skipped)
tests/security/ - 131 tests passed
```

### Recent Git History

```
97d8b053 docs: add Phase 4 continuation prompt
96c03518 feat(alembic): add migration verification utilities
1a0803d1 docs: add continuation prompt template for future phases
83b1f056 docs: reduce continuation prompt by 75% to avoid context duplication
b71b16a9 docs: fix plan coherence - Phase 1 MCP and Phase 3 Alembic incomplete
```

---

## Execution Checklist

When executing this PRD, follow this order:

- [x] 1. Read Agent Scratchpad for prior context
- [x] 2. Spawn parallel Explore agents to understand codebase
- [x] 3. Review/complete success criteria (Phase 1)
- [x] 4. Design test suite structure (Phase 1)
- [x] 5. Write feature specifications (Phase 2)
- [ ] 6. For each feature (in priority order):
  - [ ] Create test file with failing tests
  - [ ] Implement feature
  - [ ] Run tests until green
  - [ ] Commit on green
  - [ ] Run regression tests
  - [ ] Update scratchpad
- [ ] 7. Tag milestone when complete

---

## Quick Reference Commands

```bash
# Run store tests
docker compose exec codingbrain-mcp pytest tests/stores/ -v

# Run security tests (includes tenant isolation)
docker compose exec codingbrain-mcp pytest tests/security/ -v

# Run all tests (regression)
docker compose exec codingbrain-mcp pytest tests/ -v

# Run migrations
docker compose exec codingbrain-mcp alembic upgrade head

# Check migration status
docker compose exec codingbrain-mcp alembic current

# Generate new migration
docker compose exec codingbrain-mcp alembic revision --autogenerate -m "description"

# PostgreSQL shell (for RLS testing)
docker compose exec postgres psql -U postgres -d openmemory

# Git workflow
git status
git add -A
git commit -m "feat(stores): description"
git log --oneline -5

# Valkey shell (for episodic store testing)
docker compose exec valkey valkey-cli
```

---

## Key Files Reference

**Existing (read first):**
- [models.py](openmemory/api/app/models.py) - Memory, User, App, Category models
- [database.py](openmemory/api/app/database.py) - SessionLocal, get_db(), PostgreSQL support
- [test_tenant_isolation.py](openmemory/api/tests/security/test_tenant_isolation.py) - 19 existing tests
- [utils.py](openmemory/api/app/alembic/utils.py) - MigrationVerifier, BatchMigrator, RollbackManager

**To Create:**
- `openmemory/api/app/stores/` - Store implementations directory
- `openmemory/api/tests/stores/` - Store tests directory
- `openmemory/api/alembic/versions/xxx_add_rls_policies.py` - RLS migration

---

**Remember**: Tests define behavior. Write them first. Commit on green. Never skip regression tests.
