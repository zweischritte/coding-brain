# Continuation Prompt: Multi-User Memory Routing (access_entity)

## Objective
Implement multi-user memory routing using `access_entity` and JWT grants, following a TDD workflow.

## Current State (2025-12-29)

### Completed
- **Phase 2: Metadata + Validation** (DONE)
  - `access_entity` validation with format `<prefix>:<value>`
  - Valid prefixes: user, team, project, org, client, service
  - `scope!=user` requires `access_entity`
  - Scope/prefix consistency enforcement (e.g., scope=team requires team:* prefix)

- **Phase 3: Access Control Core Logic** (DONE)
  - JWT grants parsing into TokenClaims
  - Implied `user:<sub>` grant always included
  - `Principal.can_access()` with hierarchy expansion:
    - `org:X` -> `org:X`, `project:X/*`, `team:X/*`, `client:X/*`
    - `project:X` -> `project:X`, `team:X/*`
    - `team:X` -> `team:X` only
  - Access helpers in `app/security/access.py`

### Not Started
- Phase 3 integration (REST endpoints, MCP tools)
- Phase 4 retrieval backend alignment (Qdrant, Graph, OpenSearch)
- Phase 5 UI filters (optional)
- Phase 6 optional hardening

## Decisions
- Access control field is `access_entity`; `entity` stays semantic.
- Write policy for shared memories is group-editable (audit logs/backups for recovery).

## Required References
- Plan: `docs/plans/multi-user-memory-routing.md`
- Progress: `docs/plans/multi-user-memory-routing-progress.md`

## Files Changed (in previous session)
- `openmemory/api/app/utils/structured_memory.py` - access_entity validation
- `openmemory/api/app/security/types.py` - grants in TokenClaims, can_access() in Principal
- `openmemory/api/app/security/jwt.py` - grants parsing from JWT
- `openmemory/api/app/security/access.py` (NEW) - access control helpers
- `openmemory/api/tests/test_access_entity_validation.py` (NEW) - 33 tests
- `openmemory/api/tests/security/test_jwt_grants.py` (NEW) - 17 tests
- `openmemory/api/tests/test_access_entity_filtering.py` (NEW) - 38 tests
- `openmemory/api/tests/test_structured_memory.py` - added access_entity to shared scope tests

## Files Still To Change
- `openmemory/api/app/routers/memories.py` - integrate access control
- `openmemory/api/app/mcp_server.py` - integrate access control
- `openmemory/api/scripts/generate_jwt.py` - add `--grants` flag
- `openmemory/api/app/stores/qdrant_store.py` - filter by access_entity
- `openmemory/api/app/graph/metadata_projector.py` - filter by access_entity
- `openmemory/api/app/graph/graph_ops.py` - filter by access_entity
- `openmemory/api/app/stores/opensearch_store.py` - filter by access_entity

## Next TDD Tasks
1. **REST endpoint integration** (`routers/memories.py`)
   - Add tests to `test_access_entity_filtering.py` (TestListMemoriesFiltering, etc.)
   - Implement filtering in list/get/filter/related endpoints
   - Use `principal.can_access()` to check memory access
   - Build query filters using `principal.get_allowed_access_entities()`

2. **MCP tools integration** (`mcp_server.py`)
   - Implement tests in TestMCPAddMemoriesEnforcement, TestMCPUpdateMemoryEnforcement, TestMCPDeleteMemoriesEnforcement, TestMCPSearchMemoryFiltering
   - Enforce access control in add_memories, update_memory, delete_memories, search_memory
   - Use `can_write_to_access_entity()` for write operations

3. **generate_jwt.py update**
   - Add `--grants` CLI flag to specify grants in generated JWTs

4. **Qdrant store filtering**
   - Update `search_memory` to filter by allowed `access_entity` values
   - Use OR filter for all accessible access_entities

5. **Graph query filtering**
   - Update graph_ops to filter by allowed `access_entity`

6. **OpenSearch filtering**
   - Update opensearch_store to filter by allowed `access_entity`

## Key Code Patterns to Use

### Checking access to a memory
```python
from app.security.access import can_read_access_entity, can_write_to_access_entity

# For read operations
if not can_read_access_entity(principal, memory.metadata_.get("access_entity")):
    return None  # or 404

# For write operations
if not can_write_to_access_entity(principal, memory.metadata_.get("access_entity")):
    raise HTTPException(403, "Forbidden")
```

### Building query filters
```python
# Get all access_entities user can access
allowed = principal.get_allowed_access_entities()

# For SQL: WHERE access_entity IN (allowed) OR use hierarchy matching
# For Qdrant: use should conditions with match filters
```

### Validating on create
```python
from app.utils.structured_memory import build_structured_memory

# This validates access_entity format and scope/prefix consistency
text, metadata = build_structured_memory(
    text=content,
    category=category,
    scope=scope,
    access_entity=access_entity,  # Required if scope != user
    ...
)

# Then check principal has grant for the access_entity
if not principal.can_access(access_entity):
    raise HTTPException(403, "Cannot create memory with this access_entity")
```

## Running Tests
```bash
cd openmemory/api
docker compose exec api pytest tests/ -v
# Or specific test files:
docker compose exec api pytest tests/test_access_entity_filtering.py -v
```

## Deliverables
- Code changes + tests
- Updated `docs/plans/multi-user-memory-routing-progress.md`
- Mark completed items in progress file

## Change Checklist
- [x] Phase 2: access_entity validation (DONE)
- [x] Phase 3 core: JWT grants + Principal.can_access() (DONE)
- [ ] Phase 3 integration: REST endpoints
- [ ] Phase 3 integration: MCP tools
- [ ] generate_jwt.py --grants flag
- [ ] Phase 4: Qdrant filtering
- [ ] Phase 4: Graph filtering
- [ ] Phase 4: OpenSearch filtering
- [ ] Updated docs/progress tracking
