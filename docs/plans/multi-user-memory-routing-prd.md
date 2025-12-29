# PRD: Multi-User Memory Routing (access_entity)

## Context
This PRD captures the remaining work to make shared memories visible to all grant holders in REST and MCP, using `access_entity` for access control. It also removes `client:` and `service:` access_entity prefixes as requested.

Repo root: `/Users/grischadallmer/git/coding-brain`

Related docs:
- Plan: `docs/plans/multi-user-memory-routing.md`
- Progress: `docs/plans/multi-user-memory-routing-progress.md`
- Continuation: `docs/plans/multi-user-memory-routing-continuation.md`

## Decisions
- Shared memories ARE visible to all grant holders in REST list/filter/search/related and MCP search.
- Write policy: group-editable (any member of access_entity can update/delete).
- Remove `client:` and `service:` access_entity prefixes.

## Problem Statement (current issues)
The implementation is partially complete but still behaves like single-user in key paths:
- REST list/filter/related queries still filter by `Memory.user_id`, so shared memories are excluded.
- MCP search still filters Qdrant by `user_id`, so shared memories from other users never appear.
- MCP update/delete still enforce `Memory.user_id == owner`, blocking group-editable writes.
- REST search endpoints do not filter by access_entity; they only scope by org_id.
- Graph queries are still filtered by `userId`, excluding shared memories created by other users.

## Goals
1. Shared memories are visible to all grant holders in:
   - REST list/filter/search/related
   - MCP search
2. Group-editable writes for shared memories in REST and MCP.
3. Access control is enforced consistently by access_entity, with legacy fallback.
4. Remove `client:` and `service:` access_entity prefixes.

## Non-Goals
- External IdP integration.
- Full IAM or role management beyond JWT grants.
- Encrypting memory content.

## Access Model Summary
- `access_entity` is required for shared scopes (team/project/org/enterprise).
- `access_entity` is optional for user/session and defaults to `user:<sub>`.
- Hierarchy rules (already in `Principal.can_access`):
  - org grant -> org + project/team under org
  - project grant -> project + teams under project
  - team grant -> team only
  - user grant -> user only

## Required Behavior (Functional)
1. **REST list/filter/related**: return all memories accessible via access_entity grants, not just creator-owned.
2. **REST search**: filter results by access_entity; respect hierarchy for org/project grants.
3. **MCP search**: do not restrict to creator user_id; filter by access_entity grants.
4. **MCP update/delete**: allow any grant holder of access_entity to update/delete.
5. **Create**: enforce access_entity write permission; auto-default user scope.
6. **Legacy**: memories without access_entity are visible only to the owner.

## Implementation Plan (with code fragments)

### 1) Remove `client:` and `service:` prefixes
File: `openmemory/api/app/utils/structured_memory.py`

```python
VALID_ACCESS_ENTITY_PREFIXES = [
    "user",
    "team",
    "project",
    "org",
]
```

Update tests in `openmemory/api/tests/test_access_entity_validation.py` to remove client/service cases.

### 2) REST list/filter/related should include shared memories
File: `openmemory/api/app/routers/memories.py`

Use grant patterns rather than `Memory.user_id == user.id`.

```python
from sqlalchemy import or_, and_
from app.security.access import build_access_entity_patterns

exact, like_patterns = build_access_entity_patterns(principal)
access_entity_col = cast(Memory.metadata_["access_entity"], String)
legacy_owner_filter = and_(
    or_(Memory.metadata_.is_(None), access_entity_col.is_(None)),
    Memory.user_id == user.id
)
access_filter = or_(
    access_entity_col.in_(exact),
    *[access_entity_col.like(p) for p in like_patterns],
    legacy_owner_filter,
)
query = query.filter(access_filter)
```

Apply in:
- `list_memories`
- `filter_memories`
- `get_related_memories`
- `get_categories` (should only include accessible memories)

### 3) REST create must enforce access_entity grants
File: `openmemory/api/app/routers/memories.py`

```python
from app.security.access import check_create_access, get_default_access_entity

# Default for user/session scope
if normalized_metadata.get("scope") in ("user", "session") and "access_entity" not in normalized_metadata:
    normalized_metadata["access_entity"] = get_default_access_entity(principal)

access_entity = normalized_metadata.get("access_entity")
if access_entity and not check_create_access(principal, access_entity):
    raise HTTPException(status_code=403, detail="Cannot create memory for this access_entity")
```

### 4) REST search must filter by access_entity
File: `openmemory/api/app/stores/opensearch_store.py`

Expand filters to support prefix matching for org/project grants.

```python
def _create_access_entity_filter(self, exact: list[str], prefixes: list[str], additional_filters=None):
    filter_clauses = [self._create_org_filter()]
    should = []
    should.extend({"term": {"access_entity": ae}} for ae in exact)
    should.extend({"prefix": {"access_entity": p}} for p in prefixes)
    if should:
        filter_clauses.append({"bool": {"should": should, "minimum_should_match": 1}})
    ...
```

File: `openmemory/api/app/routers/search.py`

```python
from app.security.access import build_access_entity_patterns

exact, prefixes = build_access_entity_patterns(principal)
hits = store.search_with_access_control(
    query_text=request.query,
    limit=request.limit,
    access_entities=exact,
    access_entity_prefixes=prefixes,
    filters=filters_dict,
)
```

If the store API does not support prefixes yet, update it accordingly.

### 5) MCP search must not restrict to creator user_id
File: `openmemory/api/app/mcp_server.py`

Replace `filters={"user_id": uid}` with access_entity-aware filters. If vector store cannot do prefix filters, remove user_id filter and post-filter with `principal.can_access`.

```python
# Option A: access_entity + org_id filters (preferred)
access_entities = list(resolve_access_entities(principal))
hits = memory_client.vector_store.search(
    query=query,
    vectors=embeddings,
    limit=search_limit,
    filters={"org_id": principal.org_id, "access_entity": access_entities},
)

# Option B: no access_entity filters, post-filter by can_access (least efficient)
hits = memory_client.vector_store.search(
    query=query,
    vectors=embeddings,
    limit=search_limit,
)
```

Then rely on the existing `allowed` set to filter results.

Also ensure metadata includes org_id and access_entity on write:
- In MCP `add_memories`, include `org_id` in metadata payload.
- In REST create, include `org_id` in metadata payload.

### 6) MCP update/delete must be group-editable
File: `openmemory/api/app/mcp_server.py`

Update memory lookup to not restrict to `Memory.user_id == user.id`. Use access_entity checks instead.

```python
# Update memory lookup to use ID only
memory = db.query(Memory).filter(Memory.id == uuid.UUID(memory_id)).first()

# Then enforce can_write_to_access_entity(principal, access_entity)
```

### 7) Graph access across users
Files:
- `openmemory/api/app/graph/metadata_projector.py`
- `openmemory/api/app/graph/graph_ops.py`

When `allowed_memory_ids` is provided, do not filter by `userId` in Cypher.

```python
if allowed_memory_ids:
    match = "MATCH (m:OM_Memory)"
else:
    match = "MATCH (m:OM_Memory {userId: $userId})"
```

Also include `access_entity` in projection if needed for audits and debugging.

### 8) Backfill and indexes
- Add JSONB index on `metadata_.access_entity`.
- Backfill existing memories: if scope=user/session, set access_entity = user:<owner>.
- Re-sync Qdrant/OpenSearch/Neo4j after backfill.

## Test Plan (TDD)
Update or add tests:
- `openmemory/api/tests/test_access_entity_validation.py`
  - Remove client/service prefix tests.
- `openmemory/api/tests/test_access_entity_filtering.py`
  - REST list/filter/related return shared memories for grant holders.
  - REST search filters by access_entity prefixes.
- `openmemory/api/tests/test_mcp_access_entity_integration.py`
  - MCP search returns memories from other users if grant allows.
  - MCP update/delete works for group members (not just owner).
- `openmemory/api/tests/routers/test_search_router.py`
  - Ensure search endpoints apply access_entity filters (mock store call args).

## Success Criteria
- Shared memory is visible to all grant holders in REST and MCP search.
- Group members can update/delete shared memories.
- No `client:` or `service:` access_entity values accepted.
- Legacy memories (no access_entity) remain owner-only.

## Progress Tracker
- [ ] Remove client/service access_entity prefixes and update tests
- [ ] REST list/filter/related: replace user_id filtering with access_entity filtering
- [ ] REST create: enforce access_entity grants + default user scope
- [ ] REST search: add access_entity filters (exact + prefix)
- [ ] MCP search: remove user_id-only filter, add access_entity-aware filtering
- [ ] MCP update/delete: remove owner-only DB lookup, enforce group-editable
- [ ] Graph ops: remove userId filter when allowed_memory_ids is provided
- [ ] Add metadata index + backfill plan
- [ ] Re-sync Qdrant/OpenSearch/Neo4j
- [ ] Tests passing
