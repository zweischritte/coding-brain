# Continuation Prompt: Multi-User Memory Routing (Phase 3 Complete)

## Objective

Complete the remaining integration points for multi-user memory routing:

1. REST endpoint integration (`routers/memories.py`)
2. Graph query filtering by `access_entity`
3. OpenSearch filtering by `access_entity`

## Current State (code)

### Completed

- **access_entity validation** in `structured_memory.py`
- **JWT grants parsing** in `security/jwt.py`, `security/types.py`
- **Principal.can_access()** with hierarchy expansion
- **Access control helpers** in `security/access.py`:
  - `resolve_access_entities()`, `can_write_to_access_entity()`, `can_read_access_entity()`
  - `get_default_access_entity()`, `build_access_entity_patterns()`
  - `filter_memories_by_access()`, `check_create_access()`
- **Qdrant store** updated with `access_entity` index and filtering methods
- **MCP tools** enforcement:
  - `add_memories`: access_entity parameter + access control check
  - `update_memory`: access_entity parameter + write access checks
  - `delete_memories`: filtering by access_entity permissions
  - `search_memory`, `list_memories`: filtering by access_entity grants
- **generate_jwt.py** with `--grants` flag

### Still Missing

- REST endpoints in `routers/memories.py` need access_entity filtering
- Graph operations in `graph_ops.py` need `allowed_memory_ids` passed from access_entity resolution
- OpenSearch store needs access_entity filtering

## Decisions

- Access control field is `access_entity`; `entity` stays semantic.
- Write policy for shared memories: group-editable (audit logs/backups for recovery).
- Qdrant uses `MatchAny` for OR filtering of access_entities.
- MCP tools default `access_entity` to `user:<uid>` for personal scopes (user/session).

## Completed Work (since last prompt)

1. Extended `security/access.py` with helper functions
2. Added `access_entity` to Qdrant `indexed_fields`, added filtering methods
3. Updated MCP tools (add_memories, update_memory, delete_memories, search_memory, list_memories)
4. Updated `generate_jwt.py` with `--grants` flag
5. Updated progress documentation
6. **All 115 tests passing**

## Remaining Work (next steps)

1. **REST endpoints** (`routers/memories.py`):
   - Add access_entity filtering to `list_memories`, `get_memory`, `filter_memories`
   - Add access control check to `create_memory`, `update_memory`, `delete_memory`

2. **Graph operations** (`graph_ops.py`, `routers/graph.py`):
   - Pass `allowed_memory_ids` from access_entity resolution to graph functions
   - Functions needing updates: `aggregate_memories_in_graph`, `tag_cooccurrence_in_graph`, `fulltext_search_memories_in_graph`

3. **OpenSearch store** (`stores/opensearch_store.py`):
   - Add `access_entity` to index mappings
   - Add access_entity filtering to search methods

4. **Integration tests**:
   - Add end-to-end tests for MCP access_entity enforcement
   - Add tests for graph query filtering

## Required References

- Plan: `docs/plans/multi-user-memory-routing.md`
- Progress: `docs/plans/multi-user-memory-routing-progress.md`
- Core files likely to change:
  - `openmemory/api/app/routers/memories.py`
  - `openmemory/api/app/graph/graph_ops.py`
  - `openmemory/api/app/routers/graph.py`
  - `openmemory/api/app/stores/opensearch_store.py`
  - `openmemory/api/app/routers/search.py`
- Tests: `openmemory/api/tests/` (locate relevant tests with `rg`)

## TDD Workflow

1. Add or update tests first.
2. Run tests and implement minimal changes to pass.
3. Update progress and docs.

## Change Checklist

- [ ] Updated `docs/plans/multi-user-memory-routing-progress.md`
- [ ] Added/updated tests
- [ ] Implemented code changes
- [ ] Updated docs
