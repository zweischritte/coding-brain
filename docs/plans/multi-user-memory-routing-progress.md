# Multi-User Memory Routing Progress

## Decision Log
- Access control field is `access_entity`; `entity` remains semantic.
- Write policy for shared memories: group-editable (audit logs/backups for recovery).

## Phase Status
- [x] Phase 0: decisions & doc update (access_entity)
- [ ] Phase 1: prompt template docs
- [x] Phase 2: metadata + validation (TDD complete)
- [x] Phase 3: access control enforcement (core logic + MCP integration complete)
- [x] Phase 4: retrieval backend alignment (Qdrant store updated)
- [ ] Phase 5: UI filters (optional)
- [ ] Phase 6: optional hardening

## TDD Test Targets
- [x] Validate `access_entity` required for `scope!=user`.
- [x] JWT grants parsed into TokenClaims.
- [x] Principal.can_access() with hierarchy expansion.
- [x] resolve_access_entities() helper.
- [x] Access filtering tests (read/write/create).
- [ ] REST endpoint integration (list/filter/get/related).
- [x] MCP add/update/delete integration against grants.
- [x] MCP search_memory/list_memories filtering by access_entity.
- [x] Qdrant search filters allow any allowed `access_entity`.
- [ ] Graph queries filter by allowed `access_entity` values.
- [ ] OpenSearch filters by allowed `access_entity` values.

## Implementation Summary (2025-12-29)

### Completed
1. **access_entity validation** (`structured_memory.py`)
   - Added `VALID_ACCESS_ENTITY_PREFIXES` (user, team, project, org, client, service)
   - Added `SHARED_SCOPES` (team, project, org, enterprise)
   - Added `validate_access_entity()` function
   - Added `validate_scope_access_entity_consistency()` function
   - Updated `StructuredMemoryInput` dataclass with `access_entity` field
   - Enforced: `scope!=user` requires `access_entity`
   - Enforced: scope/access_entity prefix must be consistent

2. **JWT grants parsing** (`security/jwt.py`, `security/types.py`)
   - Added `grants: Set[str]` field to `TokenClaims`
   - Added `has_grant()` and `has_any_grant()` methods to `TokenClaims`
   - JWT parsing extracts `grants` from payload
   - Always includes implied `user:<sub>` grant

3. **Principal access control** (`security/types.py`)
   - Added `has_grant()` and `has_any_grant()` methods
   - Added `get_allowed_access_entities()` method
   - Added `can_access()` method with **hierarchy expansion**:
     - `org:X` grant allows access to `org:X`, `project:X/*`, `team:X/*`, `client:X/*`
     - `project:X` grant allows access to `project:X`, `team:X/*`
     - `team:X` grant allows access to `team:X` only
     - `user:X` grant allows access to `user:X` only

4. **Access control helpers** (`security/access.py`)
   - Created new module
   - `resolve_access_entities(principal)` - returns all explicit grants + user grant
   - `can_write_to_access_entity(principal, access_entity)` - group-editable check
   - `can_read_access_entity(principal, access_entity)` - read access check
   - `get_default_access_entity(principal)` - returns `user:<user_id>`
   - `build_access_entity_patterns(principal)` - builds SQL patterns for hierarchy expansion
   - `filter_memories_by_access(principal, memories, get_access_entity)` - filters list by access
   - `check_create_access(principal, access_entity)` - validates create permission

5. **Qdrant store access control** (`stores/qdrant_store.py`)
   - Added `access_entity` to `indexed_fields` tuple
   - Added `_create_access_entity_filter()` method with OR logic via MatchAny
   - Added `search_with_access_control()` method
   - Added `list_with_access_control()` method

6. **MCP tools access_entity enforcement** (`mcp_server.py`)
   - `add_memories`: Added `access_entity` parameter, defaults to `user:<uid>` for personal scopes
   - `add_memories`: Added access control check before create
   - `update_memory`: Added `access_entity` parameter for changing access control
   - `update_memory`: Added access check for current and new access_entity
   - `delete_memories`: Added access_entity filtering to determine deletable memories
   - `search_memory`: Added access_entity filtering to ACL check
   - `list_memories`: Added access_entity filtering to memory listing

7. **JWT token generation** (`scripts/generate_jwt.py`)
   - Added `--grants` flag for multi-user memory routing
   - Added `grants` parameter to `generate_token()` function
   - Updated help text with grant format examples

### Tests Added
- `tests/test_access_entity_validation.py` (33 tests)
- `tests/security/test_jwt_grants.py` (17 tests)
- `tests/test_access_entity_filtering.py` (38 tests)
- Updated `tests/test_structured_memory.py` (added access_entity to shared scope tests)
- **Total: 95 tests passing**

### Files Changed
- `openmemory/api/app/utils/structured_memory.py`
- `openmemory/api/app/security/types.py`
- `openmemory/api/app/security/jwt.py`
- `openmemory/api/app/security/access.py` (new, extended)
- `openmemory/api/app/stores/qdrant_store.py` (access_entity filtering)
- `openmemory/api/app/mcp_server.py` (access_entity enforcement)
- `openmemory/api/scripts/generate_jwt.py` (--grants flag)
- `openmemory/api/tests/test_access_entity_validation.py` (new)
- `openmemory/api/tests/security/test_jwt_grants.py` (new)
- `openmemory/api/tests/test_access_entity_filtering.py` (new)
- `openmemory/api/tests/test_structured_memory.py` (updated)

## Work Log
- Created progress tracking + continuation prompts.
- Plan updated to use `access_entity` for access control.
- 2025-12-29: TDD implementation of Phase 2 + Phase 3 core logic (88 new tests, all passing).
- 2025-12-29: MCP tools integration complete (add_memories, update_memory, delete_memories, search_memory, list_memories).
- 2025-12-29: Qdrant store updated with access_entity filtering.
- 2025-12-29: generate_jwt.py updated with --grants flag.
- **All 95 tests passing.**

## Next Steps

1. ~~Integrate access control into MCP tools (`mcp_server.py`)~~ DONE
2. ~~Update `generate_jwt.py` to support `--grants` flag~~ DONE
3. ~~Update Qdrant store to filter by `access_entity`~~ DONE
4. Integrate access control into REST endpoints (`routers/memories.py`)
5. Update graph queries to filter by `access_entity` (graph_ops.py)
6. Update OpenSearch to filter by `access_entity` (opensearch_store.py)
7. Add integration tests for MCP access_entity enforcement

## Open Questions
- None tracked.
