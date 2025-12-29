# Continuation Prompt: Multi-User Memory Routing (access_entity)

## Objective
Implement multi-user memory routing using `access_entity` and JWT grants, following a TDD workflow.

## Current State (code)
- Postgres list/filter/related endpoints filter by `Memory.user_id` (single-user).
- MCP `search_memory` filters Qdrant with `{"user_id": uid}`; `scope`/`entity` are boost-only.
- Graph projections and queries are scoped by `userId`.
- Qdrant/OpenSearch isolate only by `org_id`.
- `scope` is metadata only; no access control enforcement.

## Decisions
- Access control field is `access_entity`; `entity` stays semantic.
- Write policy for shared memories is TBD (creator-only vs group-editable).

## Required References
- Plan: `docs/plans/multi-user-memory-routing.md`
- Progress: `docs/plans/multi-user-memory-routing-progress.md`
- Core files likely to change:
  - `openmemory/api/app/mcp_server.py`
  - `openmemory/api/app/utils/structured_memory.py`
  - `openmemory/api/app/security/types.py`
  - `openmemory/api/app/security/jwt.py`
  - `openmemory/api/scripts/generate_jwt.py`
  - `openmemory/api/app/routers/memories.py`
  - `openmemory/api/app/stores/qdrant_store.py`
  - `openmemory/api/app/graph/metadata_projector.py`
  - `openmemory/api/app/graph/graph_ops.py`
  - `openmemory/api/app/routers/search.py`
  - `openmemory/api/app/stores/opensearch_store.py`
- Tests: `openmemory/api/tests/` (locate relevant tests with `rg`)

## TDD Workflow
1. Add tests for `access_entity` validation and JWT grants parsing.
2. Add tests for access filtering in list/filter/search/related and MCP tools.
3. Add tests for Qdrant/graph/OpenSearch filtering by allowed `access_entity`.
4. Run tests and implement minimal changes to pass.
5. Update docs/progress tracking.

## Implementation Notes
- `access_entity` required when `scope!=user`.
- Default `access_entity` for personal memories: `user:<sub>`.
- Add `grants` to TokenClaims; parse from JWT; extend `generate_jwt.py`.
- Add `resolve_access_entities(principal)` helper to expand grants and include `user:<sub>`.
- Enforce access control in REST + MCP paths.
- Align retrieval backends (Qdrant/Neo4j/OpenSearch) to filter by allowed `access_entity`.

## Deliverables
- Code changes + tests
- Updated progress file
- Updated docs if needed

## Change Checklist
- [ ] Updated `docs/plans/multi-user-memory-routing-progress.md`
- [ ] Added/updated tests
- [ ] Implemented code changes
- [ ] Updated docs
