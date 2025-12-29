# Continuation Prompt: Multi-User Memory Routing (access_entity)

## Objective
TBD: summarize the next milestone and goal.

## Current State (code)
- TBD: summarize what is implemented and what is still missing.

## Decisions
- Access control field is `access_entity`; `entity` stays semantic.
- Write policy for shared memories: TBD (creator-only vs group-editable).
- Add any new decisions here.

## Completed Work (since last prompt)
- TBD: list changes and tests added.

## Remaining Work (next steps)
- TBD: list concrete next steps.

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
1. Add or update tests first.
2. Run tests and implement minimal changes to pass.
3. Update progress and docs.

## Change Checklist
- [ ] Updated `docs/plans/multi-user-memory-routing-progress.md`
- [ ] Added/updated tests
- [ ] Implemented code changes
- [ ] Updated docs
