# Multi-User Memory Routing Progress

## Decision Log
- Access control field is `access_entity`; `entity` remains semantic.
- Write policy for shared memories: TBD (creator-only vs group-editable).

## Phase Status
- [x] Phase 0: decisions & doc update (access_entity)
- [ ] Phase 1: prompt template docs
- [ ] Phase 2: metadata + validation
- [ ] Phase 3: access control enforcement
- [ ] Phase 4: retrieval backend alignment
- [ ] Phase 5: UI filters (optional)
- [ ] Phase 6: optional hardening

## TDD Test Targets
- [ ] Validate `access_entity` required for `scope!=user`.
- [ ] JWT grants parsed into TokenClaims.
- [ ] Access filtering in list/filter/get/related endpoints.
- [ ] MCP add/update/delete enforcement against grants.
- [ ] Qdrant search filters allow any allowed `access_entity`.
- [ ] Graph queries filter by allowed `access_entity` values.
- [ ] OpenSearch filters by allowed `access_entity` values.

## Work Log
- Created progress tracking + continuation prompts.
- Plan updated to use `access_entity` for access control.

## Open Questions
- Shared write policy: creator-only or group-editable?
