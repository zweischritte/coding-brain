# Operations Readiness for Coding Brain

This document lists what needs to be in place for the existing system to work well: memory ingestion, indexing, retrieval, graph operations, and day-2 ops. It is a practical checklist, not new features.

## 1) Stack and environment readiness

- All core services are up and healthy: API/MCP, PostgreSQL, Qdrant, OpenSearch, Neo4j, Valkey.
- Required secrets are set in `openmemory/.env` (JWT, DB passwords, OpenAI key, OpenSearch admin password).
- Health endpoints return OK:
  - `http://localhost:8865/health/live`
  - `http://localhost:8865/health/ready`
  - `http://localhost:8865/health/deps`
- If running UI, `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_TOKEN` are set and match the user ID.
- If using DPoP, tokens and replay cache (Valkey) are configured correctly.

## 2) Auth and access control readiness

- JWT scopes include the required permissions for each workflow:
  - Memories: `memories:read`, `memories:write`, `memories:delete` (as needed)
  - Code tools: `code:read`, `code:write`
  - Graph/UI: `graph:read`, `entities:read`, `apps:read`, `stats:read`
- JWT grants match intended visibility (user/team/project/org), and `access_entity` is set for shared data.
- If you migrated legacy data, run the backfill and re-sync vector and graph stores afterward.
- Avoid ambiguous `access_entity="auto"` if multiple grants exist.

## 3) Memory ingestion quality

- Every memory includes `category`, `entity` (required), and the correct `access_entity`.
- Use `artifact_type` and `artifact_ref` for repo, file, or component references.
- Use consistent entity naming across the team to avoid graph fragmentation.
- Entity names are normalized; use displayName for UI outputs.
- Prefer updates over deletes to preserve history and keep graph links stable.
- Attach evidence links (ADR, PR, issue) for decisions and conventions.
- Define a simple tag taxonomy (e.g., `decision`, `deprecated`, `flaky`, `runbook`) and use it consistently.
- `add_memories` is async by default; poll `add_memories_status(job_id)` before assuming the memory is searchable.

## 4) Indexing readiness (code tools)

- Code indexing requires OpenSearch and Neo4j to be healthy.
- Use container-visible paths for `root_path` when indexing from MCP/REST.
- Re-index after major refactors or directory moves to keep CODE_* graph accurate.
- For large repos, use async indexing and monitor job status.
- Check queue and cancellation behavior if `MAX_QUEUED_JOBS` is low.
- If indexing is slow or failing, verify Tree-sitter/SCIP dependencies and resource limits.
- For strict call graphs, set `include_inferred_edges=false` (or `CODE_INTEL_INCLUDE_INFERRED_EDGES=false`).

## 5) Search and retrieval quality

- Use hybrid search for mixed queries (keywords + natural language).
- Use recency weighting for "current state" questions; avoid it for long-lived knowledge.
- Use tag and entity boosts to narrow results to the right component.
- Validate that OpenSearch index mappings and embeddings align with your Mem0 config.
- Use experiments and feedback endpoints to track retrieval changes over time.
- Maintain a small evaluation set of representative queries to detect regressions.

## 6) Graph readiness

- Graph queries depend on memory metadata; ensure `entity` and `tags` are populated.
- After bulk updates or access_entity backfill, re-sync graph and vector stores.
- Use graph tools for multi-hop questions and relationship-heavy queries.
- Run `backfill_graph_access_entity_hybrid.py` before entity bridge backfill.

## 7) MCP and client readiness

- MCP SSE calls include `Authorization: Bearer <JWT>` headers.
- MCP POST calls include a `session_id` query parameter.
- Client config (Claude Code or other MCP client) points to the correct user ID and token.

## 8) UI readiness

- UI token and user ID must align with JWT `sub` and scopes.
- CORS settings permit the UI origin if running separately.
- UI requires `memories:read`, `apps:read`, `stats:read`, `entities:read`, `graph:read`.

## 9) Operations and maintenance

- Backups are taken regularly and tested (Postgres + Qdrant + OpenSearch + Neo4j).
- Prometheus is scraping `/metrics`, and basic alerts exist for:
  - API error rate
  - Indexing job failures
  - Store availability (OpenSearch, Qdrant, Neo4j)
- Rate limiting and circuit breakers are configured to protect the system.
- GDPR export/deletion endpoints are documented for compliance workflows.
- Deployment runbook is followed for releases and rollbacks.

## 10) Common failure modes and quick checks

- 401/403 on API/MCP: JWT scopes or grants are missing.
- Empty results in UI: token lacks graph/entities/apps scopes.
- Shared memories not visible: `access_entity` missing or wrong grant.
- Search returns stale results: re-sync vector/graph after backfill or migration.
- Code tools return nothing: index not built or index path invalid.
- Entity casing looks wrong: run `backfill_entity_display_names.py`.

## References

- System guide: `docs/README-CODING-BRAIN.md`
- Deployment runbook: `docs/RUNBOOK-DEPLOYMENT.md`
