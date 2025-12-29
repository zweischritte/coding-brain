# PRD: Next-Step Implementation Pack (Migrations + Metrics + Code Tools)

## Summary
Implement three foundational upgrades in Coding Brain:
1) Automated Alembic migrations with a safe default.
2) Prometheus `/metrics` on the main API.
3) Expose code-intelligence tools via MCP and REST.

This PRD is self-contained and includes repo references, tool interfaces, and decision rationale so a fresh LLM can implement without guessing.

---

## Implementation Progress

### Session: 2025-12-29

| Feature | Status | Commit | Notes |
|---------|--------|--------|-------|
| AUTO_MIGRATE flag | COMPLETED | pending | Added to `app/database.py` |
| AUTO_MIGRATE tests (TDD) | COMPLETED | pending | `tests/infrastructure/test_auto_migrate.py` |
| MetricsMiddleware integration | COMPLETED | pending | Added to `main.py` |
| `/metrics` endpoint | COMPLETED | pending | Added to `main.py` |
| Metrics integration tests | COMPLETED | pending | `tests/infrastructure/test_metrics_integration.py` |
| docker-compose AUTO_MIGRATE | COMPLETED | pending | Added `AUTO_MIGRATE=true` to codingbrain-mcp service |
| Code tools router (REST) | PENDING | - | Next phase |
| Code tools MCP registration | PENDING | - | Next phase |

### Files Modified
- [app/database.py](../openmemory/api/app/database.py) - Added `auto_migrate_on_startup()`, `is_postgres_database()`, `should_auto_migrate()`, `run_alembic_upgrade()`
- [main.py](../openmemory/api/main.py) - Added MetricsMiddleware, `/metrics` endpoint, auto_migrate call
- [docker-compose.yml](../openmemory/docker-compose.yml) - Added `AUTO_MIGRATE=true` for dev environment

### Files Created
- [tests/infrastructure/test_auto_migrate.py](../openmemory/api/tests/infrastructure/test_auto_migrate.py) - TDD tests for AUTO_MIGRATE
- [tests/infrastructure/test_metrics_integration.py](../openmemory/api/tests/infrastructure/test_metrics_integration.py) - Tests for metrics integration

---

## Prerequisite Checklist (Before Implementation)
- OpenSearch `code` index exists (or a creation step is included).
- CODE_* graph is populated in Neo4j (indexing completed).
- Embedding adapter chosen and matches code index dimension (default 768).
- Alembic uses the same DB URL source as the API (`DATABASE_URL` vs settings).
- Import paths are valid in the API container (`openmemory.api.*` vs `app.*`).

---

## Context and System Map
- Backend: FastAPI in `openmemory/api/main.py`.
- REST routers in `openmemory/api/app/routers`.
- MCP server in `openmemory/api/app/mcp_server.py` with SSE auth and session binding.
- Alembic migrations in `openmemory/api/alembic/versions`.
- Observability helpers in `openmemory/api/app/observability/metrics.py`.
- Code-intel tool modules in `openmemory/api/tools`.
- Code indexing and CODE_* graph in `openmemory/api/indexing`.
- Neo4j driver in `openmemory/api/app/graph/neo4j_client.py`.
- OpenSearch client wrapper in `openmemory/api/retrieval/opensearch.py`.

---

## Decisions (Recommended)
- **Scopes**: Reuse existing scopes; do not add new ones.
  - `search_code_hybrid`: `SEARCH_READ`
  - `find_callers`, `find_callees`, `impact_analysis`: `GRAPH_READ`
  - `explain_code`, `adr_automation`, `test_generation`, `pr_analysis`: require both `SEARCH_READ` and `GRAPH_READ`
- **Migration default**: OFF in code, ON in dev Docker via env (`AUTO_MIGRATE=true`).
- **REST path**: `/api/v1/code/*` for all code-intel endpoints.

---

## Goals
- Ensure Postgres schema matches the codebase via Alembic.
- Provide a stable `/metrics` endpoint on the main API.
- Make code-intelligence tools callable via MCP and REST.
- Maintain graceful degradation if OpenSearch or Neo4j is unavailable.

## Non-Goals
- No new indexing pipeline or ingestion changes.
- No UI updates.
- No new JWT issuance or auth provider changes.
- No data model changes beyond existing migrations.

---

## Preconditions and Known Gaps (Must Address)
- **CODE_* graph and OpenSearch `code` index**: Neither is created or populated by default. Tools will fail or return empty results until both exist. Add explicit index creation and indexing steps, or document as prerequisites.
- **Graph driver interface mismatch**: Code tools expect `get_node`, `get_outgoing_edges`, and `get_incoming_edges`. `get_neo4j_driver()` returns a raw Neo4j driver. Provide an adapter that implements these methods or update tools to run Cypher directly.
- **Embedding adapter missing**: `search_code_hybrid` expects `embedding_service.embed(text)` returning a vector matching the code index dimension (default 768 via `IndexConfig.for_code`). Define an adapter (Mem0 embedder or `EmbeddingPipeline`) or explicitly set `embed_query=false`.
- **Alembic DB URL mismatch**: `openmemory/api/alembic/env.py` uses `DATABASE_URL`. Auto-migration must set `DATABASE_URL` or update Alembic to read from `app.database.get_database_url()`.
- **Absolute imports**: Some tool modules import `openmemory.api.*`. In the API container, only `app` is on the path. Use relative imports or update `PYTHONPATH`/package layout to avoid import errors.

---

## Functional Requirements

### 1) Alembic Migrations (Automated)
**Behavior**
- Add `AUTO_MIGRATE` env flag (default false).
- On API startup, if `AUTO_MIGRATE=true` and DB is PostgreSQL, run `alembic upgrade head`.
- Skip auto-migration for SQLite (`openmemory.db`) to keep dev fallback safe.
- Keep `Base.metadata.create_all` only for SQLite to avoid schema drift in Postgres.

**Implementation Notes**
- Reference Alembic config at `openmemory/api/alembic.ini`.
- Use `alembic.command.upgrade` with `alembic.config.Config`.
- Ensure CWD-independent paths using `Path(__file__).resolve()` from within `openmemory/api`.
- Ensure `DATABASE_URL` is set consistently for Alembic (or update Alembic to reuse `app.database.get_database_url()`).

**Docker Integration**
- In `openmemory/docker-compose.yml`, set `AUTO_MIGRATE=true` for the API container (dev only).
- Keep production steps explicit: `alembic upgrade head` as part of deployment.

**Smoke Test**
- Add `scripts/smoke_test.sh` (or `.py`) to run:
  - `GET /health/live`
  - `GET /health/deps`
  - `GET /mcp/health`
  - Optional: `alembic current` from inside the api container (if Docker is running)

### 2) `/metrics` Endpoint (Prometheus)
**Behavior**
- Expose `/metrics` on the main FastAPI app.
- Use `MetricsMiddleware` for request metrics.
- `/metrics` must return Prometheus text format.
- No auth on `/metrics`.

**Implementation Notes**
- `MetricsMiddleware` and `create_metrics_app` live in `openmemory/api/app/observability/metrics.py`.
- Avoid `/metrics/metrics` by adding a direct route in `openmemory/api/main.py`:
  - Use `generate_latest(REGISTRY)` and `CONTENT_TYPE_LATEST`.

### 3) Code-Intelligence Exposure (MCP + REST)

#### MCP Tools (SSE)
**Where**
- Register tools in `openmemory/api/app/mcp_server.py`.
- Use the existing `_check_tool_scope` helper (scope enforcement) and `principal_var`.
- `_check_tool_scope` expects a string scope value (use `Scope.SEARCH_READ.value` or `"search:read"`).

**Required Tools**
- `search_code_hybrid` (`openmemory/api/tools/search_code_hybrid.py`)
- `explain_code` (`openmemory/api/tools/explain_code.py`)
- `find_callers` / `find_callees` (`openmemory/api/tools/call_graph.py`)
- `impact_analysis` (`openmemory/api/tools/impact_analysis.py`)
- `adr_automation` (`openmemory/api/tools/adr_automation.py`)
- `test_generation` (`openmemory/api/tools/test_generation.py`)
- `pr_analysis` (`openmemory/api/tools/pr_workflow/pr_analysis.py`)

**Dependencies**
- OpenSearch client wrapper: `create_opensearch_client(from_env=True)` from `openmemory/api/retrieval/opensearch.py`.
- Tri-hybrid retriever: `create_trihybrid_retriever(opensearch_client, graph_driver)` from `openmemory/api/retrieval/trihybrid.py`.
- Neo4j driver adapter: wrap `get_neo4j_driver()` so it implements `get_node`, `get_outgoing_edges`, `get_incoming_edges`.
- AST parser: `ASTParser` from `openmemory/api/indexing/ast_parser.py`.
- Optional embedding service:
  - Prefer Mem0 embedding model when available (see usage in `openmemory/api/app/mcp_server.py`), or use `EmbeddingPipeline`.
  - Ensure vector length matches code index dimension (default 768).
  - If embedding is unavailable, set `embed_query=false` and run lexical-only.

**Output Handling**
- Tool return types are dataclasses; convert with `dataclasses.asdict`.
- MCP tools must return JSON strings (match existing MCP tool patterns).

**Failure/Degrade**
- If Neo4j is not configured: return JSON error or empty results with `meta.degraded_mode=true` and `missing_sources=["neo4j"]`.
- If OpenSearch `code` index is missing: return empty results with `missing_sources=["opensearch"]`.

#### REST Endpoints
**Where**
- Create a new router: `openmemory/api/app/routers/code.py`.
- Export it from `openmemory/api/app/routers/__init__.py`.
- Include it in `openmemory/api/main.py`.

**Routes**
- `POST /api/v1/code/search` -> `search_code_hybrid`
- `POST /api/v1/code/explain` -> `explain_code`
- `POST /api/v1/code/callers` -> `find_callers`
- `POST /api/v1/code/callees` -> `find_callees`
- `POST /api/v1/code/impact` -> `impact_analysis`
- `POST /api/v1/code/adr` -> `adr_automation`
- `POST /api/v1/code/test-generation` -> `test_generation`
- `POST /api/v1/code/pr-analysis` -> `pr_analysis`

**Auth**
- Use `require_scopes` from `openmemory/api/app/security/dependencies.py` and the scope mapping in Decisions.
- Use `Scope.SEARCH_READ` and `Scope.GRAPH_READ` for REST; use `.value` for MCP scope checks.

**Input Schemas (Pydantic)**
- `search_code_hybrid`: `query`, `repo_id?`, `language?`, `limit?`, `offset?`, `seed_symbols?`
- `explain_code`: `symbol_id`
- `callers/callees`: `repo_id`, `symbol_id?`, `symbol_name?`, `depth?`
- `impact_analysis`: `repo_id`, `changed_files?`, `symbol_id?`, `include_cross_language?`, `max_depth?`, `confidence_threshold?`
- `adr_automation`: align with tool input (allow JSON passthrough if needed).
- `test_generation`: align with tool input (symbol or file based).
- `pr_analysis`: `repo_id`, `diff?`, `pr_number?`, `title?`, `body?`, `base_branch?`, `head_branch?`

**Output**
- Return dataclass results as JSON (`dataclasses.asdict`).
- On tool errors: `HTTP 400` for validation, `HTTP 503` for missing dependencies.

---

## Implementation Details

### Alembic Auto-Migration
Suggested pattern in `openmemory/api/main.py`:
```
if os.getenv("AUTO_MIGRATE", "false").lower() == "true":
    if database_url_is_postgres():
        alembic_upgrade_head()
```
Where `database_url_is_postgres()` can check `openmemory/api/app/database.py`.

### Metrics
Add:
- `app.add_middleware(MetricsMiddleware)`
- A new route:
```
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
```

### Code Tool Wiring
Centralize lazy initialization to avoid heavy imports:
- Build a cached `get_code_toolkit()` that returns:
  - OpenSearch client
  - Neo4j driver
  - TriHybrid retriever
  - Embedding service adapter (or `None`)
  - AST parser
  - Tool instances

---

## Error Handling and Degradation Rules
- **Neo4j missing**: return degraded response, do not crash.
- **OpenSearch missing**: return empty results + degraded meta.
- **Embedding service missing**: disable vector query embedding.
- **LLM key missing**: return 503 for ADR/test generation.

---

## Acceptance Criteria
- `AUTO_MIGRATE=true` runs `alembic upgrade head` on container start.
- `/metrics` returns Prometheus text from the main API.
- MCP tools for code-intel are accessible over SSE and return JSON strings.
- REST endpoints under `/api/v1/code/*` are reachable and enforce scopes.
- Smoke test script exits 0 when stack is healthy.

---

## Test Plan
- Run `make up` and verify:
  - `/health/live` returns 200.
  - `/health/deps` returns 200 or 503 with details.
  - `/mcp/health` returns 200.
  - `/metrics` returns text metrics.
- Call one code endpoint with minimal input; verify JSON response.
- Optional: `alembic current` matches head.
