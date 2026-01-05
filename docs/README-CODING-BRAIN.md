# Coding Brain / OpenMemory System Guide

This document is a developer-focused overview of the Coding Brain system (an OpenMemory fork) as it exists in this repo. It covers what is implemented, how the services fit together, and how to run the stack locally.

---

## What Coding Brain Is

Coding Brain is a memory and code-intelligence backend for development assistants. It exposes:
- REST APIs for memory, search, graph, feedback, experiments, and ops flows
- MCP servers over SSE for memory tools, business concepts, and guidance
- A Next.js UI for browsing memories

It is built on a multi-store backend: PostgreSQL, Qdrant, OpenSearch, Neo4j, and Valkey.

---

## Major Capabilities (Current)

### Memory and Knowledge
- Structured memory schema with categories, scopes, artifacts, entities, tags, and evidence
- CRUD, state changes, and access logging in PostgreSQL
- Vector search via Mem0 (default: Qdrant)
- Neo4j metadata graph (OM_*), similarity edges, typed relations, tag co-occurrence
- Optional business concept extraction with a separate concept graph and vector store
- Multi-user memory routing via access_entity with grant-based visibility and group-editable writes

### Search and Retrieval
- REST search endpoints backed by OpenSearch (lexical; semantic when clients supply query vectors)
- Metadata-based re-ranking for memory search results
- Graph traversal helpers for related memories and entity networks

### Code Intelligence Modules
- Tree-sitter + SCIP indexing pipeline with CODE_* graph projection to Neo4j
- Tri-hybrid retrieval (lexical + vector + graph) powering code search
- Code tools are exposed via REST (`/api/v1/code`); MCP currently exposes
  index/search/explain/callers/callees/impact (ADR/test/pr are REST-only)

### Security, Ops, Observability
- JWT validation with scope-based RBAC
- Optional DPoP token binding
- MCP SSE session binding with memory or Valkey stores
- Health probes, circuit breakers, rate limiting, audit logging
- Backup/export and GDPR endpoints
- Prometheus metrics endpoint at `/metrics`

### Guidance
- Separate MCP SSE endpoint for serving guidance documents on demand

---

## Multi-User Memory Routing (access_entity)

Coding Brain uses `access_entity` to control read/write access to shared memories.
Shared memories are visible to all holders of matching grants; writes are group-editable.

### Access Entity Formats
Allowed prefixes (client/service removed):
- `user:<user_id>` (default for `scope=user` and `scope=session`)
- `team:<org>/<team>`
- `project:<org>/<path>`
- `org:<org>`

Shared scopes (`team`, `project`, `org`, `enterprise`) require `access_entity`.

### Grant Hierarchy
JWT grants drive access, with hierarchical expansion:
- `org:X` grants `org:X` plus all `project:X/*` and `team:X/*`
- `project:X` grants `project:X` plus all `team:X/*`
- `team:X` grants only that team
- `user:X` grants only that user

### Defaults and Auto-Resolution
- Personal scopes (`user`, `session`) default to `user:<sub>` if `access_entity` is omitted.
- For shared scopes, `access_entity="auto"` (or omitted) will default **only** when there is exactly one matching grant.
- If multiple matching grants exist, the request is rejected with an ambiguity error and options list.

### Behavior by Surface
- REST list/filter/related/search use `access_entity` (not creator `user_id`)
- MCP search/list/update/delete use `access_entity` (group-editable policy)
- Graph queries use the accessible memory IDs instead of user-only filtering
- OpenSearch filtering supports exact + prefix matching for org/project grants
- Legacy memories without `access_entity` remain owner-only

### Token Grants
Generate JWTs with grants using `openmemory/api/scripts/generate_jwt.py`:
```bash
python scripts/generate_jwt.py --user alice --org cloudfactory \
  --grants "user:alice team:cloudfactory/backend org:cloudfactory"
```

---

## Data Migrations and Backfill

Multi-user routing relies on the following migrations and scripts:
- Access entity index (Postgres): `openmemory/api/alembic/versions/add_access_entity_index.py`
- RLS disabled for shared access: `openmemory/api/alembic/versions/disable_rls_for_shared_access.py`
- Personal-scope backfill script: `openmemory/api/app/scripts/backfill_access_entity.py`

Backfill usage:
```bash
cd openmemory/api
python -m app.scripts.backfill_access_entity --dry-run
python -m app.scripts.backfill_access_entity
```

If you backfill, re-sync vector and graph stores to reflect updated metadata.

---

## Architecture (High Level)
- API/MCP: `openmemory/api` (FastAPI)
- UI: `openmemory/ui` (Next.js)
- PostgreSQL: memory metadata, users, apps, feedback, experiments, config
- Qdrant: memory embeddings (and optional business concepts)
- OpenSearch: lexical/hybrid search index for memories
- Neo4j: OM_* memory metadata graph, CODE_* code graph, concept graph
- Valkey: session binding and DPoP replay cache

---

## Ports (Default Docker Compose)

From `openmemory/docker-compose.yml`:
- API/MCP: `http://localhost:8865` (container port 8765)
- UI: `http://localhost:3433`
- Docs: `http://localhost:3080/docs/`
- PostgreSQL: `localhost:5532`
- Valkey: `localhost:6479`
- Qdrant HTTP: `localhost:6433` (gRPC: 6434)
- Neo4j Browser: `http://localhost:7574`
- Neo4j Bolt: `localhost:7787`
- OpenSearch: `http://localhost:9200` (metrics: 9600)

---

## Quickstart (Docker, Full Stack)

1) Configure environment
```bash
cd openmemory
cp .env.example .env
make env   # copies api/.env.example and ui/.env.example
```
Edit `openmemory/.env` and set required secrets:
- `JWT_SECRET_KEY` (32+ chars)
- `POSTGRES_PASSWORD`
- `NEO4J_PASSWORD`
- `OPENAI_API_KEY`
- `OPENSEARCH_INITIAL_ADMIN_PASSWORD`

Optional: adjust `USER`, `NEXT_PUBLIC_API_URL`, and CORS settings.
For the UI, set `NEXT_PUBLIC_API_TOKEN` to a JWT whose `sub` matches `USER`
and includes the scopes you need (see Troubleshooting).

Generate a token with all scopes:
```bash
cd openmemory/api
python scripts/generate_jwt.py
```
Or with specific scopes:

```bash
python scripts/generate_jwt.py --scopes "memories:read code:read code:write"
python scripts/generate_jwt.py --list-scopes  # show available scopes
```

2) Build and run
```bash
make build
make up
```

3) Verify
- API docs: `http://localhost:8865/docs`
- UI: `http://localhost:3433`
- Health: `http://localhost:8865/health/live`
- MCP SSE: `http://localhost:8865/mcp/<client>/sse/<user_id>`

---

## Code Indexing (Required for Code Tools)

The code tools depend on CODE_* graph + OpenSearch documents. Trigger indexing
after the stack is up:

REST:
```bash
curl -X POST http://localhost:8865/api/v1/code/index \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "my-repo",
    "root_path": "/path/to/repo",
    "reset": true
  }'
```

MCP:
```text
index_codebase(repo_id="my-repo", root_path="/path/to/repo", reset=true)
```

Async indexing (recommended for large repos):
```text
index_codebase(repo_id="my-repo", root_path="/path/to/repo", reset=true, async_mode=true)
index_codebase_status(job_id="<job-id>")
index_codebase_cancel(job_id="<job-id>")
```

REST job status and listing:
```bash
curl -X GET http://localhost:8865/api/v1/code/index/status/<job-id> \
  -H "Authorization: Bearer <your-jwt-token>"
curl -X GET "http://localhost:8865/api/v1/code/index/jobs?repo_id=my-repo&status=running&limit=10" \
  -H "Authorization: Bearer <your-jwt-token>"
```

Additional test ideas:
- Force behavior: start an async job, then start a second with `force=true`; the first should become canceled.
- Queue saturation: set `MAX_QUEUED_JOBS=1` and verify the second enqueue returns 429.

---

## MCP Endpoints
- Memory MCP: `/mcp/<client>/sse/<user_id>`
- Business Concepts MCP: `/concepts/<client>/sse/<user_id>` (requires `BUSINESS_CONCEPTS_ENABLED=true`)
- Guidance MCP: `/guidance/<client>/sse/<user_id>`

SSE uses a `session_id` query parameter for POSTs. All MCP calls require
`Authorization: Bearer <JWT>` and optional `DPoP` headers.

Install for a local MCP client:
```bash
npx @openmemory/install local http://localhost:8865/mcp/<client>/sse/<user_id> --client <client>
```

Auth/debug helper (MCP):
```text
whoami()
```
Returns the current MCP auth context (user_id, org_id, scopes, grants, client_name).

### Memory Tools (MCP)

#### get_memory

Retrieve a single memory by UUID with all fields:

```text
get_memory(memory_id="ba93af28-784d-4262-b8f9-adb08c45acab")
```

Returns all fields including `access_entity`, useful for:

- Verifying updates succeeded
- Debugging ACL issues
- Following evidence chains between memories
- Exploring similar memories from search results (via `OM_SIMILAR` edges)

### Claude Code Configuration

On macOS, Claude Code stores MCP settings in `~/.claude.json`. Add the following to connect to Coding Brain:

```json
{
  "mcpServers": {
    "coding-brain-memory": {
      "type": "sse",
      "url": "http://localhost:8865/mcp/claude-code/sse/<user_id>",
      "headers": {
        "Authorization": "Bearer <your-jwt-token>"
      }
    }
  }
}
```

Replace `<user_id>` with your configured user (default: `default_user`) and `<your-jwt-token>` with the token from `openmemory/.env` (`NEXT_PUBLIC_API_TOKEN`).

After adding this configuration, restart Claude Code. You should then have access to memory tools like `add_memories`, `search_memory`, `list_memories`, and the `graph_*` tools.

---

## REST API Surface (Selected)

Base: `http://localhost:8865/api/v1`

- `memories` - CRUD and listing
- `search` - lexical and hybrid search
- `graph` - aggregations, entity networks, tag co-occurrence
- `entities` - entity metadata and centrality
- `code` - code-intel endpoints (search, explain, callers/callees, impact, ADR, test generation, PR analysis, indexing)
- `feedback` - retrieval feedback and metrics
- `experiments` - A/B tests for retrieval settings
- `backup` - export/import
- `gdpr` - SAR export and deletion
- `config` - Mem0 configuration (LLM/embedder/vector store)
- `apps`, `stats` - app management and aggregates

---

## Local Development

### API only
```bash
cd openmemory/api
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```
If PostgreSQL is not configured, the API falls back to `openmemory.db` SQLite
for local dev.

### UI only
```bash
cd openmemory/ui
pnpm install
pnpm dev
```
If you are calling a secured API, add `NEXT_PUBLIC_API_TOKEN` to
`openmemory/ui/.env.local` (and keep `NEXT_PUBLIC_USER_ID` aligned with the token `sub`).

---

## Troubleshooting
- `401/403` on REST or MCP: check JWT issuer/audience/secret and required scopes.
- Shared memories not visible: ensure `access_entity` is set and the JWT includes matching grants.
- UI shows empty lists or 403s: ensure `NEXT_PUBLIC_API_TOKEN` includes
  `memories:read`, `apps:read`, `stats:read`, `entities:read`, and `graph:read`
  (plus `memories:write/delete` or `apps:write` if you edit data).
- MCP POST errors: ensure `session_id` is included in the POST URL.
- Valkey errors: verify `VALKEY_HOST` and `VALKEY_PORT` or set `MCP_SESSION_STORE=memory`.
- Neo4j auth failures: check `NEO4J_USERNAME` and `NEO4J_PASSWORD`.
- UI not loading: confirm `NEXT_PUBLIC_API_URL` points to `http://localhost:8865`.

---

## What Is In This Repo

Key directories:
- `openmemory/api` - FastAPI backend and MCP servers
- `openmemory/ui` - Next.js UI
- `openmemory/docker-compose.yml` - full stack
- `openmemory/run.sh` - simplified bootstrap for memory-only installs
- `openmemory/api/indexing` - code indexing and CODE_* graph projection
- `openmemory/api/retrieval` - tri-hybrid retrieval and reranking
- `openmemory/api/tools` - code intelligence tooling modules
- `openmemory/api/cross_repo` - cross-repo registry and analysis

---

## Next Steps
- Add a background worker or cron for continuous code indexing
- Configure Prometheus to scrape `/metrics` (and secure it if needed)
- Tune OpenSearch and Qdrant settings per workload
