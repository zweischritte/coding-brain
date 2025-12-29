# Coding Brain / OpenMemory System Guide

This document is a developer-focused overview of the Coding Brain system (an OpenMemory fork) as it exists in this repo. It covers what is implemented, how the services fit together, and how to run the stack locally.

---

## What Coding Brain Is

Coding Brain is a memory and code-intelligence backend for development assistants. It exposes:
- REST APIs for memory, search, graph, feedback, experiments, and ops flows
- MCP servers over SSE for memory tools, business concepts, and AXIS guidance
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

### Search and Retrieval
- REST search endpoints backed by OpenSearch (lexical + optional vector)
- Metadata-based re-ranking for memory search results
- Graph traversal helpers for related memories and entity networks

### Code Intelligence Modules
- Tree-sitter + SCIP indexing pipeline and CODE_* graph projection to Neo4j
- Tri-hybrid retrieval (lexical + vector + graph) and optional reranker
- Libraries for explain-code, call-graph, impact analysis, ADR automation, test generation, PR analysis
  (present in codebase, not wired to MCP/REST by default)

### Security, Ops, Observability
- JWT validation with scope-based RBAC
- Optional DPoP token binding
- MCP SSE session binding with memory or Valkey stores
- Health probes, circuit breakers, rate limiting, audit logging
- Backup/export and GDPR endpoints
- Prometheus metric helpers (mountable if you add the metrics app)

### AXIS Guidance
- Separate MCP SSE endpoint for serving AXIS protocol guides on demand

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

## MCP Endpoints
- Memory MCP: `/mcp/<client>/sse/<user_id>`
- Business Concepts MCP: `/concepts/<client>/sse/<user_id>` (requires `BUSINESS_CONCEPTS_ENABLED=true`)
- AXIS Guidance MCP: `/axis/<client>/sse/<user_id>`

SSE uses a `session_id` query parameter for POSTs. All MCP calls require
`Authorization: Bearer <JWT>` and optional `DPoP` headers.

Install for a local MCP client:
```bash
npx @openmemory/install local http://localhost:8865/mcp/<client>/sse/<user_id> --client <client>
```

---

## REST API Surface (Selected)

Base: `http://localhost:8865/api/v1`

- `memories` - CRUD and listing
- `search` - lexical and hybrid search
- `graph` - aggregations, entity networks, tag co-occurrence
- `entities` - entity metadata and centrality
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

---

## Troubleshooting
- `401/403` on REST or MCP: check JWT issuer/audience/secret and required scopes.
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
- Hook the code-intelligence tools into MCP/REST if you want them exposed
- Wire the Prometheus `/metrics` app into the main API if you need scraping
- Tune OpenSearch and Qdrant settings per workload
