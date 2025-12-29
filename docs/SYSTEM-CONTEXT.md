# System Context: Coding Brain (OpenMemory Fork)

Purpose
This system is a production-grade memory and code-intelligence backend for development assistants. It combines persistent memory, graph enrichment, and retrieval tooling and exposes them via REST APIs and MCP (Model Context Protocol) over SSE.

Current runtime shape
- FastAPI app with REST endpoints under `/api/v1` and health probes under `/health`
- MCP SSE endpoints for memory tools (`/mcp`), business concepts (`/concepts`), and guidance (`/guidance`)
- Data stores: PostgreSQL, Qdrant, OpenSearch, Neo4j, Valkey
- Next.js UI in `openmemory/ui`

Implemented capabilities
- Structured memory CRUD with categories, scopes, artifacts, entities, tags, and evidence
- Memory state tracking and access logs in PostgreSQL
- Vector search via Mem0 (default Qdrant)
- Neo4j metadata graph (OM_*), similarity edges, tag co-occurrence, typed relations, timeline queries
- Optional business concepts extraction + concept graph and vector store
- OpenSearch-backed memory search endpoints (lexical and hybrid)
- Security: JWT + scopes, optional DPoP binding, MCP SSE session binding (memory or Valkey)
- Ops and governance: backup/export, GDPR SAR and deletion, circuit breakers, rate limiting, audit logging
- Code intelligence modules: indexing pipeline, CODE_* graph, tri-hybrid retrieval, code tooling, cross-repo utilities, graph visualization (modules exist but are not exposed via MCP/REST by default)

Where to look in the repo
- `openmemory/api/app` for FastAPI routes, MCP servers, security, graph ops, and stores
- `openmemory/api/indexing`, `openmemory/api/retrieval`, `openmemory/api/tools` for code intelligence
- `openmemory/api/cross_repo` for cross-repo registry and analysis utilities
- `openmemory/api/visualization` for graph export and pagination
- `openmemory/docker-compose.yml` for deployment defaults and ports
- `docs/RUNBOOK-*` and `docs/TECHNICAL-ARCHITECTURE.md` for ops and architecture references

Notes
- If PostgreSQL is not configured, the API falls back to a local SQLite database (`openmemory.db`) for development.
- Prometheus metrics helpers exist but are not mounted into the main API by default.
