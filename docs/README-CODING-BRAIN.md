# Coding Brain / OpenMemory System Guide

This document is a comprehensive, developer-focused README for the Coding Brain (OpenMemory fork) system. It explains the major features, architecture, and how to run the full system locally or in a company server environment.

---

## Overview

Coding Brain is a production-grade memory system for AI assistants and agents. It combines:
- A FastAPI backend for memory and graph operations
- An MCP (Model Context Protocol) server for tool access (SSE transport)
- A web UI for browsing and testing memories
- Multiple data stores (PostgreSQL, Valkey, Qdrant, Neo4j, OpenSearch)
- Security (JWT + RBAC + optional DPoP) and multi-tenant support

The system is designed to work for:
- Local developer machines (Docker-based)
- A shared internal server reachable by developers
- MCP clients (Copilot-style, Gemini, local models, CLI tools)

---

## Major Features

### 1) Memory API (FastAPI)
- CRUD for memories (add/search/update/delete)
- Structured memory schema with tags/metadata
- Memory status tracking and access logs
- Per-user and per-app data scoping

### 2) MCP Server (SSE)
- MCP tools for memory operations
- SSE transport for MCP clients
- Session-bound authentication to prevent session hijacking
- Separate MCP server for business concepts (optional feature)

### 3) Security & Multi-Tenancy
- JWT validation (issuer/audience/exp/iat)
- OAuth-style scopes (RBAC)
- Optional DPoP verification for token binding
- org_id scoping for tenant separation
- Session binding for MCP SSE (session_id ↔ principal)

### 4) Graph + Search
- Neo4j graph operations for entities and relationships
- OpenSearch for full-text and hybrid search
- Qdrant for vector similarity search

### 5) Observability & Operations
- Health endpoints for API and session store
- Prometheus metrics (HTTP + session binding)
- Security audit logging for auth/session events

### 6) UI
- Web UI for memory review and exploration
- Configurable API endpoint and user ID

---

## Architecture (High Level)

- **API/MCP**: FastAPI + MCP server (`openmemory/api`)
- **UI**: Next.js app (`openmemory/ui`)
- **PostgreSQL**: relational store + metadata
- **Valkey**: caching + session binding
- **Qdrant**: vector store for memory embeddings
- **Neo4j**: graph for relationships and entities
- **OpenSearch**: full-text search

---

## Ports (Default Docker Compose)

From `openmemory/docker-compose.yml`:
- API/MCP: `http://localhost:8865`
- UI: `http://localhost:3433`
- Postgres: `localhost:5532`
- Valkey: `localhost:6479`
- Qdrant: `localhost:6433`
- Neo4j Browser: `http://localhost:7574`
- Neo4j Bolt: `localhost:7787`
- OpenSearch: `http://localhost:9200`

---

## Quickstart (Docker, Recommended)

### 1) Prerequisites
- Docker + Docker Compose v2
- OpenAI API key (for embeddings / LLM features)

### 2) Configure environment
From the repo root:

```bash
cd openmemory
cp .env.example .env
make env  # copies api/.env and ui/.env
```

Then edit:
- `openmemory/.env` (required secrets: JWT_SECRET_KEY, POSTGRES_PASSWORD, NEO4J_PASSWORD, OPENAI_API_KEY, etc.)
- `openmemory/api/.env` (OPENAI_API_KEY and USER for API development)
- `openmemory/ui/.env` (NEXT_PUBLIC_API_URL and NEXT_PUBLIC_USER_ID)

### 3) Build and run

```bash
make build
make up
```

### 4) Verify
- API docs: `http://localhost:8865/docs`
- UI: `http://localhost:3433`
- MCP SSE endpoint: `http://localhost:8865/mcp/<client>/sse/<user_id>`

---

## MCP Client Setup

Use the OpenMemory MCP installer:

```bash
npx @openmemory/install local http://localhost:8865/mcp/<client-name>/sse/<user-id> --client <client-name>
```

Notes:
- Clients must send `Authorization: Bearer <token>` on **GET and POST**.
- If you enable DPoP, clients must also send `DPoP` headers.

---

## Local Development (API + UI)

### API only
```bash
cd openmemory/api
cp .env.example .env
# fill OPENAI_API_KEY, USER, and any required settings
uvicorn main:app --host 0.0.0.0 --port 8765 --reload
```

### UI only
```bash
cd openmemory/ui
pnpm install
pnpm dev
```

### Hybrid
Run the API in Docker and UI locally:
```bash
cd openmemory
make build
make up
cd ui
pnpm install
pnpm dev
```

---

## Environment Variables (Essentials)

From `openmemory/.env.example`:
- **JWT_SECRET_KEY** (min 32 chars)
- **POSTGRES_PASSWORD**
- **NEO4J_PASSWORD**
- **OPENAI_API_KEY**

Optional (recommended):
- **JWT_ISSUER**, **JWT_AUDIENCE**
- **VALKEY_HOST/PORT**
- **QDRANT_HOST/PORT**
- **OPENSEARCH_INITIAL_ADMIN_PASSWORD**
- **CORS_ALLOWED_ORIGINS**

---

## Security Notes

- **JWT + RBAC**: MCP tool access requires scope checks.
- **Session Binding**: `session_id` is bound to authenticated principal.
- **DPoP**: Optional proof-of-possession support.
- **Multi-tenant**: `org_id` in JWT claims is used for tenant scoping.

---

## Health & Observability

- API health: `/health/live` (global)
- MCP session health: `/mcp/health`
- Prometheus metrics: `/metrics` (if enabled)

---

## Troubleshooting

- **401/403 on MCP POST**: Ensure Authorization header is sent on POST.
- **Valkey connection errors**: Verify `VALKEY_HOST` and `VALKEY_PORT`.
- **Neo4j auth failures**: Check `NEO4J_USERNAME` and `NEO4J_PASSWORD`.
- **UI not loading**: Confirm `NEXT_PUBLIC_API_URL` points to `http://localhost:8865`.

---

## What’s in the Repo

Key directories:
- `openmemory/api/` — FastAPI backend + MCP server
- `openmemory/ui/` — Next.js UI
- `openmemory/docker-compose.yml` — Full system stack
- `openmemory/run.sh` — One-command setup (vector store variants)
- `openmemory/compose/` — Compose templates for vector store selection
- `openmemory/api/app/` — main API logic, security, tools

---

## Next Steps

- Configure JWT issuance for your auth system
- Decide whether to require DPoP in production
- Add a real Valkey instance for multi-worker deployment
- Integrate your preferred MCP clients

