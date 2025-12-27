# Production Readiness Implementation Plan (2025, Rev 2)

Audience: New reader with no repository access
Purpose: Provide a self-contained, production-ready plan that resolves critical
gaps raised in recent reviews and is ready for phased execution.

---

## 1) System Summary

This system is an AI-assisted development platform for software teams. It
combines:
- FastAPI backend with MCP (Model Context Protocol) server
- Neo4j for code graph and cross-repo relationships
- Qdrant for vector embeddings
- OpenSearch for full-text and hybrid search
- PostgreSQL as the relational system of record (SQLite is legacy)
- Redis or Valkey for security caches and episodic memory

Current status:
- Feature phases 0-8 are implemented with ~2,841 tests.
- Many production-critical stores are in-memory.
- Auth components exist but are not enforced on routes.
- OpenSearch tri-hybrid search is implemented but not exposed via REST.
- Hardcoded secrets exist in compose files.

---

## 2) Goals (Condensed)

Security and compliance:
- Enforce OAuth 2.1 + DPoP, RBAC, and scopes on every request.
- Provide audit logs and GDPR SAR export + cascading delete.
- Enforce geo residency controls and deny cross-region access.

Performance and reliability:
- Retrieval P95 <= 120ms; graph queries P95 <= 500ms.
- Graceful degradation with circuit breakers and health checks.

Operability:
- Backups, restore verification, and DR runbooks for all stores.
- Pinned container versions and secure deployment defaults.

---

## 3) Key Decisions (Resolved)

Decision A: Multi-tenant isolation
- Default: shared infrastructure with org_id scoping and PostgreSQL RLS.
- Neo4j nodes/edges include org_id; all queries filter by org_id.
- Qdrant payload includes org_id with tenant-optimized payload index.
- OpenSearch uses tenant-aware index aliases; dedicated indices for large or
  compliance-sensitive tenants.

Decision B: Experiment persistence
- PostgreSQL is the system of record for experiments, variants, assignments.
- Redis/Valkey can cache assignments but is not authoritative.

Decision C: Redis vs Valkey
- Default: Valkey 8+ (drop-in replacement). Redis 7+ is acceptable only if
  licensing constraints are not a concern. Compose pins Valkey by default.

Decision D: MCP transport
- Use Streamable HTTP when clients support it; SSE remains as a fallback.

Decision E: JWT signing algorithm
- Default to EdDSA or ES256 for new deployments; RS256 accepted for compatibility.

---

## 4) Implementation Phases

### Phase 0.5: Infrastructure Prerequisites (Required First)
Deliverables:
- Add PostgreSQL 16+ and Valkey 8+ to compose with health checks.
- Remove hardcoded secrets; add .env.example placeholders.
- Pin container versions (or digests for production).
- Add health checks for API/MCP and UI services.

Exit criteria:
- Compose stack starts with .env.example values.
- All health checks pass within 60 seconds.

### Phase 1: Security Enforcement Baseline
Deliverables:
- FastAPI middleware and dependencies for get_current_principal().
- Per-router auth requirements (public vs authenticated endpoints).
- MCP tool-level permission checks.
- JWT + DPoP validation and replay cache (Valkey).
- RBAC enforcement with scope checks and org_id injection.
- Security headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options).
- Remove all user_id query param auth patterns.

Tests:
- 401/403 on unauthenticated/unauthorized access.
- DPoP replay prevention and JWT validation failures.
- Cross-tenant access attempts denied.

### Phase 2: Configuration and Secrets
Deliverables:
- Pydantic settings for all services and secrets.
- Secret rotation procedures and schedules:
  - JWT signing keys: rotate every 90 days
  - DB passwords: rotate every 180 days
  - External API keys: rotate every 30-90 days
- Validate required secrets at startup; fail fast.

### Phase 3: PostgreSQL Migration (Core App DB)
Deliverables:
- Replace SQLite with PostgreSQL for core app DB.
- Alembic migrations with explicit verification steps:
  - Pre-migration backups
  - Row count and checksum validation
  - Rollback procedure
- Enable pgcrypto if UUID generation is required.

Tests:
- Migration integrity, rollback, and transaction safety.

### Phase 4: Multi-tenant Data Plane Stores
Deliverables:
- PostgreSQL ScopedMemoryStore with org_id scoping and RLS.
  - RLS policy template: tenant_id = current_setting('app.current_org_id')
  - Session injection on every DB connection.
- PostgreSQL FeedbackStore with retention and aggregation queries.
- ExperimentStore with status history and assignment persistence.
- EpisodicMemoryStore in Valkey with TTL and recency decay.
- Neo4j stores using CODE_* schema and org_id constraints.
  - Cypher wrappers enforce WHERE org_id = $org_id.
- Qdrant EmbeddingStore with per-model collections and tenant payload index.
- OpenSearch tenant alias strategy documented and enforced.

Schema alignment:
- Contract test suite validating ABC methods against store implementations.
- Migration checks for field completeness and constraints.
- Breaking change policy documented.

### Phase 4.5: GDPR Compliance (Must-Have)
Deliverables:
- PII field inventory per store.
- SAR export orchestrator (Postgres, Neo4j, Qdrant, OpenSearch, Valkey).
- Cascading delete with explicit dependency order and audit trail.
- Backup purge strategy via crypto-shredding or retention tracking.
- SAR response format specification.

### Phase 5: API Route Wiring
Deliverables:
- REST routes for OpenSearch tri-hybrid search.
- Cross-repo routes (registry, symbol resolution, dependency graph, impact).
- Feedback + experiments routes.
- Scoped + episodic memory routes.
- Input validation and query limits on all user inputs.

### Phase 6: Operability and Resilience
Deliverables:
- /health/live, /health/ready, /health/deps endpoints.
- Circuit breakers per dependency (Neo4j, Qdrant, OpenSearch, reranker).
- Degraded-mode response schema.
- Rate limiting design:
  - Token bucket or sliding window
  - Per-endpoint and per-org quotas
  - X-RateLimit-* response headers
- OpenTelemetry instrumentation + trace correlation.
- Centralized logging (ELK/Loki) and alert thresholds.

### Phase 7: Deployment, DR, and Hardening
Deliverables:
- Backup/restore runbooks for all stores.
- Nightly restore verification to staging.
- Vulnerability scanning in CI (Trivy/Snyk).
- Container hardening (non-root, read-only FS, distroless where feasible).
- Blue-green or canary deployment playbook.

### Phase 8: Scale-Out (Deferred, Post-MVP)
Deliverables:
- Hot/cold graph partitioning strategy in Neo4j.
- Read replicas and routing policy for graph and relational stores.

---

## 5) Test Strategy and Estimates

Required test categories:
- Integration tests with real containers.
- Multi-tenant isolation fuzzing.
- Auth boundary tests (JWT/DPoP/RBAC).
- Migration verification tests.
- Chaos/failure tests for dependency outages.
- E2E workflows (indexing -> retrieval -> feedback).

Estimated new tests:
- 920 to 1,130 tests depending on DR/chaos scope.

---

## 6) Risk Register (Top Items)

- Tenant leakage: enforce RLS, org_id filters, query audit logging.
- Data loss during migration: strict runbooks and verification.
- Dependency outages: circuit breakers and degraded mode.
- Schema drift: contract tests and migration checks.
- Security regressions: mandatory auth dependency per route and tool.

---

## 7) Success Criteria

- All tests passing with persistent stores.
- Auth enforced for every non-public route and MCP tool.
- Health checks report readiness within 30 seconds.
- No hardcoded secrets in compose or code.
- Verified backup/restore for all stores.
- GDPR SAR and deletion workflows validated end-to-end.

