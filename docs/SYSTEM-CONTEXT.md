# System Context: Intelligent Development Assistant

Purpose
This system is a production-grade development assistant for software teams. It
combines persistent memory, code intelligence, and structured retrieval to
power IDEs and clients via MCP (Model Context Protocol) and REST APIs.

Documentation System

Development is tracked across multiple LLM sessions using:

- `docs/IMPLEMENTATION-PLAN-PRODUCTION-READINESS-2025-REV2.md` - Master plan (immutable goals)
- `docs/IMPLEMENTATION-PROGRESS-PROD-READINESS.md` - Progress tracker with test counts and daily log
- `docs/CONTINUATION-PROMPT-PHASE[N]-*.md` - Current phase actionable context
- `docs/CONTINUATION-PROMPT-TEMPLATE.md` - Template for creating future phase prompts

What has been implemented so far (feature phases 0-8 complete, production readiness in progress)

- Current test count: 3,459 (see Progress file for breakdown)
- Code indexing and CODE_* graph projection using Tree-sitter and SCIP symbols.
- Tri-hybrid retrieval (semantic + lexical + graph) with RRF fusion and reranker
  integration.
- Feedback loop and A/B testing framework with PostgreSQL persistence.
- Scoped memory and episodic memory models with deterministic scope precedence.
- Performance features (prefetch cache, embedding pipeline, graph scaling).
- ADR automation and test generation tools.
- Visualization endpoints for graph export and pagination.
- PR workflow tooling (diff parsing, review suggestions, GitHub MCP).
- Cross-repository intelligence (registry, symbols, dependencies, impact).

Production readiness completed (Phases 0.5-7, 4.5)

- Phase 0.5: Infrastructure (PostgreSQL, Valkey, health checks) - 37 tests
- Phase 1: Security (JWT, DPoP, RBAC, security headers) - 99 tests (MCP auth deferred)
- Phase 2: Configuration (Pydantic settings, secret rotation) - 13 tests
- Phase 3: PostgreSQL migration (tenant isolation, migration utilities) - 52 tests
- Phase 4: Multi-tenant stores (Feedback, Experiment, Episodic, Qdrant, OpenSearch) - 125 tests
- Phase 4.5: GDPR compliance (PII inventory, SAR export, cascading delete, audit) - 85 tests
- Phase 5: API routes (Feedback, Experiments, Search routers) - 67 tests
- Phase 6: Operability (health, logging, tracing, metrics, circuit breakers, rate limiting) - 56 tests
- Phase 7: Deployment/DR (backup runbooks, verifier, CI security scan, container hardening) - 47 tests

Current production gaps (see Progress file for latest status)

- MCP SSE auth pending (Phase 1 blocker - deferred)
- Scale-out deferred (Phase 8 - hot/cold partitioning, read replicas)

Technology stack
- Backend: FastAPI with MCP server integration.
- Graph store: Neo4j for CODE_* graph and cross-repo relationships.
- Vector store: Qdrant for embeddings and similarity retrieval.
- Full-text search: OpenSearch for lexical and hybrid retrieval.
- Relational DB: PostgreSQL 16+ (primary), SQLite fallback for local development.
- Cache/session: Valkey 8+ for episodic memory and security caches (DPoP replay).
- Observability: OpenTelemetry tracing, structured logging, audit hooks.
- Security: OAuth 2.1 + PKCE, JWT validation, DPoP, RBAC, prompt-injection defenses.

System goals (condensed from FINAL-SYSTEM-GOALS.md)
- User value: useful results within 30 seconds of opening a repo, consistent
  behavior across IDEs/clients, accurate impact analysis with confidence.
- Code intelligence: CODE_* graph, cross-language API boundary linking, tri-
  hybrid retrieval, degraded mode during partial indexing.
- Memory: scoped memory (session/user/team/project/org/enterprise) with
  deterministic precedence and episodic session context.
- Retrieval quality: implicit + explicit feedback, A/B testing, automated RRF
  weight tuning, dashboards for quality metrics.
- Developer workflow: explain code, generate tests, PR analysis, review comments.
- Performance: inline completion P95 <= 200ms, retrieval P95 <= 120ms, graph
  queries P95 <= 500ms, graceful degradation under failures.
- Security/compliance: enforce OAuth 2.1/DPoP/RBAC on every request, secret
  scanning, audit logging, GDPR SAR export and cascading deletes.
- Operability: circuit breakers, backup/restore for all stores, pinned service
  versions, hot/cold graph partitioning, read replicas at scale.
- Observability: OpenTelemetry traces, SLOs, cost/quality dashboards, audit logs.
- DX: CLI tools, query playground, deployable dashboards.

Intended deployment modes
- Local-first development via docker-compose.
- Enterprise deployments with strict security, compliance, and audit controls.

