# Implementation Plan v9: Intelligent Development Assistant System

Version: 9
Date: 2025-12-26
Status: Draft for execution

---

## What Is New In v9

- Added explicit Phase 1 prerequisites (Tree-sitter deps, indexing scaffolding, CODE_* schema, bootstrap state).
- Added security scanning integration points (pre-index, pre-commit, CI) and secret quarantine workflow.
- Added embedding protection requirements (encryption at rest + deletion/re-embedding on GDPR requests).
- Added retrieval quality instrumentation requirements (per-stage latency + evaluation harness).
- Added lexical tokenization guidance for code (identifier splitting, field weighting).
- Clarified reranker adapter status and MCP schema alignment requirements.
- Elevated cross-repository dependency graph and breaking change detection to a defined phase.

---

## Current Implementation Snapshot (as of 2025-12-26)

- Phase 0a complete: benchmarking framework, embedder adapters, lexical decision matrix, benchmark runner and reporter.
- Phase 0b complete: OAuth 2.1 JWT validation, DPoP binding, RBAC matrix, SCIM stubs, prompt injection defenses.
- Phase 0c complete: OpenTelemetry tracing, structured logging, audit hooks, SLO tracking.
- Existing MCP tools: memory and business concepts, hybrid retrieval and RRF, graph enrichment.
- Existing data stores: Qdrant, Neo4j (OM_*), lexical interface, MCP server (FastMCP + SSE).
- Note: AXIS memory protocol and business concepts are legacy capabilities from earlier iterations. They are not a focus of v9, but may be reused opportunistically.

---

## 0. System Context and Target Capabilities

This system addresses the stateless nature of LLM assistants by providing persistent, scoped memory and code intelligence across tools and sessions. It combines:
- mem0 memory layer with A.U.D.N pipeline (Add/Update/Delete/NoOp).
- Qdrant for semantic code embeddings and similarity search.
- Neo4j for structural code relationships and graph traversal.
- MCP as a universal adapter so IDEs and clients share context and tools.

Target user-facing capabilities:
- Context-aware code generation and recall of team conventions across tools.
- Impact analysis: callers, callees, inheritance, dependency graphs, and affected files.
- Hybrid retrieval that fuses semantic, lexical, and graph signals via RRF.
- ADRs searchable and linked to code changes.
- Memory-based code reviews and onboarding via shared team and org knowledge.
- Progressive enhancement: useful responses within 30 seconds while indexing continues.
- Retrieval quality feedback loop with tuning and A/B testing.
- Cross-language API boundary linking (REST, GraphQL, gRPC, messaging).
- PR analysis and review assistance with ADR and convention checks.
- Test generation informed by call graphs and team patterns.
- Speculative retrieval and prefetching for latency reduction.
- Cross-repository dependency insights and breaking change detection.

Memory hierarchy:
- Organization -> Team -> User -> Session (with project scope where applicable).

Performance expectations for IDE use:
- Inline completion targets <200ms (stretch), chat TTFT <1s, memory retrieval <100ms target.

Deployment targets:
- Local-first with self-hosted models where possible.
- Cloud fallback only via Gemini models, with geo-residency enforcement.

---

## 1. Goals and Non-Goals

### Goals
- Persistent, scoped memory for session, user, team, project, org, and enterprise.
- Code intelligence: semantic + lexical + structural retrieval with measurable quality.
- Impact analysis, call graph, inheritance, data flow, and dependency queries.
- Progressive indexing with bootstrap status, priority queue, and partial results.
- Retrieval quality feedback loop, dashboards, and A/B testing.
- Retrieval instrumentation with per-stage latency and evaluation harness integration.
- Cross-language API boundary linking for REST, GraphQL, gRPC, and messaging.
- MCP tools that are validated, paginated, and latency-aware.
- Enterprise security: verified identity, audit logging, GDPR deletion, and policy controls.
- Developer experience tooling: CLI, query playground, dashboards.
- Cross-repository dependency graph and breaking change detection (Phase 8).

### Non-Goals (v1 scope)
- Full IDE UI implementation (APIs and MCP only).
- Perfect static analysis for all languages (best-effort AST + optional LSP).
- Full semantic cross-language resolution beyond detected API boundaries.
- Multi-repository reasoning and system-wide impact analysis (post-v1).
- AXIS memory protocol and business concepts layer; retained as legacy capabilities, not a v9 roadmap focus.

---

## 2. Constraints and Policy

### 2.1 Cloud Model Policy
- Only local/self-hosted models or Gemini cloud models are permitted.
- Gemini text-embedding-004 is deprecated; use gemini-embedding-001 for cloud fallback.
- Cloud reranking must use Gemini (semantic-ranker-fast-004) if needed.
- Cloud usage must honor geo_scope; if residency is incompatible, disable cloud fallback.
- Gemini usage requires explicit org consent and data processing agreements.
- EU deployments use regional Gemini endpoints only.

### 2.2 Data Residency
- Support geo scopes for orgs with regional requirements.
- Enforce data residency in storage and indexing jobs.
- Retrieval path enforces geo_scope; cross-region queries are denied and audited.

### 2.3 Feedback and Privacy
- Feedback events must be scoped to org and user and stored as append-only logs.
- Query content retention is capped at 30 days unless explicitly permitted by org policy.
- Aggregate metrics must be anonymized for cross-org analysis.

---

## 3. Target Architecture

```
MCP Host (IDE/Client)
  -> MCP Server (openmemory/api/app/mcp_server.py)
     -> FastAPI Backend
        -> Indexing Orchestrator (priority queue + bootstrap status)
        -> Embedding Service (multi-model + shadow pipeline)
        -> Retrieval Service (tri-hybrid + rerank + speculative prefetch)
        -> Feedback + Experiment Service (A/B tests, metrics)
        -> Memory Service (persistent + episodic)
        -> Qdrant (memories, code embeddings, ADRs)
        -> Neo4j (memory graph + code graph)
        -> Lexical Index (Tantivy or OpenSearch)
        -> Code Analysis Pipeline (Tree-sitter + optional LSP)
        -> Observability + Audit Logs
        -> CLI + Playground (API/MCP clients)
```

### 3.1 Version and Integration Requirements
- Pin container images in production; avoid floating tags like `:latest`.
- Qdrant server must support features used by `qdrant-client>=1.9.1` (current lockfile uses 1.16.x).
- Neo4j server should be pinned (5.26.x or newer) to avoid drift between environments.
- OpenSearch is the chosen lexical backend, but the current adapter is benchmark-only; production integration is required.
- MCP schema registry must stay aligned with `mcp[cli]>=1.3.0` and the tool list in Section 11.

---

## 4. Identity, Access, and Scope Model

### 4.1 Principal Schema
Each request resolves to a verified principal:
```
principal_id
user_id
org_id
enterprise_id
project_ids[]
team_ids[]
session_id
roles[] (enterprise_owner, org_owner, admin, maintainer, user, reviewer, security_admin)
scopes[] (repository:read, repository:write, embedding:read, model:select, audit:read)
geo_scope (optional)
```

### 4.2 Scope Semantics
All items carry explicit scope fields:
```
scope: session | user | team | project | org | enterprise
session_id (optional)
user_id
team_id (optional)
project_id (optional)
org_id
enterprise_id
geo_scope (optional)
```

Retrieval semantics:
- Union of all permitted scopes with precedence: session > user > team > project > org > enterprise.
- De-duplicate by content hash; keep highest-precedence result.
- Multi-team users include all team_ids unless request narrows scope.
- geo_scope inherits from org unless explicitly overridden at project or team.

### 4.3 Session and Episodic Memory
- Session scope TTL default: 24h (configurable).
- Episodic memory stored per session and summarized over time.
- Cross-tool context handoff is allowed only within session scope unless promoted to user scope.

---

## 5. Security, Compliance, and Policy Controls

### 5.1 Audit Logging (Minimum Events)
```
auth.login, auth.logout, auth.mfa_challenge
access.query, access.context_load, access.denied
access.data_export, access.tool_invoked
admin.policy_change, admin.role_change, admin.user_provision, admin.user_deprovision
scim.user_provision, scim.user_deprovision
ai.suggestion_generated, ai.model_selected
security.secret_detected, security.content_excluded, security.break_glass, security.mcp_revoked
feedback.received, feedback.aggregated, experiment.assigned
pr.analysis_performed, pr.review_comment_suggested
```

### 5.2 GDPR Data Export (SAR)
- export_user_data(user_id) returns memories, embeddings metadata, graph entities, ADRs, episodic memory, and audit summary.
- Exports are encrypted with tenant KMS keys and logged.

### 5.3 GDPR Delete
- delete_user(user_id) cascades to Qdrant, Neo4j, lexical index, ADRs, caches, and snapshots.
- Backup purge completed within 30 days; deletion status tracked per store.
- Embedding deletion requires re-indexing affected chunks to remove residual vectors.

### 5.4 Prompt Injection Defense
- Input validation with pattern library and risk scoring.
- System prompt hardening with instruction hierarchy enforcement.
- Hidden character detection in inputs.
- Context isolation: retrieved documents treated as untrusted data.
- Human-in-the-loop for high-risk operations.
- Output validation before any code execution.

### 5.5 Secret Detection Optimization (FR-011)
- Tiered scanning: fast sync scan (<20ms), async deep scan, active verification.
- Quarantine potential secrets during sync scan; async scan determines final action.
- Verified secrets trigger security incident workflow.
- Integrations: pre-commit hooks, pre-index scanning, and CI/CD enforcement.

### 5.6 SCIM Orphan Data Handling (FR-012)
- Deprovisioned users suspended within 4 hours.
- 3-day grace period before ownership changes.
- Personal memories deleted after 30-day grace period.
- Team and org memories transferred to admin owner.
- ADRs retained and anonymized.

### 5.7 Embedding Protection
- Encrypt embeddings at rest with tenant-managed KMS keys.
- Optional application-layer encryption for high-risk tenants before storage.
- Audit all embedding queries and enforce tenant isolation on vector filters.

### 5.8 AI Governance Readiness
- Maintain controls aligned to AI governance standards (e.g., ISO/IEC 42001).
- Track model selection, data usage, and risk mitigations in audit logs.

---

## 6. Storage and Data Models

### 6.1 Content-Addressable Embedding Storage (FR-004)
- Separate code chunks from embeddings to allow multi-model storage:
```
code_chunks:
  content_hash (primary key)
  payload (file_path, repo_id, language, line_start, line_end, scope, timestamps)

code_embeddings:
  chunk_hash (FK to code_chunks)
  model_id
  model_version
  embedding vector
  created_at
  UNIQUE(chunk_hash, model_id)
```

### 6.2 Embedding Collections
- Model-specific collections remain for compatibility, but code_chunks is the source of truth.
- Matryoshka embeddings are stored as multiple dimension slices when supported.

### 6.3 Feedback Storage (FR-002)
```
feedback_events:
  query_id
  user_id
  session_id
  tool_name
  outcome (accepted|modified|rejected|ignored)
  decision_time_ms
  rrf_weights
  reranker_used
  timestamp
```

### 6.4 Episodic Memory Storage (FR-005)
- New collection: episodic_memories with session_id and recency score.
- Separate from persistent scoped memory.

### 6.5 API Boundary Graph Extensions (FR-003)
- New nodes: CODE_APIEndpoint, CODE_APIClient, CODE_ProtoService, CODE_MessageTopic.
- New edges: CODE_CALLS_API, CODE_PUBLISHES_TO, CODE_SUBSCRIBES_TO.

### 6.6 Multi-Repository Graph (Phase 8)
- New nodes: CODE_Repository, CODE_APISpec.
- New edges: CODE_DEPENDS_ON, CODE_PUBLISHES_API.

### 6.7 Bootstrap State Store (FR-001)
- Track indexing progress and capabilities by repo.
- State persisted in SQLite or Postgres for restart safety.

---

## 7. Code Ingestion and Indexing

### 7.0 Phase 1 Prerequisites
- Add Tree-sitter dependencies and language bindings before starting AST parsing.
- Create `openmemory/api/indexing/` scaffolding with tests and module layout.
- Define and enforce CODE_* graph schema constraints in Neo4j.
- Add bootstrap state persistence (SQLite/Postgres) for progressive indexing.
- Implement a priority queue to process tiered indexing order.

### 7.0.1 Phase 1 Setup Tasks (before AST parsing)
- Add Tree-sitter dependencies to `pyproject.toml` with ABI-aligned pins (e.g., `tree-sitter>=0.23.0,<0.25.0`, `tree-sitter-python>=0.23.0,<0.24.0`, `tree-sitter-typescript>=0.23.0,<0.24.0`, `tree-sitter-java>=0.23.0,<0.24.0`).
- Use individual grammar packages (avoid `tree-sitter-languages`) to keep grammars current.
- Create `openmemory/api/indexing/` with `__init__.py` and `tests/` scaffolding.
- Define CODE_* Neo4j constraints and helpers (proposed: `openmemory/api/indexing/graph_schema.py`).
- Implement bootstrap state persistence (SQLite/Postgres) and migrations.
- Implement a priority queue module for tiered indexing order.

### 7.1 Progressive Enhancement and Bootstrapping (FR-001)
- Level 0 (0-30s): open file context, lexical on open files only.
- Level 1 (30s-5m): priority files indexed first, basic embeddings for hot files.
- Level 2 (5-30m): partial graph for indexed files with confidence indicators.
- Level 3 (30m+): full repo indexed.

Priority queue tiers:
- Tier 1: open files, modified in last 24h, entry points.
- Tier 2: files in current diff, hub files, related tests.
- Tier 3: referenced files by PageRank and recency.
- Tier 4: generated, vendored, archived.

### 7.2 Bootstrap Status API
- get_bootstrap_status returns coverage, capabilities, ETA, and degraded features.

### 7.3 Incremental Indexing
- Merkle tree with file-level leaves and directory nodes.
- Periodic hash scan + git diff.
- Update only changed files and symbols.
- Transactions per commit with rollback on failure.

### 7.4 AST Parsing
- Tree-sitter plugins by language.
- Initial languages: Python, TypeScript (TS/TSX), Java.
- TypeScript and TSX require separate grammar entrypoints (select by file extension).
- Java 17 syntax is supported; Java 21 pattern matching coverage may be partial.
- Parser resilience: skip malformed files, log errors, track parse_error_rate.
- Partial indexing allowed for recoverable files.

### 7.5 Cross-Language API Boundary Detection (FR-003)
- REST: detect routes and link client calls by method + path.
- GraphQL: detect schemas and link queries to resolvers.
- gRPC: link proto services to implementations and clients.
- Messaging: link producer and consumer topic names.

---

## 8. Embedding Pipeline

### 8.1 Model Selection (Policy-Compliant)
- Primary (local): Qwen3-Embedding-8B
- Co-primary (local): Nomic Embed Code
- Local fallback: Qwen3-Embedding-0.6B
- Cloud fallback (Gemini): gemini-embedding-001

### 8.1.1 Implementation Alignment
- Update mem0 Gemini defaults to `gemini-embedding-001` (current defaults still use `text-embedding-004`).

### 8.2 Shadow Embedding Pipeline (FR-004)
- Shadow models run in parallel for comparison.
- Promotion only if >=2 percent MRR improvement and <=10 percent latency regression.

### 8.3 Matryoshka Support (FR-004)
- Store multiple dimensional slices for Gemini embeddings (3072, 1536, 768).

---

## 9. Lexical Search

- BM25 with identifier-aware tokenizers.
- Bootstrap mode: lexical index limited to open files until indexing progresses.
- Incremental updates on file changes.
- Production OpenSearch integration is required; the current OpenSearch adapter is a benchmark mock.
- Tokenization guidance: CamelCase/snake_case splitting, code stopword filtering, and filename-weighted fields.

### 9.1 OpenSearch Production Integration Tasks
- Wire `openmemory/compose/opensearch.yml` into `openmemory/docker-compose.yml` (service `mem0_store`).
- Create a production client adapter (proposed: `openmemory/api/app/lexical/opensearch_client.py`).
- Implement code-aware tokenization and field weighting (identifiers, filenames, symbols).
- Add code stopword filtering configuration for source languages.
- Add integration tests against a real OpenSearch container.

---

## 10. Retrieval System

### 10.1 Tri-Hybrid Routing
- Symbol and relationship intent drive routing.
- Sparse graph fallback to hybrid or lexical.
- Weighted RRF baseline for code: vector 0.40, lexical 0.35, graph 0.25 (tuned via feedback loop).

### 10.2 Degraded Mode Metadata (FR-001)
- Every tool response includes index_coverage, confidence, and missing_sources.

### 10.3 Speculative Retrieval (FR-010)
- Cursor-based prediction and background prefetch.
- Typing pattern prediction for imports and calls.
- Session warmup on open.

### 10.4 Feedback Loop (FR-002)
- Implicit feedback emitted on every retrieval.
- Explicit feedback via MCP tool.
- Nightly RRF weight optimizer proposes new weights.

### 10.5 Reranker Implementation Status
- Reranker adapter is not implemented yet; Phase 2 includes adapter and integration work.

### 10.5.1 Reranker Implementation Plan
- Add reranker module with base interface (proposed: `openmemory/api/app/rerankers/`).
- Implement Gemini reranker adapter (semantic-ranker-fast-004) with geo-scope checks.
- Add a local cross-encoder fallback (sentence-transformers) for offline use.
- Add reranker benchmarking harness (proposed: `openmemory/api/benchmarks/rerankers/`).
- Integrate reranking into tri-hybrid retrieval (post-RRF or stage 3) behind feature flags.

### 10.6 Optional Multi-Stage Retrieval
- Use named vectors and Matryoshka dimensions for coarse-to-fine retrieval.
- Stage 1: low-dim vectors for broad recall; Stage 2: mid-dim refinement; Stage 3: full-dim rerank.

---

## 11. MCP Tooling

### 11.1 Core Code Tools
- search_code_semantic
- search_code_lexical
- search_code_hybrid
- get_symbol_definition
- find_callers
- find_callees
- find_implementations
- find_type_definition
- inheritance_tree
- dependency_graph
- impact_analysis (supports include_cross_language)
- get_symbol_hierarchy
- find_tests_for_symbol
- generate_tests (FR-008)
- explain_code (FR-007)
- search_by_signature
- find_similar_code
- explain_code_context (alias to explain_code)
- export_code_graph
- export_call_graph
- get_project_conventions

### 11.2 Index and Admin Tools
- register_repository
- unregister_repository
- trigger_reindex
- get_index_status
- get_bootstrap_status (FR-001)
- system_health
- validate_input
- get_audit_events

### 11.3 Memory and ADR Tools
- add_memory, search_memories, update_memory, delete_memory
- create_adr, update_adr, search_adr, list_adr, get_adr_by_id

### 11.4 Feedback and Quality Tools
- provide_feedback (FR-002)
- list_experiments, get_experiment_status (internal/admin)

### 11.5 PR and Review Tools (FR-009)
- analyze_pull_request
- suggest_review_comments

### 11.6 Tool Contracts
- JSON Schema for inputs and outputs in docs/mcp/schema/v1.
- Schema registry updates must include: generate_tests, provide_feedback, get_bootstrap_status, analyze_pull_request, suggest_review_comments, and impact_analysis include_cross_language.
- Pagination defaults: 20, max 100 (graph default 50).
- Error taxonomy: UNAUTHORIZED, FORBIDDEN, RATE_LIMITED, TIMEOUT, PARSE_ERROR, SERVICE_UNAVAILABLE, PARTIAL_RESULT, NOT_FOUND.

---

## 12. Performance and Resilience

- Inline completion P95 <= 300ms baseline, target 200ms with prefetch.
- Memory retrieval P95 <= 100ms.
- Graph query P95 <= 500ms.
- Prefetch cache hit rate target > 60 percent.
- Circuit breakers for embeddings, reranker, Neo4j, lexical, auth.
- Degraded modes when graph or vector store is unavailable.

---

## 13. Observability and Monitoring

- OpenTelemetry tracing with GenAI conventions.
- Structured logs include trace_id, user_id, org_id, tool_name, latency_ms.
- Retrieval quality dashboard: acceptance rate, reranker lift, RRF contribution.
- A/B testing dashboards with guardrails and auto-rollback.
- Per-stage retrieval instrumentation (embedding, lexical, graph, fusion, rerank).

---

## 14. Testing and Evaluation

- Unit tests for AST parsing, priority queue, bootstrap states.
- Integration tests for cross-language linking (REST, GraphQL, gRPC, messaging).
- Feedback pipeline tests: event emission, aggregation, privacy rules.
- Performance tests for speculative retrieval and prefetch cache.
- Security tests for secret scanning tiers and quarantine.
- Evaluation harness for retrieval quality (MRR/NDCG + regression checks).

---

## 15. Deployment and Ops

- Docker compose updates for lexical backend and reranker.
- Feature flags for embeddings, reranking, feedback, prefetch.
- Graph scaling options: per-org database, label partitioning, or federated shards.
- Read replicas for heavy graph traversals.
- Hot/cold graph partitioning and materialized views for common traversals.
- Disaster recovery and backup policies unchanged.

---

## 16. Phased Roadmap (v9)

### Phase 0d: Security Optimization + DX (2-3 weeks)
- FR-011: secret detection optimization
- FR-013: CLI, playground, dashboard templates
- Prerequisites: secret pattern database, fast sync scanner, quarantine state store.
- Governance readiness: AI risk controls and audit alignment.

Phase 0d implementation details:
- Secret detection subsystem: `openmemory/api/security/secrets/` with `patterns.py`, `fast_scanner.py`, `deep_scanner.py`, `quarantine.py`, and tests.
- Developer CLI: `openmemory/cli/` with `__main__.py` and commands (search, memory, index, graph, health, debug).
- Query playground: `openmemory/api/app/routers/playground.py` for parameter tuning and side-by-side retrieval comparison.
- Governance controls: extend audit events (e.g., `ai.model_selected`, risk mitigation logging).

### Phase 1: Code Indexing Core + Bootstrap (4-6 weeks)
- AST parsing (Python, TypeScript, Java)
- Merkle incremental indexer
- SCIP symbol IDs
- CODE_* graph projection
- Progressive indexing + bootstrap status API (FR-001)
- Cross-language API boundary detection (FR-003)
- Prerequisites: Tree-sitter deps, indexing scaffolding, CODE_* schema constraints, bootstrap state store, priority queue.

### Phase 2: Retrieval + MCP Tools (3-4 weeks)
- Tri-hybrid retrieval and routing
- Reranker integration
- Explain code tool (FR-007)
- Core MCP code tools
- Prerequisites: OpenSearch production adapter, reranker adapter, MCP schema updates.

### Phase 2.5: Feedback Integration (2-3 weeks)
- FR-002: implicit/explicit feedback, A/B tests, RRF optimizer

### Phase 3: Scoped Memory + Conversation Memory (2-3 weeks)
- Scoped memory backfill and retrieval
- Episodic memory layer (FR-005)
- SCIM orphan handling (FR-012)

### Phase 4: Performance + Flexibility (2-3 weeks)
- Speculative retrieval and prefetch (FR-010)
- Embedding model flexibility and shadow pipeline (FR-004)
- Graph scaling operationalization (FR-006)

### Phase 5: ADR + Test Generation (4-6 weeks)
- ADR heuristics and automation (FR-014)
- Test generation tool (FR-008)

### Phase 6: Visualization (2-3 weeks)
- Graph export endpoints with pagination
- Hierarchical code graph JSON schema

### Phase 7: PR Workflow (4-6 weeks)
- PR analysis tool and review comment suggestions (FR-009)
- GitHub MCP integration

### Phase 8: Cross-Repository Intelligence (4-6 weeks)
- Repository dependency graph and cross-repo impact analysis
- Breaking change detection for shared libraries and APIs
- Optional cross-repo search and recommendation workflows

---

## 17. Phase Exit Gates

- Phase 0d: sync scan <20ms, CLI and dashboards functional.
- Phase 1: parser error rate <= 2 percent, indexing success >= 99 percent, time-to-first-useful-response < 30s.
- Phase 2: NDCG@10 >= 0.55, retrieval P95 <= 120ms.
- Phase 2.5: acceptance rate tracking in place, A/B framework operational.
- Phase 3: scoped retrieval de-dup passes, episodic memory resolves references with confidence.
- Phase 4: prefetch cache hit rate >= 60 percent, P95 inline completion <= 200ms target.
- Phase 5: ADR automation precision >= 0.6, test generation applies team patterns.
- Phase 6: graph exports pass schema validation.
- Phase 7: PR analysis and review comments validated on real PRs.
- Phase 8: cross-repo dependency graph validated and breaking change detection tested.

---

## 18. Deliverables Checklist

- Unified identity, RBAC, SCIM, and code-owner policy.
- OAuth 2.1 with PKCE S256, DPoP, and claim schema.
- Policy-compliant embedding model registry with shadow pipeline.
- Extended code graph schema with API boundary nodes.
- Progressive indexing with bootstrap status API.
- Tri-hybrid retrieval with reranking and feedback loop.
- Expanded MCP tool set including explain, generate_tests, and PR analysis.
- Prompt injection defenses and secret scanning tiers.
- GDPR SAR export and delete workflows with orphan handling.
- Embedding protection controls and re-embedding on deletion requests.
- Graph scaling strategy and operational playbooks.
- Developer experience tooling (CLI, playground, dashboards).
- Retrieval instrumentation and evaluation harness.
- Cross-repository dependency graph and breaking change detection.

---

## 19. Definition of Done

The system can:
- Ingest a repo with progressive indexing and deliver results within 30 seconds.
- Answer semantic, lexical, and structural queries within latency budgets.
- Provide scoped memory retrieval with strict policy enforcement.
- Run impact analysis with cross-language API boundary links.
- Operate securely with audit logging, GDPR delete, and secret scanning tiers.
- Provide SAR exports and enforce geo-scope residency.
- Support feedback-driven retrieval tuning and A/B tests.
- Provide PR review analysis and test generation tools.
- Enforce embedding protection controls and re-index on deletion requests.
- Provide cross-repo dependency insights and breaking change detection.
