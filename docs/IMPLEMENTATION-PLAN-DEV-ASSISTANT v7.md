# Implementation Plan v7: Intelligent Development Assistant System

This plan is self-contained and defines architecture, security, data models, indexing, retrieval, tooling, operations, and delivery phases to evolve OpenMemory into a production-grade development assistant.

---

## 0. System Context and Target Capabilities

This system addresses the stateless nature of LLM assistants by providing persistent, scoped memory and code intelligence across tools and sessions. It combines:
- **mem0 memory layer** with the A.U.D.N pipeline (Add/Update/Delete/NoOp) to maintain accurate, consolidated knowledge over time.
- **Qdrant** for semantic code embeddings and fast similarity search.
- **Neo4j** for structural code relationships (calls, inheritance, dependencies) and graph traversal.
- **MCP** as a universal adapter so IDEs and clients (VS Code, Cursor, Claude Desktop, local hosts) can share the same context and tools.

Target user-facing capabilities:
- Context-aware code generation and recall of team conventions across tools.
- Impact analysis: callers/callees, inheritance, dependency graphs, and affected files.
- Hybrid retrieval that fuses semantic, lexical, and graph signals via RRF.
- Architecture Decision Records (ADRs) that are searchable and linked to code changes.
- Memory-based code reviews and onboarding via shared team/org knowledge.

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
- MCP tools that are validated, paginated, and latency-aware.
- Enterprise security: verified identity, audit logging, GDPR deletion, and policy controls.

### Non-Goals (v1 scope)
- Full IDE UI implementation (APIs and MCP only).
- Perfect static analysis for all languages (best-effort AST + optional LSP).
- Cross-language call resolution beyond explicit links.
- IDE context capture is enabled via MCP resources; rendering/UI remains out of scope.

---

## 2. Constraints and Policy

### 2.1 Cloud Model Policy
- Only local/self-hosted models or Gemini cloud models are permitted.
- Gemini `text-embedding-004` is deprecated; use `gemini-embedding-001` for any cloud fallback.
- Cloud reranking must use Gemini (e.g., `semantic-ranker-fast-004`) if needed.
- Cloud usage must honor geo_scope; if residency is incompatible, disable cloud fallback.
- Gemini usage requires explicit org consent and data processing agreements.
- EU deployments use regional Gemini endpoints (Germany/Netherlands/France/Belgium) only.

### 2.2 Data Residency
- Support geo scopes for orgs with regional requirements.
- Enforce data residency in storage and indexing jobs.
- Retrieval path enforces geo_scope; cross-region queries are denied and audited.

---

## 3. Target Architecture

```
MCP Host (IDE/Client)
  -> MCP Server (openmemory/api/app/mcp_server.py)
     -> FastAPI Backend
        -> Qdrant (memories, code embeddings, ADRs)
        -> Neo4j (memory graph + code graph)
        -> Lexical Index (Tantivy or OpenSearch)
        -> Code Analysis Pipeline (Tree-sitter + optional LSP)
        -> Reranker (local cross-encoder or Gemini)
        -> Observability + Audit Logs
```

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
Merge strategy:
- If same content hash appears in multiple scopes, prefer higher scope unless lower scope has higher authority.
- Add `authority` property (system|admin|user) to resolve conflicts deterministically.

### 4.3 Token Validation Requirements (OAuth 2.1)
- Authorization Code + PKCE only; no implicit flow.
- PKCE uses S256 code challenge method.
- DPoP is mandatory to prevent token replay.
- Validate JWT signature via JWKs (cache + rotation).
- Mandatory claim checks: iss, aud, exp, iat, nbf, sub.
- Enforce revocation via token introspection or denylist cache.
- Map scopes -> roles -> permissions on every request.
- Refresh token rotation and short-lived access tokens (<= 1h).
- Issuer allowlist per org with configured authorization server endpoints.
- Resource Indicators (RFC 8707) required for HTTP transports.

### 4.4 OAuth Claim Schema
Required JWT claims:
```
sub, iss, aud, exp, iat, nbf,
org_id, enterprise_id, roles[], scopes[],
team_ids[], project_ids[], geo_scope
```
Optional claims:
```
session_id, user_id, token_id
```
DPoP confirmation (cnf) claim required when DPoP is enabled.

### 4.5 Enforcement Points
- MCP server validates JWT/OAuth2 and derives principal.
- API layer enforces scope and role checks for each request.
- Data access layer applies mandatory filters for Qdrant, Neo4j, and lexical search.
- Audit logger records all access and policy decisions.

### 4.6 Role Permission Matrix
Roles are mapped to permissions with least-privilege defaults:
| Role | repo_read | repo_write | memory_write | admin_policy | audit_read | model_select |
|------|-----------|------------|--------------|--------------|-----------|--------------|
| enterprise_owner | yes | yes | yes | yes | yes | yes |
| org_owner | yes | yes | yes | yes (org) | yes | yes |
| admin | yes | yes | yes | limited | no | yes |
| maintainer | yes | yes | yes | no | no | no |
| reviewer | yes | no | no | no | no | no |
| user | yes | no | personal only | no | no | no |
| security_admin | no | no | no | no | yes | no |
Scopes are required in addition to role permissions; deny if either role or scope is missing.

### 4.7 Session Scope TTL
- Session scope TTL default: 24h (configurable).
- Cleanup job removes expired session memories and cache entries.
- Session scope applies to conversational memory only, not code indexing data.

### 4.8 Enterprise Provisioning
- SCIM 2.0 for automated user provisioning/deprovisioning.
- SSO required for enterprise tenants (SAML 2.0 or OIDC).
- Code owners for path-based permissions (CODEOWNERS-style).
- SCIM groups map to IdP roles and OAuth scopes.
- CODEOWNERS affects write/admin actions; read access remains scope-based.
- Orphan detection job suspends unlinked users within 72 hours.

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
```
Retention: 1-7 years configurable; SIEM export support.
Audit integrity: append-only storage with hash chaining and WORM retention where required.
Audit includes failed operations and denied access attempts.
Audit chain verification runs hourly with external witness logs.

### 5.2 GDPR Data Export (SAR)
- `export_user_data(user_id)` returns memories, embeddings metadata, graph entities, and audit summary.
- Exports are encrypted with tenant KMS keys, logged, and delivered via secure channel.

### 5.3 GDPR Delete
- `delete_user(user_id)` cascades to Qdrant, Neo4j, lexical index, ADRs, caches, and snapshots.
- Deletion audit event preserved without content.
- Backup purge completed within 30 days; deletion status tracked per store.

### 5.4 Prompt Injection Defense
- Input validation with pattern library and risk scoring.
- System prompt hardening with instruction hierarchy enforcement.
- Hidden character detection in inputs.
- Context isolation: retrieved documents are treated as untrusted data.
- Human-in-the-loop for high-risk operations.
- Output validation before any code execution.
- Sandbox for generated code execution.
- Detection targets: >=95% detection rate, <=1% false positives on adversarial corpus.

### 5.5 Content Exclusion Policies
- Precedence: org policy > repo config > CODEOWNERS > gitignore.
- Path-based exclusions and explicit no-index tags in files or directories.

### 5.6 MCP Allowlisting
- Enterprise allowlist of MCP servers and tools.
- Central registry and policy validation.
- Server attestation via version pinning and signed hashes.
- Controlled update process for allowlisted servers (approval + staged rollout).
- Emergency revocation propagates within 1 hour.

### 5.7 Secret Detection Workflow
- Sync scan before embedding; block or redact on detection.
- Severity-based actions: block, quarantine, alert, or allow with redaction.
- Re-scan on new secret patterns or policy updates.
- Budget: <= 50ms per file for sync scan; slow paths fall back to async quarantine.
- Escalation: notify security channel, create incident ticket for verified secrets.
- Primary scanner: TruffleHog v3 with active verification; supplement with Gitleaks.

### 5.8 Compliance Targets
- SOC 2 Type II and ISO 27001 readiness for enterprise deployments.
- ISO 42001 alignment for AI governance (policies, risk controls, auditability).
- Maintain GDPR Article 30 data processing records.

### 5.9 Code Provenance and License Compliance
- Track snippet provenance and detect near-duplicates against indexed repos.
- Flag potential license conflicts for human review before use.

### 5.10 Service-to-Service Security
- mTLS or service mesh for internal service communication.
- Enforce least-privilege credentials for Qdrant, Neo4j, and lexical backend.
- Certificate rotation automated every 30 days with overlap.

### 5.11 Emergency Access
- Break-glass access is time-bound (max 4h), dual-approved, and fully audited.
- Tokens are single-use, expire on completion, and trigger real-time SIEM alerts.

---

## 6. Storage and Data Models

### 6.1 Vector Index Placement
- <50M vectors: Neo4j vector index is acceptable for simpler ops.
- >=50M vectors: Qdrant required for performance and scale.
- Migration when crossing threshold: dual-write, backfill, validate, cutover.

### 6.2 Data Isolation and Consistency
- Defense-in-depth: mandatory scope filters in code + automated tests.
- Optional per-org collections for high-risk tenants.
- Optional separate Neo4j databases per enterprise for strict isolation.
- Define consistency model: eventual consistency with saga + reconciliation jobs.
- Migration strategy: dual-write, backfill, verify, cutover with rollback.
- Per-tenant KMS keys for restricted classifications.
- Geo partitioning at storage layer for regional isolation.

### 6.3 Qdrant Collections
Use one embedding dimension per collection. Each model has its own collection:
- `openmemory` (existing)
- `code_embeddings_<model>_<dims>`
- `adr_memories`
- `business_concepts`

Model registry config:
```
CODE_EMBEDDING_MODEL=qwen3-embedding-8b
CODE_EMBEDDING_DIMS=1024
CODE_EMBEDDING_COLLECTION=code_embeddings_qwen3_1024
```

### 6.4 Code Embedding Payload Schema
Required fields:
```
repo_id, file_path, language, symbol_type, symbol_name,
chunk_id, code_text, line_start, line_end,
scope, org_id, enterprise_id, project_id, team_id, user_id, session_id,
geo_scope,
model_name, model_version, created_at, updated_at
```

### 6.5 Neo4j Code Graph (Separated Context)
Prefix all code labels and relationships:
 - CODE_* namespace reserved for code graph; OM_* remains for memory graph.

Core node labels:
- `CODE_Repo`, `CODE_File`, `CODE_Class`, `CODE_Function`, `CODE_Method`, `CODE_Module`, `CODE_Package`

Extended node labels:
- `CODE_Interface`, `CODE_Type`, `CODE_TypeAlias`, `CODE_Enum`, `CODE_Constant`
- `CODE_Parameter`, `CODE_Field`, `CODE_Variable`
- `CODE_Import`, `CODE_Call`, `CODE_Test`

Required properties (examples):
- CODE_File: path, language, repo_id, org_id, last_modified
- CODE_Function: name, signature, file, line_start, line_end, repo_id
- CODE_Class: name, file, docstring, repo_id
Property types:
- path/file/signature/name/docstring: String
- line_start/line_end: Integer
- last_modified: DateTime (ISO 8601)

Core relationships:
- `CODE_DEFINES`, `CODE_CONTAINS`, `CODE_CALLS`, `CODE_IMPORTS`, `CODE_INHERITS_FROM`, `CODE_PART_OF`

Extended relationships:
- `CODE_IMPLEMENTS`, `CODE_EXTENDS`, `CODE_DECLARES`, `CODE_REFERENCES`
- `CODE_READS`, `CODE_WRITES`, `CODE_DATA_FLOWS_TO`
- `CODE_CONTROL_FLOWS_TO`, `CODE_INSTANTIATES`, `CODE_INVOKES`
- `CODE_TESTS`, `CODE_COVERS`, `CODE_MOCKS`

Edge properties:
```
confidence: definite | probable | possible
confidence_score: 0.0-1.0
source: static | dynamic | inferred
location: { line, column }
repo_id, org_id, project_id, team_id
```
DATA_FLOWS_TO semantics:
- Direction is source -> sink.
- Optional properties: taint_type, via_callsite, sanitizer_applied.

### 6.6 Graph Indexes and Constraints
- Unique: (repo_id, path) on CODE_File.
- Unique: (repo_id, name, file) on CODE_Function/CODE_Method.
- Unique: (repo_id, name, file) on CODE_Class/CODE_Interface/CODE_Type.
- Indexes: repo_id, org_id, project_id, team_id on all CODE_* nodes.
- Relationship indexes for CODE_CALLS and CODE_DATA_FLOWS_TO by repo_id.

### 6.7 ADR Payload Schema (adr_memories)
Required fields:
```
adr_id, title, status, decision, scope, org_id, enterprise_id,
project_id, team_id, related_code[], created_at, updated_at
```

---

## 7. Code Ingestion and Indexing

### 7.1 Repo Registration
```
repo_id, repo_name, root_path, default_branch, languages[]
```
Prefer AGENTS.md (industry standard); support CLAUDE.md via symlink.

### 7.2 Incremental Indexing
- Merkle tree with file-level leaves and directory nodes (respects ignore rules).
- Periodic hash scan (every 10 minutes) + git diff for changes.
- Update only changed files and symbols.
- Symbol IDs follow SCIP-style format: `<scheme> <package> <descriptor>+`.
- Transactions per commit; atomic delete/insert; rollback on failure.
- Concurrent indexing via queue + worker pool with repo partitioning.
- Indexing SLO: file-to-searchable P95 <= 30s for active files.
- Rename detection maps old -> new symbol IDs with migration records.
- Symbol ID versioning retained for 90 days to preserve links during refactors.

### 7.3 AST Parsing
Language plugins:
- Phase 1: Python
- Phase 2: TypeScript
- Phase 3: Java + Go
- Post-v1: Rust (capacity permitting)

Parser resilience:
- Try/catch on Tree-sitter parsing.
- Skip malformed files; log parse errors.
- Track parse_error_rate.
- Partial indexing for recoverable files; error caps per repo.
Language plugin interface:
- parse(file) -> AST
- symbols(AST) -> symbol list
- references(AST) -> edges
Incremental parsing uses `ts_tree_edit()` + `ts_parser_parse()` with old tree for structural sharing.

### 7.4 Call Graph and Data Flow
- Build call edges with confidence scores.
- Create call site nodes (`CODE_Call`) for precise impact analysis.
- Add data-flow edges where supported (READS, WRITES, DATA_FLOWS_TO).
- Use Pysa (Python), Semgrep (fast patterns), and CodeQL (deep scans) where needed.

### 7.5 Code Intelligence Indexers
- Prefer SCIP indexers where available (scip-python, scip-typescript, scip-java).
- Use LSP servers as fallback for symbol extraction.
- PyCG is archived; evaluate JARVIS for Python call graphs.
- SootUp is beta; use original Soot for production Java.

### 7.6 Chunking Strategy
- AST-based chunking, 1500-2000 characters per chunk.
- Never split mid-function.
- Include signature + docstring in embedding input.
- Use content hash for cache keys and dedup.
- Optional 10-15% overlap for long functions to preserve context.
- Store provenance metadata (commit_sha, chunk_hash) for debugging.
- If a single AST node exceeds the size budget, split at safe block boundaries with explicit markers.

---

## 8. Embedding Pipeline

### 8.1 Model Selection (Policy-Compliant)
Benchmark candidates:
- Primary (local): Qwen3-Embedding-8B
- Co-primary (local): Nomic Embed Code
- Local fallback: Qwen3-Embedding-0.6B
- Cloud fallback (Gemini): gemini-embedding-001
- Optional lightweight: Jina Embeddings v3 (Apache 2.0)
- Dataflow-aware options: GraphCodeBERT, UniXcoder
- External benchmark reference (no integration due to policy): voyage-code-3

Metrics: MRR, NDCG (CodeSearchNet), latency, cost.
Selection criteria:
- Primary model chosen per language based on internal eval set.
- Co-primary used only if it beats primary by >= 2% on target queries.
- Embedding dimensions standardized per collection; no cross-dim mixing.
- Record baseline MRR/NDCG for each candidate before selection.
Gemini embeddings use separate collections due to dimension mismatch.
Gemini `gemini-embedding-001` outputs 3072 dims with Matryoshka reduction (1536/768); choose one per collection.
Qwen3 and Nomic support 32K contexts and broad language coverage; prefer them for code-heavy repos.
Dataflow-aware embeddings (GraphCodeBERT/UniXcoder) run in separate collections for optional dataflow-sensitive retrieval.
Promote GraphCodeBERT/UniXcoder to production if they outperform on dataflow-heavy internal evals.

### 8.2 Embedding Execution
- Circuit breaker parameters:
  - FAILURE_THRESHOLD=5
  - RESET_TIMEOUT=90s
  - HALF_OPEN_REQUESTS=1
- Cache embeddings (Redis, default 24h TTL, configurable up to 7 days).
- Local-first fallback; Gemini only if cloud is required.
- Warmup: load models at startup and run probe embeddings before serving traffic.
- Cold-start optimization: use safetensors and memory-mapped weights where possible.

### 8.3 Security Controls
- DLP scan before embedding.
- Redact secrets and credentials.
- Run secret scanners (TruffleHog, Gitleaks, detect-secrets) before embedding.
- Encryption at rest for restricted classifications using KMS-managed keys.

### 8.4 Model Versioning and Migration
- Embedder adapters normalize inputs across models.
- Model version recorded in payload; collection naming encodes model + dims.
- Re-embed jobs are versioned and can be rolled forward or rolled back.
- Model-agnostic adapter layer allows swapping providers without client changes.
- Shadow testing: dual-write to new collection, compare metrics before cutover.

---

## 9. Lexical Search

### 9.1 Backend Decision (Phase 0)
Evaluate Tantivy vs OpenSearch on:
- Index size at 100K and 1M symbols
- Query latency
- Memory footprint
- Operational complexity
- RRF support and neural search features
Decision matrix:
- Latency (40%), Ops complexity (20%), Scalability (20%), Feature support (20%).
Lexical default: BM25 with identifier-aware tokenizers; avoid SPLADE for code.
Optional: Qdrant BM42 hybrid for dense+sparse fusion if Query API used.
Scale guidance:
- Tantivy for <=1M docs per node; OpenSearch for multi-tenant or >1M docs.

### 9.2 Index Fields
`file_path`, `symbol_name`, `signature`, `docstring`, `code_text`, `language`
Refresh strategy:
- Incremental updates on file changes.
- Nightly full index validation and compaction.

---

## 10. Retrieval System (Tri-Hybrid + Rerank)

### 10.1 Query Routing
Decision rules:
1. Known symbol name -> GRAPH_PRIMARY
2. Relationship intent keywords -> GRAPH_PRIMARY
3. Exact identifier query or symbol-only query -> LEXICAL_PRIMARY
4. Otherwise -> HYBRID
Detection methods:
- Symbol lookup via in-memory index and language-aware regex.
- Keyword list with language-specific expansions.
- Graph coverage check; fallback to HYBRID if repo graph is sparse.
- Mandatory intent classifier for short natural-language queries.
Geo enforcement:
- Validate geo_scope before routing; deny or localize queries if required.
Sparse graph threshold:
- <60% files parsed or <0.5 call edges per symbol.

### 10.2 RRF Fusion
- Default k=60 (tune 20-100).
- Dynamic weights by query type:
  - Natural language -> Vector 0.5, Lexical 0.2, Graph 0.3
  - Exact match -> Vector 0.2, Lexical 0.5, Graph 0.3
  - Cross-file navigation -> Vector 0.3, Lexical 0.2, Graph 0.5
- Query type classification via rules + lightweight classifier (mandatory).
Hybrid execution can use Qdrant Query API (>= 1.16.2) with server-side RRF/DBSF.
Weights validated via A/B tests with statistical significance.

### 10.3 Two-Stage Retrieval
- Stage 1: top 50-100 results via tri-hybrid (default 50).
- Stage 2: rerank to top 10-20 using a local cross-encoder reranker.
- If no local reranker available, use Gemini for reranking only.
- Skip reranking for high-confidence simple queries (exact match or high lexical score).
Online default: Qwen3-Reranker-0.6B for latency; 4B reserved for async/offline.
Latency-critical mode: reduce Stage 1 to 25 and Stage 2 to 5.
Rerank thresholds:
- Skip if top-1 lexical score >= 0.95 and query length <= 3 tokens.
Rerank skip quality is monitored and adjusted based on precision drift.
Time budget allocation:
- Baseline P95 300ms: retrieval <= 120ms, rerank <= 150ms (0.6B), formatting <= 30ms.
- Offline/async 4B rerank has no strict latency budget.
Graph traversal tools follow graph SLOs and are not bound to search retrieval budgets.

### 10.4 Reranking Candidates
- Quality-critical (async/offline): Qwen3-Reranker-4B.
- Low-latency (online): Qwen3-Reranker-0.6B.
- Optional: Jina rerankers only if licensing permits (CC-BY-NC constraints).
- Cloud: Gemini semantic-ranker-fast-004 (policy-compliant).
- Use 4B for offline/async reranking when latency budgets cannot be met.

### 10.5 Result Metadata
All responses include:
```
source = vector|lexical|graph|hybrid
degraded_mode = true|false
missing_sources = [vector|lexical|graph]
confidence_adjustment = -0.2..0.2
```

### 10.6 Optional Multi-Vector Retrieval
- Qdrant multivector support enables ColBERT-style retrieval.
- Use only after code-specific fine-tuning and storage impact review.

---

## 11. MCP Tooling

### 11.1 Core Query Tools
- `search_code_semantic` (optional: language, repo_id)
- `search_code_lexical` (optional: language, repo_id)
- `search_code_hybrid` (optional: language, repo_id)
- `get_symbol_definition`
- `find_callers`
- `find_callees`
- `find_implementations`
- `find_type_definition`
- `inheritance_tree`
- `dependency_graph`
- `impact_analysis`
- `get_symbol_hierarchy`
- `find_tests_for_symbol`
- `get_recent_changes`
- `search_by_signature`
- `find_similar_code`
- `explain_code_context`
- `export_code_graph`
- `export_call_graph`
- `get_project_conventions`

### 11.2 Index and Admin Tools
- `register_repository`
- `unregister_repository`
- `trigger_reindex`
- `get_index_status`
- `system_health`
- `validate_input` (prompt-injection test harness)
- `get_audit_events`

### 11.3 Memory and ADR Tools
- `add_memory`, `search_memories`, `update_memory`, `delete_memory` (scope-aware)
- `create_adr`, `update_adr`, `search_adr`, `list_adr`, `get_adr_by_id`

### 11.4 Pattern and Review Tools
- `find_recurring_patterns`
- `detect_code_smells`
- `get_similar_past_issues`
- `get_test_coverage`

### 11.5 Tool Contracts
Each tool must define:
- JSON Schema for inputs and outputs.
- Schemas stored in versioned registry (e.g., docs/mcp/schema/v1).
- Draft schemas: `docs/mcp/schema/v1/tools.schema.json`.
- Tool-level permissions (role + scope).
- Pagination: default limit 20, max 100 (graph default 50); cursor-based for traversals.
- Cursor format: opaque base64 token with expiry and scope binding.
- Timeout budgets (search: 200ms, graph: 500ms, impact: 2000ms).
- Export/graph dump tools: 5000ms with streaming.
- Error taxonomy: UNAUTHORIZED, FORBIDDEN, RATE_LIMITED, TIMEOUT,
  PARSE_ERROR, SERVICE_UNAVAILABLE, PARTIAL_RESULT, NOT_FOUND.
- Streaming responses for large graph or search results when supported.
- Streaming schema: chunked SSE with `chunk_id`, `payload`, `has_more`.
- Tool annotations enforced by middleware.
- Annotation mapping: search tools readOnlyHint=true; delete_memory/destructive tools destructiveHint=true; register/reindex idempotentHint=true.
- Rate limits defined per org/user with RATE_LIMITED responses.
- Default limits: 500 requests/hour per user, 5,000/hour per org (configurable).
- Numeric error codes align with HTTP (401, 403, 408, 429, 503).

### 11.6 MCP Spec Alignment (2025-11-25)
- OAuth 2.1 mandatory for production deployments (PKCE S256 + resource indicators).
- Streamable HTTP transport replaces HTTP+SSE; single endpoint supports GET/POST, SSE optional for streaming.
- Session lifecycle: create, heartbeat (30s), close.
- Tool annotations are optional hints (readOnlyHint, destructiveHint, idempotentHint) and must be enforced server-side.
- Clients treat annotations as untrusted unless server attestation is verified.
- JSON-RPC batching with max batch size (<= 20) and per-request timeout.
- Endpoint versioning: `/mcp/v1/stream`.
- Resources and prompts provided for discovery.
- Support tasks, parallel tool calls (max concurrency 4), sampling with tools, and extensions framework.
 - Task lifecycle: create -> progress -> complete -> cancel.
Middleware chain: auth -> rate limit -> authz -> validation -> execution -> audit.

### 11.7 MCP Servers to Evaluate
- `mcp-language-server` for LSP-backed symbols.
- `mcp-server-tree-sitter` for multi-language AST analysis.
- `mcp-neo4j-cypher` in read-only mode where required (verify read-only enforcement).
- `mcp-server-qdrant` requires custom embedding integration for Qwen3/Nomic (FastEmbed-only by default).
- `mcp-server-github` for repo metadata and PR context.

---

## 12. Performance and Resilience

### 12.1 Budgets and SLOs
- Inline completion baseline: P95 <= 300ms, P99 <= 450ms, cold-start <= 500ms.
- Target: P95 <= 200ms with aggressive caching and warm pools.
- Memory retrieval: P95 <= 100ms target, P95 <= 120ms initial; P99 <= 200ms, cold-start <= 250ms.
- Graph query: P95 <= 500ms, P99 <= 800ms, cold-start <= 1000ms.
- File-to-searchable: P95 <= 30s for active files.
- Chat TTFT: P95 <= 1s.
- Readiness probes gate traffic until warmup passes cold-start budgets.

### 12.2 Caching Strategy
| Cache | TTL | Invalidation |
|------|-----|--------------|
| Embeddings | 1-7 days | content hash change |
| AST trees | session | file mtime |
| Query results | minutes | LRU + content hash |
| Graph traversals | hours | schema or edge update |

Tiering:
- L1: in-process memory cache for hot symbols.
- L2: distributed cache (Redis) for embeddings and query results.
- L3: vector store / graph store as source of truth.
Cache keys include repo_id, commit_sha, scope, and model_version.
Prewarm L1 caches for hot paths on startup and after deploys.
Stampede protection via single-flight locks or probabilistic early expiration.

### 12.3 Fallback Matrix
| Failure | Primary Fallback | Secondary |
|---------|------------------|-----------|
| Neo4j down | Vector + Lexical | Lexical only |
| Vector DB down | Lexical + Graph | Error + retry |
| Lexical down | Vector + Graph | Vector only |
| Embedding down | Cached embeddings | Skip semantic |

Additional resilience:
- Circuit breakers for embeddings, reranker, Neo4j, lexical, and auth validation.
- Cascading failure handling: limit fan-out, degrade to minimal mode.
- Rate limits for API and MCP endpoints by user and org.
- Retry policy: exponential backoff with max 3 attempts and jitter.
- Per-service breaker thresholds defined and tuned in staging.

---

## 13. Observability and Monitoring

- OpenTelemetry tracing for all requests.
- OpenTelemetry v1.37+ with GenAI semantic conventions (token usage, TTFT).
- SLO tracking with alerts on P95/P99 and cold-start regressions.
- Quality monitoring: retrieval acceptance rate, rerank delta, query failure rate.
- Cost monitoring for embeddings and reranking.
- Custom benchmark suite aligned to internal repos and query patterns.
- Distributed tracing: W3C TraceContext propagation across Qdrant, Neo4j, and lexical services.
- Alerting: SLO burn-rate alerts (fast and slow burn) with on-call escalation.
- Default thresholds: fast burn 2x over 1h, slow burn 4x over 6h.
- Error budget policy: at 50% burn, freeze non-critical deployments.
- Structured logs include: trace_id, user_id, org_id, repo_id, tool_name, latency_ms.

---

## 14. Testing and Evaluation

### Unit
- AST parsing per language
- Router rules
- Scope enforcement and RBAC
- Embedding pipeline, RRF fusion, Merkle tree updates, content exclusion
Coverage targets: unit >= 80%, integration >= 70%, critical paths with mutation tests.

### Integration
- Qdrant indexing + retrieval with ground truth
- Neo4j graph traversal correctness
- Lexical search consistency
- MCP tool schema validation and error handling
- Use fixed test repos with known call graphs and expected outputs.

### E2E
- Repo ingest -> query -> validate results
- Impact analysis with known call graph
- Scoped memory write/read/delete across user/team/org

### Benchmarks
- CodeSearchNet MRR/NDCG plus internal evaluation set per language.
- Latency P95/P99 vs budgets.
- Reranker uplift >= 10% precision on evaluation set.
- A/B evaluation with statistical significance for reranker changes.
Target: CodeSearchNet MRR >= 0.75 for production readiness.

### Security and Load Testing
- Prompt injection corpus with detection targets and false positive thresholds.
- RBAC boundary tests and secret detection validation.
- Load tests: steady state, burst, and degraded-mode scenarios.

---

## 15. Deployment and Ops

- Docker compose updates for lexical backend and reranker (if service-based).
- Production deployment via Kubernetes/Helm or equivalent orchestration.
- Config validation at startup.
- Feature flags for embeddings, reranking, lexical search, and graph.
- Feature flag management via centralized config service with audit.
- Feature flag lifecycle: remove within 90 days of 100% rollout.
- Backups for Qdrant and Neo4j; nightly index snapshots.
- Lexical index snapshots included in backup plan.
- Cost modeling for GPU inference vs Gemini API usage at projected request volume.
- Cost model includes request volume scenarios and GPU sizing assumptions.
- Cost scenarios: 10k, 100k, 1M requests/day with break-even analysis.
- Rollout strategy: canary or blue/green with automated rollback on SLO breach.
- Rollback triggers: P99 > 2x baseline or error rate > 1%.
- Canary progression: 5% -> 25% -> 100% with time gates.
- RPO/RTO targets: Qdrant RPO=4h/RTO=2h; Neo4j RPO=1h/RTO=4h; Lexical RPO=4h/RTO=2h.
- Scaling: stateless API autoscaling, Qdrant sharding by org_id, Neo4j read replicas.
- Autoscaling triggers: CPU > 70% or P95 latency > 150ms.
- Config validation covers env vars, model registry, connectivity, feature flags.
- Disaster recovery: cross-region backups, quarterly failover tests, and documented runbooks.
- Runbook inventory covers: auth outage, Qdrant failure, Neo4j failure, cache outage, rollback.
- Reference versions: Qdrant >= 1.16.2, Neo4j 5.26 LTS.

---

## 16. Phased Roadmap

### Phase 0a: Decisions and Baselines (2 weeks)
- Finalize embedding model (Qwen3 + Nomic local; Gemini fallback).
- Select lexical backend using decision matrix.
- Establish baseline metrics (MRR, NDCG, latency) on internal eval set.

### Phase 0b: Security Baseline (4 weeks)
- Implement JWT validation, RBAC, SCIM, and code-owner policies.
- OAuth 2.1 implementation uses Authlib with PKCE S256.
- Define prompt-injection defenses and content exclusion rules.
- Align MCP transport and auth with spec 2025-11-25.

### Phase 0c: Observability Baseline (2 weeks)
- Add core tracing, structured logs, and audit hooks.
- Define SLO dashboards and error budget policies.

### Phase 1: Code Indexing Core (4-6 weeks)
- AST parsing (Python).
- Incremental indexer with Merkle tree and SCIP symbol IDs.
- Code embeddings and CODE_* graph projection.
- Basic data-flow edges (READS/WRITES).

### Phase 2: Retrieval + MCP Tools (3-4 weeks)
- Tri-hybrid retrieval and dynamic RRF weights.
- Reranker integration (local first).
- Core MCP tools including implementations and type definitions.
- MCP allowlisting and registry.
- Capture code review signals for later pattern detection.
- Implement MCP tasks/parallel tool calls and extensions where needed.

### Phase 3: Scoped Memory and Enterprise Controls (2-3 weeks)
- Backfill scope for existing data.
- Scoped retrieval + de-dup rules.
- GDPR delete workflow and audit event completeness.

### Phase 4: Observability and Performance (2-3 weeks, parallel from Phase 2)
- Tracing, P95/P99 tracking, cost monitoring.
- Cache tuning and circuit breaker enforcement.

### Phase 5: ADR Automation and Review Patterns (4-6 weeks)
- ADR heuristics + cADR integration.
- Pattern detection + feedback loop.
- Code smell detection and incident correlation (historical bug matching).

### Phase 6: Visualization (2-3 weeks)
- Graph export endpoints with pagination.
- Hierarchical code graph JSON schema.
- API-only; UI rendering remains out of scope.

### Phase Exit Gates (Quality Thresholds)
- Phase 0 exit: benchmarks complete; lexical decision signed; auth + RBAC + SCIM validated; core observability active.
- Phase 1 exit: parser error rate <= 2% on target repos; indexing success >= 99%; CodeSearchNet MRR >= 0.70 (target 0.75+).
- Phase 2 exit: NDCG@10 >= 0.55 on evaluation set; reranker improves precision by >= 10%; retrieval P95 <= 120ms.
- Phase 3 exit: scoped retrieval de-dup passes; GDPR delete completes end-to-end in staging.
- Phase 4 exit: P95/P99 SLOs stable for 7 days; error rate < 1%.
- Stability requires >=100k requests over the window.
- Phase 5 exit: ADRs retrievable by ID and query; pattern detection precision >= 0.6 on validation set.
- Phase 6 exit: graph exports pass schema validation; interactive clients load 1k node graph under 2s.

### Phase Dependencies
- Phase 1 depends on Phase 0a + 0b completion.
- Phase 2 depends on Phase 1 (indexing + embeddings available).
- Phase 3 depends on Phase 2 (tooling + routing stable).
- Phase 4 depends on Phase 2 (retrieval/rerank instrumentation).
- Phase 5 depends on Phase 2 and GitHub MCP integration.
- Phase 6 depends on Phase 2 (graph export APIs ready).

---

## 17. Phase Checklist with Owners

### Phase 0a (Owner: ML/IR + Platform)
- [ ] Benchmark Qwen3/Nomic/Gemini on internal eval set (Owner: ML/IR).
- [ ] Choose lexical backend using decision matrix (Owner: Platform).
- [ ] Capture baseline MRR/NDCG/latency metrics (Owner: Data/QA).

### Phase 0b (Owner: Security + Platform)
- [ ] OAuth 2.1 validation with PKCE, JWKs, and revocation (Owner: Security).
- [ ] RBAC + SCIM + CODEOWNERS enforcement (Owner: Platform).
- [ ] Prompt injection defenses and content exclusions (Owner: Security).

### Phase 0c (Owner: SRE)
- [ ] Core tracing, logs, and audit hooks (Owner: Platform/SRE).
- [ ] SLO dashboards and error budget policies (Owner: SRE).
- [ ] Baseline load tests for auth and retrieval (Owner: QA/SRE).

### Phase 1 (Owner: Indexing + Graph)
- [ ] Tree-sitter parsing (Python) with error recovery (Owner: Indexing).
- [ ] Merkle-based incremental indexing with SCIP IDs (Owner: Indexing).
- [ ] Code embeddings + CODE_* graph projection (Owner: Graph/ML).

### Phase 2 (Owner: Retrieval + MCP)
- [ ] Tri-hybrid retrieval and routing (Owner: Retrieval).
- [ ] Reranker integration and latency budgets (Owner: ML/IR).
- [ ] MCP tools + schemas + annotations (Owner: MCP/Platform).

### Phase 3 (Owner: Data Governance)
- [ ] Scoped memory CRUD and de-dup rules (Owner: Data/Backend).
- [ ] GDPR SAR export + delete workflows (Owner: Security/Compliance).

### Phase 4 (Owner: SRE)
- [ ] SLO dashboards and alerting (Owner: SRE).
- [ ] Load testing and degraded-mode validation (Owner: QA/SRE).

### Phase 5 (Owner: Knowledge Systems)
- [ ] ADR heuristics + cADR integration (Owner: Knowledge).
- [ ] Pattern detection + incident correlation (Owner: Knowledge/QA).

### Phase 6 (Owner: Visualization)
- [ ] Graph export endpoints and schemas (Owner: API).
- [ ] Interactive client performance validation (Owner: Frontend/UX).

---

## 18. Deliverables Checklist

- Unified identity, RBAC, SCIM, and code-owner policy.
- OAuth 2.1 with PKCE S256, DPoP, resource indicators, and claim schema.
- Policy-compliant embedding model registry.
- Extended code graph schema with core + extended edges.
- Incremental indexing with Merkle tree + SCIP symbol IDs.
- Tri-hybrid retrieval with reranking.
- Expanded MCP tool set and allowlisting.
- MCP tool schema registry and annotations.
- Prompt injection defenses and content exclusion policies.
- GDPR SAR export and delete workflows with backup purge.
- Geo partitioning and per-tenant KMS keys for restricted data.
- Observability and benchmark suite.

---

## 19. Definition of Done

The system can:
- Ingest a repo and build AST symbols, embeddings, and code graph.
- Answer semantic, lexical, and structural queries within latency budgets.
- Provide scoped memory retrieval with strict policy enforcement.
- Run impact analysis with confidence annotations.
- Operate securely with audit logging, GDPR delete, and graceful degradation.
- Provide SAR exports and enforce geo-scope residency.
