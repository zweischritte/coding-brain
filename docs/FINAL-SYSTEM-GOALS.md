# System Goals (Current, Codebase-Aligned)

Audience: New readers with no prior context
Purpose: Capture the goals of the Coding Brain system as reflected by the current codebase
Scope: Implemented capabilities plus near-term, code-adjacent gaps

---

## 1. User Value Goals
- Provide fast, high-signal memory recall with structured metadata and graph context.
- Offer consistent access via REST and MCP so IDEs and agents share the same context.
- Make setup and local execution predictable with a single docker-compose stack.

---

## 2. Memory and Context Goals
- Store structured memories (category, artifacts, entities, tags, evidence; scope is legacy metadata only).
- Support code_refs to link memories directly to source code locations.
- Enforce multi-tenant isolation via access_entity-driven visibility.
- Project memory metadata into a Neo4j graph for aggregation and relation queries (accessEntity-filtered).
- Preserve entity displayName for UI while normalizing entity keys for matching.
- Maintain similarity edges, tag co-occurrence, and typed entity relations.
- Provide configurable output verbosity (relation_detail) to optimize token usage.
- Support optional business concept extraction and semantic search in a separate graph.

---

## 3. Code Intelligence Goals
- Build a CODE_* graph from source code using Tree-sitter and SCIP symbols.
- Extract NestJS/Angular decorators for framework-aware indexing (60+ decorators).
- Detect API boundaries and link clients to endpoints.
- Discover event publishers/subscribers and generate TRIGGERS_EVENT edges.
- Provide tri-hybrid retrieval (lexical + vector + graph) for code search.
- Support deterministic and inferred edges with explicit include_inferred_edges control.
- Implement tool fallback cascade to prevent AI hallucination on missing symbols.
- Maintain library tools for explain-code, call graph, impact analysis, ADR automation,
  test generation, and PR analysis.

---

## 4. Retrieval Quality and Feedback Goals
- Capture feedback events and aggregate metrics for retrieval quality.
- Support A/B experiments to compare retrieval configurations.
- Keep retrieval decisions explainable by exposing source contributions and ranks.

---

## 5. Security and Compliance Goals
- Enforce JWT validation and scope-based RBAC on all REST and MCP endpoints.
- Support DPoP token binding with replay prevention.
- Bind MCP SSE sessions to authenticated principals to prevent hijacking.
- Provide GDPR SAR export and deletion endpoints with audit logging.

---

## 6. Operability and Resilience Goals
- Provide health probes for liveness, readiness, and dependency checks.
- Implement circuit breaker utilities for external dependencies.
- Provide backup/export tooling and documented runbooks.
- Offer shell scripts for full-stack backup and restore (database dumps + volumes).

---

## 7. Observability Goals
- Offer Prometheus metrics helpers and tracing hooks for key subsystems.
- Keep security events and session activity auditable.
- Provide dashboard templates for metrics consumption.

---

## 8. Developer Experience Goals
- Allow live configuration of LLMs, embedders, and vector stores via `/api/v1/config`.
- Provide a CLI skeleton for search, memory, indexing, and graph actions.
- Provide graph visualization export utilities (JSON, DOT, Mermaid).

---

## 9. Near-Term Gaps (Explicit)
- Expose remaining code-intel tools via MCP (ADR automation, test generation, PR analysis).
- Wire incremental indexing modules (Merkle tree + bootstrap queue) into the indexer.
- Expand REST hybrid memory search to include on-the-fly query embeddings.
- Replace CLI stubs with real API calls for search/memory/index/graph.
- Wire reranker module into REST/MCP endpoints (currently library-only).
