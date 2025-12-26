# LLM Imagined vs Planned Implementation Gap

Date: 2025-12-26
Scope: Compare the narrative "Das Intelligente Entwicklungsassistent-System: Eine umfassende Erklaerung" to the v9 plan and current code state.
Sources: docs/IMPLEMENTATION-PLAN-DEV-ASSISTANT v9.md, docs/IMPLEMENTATION-PROGRESS v9.md, docs/IMPLEMENTATION-CODE-OVERVIEW.md.

## Current Reality (Implemented Today)

- Phase 0a/0b/0c only: benchmarks, security baseline (JWT/DPoP/RBAC/SCIM stubs), observability (tracing/logging/audit/SLO).
- Legacy memory and business concept MCP tools exist (AXIS memory + OM_* graph).
- Hybrid retrieval for memory (RRF, query routing, graph boosts) exists.
- No code indexing pipeline, no CODE_* code graph, and no code intelligence MCP tools yet.

## Gap Summary

The narrative describes a full end-state system. The majority of those capabilities are planned in v9 but not implemented yet (Phase 0d onward). Below are the features described in the narrative that the system cannot do yet.

## Features Described but NOT Implemented Yet (Planned in v9)

### Code Indexing and Code Graph (Phase 1)
- AST parsing for Python, TypeScript/TSX, and Java (Tree-sitter) with symbol extraction.
- SCIP symbol IDs for stable identifiers.
- CODE_* graph schema, callers/callees, inheritance, imports, dependency graph, and data-flow edges.
- Cross-language API boundary linking (REST/GraphQL/gRPC/messaging).
- Merkle-based incremental indexing with transactional updates.
- Bootstrap status API and progressive indexing with priority queue tiers.

### Code Search and Retrieval (Phase 2)
- Code semantic, lexical, and tri-hybrid search tools (search_code_*).
- Production OpenSearch lexical backend (current adapter is benchmark mock).
- Reranker adapter and integration into retrieval pipeline.
- Explain-code tool with context-aware output.
- Similar-code search, symbol lookup, and code graph exports.

### Feedback and Quality Loop (Phase 2.5)
- Implicit and explicit feedback collection.
- A/B experimentation and RRF weight optimizer.
- Retrieval instrumentation with per-stage latency metrics.

### Memory Scope and Episodic Memory (Phase 3)
- Scoped memory hierarchy (session/user/team/project/org/enterprise) with precedence and de-dup.
- Episodic memory storage and summarization across sessions.
- SCIM orphan data handling and lifecycle management.

### Performance and Flexibility (Phase 4)
- Speculative retrieval and prefetch cache targeting >= 60% hit rate.
- Content-addressed embedding storage and shadow embedding pipeline.
- Embedding protection (encryption at rest, re-embedding on deletion).
- Graph scaling strategies (partitioning, replicas, materialized views).

### ADR and Test Generation (Phase 5)
- ADR create/update/search tools and code linkage.
- Test generation aligned to project conventions and graph insights.

### Visualization and PR Workflow (Phase 6-7)
- Graph export endpoints with pagination and hierarchical schema.
- PR analysis and review comment suggestions.
- GitHub MCP integration.

### Cross-Repository Intelligence (Phase 8)
- Repository dependency graph and cross-repo impact analysis.
- Breaking change detection across repos.

### Security and DX Expansion (Phase 0d)
- Secret detection tiers (fast scan, deep scan, verification) with quarantine.
- CLI tooling, query playground, and dashboard templates.
- AI governance readiness controls in audit events.

## Features Described but NOT Yet Planned or Only Implicit

These appear in the narrative but are not explicitly scoped in the v9 plan:

- Co-change relationship edges ("functions often changed together") as a first-class graph signal.
- Automated contradiction detection between memory scopes (beyond precedence rules).
- Task/notes/bookmark workflows inside the memory layer.
- Full semantic type resolution for TypeScript and Java (narrative implies deep understanding; plan is best-effort AST + optional LSP).

## Bottom Line

The narrative is aligned with the long-term vision, but the current system only delivers Phase 0 capabilities. Most described features map to Phase 0d through Phase 8 and are not implemented yet. The immediate gap is code indexing + code graph + code tools, which are the foundation for most of the narrative's code intelligence claims.
