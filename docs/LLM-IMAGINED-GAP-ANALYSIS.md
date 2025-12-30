# LLM Imagined vs Planned Implementation Gap

Date: 2025-12-26
Scope: Compare the narrative "Das Intelligente Entwicklungsassistent-System: Eine umfassende Erklaerung" to the v9 plan and current code state.
Sources: docs/IMPLEMENTATION-PLAN-DEV-ASSISTANT v9.md, docs/IMPLEMENTATION-PROGRESS v9.md, docs/IMPLEMENTATION-CODE-OVERVIEW.md.

## Current Reality (Implemented Today)

- Memory CRUD with structured metadata, access_entity routing, and graph projection (OM_*).
- REST APIs for memories, search, graph, entities, feedback, experiments, backup, GDPR, and config.
- MCP servers for memory tools + graph tools; guidance tools; business concepts if enabled.
- Code indexing pipeline (Tree-sitter for Python/TS/TSX/Java, SCIP IDs, CODE_* graph, API boundary detection).
- Code search and analysis tools exposed via REST (`/api/v1/code`), including search, explain, callers/callees, impact, ADR automation, test generation, PR analysis, and indexing.
- MCP code tools exposed for index/search/explain/callers/callees/impact.
- Health probes and Prometheus metrics endpoint (`/metrics`).

## Gap Summary

The narrative still describes a full end-state system. The codebase now includes Phase 1/2 style modules and exposes core code tools via REST, but several integration and surface gaps remain.

## Features Described but Still Missing or Only Partially Implemented

### Code Indexing and Graph Operations
- Merkle-tree incremental indexing and bootstrap/priority queue modules exist but are not wired into CodeIndexingService or any API.
- No API for indexing status or bootstrap progress.

### Code Search and Tooling Surface
- MCP does not expose ADR automation, test generation, PR analysis, symbol definition, or graph export.
- REST exposes ADR/test/pr analysis, but there is no ADR storage or lifecycle endpoints; outputs are analysis-only.
- Code graph export and symbol hierarchy endpoints are not exposed (visualization modules are library-only).

### Retrieval and Feedback
- REST memory `/api/v1/search` does not compute embeddings on the fly; semantic search requires client-supplied vectors.
- Reranker adapters exist but are not integrated into tri-hybrid search.

### Cross-Repository and GitHub Integration
- Cross-repo modules exist but are not exposed via REST/MCP or scheduled in background jobs.
- No GitHub MCP integration; PR analysis expects a diff input only.

### Additional Narrative Features Not Implemented
- Co-change edges ("functions changed together") as a graph signal.
- Automated contradiction detection between memory scopes.
- Task/notes/bookmark workflows in the memory layer.
- Full semantic type resolution via LSP for TypeScript/Java.

## Bottom Line

The gap is smaller than earlier drafts: core code indexing and code tools are implemented and exposed via REST, with partial MCP coverage. Remaining work is mostly wiring (MCP surface, indexing progress, reranking, on-the-fly embeddings) and higher-level workflows (ADR lifecycle, GitHub integration).
