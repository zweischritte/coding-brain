# Radical Drop/Expand Plan: Make Coding Brain Non-Commodity

## North Star

Build a memory + index + graph system that produces **deterministic, evidence-backed answers** that a generic LLM (or Gemini CLI base tools) cannot replicate with grep + read_file + shell.

## Evidence baseline (what exists today)

### Commodity capabilities already provided by Gemini CLI

> "GrepTool (grep.ts): Searches for patterns in files." (Gemini CLI core tools API docs)

> "Use run_shell_command to interact with the underlying system, run scripts, or perform command-line operations." (Gemini CLI shell tool docs)

These are already standard in the Gemini CLI toolset. Anything that duplicates grep or shell should not be a Coding Brain feature.

### Coding Brain call graph and indexing are regex-based today

> `pattern = re.compile(rf"\b{re.escape(target_name)}\s*\(")` (openmemory/api/indexing/code_indexer.py)

> "FindCallersTool ... Traverses incoming CALLS edges in the code graph." (openmemory/api/tools/call_graph.py)

The call graph currently depends on CALLS edges that are inferred via name-matching regex within symbol bodies, so callers/callees results are not LSP-grade.

### Tri-hybrid retrieval exists and is unique

> "Tri-hybrid retrieval combining: Lexical search (BM25) ... Semantic/Vector search (kNN) ... Graph context (CODE_* relationships) from Neo4j." (openmemory/api/retrieval/trihybrid.py)

### Cross-language API boundary detection exists

> "REST endpoint detection in Python (FastAPI, Flask)" and "API client detection in TypeScript (fetch, axios)." (openmemory/api/indexing/api_boundaries.py)

### Memory already supports code references

> "code_refs: Link to source code (e.g., [{"file_path": "/src/auth.ts", "line_start": 42}])" (openmemory/api/app/mcp_server.py)

### Graph schema already declares richer edge types

> "Edge types: CONTAINS, DEFINES, IMPORTS, CALLS, READS, WRITES, DATA_FLOWS_TO" (openmemory/api/indexing/graph_projection.py)

## Drop (hard, immediate)

1) **Drop any feature that is "grep + file reads" in disguise.**
   - If it can be done by Gemini CLI GrepTool or run_shell_command, it is not a product feature.
   - Examples to remove or hide:
     - The grep fallback stage in `FallbackFindCallersTool`.
     - Any lexical-only search endpoints.

   Evidence:
   - "GrepTool (grep.ts): Searches for patterns in files." (Gemini CLI core tools API docs)
   - "Use run_shell_command to interact with the underlying system..." (Gemini CLI shell tool docs)
   - "Fallback Cascade:\n1. Graph-based search (SCIP) - Primary\n2. Grep-based pattern matching - First fallback\n3. Semantic search (embeddings) - Second fallback\n4. Structured error with suggestions - Final fallback" (openmemory/api/tools/fallback_find_callers.py)

2) **Drop callers/callees as a flagship feature until the call graph is real.**
   - Current CALLS edges are inferred with regex name matching, which is not unique and not reliably better than IDE search.

   Evidence:
   - `pattern = re.compile(rf"\b{re.escape(target_name)}\s*\(")` (openmemory/api/indexing/code_indexer.py)
   - "Traverses incoming CALLS edges" (openmemory/api/tools/call_graph.py)

3) **Drop the "graph" story if we cannot populate non-CALLS edges.**
   - The schema advertises READS/WRITES/DATA_FLOWS, but the main code indexer path shown below only emits CONTAINS and CALLS edges. Until READS/WRITES are actually emitted, the graph story is inflated.

   Evidence:
   - "Edge types: CONTAINS, DEFINES, IMPORTS, CALLS, READS, WRITES, DATA_FLOWS_TO" (openmemory/api/indexing/graph_projection.py)
   - "self._projection.create_edge(\n    edge_type=CodeEdgeType.CONTAINS,\n    source_id=str(file_path),\n    target_id=symbol_id,\n    properties={\"repo_id\": self.repo_id},\n)" (openmemory/api/indexing/code_indexer.py)
   - "self._projection.create_edge(\n    edge_type=CodeEdgeType.CALLS,\n    source_id=symbol_id,\n    target_id=target_id,\n    properties={\"inferred\": True},\n)" (openmemory/api/indexing/code_indexer.py)

## Drop unless replaced (explicit rebuild threshold)

These can stay only if they are rebuilt to be **semantically stronger than IDE search**.

1) **Call graph (find_callers/find_callees)**
   - Keep only if we replace regex-based inference with LSP/SCIP/LSIF-grade resolution that handles imports, namespaces, overloads, and dynamic dispatch.

2) **Impact analysis**
   - Keep only if it is driven by real call graph and data flow edges (READS/WRITES/DATA_FLOWS_TO), not just CALLS.

   Evidence:
   - "Traverses callers to find affected files." (openmemory/api/tools/impact_analysis.py)

## Expand (hard bets)

1) **Graph edges beyond CALLS.**
   - Implement READS/WRITES/DATA_FLOWS_TO and IMPORTS edges for real dependency and data flow analysis.
   - This is the fastest path to uniqueness because generic LLMs do not have deterministic data-flow graphs.

   Evidence:
   - Graph schema already defines the edge types (openmemory/api/indexing/graph_projection.py).

2) **Cross-language API boundary graph (make it first-class).**
   - Elevate API boundaries into distinct edge types rather than overloading CALLS/DEFINES. Make it queryable as a first-class graph surface.

   Evidence:
   - API boundary detection exists today (openmemory/api/indexing/api_boundaries.py).
   - "edge_type=CodeEdgeType.DEFINES,  # Use DEFINES as closest semantic match\n...\n\"boundary_type\": \"EXPOSES\"" (openmemory/api/indexing/api_boundaries.py)
   - "edge_type=CodeEdgeType.CALLS,  # Use CALLS as closest semantic match\n...\n\"boundary_type\": \"CONSUMES\"" (openmemory/api/indexing/api_boundaries.py)

3) **Tri-hybrid retrieval as the core search product.**
   - Lean into the lexical + vector + graph fusion with transparent provenance and scoring.
   - This is already unique and should be the canonical search surface.

   Evidence:
   - Tri-hybrid design exists (openmemory/api/retrieval/trihybrid.py).
   - "Integration points:\n- openmemory.api.retrieval.trihybrid: Tri-hybrid retrieval" (openmemory/api/tools/search_code_hybrid.py)

4) **Memory with code refs as a mandatory contract for code claims.**
   - Any code-related memory must include `code_refs`. Use this as a product differentiator: "Every claim has a code anchor."

   Evidence:
   - `add_memories` already supports `code_refs` and validates/serializes them (openmemory/api/app/mcp_server.py).

5) **Cross-repo impact analysis and dependency graphing.**
   - Expand the cross-repo impact module into a visible feature, not just an internal utility.

   Evidence:
   - "Cross-repository impact analysis.\n\nThis module provides impact analysis across repositories for detecting\nbreaking changes and understanding the blast radius of code changes." (openmemory/api/cross_repo/impact_analysis.py)

6) **Language coverage expansion with parser-backed semantics.**
   - Current AST parser supports Python/TypeScript/TSX/Java only. Expand to Go/Rust/C# or enforce explicit limits in the UI/API.

   Evidence:
   - "Supported programming languages.\n\nPYTHON = \"python\"\nTYPESCRIPT = \"typescript\"\nTSX = \"tsx\"\nJAVA = \"java\"" (openmemory/api/indexing/ast_parser.py)

## What we keep as "unique" only if it is grounded

1) **ADR automation**
   - Keep, but only when backed by graph evidence and code refs, not just diff heuristics.

   Evidence:
   - "Heuristics:\n- DependencyHeuristic: Detects significant new dependencies\n- APIChangeHeuristic: Detects new/breaking API changes\n- ConfigurationHeuristic: Detects feature flags, infrastructure changes\n- SchemaHeuristic: Detects database schema changes\n- SecurityHeuristic: Detects auth, encryption, permission changes\n- PatternHeuristic: Detects architectural pattern introductions\n- CrossCuttingHeuristic: Detects logging, monitoring, caching additions\n- PerformanceHeuristic: Detects performance optimizations" (openmemory/api/tools/adr_automation.py)

2) **PR analysis**
   - Keep if it includes graph-derived impact and links to code evidence; otherwise it is a generic LLM feature.

   Evidence:
   - "Integration points:\n- openmemory.api.indexing.graph_projection: CODE_* graph queries\n- openmemory.api.tools.impact_analysis: Impact analysis\n- openmemory.api.tools.adr_automation: ADR detection" (openmemory/api/tools/pr_workflow/pr_analysis.py)

3) **Test generation**
   - Keep only if it is pattern-anchored to repo history and coverage, and emits evidence links.

   Evidence:
   - "Features:\n- Generate tests for functions and classes\n- Apply team test patterns from existing tests\n- Generate tests for uncovered code paths\n- Support pytest and unittest frameworks\n- Generate fixtures and mocks\n- Include edge cases and error handling tests" (openmemory/api/tools/test_generation.py)

## Product positioning after the drop/expand

- **Not a CLI wrapper**: no grep or shell gimmicks.
- **A deterministic knowledge engine**: graph edges, data flow, API boundaries, and cross-repo impact with explicit code_refs.
- **Memory-first**: stored decisions and explanations are traceable to code and graph provenance.
- **LLM as a renderer**: the model summarizes deterministic evidence rather than inventing it.

## Decision gate (non-negotiable)

If a feature can be replicated by "grep + read_file + model" with similar quality, it is dropped. If a feature relies on regex call inference, it is either rebuilt or removed from the public tool surface.
