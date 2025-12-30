# PRD: REST Memory Search Embeddings + MCP Code Tool Parity

Status: Draft
Owner: TBD
Target: Next minor release

## 0. Executive Summary

This PRD covers two related gaps in the current codebase:

1) REST memory search `/api/v1/search` is lexical-only, while MCP `search_memory` already embeds queries and performs semantic + graph-aware reranking. This produces inconsistent results across clients.
2) Code-intelligence tooling is exposed via REST (`/api/v1/code`) but MCP only exposes a subset (index/search/explain/callers/callees/impact). ADR automation, test generation, and PR analysis should be available in MCP for LLM-based clients.

The goal is to bring REST and MCP to parity: add on-the-fly embeddings to `/api/v1/search` (with graceful fallback), and wire ADR/test/PR code tools into MCP with consistent scope checks and degraded-mode metadata.

---

## 1. Problem Statement

- REST search `/api/v1/search` only runs lexical OpenSearch queries. Semantic search exists at `/api/v1/search/semantic`, but clients must supply `query_vector`.
- MCP `search_memory` embeds the query on the server and can combine vector + graph signals with reranking.
- As a result, REST clients and MCP clients see different relevance and recall for the same query.
- Code tools are exposed via REST but not via MCP for ADR automation, test generation, and PR analysis. This blocks LLM clients that only consume MCP tools.

---

## 2. Goals

### 2.1 REST Search Embeddings
- `/api/v1/search` embeds the query when possible and performs hybrid search (lexical + vector) with existing access control.
- If embeddings fail or are unavailable, the endpoint falls back to lexical-only search without failing the request.
- Preserve current request/response format; add optional metadata only if needed.

### 2.2 MCP Code Tool Parity
- Add MCP tools for ADR automation, test generation, and PR analysis.
- Use the same code-tool implementations as REST (`CodeToolkit` + `tools/*`).
- Enforce the same scope requirements as REST (search:read + graph:read).
- Return consistent `meta` with degraded mode + missing sources.

---

## 3. Non-Goals

- No new ADR storage or ADR CRUD endpoints (ADR remains analysis-only).
- No new code indexing algorithms (Merkle/priority queue wiring is a separate item).
- No UI changes.
- No change to memory MCP behavior.
- No change to OpenSearch index structure beyond ensuring it is compatible with the embedder dimension.

---

## 4. Current State (Repo Context)

### Key Files
- REST memory search: `openmemory/api/app/routers/search.py`
- OpenSearch store: `openmemory/api/app/stores/opensearch_store.py`
- MCP memory search: `openmemory/api/app/mcp_server.py` (search_memory)
- Code tools REST router: `openmemory/api/app/routers/code.py`
- MCP code tools: `openmemory/api/app/mcp_server.py` (index/search/explain/callers/callees/impact)
- CodeToolkit factory: `openmemory/api/app/code_toolkit.py`
- Code tool modules: `openmemory/api/tools/adr_automation.py`, `openmemory/api/tools/test_generation.py`, `openmemory/api/tools/pr_workflow/pr_analysis.py`

### REST Search (Lexical Only Today)
`openmemory/api/app/routers/search.py`:
```python
@router.post("", response_model=SearchResponse)
async def hybrid_search(...):
    # ...
    # Perform lexical search (hybrid requires embedding which we don't have here)
    filters_dict = _filters_to_dict(request.filters)
    exact, prefixes = _access_entity_filters(principal)
    hits = store.search_with_access_control(
        query_text=request.query,
        access_entities=exact,
        access_entity_prefixes=prefixes,
        limit=request.limit,
        filters=filters_dict,
    )
```

### OpenSearch Store Supports Hybrid Search
`openmemory/api/app/stores/opensearch_store.py`:
```python
def hybrid_search_with_access_control(
    self,
    query_text: str,
    query_vector: List[float],
    access_entities: List[str],
    access_entity_prefixes: Optional[List[str]] = None,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    # Builds a bool should query with lexical + knn script_score
```

### MCP Memory Search Already Embeds
`openmemory/api/app/mcp_server.py`:
```python
embeddings = memory_client.embedding_model.embed(query, "search")
hits = memory_client.vector_store.search(
    query=query,
    vectors=embeddings,
    limit=search_limit,
    filters=search_filters,
)
```

### Code Tools in REST (ADR/Test/PR Already Exposed)
`openmemory/api/app/routers/code.py`:
```python
@router.post("/adr", response_model=ADRResponse)
async def adr_automation(...):
    # uses toolkit.adr_tool

@router.post("/test-generation", response_model=TestGenerationResponse)
async def test_generation(...):
    # uses toolkit.test_gen_tool

@router.post("/pr-analysis", response_model=PRAnalysisResponse)
async def pr_analysis(...):
    # uses tools.pr_workflow.pr_analysis
```

### MCP Code Tools (Subset)
`openmemory/api/app/mcp_server.py` currently exposes:
- `index_codebase`
- `search_code_hybrid`
- `explain_code`
- `find_callers`
- `find_callees`
- `impact_analysis`

---

## 5. Proposed Solution

### 5.1 REST /api/v1/search: On-the-Fly Embeddings

#### Behavior
- Default behavior becomes "auto" embedding:
  - Try to compute a query embedding using the configured Mem0 embedder.
  - If embedding succeeds, use `TenantOpenSearchStore.hybrid_search_with_access_control`.
  - If embedding fails or embedder is unavailable, fall back to lexical-only search and return results as today.
- Add optional request field to control mode:
  - `mode`: one of `auto` (default), `lexical`, `semantic`.
  - `semantic` uses embedding if available; if not, returns 503 or empty results (decision below).

#### Request/Response (Backward Compatible)
- Request body remains:
  ```json
  { "query": "...", "limit": 10, "filters": { ... }, "mode": "auto" }
  ```
- Response remains:
  ```json
  { "results": [...], "total_count": 0, "took_ms": 12 }
  ```
- Optionally add:
  ```json
  "meta": { "degraded_mode": true, "missing_sources": ["embedding"] }
  ```
  This field is additive and backwards compatible.

#### Embedding Source
- Reuse Mem0 embedder via `get_memory_client()` in `openmemory/api/app/utils/memory.py`.
- Use the same method MCP uses: `memory_client.embedding_model.embed(query, "search")`.
- If `memory_client` or `embedding_model` is unavailable, fallback to lexical.

#### Access Control
- Keep current access_entity filtering via `build_access_entity_patterns` and `search_with_access_control` / `hybrid_search_with_access_control`.

#### OpenSearch Index Dimensions
- `TenantOpenSearchConfig.embedding_dim` must match the embedder dimension.
- If mismatch is detected, log an error and fall back to lexical-only search.

#### Errors and Degraded Mode
- Embedding failure should not fail the request in `mode=auto`.
- If `mode=semantic` and embeddings are unavailable, return 503 with a clear error.

#### Observability
- Add metrics or logs for:
  - embedding_success
  - embedding_failure
  - semantic_fallback
  - search_mode_used

---

### 5.2 MCP: ADR Automation, Test Generation, PR Analysis

#### New MCP Tools
Add the following MCP tools in `openmemory/api/app/mcp_server.py`:

1) `adr_automation`
   - Required: `diff` or `commit_messages` or `repo_id` (align with REST input)
   - Scope: `search:read` + `graph:read`

2) `test_generation`
   - Required: `repo_id` plus `symbol_id` or `file_path`
   - Scope: `search:read` + `graph:read`

3) `pr_analysis`
   - Required: `repo_id` + `diff`
   - Optional: `title`, `body`, `pr_number`, `base_branch`, `head_branch`
   - Scope: `search:read` + `graph:read`

#### Implementation Pattern
Follow the existing code-tool wrappers in MCP:

- Scope checks via `_check_tool_scope`.
- Use `CodeToolkit` for dependencies.
- Use `_create_code_meta` for `meta` in response.
- Return JSON strings with consistent keys: `meta`, plus tool-specific fields.

#### Example Template (from existing MCP tool)
```python
@mcp.tool(description="...")
async def search_code_hybrid(...):
    scope_error = _check_tool_scope("search:read")
    if scope_error:
        return scope_error
    toolkit = get_code_toolkit()
    if not toolkit.search_tool:
        return json.dumps({"results": [], "meta": _create_code_meta(...)})
    # call tool and return dataclasses.asdict(result)
```

#### Tool Mappings to Existing Implementations
- ADR automation: `toolkit.adr_tool.analyze(...)` (see REST endpoint for inputs)
- Test generation: `toolkit.test_gen_tool.generate(...)`
- PR analysis: `tools.pr_workflow.pr_analysis.create_pr_analysis_tool(...)`

#### Response Schema
Mirror REST responses where feasible:
- ADR: `{ "detections": [...], "meta": {...} }`
- Test generation: `{ "test_cases": [...], "imports": [...], "fixtures": {...}, "meta": {...} }`
- PR analysis: `{ "summary": "...", "risks": [...], "complexity_score": ..., "meta": {...} }`

#### Degraded Mode
If Neo4j or OpenSearch is unavailable, populate `meta.degraded_mode=true` and `missing_sources`.

---

## 6. Requirements

### 6.1 Functional Requirements

#### REST Search Embeddings
- FR-1: `/api/v1/search` computes embeddings in `mode=auto` when embedder is available.
- FR-2: `/api/v1/search` uses `hybrid_search_with_access_control` when embeddings are available.
- FR-3: If embeddings fail in `mode=auto`, fall back to lexical-only search and return results.
- FR-4: Optional request `mode` allows forcing lexical-only behavior.
- FR-5: Access_entity filtering remains enforced.

#### MCP Code Tools
- FR-6: Add MCP tool for ADR automation with search+graph scope checks.
- FR-7: Add MCP tool for test generation with search+graph scope checks.
- FR-8: Add MCP tool for PR analysis with search+graph scope checks.
- FR-9: Use `CodeToolkit` and existing tool modules (no new logic).
- FR-10: Include `meta` in MCP responses for degraded mode.

### 6.2 Non-Functional Requirements
- NFR-1: No breaking change to existing REST or MCP clients.
- NFR-2: If embeddings are unavailable, requests should still return lexical results.
- NFR-3: Embedding failure must not crash the API.
- NFR-4: Logs should indicate the search mode used.

---

## 7. API and Schema Changes

### 7.1 REST /api/v1/search Request
Add an optional `mode` field:
- `auto` (default)
- `lexical`
- `semantic`

### 7.2 REST /api/v1/search Response
Optional additive `meta` field:
```json
"meta": {
  "degraded_mode": true,
  "missing_sources": ["embedding"]
}
```

### 7.3 MCP Tools
Add new tool descriptors and handlers in MCP.

---

## 8. Security and Access Control

- Maintain existing scope checks for REST (`require_scopes`) and MCP (`_check_tool_scope`).
- No changes to access_entity logic.
- No new scopes introduced.

---

## 9. Testing Plan

### Unit Tests
- REST search: ensure mode switching (auto, lexical, semantic) and fallback on embedding failure.
- MCP tool wrappers: verify tool outputs include meta and obey scope checks.

### Integration Tests
- End-to-end REST search with OpenSearch + embeddings configured.
- End-to-end MCP ADR/test/pr calls using a fixture repo and indexed graph.

### Regression Tests
- Verify existing clients still work with no mode provided.
- Verify lexical-only mode matches previous behavior.

---

## 10. Rollout Plan

1) Implement REST search embedding with fallback.
2) Add MCP code tool wrappers.
3) Update docs + MCP schema tests if needed.
4) Deploy behind a config flag if required (optional).

---

## 11. Risks and Mitigations

- Risk: Embedding dimension mismatch with OpenSearch index.
  - Mitigation: detect mismatch and fallback to lexical with a warning.

- Risk: Increased latency for REST search.
  - Mitigation: allow `mode=lexical`; add timeouts around embedding.

- Risk: MCP tools fail if backends are down.
  - Mitigation: use degraded_mode responses and explicit missing_sources.

---

## 12. Open Questions

- Should `/api/v1/search` return 503 in `mode=semantic` when embeddings are unavailable, or return lexical results with a warning?
- Should we add a query parameter instead of a JSON field for `mode`?
- Do we want a config flag to globally enable/disable on-the-fly embeddings for REST search?

---

## 13. Appendix: Relevant Code Fragments

### MCP search_memory embedding behavior
`openmemory/api/app/mcp_server.py`:
```python
embeddings = memory_client.embedding_model.embed(query, "search")
hits = memory_client.vector_store.search(
    query=query,
    vectors=embeddings,
    limit=search_limit,
    filters=search_filters,
)
```

### REST search lexical-only behavior
`openmemory/api/app/routers/search.py`:
```python
hits = store.search_with_access_control(
    query_text=request.query,
    access_entities=exact,
    access_entity_prefixes=prefixes,
    limit=request.limit,
    filters=filters_dict,
)
```

### OpenSearch hybrid search helper
`openmemory/api/app/stores/opensearch_store.py`:
```python
def hybrid_search_with_access_control(...):
    body = {
        "size": limit,
        "query": {
            "bool": {
                "should": [
                    {"match": {"content": {"query": query_text, "boost": self.config.lexical_weight}}},
                    {"script_score": {"query": {"match_all": {}}, "script": {"source": "knn_score", "lang": "knn", "params": {"field": "embedding", "query_value": query_vector}}}},
                ],
                "filter": filter_clauses,
                "minimum_should_match": 1,
            }
        }
    }
```

### REST code tool endpoints (ADR/Test/PR)
`openmemory/api/app/routers/code.py`:
```python
@router.post("/adr", response_model=ADRResponse)
async def adr_automation(...):
    # uses toolkit.adr_tool

@router.post("/test-generation", response_model=TestGenerationResponse)
async def test_generation(...):
    # uses toolkit.test_gen_tool

@router.post("/pr-analysis", response_model=PRAnalysisResponse)
async def pr_analysis(...):
    # uses tools.pr_workflow.pr_analysis
```

---

## 14. Success Criteria

- REST `/api/v1/search` returns semantic/hybrid results by default when embeddings are available.
- REST `/api/v1/search` gracefully falls back to lexical on embedding failure.
- MCP exposes ADR automation, test generation, and PR analysis tools with consistent scope checks and meta.
- Docs reflect parity between REST and MCP.
