# Post-Implementation MCP Test Use Cases (Current Tools)

This document defines MCP-driven test cases to validate that the Coding Brain system works end-to-end with the MCP tools currently exposed in this repo.

---

## 1. Purpose

Validate MCP coverage for:
- Memory CRUD with scopes and access_entity routing
- Graph projections and traversal helpers
- Code indexing + code search/call graph/impact analysis
- Guidance MCP tools
- Business concepts MCP tools (optional)

---

## 2. Preconditions

Required system state:
- MCP server running (SSE).
- PostgreSQL, Neo4j, Qdrant, OpenSearch, and Valkey healthy.
- Test principal has required scopes: `memories:read`, `memories:write`, `memories:delete`, `graph:read`, `graph:write`, `search:read`, `code:read`, `code:write`, `mcp:access`.
- If testing business concepts: `BUSINESS_CONCEPTS_ENABLED=true`.

Fixture repository for code tools:
- Repo ID: `dev-assistant-fixture`
- Repo path is accessible to the API container (bind mount or host path in dev)
- Contains:
  - `src/math.py` with `add(a,b)`, `multiply(a,b)`, `Calculator.compute()` calling `add`
  - `src/auth/user_service.py` with `authenticate_user()` calling `validate_token()` and `load_user()`
  - `src/interfaces/auth_provider.ts` with `IAuthProvider`
  - `src/auth/jwt_provider.ts` with `JwtAuthProvider implements IAuthProvider`
  - `tests/test_math.py` with tests for `add`

---

## 3. Test Execution Rules

- Use MCP tools only (no shell access for the test runner).
- Record tool inputs and outputs for each test case.
- A test passes when all expected outcomes are met.
- If a backend is unavailable, degraded_mode results are acceptable when documented.

---

## 4. Test Cases

### TC-001 Memory CRUD (User Scope)
Tools: `add_memories`, `search_memory`, `list_memories`, `update_memory`, `delete_memories`
Steps:
1) Add memory: "Team uses async/await for Python services" with `category=convention`, `scope=user`, `entity=Backend`.
2) Search memories with query "async/await".
3) Update memory text to "Team uses async/await in Python, pytest for tests".
4) List memories and confirm update is visible.
5) Delete the memory by ID.
Expected:
- Add returns an ID.
- Search returns the memory.
- Update reflected in list/search.
- Delete removes it from results.

### TC-002 Memory Routing (Shared Scope)
Tools: `add_memories`, `search_memory`
Steps:
1) Add memory with `scope=project`, `access_entity=project:default_org/coding-brain`, `entity=RoutingTest`.
2) Search without scope filter.
Expected:
- Memory is returned with correct `access_entity` metadata.

### TC-010 Graph Aggregation
Tools: `add_memories`, `graph_aggregate`
Steps:
1) Add memory with `tags={"topic":"auth"}` and `entity=AuthService`.
2) Call `graph_aggregate(group_by="tag")`.
Expected:
- Tag aggregation includes `topic` with a count >= 1.

### TC-011 Graph Related Memories
Tools: `add_memories`, `graph_related_memories`
Steps:
1) Add two memories with the same `entity=AuthService`.
2) Call `graph_related_memories(memory_id=<id_of_first>)`.
Expected:
- Second memory appears in related results.

### TC-012 Entity Network + Relations
Tools: `graph_entity_network`, `graph_entity_relations`
Steps:
1) Ensure at least one memory with `entity=AuthService` exists.
2) Call `graph_entity_network(entity_name="AuthService")`.
3) Call `graph_entity_relations(entity_name="AuthService")`.
Expected:
- Network returns connections (may be empty for small data but returns shape).
- Relations returns a list (may be empty if no typed relations exist).

### TC-013 Similar Memories + Subgraph
Tools: `graph_similar_memories`, `graph_subgraph`
Steps:
1) Use a memory ID that has been embedded.
2) Call `graph_similar_memories(memory_id=<id>)`.
3) Call `graph_subgraph(memory_id=<id>, depth=2)`.
Expected:
- Similar memories returns results or empty list with `similarity_enabled` metadata.
- Subgraph returns nodes/edges or empty graph if Neo4j unavailable.

### TC-020 Code Indexing
Tool: `index_codebase`
Steps:
1) Call `index_codebase(repo_id="dev-assistant-fixture", root_path="/path/to/fixture", reset=true)`.
Expected:
- Non-zero `files_indexed` and `symbols_indexed` if the path is accessible.
- `meta.degraded_mode` true only if backends are missing.

### TC-021 Code Search (Hybrid)
Tool: `search_code_hybrid`
Steps:
1) Query "add two numbers" with `repo_id="dev-assistant-fixture"`.
Expected:
- Results include `add` in `src/math.py`.

### TC-022 Explain Code
Tool: `explain_code`
Steps:
1) Use a symbol_id from TC-021 results.
2) Call `explain_code(symbol_id=...)`.
Expected:
- Explanation includes symbol metadata and related context.

### TC-023 Call Graph (Callers/Callees)
Tools: `find_callers`, `find_callees`
Steps:
1) Call `find_callees(repo_id="dev-assistant-fixture", symbol_name="Calculator.compute")`.
2) Call `find_callers(repo_id="dev-assistant-fixture", symbol_name="add")`.
Expected:
- Callees include `add`.
- Callers include `Calculator.compute` and/or `test_add`.

### TC-024 Impact Analysis
Tool: `impact_analysis`
Steps:
1) Call `impact_analysis(repo_id="dev-assistant-fixture", changed_files=["src/math.py"])`.
Expected:
- `tests/test_math.py` appears in affected files.

### TC-030 Guidance MCP
Tools: `list_guides`, `get_guidance`, `search_guidance`
Steps:
1) Call `list_guides`.
2) Call `get_guidance("memory")`.
3) Call `search_guidance("access_entity")`.
Expected:
- Guidance list returned.
- Guidance content returned for the requested guide.
- Search returns matches or a clear "no matches" response.

### TC-040 Business Concepts (Optional)
Tools: `extract_business_concepts`, `list_business_concepts`, `search_business_concepts`
Steps:
1) Call `extract_business_concepts(content="We launched a new billing product", store=true)`.
2) Call `list_business_concepts(limit=5)`.
3) Call `search_business_concepts(query="billing", limit=5)`.
Expected:
- Extraction returns concepts/entities with stored counts.
- List/search return results (or empty lists if the graph is empty).

---

## 5. Pass/Fail Criteria

The system passes when:
- Mandatory test cases (TC-001 to TC-024, TC-030) pass.
- Optional business concept tests pass if the feature is enabled.
- Any degraded_mode responses are documented with missing_sources.

---

## 6. Reporting Template

For each test case, record:
- Test ID
- Inputs (tool + parameters)
- Output summary
- Pass/Fail
- Notes or anomalies

---

## 7. Notes

- If a tool is not available, record as a blocking issue with the error payload.
- For code tools, ensure the fixture repo is visible to the API container.
- For graph tests, Neo4j must be configured and healthy.
