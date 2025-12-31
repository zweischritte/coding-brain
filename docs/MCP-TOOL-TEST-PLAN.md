# MCP Tool Test Plan (coding-brain)

Purpose
- Provide a complete, step-by-step MCP tool test plan for another LLM instance.
- Cover all MCP tools currently implemented in this repository.
- Use realistic memory content with correct scope and access_entity.

Prerequisites
- MCP server running and reachable by the client.
- Scopes available to the test client:
  - memories:write, memories:read
  - search:read
  - graph:read, graph:write
- Optional (for access control checks):
  - admin:read, admin:write
- Neo4j database configured:
  - `NEO4J_DATABASE=neo4j` if your mem0 graph store uses `env:NEO4J_DATABASE` in config.
- Code intelligence backends running (for Sections 4–5):
  - OpenSearch (search), Neo4j (graph), and embedding provider (Ollama/OpenAI) if vector search is enabled.
- Repo path and repo_id known:
  - repo_id: coding-brain
  - root_path (docker-compose): /usr/src/coding-brain
  - root_path (local dev): /Users/grischadallmer/git/coding-brain
- Default access_entity for this repo:
  - access_entity="project:default_org/coding-brain"
  - scope="project"
  - artifact_type="repo"
  - artifact_ref="coding-brain"
- For access_entity auto-resolution tests, have a user with:
  - Exactly one grant for a shared scope (e.g., team:default_org/dev)
  - Multiple grants for the same scope (e.g., team:default_org/dev and team:default_org/ops)

---

## 1) Seed Dataset (Memories)
Create a small, coherent memory set for graph and search tests. Capture IDs from the response or via list_memories.

Add the following memories one by one:

```
add_memories(
  text="SearchService uses OpenSearch for lexical search and Qdrant for vectors.",
  category="architecture",
  scope="project",
  entity="SearchService",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"search": true, "vector": true, "opensearch": true}
)
```

```
add_memories(
  text="Cache search embeddings in memory_client for 15 minutes to reduce latency.",
  category="performance",
  scope="project",
  entity="SearchService",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"search": true, "performance": true, "cache": true}
)
```

```
add_memories(
  text="Use ADR automation whenever a new dependency is added in requirements.txt.",
  category="workflow",
  scope="project",
  entity="ADRTool",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"adr": true, "dependency": true, "vector": true}
)
```

```
add_memories(
  text="Project memories must use access_entity project:default_org/coding-brain.",
  category="decision",
  scope="project",
  entity="AccessControl",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"access": true, "policy": true}
)
```

```
add_memories(
  text="Rotate JWT signing keys quarterly; follow SECRET-ROTATION.md.",
  category="security",
  scope="project",
  entity="SecurityOps",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"security": true, "rotation": true}
)
```

```
add_memories(
  text="MCP is the Model Context Protocol used for tool access and automation.",
  category="glossary",
  scope="project",
  entity="MCP",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"mcp": true, "protocol": true}
)
```

```
add_memories(
  text="If OpenSearch is down, restart the container and re-run the /api/v1/search health check.",
  category="runbook",
  scope="project",
  entity="OpsRunbook",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"opensearch": true, "runbook": true}
)
```

```
add_memories(
  text="searchservice is a legacy alias for SearchService in old docs.",
  category="architecture",
  scope="project",
  entity="searchservice",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"search": true, "vector": true}
)
```

Collect IDs for later steps:
```
list_memories()
```
Record the memory IDs for the eight new entries.

---

## 2) Memory CRUD Tools

### add_memories
- Already exercised in the seed dataset. Confirm each response returns an ID or is visible via list_memories.
- evidence must be a list of strings.
- tags must be a dictionary of key/value pairs.
- access_entity can be omitted or set to "auto" when exactly one matching grant exists.
- If multiple matching grants exist for the scope, the response should include:
  - code="ACCESS_ENTITY_AMBIGUOUS" with an options array.

Extra add_memories cases:
```
add_memories(
  text="Test evidence/tags standard format",
  category="workflow",
  scope="project",
  entity="Normalization",
  access_entity="project:default_org/coding-brain",
  evidence=["ADR-100", "ADR-101"],
  tags={"one": true, "two": true}
)
```

```
add_memories(
  text="Auto access_entity with single team grant",
  category="decision",
  scope="team",
  entity="AccessControl",
  access_entity="auto"
)
```

```
add_memories(
  text="Auto access_entity with multiple team grants",
  category="decision",
  scope="team",
  entity="AccessControl",
  access_entity="auto"
)
```
Expected: error with code ACCESS_ENTITY_AMBIGUOUS and options list.

Expected results (examples):
```
{
  "error": "access_entity is required for scope='team'. Multiple grants available.",
  "code": "ACCESS_ENTITY_AMBIGUOUS",
  "options": [
    "team:default_org/dev",
    "team:default_org/ops"
  ],
  "hint": "Pick one of the available access_entity values."
}
```

```
{
  "id": "<memory-id>",
  "memory": "Test evidence/tags standard format",
  "tags": {
    "one": true,
    "two": true
  },
  "evidence": [
    "ADR-100",
    "ADR-101"
  ]
}
```

### list_memories
- Verify all seed memories appear and collect IDs.

### search_memory
- Validate semantic retrieval and metadata re-ranking.
```
search_memory(query="OpenSearch lexical search", limit=5)
```
```
search_memory(query="ADR dependency", entity="ADRTool", limit=5)
```

### update_memory
- Update one memory (e.g., the ADRTool memory) to add a tag and tweak wording.
```
update_memory(
  memory_id="<ADRTool-memory-id>",
  text="Use ADR automation whenever a new dependency is added in requirements.txt or pyproject.toml.",
  add_tags={"confirmed": true}
)
```
Additional update_memory formats:
```
update_memory(
  memory_id="<ADRTool-memory-id>",
  add_tags={"priority": true, "ops": true},
  remove_tags=["draft", "old"]
)
```
Expected: tags are updated and removed as provided.

### delete_memories
- Delete only the seed memories at the end of testing (use the IDs you collected).
```
delete_memories(memory_ids=["<id-1>", "<id-2>", "<id-3>", "<id-4>", "<id-5>", "<id-6>", "<id-7>", "<id-8>"])
```

### delete_all_memories
- Only run this in a disposable test user or empty environment.
```
delete_all_memories()
```

---

## 3) Graph Tools (Memory Graph)

### graph_related_memories
- Use the SearchService architecture memory as seed.
```
graph_related_memories(memory_id="<SearchService-arch-id>")
```
Expected: related memories sharing tags like search or vector.

### graph_subgraph
```
graph_subgraph(memory_id="<SearchService-arch-id>", depth=2, related_limit=10)
```
Expected: memory + tag/entity nodes and connected memories.

### graph_aggregate
```
graph_aggregate(group_by="category")
```
```
graph_aggregate(group_by="tag")
```
Expected: buckets for architecture, performance, security, etc., and tag counts.

### graph_tag_cooccurrence
```
graph_tag_cooccurrence(limit=10, min_count=2)
```
Expected: pairs like (search, vector) or (search, performance).

### graph_related_tags
```
graph_related_tags(tag_key="vector", limit=10)
```
Expected: related tags like search or adr (if co-occurring).

### graph_path_between_entities
```
graph_path_between_entities(entity_a="SearchService", entity_b="ADRTool", max_hops=6)
```
Expected: a path via shared tags (vector).

### graph_similar_memories
```
graph_similar_memories(memory_id="<SearchService-arch-id>", limit=5)
```
Expected: similar memories if OM_SIMILAR edges exist; empty is acceptable if similarity graph is not built.

### graph_entity_network
```
graph_entity_network(entity_name="SearchService", limit=10)
```
Expected: co-mention network if entity co-mention extraction is enabled; empty is acceptable otherwise.

### graph_entity_relations
```
graph_entity_relations(entity_name="SecurityOps", direction="both", limit=10)
```
Expected: typed relations if LLM graph extraction is enabled; empty is acceptable otherwise.

### graph_biography_timeline
```
graph_biography_timeline(entity_name="SearchService", limit=10)
```
Expected: timeline entries only if temporal event extraction is enabled; empty is acceptable otherwise.

### graph_normalize_entities
```
graph_normalize_entities(dry_run=true)
```
Expected: detect potential duplicate entities like SearchService vs searchservice.

### graph_normalize_entities_semantic
```
graph_normalize_entities_semantic(mode="detect", threshold=0.7)
```
Expected: similar duplicate detection with semantic matching.

---

## 4) Code Intelligence Tools

### index_codebase
```
index_codebase(repo_id="coding-brain", root_path="/usr/src/coding-brain", reset=true)
```
Expected: non-zero counts for files_indexed and symbols_indexed if indexing is configured.

If indexing times out, run in async mode and poll status:
```
index_codebase(repo_id="coding-brain", root_path="/usr/src/coding-brain", reset=true, async_mode=true)
index_codebase_status(job_id="<job-id>")
```
Expected: status transitions (queued -> running -> succeeded/failed) and progress fields.

### index_codebase_cancel
Start a new async job and request cancellation:
```
index_codebase(repo_id="coding-brain", root_path="/usr/src/coding-brain", reset=true, async_mode=true)
index_codebase_cancel(job_id="<job-id>")
index_codebase_status(job_id="<job-id>")
```
Expected: cancel_requested immediately; status eventually becomes canceled.

### search_code_hybrid
```
search_code_hybrid(query="get_code_toolkit", repo_id="coding-brain", limit=5)
```
Expected: results with symbol info and snippets.

### explain_code
- Use a symbol_id from search_code_hybrid results.
```
explain_code(symbol_id="<symbol-id>")
```
Expected: explanation with callers/callees.

### find_callers
```
find_callers(repo_id="coding-brain", symbol_id="<symbol-id>", depth=2)
```

### find_callees
```
find_callees(repo_id="coding-brain", symbol_id="<symbol-id>", depth=2)
```

### impact_analysis
```
impact_analysis(repo_id="coding-brain", changed_files=["openmemory/api/app/code_toolkit.py"], max_depth=3)
```
Expected: affected_files list (may be empty if graph not built).

---

## 5) Code Change Tools (MCP)

### adr_automation
```
adr_automation(changes=[
  {
    "file_path": "requirements.txt",
    "change_type": "modified",
    "diff": "+redis>=4.0.0",
    "added_lines": ["redis>=4.0.0"],
    "removed_lines": []
  }
], min_confidence=0.6)
```
Expected: should_create_adr true with dependency reason.

### test_generation
- Use either symbol_id or file_path (file_path is simpler if symbols not indexed).
```
test_generation(file_path="openmemory/api/app/code_toolkit.py", framework="pytest")
```
Expected: test_cases and file_content (may be empty if parsers unavailable).

### pr_analysis
```
pr_analysis(
  repo_id="coding-brain",
  diff="diff --git a/openmemory/api/app/code_toolkit.py b/openmemory/api/app/code_toolkit.py\n+toolkit.adr_tool = create_adr_automation_tool(...)\n",
  title="Wire ADR tool factory",
  check_security=true,
  check_adr=true
)
```
Expected: summary and issues lists, meta.request_id.

---

## 6) Cleanup
- Prefer delete_memories for only the seed dataset. Use delete_all_memories only in disposable test environments.
- Re-run list_memories to confirm cleanup.

---

## 7) Optional Access Control Checks
Only run if you can authenticate as two different users.

### Job ownership checks
1) User A starts an async indexing job.
2) User B calls index_codebase_status(job_id="<job-id>") and index_codebase_cancel(job_id="<job-id>").
Expected: Access denied unless User B has admin:read/admin:write.

---

## 8) Optional Indexing Job Tests (REST + Queue)

### REST job listing
```
GET /api/v1/code/index/jobs?repo_id=coding-brain&status=running&limit=10
```
Expected: only running jobs for the repo, count matches list length.

### force=true behavior
1) Start an async job.
2) Start a second job with `force=true`.
Expected: the first job becomes cancel_requested/canceled; the second job is queued.

### Queue saturation
Set `MAX_QUEUED_JOBS=1` and create two async jobs quickly.
Expected: the second request fails with HTTP 429 (QueueFullError).
