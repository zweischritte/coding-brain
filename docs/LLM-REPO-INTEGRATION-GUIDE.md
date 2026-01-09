# LLM Repo Integration Guide (Coding Brain)

This document is for an LLM agent that works inside this repo and must integrate repo knowledge into the Coding Brain memory system the right way, so developers get value immediately. It focuses on using the existing stack (MCP/REST, memory schema, graph, code indexing) without adding features.

## Goal

Make repo knowledge discoverable and actionable by:
- Indexing code for code intelligence tools.
- Capturing high-signal repo knowledge as structured memories.
- Using consistent entities, tags, and access control.

## Required prerequisites

- Core services are healthy: API/MCP, Postgres, Qdrant, OpenSearch, Neo4j, Valkey.
- A JWT is available with the required scopes:
  - `memories:read`, `memories:write` (and `memories:delete` if needed)
  - `code:read`, `code:write`
  - `graph:read`, `entities:read` for graph use
- Set `access_entity` explicitly for any shared data; scope is legacy metadata only.

## Step 1: Index the repo (code tools)

Code tools only work after indexing. Run indexing before any code search or explain.

MCP:
```text
index_codebase(repo_id="coding-brain", root_path="/usr/src/coding-brain", reset=true)
```

REST:
```bash
curl -X POST http://localhost:8865/api/v1/code/index \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"coding-brain","root_path":"/usr/src/coding-brain","reset":true}'
```

Notes:
- Use a container-visible path when indexing from inside MCP/REST containers.
- Re-index after major refactors or directory moves.
- Use `async_mode=true` for large repos and poll job status.

## Step 2: Define entity map and metadata conventions

Memory quality depends on consistent entities and metadata. Use a stable set of entities:

Suggested entities for this repo:
- Coding Brain
- OpenMemory API
- OpenMemory UI
- MCP Servers
- Memory Routing (access_entity)
- Code Indexing Pipeline
- CODE_* Graph
- OM_* Graph
- Qdrant
- OpenSearch
- Neo4j
- Valkey

Metadata rules (required):
- Always set `entity` (required by system rules).
- Use `access_entity="project:default_org/coding-brain"` for repo knowledge.
- Use `artifact_type="repo"` and `artifact_ref="coding-brain"` for repo-level memories.
- Scope is optional legacy metadata only.
- Tag important items consistently (e.g., `decision`, `runbook`, `workflow`, `security`).

## Step 3: Ingest repo knowledge as structured memories

Do not store raw code as memory. Use code indexing for code. Use memories for:
- Decisions and conventions
- Architecture summaries
- Runbooks and operational notes
- Access control and security expectations
- Known limitations and troubleshooting tips

Priority sources to read and summarize into memories:
- `docs/README-CODING-BRAIN.md`
- `docs/TECHNICAL-ARCHITECTURE.md`
- `docs/RUNBOOK-DEPLOYMENT.md`
- `docs/cookbooks/**` (pick relevant guides)
- `docs/PRD-REST-SEARCH-EMBEDDINGS-MCP-CODE-TOOLS.md`

Memory template:
```text
add_memories(
  text="<1-2 sentence summary>",
  category="architecture|decision|runbook|workflow|dependency|security|performance|testing|convention",
  entity="<one of the defined entities>",
  access_entity="project:default_org/coding-brain",
  artifact_type="repo",
  artifact_ref="coding-brain",
  tags={"source":"docs/README-CODING-BRAIN.md"}
)
```
Async note: `add_memories` defaults to async and returns a `job_id`. Use `add_memories_status(job_id)` to fetch the
result or set `async_mode=false` if you need the memory ID immediately.

Guidance:
- Keep each memory short and atomic (one idea per memory).
- Include evidence for decisions (ADR/PR/issue IDs) when available.
- Use updates (not deletes) when knowledge changes.

## Step 4: Validate retrieval works immediately

Run quick checks after ingestion:

```text
search_memory(query="access_entity routing", limit=5)
search_memory(query="code indexing", limit=5)
search_memory(query="MCP endpoints", limit=5)
```

Graph checks:
```text
graph_related_memories(memory_id="<seed-id>")
graph_entity_network(entity_name="OpenMemory API")
```

If results look noisy, improve entity naming or add tags and evidence.

## Step 5: Use code tools together with memory

Use memory to explain intent and code tools to ground behavior.

Typical flow:
1) Search memory for decisions and constraints.
2) Use code search for implementation locations.
3) Use callers/callees for impact.
4) Update memory with new learnings and evidence.

## Step 6: Ongoing hygiene

- Re-index code after major changes.
- Update or deprecate old memories rather than deleting them.
- Keep entities and tags consistent across the team.
- Ensure `access_entity` is set for any shared memory.
- Periodically review tag co-occurrence to catch taxonomy drift.

## Quick start checklist (LLM)

- [ ] Index repo for code tools.
- [ ] Define entity list and tag taxonomy.
- [ ] Ingest core docs as structured memories.
- [ ] Validate search and graph queries.
- [ ] Begin normal work: use memory + code tools together.

## Anti-patterns

- Storing entire files or large code blocks as memories.
- Creating memories without `entity` or without `access_entity` for shared data.
- Mixing team-specific and org-wide knowledge in the same access_entity.
- Relying on recency weighting for long-lived architectural facts.

## References

- `docs/README-CODING-BRAIN.md`
- `docs/OPERATIONS-READINESS.md`
- `docs/LLM-USAGE-IDEAS.md`
