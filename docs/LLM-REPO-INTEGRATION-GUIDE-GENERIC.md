# LLM Repo Integration Guide (Generic)

This document is for an LLM agent that works inside any repository and must integrate repo knowledge into a memory system correctly so developers get immediate value. It focuses on using existing memory, graph, search, and code indexing capabilities without adding features.

---

## CLAUDE.md Template for External Repos

Copy this template to your repo's `CLAUDE.md` and fill in the `[PLACEHOLDERS]`:

```markdown
# [YOUR_PROJECT] with Coding Brain

You have access to Coding Brain, a long-term memory system via MCP tools.

## Memory Config for This Repo

When storing memories:
- **scope**: `project`
- **access_entity**: `project:[YOUR_ORG]/[YOUR_REPO]`
- **artifact_type**: `repo`
- **artifact_ref**: `[YOUR_REPO]`
- **entity**: Always specify (e.g., "API", "Frontend", "AuthService")

## Key Rules

1. **Code > Memory**: Always verify memory against actual code files
2. **Store decisions**: Use `add_memories` for decisions, conventions, architecture
3. **Search first**: Use `search_memory` before asking about past context

## Tool Selection

| Task | Tool | Notes |
|------|------|-------|
| Find past decisions | `search_memory` | Add entity/category filters |
| Store new knowledge | `add_memories` | Always include entity |
| Find similar context | `graph_similar_memories` | Semantic similarity |
| Code discovery | `search_code_hybrid` | Then use Read for full context |

## Memory Categories

- `decision` - Choices made (why X over Y)
- `convention` - Patterns to follow
- `architecture` - System design
- `workflow` - Processes (deploy, review)
- `runbook` - Operational procedures

## Example: Store a Decision

```
add_memories(
  text="Use Tailwind for styling, avoid custom CSS",
  category="convention",
  scope="project",
  entity="Frontend",
  access_entity="project:[YOUR_ORG]/[YOUR_REPO]",
  evidence=["PR-123"]
)
```

## Example: Find Context

```
search_memory(query="authentication", entity="API", limit=5)
```

## Entities for This Repo

Define your main components:
- [Entity1] - Description
- [Entity2] - Description
- [Entity3] - Description
```

---

## Goal

Make repo knowledge discoverable and actionable by:
- Indexing code for code intelligence tools.
- Capturing high-signal repo knowledge as structured memories.
- Using consistent entities, tags, and access control.

## Required prerequisites

- Core services are healthy: API/MCP, metadata store, vector store, search index, graph store, and session store.
- A JWT or API token exists with the required scopes:
  - Memories: read/write (and delete if needed)
  - Code tools: read/write
  - Graph/UI: graph and entity read permissions
- For shared scope, set `access_entity` explicitly.

## Step 1: Index the repo (code tools)

Code tools only work after indexing. Run indexing before any code search or explain.

MCP (example):
```text
index_codebase(repo_id="<repo-id>", root_path="<container-visible-path>", reset=true)
```

REST (example):
```bash
curl -X POST http://<host>/api/v1/code/index \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"repo_id":"<repo-id>","root_path":"<container-visible-path>","reset":true}'
```

Notes:
- Use a container-visible path when indexing inside containers.
- Re-index after major refactors or directory moves.
- Use async indexing for large repos and poll job status.

## Step 2: Define entity map and metadata conventions

Memory quality depends on consistent entities and metadata. Define a stable entity map:

Suggested entity categories:
- Repo name
- Services or modules
- Key subsystems (auth, storage, search, billing, etc.)
- Data stores and infra components

Metadata rules (required):
- Always set `entity` (required by most systems for graph linking).
- Use the narrowest `scope` that fits (user, team, project, org).
- Use `access_entity` for any shared scope.
- Use `artifact_type` and `artifact_ref` for repo/file/component references.
- Keep tags consistent across the team.

## Step 3: Ingest repo knowledge as structured memories

Do not store raw code as memory. Use code indexing for code. Use memories for:
- Decisions and conventions
- Architecture summaries
- Runbooks and operational notes
- Access control and security expectations
- Known limitations and troubleshooting tips

Priority sources to read and summarize into memories:
- README and docs
- Architecture diagrams and ADRs
- Runbooks and incident notes
- Contribution guides and CI/CD docs

Memory template (example):
```text
add_memories(
  text="<1-2 sentence summary>",
  category="architecture|decision|runbook|workflow|dependency|security|performance|testing|convention",
  scope="project",
  entity="<entity>",
  access_entity="<access-entity>",
  artifact_type="repo",
  artifact_ref="<repo-name>",
  tags={"source":"docs/README.md"}
)
```

Guidance:
- Keep each memory short and atomic (one idea per memory).
- Include evidence for decisions (ADR/PR/issue IDs) when available.
- Use updates (not deletes) when knowledge changes.

## Step 4: Validate retrieval works immediately

Run quick checks after ingestion:

```text
search_memory(query="<key topic>", limit=5)
search_memory(query="<system name>", limit=5)
```

Graph checks:
```text
graph_related_memories(memory_id="<seed-id>")
graph_entity_network(entity_name="<entity>")
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
- Creating memories without `entity` or without `access_entity` for shared scopes.
- Mixing team-specific and org-wide knowledge in the same scope.
- Relying on recency weighting for long-lived architectural facts.
