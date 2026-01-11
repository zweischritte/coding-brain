# Coding Brain - MCP Tools Guide

You have access to the Coding Brain memory system via MCP tools. Use these tools to store, search, and retrieve information.

## Quick Reference

### Most Used Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `search_memory` | Find memories | `search_memory(query="auth flow", limit=5)` |
| `add_memories` | Store new memory | `add_memories(text="...", category="decision", scope="user", entity="Auth")` |
| `add_memories_status` | Check async add result | `add_memories_status(job_id="<job-id>")` |
| `list_memories` | Show all memories | `list_memories()` |
| `update_memory` | Modify memory | `update_memory(memory_id="<uuid>", text="new text")` |
| `delete_memories` | Remove memories | `delete_memories(memory_ids=["<uuid>"])` |

---

## Adding Memories

Always include: `text`, `category`, `entity`. Include `scope` or `access_entity` (scope can be derived from access_entity).

```
add_memories(
  text="Use pytest for all tests",
  category="workflow",
  scope="user",
  entity="Testing"
)
```
Async default: `add_memories` returns a `job_id`; call `add_memories_status(job_id)` to fetch the result, or pass `async_mode=false`.

### Categories
`decision` | `convention` | `architecture` | `dependency` | `workflow` | `testing` | `security` | `performance` | `runbook` | `glossary`

### Scopes
`user` (personal) | `project` (this repo) | `team` (shared team) | `org` (company-wide)

---

## Searching Memories

Basic search:
```
search_memory(query="database connection", limit=10)
```

Filtered search:
```
search_memory(query="auth", category="architecture", scope="project")
```

Time-bounded search:
```
search_memory(
  query="Project XXX",
  scope="project",
  entity="XXX",
  created_after="2025-02-14T12:10:00Z",
  created_before="2025-02-14T12:30:00Z",
  limit=50
)
```
Note: `list_memories()` is unfiltered and can be noisy; use `created_after`/`created_before` with `search_memory` for time windows. `search_memory` caps at 50 results, so page by narrowing the window if needed.

Hard filters (pre-search, applied before vector search):
```
search_memory(query="auth", filter_tags="layer=backend,shared", filter_mode="all", limit=5)
search_memory(query="incident", filter_evidence="INC-123,ADR-04", filter_mode="any", limit=5)
search_memory(query="routing", filter_access_entity="project:default_org/coding-brain", limit=5)
```
Notes:
- `filter_tags` accepts `key` (boolean tag == true) or `key=value` (string tag value).
- `filter_mode` controls tag/evidence matching (`all` default, or `any`).
- `tags` remains a soft boost; use `filter_tags` for strict filtering.

---

## Updating Memories

Change text:
```
update_memory(memory_id="abc-123", text="Updated content here")
```

Add tags:
```
update_memory(memory_id="abc-123", add_tags={"priority": "high"})
```

---

## Graph Tools

Find related memories:
```
graph_related_memories(memory_id="<uuid>")
```

Find entity connections:
```
graph_entity_network(entity_name="Auth")
```

---

## Code Intelligence

Search code (requires indexed repo):
```
search_code_hybrid(query="authentication handler", limit=10)
```

Explain a symbol:
```
explain_code(symbol_id="<symbol-id>")
```

---

## Defaults for This Repo

When saving memories about this codebase:
- `scope="project"`
- `access_entity="project:default_org/coding-brain"`
- `artifact_type="repo"`
- `artifact_ref="coding-brain"`

---

## Shared Memory Entry Points (cloudfactory/shared)

- System Prompt Template: `03e4db30-daaa-422f-bb0b-e4417e9c263b`
- Shared Memory Index: `3dc502f7-eaeb-4efc-a9fc-99b09655934a`
- Coding Brain Shared Index: `ba93af28-784d-4262-b8f9-adb08c45acab` (load Friendly Quickstart `e02b4b2a-b976-4d19-85b7-c61f759793fb`)
- Tool-use policy: `f894b62b-a912-449b-b34a-9c425f70b795`
- Response style guide: `c7993fc9-2c92-4b1e-b80d-330b60bb2336`

---

## Rules

1. Always include `entity` - identifies what the memory is about
2. Shared scopes require `access_entity`; scope can be derived from `access_entity`
3. Use `search_memory` before adding to avoid duplicates
4. Prefer `update_memory` over delete + add
5. Be specific in memory text - include context

---

## Agent Guidelines (Verification Protocol)

When answering code questions:

- **VERIFY** before claiming (Read/Grep -> Quote -> Answer)
- **EXPRESS** uncertainty when unsure ("I don't know - let me check")
- **FOLLOW** user hints actively (search for mentioned terms)
- **QUOTE** code before making claims (no quotes = no claims)

---

## Common Patterns

### Save a decision
```
add_memories(
  text="Use Redis for session caching instead of in-memory",
  category="decision",
  scope="project",
  entity="SessionCache",
  access_entity="project:default_org/coding-brain",
  evidence=["PR-123"]
)
```

### Find and update
```
search_memory(query="session cache", limit=3)
# Then update with the returned memory_id
update_memory(memory_id="<uuid>", add_tags={"status": "implemented"})
```

### Check what you know
```
search_memory(query="<topic>", limit=10)
```
