# Coding Brain - MCP Tools Guide

You have access to the Coding Brain memory system via MCP tools. Use these tools to store, search, and retrieve information.

## Quick Reference

### Most Used Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `search_memory` | Find memories | `search_memory(query="auth flow", limit=5)` |
| `add_memories` | Store new memory | `add_memories(text="...", category="decision", scope="user", entity="Auth")` |
| `list_memories` | Show all memories | `list_memories()` |
| `update_memory` | Modify memory | `update_memory(memory_id="<uuid>", text="new text")` |
| `delete_memories` | Remove memories | `delete_memories(memory_ids=["<uuid>"])` |

---

## Adding Memories

Always include: `text`, `category`, `scope`, `entity`

```
add_memories(
  text="Use pytest for all tests",
  category="workflow",
  scope="user",
  entity="Testing"
)
```

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

## Rules

1. Always include `entity` - identifies what the memory is about
2. Use `search_memory` before adding to avoid duplicates
3. Prefer `update_memory` over delete + add
4. Be specific in memory text - include context

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
