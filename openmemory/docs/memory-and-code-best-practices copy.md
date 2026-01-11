# Memory and Code Inspection Best Practices

This document merges best practices from Coding Brain memories and code tool documentation in this repo.

## Memory capture and hygiene (MCP)

- Search before adding a new memory to avoid duplicates.
- Keep memories short and atomic: one idea per memory.
- Store decisions and conventions as memories; include evidence links (ADR/PR/issue) when available.
- Use consistent `entity` labels to build a navigable knowledge graph.
- Use `access_entity` to control visibility; store short-lived notes under user access and shared knowledge under shared project access.
- Tag memories with lifecycle states like `draft`, `validated`, `deprecated`, and update memories when knowledge changes (prefer update over delete).
- Save prompts that worked well as memories for reuse.
- Do not store raw code as memory. Use memories for decisions, conventions, architecture summaries, runbooks, access control/security expectations, and known limitations/troubleshooting tips.
- Prioritize summarizing README/docs, architecture diagrams/ADRs, runbooks/incident notes, and contribution guides/CI/CD docs.
- Hygiene triggers: consolidate duplicates, remove hallucinated or incorrect memories, and keep entries atomic; run cleanup when search results are noisy (example threshold: >30% irrelevant).

## Memory graph enrichment

- Choose the most specific entity (module/component/service) rather than generic labels like "Backend".
- Add systematic tags with namespaces to improve graph links: `framework:`, `module:`, `component:`, `pattern:`, `layer:`, `api:`, `principle:`.
- Add code references to the primary implementation file(s) when applicable.
- Avoid single-tag entries or non-namespaced tags; include `layer:` tags for code-related memories.

## Code inspection and code tools

- Indexing is required before using code tools; re-index after major refactors.
- For large repos, use async indexing and poll status with `index_codebase_status(job_id)`.
- Graph tools only reflect memory metadata; to validate the code index, run `search_code_hybrid` or `explain_code` and confirm results are returned.
- Use `search_code_hybrid` to locate entry points, then `find_callers` / `find_callees` to map dependencies.
- Read the full file after locating symbols; search results are snippets only.
- Use `impact_analysis` to estimate blast radius before changes; use `test_generation` for risky changes.
- Use `pr_analysis` to surface risks, conventions, and security issues; use ADR automation for architectural changes and store the ADR as a memory.
- If a symbol is not found, follow fallbacks (grep, then `search_code_hybrid`) and never guess callers.
- When semantic search is used as a fallback, verify relevance manually.
- If results look incomplete, the index may be stale; consider re-indexing with `index_codebase(reset=true)`.

## Grounding and verification

- Treat code as the source of truth; verify memory against code before claiming behavior.
- Use a memory + code flow: search memories for intent/constraints, use code tools for implementation, then update memory with new evidence.

## Sources

### Memories

- `memory 03e4db30-daaa-422f-bb0b-e4417e9c263b`: "Code > Memory: Always verify memory against actual code files" and "Search first: Use search_memory before asking about past context."
- `memory 9a98426c-0062-4193-9f79-b6a1f6cb5931`: "Store decisions and conventions..." "Use entity labels..." "Tag memories by lifecycle stage..." "update, not delete."
- `memory e74e0474-df5c-4c9e-beb3-d3c5da6184e4`: "Priority sources to read and summarize into memories..." and "Keep each memory short and atomic."
- `memory 778dd1a8-46ed-45eb-95b2-a8c87fa27807`: "Do not store raw code as memory..."
- `memory 37460cb9-c894-4ee2-9fde-b287ee006dcb`: "Save prompts that work well as memory entries..."
- `memory d1b59c0e-0798-4820-b4b2-a3cca70bdef5`: "Capture decisions, runbooks, and key learnings..." and "use search and graph queries..."
- `memory ee80a1ca-ab73-4a74-9bd6-642aaff03ed9`: "Code tools require indexing; refresh after major refactors..."
- `memory 232be81f-6a9f-44a9-b72d-b3d0d53042df`: "Code tools require indexing... use async_mode and poll index_codebase_status."
- `memory 2c038f0b-806a-4356-87cc-cba10ae02efc`: "graph_* tools only reflect memory metadata..." and "validate code tools with search_code_hybrid or explain_code."
- `memory b0c3be83-5de5-4d31-a03f-21d44834d512`: "Use memory to explain intent and code tools to ground behavior..."
- `memory a5e06e81-5f4e-4d14-aa71-1c35dca2006b`: "Use callers/callees... impact_analysis... test_generation... pr_analysis... ADR automation..."
- `memory f095dddf-2a3a-49c2-92f0-369faa6c7a05`: "Workflow: search_code_hybrid -> find_callers/callees -> read_file -> impact_analysis -> synthesize."
- `memory 6b41cdc6-e769-489d-a94f-40c3a7adb5a1`: "Graph enrichment checklist" (entities, tag namespaces, code references, avoid generic entities).
- `memory 449e0ab1-4473-42f6-a311-fc8f1932fff0`: "Memory hygiene: identify duplicates, use lifecycle tags, delete hallucinations, keep memories atomic; cleanup if results are noisy."

### Code references

- `api/app/mcp_server.py:3667`: "IMPORTANT: Results are snippets only. Use Read tool to see full file context before answering."
- `api/app/mcp_server.py:3748`: "FALLBACK: If symbol not found, use search_code_hybrid first, then Read the file directly."
- `api/app/mcp_server.py:3830`: "FALLBACK: If \"Symbol not found\", automatic cascade tries: 1. Grep ... 2. search_code_hybrid ..."
- `api/app/mcp_server.py:3835`: "NEVER guess callers - use fallback results or admit uncertainty."
- `api/tools/fallback_find_callers.py:507`: "Results from semantic search - verify relevance manually."
- `api/tools/fallback_find_callers.py:527`: "Index may be stale - consider re-indexing with index_codebase(reset=true)."
