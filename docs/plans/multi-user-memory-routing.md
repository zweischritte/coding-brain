# Multi-User Memory Routing Plan

## Goal
Enable multiple users to share a single Coding Brain instance while keeping personal memories private and shared memories visible only to explicitly granted groups. Use metadata-based routing so each user needs only one MCP connection.

## Non-Goals
- Cross-org sharing (org_id isolation remains).
- Full IAM/role management (JWT claims only).
- Protection from server admins with DB access.

## Current State (as implemented)
- Postgres list/filter/related endpoints filter by `Memory.user_id == current user`.
- MCP search filters Qdrant with `{"user_id": uid}`; `scope`/`entity` are boost-only.
- Graph projections and queries are keyed by `userId`; graph ops assume a single user.
- Qdrant/OpenSearch isolate by `org_id` only (no user grant filtering).
- `scope` is metadata; it is not enforced for access control.

## Gaps
- Shared memories cannot be read across users.
- Vector search and graph retrieval are hard-scoped to one `user_id`.
- Grant hierarchy (org -> project -> team) is undefined.

## Terminology
- Memory scope: `session | user | team | project | org | enterprise`
- Token scopes: OAuth permissions like `memories:read` (unrelated to memory scope)
- Access key: the string used for ACL decisions (field: `access_entity`)

## Access Model

Each memory has:
- owner: creator `user_id` (stored in `Memory.user_id`)
- scope: `user`/`team`/`project`/`org`/`enterprise`
- access_entity: identifies the sharing group

Read access:
- `scope=user` -> only owner
- `scope!=user` -> access_entity must be in resolved grants

Write access (decision):
- Decision: group-editable. Any member of access_entity can edit/delete.

### Access Key Field (decision)
Decision: add `access_entity` in metadata to avoid overloading `entity` (used for semantic graph).
Decision: group-editable shared memories. Rely on audit logs/backups for recovery.

### Grant Expansion
JWT includes a `grants` list (`team:...`, `project:...`, `org:...`).
Expand in a helper, for example:
- `org` grant -> all project/team under that org
- `project` grant -> all teams under that project
- `team` grant -> that team only

## Memory Classification

| Memory Type | scope | access_entity | Example |
|-------------|-------|------------|---------|
| Personal | `user` | `user:<username>` | "I prefer vim keybindings" |
| Team-shared | `team` | `team:<org>/<client>/<project>/<team>` | "Use conventional commits" |
| Project | `project` | `project:<org>/<client>/<project>` | "This repo uses pnpm" |
| Org-wide | `org` | `org:<org>` | "Company uses AWS" |

## Routing Rules (Prompt-Level)

1. "I prefer..." / "My setup..." -> personal
2. "We decided..." / "Team standard..." -> team
3. Working in shared repo -> default to project
4. Org-wide policy -> org
5. When uncertain, ask the user

Examples (use `access_entity` for access control; keep `entity` for semantics when needed):

~~~
# Personal
add_memories(text="I prefer 2-space indent", category="convention", scope="user", access_entity="user:grischa")

# Team
add_memories(text="Use conventional commits", category="convention", scope="team", access_entity="team:cloudfactory/acme/billing/backend")

# Project
add_memories(text="API uses JWT auth", category="architecture", scope="project", access_entity="project:cloudfactory/acme/billing-api")
```

## Implementation

### Phase 0: Decisions and docs
- Access_entity is the access-control field; `entity` stays semantic.
- Define grant hierarchy rules.
- Document that prompt routing is not security.

### Phase 1: Prompt template (no code changes)
Create `docs/templates/memory-routing-prompt.md`:

````markdown
## Memory Management

You have access to Coding Brain memory tools. Route memories based on context:

### Personal Memories (scope="user", access_entity="user:<USERNAME>")

- Personal preferences and working style
- Local environment setup
- Private notes and drafts
- Individual learning

### Team Memories (scope="team", access_entity="team:<ORG>/<CLIENT>/<PROJECT>/<TEAM>")

- Coding conventions and standards
- Architecture decisions (ADRs)
- Common patterns and solutions
- Onboarding knowledge

### Project Memories (scope="project", access_entity="project:<ORG>/<CLIENT>/<PROJECT>")

- Project-specific patterns
- Build/deploy configurations
- API contracts
- Tech debt notes

### Routing Rules

1. "I prefer..." / "My setup..." -> personal
2. "We decided..." / "Team standard..." -> team
3. Working in shared repo -> default to project
4. Architecture/convention -> team or org
5. When uncertain, ask user

### Examples

~~~
# Personal
add_memories(text="I prefer 2-space indent", category="convention", scope="user", access_entity="user:grischa")

# Team
add_memories(text="Use conventional commits", category="convention", scope="team", access_entity="team:cloudfactory/acme/billing/backend")

# Project
add_memories(text="API uses JWT auth", category="architecture", scope="project", access_entity="project:cloudfactory/acme/billing-api")
~~~
````

`entity` remains for semantic meaning; `access_entity` drives access control.

### Phase 2: Metadata + validation
- Add access_entity validation (format + required for `scope!=user`).
- Default access_entity for personal memories to `user:<sub>`.
- Add JSONB indexes on `metadata_.scope` and `metadata_.access_entity`.
- Backfill missing scope/access_entity (see migration section).

### Phase 3: Access control enforcement (core)
- Add `grants` to `TokenClaims` and JWT parsing; extend `generate_jwt.py`.
- Implement `resolve_access_keys(principal)` with hierarchy expansion.
- Update Postgres queries (list/filter/get/related) to include shared memories.
- Update MCP tools:
  - `add_memories`: reject writes to a shared access_entity outside grants
  - `update/delete`: enforce chosen write policy
- Include actor user id in `MemoryAccessLog` metadata for shared reads.

### Phase 4: Retrieval backend alignment
- Qdrant:
  - store access_entity in payload
  - allow filters that match any allowed access_entity (MatchAny/OR)
  - remove hard filter on `user_id` only
- Neo4j:
  - store access_entity on `OM_Memory` and edges
  - update graph queries to accept multiple access_entity values
- OpenSearch:
  - index access_entity
  - filter search results by allowed access_entity values

### Phase 5: UI filters (optional)
- Add scope and access_entity filters.
- Show scope badge and access_entity on memory cards.

### Phase 6: Optional hardening
- Per-user secret for JWT generation (optional, protects issuance only).
- External IdP for stronger isolation.

## Backfill and Migration
- For memories missing scope/access_entity:
  - `scope=user`
  - access_entity = `user:<owner>`
- Re-sync Qdrant/OpenSearch/Neo4j using existing sync scripts.
- Consider a feature flag to switch access model (single-user vs grants).

## Files to Change

### Required (Phase 1)
- `docs/templates/memory-routing-prompt.md` (create)
- `docs/README-CODING-BRAIN.md` (update)

### Required (Phase 3/4)
- `openmemory/api/app/security/types.py` (add `grants`)
- `openmemory/api/app/security/jwt.py` (parse `grants`)
- `openmemory/api/scripts/generate_jwt.py` (add `--grants`)
- `openmemory/api/app/routers/memories.py` (shared access filtering)
- `openmemory/api/app/mcp_server.py` (read/write enforcement, search filters)
- `openmemory/api/app/utils/structured_memory.py` (access_entity validation)
- `openmemory/api/app/stores/qdrant_store.py` (MatchAny/OR filters)
- `openmemory/api/app/routers/search.py` or `openmemory/api/app/stores/opensearch_store.py` (filter by access_entity)
- `openmemory/api/app/graph/metadata_projector.py` and `openmemory/api/app/graph/graph_ops.py` (access_entity filters)

### Optional (Phase 5)
- `openmemory/ui/app/memories/page.tsx`
- `openmemory/ui/components/memory-card.tsx`

### Optional (Phase 6)
- `openmemory/.env.example` (USER_SECRET_<user>)
- `openmemory/api/scripts/generate_jwt.py` (`--user-secret`)

## Access Entity Naming Convention

| Access Key Type | Format | Example |
|-----------------|--------|---------|
| User | `user:<username>` | `user:grischa` |
| Org | `org:<org>` | `org:cloudfactory` |
| Client | `client:<org>/<client>` | `client:cloudfactory/acme` |
| Project | `project:<org>/<client>/<project>` | `project:cloudfactory/acme/billing-api` |
| Team | `team:<org>/<client>/<project>/<team>` | `team:cloudfactory/acme/billing/backend` |
| Service | `service:<org>/<service>` | `service:cloudfactory/auth-gateway` |

### CloudFactory Hierarchy Example
```
org:cloudfactory
├── client:cloudfactory/acme
│   ├── project:cloudfactory/acme/billing-api
│   │   ├── team:cloudfactory/acme/billing/backend
│   │   ├── team:cloudfactory/acme/billing/frontend
│   │   └── team:cloudfactory/acme/billing/devops
│   └── project:cloudfactory/acme/auth-service
└── client:cloudfactory/bigcorp
    └── project:cloudfactory/bigcorp/dashboard
```

## Tagging Guidelines
Use `tags` for technical attributes that are not access control.

| Tag Key | Values | Purpose |
|---------|--------|---------|
| `lang` | `python`, `typescript`, `java`, `go`, `rust` | Programming language |
| `framework` | `fastapi`, `react`, `spring`, `gin` | Framework/library |
| `pattern` | `error-handling`, `auth`, `caching`, `logging` | Pattern type |
| `domain` | `billing`, `auth`, `notifications` | Business domain |
| `status` | `draft`, `approved`, `deprecated` | Decision status |

## User Setup

1. Admin adds org/user config and optional user secrets.
2. Each user generates a JWT with `grants`.
3. Each user configures `~/.claude.json` with their JWT.
4. Add the routing instructions to project `CLAUDE.md` or personal `~/.claude/CLAUDE.md`.

## Considerations

### Pros
- Personal memories stay private.
- Shared memories are first-class and enforceable.
- Single MCP connection per user.

### Cons
- Requires grant management and consistent access_entity naming.
- More complex search/graph filtering logic.

### Risks
- Access key hierarchy must be correct to avoid leakage.
- Graph and vector stores require reindex/backfill for new filters.

## Future Enhancements
- Auto-detect project context from git remote.
- Cache grant expansion results per request/session.
- Expose access_entity filters in the UI and MCP helpers.
