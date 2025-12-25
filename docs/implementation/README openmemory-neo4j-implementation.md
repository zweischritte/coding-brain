# OpenMemory + Neo4j Integration (Qdrant + Graph Memory) — Implementation Documentation

This repository now integrates **Neo4j** into **OpenMemory (mem0)** in two complementary ways:

1) **Deterministic metadata graph** (`OM_*` namespace): projects your OpenMemory/Qdrant metadata (vault/layer/tags/etc.) into Neo4j as a queryable graph.
2) **Mem0 Graph Memory** (LLM‑extracted entity graph): builds **direct entity→entity relationships** from memory text using Mem0’s Graph Memory feature (stored as `:__Entity__` nodes when `base_label=true`).

The vector store (**Qdrant**) remains the primary semantic index; Neo4j adds **structure, traversal, and graph context**.

---

## 1) Architecture (what talks to what)

**Write path (new memories / updates)**

- `add_memories` (OpenMemory MCP) calls `mem0.Memory.add(...)`
  - writes **embeddings + payload** to **Qdrant**
  - if `mem0.graph_store` is configured, Mem0 also writes **entity graph** to **Neo4j**
- OpenMemory additionally projects deterministic metadata into Neo4j (`OM_*` graph)

**Read path (search)**

- `search_memory` remains **vector-first** (Qdrant retrieval + reranking)
- results are enriched with:
  - `meta_relations` (deterministic `OM_*` relations for the returned memory IDs)
  - `relations` (Mem0 Graph Memory relations returned by `memory_client.graph.search(...)`)

```mermaid
flowchart LR
  MCP[OpenMemory MCP] -->|add_memories| M[mem0.Memory]
  M -->|embeddings + payload| Q[Qdrant]
  M -->|LLM entity extraction| N[Neo4j :__Entity__ Graph]
  MCP -->|metadata projector| N2[Neo4j OM_* Metadata Graph]

  MCP -->|search_memory (vector)| Q
  MCP -->|enrich meta_relations| N2
  MCP -->|enrich relations| N
```

---

## 2) Neo4j deployment (Docker in this repo)

Neo4j runs as a service in `openmemory/docker-compose.yml:14`.

- Neo4j Browser: `http://localhost:7474`
- Bolt (drivers): `bolt://localhost:7687`
- Default credentials (from compose):
  - Username: `neo4j`
  - Password: `openmemory123`

OpenMemory MCP receives these via environment variables (compose sets them for the container):
- `NEO4J_URL=bolt://neo4j:7687`
- `NEO4J_USERNAME=neo4j`
- `NEO4J_PASSWORD=openmemory123`

Security note: change the password and restrict port exposure if you run this beyond localhost.

---

## 3) Dependencies (why Neo4j works in the MCP container)

The OpenMemory MCP image installs graph dependencies in `openmemory/api/requirements.txt:1`:

- `langchain-neo4j` (Mem0’s Neo4j integration via `Neo4jGraph`)
- `neo4j` (Neo4j Python driver)
- `rank-bm25` (Mem0 graph search reranking)

---

## 4) Configuration storage and enablement

### 4.1 Where config is stored

OpenMemory stores configuration in the SQL DB (SQLite by default) in `configs.key='main'`:
- DB file: `openmemory/api/openmemory.db`
- API router: `openmemory/api/app/routers/config.py`
- Mem0 client loader: `openmemory/api/app/utils/memory.py`

### 4.2 Mem0 `graph_store` config (Neo4j Graph Memory)

Mem0 Graph Memory is enabled when `mem0.graph_store` is present in the stored config.

Example (as stored; secrets are *env references*):

```json
{
  "mem0": {
    "graph_store": {
      "provider": "neo4j",
      "config": {
        "url": "env:NEO4J_URL",
        "username": "env:NEO4J_USERNAME",
        "password": "env:NEO4J_PASSWORD",
        "database": "neo4j",
        "base_label": true
      },
      "threshold": 0.75
    }
  }
}
```

Notes:
- `threshold` is a **top‑level** field of `graph_store` (not inside `graph_store.config`).
- `base_label=true` makes all entities use the `:__Entity__` label (recommended for querying/indexing).

### 4.3 Enable script (writes config into DB)

Script: `openmemory/api/app/scripts/enable_mem0_graph_memory.py`

Runs inside the MCP container:

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.enable_mem0_graph_memory --base-label --threshold 0.75 --database neo4j
```

### 4.4 Config persistence fix (important)

OpenMemory’s config is stored in a JSON column. In-place mutation of nested dicts is not reliably tracked by SQLAlchemy, so updates could silently not persist.

Fix: deep-copy on read/write in `openmemory/api/app/routers/config.py:1` and scripts (e.g. `openmemory/api/app/scripts/enable_mem0_graph_memory.py:1`).

### 4.5 Config API endpoints (REST)

All config endpoints are under `/api/v1/config` (see `openmemory/api/app/routers/config.py`).

Graph store (Mem0 Graph Memory):
- `GET /api/v1/config/mem0/graph_store`
- `PUT /api/v1/config/mem0/graph_store`
- `DELETE /api/v1/config/mem0/graph_store`

Whole-config update options:
- `GET /api/v1/config/`
- `PATCH /api/v1/config/` (partial update, recommended)
- `PUT /api/v1/config/` (full overwrite)

---

## 5) Neo4j graphs & schema

Neo4j holds **two “subgraphs”** in the same database, separated by label/relationship namespaces.

### 5.1 Deterministic metadata graph (`OM_*`)

Purpose:
- translate **vector metadata** (vault/layer/tags/re/etc.) into **queryable relations**
- support deterministic traversal/aggregation from MCP tools

Implementation:
- projector: `openmemory/api/app/graph/metadata_projector.py`
- Neo4j driver wrapper: `openmemory/api/app/graph/neo4j_client.py`
- MCP integration helpers: `openmemory/api/app/graph/graph_ops.py`

**Naming Conventions** (per Neo4j best practices)

- Node labels: CamelCase (e.g., `OM_Memory`, `OM_Entity`)
- Relationship types: SNAKE_CASE (e.g., `OM_ABOUT`, `OM_IN_VAULT`)
- Properties: camelCase (e.g., `userId`, `createdAt`, `memoryIds`)

**Node labels**
- `OM_Memory` (one per memory UUID)
- `OM_Entity` (`metadata.re`)
- `OM_Vault` (`metadata.vault`)
- `OM_Layer` (`metadata.layer`)
- `OM_Vector` (`metadata.vector`)
- `OM_Circuit` (`metadata.circuit`)
- `OM_Tag` (`metadata.tags` keys)
- `OM_Origin` (`metadata.from` / `metadata.origin`)
- `OM_Evidence` (`metadata.ev`)
- `OM_App` (`metadata.source_app`, `metadata.mcp_client`)

**Relationship types**
- `(OM_Memory)-[:OM_ABOUT]->(OM_Entity)`
- `(OM_Memory)-[:OM_IN_VAULT]->(OM_Vault)`
- `(OM_Memory)-[:OM_IN_LAYER]->(OM_Layer)`
- `(OM_Memory)-[:OM_HAS_VECTOR]->(OM_Vector)`
- `(OM_Memory)-[:OM_IN_CIRCUIT]->(OM_Circuit)`
- `(OM_Memory)-[:OM_TAGGED {tagValue: <serialized> }]->(OM_Tag)`
- `(OM_Memory)-[:OM_DERIVED_FROM]->(OM_Origin)`
- `(OM_Memory)-[:OM_EVIDENCE]->(OM_Evidence)`
- `(OM_Memory)-[:OM_WRITTEN_VIA]->(OM_App)`

**Enhanced Semantic Connection Edges** (materialized graph edges for fast traversal)

Beyond memory→dimension edges, the `OM_*` graph now includes **three additional edge types** that materialize semantic connections for O(1) graph lookups:

1. **`OM_SIMILAR`** (Memory-to-Memory similarity edges)
   - Schema: `(OM_Memory)-[:OM_SIMILAR {score, rank, userId, createdAt, updatedAt}]->(OM_Memory)`
   - Source: Qdrant embeddings (cosine similarity)
   - Pre-computed K nearest neighbors per memory
   - Config (env vars):
     - `OM_SIMILARITY_K`: neighbors per memory (default: 20)
     - `OM_SIMILARITY_THRESHOLD`: min cosine similarity (default: 0.6)
     - `OM_SIMILARITY_MAX_EDGES`: max edges per memory (default: 30)

2. **`OM_CO_MENTIONED`** (Entity-to-Entity co-mention edges)
   - Schema: `(OM_Entity)-[:OM_CO_MENTIONED {count, memoryIds, userId, createdAt, updatedAt}]->(OM_Entity)`
   - Source: deterministic — entities that appear in the same memory(ies)
   - `memoryIds` provides provenance (which memories mention both)

3. **`OM_COOCCURS`** (Tag-to-Tag co-occurrence edges with PMI)
   - Schema: `(OM_Tag)-[:OM_COOCCURS {count, pmi, npmi, userId, createdAt, updatedAt}]->(OM_Tag)`
   - Source: deterministic — tags that appear together on the same memories
   - Includes **Pointwise Mutual Information** (PMI/NPMI) for relevance scoring
   - NPMI formula: `log(P(a,b) / (P(a)*P(b))) / -log(P(a,b))`

**Design notes:**
- This graph is **deterministic** (no LLM for the base `OM_*` edges).
- It encodes your metadata as shared dimension nodes, enabling aggregations and traversals.
- The enhanced edges (`OM_SIMILAR`, `OM_CO_MENTIONED`, `OM_COOCCURS`) add direct entity→entity and tag→tag connections for richer graph traversal.
- `OM_SIMILAR` edges are computed from Qdrant embeddings; `OM_CO_MENTIONED` and `OM_COOCCURS` are pure Cypher projections.

### 5.2 Mem0 Graph Memory entity graph (`:__Entity__`)

Purpose:
- build **direct relationships between entities** based on memory text (e.g. “Gisela collaborated with Martin”)
- answer graph-style questions that require entity adjacency

Implementation (Mem0 core):
- `mem0/memory/graph_memory.py`
- enabled via OpenMemory config (`mem0.graph_store`) loaded by `openmemory/api/app/utils/memory.py`

When `base_label=true`, entities are stored as:
- Nodes: `(:__Entity__ {user_id, name, embedding, ...})`
- Relationships: dynamic relationship types extracted from text (e.g. `:kollege_von`, `:worked_at`, etc.)

Mem0 Graph Memory internally uses:
- LLM extraction for entities and relations (tool calls in Mem0)
- embeddings for node matching and similarity search in Neo4j (`vector.similarity.cosine`)
- BM25 reranking for returned relation triples in `graph.search(...)`

Important limitation:
- Mem0’s Neo4j graph backend does **not store per-memory provenance** on edges. Edges are a derived, evolving view.

### 5.3 Indexes and constraints (performance + idempotency)

Deterministic `OM_*` graph:
- Constraints/indexes are created by the projector (`ensure_constraints()`), see `openmemory/api/app/graph/metadata_projector.py`.
- Key ones include:
  - unique `OM_Memory.id`
  - composite unique `OM_Entity (userId, name)`
  - unique dimension nodes like `OM_Vault.name`, `OM_Tag.key`, etc.
  - index on `OM_Memory.userId`
  - composite index on `OM_Memory.(userId, createdAt)` for time-range queries
  - indexes on `OM_CO_MENTIONED.userId`, `OM_COOCCURS.userId`, `OM_SIMILAR.userId`

Mem0 Graph Memory (`:__Entity__` graph):
- When `base_label=true`, Mem0 creates:
  - `CREATE INDEX entity_single IF NOT EXISTS FOR (n:`__Entity__`) ON (n.user_id)`
  - a composite `(n.name, n.user_id)` index is attempted but only succeeds on editions that support it

### 5.4 What is stored where (SQL vs Qdrant vs Neo4j)

| Store | Role | Source of truth? | What it contains |
|---|---|---:|---|
| SQL (`openmemory.db`) | OpenMemory system-of-record | Yes | Memory lifecycle (active/deleted), metadata edits, permissions, timestamps |
| Qdrant (`mem0_store`) | Vector index | No (derived) | Embeddings + payload (used for semantic search); payload metadata can become stale if SQL metadata changes |
| Neo4j (`neo4j`) | Graph context store | No (derived) | `OM_*` deterministic metadata graph + Mem0 `:__Entity__` entity relationship graph |

Operational implication:
- For deterministic metadata backfills, prefer SQL (`backfill_neo4j_from_db.py`) because it reflects the current OpenMemory state and metadata.

---

## 6) MCP tools (what you can do from OpenMemory MCP)

All tools live in `openmemory/api/app/mcp_server.py`.

### 6.1 Vector search + graph enrichment

Tool: `search_memory`

Returns:
- `results[]` (vector hits from Qdrant, reranked with metadata boosts)
- optional `meta_relations` (metadata graph relations for the returned memory IDs)
- optional `relations` (Mem0 Graph Memory relations related to the query)

This is the “best of both worlds” default: vector recall + structured graph context.

Response shapes (high level):
- `meta_relations` is keyed by memory UUID: `{ "<memory_id>": [ {type, target_label, target_value, value?}, ... ] }`
- `relations` is a list of entity triples: `[ {source, relationship, destination}, ... ]`

### 6.2 Deterministic graph query tools (OM_* metadata graph)

These tools query the deterministic `OM_*` graph (no LLM):

- `graph_related_memories(memory_id, via?, limit?)`
  - “Find memories that share tags/entities/vault/layer/etc. with a seed memory.”

- `graph_subgraph(memory_id, depth?, via?, related_limit?)`
  - Returns a small JSON subgraph around a memory (memory + dimension nodes + optionally related memories).
  - **Token Optimization:** The `related[]` array contains only non-redundant fields (`id`, `sharedCount`, `sharedRelations`). Full memory content and metadata are available in the `nodes[]` array, avoiding ~23% token overhead from content duplication.

- `graph_aggregate(group_by, limit?)`
  - Aggregates memories by `vault|layer|tag|entity|app|vector|circuit|origin|evidence|source|state`.

- `graph_tag_cooccurrence(limit?, min_count?, sample_size?)`
  - Returns tag pairs that frequently co-occur across memories.

- `graph_path_between_entities(entity_a, entity_b, max_hops?)`
  - Shortest path between two `OM_Entity` nodes via memory/dimension nodes.
  - **Now includes typed relationships:** Traverses `OM_RELATION` (e.g., `bruder_von`, `arbeitet_bei`) and `OM_CO_MENTIONED` edges for semantic path finding.
  - Example: `paul → [bruder_von] → marie → [plant_besuch] → el_juego → [gruendete] ← marius`

### 6.3 Enhanced semantic connection tools (materialized edges)

These tools query the pre-computed `OM_SIMILAR`, `OM_CO_MENTIONED`, and `OM_COOCCURS` edges:

- `graph_similar_memories(memory_id, min_score?, limit?)`
  - Get semantically similar memories via pre-computed `OM_SIMILAR` edges
  - O(1) graph lookup — no embedding computation at query time
  - Returns: `[{id, content, vault, layer, similarity_score, rank}, ...]`

- `graph_entity_network(entity_name, min_count?, limit?)`
  - Get entities that co-occur with a given entity via `OM_CO_MENTIONED` edges
  - Returns: `[{entity, count, memory_ids}, ...]`
  - `memory_ids` provides provenance for each co-mention relationship

- `graph_related_tags(tag_key, min_count?, limit?)`
  - Get tags that co-occur with a given tag via `OM_COOCCURS` edges
  - Returns: `[{tag, count, pmi, npmi}, ...]`
  - PMI/NPMI scores indicate statistical relevance of the co-occurrence

### 6.4 Entity Normalization Tools

OpenMemory provides two complementary entity normalization approaches:

**Tool 1: `graph_normalize_entities` (Basic Case-Based Normalization)**

Finds and merges simple case variants and underscore/space variations (e.g., "BMG" and "bmg", "Matthias Coers" and "matthias_coers").

**Parameters:**

- `dry_run` (bool, default: true): If true, only show what would be merged
- `auto` (bool): If true, automatically merge all detected duplicates
- `canonical` (str): For manual merge: the target entity name
- `variants` (str): For manual merge: comma-separated list of variant names

**Examples:**

```python
# Detect duplicates (dry run)
graph_normalize_entities(dry_run=true)
# Returns: {duplicates: [{canonical: "bmg", variants: [{name: "BMG", memories: 47}, ...]}]}

# Auto-merge all detected duplicates
graph_normalize_entities(auto=true, dry_run=false)

# Manual merge specific variants
graph_normalize_entities(canonical="matthias_coers", variants="Matthias,matthias", dry_run=false)
```

**Implementation:** `openmemory/api/app/graph/entity_normalizer.py`

**Tool 2: `graph_normalize_entities_semantic` (Advanced Multi-Phase Normalization)**

Extends basic normalization with semantic similarity detection using multiple phases:

1. **String Similarity** (Levenshtein/fuzzy matching) - e.g., "matthias" ↔ "mathias"
2. **Prefix/Suffix Matching** - e.g., "marie" ↔ "marie_schubenz"
3. **Domain Normalization** - e.g., "eljuego.community" ↔ "el_juego"
4. **Embedding Similarity** (optional, API-based) - e.g., "CloudFactory" ↔ "CF GmbH"

Each phase contributes to a confidence score that determines whether entities should be merged.

**Parameters:**

- `mode` (str, default: "detect"): Operation mode - "detect", "preview", "execute"
- `threshold` (float, default: 0.7): Minimum confidence for merge (0.0-1.0)
- `canonical` (str): For manual merge: the target entity name
- `variants` (str): For manual merge: comma-separated variant names

**Examples:**

```python
# Detect semantic duplicates
graph_normalize_entities_semantic(mode="detect", threshold=0.7)
# Returns: {groups: [{canonical: "bmg", variants: ["BMG", "Bmg"], confidence: 0.95, sources: {...}}]}

# Preview merge impact
graph_normalize_entities_semantic(mode="preview", canonical="matthias_coers", variants="Matthias,matthias")
# Returns: estimated edge migrations

# Execute merge
graph_normalize_entities_semantic(mode="execute", canonical="matthias_coers", variants="Matthias,matthias")
# Returns: actual migration stats
```

**Implementation:** `openmemory/api/app/graph/semantic_entity_normalizer.py`

**Normalization Pipeline:**

When entities are merged, the following operations are performed in a safe, transactional manner:

1. **Edge Migration** (all edge types are migrated from variant to canonical):
   - `OM_ABOUT` edges (Memory → Entity)
   - `OM_CO_MENTIONED` edges (Entity ↔ Entity) - counts are aggregated
   - `OM_RELATION` edges (Entity → Entity with type) - counts are incremented
   - `OM_TEMPORAL` edges (Entity → TemporalEvent)

2. **Self-Referential Edge Cleanup**: Delete any `OM_CO_MENTIONED` edges between canonical and variant nodes

3. **Orphan Node Deletion**: Delete variant nodes that have no remaining edges

4. **Mem0 Graph Sync** (optional): Synchronize changes to Mem0's `__Entity__` graph if enabled

5. **GDS Signal Refresh** (optional): Update PageRank and degree metrics if GDS is available

**Edge Migration Implementation:** `openmemory/api/app/graph/entity_edge_migrator.py`

### 6.5 Typed Entity Relations Tool

Tool: `graph_entity_relations`

Get semantic relationships for an entity (e.g., "Grischa -[schwester_von]-> Julia").

**Parameters:**
- `entity_name` (str, required): Name of the entity
- `relation_types` (str): Comma-separated relation types (e.g., "schwester_von,bruder_von")
- `category` (str): Category filter (family, social, work, location, creative, membership, travel)
- `direction` (str): "outgoing", "incoming", or "both" (default)
- `limit` (int): Max relations to return (default: 50)

**Example:**
```python
graph_entity_relations(entity_name="grischa", category="family")
# Returns: {entity: "grischa", relations: [{target: "julia", type: "schwester_von", direction: "outgoing", count: 3}]}
```

**Implementation:** `openmemory/api/app/graph/relation_types.py` (category definitions) + `graph_ops.py` (queries)

**Note on Relation Types:**
The LLM can extract arbitrary relationship types - the categories in `relation_types.py` are only query filters, not extraction constraints. Unknown types are stored and traversable, they just don't belong to any category.

**Available Categories:**

- `family`: eltern_von, kind_von, mutter_von, vater_von, schwester_von, bruder_von, verwandt_mit, onkel_von, tante_von, cousin_von, grosseltern_von, enkel_von
- `social`: partner_von, freund_von, mitbewohner_von, bekannt_mit, verlobt_mit, verheiratet_mit
- `work`: arbeitet_bei, arbeitete_bei, kollege_von, arbeitspartner_von, gruendete, leitet, angestellt_bei, chef_von, mitarbeiter_von
- `location`: wohnt_in, wohnte_in, geboren_in, aufgewachsen_in, befindet_sich_in, lebt_in, stammt_aus
- `creative`: produzierte, wirkte_mit_in, regie_bei, schrieb, erschuf, komponierte, entwickelte
- `membership`: mitglied_von, aktiv_in, engagiert_in, teil_von, gehoert_zu
- `travel`: plant_besuch, reist_nach, besucht, fliegt_nach, faehrt_nach, war_in

### 6.6 Biographical Timeline Tool

Tool: `graph_biography_timeline`

Get chronological events (residences, projects, work history) for a person.

**Parameters:**
- `entity_name` (str): Person name (if omitted, shows all events)
- `event_types` (str): Comma-separated types (residence, education, work, project, relationship, health, travel, milestone)
- `start_year` (int): Filter events starting from this year
- `end_year` (int): Filter events ending by this year
- `limit` (int): Max events (default: 50)

**Examples:**
```python
# Get timeline for a specific person
graph_biography_timeline(entity_name="grischa")

# Get all projects between 2014-2018
graph_biography_timeline(event_types="project", start_year=2014, end_year=2018)
```

**Implementation:** `openmemory/api/app/graph/temporal_events.py`

### 6.7 Mem0 entity graph query (LLM)

Mem0 Graph Memory is exposed indirectly:
- `search_memory` adds `relations` by calling `memory_client.graph.search(query, filters={"user_id": ...})` via `openmemory/api/app/graph/graph_ops.py`.

If you want full flexibility (raw Cypher) for the `:__Entity__` graph, use Neo4j Browser/cypher-shell (see verification section).

---

## 7) Embedding mechanism (vector store + graph)

### 7.1 Qdrant embeddings (semantic memory)

OpenMemory uses a Mem0 embedder configured in `openmemory/api/app/utils/memory.py`:
- default: OpenAI `text-embedding-3-small`

Used for:
- embedding stored memories (Qdrant vectors)
- embedding search queries (Qdrant search)

### 7.2 Neo4j entity embeddings (graph memory)

Mem0 Graph Memory also embeds entities (typically based on entity names) and stores embeddings on Neo4j nodes.

Used for:
- matching new extracted entities to existing nodes (via cosine similarity + `threshold`)
- retrieving relevant neighborhood relations for a query

Tuning knobs:
- `graph_store.threshold`: higher = fewer merges (more distinct nodes), lower = more merges (risk of over-merging)
- `graph_store.custom_prompt`: constrain extraction to reduce noise and improve consistency

---

## 8) Backfill mechanisms (how existing memories were integrated)

There are **two separate backfills** depending on which graph you want populated.

### 8.1 Backfill the deterministic `OM_*` metadata graph (recommended baseline)

Script (from SQL source of truth):
- `openmemory/api/app/scripts/backfill_neo4j_from_db.py`

Why SQL and not Qdrant:
- SQL is OpenMemory’s source of truth for state + metadata edits.
- In this repo, `update_memory` updates SQL metadata; Qdrant payload metadata can be stale.

Run:
```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_neo4j_from_db --user-id grischadallmer
```

Logs:
- `openmemory/api/logs/neo4j_backfill_from_db_<timestamp>.log`

Alternative (from an OpenMemory export JSON):
- `openmemory/api/app/scripts/backfill_neo4j_metadata.py`

### 8.2 Backfill Mem0 Graph Memory (`:__Entity__` LLM graph)

Script:
- `openmemory/api/app/scripts/backfill_mem0_graph_from_db.py`

This is LLM + embeddings heavy and can be slow/costly.

Run:
```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_mem0_graph_from_db --user-id grischadallmer
```

Targeted retry support:
- `--retry-failed-from-log <logfile>` (parses `FAIL memory_id=...` lines)
- `--memory-ids <uuid,uuid,...>` (process only selected memories)

Example retry:
```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_mem0_graph_from_db \
    --user-id grischadallmer \
    --retry-failed-from-log logs/mem0_graph_backfill_20251219T084120Z.log \
    --log-file logs/mem0_graph_backfill_retry.log
```

Logs:
- `openmemory/api/logs/mem0_graph_backfill_*.log`

### 8.3 Backfill enhanced semantic connection edges

Three additional backfill scripts create the materialized `OM_SIMILAR`, `OM_CO_MENTIONED`, and `OM_COOCCURS` edges.

**Entity co-mention edges (`OM_CO_MENTIONED`)**

Script: `openmemory/api/app/scripts/backfill_entity_edges.py`

Fast operation — uses pure Cypher (no vector queries).

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_entity_edges --user-id grischadallmer
```

**Options:**

- `--min-count N` — minimum co-occurrences to create edge (default: 1)
- `--dry-run` — preview without writing

**Tag co-occurrence edges (`OM_COOCCURS`)**

Script: `openmemory/api/app/scripts/backfill_tag_edges.py`

Fast operation — uses pure Cypher with PMI calculation.

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_tag_edges --user-id grischadallmer --min-count 2
```

**Options:**

- `--min-count N` — minimum co-occurrences to create edge (default: 2)
- `--min-pmi FLOAT` — minimum PMI score to create edge (default: 0.0)
- `--dry-run` — preview without writing

**Similarity edges (`OM_SIMILAR`)**

Script: `openmemory/api/app/scripts/backfill_similarity_edges.py`

Slower operation — queries Qdrant for K nearest neighbors per memory.

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_similarity_edges --user-id grischadallmer
```

**Options:**

- `--limit N` — process at most N memories
- `--batch-size N` — progress log interval (default: 50)
- `--dry-run` — preview without writing
- `--verbose` — enable debug logging

**Configuration (environment variables):**

- `OM_SIMILARITY_K` — neighbors per memory (default: 20)
- `OM_SIMILARITY_THRESHOLD` — minimum cosine similarity (default: 0.6)
- `OM_SIMILARITY_MAX_EDGES` — maximum edges per memory (default: 30)

**Logs:**

- `openmemory/api/logs/backfill_entity_edges_*.log`
- `openmemory/api/logs/backfill_tag_edges_*.log`
- `openmemory/api/logs/backfill_similarity_edges_*.log`

**Entity Bridge backfill (multi-entity + typed relations)**

Script: `openmemory/api/app/scripts/backfill_entity_bridge.py`

This script uses Mem0's LLM to extract multiple entities from each memory's content, creating:
- Multiple `OM_ABOUT` edges per memory (fixes the single-entity limitation)
- `OM_RELATION` edges with typed relationships (e.g., "vater_von", "works_at")
- Updated `OM_CO_MENTIONED` edges (now functional with multi-entity data)

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_entity_bridge --user-id grischadallmer
```

**Options:**

- `--limit N` — process at most N memories
- `--batch-size N` — progress log interval (default: 10)
- `--dry-run` — extract entities but don't write to Neo4j
- `--verbose` — enable debug logging
- `--memory-ids ID,ID,...` — process only specific memory IDs

**Note:** This backfill requires Mem0 Graph Memory to be enabled (uses LLM extraction).

**Logs:**

- `openmemory/api/logs/backfill_entity_bridge_*.log`

**Combined Graph Enhancements Backfill**

Script: `openmemory/api/app/scripts/backfill_graph_enhancements.py`

This script runs all three graph enhancement phases in sequence:

1. **Entity Normalization** — Merge duplicate entity variants (case-insensitive, underscore handling)
2. **Relation Extraction** — Extract typed relations from existing memories using LLM
3. **Temporal Event Extraction** — Extract biographical events from date-tagged memories

```bash
# Dry run (preview what would happen)
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_graph_enhancements --user-id grischadallmer --dry-run

# Full execution
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_graph_enhancements --user-id grischadallmer --execute

# Run specific phases only
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_graph_enhancements --user-id grischadallmer --execute \
    --phases normalize relations
```

**Options:**

- `--user-id` (required): User ID to process
- `--dry-run`: Preview changes without writing
- `--execute`: Actually perform changes (required if not dry-run)
- `--phases`: Which phases to run (normalize, relations, temporal, all)

**Example Output:**

```
=== Phase 1: Entity Normalization ===
Found 85 duplicate groups
  → bmg ← [bmg(63), BMG(47)]
  → grischa ← [grischa(61), Grischa(26)]
Normalization results:
  - Groups merged: 85
  - OM_CO_MENTIONED migrated: 793
  - OM_RELATION migrated: 33

=== Phase 2: Relation Extraction ===
Processing 488 memories...
Relation extraction results:
  - Memories processed: 488
  - Entities bridged: 1200+
  - Relations created: 600+

=== Phase 3: Temporal Event Extraction ===
Scanning 488 memories for temporal data...
Temporal extraction results:
  - Memories scanned: 488
  - Events found: 45
  - Events created: 45
```

### 8.4 Sync Qdrant from SQLite (fixing missing vector embeddings)

When memories exist in SQLite but are missing from Qdrant (e.g., due to import issues or failed writes), use these sync scripts.

**Important:** SQLite is the source of truth. Qdrant should contain embeddings for all active memories in SQLite.

**Fast sync (recommended):**

Script: `openmemory/api/app/scripts/sync_qdrant_fast.py`

This script directly computes embeddings via OpenAI and upserts to Qdrant in batches, bypassing mem0's deduplication logic. It's significantly faster (~20 memories/second vs ~0.1 memories/second with the slow method).

```bash
# Dry run (show what would be synced)
docker exec openmemory-openmemory-mcp-1 \
  python -m app.scripts.sync_qdrant_fast --user-id grischadallmer --dry-run

# Full sync
docker exec openmemory-openmemory-mcp-1 \
  python -m app.scripts.sync_qdrant_fast --user-id grischadallmer

# Limit to N memories
docker exec openmemory-openmemory-mcp-1 \
  python -m app.scripts.sync_qdrant_fast --user-id grischadallmer --limit 50
```

**Slow sync (uses mem0's add method):**

Script: `openmemory/api/app/scripts/sync_qdrant_from_db.py`

This uses mem0's `add()` method which handles embedding but is slow (~10s per memory) and may create duplicate entries with new UUIDs.

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.sync_qdrant_from_db --user-id grischadallmer --dry-run
```

**UUID Format Note:**

SQLite stores UUIDs without dashes (e.g., `e8a2c353e9e8494c8e1050f16cc73df8`), while Qdrant uses the standard format with dashes (e.g., `e8a2c353-e9e8-494c-8e10-50f16cc73df8`). The sync scripts handle this conversion automatically via `normalize_uuid()` and `to_qdrant_uuid()` functions.

**After syncing Qdrant, run similarity edge backfill:**

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_similarity_edges --user-id grischadallmer
```

---

## 9) Relationship-type hardening (Neo4j syntax error fix)

Problem observed during backfill:
- Some LLM-extracted relationship types contained invalid characters for Cypher relationship type identifiers (e.g. `-`), causing Neo4j `SyntaxError`.

Fix implemented:
- OpenMemory patches Mem0’s relationship sanitizer at runtime in `openmemory/api/app/utils/memory.py:45` to ensure relationship type names are always valid Cypher identifiers (no hyphens, no leading digits).

This makes Mem0 Graph Memory ingestion robust against noisy LLM outputs.

---

## 10) Verification (how to check everything is working)

### 10.1 Via MCP responses

- `search_memory(...)` should include:
  - `meta_relations` when Neo4j metadata projection is available
  - `relations` when Mem0 Graph Memory is enabled

- Use the deterministic graph tools:
  - `graph_subgraph(memory_id=...)`
  - `graph_related_memories(memory_id=..., via="tag,entity")`
  - `graph_aggregate(group_by="vault")`

### 10.2 Via Neo4j (cypher-shell)

Count Mem0 Graph Memory edges:
```cypher
MATCH (a:`__Entity__` {user_id:'grischadallmer'})-[r]->(b:`__Entity__` {user_id:'grischadallmer'})
RETURN count(r);
```

Inspect a direct relationship:
```cypher
MATCH (a:`__Entity__` {user_id:'grischadallmer', name:'gisela_jahn'})-[r]-(b:`__Entity__` {user_id:'grischadallmer', name:'martin'})
RETURN type(r), properties(r);
```

Inspect deterministic metadata relations:
```cypher
MATCH (m:OM_Memory {userId:'grischadallmer'})-[r]->(d)
RETURN type(r), labels(d)[0], count(*) ORDER BY count(*) DESC;
```

Count enhanced semantic edges:
```cypher
// Similarity edges
MATCH ()-[r:OM_SIMILAR {userId:'grischadallmer'}]->() RETURN count(r);

// Entity co-mention edges
MATCH ()-[r:OM_CO_MENTIONED {userId:'grischadallmer'}]->() RETURN count(r);

// Tag co-occurrence edges
MATCH ()-[r:OM_COOCCURS {userId:'grischadallmer'}]->() RETURN count(r);
```

Explore similar memories:
```cypher
MATCH (m:OM_Memory {userId:'grischadallmer'})-[r:OM_SIMILAR]->(similar)
WHERE r.score > 0.7
RETURN m.id, similar.id, r.score, r.rank
ORDER BY r.score DESC LIMIT 20;
```

Explore tag co-occurrences with PMI:
```cypher
MATCH (t1:OM_Tag)-[r:OM_COOCCURS {userId:'grischadallmer'}]->(t2:OM_Tag)
RETURN t1.key, t2.key, r.count, r.npmi
ORDER BY r.npmi DESC LIMIT 20;
```

### 10.3 Memory lifecycle notes (updates/deletes)

- `add_memories` writes to Qdrant via Mem0 and updates SQL; it also projects metadata into Neo4j (`OM_*`) and (if enabled) Mem0 writes to the entity graph.
- `update_memory` updates SQL and re-projects the `OM_*` metadata graph for that memory.
- `delete_memories` removes from Qdrant and marks deleted in SQL; it also deletes the `OM_Memory` node from the `OM_*` graph.
- Mem0's `:__Entity__` graph is a derived view without per-memory provenance; deletes do not automatically "subtract" edges from Neo4j.

**Real-time hooks for enhanced edges:**

- On `add_memories`: similarity edges (`OM_SIMILAR`) are projected for new memories, entity co-mention edges (`OM_CO_MENTIONED`) are updated
- On `delete_memories`: similarity edges and entity co-mention edges are cleaned up for the deleted memory
- Tag co-occurrence edges (`OM_COOCCURS`) are batch-computed via backfill (not real-time)

---

## 11) Common questions & operational notes

### “Warum zwei Subgraphs?”

They’re in the same DB, but separated by naming:
- `OM_*` graph is deterministic + auditable and models metadata dimensions.
- `:__Entity__` graph is LLM-derived and models direct entity relations.

Keeping them separate avoids schema collisions and keeps deterministic metadata traversal stable even if LLM behavior changes.

If you want a single unified schema later, the next step is a **provenance graph** (memory nodes linked to entities + edges with `memory_id` attribution), but that requires extending Mem0’s graph writer or adding a third projector.

### “Warum sehe ich zu wenige Verbindungen?”

Two typical reasons:
- You’re looking at the `OM_*` graph: it links `Memory -> Entity/Tag/Layer/...` but does not create entity→entity edges.
- Mem0 Graph Memory wasn’t enabled/backfilled yet: enable `mem0.graph_store` and run the Mem0 graph backfill.

### “Wo sind Username/Passwort?”

For Docker deployment in this repo:
- `openmemory/docker-compose.yml` sets `NEO4J_AUTH=neo4j/openmemory123`
- OpenMemory MCP uses `NEO4J_USERNAME=neo4j` and `NEO4J_PASSWORD=openmemory123`

In Mem0 config we store `env:` references (not plaintext secrets) in the SQL config.

---

## 12) Tests

Unit tests were added to validate graph enrichment + projector behavior without a live Neo4j:
- `openmemory/api/tests/test_mcp_search_graph_enrichment.py`
- `openmemory/api/tests/test_neo4j_metadata_projector.py`

Run inside the MCP container:
```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 pytest -q
```

---

## 13) Best Practices Implementation

This section documents how OpenMemory's Neo4j integration follows industry best practices for semantic and relational graph structures.

### 13.1 Graph Data Modeling Best Practices

**Naming Conventions** (strictly followed):
- Node labels: CamelCase (`OM_Memory`, `OM_Entity`)
- Relationship types: SNAKE_CASE (`OM_ABOUT`, `OM_IN_VAULT`)
- Properties: camelCase (`userId`, `createdAt`, `memoryIds`)

**Query-First Design**:
The graph schema is designed around common query patterns:
- **Find related memories**: `Memory → Dimension ← Memory` (dimension nodes as join points)
- **Entity networks**: `Entity → OM_CO_MENTIONED → Entity` (pre-computed edges)
- **Semantic similarity**: `Memory → OM_SIMILAR → Memory` (vector-based edges)
- **Tag relationships**: `Tag → OM_COOCCURS → Tag` (statistical co-occurrence)

**Dimension Nodes (Fan-Out Pattern)**:
Rather than storing metadata as properties, shared values become dimension nodes:
- `OM_Vault`, `OM_Layer`, `OM_Vector`, `OM_Circuit` — AXIS taxonomy dimensions
- `OM_Entity`, `OM_Tag`, `OM_Origin` — semantic classification nodes

This enables efficient aggregations and multi-hop traversals.

### 13.2 Semantic Graph Structures

**Ontology Integration**:
The AXIS taxonomy (Vault/Layer/Vector/Circuit) creates a hierarchical semantic structure:
- Vault: Domain/sovereignty classification (SOV, WLT, SIG, FRC, DIR, FGP, Q)
- Layer: Processing depth (somatic → meta, 12 levels)
- Vector: Behavioral dimension (say, want, do)
- Circuit: Energy/integration level (1-8)

**Knowledge Graph Patterns**:
```
Memory → [dimensions] → Vault/Layer/etc.  # Classification
Memory → OM_ABOUT → Entity                 # Entity extraction
Entity → OM_CO_MENTIONED → Entity          # Relationship inference
Memory → OM_SIMILAR → Memory               # Semantic similarity
Tag → OM_COOCCURS → Tag                    # Statistical patterns
```

**Relationship Semantics**:
All relationship types are specific and meaningful:
- `OM_ABOUT`: Memory mentions/references entity
- `OM_IN_VAULT`, `OM_IN_LAYER`: Hierarchical classification
- `OM_DERIVED_FROM`: Provenance tracking
- `OM_CO_MENTIONED`: Statistical co-occurrence with count
- `OM_SIMILAR`: Semantic similarity with score

### 13.3 Vector Embeddings Integration

**Hybrid Vector + Graph Architecture**:
- **Qdrant**: Primary vector store for semantic search (dense embeddings)
- **Neo4j**: Graph store for structural relationships and pre-computed similarity
- **OM_SIMILAR edges**: Materialize vector similarity as graph edges

**Benefits**:
- O(1) graph traversal for similarity (no embedding computation at query time)
- Combine vector similarity with graph context in single query
- Richer context through relationship paths

**Configuration**:
```
OM_SIMILARITY_K=20              # K nearest neighbors per memory
OM_SIMILARITY_THRESHOLD=0.6     # Minimum cosine similarity
OM_SIMILARITY_MAX_EDGES=30      # Maximum edges per memory
```

### 13.4 Performance Optimization

**Index Strategy**:
Follows the principle of indexing high-cardinality properties used for filtering:

```cypher
// Unique constraints (auto-create indexes)
CREATE CONSTRAINT om_memory_id FOR (m:OM_Memory) REQUIRE m.id IS UNIQUE
CREATE CONSTRAINT om_entity_user_name FOR (e:OM_Entity) REQUIRE (e.userId, e.name) IS UNIQUE

// Range indexes for filtering
CREATE INDEX om_memory_user_id FOR (m:OM_Memory) ON (m.userId)
CREATE INDEX om_memory_user_created FOR (m:OM_Memory) ON (m.userId, m.createdAt)

// Relationship property indexes for edge filtering
CREATE INDEX om_similar_score FOR ()-[r:OM_SIMILAR]-() ON (r.score)
CREATE INDEX om_co_mentioned_user FOR ()-[r:OM_CO_MENTIONED]-() ON (r.userId)
```

**Query Optimization Techniques**:
1. **Parameter usage**: All queries use `$parameters` for query plan reuse
2. **Early filtering**: ACL (`allowedMemoryIds`) applied in WHERE clause
3. **Limited results**: All queries use LIMIT to prevent unbounded results
4. **Composite indexes**: Used for common multi-property filters (userId + createdAt)

**Supernode Prevention**:
- `OM_SIMILAR` edges limited to `max_edges_per_memory` (default: 30)
- Entity co-mentions store only top 5 `memoryIds` for provenance
- Dimension nodes (Vault, Layer, etc.) are finite/controlled

### 13.5 Memory/Knowledge System Patterns

**Temporal Context**:
- `createdAt`, `updatedAt`, `projectedAt` timestamps on memory nodes
- Edge timestamps for tracking relationship evolution

**Provenance Tracking**:
- `OM_DERIVED_FROM` → `OM_Origin`: Where memory came from
- `OM_EVIDENCE` → `OM_Evidence`: Supporting evidence
- `OM_WRITTEN_VIA` → `OM_App`: Application source
- `memoryIds` on `OM_CO_MENTIONED`: Which memories mention both entities

**Context Management**:
- `userId` on all user-scoped nodes and edges for multi-tenant isolation
- `allowedMemoryIds` ACL parameter for permission filtering
- Dual graph: deterministic `OM_*` graph + LLM-extracted `:__Entity__` graph

### 13.6 Statistical Edge Properties

**PMI (Pointwise Mutual Information) on OM_COOCCURS**:
```
PMI = log2(P(a,b) / (P(a) * P(b)))
NPMI = PMI / -log2(P(a,b))  # Normalized to [-1, 1]
```
Higher NPMI indicates statistically significant co-occurrence beyond chance.

**Scoring on OM_SIMILAR**:
- `score`: Cosine similarity from Qdrant (0.0-1.0)
- `rank`: Position in K nearest neighbors (0 = most similar)

**Count tracking on OM_CO_MENTIONED**:
- `count`: Number of memories mentioning both entities
- `memoryIds`: Sample of memory IDs for provenance (up to 5)

### 13.7 Graceful Degradation

All graph operations fail gracefully:
- If Neo4j is not configured: operations return empty results, don't block main flow
- If projection fails: logs warning, returns True to not fail parent operation
- If query fails: returns empty list/dict, caller handles gracefully

This ensures the system works with or without Neo4j, with Qdrant as the primary store.

---

## 14) Advanced Patterns (Future Enhancements)

### 14.1 Temporal Versioning

For tracking memory evolution over time:
```cypher
// Bi-temporal versioning pattern
(m:OM_Memory)-[:PREVIOUS_VERSION]->(old:OM_Memory)
// Properties: validFrom, validTo (transaction time)
// Properties: effectiveFrom, effectiveTo (event time)
```

### 14.2 Entity Resolution

Using Neo4j GDS for deduplicating similar entities:
```cypher
// FastRP embeddings + community detection
CALL gds.fastRP.stream('entities', {embeddingDimension: 128})
YIELD nodeId, embedding
// Then cluster with Leiden/Louvain for merge candidates
```

### 14.3 Semantic Hierarchy (Ontology)

For category-based inference:
```cypher
// Vault hierarchy with inference
(subVault)-[:IS_A]->(parentVault)
// Query: Find all memories in parent category (transitive)
MATCH (m:OM_Memory)-[:OM_IN_VAULT]->(v)-[:IS_A*0..]->(parent:OM_Vault {name: $vaultName})
```

### 14.4 Graph Algorithms for Analysis

Centrality and community detection:
```cypher
// Entity importance via PageRank
CALL gds.pageRank.stream('entity_graph')
YIELD nodeId, score
// Community detection for topic clustering
CALL gds.leiden.stream('memory_graph')
YIELD nodeId, communityId
```

### 14.5 Full-Text Search Index

For content search within the graph:
```cypher
CREATE FULLTEXT INDEX om_memory_content IF NOT EXISTS
FOR (m:OM_Memory) ON EACH [m.content]
// Query:
CALL db.index.fulltext.queryNodes('om_memory_content', 'search terms')
YIELD node, score
```

---

## 15) Schema Reference

### Node Labels

| Label | Properties | Unique Constraint | Description |
|-------|-----------|-------------------|-------------|
| `OM_Memory` | id, userId, content, createdAt, updatedAt, state, vault, layer, vector, circuit, axisCategory, source, was, projectedAt | id | Core memory node |
| `OM_Entity` | userId, name | (userId, name) | Referenced entities |
| `OM_Vault` | name | name | AXIS vault dimension |
| `OM_Layer` | name | name | AXIS layer dimension |
| `OM_Vector` | name | name | AXIS vector dimension |
| `OM_Circuit` | level | level | AXIS circuit level |
| `OM_Tag` | key | key | Tag keys |
| `OM_Origin` | name | name | Origin sources |
| `OM_Evidence` | name | name | Evidence items |
| `OM_App` | name | name | Application sources |
| `OM_TemporalEvent` | userId, name, eventType, startDate, endDate, description, memoryIds, createdAt, updatedAt | (userId, name) | Biographical timeline events |

### Relationship Types

| Type | Source | Target | Properties | Description |
|------|--------|--------|------------|-------------|
| `OM_ABOUT` | OM_Memory | OM_Entity | — | Memory references entity |
| `OM_IN_VAULT` | OM_Memory | OM_Vault | — | Memory in vault |
| `OM_IN_LAYER` | OM_Memory | OM_Layer | — | Memory at layer |
| `OM_HAS_VECTOR` | OM_Memory | OM_Vector | — | Memory has vector |
| `OM_IN_CIRCUIT` | OM_Memory | OM_Circuit | — | Memory at circuit |
| `OM_TAGGED` | OM_Memory | OM_Tag | tagValue | Tagged with value |
| `OM_DERIVED_FROM` | OM_Memory | OM_Origin | — | Origin tracking |
| `OM_HAS_EVIDENCE` | OM_Memory | OM_Evidence | — | Evidence link |
| `OM_WRITTEN_VIA` | OM_Memory | OM_App | — | App source |
| `OM_SIMILAR` | OM_Memory | OM_Memory | userId, score, rank, createdAt, updatedAt | Semantic similarity |
| `OM_CO_MENTIONED` | OM_Entity | OM_Entity | userId, count, memoryIds, createdAt, updatedAt | Entity co-occurrence |
| `OM_COOCCURS` | OM_Tag | OM_Tag | userId, count, pmi, createdAt, updatedAt | Tag co-occurrence |
| `OM_RELATION` | OM_Entity | OM_Entity | userId, type, memoryId, count, createdAt, updatedAt | Typed entity relationship (e.g., "vater_von", "works_at") |
| `OM_TEMPORAL` | OM_Entity | OM_TemporalEvent | — | Entity linked to biographical event |

### Indexes

| Index Name | Type | On | Purpose |
|------------|------|-----|---------|
| om_memory_user_id | Range | OM_Memory.userId | User-scoped queries |
| om_memory_user_created | Composite | OM_Memory.(userId, createdAt) | Time-range queries |
| om_entity_user_id | Range | OM_Entity.userId | User-scoped entity lookups |
| om_similar_user_id | Relationship | OM_SIMILAR.userId | User-scoped similarity |
| om_similar_score | Relationship | OM_SIMILAR.score | Score filtering |
| om_co_mentioned_user_id | Relationship | OM_CO_MENTIONED.userId | User-scoped co-mentions |
| om_cooccurs_user_id | Relationship | OM_COOCCURS.userId | User-scoped tag co-occurrence |
| om_relation_user_id | Relationship | OM_RELATION.userId | User-scoped typed relations |
| om_relation_type | Relationship | OM_RELATION.type | Relationship type filtering |
| om_temporal_event_unique | Constraint | OM_TemporalEvent.(userId, name) | Unique temporal events per user |
| om_temporal_event_date | Range | OM_TemporalEvent.startDate | Date-based timeline queries |

---

## 16) Neo4j Graph Data Science (GDS) Integration

OpenMemory now includes support for Neo4j's Graph Data Science library for advanced graph analytics.

### 16.1 Enabling GDS

GDS is enabled in `docker-compose.yml` via environment variables:

```yaml
neo4j:
  image: neo4j:5-community
  environment:
    - NEO4J_PLUGINS=["graph-data-science"]
    - NEO4J_dbms_security_procedures_unrestricted=gds.*
    - NEO4J_dbms_security_procedures_allowlist=gds.*
```

After starting Docker, verify GDS is available:
```cypher
RETURN gds.version() AS version
```

### 16.2 Available GDS Operations

**File:** `openmemory/api/app/graph/gds_operations.py`

| Operation | Function | Description |
|-----------|----------|-------------|
| **PageRank** | `entity_pagerank()` | Identify most influential entities |
| **Community Detection** | `detect_entity_communities()` | Find clusters of related entities (Louvain) |
| **Memory Communities** | `detect_memory_communities()` | Find clusters of similar memories (WCC) |
| **Node Similarity** | `find_similar_entities()` | Find similar entities (Jaccard) |
| **Betweenness Centrality** | `entity_betweenness()` | Find bridge entities |
| **FastRP Embeddings** | `generate_entity_embeddings()` | Generate graph-based embeddings |

### 16.3 Usage Examples

**PageRank for Entity Influence:**
```python
from app.graph.graph_ops import get_entity_pagerank

# Get most influential entities
rankings = get_entity_pagerank(user_id="grischadallmer", limit=20)
for entity in rankings:
    print(f"{entity['name']}: {entity['pageRankScore']:.4f}")
```

**Community Detection:**
```python
from app.graph.graph_ops import detect_entity_communities

# Find entity clusters
result = detect_entity_communities(user_id="grischadallmer")
print(f"Found {result['stats']['communityCount']} communities")
for community in result['communities'][:5]:
    members = [m['name'] for m in community['members']]
    print(f"  Community {community['communityId']}: {members}")
```

**Finding Similar Entities (Graph-based):**
```python
from app.graph.graph_ops import find_similar_entities_gds

# Find entities structurally similar to "BMG"
similar = find_similar_entities_gds(
    user_id="grischadallmer",
    entity_name="BMG",
    limit=10
)
for pair in similar:
    print(f"{pair['entity1']} ↔ {pair['entity2']}: {pair['similarity']:.3f}")
```

**Graph Embeddings for ML:**
```python
from app.graph.graph_ops import generate_entity_embeddings

# Generate 128-dim FastRP embeddings
embeddings = generate_entity_embeddings(
    user_id="grischadallmer",
    write_to_nodes=True,  # Store on nodes as 'graphEmbedding'
    limit=100
)
```

### 16.4 GDS Graph Projections

GDS algorithms operate on in-memory graph projections:

| Projection | Nodes | Relationships | Use Cases |
|------------|-------|---------------|-----------|
| Entity Graph | OM_Entity | OM_CO_MENTIONED | PageRank, betweenness, communities |
| Memory Graph | OM_Memory | OM_SIMILAR | Memory clustering, similarity |
| Full Graph | All OM_* | All edges | Cross-type analysis |

### 16.5 Configuration

GDS parameters can be configured via `GDSConfig`:

```python
from app.graph.gds_operations import GDSConfig, GDSOperations

config = GDSConfig(
    pagerank_damping_factor=0.85,      # Default: 0.85
    pagerank_max_iterations=20,         # Default: 20
    louvain_max_iterations=10,          # Default: 10
    similarity_top_k=10,                # Default: 10
    similarity_min_similarity=0.5,      # Default: 0.5
    fastrp_embedding_dimension=128,     # Default: 128
)
```

### 16.6 Entity Resolution with GDS

For deduplicating similar entities, use FastRP + cosine similarity:

```cypher
// Generate embeddings
CALL gds.fastRP.stream('entity-graph', {embeddingDimension: 128})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
SET node.graphEmbedding = embedding;

// Find potential duplicates
MATCH (e1:OM_Entity), (e2:OM_Entity)
WHERE e1.userId = $userId AND e2.userId = $userId
  AND id(e1) < id(e2)
  AND e1.graphEmbedding IS NOT NULL
  AND e2.graphEmbedding IS NOT NULL
WITH e1, e2,
     gds.similarity.cosine(e1.graphEmbedding, e2.graphEmbedding) AS sim
WHERE sim > 0.9
RETURN e1.name AS entity1, e2.name AS entity2, sim
ORDER BY sim DESC
```

---

## 17) Quick Reference

### Check Available Features

```python
from app.graph.graph_ops import (
    is_graph_enabled,        # Neo4j metadata projection
    is_mem0_graph_enabled,   # Mem0 LLM entity extraction
    is_similarity_enabled,   # Qdrant→Neo4j similarity edges
    is_gds_available,        # GDS advanced analytics
)

print(f"Graph enabled: {is_graph_enabled()}")
print(f"Mem0 Graph: {is_mem0_graph_enabled()}")
print(f"Similarity: {is_similarity_enabled()}")
print(f"GDS: {is_gds_available()}")
```

### Get Graph Statistics

```python
from app.graph.graph_ops import get_graph_statistics

stats = get_graph_statistics(user_id="grischadallmer")
# Returns: memoryCount, entityCount, coMentionEdges, similarityEdges, totalEdges
```

---

## 18) Known Limitation: Entity Co-Mention Gap

### The Problem

The `OM_CO_MENTIONED` edges are **structurally correct but empty** in practice.

**Root cause:** Each memory has only one entity reference (`metadata.re`). The co-mention query:

```cypher
MATCH (m:OM_Memory)-[:OM_ABOUT]->(e1:OM_Entity)
MATCH (m)-[:OM_ABOUT]->(e2:OM_Entity)
WHERE e1.name < e2.name
```

Never finds pairs because there's only ever one `OM_ABOUT` edge per memory.

**Symptoms:**
- `graph_entity_network(entity_name="BMG")` returns empty results
- `graph_path_between_entities(entity_a="BMG", entity_b="Matthias")` finds only structural paths (shared layer/vault/app) — not semantic relationships
- Backfill creates 0 edges

### The Solution: Leverage Mem0 Graph Memory

Mem0's Graph Memory already extracts multiple entities per memory with typed relationships. The solution is to **bridge Mem0's `__Entity__` graph to OpenMemory's `OM_*` graph**.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Memory Creation                              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ mem0.Memory.add(text, metadata)                                      │
│   1. Stores embedding in Qdrant                                      │
│   2. If graph_store configured:                                      │
│      - LLM extracts entities + relationships from text               │
│      - Creates :__Entity__ nodes with typed relationship edges       │
│      - Example: (grischa)-[:vater_von]->(charlie)                    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OpenMemory Projector (NEW: Bridge Layer)                             │
│   1. Query Mem0's :__Entity__ graph for this memory's entities       │
│   2. Create OM_ABOUT edges: (OM_Memory)-[:OM_ABOUT]->(OM_Entity)     │
│      for EACH extracted entity (not just metadata.re)                │
│   3. Update OM_CO_MENTIONED edges between entity pairs               │
│   4. Optionally: Create OM_RELATION edges with type from Mem0        │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Multi-Entity Bridging (Core Fix)

**Goal:** After each memory is created, query Mem0's graph for extracted entities and create corresponding `OM_ABOUT` edges.

**New file:** `openmemory/api/app/graph/entity_bridge.py`

```python
def bridge_mem0_entities_to_om(memory_id: str, user_id: str, memory_content: str) -> int:
    """
    Query Mem0's :__Entity__ graph for entities mentioned in this memory's content,
    then create OM_ABOUT edges for each entity.

    Returns: number of entities bridged
    """
    # 1. Find :__Entity__ nodes that have relationships involving this content
    #    (Mem0 stores embeddings on entity nodes for matching)

    # 2. For each entity found, ensure corresponding OM_Entity node exists

    # 3. Create (OM_Memory)-[:OM_ABOUT]->(OM_Entity) edge

    # 4. Trigger update_entity_edges_on_add() to create OM_CO_MENTIONED
```

**Integration point:** Call `bridge_mem0_entities_to_om()` after memory creation in `mcp_server.py:276-282`.

#### Phase 2: Typed Relationship Edges (Enhanced)

**Goal:** Store Mem0's typed relationships (e.g., `vater_von`, `works_at`) as explicit edges.

**New relationship type:** `OM_RELATION`

```cypher
(OM_Entity)-[:OM_RELATION {type: "vater_von", memoryId: "...", createdAt: ...}]->(OM_Entity)
```

**Benefits:**
- Preserves semantic relationship types
- Links back to source memory (provenance)
- Can be queried via MCP tools

#### Phase 3: Backfill for Existing Memories

**New script:** `openmemory/api/app/scripts/backfill_entity_bridge.py`

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_entity_bridge --user-id grischadallmer
```

### Alternative Approaches Considered

| Approach | Pros | Cons |
|----------|------|------|
| **Option A: Multi-Entity Metadata Field** | No LLM needed, simple | Requires schema change, no relationship types |
| **Option B: Bridge Mem0 Graph (Chosen)** | Leverages existing LLM extraction, typed relationships, no metadata change | Requires Mem0 Graph Memory enabled |
| **Option C: Inline LLM Extraction** | Independent of Mem0 | Duplicates LLM calls, increases latency |

### Configuration Requirements

For this solution to work, Mem0 Graph Memory must be enabled:

```bash
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.enable_mem0_graph_memory --base-label --threshold 0.75
```

### Expected Results After Implementation

1. **Entity Network Works:**
   ```python
   graph_entity_network("BMG")
   # Returns: [{entity: "Matthias", count: 5, memory_ids: [...]}, ...]
   ```

2. **Path Finding Semantic:**
   ```python
   graph_path_between_entities("BMG", "Charlie")
   # Returns path through shared memories with relationship types
   ```

3. **Statistics Show Edges:**
   ```cypher
   MATCH ()-[r:OM_CO_MENTIONED {userId:'grischadallmer'}]->()
   RETURN count(r)
   // Previously: 0
   // After: >0 based on multi-entity memories
   ```

### Quality Differentiation in Path Results

A key improvement is distinguishing path quality:

| Path Type | Quality | Example |
|-----------|---------|---------|
| **Direct** | Strong | Both entities in same memory (`OM_ABOUT`) |
| **Via Relationship** | Strong | `Entity → OM_RELATION → Entity` |
| **Via Dimension** | Weak | `Entity ← OM_ABOUT ← Memory → OM_IN_LAYER → Layer ← OM_IN_LAYER ← Memory → OM_ABOUT → Entity` |

The `path_between_entities` tool will be enhanced to report `path_quality` for each result.

---

## 19. Hybrid Retrieval Architecture

### Overview

Hybrid retrieval combines three retrieval sources:
1. **Semantic similarity** (Qdrant vector search)
2. **Graph topology** (Neo4j OM_SIMILAR, OM_CO_MENTIONED edges)
3. **Entity relationships** (multi-entity extraction via Mem0)

The implementation follows a 3-phase architecture:
- **Phase 1**: Graph-Enhanced Reranking (boost factors from graph signals)
- **Phase 2**: RRF Multi-Source Fusion (combine vector and graph results)
- **Phase 3**: Intelligent Query Routing (auto-select optimal strategy)

### New Files Created

| File | Purpose |
|------|---------|
| `app/graph/graph_cache.py` | Batch-fetches graph signals for reranking |
| `app/utils/rrf_fusion.py` | Reciprocal Rank Fusion algorithm |
| `app/utils/query_router.py` | Intelligent query routing |
| `app/scripts/migrate_hybrid_retrieval.py` | Schema migration script |

### Modified Files

| File | Changes |
|------|---------|
| `app/graph/neo4j_client.py` | Added health check with circuit breaker pattern |
| `app/utils/reranking.py` | Extended BoostConfig with graph weights, added `compute_graph_boost()` |
| `app/graph/graph_ops.py` | Added `retrieve_via_similarity_graph()` and `retrieve_via_entity_graph()` |
| `app/mcp_server.py` | Integrated hybrid retrieval into `search_memory` |

### Phase 1: Graph-Enhanced Reranking

New boost signals added to reranking:

| Signal | Weight | Source | Description |
|--------|--------|--------|-------------|
| `entity_centrality` | 0.25 | OM_Entity.pageRank | Important entities boost memory relevance |
| `similarity_cluster` | 0.20 | OM_Memory.similarityClusterSize | Well-connected memories are more reliable |
| `entity_density` | 0.15 | OM_Entity.degree | Entities with many co-mentions are central |
| `tag_pmi_relevance` | 0.10 | OM_COOCCURS.pmi | Tag co-occurrence indicates thematic relevance |

Max graph boost capped at 0.50 to prevent graph signals from dominating.

### Phase 2: RRF Multi-Source Fusion

Reciprocal Rank Fusion combines rankings from different sources:

```
score = α/(k + vector_rank) + (1-α)/(k + graph_rank)
```

Where:
- `k = 60` (smoothing constant)
- `α = 0.6` (default vector preference)
- Missing ranks penalized with `penalty = 100`

### Phase 3: Intelligent Query Routing

Query analysis determines optimal search strategy without LLM calls:

| Entity Count | Relationship Keywords | Route | RRF Alpha |
|--------------|----------------------|-------|-----------|
| 0 | No | VECTOR_ONLY | 1.0 |
| 0 | Yes | HYBRID | 0.6 |
| 1 | No | HYBRID | 0.6 |
| 1 | Yes | GRAPH_PRIMARY | 0.4 |
| 2+ | Any | GRAPH_PRIMARY | 0.4 |

Entity detection uses Neo4j fulltext index (~5-20ms latency).
Keyword detection uses regex patterns (English + German).

### New search_memory Parameters

```python
search_memory(
    query="...",
    # Existing parameters...

    # New hybrid retrieval parameters:
    use_rrf=True,           # Enable RRF fusion (default: True)
    graph_seed_count=5,     # Seed memories for graph traversal (default: 5)
    auto_route=True,        # Enable intelligent routing (default: True)
)
```

### Verbose Output Extension

When `verbose=True`, the response includes:

```json
{
  "results": [...],
  "hybrid_retrieval": {
    "route": "graph",
    "route_confidence": 0.9,
    "detected_entities": [["julia", 0.95], ["bob", 0.88]],
    "relationship_keywords": ["connected to"],
    "rrf_stats": {
      "vector_candidates": 10,
      "graph_candidates": 8,
      "fused_total": 15,
      "in_both_sources": 6,
      "alpha": 0.4
    },
    "analysis_time_ms": 15.2
  }
}
```

### Neo4j Schema Requirements

The hybrid retrieval requires these properties/indexes:

```cypher
-- Fulltext index for entity detection
CREATE FULLTEXT INDEX om_entity_name_fulltext IF NOT EXISTS
FOR (e:OM_Entity) ON EACH [e.name]

-- Properties (added by migration script)
OM_Memory.similarityClusterSize  -- Count of OM_SIMILAR edges
OM_Entity.pageRank               -- PageRank score from GDS
OM_Entity.degree                 -- Count of OM_CO_MENTIONED edges
```

### Migration Script Usage

```bash
# From the openmemory-api container:

# Dry run (see what would be done):
python -m app.scripts.migrate_hybrid_retrieval --dry-run --all-users

# Migrate for specific user:
python -m app.scripts.migrate_hybrid_retrieval --user-id YOUR_USER_ID

# Migrate all users:
python -m app.scripts.migrate_hybrid_retrieval --all-users
```

Migration steps:
1. Creates fulltext index for entity detection
2. Refreshes `similarityClusterSize` on memories
3. Refreshes `degree` on entities
4. Computes PageRank on entities (if GDS available)

**Neo4j Community Edition Note:**

The migration script uses `SHOW INDEXES YIELD name WHERE name = $indexName` syntax which is required for Neo4j Community Edition. The older `SHOW INDEXES WHERE ...` syntax without `YIELD` only works in Enterprise Edition.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OM_ROUTING_ENABLED` | `true` | Enable intelligent routing |
| `OM_ROUTING_MIN_ENTITY_SCORE` | `0.7` | Minimum fulltext score for entity match |
| `OM_ROUTING_FALLBACK_MIN_RESULTS` | `3` | Minimum results before fallback |

### Circuit Breaker Pattern

Neo4j health is checked with a circuit breaker to prevent cascading failures:

- Health check interval: 60 seconds
- Health check timeout: 0.5 seconds
- When unhealthy, graph features gracefully degrade (vector-only mode)

### Graceful Degradation

The system degrades gracefully when Neo4j is unavailable:

| Component | Fallback Behavior |
|-----------|-------------------|
| Entity detection | Returns empty list, route defaults to HYBRID |
| Graph retrieval | Returns empty list, uses vector-only results |
| Graph boost | Returns 0.0, only metadata boosts apply |
| RRF fusion | With empty graph list, effectively vector-only |

### Test Files

| File | Coverage |
|------|----------|
| `tests/test_graph_boost.py` | Graph boost calculation, capping, edge cases |
| `tests/test_query_router.py` | Route determination, keyword detection |
| `tests/test_rrf_fusion.py` | RRF algorithm, alpha weights, statistics |

---

## 20. Entity-Aware Query Expansion

### Overview

Entity-aware query expansion automatically finds transitive connections between entities when intermediate entities are not mentioned in the query. This solves the problem of missing connections for relationship queries like "Wie hängen Paul und Marius zusammen?" (How are Paul and Marius connected?).

**Problem:** Queries for entity connections return no results because intermediate entities (bridge entities) are not in the query.

**Solution:** When 2+ entities are detected and route is GRAPH_PRIMARY, the system:
1. Finds bridge entities via `OM_CO_MENTIONED` network (up to 3 hops)
2. Expands the entity list with bridges
3. Retrieves memories via entity graph
4. Merges results into RRF fusion

### Implementation

**New function:** `find_bridge_entities()` in [graph_ops.py:1152](openmemory/api/app/graph/graph_ops.py#L1152)

```python
def find_bridge_entities(
    user_id: str,
    entity_names: List[str],
    max_bridges: int = 5,
    min_count: int = 2,
    max_hops: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find entities that bridge between query entities via OM_CO_MENTIONED.

    For 2+ entities, finds intermediate entities that connect them
    through the co-mention network (up to max_hops).

    Returns:
        List of bridge entity dicts with:
        - name: Bridge entity name
        - connectionStrength: Sum of co-mention counts
        - connects: List of query entities this bridges
    """
```

**Integration point:** [mcp_server.py:596-665](openmemory/api/app/mcp_server.py#L596) inside `search_memory()`

### How It Works

1. **Query Analysis:** Router detects entities via fulltext index (`om_entity_name`)
2. **Route Determination:** 2+ entities triggers `GRAPH_PRIMARY` route
3. **Bridge Detection:** `find_bridge_entities()` finds entities connected to 2+ query entities via `OM_CO_MENTIONED` paths
4. **Entity Expansion:** Query entities + bridge entities form the expanded list
5. **Graph Retrieval:** `retrieve_via_entity_graph()` finds memories mentioning any expanded entity
6. **RRF Fusion:** Entity graph results merged with vector + similarity results

### Example

**Query:** "Paul Marius Verbindung"

**Process:**
1. Router detects: `["paul", "marius", "Paul Schubenz", "Marius Waldau"]`
2. Route: `GRAPH_PRIMARY` (4 entities detected)
3. Bridge detection finds: `["matthias", "grischa", "bmg", "matthias_coers", "delegiertenrat"]`
4. Expanded entity list: 9 entities total
5. Memory retrieval includes memories mentioning any of these 9 entities

**Verbose output:**
```json
{
  "hybrid_retrieval": {
    "routing": {
      "route": "graph",
      "detected_entities": ["paul", "marius", "Paul Schubenz", "Marius Waldau"]
    },
    "entity_expansion": {
      "detected_entities": ["paul", "marius", "Paul Schubenz", "Marius Waldau"],
      "bridge_entities": ["matthias", "grischa", "bmg", "matthias_coers", "delegiertenrat"],
      "expanded_count": 9
    }
  }
}
```

### Bridge Detection Query

The Cypher query finds entities connected to 2+ query entities:

```cypher
UNWIND $entityNames AS name
MATCH (e:OM_Entity {userId: $userId, name: name})
MATCH (e)-[r:OM_CO_MENTIONED*1..3]-(bridge:OM_Entity {userId: $userId})
WHERE NOT bridge.name IN $entityNames
WITH bridge,
     collect(DISTINCT name) AS connectedEntities,
     sum(reduce(s = 0, rel IN r | s + coalesce(rel.count, 1))) AS totalStrength
WHERE size(connectedEntities) >= 2
RETURN bridge.name AS name,
       totalStrength AS connectionStrength,
       connectedEntities AS connects
ORDER BY size(connectedEntities) DESC, totalStrength DESC
LIMIT 5
```

### Performance

| Operation | Latency |
|-----------|---------|
| Bridge detection | ~20-40ms |
| Entity graph retrieval | ~30-50ms |
| **Total added** | **~50-90ms** |

Only triggers when:
- 2+ entities detected in query
- Route is `GRAPH_PRIMARY`
- `use_rrf=True` (default)

### Requirements

1. **OM_Entity nodes:** Must exist in Neo4j (created by metadata projector or backfill)
2. **OM_CO_MENTIONED edges:** Must be populated (run `backfill_entity_bridge.py` or `backfill_entity_edges.py`)
3. **Fulltext index:** `om_entity_name` for entity detection in queries

### Circuit Breaker Note

The `find_bridge_entities()` function intentionally does **not** check `is_neo4j_healthy()`. This is because:

1. The circuit breaker may have been tripped by unrelated operations
2. The function has its own exception handling
3. We want to give Neo4j a chance to work even if previous ops failed

This ensures entity expansion continues to work even if earlier graph operations (like similarity retrieval) failed and marked Neo4j as unhealthy.

### Backfill Requirements

For entity expansion to work, you need:

1. **Entity bridge backfill** (creates entities + relations):
   ```bash
   docker exec openmemory-openmemory-mcp-1 \
     python -m app.scripts.backfill_entity_bridge --user-id YOUR_USER_ID
   ```

2. **Entity edges backfill** (creates OM_CO_MENTIONED):
   ```bash
   docker exec openmemory-openmemory-mcp-1 \
     python -m app.scripts.backfill_entity_edges --user-id YOUR_USER_ID
   ```

### Verbose Response Fields

When `verbose=True`, `entity_expansion` is added to `hybrid_retrieval`:

| Field | Type | Description |
|-------|------|-------------|
| `detected_entities` | List[str] | Entities found in query |
| `bridge_entities` | List[str] | Intermediate entities connecting query entities |
| `expanded_count` | int | Total entity count (detected + bridges) |

---

## 21. Semantic Entity Normalization

### Overview

Semantic entity normalization extends the basic case-based entity normalization with multi-phase similarity detection. While basic normalization handles simple case variants (`BMG` ↔ `bmg`), semantic normalization catches more complex duplicates:

| Pattern | Example | Detection Phase |
|---------|---------|-----------------|
| Case variants | `BMG` ↔ `bmg` | Basic normalizer |
| Prefix/suffix | `marie` ↔ `marie_schubenz` | Prefix matching |
| String similarity | `matthias` ↔ `mathias` | Levenshtein |
| Domain normalization | `eljuego.community` ↔ `el_juego` | Domain matcher |
| Semantic similarity | `CloudFactory` ↔ `CF GmbH` | Embedding (optional) |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Semantic Entity Normalizer                        │
│                  (app/graph/semantic_entity_normalizer.py)           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ String Similarity│  │  Prefix Matcher  │  │ Domain Normalizer│
│   (rapidfuzz)    │  │   (overlap %)    │  │  (.community etc)│
└──────────────────┘  └──────────────────┘  └──────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Union-Find Clustering                             │
│              (transitive grouping of variants)                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Edge Migrator   │  │  Mem0 Sync       │  │  GDS Refresh     │
│ (OM_ABOUT etc.)  │  │ (__Entity__)     │  │ (PageRank/degree)│
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Implementation Files

| File | Purpose |
|------|---------|
| `app/graph/similarity/__init__.py` | Module exports |
| `app/graph/similarity/string_similarity.py` | Levenshtein distance matching (rapidfuzz) |
| `app/graph/similarity/prefix_matcher.py` | Prefix/suffix relationship detection |
| `app/graph/similarity/domain_normalizer.py` | Domain suffix extraction (.community, .de, etc.) |
| `app/graph/similarity/embedding_similarity.py` | Optional embedding-based similarity |
| `app/graph/semantic_entity_normalizer.py` | Main orchestrator with Union-Find clustering |
| `app/graph/entity_edge_migrator.py` | Transactional edge migration with ACL |
| `app/graph/mem0_entity_sync.py` | Mem0 __Entity__ graph synchronization |
| `app/graph/gds_signal_refresh.py` | PageRank/degree signal updates |
| `app/scripts/backfill_semantic_normalization.py` | CLI backfill script |

### Detection Phases

**Phase 1: String Similarity (Levenshtein)**

Uses rapidfuzz library for fast fuzzy matching:

```python
from app.graph.similarity import find_string_similar_entities

matches = find_string_similar_entities(
    entities=["matthias", "mathias", "Matthias Coers"],
    threshold=0.85  # 85% similarity required
)
# Returns: [StringSimilarityMatch(entity_a="matthias", entity_b="mathias", score=0.92)]
```

**Phase 2: Prefix/Suffix Matching**

Detects entities where one is a prefix/suffix of another:

```python
from app.graph.similarity import find_prefix_matches

matches = find_prefix_matches(
    entities=["marie", "marie_schubenz", "paul"],
    min_prefix_len=4,      # Minimum prefix length
    min_overlap_ratio=0.5  # Minimum overlap percentage
)
# Returns: [PrefixMatch(shorter="marie", longer="marie_schubenz", overlap_ratio=0.71)]
```

**Phase 3: Domain Normalization**

Handles domain-based entity names:

```python
from app.graph.similarity import find_domain_matches

matches = find_domain_matches(
    entities=["eljuego.community", "el_juego", "El Juego"]
)
# Returns: [("eljuego.community", "el_juego", 0.95)]
```

Recognized suffixes: `.community`, `.org`, `.net`, `.com`, `.de`, `.io`, `-community`, `_community`

**Phase 4: Embedding Similarity (Optional)**

For semantically similar but textually different entities:

```python
from app.graph.similarity.embedding_similarity import find_embedding_similar_entities

matches = await find_embedding_similar_entities(
    entities=["CloudFactory", "CF GmbH"],
    embedding_model=openai_client,
    threshold=0.90
)
```

Note: Embedding phase is disabled by default to avoid API costs.

### Confidence Scoring

Each phase contributes to a weighted confidence score:

| Phase | Weight | Description |
|-------|--------|-------------|
| `case_exact` | 1.0 | Exact case variants (basic normalizer) |
| `domain_match` | 0.9 | Domain normalization |
| `embedding` | 0.85 | Embedding similarity |
| `string_similarity` | 0.8 | Levenshtein distance |
| `prefix_match` | 0.7 | Prefix/suffix matching |

Final merge threshold: **0.7** (configurable)

### Union-Find Clustering

Candidates are transitively clustered using Union-Find:

```
If A matches B (score 0.85)
And B matches C (score 0.75)
Then A, B, C form one cluster
```

Canonical entity selection criteria (in order):
1. No domain suffixes preferred
2. Shorter names preferred
3. Fewer special characters preferred
4. Uppercase at start preferred

### MCP Tool: graph_normalize_entities_semantic

```python
# Detect duplicates (dry run)
graph_normalize_entities_semantic(mode="detect", threshold=0.7)
# Returns: {groups: [{canonical: "bmg", variants: ["BMG", "Bmg"], confidence: 0.95}]}

# Preview specific merge
graph_normalize_entities_semantic(
    mode="preview",
    canonical="matthias_coers",
    variants="Matthias,matthias"
)
# Returns: estimated edge migrations

# Execute merge
graph_normalize_entities_semantic(
    mode="execute",
    canonical="matthias_coers",
    variants="Matthias,matthias"
)
# Returns: actual migration stats
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "detect" | Operation mode: "detect", "preview", "execute" |
| `threshold` | float | 0.7 | Minimum confidence for merge |
| `canonical` | str | None | Target entity name (for manual merge) |
| `variants` | str | None | Comma-separated variant names |

### Edge Migration Pattern

The entity normalization implementation follows a **safe merge pattern** designed to prevent data loss:

**Phase 1: Migrate Edges**
All edges are migrated from variant nodes to the canonical node before any deletion occurs:

| Edge Type | Migration Strategy | Implementation |
|-----------|-------------------|----------------|
| `OM_ABOUT` | Re-point from variant to canonical | Avoids duplicates with `NOT EXISTS((m)-[:OM_ABOUT]->(canonical))` |
| `OM_CO_MENTIONED` | Aggregate counts, merge memoryIds (top 5) | `ON MATCH SET newR.count = coalesce(newR.count, 0) + oldCount` |
| `OM_RELATION` | Preserve type, increment count per direction | Separate queries for outgoing and incoming edges |
| `OM_TEMPORAL` | Move to canonical if not duplicate | Links to biographical timeline events |

**Phase 2: Delete Self-Referential Edges**
Remove any `OM_CO_MENTIONED` edges between canonical and variant nodes (artifact of migration).

**Phase 3: Delete Orphan Nodes**
Only delete variant nodes that have **no remaining edges** of any type:

```cypher
WHERE NOT EXISTS((variant)<-[:OM_ABOUT]-())
  AND NOT EXISTS((variant)-[:OM_CO_MENTIONED]-())
  AND NOT EXISTS((variant)-[:OM_RELATION]-())
  AND NOT EXISTS(()-[:OM_RELATION]->(variant))
  AND NOT EXISTS((variant)-[:OM_TEMPORAL]->())
```

**ACL Enforcement:** Only memories in `allowed_memory_ids` are affected (if specified). This allows entity normalization to be scoped to specific permissions.

**Transactional Safety:** Each edge type migration is a separate transaction. If one fails, others can still succeed. The orphan deletion only happens after all edges are migrated.

### Mem0 Graph Sync

Optional synchronization with Mem0's `__Entity__` graph:

1. Find `__Entity__` nodes matching variant names
2. Merge relationships from variants to canonical
3. Use APOC for dynamic relationship types (with fallback)
4. Delete orphaned variant nodes

### GDS Signal Refresh

After merge, graph-derived signals are updated:

- Entity degree (OM_CO_MENTIONED edge count)
- Entity PageRank (if GDS available)
- Memory cluster sizes (similarity neighbors)

### Backfill Script Usage

The semantic normalization backfill script automates the entire pipeline:

```bash
# Dry run (show what would be merged)
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_semantic_normalization \
    --user-id grischadallmer \
    --dry-run

# Execute with custom threshold (more conservative)
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_semantic_normalization \
    --user-id grischadallmer \
    --execute \
    --threshold 0.8

# Execute with lower threshold (more aggressive)
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_semantic_normalization \
    --user-id grischadallmer \
    --execute \
    --threshold 0.65

# Skip optional post-processing steps
docker exec -w /usr/src/openmemory openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_semantic_normalization \
    --user-id grischadallmer \
    --execute \
    --skip-mem0-sync \
    --skip-gds-refresh
```

**Options:**

| Flag | Description |
|------|-------------|
| `--user-id` | Required: User ID to process |
| `--dry-run` | Preview mode (default, mutually exclusive with --execute) |
| `--execute` | Actually perform merges (required if not dry-run) |
| `--threshold` | Minimum confidence (default: 0.7, range: 0.0-1.0) |
| `--skip-mem0-sync` | Don't sync Mem0 __Entity__ graph after merge |
| `--skip-gds-refresh` | Don't refresh PageRank/degree signals after merge |
| `--log-file` | Custom log file path (auto-generated if omitted) |

**Script Workflow:**

1. **Collect all entities** from OM_Entity nodes, metadata.re fields, and OM_ABOUT edges
2. **Run multi-phase detection** (string similarity, prefix matching, domain normalization)
3. **Cluster candidates** using Union-Find algorithm for transitive grouping
4. **Choose canonical forms** based on: no domain suffixes, shorter names, fewer special chars, uppercase preferred
5. **Execute or preview merges** for each cluster
6. **Sync Mem0 graph** (if not skipped) - migrates `:__Entity__` relationships
7. **Refresh GDS signals** (if not skipped and GDS available) - updates PageRank and degree

**Log Output:**

Logs are written to `openmemory/api/logs/semantic_normalization_<timestamp>.log` with detailed progress for each phase.

### Graceful Degradation

All operations fail gracefully when Neo4j is unavailable:

```python
from app.graph.semantic_entity_normalizer import with_neo4j_fallback

@with_neo4j_fallback(fallback_value=[])
async def find_semantic_duplicates(user_id: str) -> List[Dict]:
    # If Neo4j is down, returns [] instead of raising
    ...
```

### Neo4j Index Requirements

For optimal performance, ensure these indexes exist:

```cypher
-- Entity name lookup (case-insensitive)
CREATE INDEX om_entity_name_lower IF NOT EXISTS
FOR (e:OM_Entity) ON (toLower(e.name));

-- User-scoped entity queries
CREATE INDEX om_entity_userid IF NOT EXISTS
FOR (e:OM_Entity) ON (e.userId);
```

### Best Practices

1. **Always dry-run first** to review merge candidates
2. **Backup recommended** before first auto-merge on production data
3. **Start with higher threshold** (0.8) and lower gradually
4. **Review false positives** especially for prefix matches (may merge unrelated entities like "Jo" and "Johannes")
5. **Run periodically** as new memories may introduce new variants

### Example Workflow

```bash
# 1. Detect all duplicates
graph_normalize_entities_semantic(mode="detect")

# 2. Review output - identify false positives

# 3. Manual merge for specific entities
graph_normalize_entities_semantic(
    mode="execute",
    canonical="matthias_coers",
    variants="Matthias,matthias,MATTHIAS"
)

# 4. For remaining safe merges, use auto mode via backfill
docker exec openmemory-openmemory-mcp-1 \
  python -m app.scripts.backfill_semantic_normalization \
    --user-id grischadallmer \
    --execute \
    --threshold 0.85
```

### Actual Results from Implementation

Based on the entity normalization work completed:

**Case-insensitive normalization (14 groups):**

- grischa/Grischa → grischa
- matthias/Matthias → matthias
- bmg/BMG → bmg
- And 11 more groups

**Semantic normalization (35 groups):**

- Matthias Coers → matthias (via prefix matching)
- grischas → grischa (via string similarity)
- Philipp Mattern → philipp_mattern (via case normalization)
- eljuego.community → el_juego (via domain normalization)
- And 31 more groups

**Total impact:**

- 200 unique entities after deduplication (down from ~249)
- 183+ edges migrated successfully
- 49 variant nodes deleted cleanly
- No data loss - all edges preserved through migration

**Technical correctness verified:**

- Edge migration completed before node deletion (safe merge pattern)
- All edge types properly migrated (OM_ABOUT, OM_CO_MENTIONED, OM_RELATION, OM_TEMPORAL)
- Counts properly aggregated on OM_CO_MENTIONED edges
- Orphan cleanup only deleted nodes with no remaining edges

### Difference from Basic Normalization

| Feature | Basic (`graph_normalize_entities`) | Semantic (`graph_normalize_entities_semantic`) |
|---------|-----------------------------------|----------------------------------------------|
| Case matching | ✅ | ✅ |
| Underscore handling | ✅ | ✅ |
| String similarity | ❌ | ✅ |
| Prefix/suffix | ❌ | ✅ |
| Domain normalization | ❌ | ✅ |
| Embedding similarity | ❌ | ✅ (optional) |
| Confidence scores | ❌ | ✅ |
| Transitive clustering | ❌ | ✅ |
| GDS signal refresh | ❌ | ✅ |

The basic normalizer remains useful for quick, deterministic merges. The semantic normalizer is for comprehensive duplicate detection with confidence scoring.
