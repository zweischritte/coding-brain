# Memory Graph Protocol — Neo4j Operations

## Architecture

OpenMemory uses a **dual-graph architecture**:

1. **Deterministic `OM_*` Graph** — Metadata projection (category/scope/tags/artifacts → Neo4j)
2. **LLM-Extracted `:__Entity__` Graph** — Mem0 Graph Memory (semantic relations)

### Enhanced Semantic Edges

| Edge Type | Connects | Source |
|-----------|----------|--------|
| `OM_SIMILAR` | Memory ↔ Memory | Qdrant Embeddings (pre-computed) |
| `OM_CO_MENTIONED` | Entity ↔ Entity | Shared memories |
| `OM_COOCCURS` | Tag ↔ Tag | Co-occurrence with PMI |
| `OM_RELATION` | Entity ↔ Entity | LLM-extracted types |

### Graceful Degradation

When Neo4j is unavailable:
- All graph_* tools return `{"graph_enabled": false}`
- search_memory continues with vector-only results
- Responses do not include `graph_enabled`, but may omit graph enrichment

---

## Decision Tree: Which Tool When?

```
Query arrives
    │
    ├─ Semantic search → search_memory (Hybrid default)
    │
    ├─ "Who is connected to X?" → graph_entity_network
    │
    ├─ "How is X related to Y?" → graph_entity_relations OR graph_path_between_entities
    │
    ├─ "Similar memories" → graph_similar_memories (after initial search)
    │
    ├─ "Pattern distribution" → graph_aggregate
    │
    ├─ "Tags related to X" → graph_related_tags
    │
    └─ "Tags that co-occur" → graph_tag_cooccurrence
```

---

## Hybrid Retrieval (search_memory)

Parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_rrf` | true | RRF multi-source fusion |
| `graph_seed_count` | 5 | Seeds for graph traversal |
| `auto_route` | true | Internal query routing; routing metadata appears in verbose responses |

### Query Routing (Internal)

| Entities Detected | Relationship Keywords | Route | Description |
|-------------------|-----------------------|-------|-------------|
| 0 | No | VECTOR_ONLY | Pure semantic search |
| 0 | Yes | HYBRID | Relationship terms without entities |
| 1 | No | HYBRID | Balanced vector + graph |
| 1 | Yes | GRAPH_PRIMARY | Graph-preferred |
| 2+ | Any | GRAPH_PRIMARY | Graph-preferred |

Notes:
- Entity detection uses the Neo4j fulltext index `om_entity_name`.
- If entity detection is unavailable, routing falls back to relationship keyword signals only.

### Verbose Mode

With `verbose=True`, search_memory returns additional debug fields:
- `query`
- `context_applied`
- `filters_applied`
- `total_candidates`
- `hybrid_retrieval` (when `use_rrf` or `auto_route`): routing, RRF stats, graph boost, entity expansion (when available)

If graph is available, search_memory may also include:
- `meta_relations` (deterministic OM_* relations)
- `relations` (Mem0 LLM-extracted relations)
These graph enrichment fields can appear in non-verbose responses too.

---

## Graph Tools — Complete Reference

### graph_similar_memories
Pre-computed similarity via OM_SIMILAR edges.

```python
graph_similar_memories(
    memory_id="uuid",      # Required
    min_score=0.0,         # Default 0.0, recommended: 0.7 for quality
    limit=10               # Max results (default: 10)
)
# Returns: [{id, content, category, scope, artifactType, artifactRef, source, createdAt, similarityScore, rank}]
```

### graph_entity_network
Entity co-mention network via OM_CO_MENTIONED.

```python
graph_entity_network(
    entity_name="Platform",     # Required
    min_count=1,           # Default: 1
    limit=20               # Max connections (default: 20)
)
# Returns: {entity, connections: [{entity, count, memoryIds}], total, graph_enabled}
```

### graph_entity_relations
Typed semantic relations via OM_RELATION.

```python
graph_entity_relations(
    entity_name="Platform", # Required
    relation_types=None,   # Filter: "schwester_von,bruder_von"
    category="family",     # family|social|work|location|creative|membership|travel
    direction="both",      # outgoing|incoming|both (default: both)
    limit=50               # default: 50
)
# Returns: [{target, type, direction, count, memory_id}]
```

**Categories:**
- **family:** eltern_von, kind_von, schwester_von, bruder_von, etc.
- **social:** partner_von, freund_von, bekannt_mit
- **work:** arbeitet_bei, kollege_von, gruendete, leitet
- **location:** wohnt_in, geboren_in, aufgewachsen_in
- **creative:** produzierte, schrieb, erschuf
- **membership:** mitglied_von, aktiv_in
- **travel:** plant_besuch, reist_nach, war_in

### graph_related_tags
Tag co-occurrence with PMI scoring via OM_COOCCURS.

```python
graph_related_tags(
    tag_key="trigger",     # Required
    min_count=1,           # Default: 1
    limit=20               # default: 20
)
# Returns: [{tag, count, pmi}]
```

### graph_biography_timeline
Biographical events from OM_TemporalEvent.

```python
graph_biography_timeline(
    entity_name="Alex",    # Optional — if empty, all events
    event_types="project", # residence|education|work|project|relationship|health|travel|milestone
    start_year=2014,       # Optional
    end_year=2018,         # Optional
    limit=50               # default: 50
)
# Returns: [{event_type, start_date, end_date, description, memory_ids}]
```

### graph_normalize_entities
Entity deduplication.

```python
# Dry run (preview only)
graph_normalize_entities(dry_run=True)

# Auto-merge all
graph_normalize_entities(auto=True, dry_run=False)

# Manual merge
graph_normalize_entities(canonical="matthias_coers", variants="Matthias,matthias", dry_run=False)
```

### graph_path_between_entities
Semantic path between entities.

```python
graph_path_between_entities(
    entity_a="Platform",
    entity_b="Security",
    max_hops=6             # default: 6
)
# Returns: Path with relationship types
# Uses OM_RELATION and OM_CO_MENTIONED for semantic paths
```

### graph_tag_cooccurrence
Tag pairs that frequently co-occur.

```python
graph_tag_cooccurrence(
    limit=20,              # default: 20
    min_count=2,           # default: 2
    sample_size=3          # Example memory IDs per pair
)
# Returns: [{tag1, tag2, count, exampleMemoryIds}]
```

---

## Workflow Examples

### 1. Explore Entity Network

```python
# Who is connected to Platform?
graph_entity_network(entity_name="Platform")

# Typed relationships
graph_entity_relations(entity_name="Platform", category="work")

# Path to specific person
graph_path_between_entities(entity_a="Platform", entity_b="Security")
```

### 2. Memory Expansion

```python
# Start with search
results = search_memory(query="review feedback")

# Expand high-signal result
graph_similar_memories(memory_id=results[0]["id"], min_score=0.75)

# Full subgraph
graph_subgraph(memory_id=results[0]["id"], depth=2)
```

### 3. Pattern Topology

```python
# Tag clusters
graph_related_tags(tag_key="trigger")

# Which tags co-occur?
graph_tag_cooccurrence(min_count=3)

# Distribution analysis
graph_aggregate(group_by="category")
graph_aggregate(group_by="entity", limit=30)
```

---

## Routing Signals (search_memory)

Routing only affects `search_memory` and does not auto-invoke graph tools.

| Pattern in Query | Expected Route |
|------------------|----------------|
| Relationship keywords (e.g., "connected to", "related to", "between", "path", "network", "relationship", "verbunden mit", "beziehung", "zwischen", "who knows", "wer kennt") | GRAPH_PRIMARY if entities are detected, otherwise HYBRID |
| Multiple entity names | GRAPH_PRIMARY |
| No entities + no relationship keywords | VECTOR_ONLY |
