# AXIS Graph Protocol — Neo4j Operations

## Architecture

OpenMemory uses a **dual-graph architecture**:

1. **Deterministic `OM_*` Graph** — Metadata projection (vault/layer/tags → Neo4j)
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
- search_memory falls back to pure vector search
- No errors, just reduced functionality

---

## Decision Tree: Which Tool When?

```
Query arrives
    │
    ├─ Semantic search → search_memory (Hybrid default)
    │
    ├─ "Wer ist mit X verbunden?" → graph_entity_network
    │
    ├─ "Welche Beziehung hat X zu Y?" → graph_entity_relations OR graph_path_between_entities
    │
    ├─ "Ähnliche Memories" → graph_similar_memories (after initial search)
    │
    ├─ "Pattern-Verteilung" → graph_aggregate
    │
    ├─ "Tags wie X" → graph_related_tags
    │
    └─ "Tags die zusammen vorkommen" → graph_tag_cooccurrence
```

---

## Hybrid Retrieval (search_memory)

New parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_rrf` | true | RRF Multi-Source Fusion |
| `graph_seed_count` | 5 | Seeds for graph traversal |
| `auto_route` | true | Intelligent query routing |

### Query Routing

| Entities Detected | Route | Description |
|-------------------|-------|-------------|
| 0 | VECTOR_ONLY | Pure semantic search |
| 1 | HYBRID | Balanced vector + graph |
| 2+ | GRAPH_PRIMARY | Graph-preferred |

### Verbose Mode

With `verbose=True`, search_memory returns:
- `hybrid_retrieval.detected_entities`: Detected entities with scores
- `hybrid_retrieval.route`: Chosen strategy
- `hybrid_retrieval.rrf_stats`: Fusion statistics (vector_candidates, graph_candidates, etc.)

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
# Returns: [{id, content, vault, layer, similarity_score, rank}]
```

### graph_entity_network
Entity co-mention network via OM_CO_MENTIONED.

```python
graph_entity_network(
    entity_name="BMG",     # Required
    min_count=1,           # Default: 1
    limit=20               # Max connections (default: 20)
)
# Returns: [{entity, count, memory_ids}]
```

### graph_entity_relations
Typed semantic relations via OM_RELATION.

```python
graph_entity_relations(
    entity_name="Grischa", # Required
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
# Returns: [{tag, count, pmi, npmi}]
```

### graph_biography_timeline
Biographical events from OM_TemporalEvent.

```python
graph_biography_timeline(
    entity_name="Grischa", # Optional — if empty, all events
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
    entity_a="BMG",
    entity_b="Matthias",
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
# Returns: [{tag1, tag2, count, sample_memory_ids}]
```

---

## Workflow Examples

### 1. Explore Entity Network

```python
# Who is connected to BMG?
graph_entity_network(entity_name="BMG")

# Typed relationships
graph_entity_relations(entity_name="BMG", category="work")

# Path to specific person
graph_path_between_entities(entity_a="BMG", entity_b="Charlie")
```

### 2. Memory Expansion

```python
# Start with search
results = search_memory(query="Kritik-Pattern")

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
graph_aggregate(group_by="vault")
graph_aggregate(group_by="entity", limit=30)
```

---

## Auto-Triggers

| Pattern in Query | Auto-Action |
|------------------|-------------|
| New entity stored | `graph_path_between_entities` |
| "Fingerprint" | `graph_aggregate` by vault+layer |
| Pattern language | `graph_tag_cooccurrence` |
| "Netzwerk" / "Verbindungen" | `graph_entity_network` |
| "Beziehung" / "Relation" | `graph_entity_relations` |
| Multiple entity names | Route to GRAPH_PRIMARY |
