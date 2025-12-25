# OpenMemory Technical Architecture

A comprehensive technical documentation of the OpenMemory system's hybrid knowledge graph architecture, combining vector similarity search, graph-based knowledge representation, and intelligent retrieval mechanisms.

---

## Executive Summary

OpenMemory implements a **dual-database hybrid architecture** that combines:

- **Qdrant Vector Store** - Semantic embedding storage and similarity search
- **Neo4j Graph Database** - Structured relationship modeling and graph traversal
- **Reciprocal Rank Fusion (RRF)** - Multi-source retrieval fusion
- **Intelligent Query Routing** - Dynamic strategy selection per query

This architecture enables both semantic similarity search and explicit relationship traversal, with graceful degradation when components are unavailable.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OpenMemory Architecture                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌────────────────┐        ┌────────────────┐        ┌──────────────┐  │
│   │   MCP Client   │───────▶│  MCP Server    │───────▶│   FastAPI    │  │
│   │ (Claude, etc.) │◀───────│  (SSE/STDIO)   │◀───────│   Backend    │  │
│   └────────────────┘        └────────────────┘        └──────────────┘  │
│                                                               │          │
│                              ┌────────────────────────────────┤          │
│                              │                                │          │
│                              ▼                                ▼          │
│   ┌──────────────────────────────────┐  ┌───────────────────────────┐   │
│   │       Qdrant Vector Store        │  │     Neo4j Graph Store     │   │
│   │                                  │  │                           │   │
│   │  • Memory embeddings (1536-dim)  │  │  • OM_Memory nodes        │   │
│   │  • Cosine similarity search      │  │  • OM_Entity nodes        │   │
│   │  • Payload metadata storage      │  │  • Relationship edges     │   │
│   │  • Scroll/scroll_next pagination │  │  • GDS analytics          │   │
│   │                                  │  │                           │   │
│   │  Collections:                    │  │  Labels:                  │   │
│   │  • openmemory (memories)         │  │  • OM_Memory, OM_Entity   │   │
│   │  • business_concepts (concepts)  │  │  • OM_Vault, OM_Layer     │   │
│   │                                  │  │  • OM_Concept, etc.       │   │
│   └──────────────────────────────────┘  └───────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Graceful Degradation**: Neo4j is optional; Qdrant is the primary store
2. **User Isolation**: All data scoped by `userId` parameter
3. **Multi-Tenant Ready**: Multiple user namespaces in shared infrastructure
4. **Fail-Safe Operations**: Graph operations return True on failure to not block main flow

---

## 2. Vector Store Layer (Qdrant)

### 2.1 Embedding Pipeline

```
Memory Text ──▶ Tokenization ──▶ Embedding Model ──▶ 1536-dim Vector ──▶ Qdrant
```

**Embedding Models Supported**:
- `text-embedding-3-small` (OpenAI, default)
- `text-embedding-ada-002` (OpenAI, legacy)
- Custom models via configuration

**Vector Dimensions**: 1536 (configurable via `BUSINESS_CONCEPTS_EMBEDDING_DIMS`)

### 2.2 Qdrant Collections

| Collection | Purpose | Isolation |
|------------|---------|-----------|
| `openmemory` | Personal memories (AXIS layer) | `userId: "grischa"` |
| `business_concepts` | Business concept embeddings | `userId: "concepts"` |

### 2.3 Point Structure

Each Qdrant point contains:

```json
{
  "id": "uuid-string",
  "vector": [0.123, -0.456, ...],  // 1536 dimensions
  "payload": {
    "data": "Memory content text",
    "created_at": "2025-12-25T10:00:00Z",
    "updated_at": "2025-12-25T10:00:00Z",
    "user_id": "grischadallmer",
    "vault": "WLT",
    "layer": "cognitive",
    "vector": "say",
    "circuit": 4,
    "re": "BMG",
    "tags": {"important": true, "quarterly": "Q4"},
    "source": "user",
    "source_app": "claude-code"
  }
}
```

### 2.4 Search Operations

**Semantic Search**:
```python
# Cosine similarity with metadata filtering
results = qdrant.search(
    collection="openmemory",
    query_vector=embedding,
    limit=20,
    with_payload=True,
    filter=Filter(
        must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    )
)
```

**Scroll API** for bulk operations:
```python
# Efficient pagination for large datasets
points, next_offset = qdrant.scroll(
    collection_name="openmemory",
    limit=100,
    offset=next_offset,
    with_payload=True
)
```

---

## 3. Graph Store Layer (Neo4j)

### 3.1 Schema Overview

OpenMemory maintains a **dual graph architecture**:

1. **Deterministic Metadata Graph** (`OM_*` prefix) - Projected from Qdrant payload
2. **LLM-Extracted Entity Graph** (`__Entity__`) - Mem0's native graph memory

### 3.2 Node Labels

| Label | Properties | Unique Constraint | Description |
|-------|-----------|-------------------|-------------|
| `OM_Memory` | id, userId, content, createdAt, updatedAt, state, vault, layer, vector, circuit, source | id | Core memory node |
| `OM_Entity` | userId, name | (userId, name) | Referenced entities |
| `OM_Vault` | name | name | Vault dimension (SOV, WLT, SIG, FRC, DIR, FGP, Q) |
| `OM_Layer` | name | name | Layer dimension (somatic, emotional, narrative, cognitive, etc.) |
| `OM_Vector` | name | name | Vector dimension (say, want, do) |
| `OM_Circuit` | level | level | Circuit level (1-8) |
| `OM_Tag` | key | key | Tag keys |
| `OM_Origin` | name | name | Origin sources |
| `OM_Evidence` | name | name | Evidence items |
| `OM_App` | name | name | Application sources |
| `OM_TemporalEvent` | userId, name, eventType, startDate, endDate, description, memoryIds | (userId, name) | Biographical timeline events |

### 3.3 Relationship Types

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
| `OM_RELATION` | OM_Entity | OM_Entity | userId, type, memoryId, count | Typed semantic relationship |
| `OM_TEMPORAL` | OM_Entity | OM_TemporalEvent | — | Entity linked to biographical event |

### 3.4 Statistical Edge Properties

**PMI (Pointwise Mutual Information) on OM_COOCCURS**:
```
PMI = log2(P(a,b) / (P(a) * P(b)))
NPMI = PMI / -log2(P(a,b))  // Normalized to [-1, 1]
```
Higher NPMI indicates statistically significant co-occurrence beyond chance.

**Scoring on OM_SIMILAR**:
- `score`: Cosine similarity from Qdrant (0.0-1.0)
- `rank`: Position in K nearest neighbors (0 = most similar)

**Count tracking on OM_CO_MENTIONED**:
- `count`: Number of memories mentioning both entities
- `memoryIds`: Sample of memory IDs for provenance (up to 5)

### 3.5 Projection Pipeline

Memory creation triggers a synchronous projection to Neo4j:

```
Memory Created in Qdrant
        │
        ▼
┌───────────────────────────────────┐
│  metadata_projector.py            │
│                                   │
│  1. Parse metadata payload        │
│  2. Create OM_Memory node         │
│  3. Create/link dimension nodes   │
│  4. Create entity relationships   │
│  5. Update co-occurrence stats    │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  similarity_projector.py          │
│  (async/batch process)            │
│                                   │
│  1. Query Qdrant for k-NN         │
│  2. Create OM_SIMILAR edges       │
│  3. Bidirectional edge creation   │
└───────────────────────────────────┘
```

---

## 4. Similarity Projection System

### 4.1 Configuration

Environment variables:
```bash
OM_SIMILARITY_K=20              # Number of nearest neighbors per memory
OM_SIMILARITY_THRESHOLD=0.6     # Minimum cosine similarity score
OM_SIMILARITY_MAX_EDGES=30      # Maximum edges per memory
OM_SIMILARITY_BIDIRECTIONAL=true
```

### 4.2 Projection Algorithm

```python
# For each memory in Qdrant:
def project_similarity_edges(memory_id, user_id):
    # 1. Get memory's embedding from Qdrant
    embedding = qdrant.retrieve(memory_id).vector

    # 2. Find k nearest neighbors
    neighbors = qdrant.search(
        query_vector=embedding,
        limit=k_neighbors,
        filter=user_filter
    )

    # 3. Filter by threshold
    similar = [n for n in neighbors if n.score >= threshold]

    # 4. Create OM_SIMILAR edges in Neo4j
    for neighbor in similar[:max_edges]:
        neo4j.merge_edge(
            source=memory_id,
            target=neighbor.id,
            type="OM_SIMILAR",
            properties={
                "score": neighbor.score,
                "rank": rank,
                "userId": user_id
            }
        )
```

### 4.3 Bidirectional Edge Strategy

Creating edges in both directions ensures traversability:

```cypher
// Forward edge (source → target)
MERGE (source)-[r1:OM_SIMILAR]->(target)
SET r1.score = $score, r1.rank = $rank

// Reverse edge (target → source)
MERGE (target)-[r2:OM_SIMILAR]->(source)
SET r2.score = $score, r2.reverseRank = $rank
```

---

## 5. Hybrid Retrieval System

### 5.1 Architecture

Hybrid retrieval combines three retrieval sources with Reciprocal Rank Fusion:

```
                    Query
                      │
                      ▼
           ┌──────────────────┐
           │  Query Router    │
           │  (RouteType)     │
           └──────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Vector   │  │  Graph   │  │ Entity   │
  │ Search   │  │ Traverse │  │ Network  │
  │ (Qdrant) │  │ (Neo4j)  │  │ (Neo4j)  │
  └──────────┘  └──────────┘  └──────────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
           ┌──────────────────┐
           │   RRF Fusion     │
           │   (Rank-based)   │
           └──────────────────┘
                      │
                      ▼
               Fused Results
```

### 5.2 Intelligent Query Routing

The system analyzes queries to determine optimal search strategy **without LLM calls**:

| Entity Count | Relationship Keywords | Route | RRF Alpha |
|--------------|----------------------|-------|-----------|
| 0 | No | VECTOR_ONLY | 1.0 |
| 1 | No | HYBRID | 0.6 |
| 1 | Yes | GRAPH_PRIMARY | 0.4 |
| 2+ | Any | GRAPH_PRIMARY | 0.4 |

**Relationship Keywords** (English + German):
```python
patterns = [
    r"\bconnected to\b", r"\brelated to\b", r"\bbetween\b",
    r"\bverbunden\b", r"\bbeziehung\b", r"\bzwischen\b"
]
```

### 5.3 Reciprocal Rank Fusion (RRF)

Algorithm from Cormack et al. (2009):

```python
RRF_score = α/(k + vector_rank) + (1-α)/(k + graph_rank)

# Where:
k = 60                    # Smoothing constant (from research)
α = 0.6                   # Vector weight (configurable per route)
missing_rank = 100        # Penalty for single-source results
```

**Advantages**:
- No score normalization needed
- Robust to different score distributions
- Rank-based fusion is inherently fair

### 5.4 Graph Cache for Reranking

The `graph_cache.py` module batch-fetches graph signals for reranking:

```python
class GraphSignalCache:
    def fetch_signals_batch(self, memory_ids: List[str], user_id: str):
        # Single Neo4j query for all signals
        signals = {}
        for memory_id in memory_ids:
            signals[memory_id] = {
                "entity_centrality": pagerank_score,
                "similarity_count": len(similar_memories),
                "entity_count": len(entities)
            }
        return signals
```

---

## 6. Entity Management

### 6.1 Multi-Entity Extraction

The `entity_bridge.py` module bridges Mem0's LLM-extracted entities to OpenMemory's graph:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Memory Creation                              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Mem0 Graph Memory (if enabled)                                       │
│   - LLM extracts entities + relationships from text                  │
│   - Creates :__Entity__ nodes with typed relationship edges         │
│   - Example: (grischa)-[:vater_von]->(charlie)                      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Entity Bridge Layer                                                  │
│   1. Query Mem0's :__Entity__ graph for this memory's entities      │
│   2. Create OM_ABOUT edges for EACH extracted entity                │
│   3. Update OM_CO_MENTIONED edges between entity pairs              │
│   4. Create OM_RELATION edges with type from Mem0                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Entity Normalization

The `entity_normalizer.py` handles duplicate entity variants:

**Problem**:
```
grischa (61 memories) vs Grischa (26 memories)
matthias (43) vs Matthias (6) vs matthias_coers (16)
```

**Solution**:
```python
def normalize_entity_name(name: str) -> str:
    """
    1. Lowercase
    2. Spaces → Underscores
    3. Multiple underscores → Single
    4. Trim
    """
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    return normalized.strip('_')
```

**Merge Algorithm**:
1. Migrate OM_ABOUT edges to canonical entity
2. Aggregate OM_CO_MENTIONED edge counts
3. Transfer OM_RELATION edges
4. Delete variant nodes

### 6.3 Semantic Entity Normalization

Extended normalization with multi-phase detection (`semantic_entity_normalizer.py`):

1. **String similarity** (Levenshtein/fuzzy matching)
2. **Prefix/suffix matching** (e.g., "marie" → "marie_schubenz")
3. **Domain normalization** (e.g., "eljuego.community" → "el_juego")

---

## 7. Typed Relationships

### 7.1 Relation Type Categories

```python
RELATION_CATEGORIES = {
    "family": ["eltern_von", "kind_von", "mutter_von", "vater_von",
               "schwester_von", "bruder_von", "verwandt_mit"],
    "social": ["partner_von", "freund_von", "mitbewohner_von", "bekannt_mit"],
    "work": ["arbeitet_bei", "arbeitete_bei", "kollege_von",
             "arbeitspartner_von", "gruendete", "leitet"],
    "location": ["wohnt_in", "wohnte_in", "geboren_in",
                 "aufgewachsen_in", "befindet_sich_in"],
    "creative": ["produzierte", "wirkte_mit_in", "regie_bei",
                 "schrieb", "erschuf"],
    "membership": ["mitglied_von", "aktiv_in", "engagiert_in", "teil_von"],
}
```

### 7.2 Inverse Relations

For bidirectional traversal:
```python
INVERSE_RELATIONS = {
    "eltern_von": "kind_von",
    "kind_von": "eltern_von",
    "partner_von": "partner_von",  # symmetric
    "freund_von": "freund_von",    # symmetric
}
```

---

## 8. Temporal Events

### 8.1 Biographical Timeline

The `temporal_events.py` module enables temporal queries:

```python
class EventType(Enum):
    RESIDENCE = "residence"     # Wohnort
    EDUCATION = "education"     # Ausbildung
    WORK = "work"              # Beruf
    PROJECT = "project"        # Projekt
    RELATIONSHIP = "relationship"
    HEALTH = "health"
    TRAVEL = "travel"
    MILESTONE = "milestone"    # Birth, marriage, etc.
```

### 8.2 Date Parsing

Supported formats:
```python
# "2014" → ("2014", None)
# "2014-2018" → ("2014", "2018")
# "seit 2020" → ("2020", None)
# "bis 2019" → (None, "2019")

def parse_date_from_text(text: str) -> Optional[Tuple[str, str]]:
    range_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', text)
    if range_match:
        return (range_match.group(1), range_match.group(2))
    # ... more patterns
```

---

## 9. Neo4j Graph Data Science (GDS)

### 9.1 Available Operations

| Operation | Function | Description |
|-----------|----------|-------------|
| **PageRank** | `entity_pagerank()` | Identify most influential entities |
| **Community Detection** | `detect_entity_communities()` | Find clusters (Louvain) |
| **Memory Communities** | `detect_memory_communities()` | Find similar memory clusters (WCC) |
| **Node Similarity** | `find_similar_entities()` | Find similar entities (Jaccard) |
| **Betweenness Centrality** | `entity_betweenness()` | Find bridge entities |
| **FastRP Embeddings** | `generate_entity_embeddings()` | Generate graph-based embeddings |

### 9.2 Graph Projections

| Projection | Nodes | Relationships | Use Cases |
|------------|-------|---------------|-----------|
| Entity Graph | OM_Entity | OM_CO_MENTIONED | PageRank, betweenness, communities |
| Memory Graph | OM_Memory | OM_SIMILAR | Memory clustering, similarity |
| Full Graph | All OM_* | All edges | Cross-type analysis |

### 9.3 Entity Resolution with GDS

```cypher
// Generate FastRP embeddings
CALL gds.fastRP.stream('entity-graph', {embeddingDimension: 128})
YIELD nodeId, embedding

// Find potential duplicates via cosine similarity
MATCH (e1:OM_Entity), (e2:OM_Entity)
WHERE id(e1) < id(e2)
WITH e1, e2, gds.similarity.cosine(e1.embedding, e2.embedding) AS sim
WHERE sim > 0.9
RETURN e1.name, e2.name, sim
```

---

## 10. Business Concepts Layer

### 10.1 Architecture

A separate knowledge layer that extracts structured concepts from memories:

```
┌─────────────────────────────┬───────────────────────────────────┐
│     Memory Layer            │      Business Concepts Layer       │
│                             │                                    │
│  • Raw memories             │  • Synthesized concepts            │
│  • Personal knowledge       │  • Business patterns               │
│  • user_id: "grischa"       │  • user_id: "concepts"             │
│  • Qdrant: openmemory       │  • Qdrant: business_concepts       │
└─────────────────────────────┴───────────────────────────────────┘
                              ↓
                    Neo4j Shared Graph Database
                    (OM_Concept, OM_BizEntity nodes)
```

### 10.2 Concept Types

| Type | Description | Example |
|------|-------------|---------|
| `causal` | Cause-effect relationships | "Community engagement drives retention" |
| `pattern` | Recurring observations | "Enterprise clients need 3+ touchpoints" |
| `comparison` | Relative assessments | "Product A outperforms B in X" |
| `trend` | Directional changes | "Market shifting toward subscription" |
| `contradiction` | Conflicting observations | "Price sensitivity vs premium positioning" |
| `hypothesis` | Untested theories | "AI could automate X process" |
| `fact` | Verified information | "Q3 revenue was 500k" |

### 10.3 Concept Schema

**New Node Labels**:
- `OM_Concept` - Business concept entries
- `OM_ConceptVersion` - Version history
- `OM_Contradiction` - Detected contradictions
- `OM_BizEntity` - Business entities (companies, people, products)

**New Relationships**:
```
SUPPORTS    - Memory → Concept (evidence)
REFINES     - Concept → Concept (narrowing)
SUPERSEDES  - Concept → Concept (replacement)
SYNTHESIZES - Concept → [Concepts] (combination)
CONTRADICTS - Concept → Contradiction
ENABLES     - Concept → Concept (causation)
REQUIRES    - Concept → Concept (dependency)
```

### 10.4 Concept Vector Store

Separate Qdrant collection for semantic concept search:

```python
class ConceptVectorStore:
    collection: str = "business_concepts"
    embedding_model: str = "text-embedding-3-small"
    dimensions: int = 1536

    def embed_concept(self, concept):
        # Embed: name + type + summary
        text = f"{concept.name} ({concept.type}): {concept.summary}"
        return self.embedder.embed(text)
```

### 10.5 Contradiction Detection

The `convergence_detector.py` implements semantic contradiction detection:

```python
def find_contradictions(concept_id, user_id, min_severity=0.5):
    # 1. Get concept's embedding from Qdrant
    # 2. Find semantically similar concepts
    # 3. Detect opposing semantic signals
    # 4. Score severity based on evidence strength
    return contradictions
```

---

## 11. MCP Integration

### 11.1 Endpoint Separation

```
Port 8765 - OpenMemory MCP Server
├── /mcp/claude/sse/{user_id}        → Memory Tools (~25 tools)
├── /concepts/claude/sse/{user_id}   → Business Concept Tools (~10 tools)
└── /axis/claude/sse/{user_id}       → AXIS Guidance Tools
```

### 11.2 Tool Categories

**Memory Operations**:
- `add_memories`, `search_memory`, `list_memories`, `update_memory`, `delete_memories`

**Graph Operations**:
- `graph_related_memories`, `graph_subgraph`, `graph_aggregate`
- `graph_similar_memories`, `graph_entity_network`, `graph_entity_relations`
- `graph_normalize_entities`, `graph_path_between_entities`
- `graph_biography_timeline`

**Business Concept Operations**:
- `extract_business_concepts`, `list_business_concepts`, `search_business_concepts`
- `find_similar_business_concepts`, `find_concept_contradictions`
- `analyze_concept_convergence`, `get_concept_network`

---

## 12. Data Flow Diagrams

### 12.1 Memory Creation Flow

```
User Input
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  MCP Tool: add_memories                                        │
│                                                                │
│  1. Parse metadata (vault, layer, vector, circuit, tags, etc.) │
│  2. Generate embedding via configured embedder                  │
│  3. Store in Qdrant with payload                               │
└───────────────────────────────────────────────────────────────┘
    │
    ├──────────────────────────┐
    ▼                          ▼
┌──────────────────┐    ┌──────────────────────────────┐
│  Qdrant          │    │  Neo4j Projection            │
│                  │    │                              │
│  • Vector stored │    │  • OM_Memory node created    │
│  • Payload saved │    │  • Dimension nodes linked    │
│  • ID returned   │    │  • Entity edges created      │
└──────────────────┘    │  • Co-occurrence updated     │
                        └──────────────────────────────┘
                               │
                               ▼ (async)
                        ┌──────────────────────────────┐
                        │  Similarity Projection       │
                        │                              │
                        │  • Query Qdrant for k-NN     │
                        │  • Create OM_SIMILAR edges   │
                        └──────────────────────────────┘
```

### 12.2 Hybrid Search Flow

```
Search Query
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  Query Router                                                  │
│                                                                │
│  • Detect entities via Neo4j fulltext                          │
│  • Check for relationship keywords                             │
│  • Select route: VECTOR_ONLY | HYBRID | GRAPH_PRIMARY         │
└───────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────┬─────────────────────┐
    ▼                     ▼                     ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Vector Search│   │ Graph Search │   │ Entity Network   │
│ (Qdrant)     │   │ (Neo4j)      │   │ (Neo4j)          │
│              │   │              │   │                  │
│ k=20 results │   │ OM_SIMILAR   │   │ OM_CO_MENTIONED  │
└──────────────┘   │ traversal    │   │ traversal        │
    │              └──────────────┘   └──────────────────┘
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          ▼
              ┌──────────────────────────────┐
              │  RRF Fusion                  │
              │                              │
              │  score = α/(k+r_v) + (1-α)   │
              │         /(k+r_g)             │
              │                              │
              │  α = 0.6 (HYBRID)            │
              │  α = 0.4 (GRAPH_PRIMARY)     │
              └──────────────────────────────┘
                          │
                          ▼
              ┌──────────────────────────────┐
              │  Graph Signal Reranking      │
              │                              │
              │  • PageRank centrality boost │
              │  • Similarity edge count     │
              │  • Entity count boost        │
              └──────────────────────────────┘
                          │
                          ▼
                   Final Results
```

---

## 13. Configuration Reference

### 13.1 Environment Variables

**Qdrant**:
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=openmemory
```

**Neo4j**:
```bash
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
```

**Similarity Projection**:
```bash
OM_SIMILARITY_K=20
OM_SIMILARITY_THRESHOLD=0.6
OM_SIMILARITY_MAX_EDGES=30
OM_SIMILARITY_BIDIRECTIONAL=true
```

**Hybrid Retrieval**:
```bash
OM_RRF_ENABLED=true
OM_RRF_ALPHA=0.6
OM_RRF_K=60
OM_ROUTING_ENABLED=true
```

**Business Concepts**:
```bash
BUSINESS_CONCEPTS_ENABLED=true
BUSINESS_CONCEPTS_EMBEDDING_ENABLED=true
BUSINESS_CONCEPTS_EMBEDDING_MODEL=text-embedding-3-small
BUSINESS_CONCEPTS_COLLECTION=business_concepts
BUSINESS_CONCEPTS_SIMILARITY_THRESHOLD=0.75
```

### 13.2 Feature Flags

```python
def is_graph_enabled() -> bool:
    """Neo4j metadata projection"""

def is_mem0_graph_enabled() -> bool:
    """Mem0 LLM entity extraction"""

def is_similarity_enabled() -> bool:
    """Qdrant→Neo4j similarity edges"""

def is_gds_available() -> bool:
    """GDS advanced analytics"""
```

---

## 14. Graceful Degradation

All graph operations fail gracefully:

1. **Neo4j unavailable**: Operations return empty results, don't block main flow
2. **Projection failure**: Logs warning, returns True to not fail parent operation
3. **Query failure**: Returns empty list/dict, caller handles gracefully

This ensures the system works with or without Neo4j, with Qdrant as the primary store.

```python
def project_memory_to_graph(...) -> bool:
    projector = _get_projector()

    if projector is None:
        # Neo4j not configured, silently succeed
        return True

    try:
        return projector.upsert_memory(memory_metadata)
    except Exception as e:
        logger.warning(f"Error projecting memory: {e}")
        return False  # Don't fail the main operation
```

---

## 15. Index Reference

### Neo4j Indexes

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
| om_temporal_event_date | Range | OM_TemporalEvent.startDate | Date-based timeline queries |

---

## 16. File Structure

```
openmemory/api/app/
├── graph/
│   ├── neo4j_client.py           # Connection management
│   ├── metadata_projector.py     # Core metadata → Neo4j projection
│   ├── similarity_projector.py   # Qdrant similarity → Neo4j edges
│   ├── graph_ops.py              # High-level graph operations
│   ├── graph_cache.py            # Batch signal fetching for reranking
│   ├── gds_operations.py         # Neo4j GDS algorithms
│   ├── entity_bridge.py          # Mem0 entity → OpenMemory bridging
│   ├── entity_normalizer.py      # Case-based entity normalization
│   ├── semantic_entity_normalizer.py  # Semantic entity deduplication
│   ├── relation_types.py         # Typed relationship definitions
│   ├── temporal_events.py        # Biographical timeline events
│   ├── concept_ops.py            # Business concept CRUD
│   ├── concept_projector.py      # Concept graph projection
│   ├── concept_vector_store.py   # Concept embeddings in Qdrant
│   └── convergence_detector.py   # Contradiction detection
├── utils/
│   ├── rrf_fusion.py             # Reciprocal Rank Fusion
│   ├── query_router.py           # Intelligent query routing
│   └── concept_extractor.py      # LLM concept extraction
├── mcp_server.py                 # MCP tool definitions
└── config.py                     # Feature flags + configuration
```

---

## 17. Version Information

- **Qdrant**: 1.7+ (collection API)
- **Neo4j**: 5.x Community (with GDS plugin)
- **Python**: 3.10+
- **Key Dependencies**: `mem0ai[graph]`, `qdrant-client`, `neo4j`, `langchain-neo4j`

---

*This documentation reflects the system architecture as of December 2025.*
