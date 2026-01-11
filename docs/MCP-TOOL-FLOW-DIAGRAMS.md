# Coding Brain MCP Tool Flow Diagrams

Dieses Dokument beschreibt detailliert, was bei jedem MCP Tool Call in Coding Brain passiert, inklusive aller internen LLM-Calls, Datenbank-Interaktionen und Store-Zugriffe.

---

## Architektur-Übersicht

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MCP CLIENT                                      │
│                    (Claude Code, Cursor, IDE Extension)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ SSE/POST
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MCP SERVER (FastMCP)                              │
│                         /mcp/<client>/sse/<user_id>                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SessionAwareSseTransport                         │    │
│  │              (Session Binding, JWT Validation, DPoP)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         TOOL HANDLERS                                │    │
│  │  add_memories, search_memory, update_memory, delete_memories,       │    │
│  │  list_memories, graph_*, index_codebase, search_code_hybrid, ...    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   Mem0 Client     │    │   Graph Ops       │    │   Code Intel      │
│   (Memory CRUD)   │    │   (Neo4j)         │    │   (Indexing)      │
└───────────────────┘    └───────────────────┘    └───────────────────┘
         │                        │                        │
    ┌────┴────┐              ┌────┴────┐              ┌────┴────┐
    │         │              │         │              │         │
    ▼         ▼              ▼         │              ▼         ▼
┌───────┐ ┌───────┐    ┌───────────┐   │       ┌───────────┐ ┌───────────┐
│Qdrant │ │Ollama │    │   Neo4j   │   │       │OpenSearch │ │   Neo4j   │
│Vector │ │LLM/   │    │  (Graph)  │   │       │  (Index)  │ │ (CODE_*)  │
│ Store │ │Embed  │    └───────────┘   │       └───────────┘ └───────────┘
└───────┘ └───────┘                    │
                                       ▼
                              ┌───────────────────┐
                              │   PostgreSQL      │
                              │   (Metadata)      │
                              └───────────────────┘
```

---

## Store-Übersicht

| Store | Zweck | Daten |
|-------|-------|-------|
| **PostgreSQL** | Persistente Metadaten | Memory, User, App, Config, AccessLog, StatusHistory |
| **Qdrant** | Vektor-Embeddings (Mem0) | Memory-Embeddings (BGE-M3, 1024 dims) |
| **Neo4j (OM_*)** | Memory-Graph | Memory, Entity, Tag, Category, Scope + OM_ABOUT, OM_SIMILAR, etc. |
| **Neo4j (CODE_*)** | Code-Graph | CODE_FILE, CODE_SYMBOL + CALLS, IMPORTS, CONTAINS |
| **OpenSearch** | Code-Index (BM25 + kNN) | Symbol-Dokumente mit Embeddings |
| **Ollama** | LLM + Embedder | Fact Extraction (qwen3:8b), Embeddings (bge-m3) |
| **Valkey** | Session Cache | MCP Session Binding, DPoP Replay Cache |

---

## 1. add_memories

### Sequenzdiagramm

```
Client                MCP Server           Mem0 Client      Ollama        Qdrant       PostgreSQL      Neo4j
  │                       │                    │              │             │              │             │
  │ add_memories(text,    │                    │              │             │              │             │
  │   category, entity,   │                    │              │             │              │             │
  │   access_entity,      │                    │              │             │              │             │
  │   infer=false)        │                    │              │             │              │             │
  │──────────────────────>│                    │              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 1. validate_metadata              │             │              │             │
  │                       │ (build_structured_memory)         │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 2. resolve_access_entity          │             │              │             │
  │                       │ (JWT grants check)│               │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 3. create_job(status="queued")    │             │              │             │
  │                       │ (if async_mode=true)              │             │              │             │
  │                       │                    │              │             │              │             │
  │<──────────────────────│ return job_id      │              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │====== ASYNC BACKGROUND ======     │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 4. memory_client   │              │             │              │             │
  │                       │    .add(text,      │              │             │              │             │
  │                       │    metadata, infer)│              │             │              │             │
  │                       │───────────────────>│              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │                    │ 5a. IF infer=true:         │              │             │
  │                       │                    │ LLM fact extraction        │              │             │
  │                       │                    │─────────────>│             │              │             │
  │                       │                    │              │             │              │             │
  │                       │                    │<─────────────│             │              │             │
  │                       │                    │ entities,    │             │              │             │
  │                       │                    │ relations    │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │                    │ 5b. Generate │             │              │             │
  │                       │                    │ embedding    │             │              │             │
  │                       │                    │─────────────>│             │              │             │
  │                       │                    │<─────────────│             │              │             │
  │                       │                    │ vector[1024] │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │                    │ 6. Store embedding in Qdrant             │             │
  │                       │                    │──────────────────────────────>│            │             │
  │                       │                    │              │             │              │             │
  │                       │<───────────────────│              │             │              │             │
  │                       │ {id, memory, event}│              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 7. Save Memory metadata to PostgreSQL           │              │             │
  │                       │─────────────────────────────────────────────────────────────────>│             │
  │                       │                    │              │             │              │             │
  │                       │ 8. project_memory_to_graph (Neo4j)│             │              │             │
  │                       │────────────────────────────────────────────────────────────────────────────────>│
  │                       │                    │              │             │              │             │
  │                       │ 9. bridge_entities_to_om_graph (Neo4j)          │              │             │
  │                       │────────────────────────────────────────────────────────────────────────────────>│
  │                       │                    │              │             │              │             │
  │                       │ 10. update_entity_edges (Neo4j)   │             │              │             │
  │                       │────────────────────────────────────────────────────────────────────────────────>│
  │                       │                    │              │             │              │             │
  │                       │ 11. update_tag_edges (Neo4j)      │             │              │             │
  │                       │────────────────────────────────────────────────────────────────────────────────>│
  │                       │                    │              │             │              │             │
  │                       │ 12. project_similarity_edges (Neo4j)            │              │             │
  │                       │────────────────────────────────────────────────────────────────────────────────>│
  │                       │                    │              │             │              │             │
  │                       │ 13. update_job(status="succeeded")│             │              │             │
  │                       │                    │              │             │              │             │
```

### Detaillierter Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            add_memories Tool                                 │
│                         mcp_server.py:925-1082                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. VALIDIERUNG                                                              │
│    build_structured_memory() → Validiert category, scope, entity, tags      │
│    validate_code_refs_input() → Validiert code_refs (optional)              │
│    Fehler bei ungültigen Werten                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. ACCESS CONTROL                                                           │
│    resolve_access_entity_for_scope(principal, scope)                        │
│    • Personal scope (user/session) → Default: user:<sub>                    │
│    • Shared scope (team/project/org) → MUSS access_entity haben             │
│    • Mehrere Grants → Fehler (Ambiguität)                                   │
│    can_write_to_access_entity(principal, access_entity) → bool              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. ASYNC JOB (wenn async_mode=true, Default)                                │
│    create_memory_job() → job_id                                             │
│    Status: "queued" → "running" → "succeeded"/"failed"/"partial"            │
│    RETURN job_id sofort                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (asyncio.create_task)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. MEM0 CLIENT ADD                                                          │
│    memory_client.add(text, user_id, metadata, infer)                        │
│                                                                             │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │ WENN infer=true (LLM Fact Extraction)                           │     │
│    │ • Ollama LLM (qwen3:8b) extrahiert:                             │     │
│    │   - Entities (Personen, Services, Konzepte)                     │     │
│    │   - Relationships (z.B. "works_at", "vater_von")                │     │
│    │   - Facts für Graph                                             │     │
│    │ • Erstellt __Entity__ Nodes in Neo4j (Mem0 Graph)               │     │
│    │ • Kann DELETE events generieren (wenn Memory obsolet)           │     │
│    │ • LANGSAMER (~500ms-2s)                                         │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │ WENN infer=false (Default, Raw Storage)                         │     │
│    │ • Kein LLM-Call                                                 │     │
│    │ • Text wird direkt gespeichert                                  │     │
│    │ • NUR Embedding generiert                                       │     │
│    │ • SCHNELLER (~100-200ms)                                        │     │
│    └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. EMBEDDING GENERATION (IMMER)                                             │
│    Ollama Embedder (bge-m3)                                                 │
│    POST http://ollama:11434/api/embed                                       │
│    Input: text → Output: vector[1024]                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. QDRANT VECTOR STORE                                                      │
│    collection: "openmemory_bge_m3"                                          │
│    • Speichert: id, vector[1024], payload{metadata}                         │
│    • Index: HNSW für Approximate Nearest Neighbor                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. POSTGRESQL (Metadata)                                                    │
│    Memory Tabelle:                                                          │
│    • id (UUID)                                                              │
│    • user_id (FK → users)                                                   │
│    • app_id (FK → apps)                                                     │
│    • content (text)                                                         │
│    • metadata_ (JSON: category, scope, entity, access_entity, tags, ...)    │
│    • state (active/archived/deleted)                                        │
│    • created_at, updated_at                                                 │
│                                                                             │
│    MemoryStatusHistory Tabelle:                                             │
│    • old_state → new_state                                                  │
│    • changed_by (user_id)                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (asyncio.create_task - Non-Blocking)
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8-12. GRAPH PROJECTION (Best-Effort, Background)                            │
│                                                                             │
│ 8. project_memory_to_graph()                                                │
│    • Erstellt Memory Node mit access_entity                                 │
│    • Erstellt Entity/Category/Scope/Tag Nodes (falls nicht existent)        │
│    • Erstellt Edges: OM_ABOUT, OM_IN_CATEGORY, OM_TAGGED, etc.              │
│                                                                             │
│ 9. bridge_entities_to_om_graph() (wenn Mem0 Graph aktiviert)                │
│    • Synct __Entity__ Nodes von Mem0 zu OM Entity Nodes                     │
│    • Erstellt OM_RELATION Edges mit Typen (z.B. "works_at")                 │
│                                                                             │
│ 10. update_entity_edges_on_memory_add()                                     │
│     • Aktualisiert Entity-Entity Co-Mention Edges                           │
│     • Zählt, wie oft Entities zusammen vorkommen                            │
│                                                                             │
│ 11. update_tag_edges_on_memory_add()                                        │
│     • Aktualisiert Tag-Tag Co-Occurrence Edges (OM_COOCCURS)                │
│     • Berechnet PMI (Pointwise Mutual Information)                          │
│                                                                             │
│ 12. project_similarity_edges_for_memory()                                   │
│     • Findet K nächste Nachbarn in Qdrant                                   │
│     • Erstellt OM_SIMILAR Edges (similarity_score, rank)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-Calls bei add_memories

| Bedingung | LLM-Call? | Was passiert |
|-----------|-----------|--------------|
| `infer=false` (Default) | **NEIN** | Nur Embedding (bge-m3) |
| `infer=true` | **JA** | LLM (qwen3:8b) + Embedding (bge-m3) |

---

## 2. search_memory

### Sequenzdiagramm

```
Client                MCP Server           Mem0 Client      Ollama        Qdrant       PostgreSQL      Neo4j
  │                       │                    │              │             │              │             │
  │ search_memory(query,  │                    │              │             │              │             │
  │   entity, category,   │                    │              │             │              │             │
  │   filter_*,           │                    │              │             │              │             │
  │   use_rrf=true)       │                    │              │             │              │             │
  │──────────────────────>│                    │              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 1. ACL Check (PostgreSQL)         │             │              │             │
  │                       │ (get accessible_memory_ids)       │             │              │             │
  │                       │────────────────────────────────────────────────────────────────>│             │
  │                       │<────────────────────────────────────────────────────────────────│             │
  │                       │                    │              │             │              │             │
  │                       │ 2. Build hard filters             │             │              │             │
  │                       │ (filter_* + filter_mode)          │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 3. Query Routing   │              │             │              │             │
  │                       │ (in-memory, uses   │              │             │              │             │
  │                       │  cached entity list)              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ ════════════ PHASE 1: VECTOR SEARCH ════════════│             │             │
  │                       │                    │              │             │              │             │
  │                       │ 4. Embed query (Ollama)           │             │              │             │
  │                       │───────────────────>│              │             │              │             │
  │                       │                    │─────────────>│             │              │             │
  │                       │                    │<─────────────│             │              │             │
  │                       │<───────────────────│ vector[1024] │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 5. Vector search (Qdrant)         │             │              │             │
  │                       │ (limit * 3 candidates + filters)  │             │              │             │
  │                       │───────────────────>│──────────────────────────>│              │             │
  │                       │<───────────────────│<──────────────────────────│              │             │
  │                       │ hits[score, id, payload]          │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ ════════════ PHASE 2: GRAPH RETRIEVAL ══════════│             │             │
  │                       │                    │              │             │              │             │
  │                       │ 6. Similarity graph (Neo4j OM_SIMILAR)          │             │             │
  │                       │ (retrieve_via_similarity_graph)   │             │              │             │
  │                       │─────────────────────────────────────────────────────────────────────────────>│
  │                       │<─────────────────────────────────────────────────────────────────────────────│
  │                       │ graph_results      │              │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 7. Entity bridge (Neo4j OM_MENTIONS)            │             │             │
  │                       │ (find_bridge_entities)            │             │              │             │
  │                       │─────────────────────────────────────────────────────────────────────────────>│
  │                       │<─────────────────────────────────────────────────────────────────────────────│
  │                       │                    │              │             │              │             │
  │                       │ ════════════ PHASE 3: RE-RANKING ═══════════════│             │             │
  │                       │                    │              │             │              │             │
  │                       │ 8. RRF Fusion (in-memory)         │             │              │             │
  │                       │ (merge vector + graph results)    │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 9. compute_boost() (in-memory)    │             │              │             │
  │                       │ (metadata, recency, graph_context)│             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 10. calculate_final_score()       │             │              │             │
  │                       │                    │              │             │              │             │
  │                       │ 11. Meta-relations (Neo4j OM_*)   │             │              │             │
  │                       │ (get_meta_relations_for_memories) │             │              │             │
  │                       │─────────────────────────────────────────────────────────────────────────────>│
  │                       │<─────────────────────────────────────────────────────────────────────────────│
  │                       │                    │              │             │              │             │
  │                       │ 12. Mem0 Graph Relations (Neo4j + LLM)          │             │             │
  │                       │ (get_graph_relations) ─ SLOW!     │             │              │             │
  │                       │───────────────────>│─────────────>│ LLM extract │              │             │
  │                       │                    │<─────────────│ entities    │              │             │
  │                       │                    │───────────────────────────────────────────────────────>│
  │                       │                    │<───────────────────────────────────────────────────────│
  │                       │<───────────────────│ relations    │             │              │             │
  │                       │                    │              │             │              │             │
  │<──────────────────────│ results[id, memory,│              │             │              │             │
  │                       │   score, metadata, │              │             │              │             │
  │                       │   meta_relations,  │              │             │              │             │
  │                       │   relations]       │              │             │              │             │
```

### Detaillierter Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           search_memory Tool                                 │
│                        mcp_server.py:1188-1750                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: ACCESS CONTROL                                                     │
│                                                                             │
│ _get_accessible_memories(db, principal, user, app)                          │
│ • Query PostgreSQL für alle Memories des Users                              │
│ • Filtert nach access_entity basierend auf JWT grants                       │
│ • Gibt Set von erlaubten Memory IDs zurück                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1B: HARD FILTERS (optional, pre-search)                               │
│                                                                             │
│ normalize_filter_list(filter_*)                                             │
│ • filter_tags: key oder key=value                                           │
│ • filter_evidence, filter_category, filter_scope                            │
│ • filter_artifact_type, filter_artifact_ref, filter_entity                  │
│ • filter_source, filter_access_entity                                       │
│ filter_mode=all|any (tags/evidence)                                         │
│ → vector_filters {must, should}                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: QUERY ROUTING (wenn auto_route=true)                               │
│                                                                             │
│ analyze_query(query, user_id, config, access_entities)                      │
│ • Erkennt Entities im Query (via Neo4j Entity-Liste)                        │
│ • Erkennt Relationship-Keywords                                             │
│ • Entscheidet Route:                                                        │
│   - VECTOR_ONLY: Nur semantische Suche                                      │
│   - GRAPH_PRIMARY: Betont Graph-Traversal                                   │
│   - HYBRID: Balanciert (Default)                                            │
│ • Setzt RRF Alpha entsprechend                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: VECTOR SEARCH                                                      │
│                                                                             │
│ 1. Query Embedding:                                                         │
│    memory_client.embedding_model.embed(query, "search")                     │
│    → Ollama bge-m3 → vector[1024]                                           │
│                                                                             │
│ 2. Qdrant kNN Search:                                                       │
│    memory_client.vector_store.search(                                       │
│        query=query,                                                         │
│        vectors=embeddings,                                                  │
│        limit=limit * 3,  # Pool für Re-Ranking                              │
│        filters=vector_filters                                               │
│    )                                                                        │
│    → hits[{id, score, payload}]                                             │
│    • vector_filters enthält filter_* + user_id (wenn kein principal)        │
│                                                                             │
│ 3. ACL Filter:                                                              │
│    • Nur Hits behalten, deren ID in accessible_memory_ids                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: GRAPH RETRIEVAL (wenn use_rrf=true und Graph aktiviert)            │
│                                                                             │
│ A) Similarity Graph:                                                        │
│    retrieve_via_similarity_graph(user_id, seed_ids, allowed_ids, limit)     │
│    • Nimmt Top-K Vector-Hits als Seeds                                      │
│    • Traversiert OM_SIMILAR Edges in Neo4j                                  │
│    • Gibt zusätzliche Memory IDs mit Graph-Scores zurück                    │
│                                                                             │
│ B) Entity Bridge (wenn GRAPH_PRIMARY Route):                                │
│    find_bridge_entities(user_id, entity_names, max_hops)                    │
│    • Findet Entities, die Query-Entities verbinden                          │
│    • Expandiert Suche auf verbundene Memories                               │
│                                                                             │
│    retrieve_via_entity_graph(user_id, entity_names, allowed_ids)            │
│    • Findet Memories über Entity-Verbindungen                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: RRF FUSION                                                         │
│                                                                             │
│ RRFFusion.fuse(vector_results, graph_results, alpha)                        │
│                                                                             │
│ Reciprocal Rank Fusion:                                                     │
│ score = Σ (1 / (k + rank)) für jede Liste                                   │
│                                                                             │
│ Gewichte (Default):                                                         │
│ • alpha = 0.6 (60% Vector, 40% Graph)                                       │
│ • Angepasst je nach Route:                                                  │
│   - VECTOR_ONLY: alpha = 1.0                                                │
│   - GRAPH_PRIMARY: alpha = 0.4                                              │
│   - HYBRID: alpha = 0.6                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: RE-RANKING                                                         │
│                                                                             │
│ Für jedes Result:                                                           │
│                                                                             │
│ 1. Graph Context:                                                           │
│    get_graph_cooccurrence_context(memory_id, context_entity)                │
│    • Holt Entity Co-Mention Scores aus Neo4j                                │
│                                                                             │
│ 2. Boost Berechnung:                                                        │
│    compute_boost(metadata, stored_tags, context, created_at, graph_context) │
│    • +0.1 für Category Match                                                │
│    • +0.1 für Entity Match                                                  │
│    • +0.05 pro Tag Match                                                    │
│    • +recency_weight * exp(-days/halflife) für Recency                      │
│    • +graph_boost für Co-Occurrence                                         │
│                                                                             │
│ 3. Final Score:                                                             │
│    final_score = semantic_score * (1 + boost)                               │
│                                                                             │
│ 4. Exclusion Filter:                                                        │
│    should_exclude(metadata, filters)                                        │
│    • Excludiert deleted States                                              │
│    • Excludiert bestimmte Tags                                              │
│    • Date Range Filter (created_after/before)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: META-RELATIONS (Output Enrichment)                                 │
│                                                                             │
│ get_meta_relations_for_memories(memory_ids, relation_detail)                │
│                                                                             │
│ relation_detail Levels:                                                     │
│ • "none": Keine meta_relations (minimal tokens)                             │
│ • "minimal": Nur artifact + similar IDs                                     │
│ • "standard": + entities + tags + evidence (Default)                        │
│ • "full": Alle OM_* Relationen (verbose)                                    │
│                                                                             │
│ Abfrage aus Neo4j:                                                          │
│ • OM_ABOUT → entities                                                       │
│ • OM_SIMILAR → similar_memory_ids                                           │
│ • OM_TAGGED → tags                                                          │
│ • OM_HAS_EVIDENCE → evidence                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-Calls bei search_memory

| Aktion | LLM-Call? | Was passiert |
|--------|-----------|--------------|
| Query Embedding | **NEIN** (nur Embedder) | bge-m3 generiert Query-Vector |
| Query Routing | **NEIN** | Regelbasierte Entscheidung |
| Re-Ranking | **NEIN** | Mathematische Score-Berechnung |

**Kein LLM-Call bei search_memory!** Nur Embedding-Generierung.

---

## 3. update_memory

### Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           update_memory Tool                                 │
│                        mcp_server.py:2637-2800                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ACL CHECK                                                                │
│    • Get Memory from PostgreSQL                                             │
│    • Check access_entity gegen Principal Grants                             │
│    • Group-Editable: Jeder Grant-Holder kann editieren                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. WENN text geändert:                                                      │
│    memory_client.update(memory_id, new_text)                                │
│    • Generiert neues Embedding (Ollama bge-m3)                              │
│    • Aktualisiert Qdrant Vector                                             │
│                                                                             │
│    WENN text NICHT geändert:                                                │
│    • Nur PostgreSQL Metadata Update                                         │
│    • Kein Embedding, kein Qdrant Update                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. POSTGRESQL UPDATE                                                        │
│    • metadata_ JSON aktualisieren (merge)                                   │
│    • add_tags → fügt neue Tags hinzu                                        │
│    • remove_tags → entfernt Tags                                            │
│    • updated_at Timestamp                                                   │
│    • StatusHistory Entry                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. GRAPH UPDATE (wenn Graph aktiviert)                                      │
│    project_memory_to_graph() mit merge_mode=true                            │
│    • Aktualisiert Memory Node Properties                                    │
│    • Aktualisiert Entity/Tag Edges bei Änderungen                           │
│    • Re-Projiziert Similarity Edges (wenn text geändert)                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-Calls bei update_memory

| Bedingung | LLM-Call? | Was passiert |
|-----------|-----------|--------------|
| Nur Metadata-Update | **NEIN** | Nur PostgreSQL + Neo4j |
| Text-Update | **NEIN** (nur Embedder) | Neues Embedding generiert |

---

## 4. delete_memories

### Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          delete_memories Tool                                │
│                        mcp_server.py:2404-2520                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ACL CHECK (für jede Memory ID)                                           │
│    • Get Memory from PostgreSQL                                             │
│    • Check access_entity gegen Principal Grants                             │
│    • Nur löschen wenn Grant matched                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. MEM0 DELETE                                                              │
│    memory_client.delete(memory_id)                                          │
│    • Löscht Vector aus Qdrant                                               │
│    • Mem0 internal cleanup                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. POSTGRESQL                                                               │
│    • state = MemoryState.deleted                                            │
│    • deleted_at = now()                                                     │
│    • StatusHistory Entry                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. GRAPH CLEANUP                                                            │
│    delete_memory_from_graph(memory_id, user_id)                             │
│    • Löscht Memory Node                                                     │
│    • Löscht alle verbundenen Edges (OM_ABOUT, OM_SIMILAR, etc.)             │
│                                                                             │
│    delete_similarity_edges_for_memory(memory_id, user_id)                   │
│    • Explizit OM_SIMILAR Edges entfernen                                    │
│                                                                             │
│    update_entity_edges_on_memory_delete(memory_id, user_id)                 │
│    • Aktualisiert Entity Co-Mention Counts                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. index_codebase

### Sequenzdiagramm

```
Client                MCP Server           CodeIndexer      TreeSitter     OpenSearch      Neo4j       Ollama
  │                       │                    │               │              │             │            │
  │ index_codebase(       │                    │               │              │             │            │
  │   repo_id, root_path, │                    │               │              │             │            │
  │   reset=true,         │                    │               │              │             │            │
  │   async_mode=true)    │                    │               │              │             │            │
  │──────────────────────>│                    │               │              │             │            │
  │                       │                    │               │              │             │            │
  │                       │ 1. create_index_job│               │              │             │            │
  │                       │ (status="queued")  │               │              │             │            │
  │                       │                    │               │              │             │            │
  │<──────────────────────│ return job_id     │               │              │             │            │
  │                       │                    │               │              │             │            │
  │                       │====== ASYNC BACKGROUND ======      │              │             │            │
  │                       │                    │               │              │             │            │
  │                       │ 2. CodeIndexer    │               │              │             │            │
  │                       │───────────────────>│               │              │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ PHASE 1: Symbol Discovery    │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 3. Parse files│              │             │            │
  │                       │                    │──────────────>│              │             │            │
  │                       │                    │<──────────────│              │             │            │
  │                       │                    │ AST + symbols │              │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 4. Extract SCIP IDs          │             │            │
  │                       │                    │ (package + descriptor)       │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ PHASE 2: Graph Projection    │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 5. Create CODE_FILE nodes    │             │            │
  │                       │                    │────────────────────────────────────────────>│            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 6. Create CODE_SYMBOL nodes  │             │            │
  │                       │                    │────────────────────────────────────────────>│            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 7. Create CONTAINS edges     │             │            │
  │                       │                    │────────────────────────────────────────────>│            │
  │                       │                    │               │              │             │            │
  │                       │                    │ PHASE 3: Edge Extraction     │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 8. Inferred CALLS edges      │             │            │
  │                       │                    │ (regex-based) │              │             │            │
  │                       │                    │────────────────────────────────────────────>│            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 9. Deterministic edges       │             │            │
  │                       │                    │ (AST-based)   │              │             │            │
  │                       │                    │──────────────>│              │             │            │
  │                       │                    │<──────────────│              │             │            │
  │                       │                    │────────────────────────────────────────────>│            │
  │                       │                    │               │              │             │            │
  │                       │                    │ PHASE 4: OpenSearch Indexing │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 10. Generate embeddings      │             │            │
  │                       │                    │───────────────────────────────────────────────────────>│
  │                       │                    │<───────────────────────────────────────────────────────│
  │                       │                    │ vectors       │              │             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 11. Bulk index documents     │             │            │
  │                       │                    │───────────────────────────────>│             │            │
  │                       │                    │               │              │             │            │
  │                       │                    │ 12. update_job(succeeded)    │             │            │
  │                       │<───────────────────│               │              │             │            │
```

### Detaillierter Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         index_codebase Tool                                  │
│                      mcp_server.py:3414-3550                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SYMBOL DISCOVERY                                                   │
│                                                                             │
│ Für jede Datei (.py, .ts, .tsx, .java, .go, .js, .jsx):                     │
│                                                                             │
│ 1. Tree-sitter AST Parsing:                                                 │
│    ASTParser.parse_file(file_path)                                          │
│    → ParseResult{symbols, imports, calls}                                   │
│                                                                             │
│ 2. Python Fallback (wenn AST fehlschlägt):                                  │
│    extract_python_symbols(content)                                          │
│    → Regex-basierte Symbol-Extraktion                                       │
│                                                                             │
│ 3. SCIP ID Generation:                                                      │
│    SCIPExtractor.extract(symbol, file_path)                                 │
│    Format: <scheme> <package> <descriptor>+                                 │
│    Beispiel: "scip-python myapp.module file#ClassName.method."              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: GRAPH PROJECTION (Neo4j CODE_*)                                    │
│                                                                             │
│ Node Types:                                                                 │
│ ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
│ │  CODE_FILE     │  │  CODE_SYMBOL   │  │  CODE_PACKAGE  │                  │
│ │  - path        │  │  - scip_id     │  │  - name        │                  │
│ │  - language    │  │  - name        │  │  - language    │                  │
│ │  - repo_id     │  │  - symbol_type │  │  - repo_id     │                  │
│ │  - content_hash│  │  - signature   │  └────────────────┘                  │
│ └────────────────┘  │  - docstring   │                                      │
│                     │  - line_start  │                                      │
│                     │  - line_end    │                                      │
│                     └────────────────┘                                      │
│                                                                             │
│ Edge Types:                                                                 │
│ • CONTAINS: FILE → SYMBOL                                                   │
│ • IMPORTS: FILE/SYMBOL → FILE/SYMBOL                                        │
│ • CALLS: SYMBOL → SYMBOL (mit inferred=true/false)                          │
│ • TRIGGERS_EVENT: SYMBOL → EVENT (für Decorators)                           │
│                                                                             │
│ Constraints:                                                                │
│ CREATE CONSTRAINT code_symbol_scip_id_unique                                │
│ FOR (s:CODE_SYMBOL) REQUIRE s.scip_id IS UNIQUE                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: EDGE EXTRACTION                                                    │
│                                                                             │
│ A) Inferred Edges (First Pass, regex-based):                                │
│    • Pattern: r"\b{function_name}\s*\("                                     │
│    • Für jede Funktion: Suche nach Aufrufen im Body                         │
│    • Markiert als inferred=true                                             │
│                                                                             │
│ B) Deterministic Edges (Second Pass, AST-based):                            │
│    DeterministicEdgeExtractor.extract_edges(file, language, content)        │
│    • Tree-sitter Call-Expression Traversal                                  │
│    • Auflösung von import Aliase                                            │
│    • Markiert als inferred=false, resolution="ast"                          │
│                                                                             │
│ C) Decorator Extraction (NestJS/Angular):                                   │
│    • 60+ Decorator Patterns (@OnEvent, @Subscribe, etc.)                    │
│    • Erstellt TRIGGERS_EVENT Edges                                          │
│    • Event Registry für Publisher/Subscriber Discovery                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: OPENSEARCH INDEXING                                                │
│                                                                             │
│ 1. Embedding Generation:                                                    │
│    Für jedes Symbol:                                                        │
│    content = function_body OR symbol_name                                   │
│    embedding = embed(content) → vector[768 oder 1024]                       │
│                                                                             │
│ 2. Document Structure:                                                      │
│    {                                                                        │
│      "id": scip_id,                                                         │
│      "content": function_body,                                              │
│      "embedding": vector[],                                                 │
│      "metadata": {                                                          │
│        "symbol_name": "myFunction",                                         │
│        "symbol_type": "function",                                           │
│        "file_path": "/src/module.py",                                       │
│        "language": "python",                                                │
│        "line_start": 42,                                                    │
│        "line_end": 67,                                                      │
│        "signature": "def myFunction(arg: str) -> int",                      │
│        "docstring": "...",                                                  │
│        "repo_id": "my-repo"                                                 │
│      }                                                                      │
│    }                                                                        │
│                                                                             │
│ 3. Bulk Index:                                                              │
│    opensearch_client.bulk_index(index_name, documents)                      │
│    • BM25 auf content, symbol_name, signature                               │
│    • kNN Index für dense vectors                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-Calls bei index_codebase

| Phase | LLM-Call? | Was passiert |
|-------|-----------|--------------|
| AST Parsing | **NEIN** | Tree-sitter (deterministisch) |
| SCIP ID Generation | **NEIN** | Regelbasiert |
| Graph Projection | **NEIN** | Neo4j Writes |
| Edge Extraction | **NEIN** | AST + Regex |
| Embedding Generation | **NEIN** (nur Embedder) | Ollama bge-m3 |

**Kein LLM-Call bei index_codebase!** Nur Embeddings.

---

## 6. search_code_hybrid (Tri-Hybrid Retrieval)

### Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        search_code_hybrid Tool                               │
│                      mcp_server.py:3600-3700                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 1: LEXICAL SEARCH (BM25)                                          │
│                                                                             │
│ OpenSearch BM25 Query:                                                      │
│ {                                                                           │
│   "query": {                                                                │
│     "multi_match": {                                                        │
│       "query": "auth middleware",                                           │
│       "fields": ["content", "symbol_name^2", "signature"]                   │
│     }                                                                       │
│   }                                                                         │
│ }                                                                           │
│ → lexical_results[{id, score, rank}]                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 2: SEMANTIC SEARCH (kNN)                                          │
│                                                                             │
│ 1. Query Embedding:                                                         │
│    embed(query) → vector[768/1024]                                          │
│                                                                             │
│ 2. OpenSearch kNN Query:                                                    │
│    {                                                                        │
│      "knn": {                                                               │
│        "embedding": {"vector": [...], "k": 20}                              │
│      }                                                                      │
│    }                                                                        │
│    → vector_results[{id, score, rank}]                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPONENT 3: GRAPH CONTEXT                                                  │
│                                                                             │
│ GraphContextFetcher.fetch_context(symbol_ids, depth=2)                      │
│                                                                             │
│ Neo4j Traversal:                                                            │
│ MATCH (s:CODE_SYMBOL)-[r:CALLS|IMPORTS|CONTAINS]->(t:CODE_SYMBOL)           │
│ WHERE s.scip_id IN $seed_ids                                                │
│ RETURN t.scip_id, type(r)                                                   │
│                                                                             │
│ → graph_results[{id, score, rank}]                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FUSION: Reciprocal Rank Fusion (RRF)                                        │
│                                                                             │
│ 1. Min-Max Normalize each list to [0, 1]                                    │
│                                                                             │
│ 2. RRF Score:                                                               │
│    score(doc) = Σ (1 / (k + rank_i))                                        │
│    k = 60 (default)                                                         │
│                                                                             │
│ 3. Weighted Combination:                                                    │
│    final = 0.40 * vector + 0.35 * lexical + 0.25 * graph                    │
│                                                                             │
│ → fused_results[{id, score, symbol_info}]                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-Calls bei search_code_hybrid

| Komponente | LLM-Call? | Was passiert |
|------------|-----------|--------------|
| BM25 Lexical | **NEIN** | OpenSearch Full-Text |
| kNN Semantic | **NEIN** (nur Embedder) | Query Embedding |
| Graph Context | **NEIN** | Neo4j Traversal |
| RRF Fusion | **NEIN** | Mathematische Fusion |

---

## 7. find_callers / find_callees

### Flow mit Fallback Cascade

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         find_callers Tool                                    │
│                      (FallbackFindCallers)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: GRAPH SEARCH (Primary)                                             │
│ Timeout: 150ms                                                              │
│                                                                             │
│ Neo4j Query:                                                                │
│ MATCH (caller:CODE_SYMBOL)-[r:CALLS]->(target:CODE_SYMBOL)                  │
│ WHERE target.scip_id = $symbol_id                                           │
│ RETURN caller, r                                                            │
│                                                                             │
│ IF found → RETURN nodes, edges                                              │
│ IF SymbolNotFoundError → STAGE 2                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: GREP FALLBACK                                                      │
│ Timeout: 150ms                                                              │
│                                                                             │
│ GrepTool.search(                                                            │
│   pattern=symbol_name,                                                      │
│   include_patterns=["*.ts", "*.py", "*.js"],                                │
│   max_results=50                                                            │
│ )                                                                           │
│                                                                             │
│ IF found → RETURN with degraded_mode=true, fallback_stage=2                 │
│ IF empty → STAGE 3                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SEMANTIC SEARCH                                                    │
│ Timeout: 200ms                                                              │
│                                                                             │
│ 1. Extract keywords:                                                        │
│    "moveFilesToPermanentStorage" → ["move", "files", "permanent", "storage"]│
│                                                                             │
│ 2. search_code_hybrid(query=keywords, limit=20)                             │
│                                                                             │
│ 3. Filter by min_score >= 0.5                                               │
│                                                                             │
│ IF found → RETURN with degraded_mode=true, fallback_stage=3                 │
│ IF empty → STAGE 4                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: STRUCTURED ERROR                                                   │
│                                                                             │
│ RETURN {                                                                    │
│   nodes: [],                                                                │
│   edges: [],                                                                │
│   meta: {                                                                   │
│     degraded_mode: true,                                                    │
│     fallback_stage: 4,                                                      │
│     missing_sources: ["graph_index", "grep", "semantic_search"]             │
│   },                                                                        │
│   suggestions: [                                                            │
│     "Try: grep -r 'symbol_name' --include='*.ts'",                          │
│     "Symbol may be called via decorator (@OnEvent)",                        │
│     "Symbol may be injected via DI",                                        │
│     "Index may be stale - try index_codebase(reset=true)"                   │
│   ]                                                                         │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LLM-Calls bei find_callers

| Stage | LLM-Call? | Was passiert |
|-------|-----------|--------------|
| Graph Search | **NEIN** | Neo4j Query |
| Grep Fallback | **NEIN** | Pattern Match |
| Semantic Search | **NEIN** (nur Embedder) | search_code_hybrid |
| Structured Error | **NEIN** | Statische Suggestions |

---

## 8. Graph Tools (graph_*)

### Übersicht

| Tool | Store | LLM-Call? | Beschreibung |
|------|-------|-----------|--------------|
| `graph_related_memories` | Neo4j | **NEIN** | Findet Memories über Metadata-Verbindungen |
| `graph_similar_memories` | Neo4j | **NEIN** | Findet semantisch ähnliche Memories (OM_SIMILAR) |
| `graph_subgraph` | Neo4j | **NEIN** | Holt Nachbarschaft eines Memory Nodes |
| `graph_aggregate` | Neo4j | **NEIN** | Gruppiert Memories nach Dimension |
| `graph_tag_cooccurrence` | Neo4j | **NEIN** | Findet Tag-Paare mit hohem PMI |
| `graph_entity_network` | Neo4j | **NEIN** | Entities, die oft zusammen erwähnt werden |
| `graph_entity_relations` | Neo4j | **NEIN** | Typisierte Beziehungen zwischen Entities |
| `graph_path_between_entities` | Neo4j | **NEIN** | Kürzester Pfad zwischen zwei Entities |
| `graph_biography_timeline` | Neo4j | **NEIN** | Chronologische Events für eine Person |
| `graph_normalize_entities` | Neo4j | **NEIN** | Merged Duplicate Entities |

**Alle Graph-Tools sind reine Neo4j-Queries ohne LLM-Calls!**

---

## Zusammenfassung: LLM-Calls

| Tool | LLM für Fact Extraction? | Embedding? |
|------|--------------------------|------------|
| `add_memories` (infer=false) | **NEIN** | **JA** (bge-m3) |
| `add_memories` (infer=true) | **JA** (qwen3:8b) | **JA** (bge-m3) |
| `search_memory` | **NEIN** | **JA** (Query Embedding) |
| `update_memory` (text changed) | **NEIN** | **JA** (Re-Embedding) |
| `update_memory` (metadata only) | **NEIN** | **NEIN** |
| `delete_memories` | **NEIN** | **NEIN** |
| `list_memories` | **NEIN** | **NEIN** |
| `get_memory` | **NEIN** | **NEIN** |
| `graph_*` | **NEIN** | **NEIN** |
| `index_codebase` | **NEIN** | **JA** (Code Embeddings) |
| `search_code_hybrid` | **NEIN** | **JA** (Query Embedding) |
| `find_callers/callees` | **NEIN** | **JA** (bei Semantic Fallback) |
| `explain_code` | **NEIN** | **NEIN** |
| `impact_analysis` | **NEIN** | **NEIN** |

---

## Datenfluss-Matrix

| Operation | PostgreSQL | Qdrant | Neo4j (OM_*) | Neo4j (CODE_*) | OpenSearch | Ollama |
|-----------|------------|--------|--------------|----------------|------------|--------|
| add_memories | WRITE | WRITE | WRITE | - | - | EMBED (+LLM wenn infer) |
| search_memory | READ | READ | READ | - | - | EMBED |
| update_memory | WRITE | WRITE | WRITE | - | - | EMBED (wenn text) |
| delete_memories | WRITE | DELETE | DELETE | - | - | - |
| list_memories | READ | - | READ | - | - | - |
| graph_* | - | - | READ | - | - | - |
| index_codebase | - | - | - | WRITE | WRITE | EMBED |
| search_code_hybrid | - | - | - | READ | READ | EMBED |
| find_callers | - | - | - | READ | READ | EMBED (fallback) |

---

## Konfigurierbare Komponenten

### LLM (Fact Extraction)
```python
"llm": {
    "provider": "ollama",  # oder openai, claude, gemini
    "config": {
        "model": "qwen3:8b",
        "temperature": 0.1,
        "max_tokens": 2000,
        "ollama_base_url": "http://ollama:11434"
    }
}
```

### Embedder
```python
"embedder": {
    "provider": "ollama",  # oder openai, gemini
    "config": {
        "model": "bge-m3",
        "embedding_dims": 1024,
        "ollama_base_url": "http://ollama:11434"
    }
}
```

### Vector Store
```python
"vector_store": {
    "provider": "qdrant",  # oder chroma, weaviate, redis, pgvector, milvus, elasticsearch, opensearch, faiss
    "config": {
        "collection_name": "openmemory_bge_m3",
        "host": "mem0_store",
        "port": 6333,
        "embedding_model_dims": 1024
    }
}
```

### Graph Store
```python
"graph_store": {
    "provider": "neo4j",
    "config": {
        "url": "bolt://neo4j:7687",
        "username": "neo4j",
        "password": "..."
    }
}
```

---

## 9. Debug Timing

### Übersicht

Die MCP Tools `add_memories` und `search_memory` unterstützen einen `debug=True` Parameter, der detaillierte Timing-Informationen zurückgibt. Dies hilft bei der Performance-Analyse und Identifikation von Bottlenecks.

### Verwendung

```python
# add_memories mit Debug-Timing
add_memories(
    text="Test memory",
    category="decision",
    entity="TestEntity",
    debug=True  # Aktiviert Timing-Ausgabe
)

# search_memory mit Debug-Timing
search_memory(
    query="test query",
    limit=10,
    debug=True  # Aktiviert Timing-Ausgabe
)
```

### add_memories Debug-Output

```json
{
  "id": "uuid-...",
  "status": "created",
  "_debug_timing": {
    "total_ms": 523.4,
    "breakdown": {
      "access_resolution_ms": 1.2,
      "validation_ms": 2.1,
      "mem0_add_ms": 312.5,
      "graph_projection_trigger_ms": 0.8
    },
    "details": {
      "mem0_add": {
        "db_get_user_app_ms": 5.2,
        "mem0_client_add_ms": 298.4,
        "postgresql_write_ms": 3.1,
        "postgresql_commit_ms": 5.8,
        "infer": false
      }
    }
  }
}
```

### search_memory Debug-Output

```json
{
  "results": [...],
  "_debug_timing": {
    "total_ms": 245.6,
    "breakdown": {
      "acl_check_ms": 12.3,
      "query_routing_ms": 2.1,
      "vector_embedding_ms": 45.2,
      "vector_search_ms": 28.7,
      "graph_retrieval_ms": 65.4,
      "entity_expansion_ms": 15.2,
      "graph_context_ms": 8.9,
      "result_processing_ms": 22.1,
      "rrf_fusion_ms": 12.5,
      "access_logging_ms": 8.4,
      "response_format_ms": 2.1,
      "meta_relations_ms": 22.7
    },
    "details": {
      "acl_check": {"accessible_count": 150},
      "query_routing": {"route": "hybrid"},
      "vector_search": {"hits": 30},
      "graph_retrieval": {"candidates": 25},
      "entity_expansion": {"bridges": 3},
      "result_processing": {"processed": 28},
      "rrf_fusion": {"used_graph": true}
    }
  }
}
```

### Timing-Phasen (search_memory)

| Phase | Beschreibung | Typische Dauer |
|-------|-------------|----------------|
| `acl_check` | ACL-Prüfung, accessible memories laden | 5-20ms |
| `query_routing` | Intelligente Query-Routing-Analyse | 1-5ms |
| `vector_embedding` | Query-Embedding generieren (Ollama) | 30-80ms |
| `vector_search` | Qdrant kNN-Suche | 10-50ms |
| `graph_retrieval` | Neo4j OM_SIMILAR Graph-Traversal | 20-100ms |
| `entity_expansion` | Bridge-Entity-Erweiterung | 5-30ms |
| `graph_context` | Graph-Kontext für Boost laden | 5-20ms |
| `result_processing` | Ergebnisse verarbeiten und re-ranken | 10-40ms |
| `rrf_fusion` | RRF-Fusion von Vector + Graph | 5-20ms |
| `access_logging` | Access-Logs schreiben | 5-15ms |
| `response_format` | Response formatieren | 1-5ms |
| `meta_relations` | Neo4j meta_relations abrufen | 10-40ms |

### Timing-Phasen (add_memories)

| Phase | Beschreibung | Typische Dauer |
|-------|-------------|----------------|
| `access_resolution` | access_entity auflösen | 1-3ms |
| `validation` | Input validieren | 1-5ms |
| `mem0_add` | Mem0 Client add (inkl. Embedding) | 200-500ms |
| `graph_projection_trigger` | Graph-Projektion anstoßen | 0.5-2ms |

### Implementierung

Die Debug-Timing-Funktionalität ist in `app/utils/debug_timing.py` implementiert:

```python
from app.utils.debug_timing import DebugTimer, TimingContext

# Manuelle Start/Stop
timer = DebugTimer(enabled=debug)
timer.start("operation_name")
# ... code ...
timer.stop("operation_name", metadata={"key": "value"})

# Context Manager
with TimingContext(timer, "operation_name"):
    # ... code ...

# Timing abrufen
timing = timer.get_timing()
# Returns: {"total_ms": 123.4, "breakdown": {...}, "details": {...}}
```

### Performance-Hinweise

- Debug-Timing hat minimalen Overhead (~0.1ms pro Operation)
- Bei `debug=False` (Standard) werden keine Timing-Daten erfasst
- Für Produktions-Monitoring verwende Prometheus-Metriken (`/metrics`)
