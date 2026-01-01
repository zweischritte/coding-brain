# PRD: Switch Memory Embeddings to Local BGE‑M3 (Ollama)

**Version:** 1.0
**Date:** 2026-01-01
**Status:** Draft (for later)
**Owner:** TBD

---

## Executive Summary
Switch the OpenMemory/Mem0 memory embedding model from OpenAI `text-embedding-3-small` (1536 dims, remote API) to a **local** BGE‑M3 model served via **Ollama**. This removes OpenAI dependency, keeps embeddings local, and improves retrieval quality at the cost of higher local inference latency and a Qdrant re-embedding migration (1536 → 1024 dims).

---

## Goals
- Remove OpenAI embeddings from memory pipeline.
- Use local embeddings (BGE‑M3) via Ollama on macOS/Linux hosts.
- Preserve or improve semantic retrieval quality.
- Provide a clear migration path for Qdrant dimension changes.

## Non‑Goals
- Switching LLM generation models.
- Changing vector store provider (remain on Qdrant).
- Implementing new retrieval algorithms beyond existing RRF/graph steps.

---

## Current State
- Memory embeddings use OpenAI `text-embedding-3-small` with **1536 dims**.
- Config defined in `openmemory/api/app/utils/memory.py`.
- Qdrant collection `openmemory` is dimension‑locked at 1536.

---

## Proposed Change
- Embedder provider: `ollama`.
- Model: `bge-m3` (local, via Ollama).
- Output dims: **1024** (BGE‑M3 default).
- Update Qdrant collection to 1024 dims (new collection + re-embed).

---

## Success Criteria
1. Memory embeddings are generated locally via Ollama BGE‑M3.
2. No OpenAI embedding calls are made for memory add/search.
3. Qdrant collection is 1024 dims and contains all memory embeddings.
4. Search quality does not regress in evaluation queries.
5. Add/search latency remains within acceptable thresholds.

---

## Constraints and Risks
- **Dimension change requires re-embedding** all memories.
- BGE‑M3 is slower than Nomic on macOS (~83 ms per embed vs ~22 ms).
- If Ollama is not running, memory add/search fails.
- Graph similarity edges may need regeneration post‑migration.

---

## Architecture & Integration

### Embedder config (Mem0)
Update default config in `openmemory/api/app/utils/memory.py`:

```python
"embedder": {
    "provider": "ollama",
    "config": {
        "model": "bge-m3",
        "ollama_base_url": "http://ollama:11434",
        "embedding_dims": 1024
    }
}
```

### Vector store config (Qdrant)
Update embedding dims to match:

```python
"vector_store": {
    "provider": "qdrant",
    "config": {
        "collection_name": "openmemory_bge_m3",
        "host": "qdrant",
        "port": 6333,
        "embedding_model_dims": 1024
    }
}
```

---

## Migration Plan (Required)

### Phase 0: Prep
- Ensure Ollama is running and has `bge-m3` pulled.
- Confirm Qdrant is healthy.
- Schedule maintenance window (migration requires re-embed).

### Phase 1: Create new Qdrant collection
- Create `openmemory_bge_m3` with 1024 dims.

### Phase 2: Re-embed and backfill
Preferred approach: **backup export/import** (provider‑agnostic).
1) Export memories via `/api/v1/backup/export`.
2) Switch embedder config to Ollama + BGE‑M3.
3) Import via `/api/v1/backup/import` to re‑embed into new collection.

### Phase 3: Switch reads
- Update `collection_name` to `openmemory_bge_m3`.
- Verify add/search works end‑to‑end.

### Phase 4: Graph similarity refresh
- Rebuild similarity edges (if used) after re‑embedding.

### Phase 5: Cleanup
- Archive or delete old `openmemory` collection after validation.

---

## Tests & Validation
- **Embedding check**: confirm embedder provider is `ollama` and no OpenAI calls.
- **Add memory**: `add_memories` completes and stores new vectors.
- **Search**: baseline query set matches or improves top‑k recall.
- **Latency**: measure P95 for add/search before/after migration.

---

## Rollback Plan
- Keep old Qdrant collection until validation passes.
- If issues found, revert config to OpenAI embedder and old collection.
- Re-deploy and restore previous config.

---

## Benchmark Reference (Local macOS)
- `nomic-embed-text` (Ollama): ~22 ms mean
- `bge-m3` (Ollama): ~83 ms mean
- Gemini API (text‑embedding‑004): ~372 ms mean

---

## Open Questions
- Expected acceptable P95 for add/search after migration?
- Is Ollama acceptable for production (macOS/Linux), or should we add a Linux TEI deployment for servers?
- Should we add automated backfill scripts and a migration CLI for future swaps?

