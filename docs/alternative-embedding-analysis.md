# Alternative Embeddings Analysis (Local or Gemini Only)

## Objective
Replace OpenAI `text-embedding-3-small` with either a local embedder or a Gemini-based embedder. This analysis is based on:
- `alternative-embedding-claude-researcher-good.md`
- `alternative-embedding-codex-researcher.md`

## System Constraints (from repo)
- Memory embeddings are configured in `openmemory/api/app/utils/memory.py` (Mem0 config).
- Qdrant collection `openmemory` is currently 1536 dims (OpenAI default).
- Allowed embedder providers in code: `ollama`, `huggingface`, `gemini`, `vertexai`, `fastembed`, `openai` (see `mem0/utils/factory.py`).
- `gemini` provider is implemented in `mem0/embeddings/gemini.py` (uses Google GenAI `embed_content`).
- `ollama` provider exists in `mem0/embeddings/ollama.py` and expects `ollama_base_url` plus model name.

## Evaluation Criteria
- Compliance: local or Gemini only.
- Quality: strong semantic retrieval for short-to-medium text memories.
- Operational simplicity: minimal infra changes.
- Performance: acceptable latency for `add_memories`.
- Migration cost: avoid or minimize re-embedding if possible.

## Candidate Options (from research docs)

### Local (Ollama or TEI)
1) **Nomic embed text v1.5** (768 dims)
   - Pros: strong quality, efficient, long context, Apache 2.0 license.
   - Cons: dimension change (1536 -> 768) requires Qdrant re-embed.
   - Serving: Ollama or TEI.

2) **BGE-M3** (1024 dims)
   - Pros: top quality, multilingual, supports hybrid retrieval.
   - Cons: heavier model, higher resource use, dimension change.
   - Serving: TEI recommended for production.

3) **E5 (base/large)** (768/1024 dims)
   - Pros: strong quality, widely used; low latency for base.
   - Cons: requires query prefixing; dimension change.

4) **Jina Embeddings v3** (1024 dims)
   - Pros: strong multilingual, long context.
   - Cons: non-commercial license unless licensed.

### Gemini
1) **Gemini Embedding** (`gemini-embedding-001` or `models/text-embedding-004`)
   - Pros: high quality, adjustable output dimensionality (can match 1536), no local infra.
   - Cons: external API (not local), cost and latency variability.

## Key Findings
- **Local models almost always require dimension changes**, which implies re-embedding Qdrant.
- **Gemini supports configurable output dimensions** and can match 1536, which avoids Qdrant migration.
- The repository already supports `gemini` embedder provider in Mem0, but the Claude doc mentions `google_ai` provider. In this codebase, the correct provider is `gemini` (see `mem0/utils/factory.py`).

## Local Benchmarks (MacBook Pro M4 Max, macOS, Ollama)
Measured locally with a representative short memory payload (~6x repeated sentence).

| Model | Provider | Mean | P95 | Throughput |
| --- | --- | --- | --- | --- |
| `nomic-embed-text:latest` | Ollama | ~22 ms | ~28.6 ms | ~45 req/s |
| `bge-m3` | Ollama | ~83 ms | ~86.4 ms | ~12 req/s |

**Notes**:
- TEI `BAAI/bge-m3` on macOS runs under linux/amd64 emulation and was ~1.6 s per embed, which is not representative of Linux/AVX or GPU deployments.
- Ollama is the practical local path on macOS because it uses Metal acceleration.

## Gemini Benchmarks (API)
Measured with `models/text-embedding-004` and `output_dimensionality=1536` using the same payload.

| Model | Provider | Mean | P95 | Throughput |
| --- | --- | --- | --- | --- |
| `text-embedding-004` | Gemini API | ~372 ms | ~442 ms | ~2.7 req/s |

## Impact on Add and Retrieval Latency
- **Add path**: embedding time is only one part of `add_memories` (Qdrant + Postgres + graph updates). On this machine, switching from Nomic (~22 ms) to BGE‑M3 (~83 ms) adds ~60 ms per memory. If the rest of the pipeline is ~200–600 ms, the total difference is noticeable but not dominant.
- **Retrieval path**: query embedding time is on the critical path for search. The Nomic vs BGE‑M3 gap is ~60 ms per query on macOS with Ollama. Qdrant search time differences between 768 vs 1024 dims are typically small compared to embedding generation.
- **Quality tradeoff**: BGE‑M3 generally improves multilingual and overall retrieval quality, but costs ~4x latency per embed on macOS compared to Nomic. For strict latency or high QPS, Nomic is usually the better balance.

## Recommended Solutions

### Primary (Local, simplest ops): Ollama + Nomic Embed Text v1.5
Best balance of quality, speed, and operational simplicity.
- **Model**: `nomic-embed-text` (768 dims)
- **Serving**: Ollama (`http://ollama:11434`)
- **Impact**: requires Qdrant re-embed (dimension change).
- **Why this wins**: lowest operational complexity; widely used; good quality; Apache 2.0.

### Primary (Local, highest quality): TEI + BGE-M3
Best quality, especially if multilingual or hybrid retrieval is important.
- **Model**: `BAAI/bge-m3` (1024 dims)
- **Serving**: Hugging Face TEI (OpenAI-compatible `/v1/embeddings`)
- **Impact**: requires Qdrant re-embed.
- **Why this wins**: strong benchmark performance; production-grade inference server.

### Secondary (Gemini, avoids migration): Gemini embedder with output_dimensionality=1536
Best option if you want no Qdrant migration and can accept external API usage.
- **Model**: `models/gemini-embedding-001` (supports 1536 via output_dimensionality)
- **Output dims**: set to 1536 (Gemini defaults to 768 if unset)
- **Impact**: minimal (no collection change).
- **Why this wins**: no re-embed; good quality; meets Gemini requirement.

## Suggested Config (Memory Pipeline)

### Local Ollama (Nomic)
```json
{
  "embedder": {
    "provider": "ollama",
    "config": {
      "model": "nomic-embed-text",
      "ollama_base_url": "http://ollama:11434",
      "embedding_dims": 768
    }
  }
}
```

### Local TEI (BGE-M3)
```json
{
  "embedder": {
    "provider": "huggingface",
    "config": {
      "huggingface_base_url": "http://tei:80/v1",
      "embedding_dims": 1024
    }
  }
}
```

### Gemini (no migration if dims stay 1536)
```json
{
  "embedder": {
    "provider": "gemini",
    "config": {
      "model": "models/gemini-embedding-001",
      "api_key": "env:GOOGLE_API_KEY",
      "embedding_dims": 1536
    }
  }
}
```
Note: If `embedding_dims` is not set for Gemini, the runtime defaults to 768 and will not match the existing 1536-dim vector store.
Note: `models/text-embedding-004` currently returns 768 even when `output_dimensionality` is set.

## Migration Impact
- If you switch to 768 or 1024 dims, **Qdrant must be rebuilt** (new collection + re-embed).
- If you use Gemini with 1536 dims, **no migration required**.

## Decision Summary
- If local-only is mandatory: **Ollama + Nomic Embed v1.5** is the best default.
- If highest quality is the goal and you can run heavier infra: **TEI + BGE-M3**.
- If you want no reindex and can use Gemini API: **Gemini embeddings with 1536 dims**.

## Next Actions
1) Pick between local vs Gemini based on compliance and migration budget.
2) Stand up Ollama or TEI if local.
3) Update embedder config (UI or API) and set `embedding_dims` to 1536 for Gemini.
4) Re-embed memories if dimensions change.
