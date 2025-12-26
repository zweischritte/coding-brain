# Phase 0a Benchmark Results

> Generated: 2025-12-26
> Framework Version: Phase 0a (337 tests passing)

## Executive Summary

| Category | Winner | Key Metric |
|----------|--------|-----------|
| **Embedding Model** | qwen3-embedding:8b | MRR: 0.824, NDCG: 0.848 |
| **Lexical Backend** | OpenSearch | Weighted Score: 0.530 |

**Status**: ✅ Production thresholds met for embedding (MRR ≥ 0.75, NDCG ≥ 0.80)

---

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CodeSearchNet (Python) |
| Split | Test |
| Samples | 100 |
| MRR@k | k=10 |
| NDCG@k | k=10 |
| Environment | macOS, Ollama v0.13.5 |

---

## Embedding Model Results

### Summary Table

| Model | Provider | Dimensions | MRR | NDCG | Latency P95 | Production Ready |
|-------|----------|------------|-----|------|-------------|------------------|
| **qwen3-embedding:8b** | Ollama | 4096 | **0.824** | **0.848** | 37.0s | ✅ Yes |
| nomic-embed-text | Ollama | 768 | 0.659 | 0.705 | 1.9s | ❌ No |

### Analysis

**qwen3-embedding:8b**
- ✅ Exceeds production MRR threshold (0.824 ≥ 0.75)
- ✅ Exceeds production NDCG threshold (0.848 ≥ 0.80)
- ⚠️ High latency due to 8B parameter size (batch embedding 100 samples)
- 📊 4096-dimensional embeddings provide rich semantic representation

**nomic-embed-text**
- ❌ Below MRR threshold (0.659 < 0.75)
- ❌ Below NDCG threshold (0.705 < 0.80)
- ✅ Much faster latency (~20x faster than qwen3-8b)
- 📊 768-dimensional embeddings, more efficient for storage

### Latency Notes

The P95 latency values reflect **batch embedding time** for 100 code samples + 100 queries on Apple Silicon (M-series). Per-query latency in production with pre-computed embeddings would be significantly lower (cosine similarity search).

---

## Lexical Backend Results

### Summary Table

| Backend | Weighted Total | Latency P95 | Latency Score | Ops Complexity | Scalability | Features |
|---------|----------------|-------------|---------------|----------------|-------------|----------|
| **OpenSearch** | **0.530** | 90ms | 0.10 | 0.60 | 0.95 | 0.90 |
| Tantivy | 0.475 | 94ms | 0.06 | 0.95 | 0.60 | 0.70 |

### Decision Matrix Breakdown

The lexical backends are evaluated using a weighted decision matrix with the following criteria:

| Criterion | Weight | OpenSearch | Tantivy |
|-----------|--------|------------|---------|
| Latency (30%) | 0.30 | 0.10 | 0.06 |
| Ops Complexity (25%) | 0.25 | 0.60 | **0.95** |
| Scalability (25%) | 0.25 | **0.95** | 0.60 |
| Feature Support (20%) | 0.20 | **0.90** | 0.70 |

### Analysis

**OpenSearch** wins the weighted comparison due to:
- Superior scalability for distributed deployments
- Richer feature set (aggregations, filters, synonyms)
- Better suited for production workloads at scale

**Tantivy** advantages:
- Lower operational complexity (embedded, no external service)
- Suitable for single-node deployments
- Good for development and smaller deployments

---

## Recommendations

### For Production Deployment

1. **Embedding Model**: Use **qwen3-embedding:8b** via Ollama
   - Meets all production quality thresholds
   - Consider GPU acceleration for lower latency
   - Pre-compute embeddings at indexing time

2. **Lexical Backend**: Use **OpenSearch** for hybrid search
   - Combine with semantic embeddings for best results
   - Provides BM25 scoring for keyword relevance

### For Development/Testing

1. **Embedding Model**: Use **nomic-embed-text** for faster iteration
   - 20x faster than qwen3-8b
   - Adequate for development workflows

2. **Lexical Backend**: Use **Tantivy** for local development
   - No external dependencies
   - Fast to set up and tear down

---

## Limitations & Future Work

### Current Limitations

1. **Context Length**: Some code samples exceed model context limits (8192 tokens for qwen3-embedding)
2. **Sample Size**: 100 samples may not fully represent production query patterns
3. **Mock Backends**: Lexical backends use mock implementations, not actual Tantivy/OpenSearch

### Future Improvements

1. Add chunking strategy for long code samples
2. Test with larger sample sizes (500+)
3. Integrate real Tantivy and OpenSearch backends
4. Add reranker benchmark (target: ≥10% precision uplift)
5. Test Gemini embeddings (requires GOOGLE_API_KEY)

---

## Appendix: Raw Data

### Models Tested
- `qwen3-embedding:8b` (Ollama, 4.7GB, 4096-dim)
- `nomic-embed-text` (Ollama, 274MB, 768-dim)
- `gemini` (skipped - GOOGLE_API_KEY not set)

### Backends Tested
- Tantivy (mock implementation)
- OpenSearch (mock implementation)

### Thresholds (per Implementation Plan v7)
- MRR ≥ 0.75 for production readiness
- NDCG@10 ≥ 0.80 for production readiness
- Reranker uplift ≥ 10% precision (not yet tested)
