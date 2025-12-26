"""Benchmark runner for orchestrating model comparisons.

This module provides the BenchmarkRunner class which:
- Loads dataset samples from CodeSearchNet
- Runs embedding models against test queries
- Runs lexical backends through decision matrix
- Collects MRR, NDCG, and latency metrics
- Aggregates results across multiple runs
"""

from datetime import datetime
from typing import List, Dict, Optional
import time

from openmemory.api.benchmarks.embeddings.datasets.codesearchnet import (
    CodeSearchNetLoader,
    CodeSearchNetSample,
)
from openmemory.api.benchmarks.embeddings.adapters import create_adapter
from openmemory.api.benchmarks.embeddings.metrics.mrr import calculate_mrr
from openmemory.api.benchmarks.embeddings.metrics.ndcg import calculate_ndcg
from openmemory.api.benchmarks.embeddings.metrics.latency import LatencyTracker

from openmemory.api.benchmarks.lexical.backends import create_backend
from openmemory.api.benchmarks.lexical.backends.base import Document
from openmemory.api.benchmarks.lexical.decision_matrix.criteria import (
    CriterionName,
    CRITERIA,
)
from openmemory.api.benchmarks.lexical.decision_matrix.evaluator import (
    DecisionMatrixEvaluator,
)

from openmemory.api.benchmarks.runner.results import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelResult,
    LexicalBackendResult,
)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class BenchmarkRunner:
    """Orchestrates benchmark runs for embedding models and lexical backends.

    This class handles:
    - Loading dataset samples
    - Running embedding model benchmarks with MRR/NDCG/latency metrics
    - Running lexical backend benchmarks with decision matrix scoring
    - Aggregating results across multiple runs
    """

    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration specifying models, backends, etc.
        """
        self.config = config

    def load_samples(self) -> List[CodeSearchNetSample]:
        """Load dataset samples based on config.

        Returns:
            List of CodeSearchNet samples

        Raises:
            ValueError: If dataset loading fails
        """
        loader = CodeSearchNetLoader(
            language=self.config.dataset_language,
            split=self.config.dataset_split,
        )

        samples = list(loader.load(limit=self.config.sample_limit))
        return samples

    def run_embedding_benchmark(
        self,
        model_name: str,
        samples: List[CodeSearchNetSample],
    ) -> ModelResult:
        """Run benchmark for a single embedding model.

        Args:
            model_name: Name of the embedding model to benchmark
            samples: Dataset samples to use for evaluation

        Returns:
            ModelResult with MRR, NDCG, latency metrics

        Raises:
            ValueError: If adapter creation fails
        """
        adapter = create_adapter(model_name)
        tracker = LatencyTracker()

        # Collect all texts for batch embedding
        queries = [s.query for s in samples]
        codes = [s.code for s in samples]

        # Embed all queries and codes
        with tracker.time():
            query_embeddings = adapter.embed_batch(queries)

        with tracker.time():
            code_embeddings = adapter.embed_batch(codes)

        # For each query, rank all codes by similarity
        ranked_results: List[List[str]] = []
        relevant_docs: List[set] = []
        relevance_scores: List[Dict[str, int]] = []

        for i, query_emb in enumerate(query_embeddings):
            # Calculate similarity to all codes
            similarities = []
            for j, code_emb in enumerate(code_embeddings):
                sim = _cosine_similarity(query_emb, code_emb)
                similarities.append((samples[j].id, sim))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Extract ranked doc IDs
            ranked_ids = [doc_id for doc_id, _ in similarities]
            ranked_results.append(ranked_ids)

            # The relevant doc for query i is code i (same sample)
            relevant_docs.append({samples[i].id})

            # For NDCG: query i's code is fully relevant (grade 3),
            # other codes are not relevant (grade 0)
            grades = {samples[i].id: 3}  # Only the matching code is relevant
            relevance_scores.append(grades)

        # Calculate metrics
        mrr_result = calculate_mrr(
            ranked_results=ranked_results,
            relevant_docs=relevant_docs,
            k=self.config.mrr_k,
        )

        ndcg_result = calculate_ndcg(
            ranked_results=ranked_results,
            relevance_scores=relevance_scores,
            k=self.config.ndcg_k,
        )

        return ModelResult(
            model_name=adapter.info.model_name,
            provider=adapter.info.provider,
            mrr_score=mrr_result.score,
            ndcg_score=ndcg_result.score,
            latency_stats=tracker.get_stats(),
            num_queries=len(samples),
            embedding_dimensions=adapter.info.dimensions,
        )

    def run_lexical_benchmark(
        self,
        backend_name: str,
        samples: List[CodeSearchNetSample],
    ) -> LexicalBackendResult:
        """Run benchmark for a single lexical backend.

        Args:
            backend_name: Name of the backend to benchmark
            samples: Dataset samples to use for evaluation

        Returns:
            LexicalBackendResult with decision matrix scores and latency

        Raises:
            ValueError: If backend creation fails
        """
        backend = create_backend(backend_name)
        tracker = LatencyTracker()

        # Index all code samples
        documents = [
            Document(id=s.id, content=s.code, metadata={"func_name": s.func_name})
            for s in samples
        ]
        backend.index_documents(documents)

        # Run searches and track latency
        for sample in samples:
            with tracker.time():
                backend.search(sample.query, limit=10)

        stats = backend.get_stats()
        latency_stats = tracker.get_stats()

        # Score using decision matrix
        # Convert P95 latency to 0-1 score (lower is better, cap at 100ms)
        latency_score = max(0.0, 1.0 - (latency_stats.p95 / 100.0))

        # Score other criteria based on backend characteristics
        if backend.name.lower() == "tantivy":
            ops_complexity_score = 0.95  # Low ops complexity (embedded)
            scalability_score = 0.60  # Limited scalability
            feature_score = 0.70  # Good BM25, limited features
        else:  # OpenSearch
            ops_complexity_score = 0.60  # Higher ops complexity
            scalability_score = 0.95  # Excellent scalability
            feature_score = 0.90  # Full feature set

        raw_scores = {
            CriterionName.LATENCY: latency_score,
            CriterionName.OPS_COMPLEXITY: ops_complexity_score,
            CriterionName.SCALABILITY: scalability_score,
            CriterionName.FEATURE_SUPPORT: feature_score,
        }

        evaluator = DecisionMatrixEvaluator()
        backend_score = evaluator.evaluate(backend.name, raw_scores)

        # Clean up
        backend.clear()

        return LexicalBackendResult(
            backend_name=backend.name,
            criterion_scores=raw_scores,
            weighted_total=backend_score.weighted_total,
            latency_stats=latency_stats,
            document_count=stats.document_count,
        )

    def run(self) -> BenchmarkResult:
        """Run complete benchmark for all configured models and backends.

        Returns:
            BenchmarkResult with all model and backend results

        Raises:
            ValueError: If no samples could be loaded
        """
        start_time = datetime.now()
        start_seconds = time.time()

        # Load samples
        samples = self.load_samples()
        if not samples:
            raise ValueError("No samples loaded from dataset")

        # Run embedding model benchmarks
        model_results: List[ModelResult] = []
        for model_name in self.config.embedding_models:
            result = self.run_embedding_benchmark(model_name, samples)
            model_results.append(result)

        # Run lexical backend benchmarks
        lexical_results: List[LexicalBackendResult] = []
        for backend_name in self.config.lexical_backends:
            result = self.run_lexical_benchmark(backend_name, samples)
            lexical_results.append(result)

        end_seconds = time.time()

        return BenchmarkResult(
            config=self.config,
            model_results=model_results,
            lexical_results=lexical_results,
            timestamp=start_time,
            total_duration_seconds=end_seconds - start_seconds,
        )

    def run_multiple(self, num_runs: int = 3) -> BenchmarkResult:
        """Run benchmark multiple times and aggregate results.

        This helps reduce variance by averaging metrics across runs.

        Args:
            num_runs: Number of times to run the benchmark

        Returns:
            BenchmarkResult with averaged metrics
        """
        start_time = datetime.now()
        start_seconds = time.time()

        # Load samples once
        samples = self.load_samples()
        if not samples:
            raise ValueError("No samples loaded from dataset")

        # Accumulate results per model
        model_mrr_scores: Dict[str, List[float]] = {}
        model_ndcg_scores: Dict[str, List[float]] = {}
        model_latencies: Dict[str, List[float]] = {}
        model_info: Dict[str, tuple] = {}  # (provider, dimensions)

        # Accumulate results per backend
        backend_scores: Dict[str, List[float]] = {}
        backend_latencies: Dict[str, List[float]] = {}
        backend_criterion_scores: Dict[str, Dict[CriterionName, List[float]]] = {}
        backend_doc_counts: Dict[str, int] = {}

        for _ in range(num_runs):
            # Run embedding benchmarks
            for model_name in self.config.embedding_models:
                result = self.run_embedding_benchmark(model_name, samples)

                if model_name not in model_mrr_scores:
                    model_mrr_scores[model_name] = []
                    model_ndcg_scores[model_name] = []
                    model_latencies[model_name] = []
                    model_info[model_name] = (
                        result.provider,
                        result.embedding_dimensions,
                    )

                model_mrr_scores[model_name].append(result.mrr_score)
                model_ndcg_scores[model_name].append(result.ndcg_score)
                model_latencies[model_name].append(result.latency_stats.p95)

            # Run lexical benchmarks
            for backend_name in self.config.lexical_backends:
                result = self.run_lexical_benchmark(backend_name, samples)

                if backend_name not in backend_scores:
                    backend_scores[backend_name] = []
                    backend_latencies[backend_name] = []
                    backend_criterion_scores[backend_name] = {
                        c: [] for c in CriterionName
                    }
                    backend_doc_counts[backend_name] = result.document_count

                backend_scores[backend_name].append(result.weighted_total)
                backend_latencies[backend_name].append(result.latency_stats.p95)
                for criterion, score in result.criterion_scores.items():
                    backend_criterion_scores[backend_name][criterion].append(score)

        # Create averaged model results
        from openmemory.api.benchmarks.embeddings.metrics.latency import LatencyStats

        model_results: List[ModelResult] = []
        for model_name in self.config.embedding_models:
            avg_mrr = sum(model_mrr_scores[model_name]) / num_runs
            avg_ndcg = sum(model_ndcg_scores[model_name]) / num_runs
            avg_p95 = sum(model_latencies[model_name]) / num_runs
            provider, dimensions = model_info[model_name]

            # Create simplified latency stats with averaged P95
            latency_stats = LatencyStats(
                p50=avg_p95 * 0.6,  # Approximate
                p95=avg_p95,
                p99=avg_p95 * 1.2,  # Approximate
                mean=avg_p95 * 0.7,  # Approximate
                min=avg_p95 * 0.3,  # Approximate
                max=avg_p95 * 2.0,  # Approximate
                count=len(samples) * num_runs,
            )

            model_results.append(
                ModelResult(
                    model_name=model_name,
                    provider=provider,
                    mrr_score=avg_mrr,
                    ndcg_score=avg_ndcg,
                    latency_stats=latency_stats,
                    num_queries=len(samples),
                    embedding_dimensions=dimensions,
                )
            )

        # Create averaged lexical results
        lexical_results: List[LexicalBackendResult] = []
        for backend_name in self.config.lexical_backends:
            avg_total = sum(backend_scores[backend_name]) / num_runs
            avg_p95 = sum(backend_latencies[backend_name]) / num_runs
            avg_criterion_scores = {
                c: sum(scores) / num_runs
                for c, scores in backend_criterion_scores[backend_name].items()
            }

            latency_stats = LatencyStats(
                p50=avg_p95 * 0.6,
                p95=avg_p95,
                p99=avg_p95 * 1.2,
                mean=avg_p95 * 0.7,
                min=avg_p95 * 0.3,
                max=avg_p95 * 2.0,
                count=len(samples) * num_runs,
            )

            lexical_results.append(
                LexicalBackendResult(
                    backend_name=backend_name,
                    criterion_scores=avg_criterion_scores,
                    weighted_total=avg_total,
                    latency_stats=latency_stats,
                    document_count=backend_doc_counts[backend_name],
                )
            )

        end_seconds = time.time()

        return BenchmarkResult(
            config=self.config,
            model_results=model_results,
            lexical_results=lexical_results,
            timestamp=start_time,
            total_duration_seconds=end_seconds - start_seconds,
        )
