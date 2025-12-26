"""Result dataclasses for benchmark runner.

These dataclasses store the results of benchmark runs including:
- Per-model embedding metrics (MRR, NDCG, latency)
- Per-backend lexical search metrics
- Aggregated benchmark results
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any

from openmemory.api.benchmarks.embeddings.metrics.latency import LatencyStats
from openmemory.api.benchmarks.lexical.decision_matrix.criteria import CriterionName


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        dataset_name: Name of the dataset (e.g., "codesearchnet")
        dataset_language: Programming language for dataset (e.g., "python")
        dataset_split: Dataset split to use (e.g., "test", "valid", "train")
        sample_limit: Maximum number of samples to load (None for all)
        embedding_models: List of embedding model names to benchmark
        lexical_backends: List of lexical backend names to benchmark
        mrr_k: Cutoff for MRR@k calculation
        ndcg_k: Cutoff for NDCG@k calculation
    """

    dataset_name: str
    embedding_models: List[str]
    dataset_language: str = "python"
    dataset_split: str = "test"
    sample_limit: Optional[int] = None
    lexical_backends: List[str] = field(default_factory=list)
    mrr_k: int = 10
    ndcg_k: int = 10


@dataclass
class ModelResult:
    """Result of benchmarking a single embedding model.

    Attributes:
        model_name: Name of the embedding model
        provider: Provider of the model (e.g., "ollama", "gemini")
        mrr_score: Mean Reciprocal Rank score (0-1)
        ndcg_score: NDCG@k score (0-1)
        latency_stats: Latency statistics for embedding operations
        num_queries: Number of queries evaluated
        embedding_dimensions: Dimension of embeddings produced
    """

    model_name: str
    provider: str
    mrr_score: float
    ndcg_score: float
    latency_stats: LatencyStats
    num_queries: int
    embedding_dimensions: int

    def meets_mrr_threshold(self, threshold: float) -> bool:
        """Check if model meets MRR threshold.

        Args:
            threshold: Minimum MRR score required (e.g., 0.75)

        Returns:
            True if mrr_score >= threshold
        """
        return self.mrr_score >= threshold

    def meets_ndcg_threshold(self, threshold: float) -> bool:
        """Check if model meets NDCG threshold.

        Args:
            threshold: Minimum NDCG score required (e.g., 0.80)

        Returns:
            True if ndcg_score >= threshold
        """
        return self.ndcg_score >= threshold

    def is_production_ready(
        self, mrr_threshold: float = 0.75, ndcg_threshold: float = 0.80
    ) -> bool:
        """Check if model meets production readiness thresholds.

        Args:
            mrr_threshold: Minimum MRR score (default 0.75 per v7 plan)
            ndcg_threshold: Minimum NDCG score (default 0.80 per v7 plan)

        Returns:
            True if model meets both thresholds
        """
        return self.meets_mrr_threshold(mrr_threshold) and self.meets_ndcg_threshold(
            ndcg_threshold
        )


@dataclass
class LexicalBackendResult:
    """Result of benchmarking a single lexical backend.

    Attributes:
        backend_name: Name of the backend (e.g., "tantivy", "opensearch")
        criterion_scores: Scores per decision matrix criterion (0-1)
        weighted_total: Weighted total score from decision matrix
        latency_stats: Search latency statistics
        document_count: Number of documents indexed
    """

    backend_name: str
    criterion_scores: Dict[CriterionName, float]
    weighted_total: float
    latency_stats: LatencyStats
    document_count: int


@dataclass
class BenchmarkResult:
    """Aggregated result of a complete benchmark run.

    Attributes:
        config: The benchmark configuration used
        model_results: Results for each embedding model
        lexical_results: Results for each lexical backend
        timestamp: When the benchmark started
        total_duration_seconds: Total time taken for benchmark
    """

    config: BenchmarkConfig
    model_results: List[ModelResult]
    lexical_results: List[LexicalBackendResult]
    timestamp: datetime
    total_duration_seconds: float

    def get_best_model(self, metric: str = "mrr") -> Optional[ModelResult]:
        """Get the best performing embedding model.

        Args:
            metric: Metric to use for comparison ("mrr" or "ndcg")

        Returns:
            ModelResult with highest score, or None if no results
        """
        if not self.model_results:
            return None

        if metric == "mrr":
            return max(self.model_results, key=lambda r: r.mrr_score)
        elif metric == "ndcg":
            return max(self.model_results, key=lambda r: r.ndcg_score)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'mrr' or 'ndcg'.")

    def get_best_lexical_backend(self) -> Optional[LexicalBackendResult]:
        """Get the best performing lexical backend.

        Returns:
            LexicalBackendResult with highest weighted_total, or None if no results
        """
        if not self.lexical_results:
            return None

        return max(self.lexical_results, key=lambda r: r.weighted_total)

    def get_production_ready_models(
        self, mrr_threshold: float = 0.75, ndcg_threshold: float = 0.80
    ) -> List[ModelResult]:
        """Get all models that meet production thresholds.

        Args:
            mrr_threshold: Minimum MRR score (default 0.75)
            ndcg_threshold: Minimum NDCG score (default 0.80)

        Returns:
            List of ModelResult that meet both thresholds
        """
        return [
            r
            for r in self.model_results
            if r.is_production_ready(mrr_threshold, ndcg_threshold)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the benchmark result
        """
        return {
            "config": {
                "dataset_name": self.config.dataset_name,
                "dataset_language": self.config.dataset_language,
                "dataset_split": self.config.dataset_split,
                "sample_limit": self.config.sample_limit,
                "embedding_models": self.config.embedding_models,
                "lexical_backends": self.config.lexical_backends,
                "mrr_k": self.config.mrr_k,
                "ndcg_k": self.config.ndcg_k,
            },
            "model_results": [
                {
                    "model_name": r.model_name,
                    "provider": r.provider,
                    "mrr_score": r.mrr_score,
                    "ndcg_score": r.ndcg_score,
                    "latency_stats": {
                        "p50": r.latency_stats.p50,
                        "p95": r.latency_stats.p95,
                        "p99": r.latency_stats.p99,
                        "mean": r.latency_stats.mean,
                        "min": r.latency_stats.min,
                        "max": r.latency_stats.max,
                        "count": r.latency_stats.count,
                    },
                    "num_queries": r.num_queries,
                    "embedding_dimensions": r.embedding_dimensions,
                }
                for r in self.model_results
            ],
            "lexical_results": [
                {
                    "backend_name": r.backend_name,
                    "criterion_scores": {
                        k.value: v for k, v in r.criterion_scores.items()
                    },
                    "weighted_total": r.weighted_total,
                    "latency_stats": {
                        "p50": r.latency_stats.p50,
                        "p95": r.latency_stats.p95,
                        "p99": r.latency_stats.p99,
                        "mean": r.latency_stats.mean,
                        "min": r.latency_stats.min,
                        "max": r.latency_stats.max,
                        "count": r.latency_stats.count,
                    },
                    "document_count": r.document_count,
                }
                for r in self.lexical_results
            ],
            "timestamp": self.timestamp.isoformat(),
            "total_duration_seconds": self.total_duration_seconds,
        }
