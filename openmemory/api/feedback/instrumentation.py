"""Retrieval instrumentation and evaluation harness.

Implements FR-002: Retrieval quality instrumentation requirements:
- Per-stage latency tracking (embedding, lexical, graph, fusion, rerank)
- Evaluation harness for retrieval quality (MRR/NDCG + regression checks)
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


class StageType(Enum):
    """Types of retrieval stages."""

    EMBEDDING = "embedding"  # Embedding generation
    LEXICAL = "lexical"  # Lexical/BM25 search
    VECTOR = "vector"  # Vector/kNN search
    GRAPH = "graph"  # Graph traversal
    FUSION = "fusion"  # RRF or weighted fusion
    RERANK = "rerank"  # Reranking stage


@dataclass
class StageLatency:
    """Latency measurement for a single stage.

    Attributes:
        stage: Type of stage
        duration_ms: Duration in milliseconds
        success: Whether stage completed successfully
        error: Error message if failed
        metadata: Additional stage-specific data
    """

    stage: StageType
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage": self.stage.value,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalMetrics:
    """Complete metrics for a single retrieval query.

    Attributes:
        query_id: Unique identifier for the query
        total_duration_ms: Total wall-clock time
        stages: List of stage latencies
        result_count: Number of results returned
        reranker_used: Whether reranker was applied
        timestamp: When the query was executed
    """

    query_id: str
    total_duration_ms: float
    stages: list[StageLatency]
    result_count: Optional[int] = None
    reranker_used: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def stage_total_ms(self) -> float:
        """Sum of all stage durations."""
        return sum(s.duration_ms for s in self.stages)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query_id": self.query_id,
            "total_duration_ms": self.total_duration_ms,
            "stage_total_ms": self.stage_total_ms,
            "stages": [s.to_dict() for s in self.stages],
            "result_count": self.result_count,
            "reranker_used": self.reranker_used,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class _QueryState:
    """Internal state for tracking an in-progress query."""

    start_time: float
    stages: list[StageLatency] = field(default_factory=list)


class RetrievalInstrumentation:
    """Instrumentation for retrieval performance tracking.

    Tracks per-stage latencies and provides aggregate statistics
    for monitoring and optimization.
    """

    def __init__(self):
        """Initialize instrumentation."""
        self._active_queries: dict[str, _QueryState] = {}
        self._completed_queries: list[RetrievalMetrics] = []

    def start_query(self, query_id: str) -> str:
        """Start tracking a query.

        Args:
            query_id: Unique identifier for the query

        Returns:
            The query_id
        """
        self._active_queries[query_id] = _QueryState(start_time=time.perf_counter())
        return query_id

    def record_stage(
        self,
        query_id: str,
        stage: StageType,
        duration_ms: float,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a stage completion.

        Args:
            query_id: Query identifier
            stage: Type of stage
            duration_ms: Duration in milliseconds
            success: Whether stage succeeded
            error: Error message if failed
            metadata: Additional data
        """
        if query_id not in self._active_queries:
            logger.warning(f"Recording stage for unknown query: {query_id}")
            return

        latency = StageLatency(
            stage=stage,
            duration_ms=duration_ms,
            success=success,
            error=error,
            metadata=metadata or {},
        )
        self._active_queries[query_id].stages.append(latency)

    @contextmanager
    def stage(
        self,
        query_id: str,
        stage: StageType,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Generator[None, None, None]:
        """Context manager for timing a stage.

        Usage:
            with instrumentation.stage("q123", StageType.EMBEDDING):
                embedding = embed(query)

        Args:
            query_id: Query identifier
            stage: Type of stage
            metadata: Additional data
        """
        start = time.perf_counter()
        error = None
        success = True

        try:
            yield
        except Exception as e:
            error = str(e)
            success = False
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record_stage(
                query_id=query_id,
                stage=stage,
                duration_ms=duration_ms,
                success=success,
                error=error,
                metadata=metadata,
            )

    def finish_query(
        self,
        query_id: str,
        result_count: Optional[int] = None,
        reranker_used: bool = False,
    ) -> RetrievalMetrics:
        """Finish tracking a query and get metrics.

        Args:
            query_id: Query identifier
            result_count: Number of results returned
            reranker_used: Whether reranker was used

        Returns:
            RetrievalMetrics for the query
        """
        if query_id not in self._active_queries:
            raise ValueError(f"Unknown query: {query_id}")

        state = self._active_queries.pop(query_id)
        total_duration_ms = (time.perf_counter() - state.start_time) * 1000

        metrics = RetrievalMetrics(
            query_id=query_id,
            total_duration_ms=total_duration_ms,
            stages=state.stages,
            result_count=result_count,
            reranker_used=reranker_used,
        )
        self._completed_queries.append(metrics)

        return metrics

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all completed queries.

        Returns:
            Dict with query_count, stage_stats (mean, p50, p95 per stage)
        """
        if not self._completed_queries:
            return {"query_count": 0, "stage_stats": {}}

        # Collect latencies by stage
        stage_latencies: dict[str, list[float]] = {}
        for metrics in self._completed_queries:
            for stage in metrics.stages:
                key = stage.stage.value
                if key not in stage_latencies:
                    stage_latencies[key] = []
                stage_latencies[key].append(stage.duration_ms)

        # Calculate stats per stage
        stage_stats = {}
        for stage_name, latencies in stage_latencies.items():
            sorted_latencies = sorted(latencies)
            stage_stats[stage_name] = {
                "count": len(latencies),
                "mean": statistics.mean(latencies),
                "p50": self._percentile(sorted_latencies, 50),
                "p95": self._percentile(sorted_latencies, 95),
                "p99": self._percentile(sorted_latencies, 99),
                "min": min(latencies),
                "max": max(latencies),
            }

        return {
            "query_count": len(self._completed_queries),
            "stage_stats": stage_stats,
        }

    def _percentile(self, sorted_data: list[float], percentile: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

    def clear(self) -> None:
        """Clear all stored metrics (for testing)."""
        self._active_queries.clear()
        self._completed_queries.clear()


# =============================================================================
# Evaluation Harness
# =============================================================================


@dataclass
class QueryExecution:
    """Record of a query execution for evaluation.

    Attributes:
        query_id: Unique identifier
        query_text: The query text
        expected_results: Ground truth relevant document IDs
        actual_results: Returned document IDs in order
        expected_relevance: Optional relevance scores per document
    """

    query_id: str
    query_text: str
    expected_results: list[str]
    actual_results: list[str]
    expected_relevance: Optional[dict[str, int]] = None


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics.

    Attributes:
        k_values: Values of k for NDCG@k
        regression_threshold: Threshold for regression detection
    """

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])
    regression_threshold: float = 0.02  # 2% regression


@dataclass
class EvaluationResult:
    """Result of an evaluation run.

    Attributes:
        mrr: Mean Reciprocal Rank
        ndcg: NDCG@k for each k
        passed: Whether evaluation passed thresholds
        regressions: List of detected regressions
    """

    mrr: float
    ndcg: dict[int, float]
    passed: bool
    regressions: list[str] = field(default_factory=list)


class EvaluationHarness:
    """Evaluation harness for retrieval quality.

    Calculates MRR, NDCG, and detects regressions from baseline.
    """

    def __init__(self, config: Optional[MetricConfig] = None):
        """Initialize harness.

        Args:
            config: Optional configuration
        """
        self._config = config or MetricConfig()

    def calculate_mrr(self, executions: list[QueryExecution]) -> float:
        """Calculate Mean Reciprocal Rank.

        MRR = (1/|Q|) * sum(1/rank_i) for first relevant result.

        Args:
            executions: List of query executions

        Returns:
            MRR score (0.0 to 1.0)
        """
        if not executions:
            return 0.0

        total_rr = 0.0
        for exec in executions:
            expected_set = set(exec.expected_results)
            for i, doc_id in enumerate(exec.actual_results, start=1):
                if doc_id in expected_set:
                    total_rr += 1.0 / i
                    break

        return total_rr / len(executions)

    def calculate_ndcg(
        self,
        executions: list[QueryExecution],
        k: int = 10,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        NDCG@k = DCG@k / IDCG@k

        Args:
            executions: List of query executions
            k: Cutoff for evaluation

        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if not executions:
            return 0.0

        total_ndcg = 0.0
        for exec in executions:
            # Get relevance scores (default to 1 for expected, 0 for others)
            relevance = exec.expected_relevance or {
                doc_id: 1 for doc_id in exec.expected_results
            }

            # Calculate DCG
            dcg = 0.0
            for i, doc_id in enumerate(exec.actual_results[:k], start=1):
                rel = relevance.get(doc_id, 0)
                dcg += (2**rel - 1) / math.log2(i + 1)

            # Calculate ideal DCG
            ideal_rels = sorted(relevance.values(), reverse=True)[:k]
            idcg = 0.0
            for i, rel in enumerate(ideal_rels, start=1):
                idcg += (2**rel - 1) / math.log2(i + 1)

            # NDCG for this query
            if idcg > 0:
                total_ndcg += dcg / idcg

        return total_ndcg / len(executions)

    def evaluate(
        self,
        executions: list[QueryExecution],
        baseline: Optional[EvaluationResult] = None,
    ) -> EvaluationResult:
        """Run full evaluation.

        Args:
            executions: Query executions to evaluate
            baseline: Optional baseline to compare against

        Returns:
            EvaluationResult with metrics and regression info
        """
        mrr = self.calculate_mrr(executions)
        ndcg = {k: self.calculate_ndcg(executions, k) for k in self._config.k_values}

        regressions = []
        passed = True

        if baseline:
            # Check MRR regression
            mrr_diff = baseline.mrr - mrr
            if mrr_diff > self._config.regression_threshold:
                regressions.append(
                    f"MRR dropped from {baseline.mrr:.3f} to {mrr:.3f}"
                )
                passed = False

            # Check NDCG regressions
            for k, score in ndcg.items():
                if k in baseline.ndcg:
                    diff = baseline.ndcg[k] - score
                    if diff > self._config.regression_threshold:
                        regressions.append(
                            f"NDCG@{k} dropped from {baseline.ndcg[k]:.3f} to {score:.3f}"
                        )
                        passed = False

        return EvaluationResult(
            mrr=mrr,
            ndcg=ndcg,
            passed=passed,
            regressions=regressions,
        )
