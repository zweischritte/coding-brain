"""Tests for retrieval instrumentation and evaluation harness.

TDD tests for per-stage latency tracking and evaluation harness following v9 plan:
- Per-stage retrieval instrumentation (embedding, lexical, graph, fusion, rerank)
- Evaluation harness for retrieval quality (MRR/NDCG + regression checks)
- Integration with feedback events
"""

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmemory.api.feedback.instrumentation import (
    EvaluationHarness,
    EvaluationResult,
    MetricConfig,
    QueryExecution,
    RetrievalInstrumentation,
    RetrievalMetrics,
    StageLatency,
    StageType,
)


class TestStageType:
    """Tests for StageType enum."""

    def test_has_embedding_stage(self):
        """Has embedding stage."""
        assert StageType.EMBEDDING.value == "embedding"

    def test_has_lexical_stage(self):
        """Has lexical stage."""
        assert StageType.LEXICAL.value == "lexical"

    def test_has_vector_stage(self):
        """Has vector stage."""
        assert StageType.VECTOR.value == "vector"

    def test_has_graph_stage(self):
        """Has graph stage."""
        assert StageType.GRAPH.value == "graph"

    def test_has_fusion_stage(self):
        """Has fusion stage."""
        assert StageType.FUSION.value == "fusion"

    def test_has_rerank_stage(self):
        """Has rerank stage."""
        assert StageType.RERANK.value == "rerank"


class TestStageLatency:
    """Tests for StageLatency dataclass."""

    def test_create_stage_latency(self):
        """Can create stage latency."""
        latency = StageLatency(
            stage=StageType.EMBEDDING,
            duration_ms=15.5,
            success=True,
        )
        assert latency.stage == StageType.EMBEDDING
        assert latency.duration_ms == 15.5
        assert latency.success is True

    def test_stage_latency_with_error(self):
        """Stage latency can track errors."""
        latency = StageLatency(
            stage=StageType.GRAPH,
            duration_ms=50.0,
            success=False,
            error="Connection timeout",
        )
        assert latency.success is False
        assert latency.error == "Connection timeout"

    def test_stage_latency_with_metadata(self):
        """Stage latency can include metadata."""
        latency = StageLatency(
            stage=StageType.VECTOR,
            duration_ms=25.0,
            success=True,
            metadata={"num_results": 50, "score_range": [0.7, 0.95]},
        )
        assert latency.metadata["num_results"] == 50


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics dataclass."""

    def test_create_metrics(self):
        """Can create retrieval metrics."""
        metrics = RetrievalMetrics(
            query_id="q123",
            total_duration_ms=75.5,
            stages=[
                StageLatency(stage=StageType.EMBEDDING, duration_ms=10.0, success=True),
                StageLatency(stage=StageType.VECTOR, duration_ms=25.0, success=True),
                StageLatency(stage=StageType.FUSION, duration_ms=5.0, success=True),
            ],
        )
        assert metrics.query_id == "q123"
        assert metrics.total_duration_ms == 75.5
        assert len(metrics.stages) == 3

    def test_metrics_calculates_stage_total(self):
        """Metrics can calculate sum of stage durations."""
        metrics = RetrievalMetrics(
            query_id="q123",
            total_duration_ms=100.0,
            stages=[
                StageLatency(stage=StageType.EMBEDDING, duration_ms=10.0, success=True),
                StageLatency(stage=StageType.VECTOR, duration_ms=25.0, success=True),
                StageLatency(stage=StageType.LEXICAL, duration_ms=20.0, success=True),
                StageLatency(stage=StageType.FUSION, duration_ms=5.0, success=True),
            ],
        )
        assert metrics.stage_total_ms == 60.0

    def test_metrics_with_result_count(self):
        """Metrics can track result count."""
        metrics = RetrievalMetrics(
            query_id="q123",
            total_duration_ms=50.0,
            stages=[],
            result_count=10,
        )
        assert metrics.result_count == 10

    def test_metrics_with_reranker_flag(self):
        """Metrics can indicate reranker usage."""
        metrics = RetrievalMetrics(
            query_id="q123",
            total_duration_ms=100.0,
            stages=[
                StageLatency(stage=StageType.RERANK, duration_ms=30.0, success=True),
            ],
            reranker_used=True,
        )
        assert metrics.reranker_used is True

    def test_metrics_to_dict(self):
        """Metrics can be serialized."""
        metrics = RetrievalMetrics(
            query_id="q123",
            total_duration_ms=50.0,
            stages=[
                StageLatency(stage=StageType.EMBEDDING, duration_ms=10.0, success=True),
            ],
        )
        d = metrics.to_dict()
        assert d["query_id"] == "q123"
        assert d["total_duration_ms"] == 50.0
        assert len(d["stages"]) == 1


class TestRetrievalInstrumentation:
    """Tests for RetrievalInstrumentation class."""

    @pytest.fixture
    def instrumentation(self) -> RetrievalInstrumentation:
        """Create instrumentation instance."""
        return RetrievalInstrumentation()

    def test_start_query(self, instrumentation: RetrievalInstrumentation):
        """Can start tracking a query."""
        query_id = instrumentation.start_query("q123")
        assert query_id == "q123"

    def test_record_stage(self, instrumentation: RetrievalInstrumentation):
        """Can record stage latency."""
        instrumentation.start_query("q123")
        instrumentation.record_stage(
            query_id="q123",
            stage=StageType.EMBEDDING,
            duration_ms=15.0,
            success=True,
        )

        metrics = instrumentation.finish_query("q123")
        assert len(metrics.stages) == 1
        assert metrics.stages[0].stage == StageType.EMBEDDING
        assert metrics.stages[0].duration_ms == 15.0

    def test_record_multiple_stages(self, instrumentation: RetrievalInstrumentation):
        """Can record multiple stages in order."""
        instrumentation.start_query("q123")
        instrumentation.record_stage("q123", StageType.EMBEDDING, 10.0, True)
        instrumentation.record_stage("q123", StageType.VECTOR, 25.0, True)
        instrumentation.record_stage("q123", StageType.LEXICAL, 20.0, True)
        instrumentation.record_stage("q123", StageType.FUSION, 5.0, True)

        metrics = instrumentation.finish_query("q123")
        assert len(metrics.stages) == 4
        stages = [s.stage for s in metrics.stages]
        assert stages == [
            StageType.EMBEDDING,
            StageType.VECTOR,
            StageType.LEXICAL,
            StageType.FUSION,
        ]

    def test_finish_query_calculates_total_duration(
        self, instrumentation: RetrievalInstrumentation
    ):
        """Finishing a query calculates total duration."""
        import time

        instrumentation.start_query("q123")
        time.sleep(0.01)  # 10ms
        metrics = instrumentation.finish_query("q123")

        assert metrics.total_duration_ms >= 10.0

    def test_context_manager_for_stage(self, instrumentation: RetrievalInstrumentation):
        """Can use context manager for stage timing."""
        import time

        instrumentation.start_query("q123")
        with instrumentation.stage("q123", StageType.EMBEDDING):
            time.sleep(0.01)  # 10ms

        metrics = instrumentation.finish_query("q123")
        assert len(metrics.stages) == 1
        assert metrics.stages[0].duration_ms >= 10.0
        assert metrics.stages[0].success is True

    def test_context_manager_captures_errors(
        self, instrumentation: RetrievalInstrumentation
    ):
        """Context manager captures errors."""
        instrumentation.start_query("q123")

        try:
            with instrumentation.stage("q123", StageType.GRAPH):
                raise ConnectionError("Neo4j unavailable")
        except ConnectionError:
            pass

        metrics = instrumentation.finish_query("q123")
        assert len(metrics.stages) == 1
        assert metrics.stages[0].success is False
        assert "Neo4j unavailable" in metrics.stages[0].error

    def test_get_aggregate_stats(self, instrumentation: RetrievalInstrumentation):
        """Can get aggregate statistics."""
        for i in range(10):
            instrumentation.start_query(f"q{i}")
            instrumentation.record_stage(f"q{i}", StageType.EMBEDDING, 10.0 + i, True)
            instrumentation.record_stage(f"q{i}", StageType.VECTOR, 20.0 + i, True)
            instrumentation.finish_query(f"q{i}")

        stats = instrumentation.get_aggregate_stats()

        assert stats["query_count"] == 10
        assert StageType.EMBEDDING.value in stats["stage_stats"]
        assert StageType.VECTOR.value in stats["stage_stats"]

        embedding_stats = stats["stage_stats"][StageType.EMBEDDING.value]
        assert embedding_stats["count"] == 10
        assert embedding_stats["mean"] >= 10.0
        assert embedding_stats["p50"] >= 10.0
        assert embedding_stats["p95"] >= 10.0

    def test_get_percentiles(self, instrumentation: RetrievalInstrumentation):
        """Can calculate percentile latencies."""
        for i in range(100):
            instrumentation.start_query(f"q{i}")
            instrumentation.record_stage(
                f"q{i}", StageType.EMBEDDING, float(i + 1), True
            )
            instrumentation.finish_query(f"q{i}")

        stats = instrumentation.get_aggregate_stats()
        embedding_stats = stats["stage_stats"][StageType.EMBEDDING.value]

        # p50 should be around 50, p95 around 95
        assert 45 <= embedding_stats["p50"] <= 55
        assert 90 <= embedding_stats["p95"] <= 100


class TestQueryExecution:
    """Tests for QueryExecution dataclass (evaluation harness input)."""

    def test_create_query_execution(self):
        """Can create query execution record."""
        execution = QueryExecution(
            query_id="q123",
            query_text="find login function",
            expected_results=["doc_a", "doc_b", "doc_c"],
            actual_results=["doc_b", "doc_a", "doc_d"],
        )
        assert execution.query_id == "q123"
        assert len(execution.expected_results) == 3
        assert len(execution.actual_results) == 3

    def test_query_execution_with_relevance_scores(self):
        """Query execution can have relevance scores."""
        execution = QueryExecution(
            query_id="q123",
            query_text="find login function",
            expected_results=["doc_a", "doc_b"],
            expected_relevance={
                "doc_a": 3,  # Highly relevant
                "doc_b": 1,  # Marginally relevant
            },
            actual_results=["doc_b", "doc_a"],
        )
        assert execution.expected_relevance["doc_a"] == 3


class TestMetricConfig:
    """Tests for MetricConfig."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = MetricConfig()
        assert config.k_values == [1, 5, 10]
        assert config.regression_threshold == 0.02  # 2% regression

    def test_custom_config(self):
        """Can customize config."""
        config = MetricConfig(k_values=[3, 10, 20], regression_threshold=0.05)
        assert config.k_values == [3, 10, 20]


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_create_result(self):
        """Can create evaluation result."""
        result = EvaluationResult(
            mrr=0.75,
            ndcg={10: 0.85},
            passed=True,
        )
        assert result.mrr == 0.75
        assert result.ndcg[10] == 0.85
        assert result.passed is True

    def test_result_with_regression(self):
        """Result can indicate regression."""
        result = EvaluationResult(
            mrr=0.70,
            ndcg={10: 0.75},
            passed=False,
            regressions=["MRR dropped from 0.75 to 0.70"],
        )
        assert result.passed is False
        assert len(result.regressions) == 1


class TestEvaluationHarness:
    """Tests for EvaluationHarness class."""

    @pytest.fixture
    def harness(self) -> EvaluationHarness:
        """Create evaluation harness."""
        return EvaluationHarness()

    def test_calculate_mrr_perfect_first(self, harness: EvaluationHarness):
        """MRR is 1.0 when first result is correct."""
        execution = QueryExecution(
            query_id="q1",
            query_text="test",
            expected_results=["doc_a", "doc_b"],
            actual_results=["doc_a", "doc_c", "doc_b"],
        )
        mrr = harness.calculate_mrr([execution])
        assert mrr == 1.0

    def test_calculate_mrr_second_position(self, harness: EvaluationHarness):
        """MRR is 0.5 when first hit is at position 2."""
        execution = QueryExecution(
            query_id="q1",
            query_text="test",
            expected_results=["doc_a"],
            actual_results=["doc_x", "doc_a", "doc_y"],
        )
        mrr = harness.calculate_mrr([execution])
        assert mrr == 0.5

    def test_calculate_mrr_not_found(self, harness: EvaluationHarness):
        """MRR is 0.0 when no relevant results found."""
        execution = QueryExecution(
            query_id="q1",
            query_text="test",
            expected_results=["doc_a"],
            actual_results=["doc_x", "doc_y", "doc_z"],
        )
        mrr = harness.calculate_mrr([execution])
        assert mrr == 0.0

    def test_calculate_mrr_average(self, harness: EvaluationHarness):
        """MRR is averaged across queries."""
        executions = [
            QueryExecution(
                query_id="q1",
                query_text="test1",
                expected_results=["doc_a"],
                actual_results=["doc_a", "doc_b"],  # RR = 1.0
            ),
            QueryExecution(
                query_id="q2",
                query_text="test2",
                expected_results=["doc_a"],
                actual_results=["doc_x", "doc_a"],  # RR = 0.5
            ),
        ]
        mrr = harness.calculate_mrr(executions)
        assert mrr == 0.75  # (1.0 + 0.5) / 2

    def test_calculate_ndcg_perfect(self, harness: EvaluationHarness):
        """NDCG is 1.0 for perfect ordering."""
        execution = QueryExecution(
            query_id="q1",
            query_text="test",
            expected_results=["doc_a", "doc_b", "doc_c"],
            expected_relevance={"doc_a": 3, "doc_b": 2, "doc_c": 1},
            actual_results=["doc_a", "doc_b", "doc_c"],
        )
        ndcg = harness.calculate_ndcg([execution], k=3)
        assert ndcg == pytest.approx(1.0, rel=0.01)

    def test_calculate_ndcg_reversed(self, harness: EvaluationHarness):
        """NDCG is lower for reversed ordering."""
        execution = QueryExecution(
            query_id="q1",
            query_text="test",
            expected_results=["doc_a", "doc_b", "doc_c"],
            expected_relevance={"doc_a": 3, "doc_b": 2, "doc_c": 1},
            actual_results=["doc_c", "doc_b", "doc_a"],  # Reversed
        )
        ndcg = harness.calculate_ndcg([execution], k=3)
        assert ndcg < 1.0  # Not perfect

    def test_evaluate_runs_all_metrics(self, harness: EvaluationHarness):
        """Evaluate method calculates all metrics."""
        executions = [
            QueryExecution(
                query_id="q1",
                query_text="test",
                expected_results=["doc_a", "doc_b"],
                expected_relevance={"doc_a": 2, "doc_b": 1},
                actual_results=["doc_a", "doc_b", "doc_c"],
            ),
        ]
        result = harness.evaluate(executions)

        assert result.mrr > 0
        assert 10 in result.ndcg
        assert result.passed is not None

    def test_evaluate_detects_regression(self, harness: EvaluationHarness):
        """Evaluate detects regression from baseline."""
        executions = [
            QueryExecution(
                query_id="q1",
                query_text="test",
                expected_results=["doc_a"],
                actual_results=["doc_x", "doc_y", "doc_z"],  # All wrong
            ),
        ]
        baseline = EvaluationResult(
            mrr=0.8,
            ndcg={10: 0.85},
            passed=True,
        )
        result = harness.evaluate(executions, baseline=baseline)

        assert result.passed is False
        assert len(result.regressions) > 0

    def test_evaluate_passes_when_above_threshold(self, harness: EvaluationHarness):
        """Evaluate passes when within threshold of baseline."""
        executions = [
            QueryExecution(
                query_id="q1",
                query_text="test",
                expected_results=["doc_a"],
                actual_results=["doc_a", "doc_b", "doc_c"],
            ),
        ]
        baseline = EvaluationResult(
            mrr=0.99,  # Baseline is 0.99
            ndcg={10: 0.98},
            passed=True,
        )
        config = MetricConfig(regression_threshold=0.02)
        harness_with_config = EvaluationHarness(config=config)

        result = harness_with_config.evaluate(executions, baseline=baseline)
        # MRR is 1.0 (better than baseline), should pass
        assert result.passed is True


class TestRetrievalInstrumentationIntegration:
    """Integration tests combining instrumentation and evaluation."""

    def test_metrics_can_feed_evaluation(self):
        """Instrumentation metrics can be used for evaluation."""
        instrumentation = RetrievalInstrumentation()
        harness = EvaluationHarness()

        # Simulate a query execution
        instrumentation.start_query("q123")
        instrumentation.record_stage("q123", StageType.EMBEDDING, 10.0, True)
        instrumentation.record_stage("q123", StageType.VECTOR, 20.0, True)
        metrics = instrumentation.finish_query("q123", result_count=5)

        # Create execution for evaluation
        execution = QueryExecution(
            query_id=metrics.query_id,
            query_text="find function",
            expected_results=["doc_a", "doc_b"],
            actual_results=["doc_a", "doc_c", "doc_b"],
        )

        result = harness.evaluate([execution])
        assert result.mrr == 1.0  # doc_a is first
