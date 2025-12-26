"""Tests for benchmark result dataclasses.

TDD: Tests written first to define the contract for result dataclasses.
"""

import pytest
from dataclasses import fields

from openmemory.api.benchmarks.embeddings.metrics.latency import LatencyStats
from openmemory.api.benchmarks.lexical.decision_matrix.criteria import CriterionName


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_benchmark_config_import(self):
        """BenchmarkConfig should be importable from results module."""
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        assert BenchmarkConfig is not None

    def test_benchmark_config_has_required_fields(self):
        """BenchmarkConfig should have all required fields."""
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        field_names = {f.name for f in fields(BenchmarkConfig)}
        required_fields = {
            "dataset_name",
            "dataset_language",
            "dataset_split",
            "sample_limit",
            "embedding_models",
            "lexical_backends",
            "mrr_k",
            "ndcg_k",
        }
        assert required_fields.issubset(field_names)

    def test_benchmark_config_creation(self):
        """BenchmarkConfig should be creatable with required args."""
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            dataset_language="python",
            dataset_split="test",
            sample_limit=100,
            embedding_models=["qwen3-8b", "nomic"],
            lexical_backends=["tantivy", "opensearch"],
            mrr_k=10,
            ndcg_k=10,
        )

        assert config.dataset_name == "codesearchnet"
        assert config.dataset_language == "python"
        assert config.sample_limit == 100
        assert "qwen3-8b" in config.embedding_models
        assert "tantivy" in config.lexical_backends

    def test_benchmark_config_defaults(self):
        """BenchmarkConfig should have sensible defaults."""
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["qwen3-8b"],
        )

        # Default values
        assert config.dataset_language == "python"
        assert config.dataset_split == "test"
        assert config.sample_limit is None  # No limit by default
        assert config.lexical_backends == []
        assert config.mrr_k == 10
        assert config.ndcg_k == 10


class TestModelResult:
    """Tests for ModelResult dataclass."""

    def test_model_result_import(self):
        """ModelResult should be importable from results module."""
        from openmemory.api.benchmarks.runner.results import ModelResult

        assert ModelResult is not None

    def test_model_result_has_required_fields(self):
        """ModelResult should have all required fields."""
        from openmemory.api.benchmarks.runner.results import ModelResult

        field_names = {f.name for f in fields(ModelResult)}
        required_fields = {
            "model_name",
            "provider",
            "mrr_score",
            "ndcg_score",
            "latency_stats",
            "num_queries",
            "embedding_dimensions",
        }
        assert required_fields.issubset(field_names)

    def test_model_result_creation(self):
        """ModelResult should be creatable with all fields."""
        from openmemory.api.benchmarks.runner.results import ModelResult

        latency = LatencyStats(
            p50=10.0,
            p95=25.0,
            p99=50.0,
            mean=12.0,
            min=5.0,
            max=100.0,
            count=100,
        )

        result = ModelResult(
            model_name="qwen3-8b",
            provider="ollama",
            mrr_score=0.85,
            ndcg_score=0.90,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=4096,
        )

        assert result.model_name == "qwen3-8b"
        assert result.provider == "ollama"
        assert result.mrr_score == 0.85
        assert result.ndcg_score == 0.90
        assert result.latency_stats.p95 == 25.0
        assert result.num_queries == 100

    def test_model_result_meets_mrr_threshold(self):
        """ModelResult should be able to check MRR threshold."""
        from openmemory.api.benchmarks.runner.results import ModelResult

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        passing_result = ModelResult(
            model_name="good-model",
            provider="local",
            mrr_score=0.80,
            ndcg_score=0.85,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        failing_result = ModelResult(
            model_name="bad-model",
            provider="local",
            mrr_score=0.50,
            ndcg_score=0.60,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        assert passing_result.meets_mrr_threshold(0.75) is True
        assert failing_result.meets_mrr_threshold(0.75) is False

    def test_model_result_meets_ndcg_threshold(self):
        """ModelResult should be able to check NDCG threshold."""
        from openmemory.api.benchmarks.runner.results import ModelResult

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        passing_result = ModelResult(
            model_name="good-model",
            provider="local",
            mrr_score=0.80,
            ndcg_score=0.85,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        failing_result = ModelResult(
            model_name="bad-model",
            provider="local",
            mrr_score=0.80,
            ndcg_score=0.60,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        assert passing_result.meets_ndcg_threshold(0.80) is True
        assert failing_result.meets_ndcg_threshold(0.80) is False

    def test_model_result_is_production_ready(self):
        """ModelResult should check if model meets all production thresholds."""
        from openmemory.api.benchmarks.runner.results import ModelResult

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        # Meets both thresholds
        passing = ModelResult(
            model_name="good-model",
            provider="local",
            mrr_score=0.80,
            ndcg_score=0.85,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        # Fails MRR
        fails_mrr = ModelResult(
            model_name="bad-mrr",
            provider="local",
            mrr_score=0.50,
            ndcg_score=0.85,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        # Fails NDCG
        fails_ndcg = ModelResult(
            model_name="bad-ndcg",
            provider="local",
            mrr_score=0.80,
            ndcg_score=0.60,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=1024,
        )

        assert passing.is_production_ready(mrr_threshold=0.75, ndcg_threshold=0.80)
        assert not fails_mrr.is_production_ready(mrr_threshold=0.75, ndcg_threshold=0.80)
        assert not fails_ndcg.is_production_ready(mrr_threshold=0.75, ndcg_threshold=0.80)


class TestLexicalBackendResult:
    """Tests for LexicalBackendResult dataclass."""

    def test_lexical_backend_result_import(self):
        """LexicalBackendResult should be importable from results module."""
        from openmemory.api.benchmarks.runner.results import LexicalBackendResult

        assert LexicalBackendResult is not None

    def test_lexical_backend_result_has_required_fields(self):
        """LexicalBackendResult should have all required fields."""
        from openmemory.api.benchmarks.runner.results import LexicalBackendResult

        field_names = {f.name for f in fields(LexicalBackendResult)}
        required_fields = {
            "backend_name",
            "criterion_scores",
            "weighted_total",
            "latency_stats",
            "document_count",
        }
        assert required_fields.issubset(field_names)

    def test_lexical_backend_result_creation(self):
        """LexicalBackendResult should be creatable with all fields."""
        from openmemory.api.benchmarks.runner.results import LexicalBackendResult

        latency = LatencyStats(
            p50=5.0, p95=15.0, p99=30.0, mean=7.0, min=2.0, max=50.0, count=100
        )

        criterion_scores = {
            CriterionName.LATENCY: 0.9,
            CriterionName.OPS_COMPLEXITY: 0.95,
            CriterionName.SCALABILITY: 0.6,
            CriterionName.FEATURE_SUPPORT: 0.7,
        }

        result = LexicalBackendResult(
            backend_name="tantivy",
            criterion_scores=criterion_scores,
            weighted_total=0.82,
            latency_stats=latency,
            document_count=1000,
        )

        assert result.backend_name == "tantivy"
        assert result.criterion_scores[CriterionName.LATENCY] == 0.9
        assert result.weighted_total == 0.82
        assert result.latency_stats.p95 == 15.0
        assert result.document_count == 1000


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_import(self):
        """BenchmarkResult should be importable from results module."""
        from openmemory.api.benchmarks.runner.results import BenchmarkResult

        assert BenchmarkResult is not None

    def test_benchmark_result_has_required_fields(self):
        """BenchmarkResult should have all required fields."""
        from openmemory.api.benchmarks.runner.results import BenchmarkResult

        field_names = {f.name for f in fields(BenchmarkResult)}
        required_fields = {
            "config",
            "model_results",
            "lexical_results",
            "timestamp",
            "total_duration_seconds",
        }
        assert required_fields.issubset(field_names)

    def test_benchmark_result_creation(self):
        """BenchmarkResult should be creatable with all fields."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
            ModelResult,
            LexicalBackendResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["qwen3-8b"],
        )

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        model_result = ModelResult(
            model_name="qwen3-8b",
            provider="ollama",
            mrr_score=0.85,
            ndcg_score=0.90,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=4096,
        )

        result = BenchmarkResult(
            config=config,
            model_results=[model_result],
            lexical_results=[],
            timestamp=datetime.now(),
            total_duration_seconds=60.5,
        )

        assert result.config.dataset_name == "codesearchnet"
        assert len(result.model_results) == 1
        assert result.model_results[0].model_name == "qwen3-8b"
        assert result.total_duration_seconds == 60.5

    def test_benchmark_result_get_best_model(self):
        """BenchmarkResult should return the best model by MRR score."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
            ModelResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["model1", "model2", "model3"],
        )

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        model_results = [
            ModelResult(
                model_name="model1",
                provider="local",
                mrr_score=0.70,
                ndcg_score=0.75,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
            ModelResult(
                model_name="model2",
                provider="local",
                mrr_score=0.85,  # Best MRR
                ndcg_score=0.90,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
            ModelResult(
                model_name="model3",
                provider="local",
                mrr_score=0.75,
                ndcg_score=0.80,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
        ]

        result = BenchmarkResult(
            config=config,
            model_results=model_results,
            lexical_results=[],
            timestamp=datetime.now(),
            total_duration_seconds=60.5,
        )

        best = result.get_best_model(metric="mrr")
        assert best is not None
        assert best.model_name == "model2"

    def test_benchmark_result_get_best_model_by_ndcg(self):
        """BenchmarkResult should return the best model by NDCG score."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
            ModelResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["model1", "model2"],
        )

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        model_results = [
            ModelResult(
                model_name="model1",
                provider="local",
                mrr_score=0.85,
                ndcg_score=0.75,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
            ModelResult(
                model_name="model2",
                provider="local",
                mrr_score=0.70,
                ndcg_score=0.90,  # Best NDCG
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
        ]

        result = BenchmarkResult(
            config=config,
            model_results=model_results,
            lexical_results=[],
            timestamp=datetime.now(),
            total_duration_seconds=60.5,
        )

        best = result.get_best_model(metric="ndcg")
        assert best is not None
        assert best.model_name == "model2"

    def test_benchmark_result_get_best_model_empty(self):
        """BenchmarkResult should return None if no model results."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=[],
        )

        result = BenchmarkResult(
            config=config,
            model_results=[],
            lexical_results=[],
            timestamp=datetime.now(),
            total_duration_seconds=0.0,
        )

        assert result.get_best_model(metric="mrr") is None

    def test_benchmark_result_get_best_lexical_backend(self):
        """BenchmarkResult should return the best lexical backend."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
            LexicalBackendResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=[],
            lexical_backends=["tantivy", "opensearch"],
        )

        latency = LatencyStats(
            p50=5.0, p95=15.0, p99=30.0, mean=7.0, min=2.0, max=50.0, count=100
        )

        criterion_scores = {
            CriterionName.LATENCY: 0.9,
            CriterionName.OPS_COMPLEXITY: 0.95,
            CriterionName.SCALABILITY: 0.6,
            CriterionName.FEATURE_SUPPORT: 0.7,
        }

        lexical_results = [
            LexicalBackendResult(
                backend_name="tantivy",
                criterion_scores=criterion_scores,
                weighted_total=0.82,
                latency_stats=latency,
                document_count=1000,
            ),
            LexicalBackendResult(
                backend_name="opensearch",
                criterion_scores=criterion_scores,
                weighted_total=0.78,
                latency_stats=latency,
                document_count=1000,
            ),
        ]

        result = BenchmarkResult(
            config=config,
            model_results=[],
            lexical_results=lexical_results,
            timestamp=datetime.now(),
            total_duration_seconds=30.0,
        )

        best = result.get_best_lexical_backend()
        assert best is not None
        assert best.backend_name == "tantivy"

    def test_benchmark_result_get_production_ready_models(self):
        """BenchmarkResult should filter models that meet production thresholds."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
            ModelResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["good", "bad_mrr", "bad_ndcg"],
        )

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        model_results = [
            ModelResult(
                model_name="good",
                provider="local",
                mrr_score=0.80,
                ndcg_score=0.85,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
            ModelResult(
                model_name="bad_mrr",
                provider="local",
                mrr_score=0.50,
                ndcg_score=0.85,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
            ModelResult(
                model_name="bad_ndcg",
                provider="local",
                mrr_score=0.80,
                ndcg_score=0.60,
                latency_stats=latency,
                num_queries=100,
                embedding_dimensions=1024,
            ),
        ]

        result = BenchmarkResult(
            config=config,
            model_results=model_results,
            lexical_results=[],
            timestamp=datetime.now(),
            total_duration_seconds=60.0,
        )

        production_ready = result.get_production_ready_models(
            mrr_threshold=0.75, ndcg_threshold=0.80
        )
        assert len(production_ready) == 1
        assert production_ready[0].model_name == "good"

    def test_benchmark_result_to_dict(self):
        """BenchmarkResult should be serializable to dict."""
        from openmemory.api.benchmarks.runner.results import (
            BenchmarkConfig,
            BenchmarkResult,
            ModelResult,
        )
        from datetime import datetime

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["qwen3-8b"],
        )

        latency = LatencyStats(
            p50=10.0, p95=25.0, p99=50.0, mean=12.0, min=5.0, max=100.0, count=100
        )

        model_result = ModelResult(
            model_name="qwen3-8b",
            provider="ollama",
            mrr_score=0.85,
            ndcg_score=0.90,
            latency_stats=latency,
            num_queries=100,
            embedding_dimensions=4096,
        )

        result = BenchmarkResult(
            config=config,
            model_results=[model_result],
            lexical_results=[],
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            total_duration_seconds=60.5,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["config"]["dataset_name"] == "codesearchnet"
        assert len(result_dict["model_results"]) == 1
        assert result_dict["model_results"][0]["model_name"] == "qwen3-8b"
        assert result_dict["total_duration_seconds"] == 60.5
