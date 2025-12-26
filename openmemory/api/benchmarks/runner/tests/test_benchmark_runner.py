"""Tests for BenchmarkRunner.

TDD: Tests written first to define the contract for the benchmark runner.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from openmemory.api.benchmarks.embeddings.metrics.latency import LatencyStats
from openmemory.api.benchmarks.lexical.decision_matrix.criteria import CriterionName


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_embedder():
    """Create a mock embedder adapter."""
    embedder = Mock()
    embedder.info = Mock()
    embedder.info.model_name = "mock-model"
    embedder.info.provider = "mock"
    embedder.info.dimensions = 1024
    embedder.info.max_sequence_length = 8192
    embedder.info.is_code_optimized = True

    # Return consistent embeddings (normalized)
    embedder.embed.return_value = [0.1] * 1024
    embedder.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]

    return embedder


@pytest.fixture
def mock_lexical_backend():
    """Create a mock lexical backend."""
    from openmemory.api.benchmarks.lexical.backends.base import (
        BackendStats,
        SearchResult,
    )

    backend = Mock()
    backend.name = "mock-backend"
    backend.index_documents.return_value = 100
    backend.search.return_value = [
        SearchResult(doc_id="doc1", score=0.9, content="content 1"),
        SearchResult(doc_id="doc2", score=0.8, content="content 2"),
        SearchResult(doc_id="doc3", score=0.7, content="content 3"),
    ]
    backend.get_stats.return_value = BackendStats(
        index_size_bytes=1024,
        document_count=100,
        backend_name="mock-backend",
    )
    backend.clear.return_value = None

    return backend


@pytest.fixture
def sample_codesearchnet_data():
    """Sample CodeSearchNet data for testing."""
    from openmemory.api.benchmarks.embeddings.datasets.codesearchnet import (
        CodeSearchNetSample,
    )

    return [
        CodeSearchNetSample(
            query="Calculate the sum of two numbers",
            code="def add(a, b): return a + b",
            docstring="Calculate the sum of two numbers",
            func_name="add",
            url="https://github.com/example/repo",
            id="abc123",
        ),
        CodeSearchNetSample(
            query="Find the maximum value in a list",
            code="def find_max(lst): return max(lst)",
            docstring="Find the maximum value in a list",
            func_name="find_max",
            url="https://github.com/example/repo",
            id="def456",
        ),
        CodeSearchNetSample(
            query="Sort a list in ascending order",
            code="def sort_list(lst): return sorted(lst)",
            docstring="Sort a list in ascending order",
            func_name="sort_list",
            url="https://github.com/example/repo",
            id="ghi789",
        ),
    ]


@pytest.fixture
def benchmark_config():
    """Create a benchmark config for testing."""
    from openmemory.api.benchmarks.runner.results import BenchmarkConfig

    return BenchmarkConfig(
        dataset_name="codesearchnet",
        dataset_language="python",
        dataset_split="test",
        sample_limit=100,
        embedding_models=["mock-model"],
        lexical_backends=["mock-backend"],
        mrr_k=10,
        ndcg_k=10,
    )


# ============================================================================
# BenchmarkRunner Initialization Tests
# ============================================================================


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization."""

    def test_benchmark_runner_import(self):
        """BenchmarkRunner should be importable from runner module."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        assert BenchmarkRunner is not None

    def test_benchmark_runner_init_with_config(self, benchmark_config):
        """BenchmarkRunner should initialize with a config."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        runner = BenchmarkRunner(config=benchmark_config)

        assert runner.config == benchmark_config

    def test_benchmark_runner_init_validates_config(self):
        """BenchmarkRunner should validate config on init."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        # Empty config should be valid (uses defaults)
        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=[],
        )

        runner = BenchmarkRunner(config=config)
        assert runner.config.embedding_models == []


# ============================================================================
# Dataset Loading Tests
# ============================================================================


class TestBenchmarkRunnerDatasetLoading:
    """Tests for loading dataset samples."""

    def test_load_samples_returns_list(self, benchmark_config, sample_codesearchnet_data):
        """BenchmarkRunner should load dataset samples as a list."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            runner = BenchmarkRunner(config=benchmark_config)
            samples = runner.load_samples()

            assert len(samples) == 3
            assert samples[0].query == "Calculate the sum of two numbers"

    def test_load_samples_respects_limit(self, sample_codesearchnet_data):
        """BenchmarkRunner should respect sample_limit config."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["mock"],
            sample_limit=2,
        )

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data[:2])
            MockLoader.return_value = mock_loader

            runner = BenchmarkRunner(config=config)
            samples = runner.load_samples()

            mock_loader.load.assert_called_with(limit=2)
            assert len(samples) == 2

    def test_load_samples_uses_language_and_split(self, sample_codesearchnet_data):
        """BenchmarkRunner should pass language and split to loader."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            dataset_language="javascript",
            dataset_split="valid",
            embedding_models=["mock"],
        )

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader:
            mock_loader = Mock()
            mock_loader.load.return_value = iter([])
            MockLoader.return_value = mock_loader

            runner = BenchmarkRunner(config=config)
            runner.load_samples()

            MockLoader.assert_called_with(
                language="javascript",
                split="valid",
            )


# ============================================================================
# Embedding Model Benchmark Tests
# ============================================================================


class TestBenchmarkRunnerEmbeddingModels:
    """Tests for running embedding model benchmarks."""

    def test_run_embedding_benchmark_single_model(
        self, benchmark_config, mock_embedder, sample_codesearchnet_data
    ):
        """BenchmarkRunner should benchmark a single embedding model."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import ModelResult

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create:
            mock_create.return_value = mock_embedder

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_embedding_benchmark(
                model_name="mock-model",
                samples=sample_codesearchnet_data,
            )

            assert isinstance(result, ModelResult)
            assert result.model_name == "mock-model"
            assert result.num_queries == 3

    def test_run_embedding_benchmark_calculates_mrr(
        self, benchmark_config, sample_codesearchnet_data
    ):
        """BenchmarkRunner should calculate MRR for embedding model."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        mock_embedder = Mock()
        mock_embedder.info.model_name = "mock-model"
        mock_embedder.info.provider = "mock"
        mock_embedder.info.dimensions = 1024

        # Simulate embeddings that produce perfect ranking
        # (query embedding similar to its own code embedding)
        def mock_embed_batch(texts: List[str]) -> List[List[float]]:
            embeddings = []
            for i, text in enumerate(texts):
                # Create unique embedding per text
                emb = [0.0] * 1024
                emb[i % 1024] = 1.0  # Different position for each
                embeddings.append(emb)
            return embeddings

        mock_embedder.embed_batch.side_effect = mock_embed_batch

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create:
            mock_create.return_value = mock_embedder

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_embedding_benchmark(
                model_name="mock-model",
                samples=sample_codesearchnet_data,
            )

            # MRR should be between 0 and 1
            assert 0.0 <= result.mrr_score <= 1.0

    def test_run_embedding_benchmark_calculates_ndcg(
        self, benchmark_config, sample_codesearchnet_data
    ):
        """BenchmarkRunner should calculate NDCG for embedding model."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        mock_embedder = Mock()
        mock_embedder.info.model_name = "mock-model"
        mock_embedder.info.provider = "mock"
        mock_embedder.info.dimensions = 1024
        # Return 3 embeddings per call (once for queries, once for codes)
        mock_embedder.embed_batch.side_effect = [
            [[0.1] * 1024] * 3,  # query embeddings
            [[0.1] * 1024] * 3,  # code embeddings
        ]

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create:
            mock_create.return_value = mock_embedder

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_embedding_benchmark(
                model_name="mock-model",
                samples=sample_codesearchnet_data,
            )

            # NDCG should be between 0 and 1
            assert 0.0 <= result.ndcg_score <= 1.0

    def test_run_embedding_benchmark_tracks_latency(
        self, benchmark_config, mock_embedder, sample_codesearchnet_data
    ):
        """BenchmarkRunner should track latency for embedding model."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create:
            mock_create.return_value = mock_embedder

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_embedding_benchmark(
                model_name="mock-model",
                samples=sample_codesearchnet_data,
            )

            # Latency stats should be populated
            assert result.latency_stats is not None
            assert result.latency_stats.count > 0
            assert result.latency_stats.p50 >= 0
            assert result.latency_stats.p95 >= 0

    def test_run_embedding_benchmark_stores_dimensions(
        self, benchmark_config, mock_embedder, sample_codesearchnet_data
    ):
        """BenchmarkRunner should store embedding dimensions."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        mock_embedder.info.dimensions = 4096

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create:
            mock_create.return_value = mock_embedder

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_embedding_benchmark(
                model_name="mock-model",
                samples=sample_codesearchnet_data,
            )

            assert result.embedding_dimensions == 4096


# ============================================================================
# Lexical Backend Benchmark Tests
# ============================================================================


class TestBenchmarkRunnerLexicalBackends:
    """Tests for running lexical backend benchmarks."""

    def test_run_lexical_benchmark_single_backend(
        self, benchmark_config, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner should benchmark a single lexical backend."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import LexicalBackendResult

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create:
            mock_create.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_lexical_benchmark(
                backend_name="mock-backend",
                samples=sample_codesearchnet_data,
            )

            assert isinstance(result, LexicalBackendResult)
            assert result.backend_name == "mock-backend"

    def test_run_lexical_benchmark_indexes_documents(
        self, benchmark_config, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner should index documents before searching."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create:
            mock_create.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            runner.run_lexical_benchmark(
                backend_name="mock-backend",
                samples=sample_codesearchnet_data,
            )

            mock_lexical_backend.index_documents.assert_called_once()

    def test_run_lexical_benchmark_tracks_latency(
        self, benchmark_config, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner should track search latency."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create:
            mock_create.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_lexical_benchmark(
                backend_name="mock-backend",
                samples=sample_codesearchnet_data,
            )

            assert result.latency_stats is not None
            assert result.latency_stats.count > 0

    def test_run_lexical_benchmark_uses_decision_matrix(
        self, benchmark_config, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner should use decision matrix for scoring."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create:
            mock_create.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_lexical_benchmark(
                backend_name="mock-backend",
                samples=sample_codesearchnet_data,
            )

            # Should have criterion scores
            assert result.criterion_scores is not None
            assert CriterionName.LATENCY in result.criterion_scores
            assert result.weighted_total >= 0.0

    def test_run_lexical_benchmark_clears_backend_after(
        self, benchmark_config, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner should clear backend after benchmark."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create:
            mock_create.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            runner.run_lexical_benchmark(
                backend_name="mock-backend",
                samples=sample_codesearchnet_data,
            )

            mock_lexical_backend.clear.assert_called_once()


# ============================================================================
# Full Benchmark Run Tests
# ============================================================================


class TestBenchmarkRunnerFullRun:
    """Tests for running complete benchmarks."""

    def test_run_returns_benchmark_result(
        self, benchmark_config, mock_embedder, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner.run() should return a BenchmarkResult."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkResult

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create_adapter, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create_backend:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_adapter.return_value = mock_embedder
            mock_create_backend.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run()

            assert isinstance(result, BenchmarkResult)

    def test_run_includes_all_model_results(
        self, mock_embedder, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner.run() should include results for all models."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["model1", "model2", "model3"],
            lexical_backends=[],
            sample_limit=3,
        )

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create_adapter:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_adapter.return_value = mock_embedder

            runner = BenchmarkRunner(config=config)
            result = runner.run()

            assert len(result.model_results) == 3

    def test_run_includes_all_lexical_results(
        self, mock_embedder, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner.run() should include results for all backends."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=[],
            lexical_backends=["backend1", "backend2"],
            sample_limit=3,
        )

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create_backend:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_backend.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=config)
            result = runner.run()

            assert len(result.lexical_results) == 2

    def test_run_records_total_duration(
        self, benchmark_config, mock_embedder, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner.run() should record total benchmark duration."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create_adapter, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create_backend:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_adapter.return_value = mock_embedder
            mock_create_backend.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run()

            assert result.total_duration_seconds >= 0.0

    def test_run_records_timestamp(
        self, benchmark_config, mock_embedder, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner.run() should record start timestamp."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from datetime import datetime

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create_adapter, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create_backend:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_adapter.return_value = mock_embedder
            mock_create_backend.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            before = datetime.now()
            result = runner.run()
            after = datetime.now()

            assert result.timestamp >= before
            assert result.timestamp <= after


# ============================================================================
# Aggregation Tests
# ============================================================================


class TestBenchmarkRunnerAggregation:
    """Tests for aggregating results across multiple runs."""

    def test_run_multiple_aggregates_results(
        self, benchmark_config, mock_embedder, mock_lexical_backend, sample_codesearchnet_data
    ):
        """BenchmarkRunner should aggregate results from multiple runs."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkResult

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create_adapter, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create_backend:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_adapter.return_value = mock_embedder
            mock_create_backend.return_value = mock_lexical_backend

            runner = BenchmarkRunner(config=benchmark_config)
            result = runner.run_multiple(num_runs=3)

            assert isinstance(result, BenchmarkResult)
            # Total duration should be sum of all runs
            assert result.total_duration_seconds > 0

    def test_run_multiple_averages_scores(
        self, sample_codesearchnet_data
    ):
        """BenchmarkRunner should average scores across runs."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            embedding_models=["mock-model"],
            lexical_backends=[],
            sample_limit=3,
        )

        mock_embedder = Mock()
        mock_embedder.info.model_name = "mock-model"
        mock_embedder.info.provider = "mock"
        mock_embedder.info.dimensions = 1024
        # Return 3 embeddings per call (2 calls per run * 3 runs = 6 calls)
        mock_embedder.embed_batch.side_effect = [
            [[0.1] * 1024] * 3,  # run 1: query embeddings
            [[0.1] * 1024] * 3,  # run 1: code embeddings
            [[0.1] * 1024] * 3,  # run 2: query embeddings
            [[0.1] * 1024] * 3,  # run 2: code embeddings
            [[0.1] * 1024] * 3,  # run 3: query embeddings
            [[0.1] * 1024] * 3,  # run 3: code embeddings
        ]

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader, patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create_adapter:
            mock_loader = Mock()
            mock_loader.load.return_value = iter(sample_codesearchnet_data)
            MockLoader.return_value = mock_loader

            mock_create_adapter.return_value = mock_embedder

            runner = BenchmarkRunner(config=config)
            result = runner.run_multiple(num_runs=3)

            # Should have averaged MRR and NDCG scores
            assert result.model_results[0].mrr_score >= 0.0
            assert result.model_results[0].ndcg_score >= 0.0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestBenchmarkRunnerErrorHandling:
    """Tests for error handling in benchmark runner."""

    def test_run_embedding_benchmark_handles_adapter_error(
        self, benchmark_config, sample_codesearchnet_data
    ):
        """BenchmarkRunner should handle adapter creation errors gracefully."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_adapter"
        ) as mock_create:
            mock_create.side_effect = ValueError("Unknown model: bad-model")

            runner = BenchmarkRunner(config=benchmark_config)

            with pytest.raises(ValueError, match="Unknown model"):
                runner.run_embedding_benchmark(
                    model_name="bad-model",
                    samples=sample_codesearchnet_data,
                )

    def test_run_lexical_benchmark_handles_backend_error(
        self, benchmark_config, sample_codesearchnet_data
    ):
        """BenchmarkRunner should handle backend creation errors gracefully."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.create_backend"
        ) as mock_create:
            mock_create.side_effect = ValueError("Unknown backend: bad-backend")

            runner = BenchmarkRunner(config=benchmark_config)

            with pytest.raises(ValueError, match="Unknown backend"):
                runner.run_lexical_benchmark(
                    backend_name="bad-backend",
                    samples=sample_codesearchnet_data,
                )

    def test_run_handles_empty_samples(self, benchmark_config):
        """BenchmarkRunner should handle empty sample list."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner

        with patch(
            "openmemory.api.benchmarks.runner.benchmark_runner.CodeSearchNetLoader"
        ) as MockLoader:
            mock_loader = Mock()
            mock_loader.load.return_value = iter([])
            MockLoader.return_value = mock_loader

            runner = BenchmarkRunner(config=benchmark_config)

            with pytest.raises(ValueError, match="No samples loaded"):
                runner.run()


# ============================================================================
# Integration Test Marker
# ============================================================================


class TestBenchmarkRunnerIntegration:
    """Integration tests that use real components (marked for separate execution)."""

    @pytest.mark.integration
    def test_integration_with_codesearchnet_loader(self):
        """Test with real CodeSearchNet loader."""
        from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
        from openmemory.api.benchmarks.runner.results import BenchmarkConfig

        config = BenchmarkConfig(
            dataset_name="codesearchnet",
            dataset_language="python",
            dataset_split="test",
            sample_limit=5,
            embedding_models=[],
            lexical_backends=["tantivy"],
        )

        runner = BenchmarkRunner(config=config)
        samples = runner.load_samples()

        assert len(samples) == 5
        assert all(hasattr(s, "query") for s in samples)
        assert all(hasattr(s, "code") for s in samples)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_integration_full_benchmark_run(self):
        """Full integration test with all components."""
        # This test would use real adapters and backends
        # Only run with explicit integration test flag
        pass
