"""Tests for reranker integration with tri-hybrid retrieval.

TDD tests covering:
- Reranker adapter interface (abstract base)
- Cross-encoder reranker implementation
- Integration with tri-hybrid results
- Score normalization after reranking
- Fallback when reranker unavailable
- Latency requirements (<50ms for reranking step)
- Multiple reranker backends (cross-encoder, Cohere, etc.)
- Configurable reranking depth (top-k results to rerank)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test: RerankerConfig
# =============================================================================


class TestRerankerConfig:
    """Tests for reranker configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        from openmemory.api.retrieval.reranker import RerankerConfig

        config = RerankerConfig()
        assert config.top_k == 20  # Rerank top 20 by default
        assert config.enabled is True
        assert config.timeout_ms == 50  # 50ms timeout target

    def test_custom_top_k(self):
        """Should accept custom top_k."""
        from openmemory.api.retrieval.reranker import RerankerConfig

        config = RerankerConfig(top_k=50)
        assert config.top_k == 50

    def test_can_disable_reranker(self):
        """Should be able to disable reranker."""
        from openmemory.api.retrieval.reranker import RerankerConfig

        config = RerankerConfig(enabled=False)
        assert config.enabled is False

    def test_batch_size_config(self):
        """Should support batch size for efficient reranking."""
        from openmemory.api.retrieval.reranker import RerankerConfig

        config = RerankerConfig(batch_size=32)
        assert config.batch_size == 32

    def test_min_score_threshold(self):
        """Should support minimum score threshold."""
        from openmemory.api.retrieval.reranker import RerankerConfig

        config = RerankerConfig(min_score=0.5)
        assert config.min_score == 0.5


# =============================================================================
# Test: RerankerAdapter (Abstract Base)
# =============================================================================


class TestRerankerAdapter:
    """Tests for the reranker adapter interface."""

    def test_adapter_is_abstract(self):
        """RerankerAdapter should be abstract."""
        from openmemory.api.retrieval.reranker import RerankerAdapter

        with pytest.raises(TypeError, match="abstract"):
            RerankerAdapter()

    def test_adapter_requires_rerank_method(self):
        """Adapter must implement rerank method."""
        from openmemory.api.retrieval.reranker import RerankerAdapter

        # Create a minimal concrete implementation
        class IncompleteAdapter(RerankerAdapter):
            pass

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_adapter_contract(self):
        """Adapter should define required methods."""
        from openmemory.api.retrieval.reranker import RerankerAdapter

        # Check that abstract methods are defined
        assert hasattr(RerankerAdapter, "rerank")
        assert hasattr(RerankerAdapter, "is_available")


# =============================================================================
# Test: CrossEncoderReranker
# =============================================================================


class TestCrossEncoderReranker:
    """Tests for cross-encoder based reranking."""

    def test_create_cross_encoder_reranker(self):
        """Should create cross-encoder reranker."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            config=RerankerConfig(),
        )
        assert reranker is not None

    def test_rerank_returns_scores(self):
        """Reranker should return scores for query-document pairs."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
            RerankedResult,
        )

        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.3]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        results = reranker.rerank(
            query="database connection",
            documents=[
                {"id": "doc1", "content": "DB pool manager"},
                {"id": "doc2", "content": "HTTP handler"},
                {"id": "doc3", "content": "Test utils"},
            ],
        )

        assert len(results) == 3
        assert all(isinstance(r, RerankedResult) for r in results)
        # Results should be sorted by score descending
        assert results[0].score >= results[1].score >= results[2].score

    def test_rerank_preserves_document_metadata(self):
        """Reranked results should preserve original document metadata."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        results = reranker.rerank(
            query="test",
            documents=[
                {
                    "id": "doc1",
                    "content": "test content",
                    "score": 0.85,
                    "metadata": {"file": "test.py", "language": "python"},
                },
            ],
        )

        assert results[0].id == "doc1"
        assert results[0].original_score == 0.85
        assert "metadata" in results[0].document

    def test_rerank_respects_top_k(self):
        """Reranker should only process top_k documents."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.7]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(top_k=3),
        )
        reranker._model = mock_model

        documents = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(10)]
        results = reranker.rerank(query="test", documents=documents)

        # Should only have called model with top_k documents
        call_args = mock_model.predict.call_args
        assert len(call_args[0][0]) <= 3

    def test_rerank_handles_empty_documents(self):
        """Reranker should handle empty document list."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )

        results = reranker.rerank(query="test", documents=[])
        assert results == []

    def test_is_available_with_loaded_model(self):
        """Should report available when model is loaded."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = MagicMock()

        assert reranker.is_available() is True

    def test_is_available_without_model(self):
        """Should report unavailable when model not loaded."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
            lazy_load=True,  # Don't load model immediately
        )
        reranker._model = None

        assert reranker.is_available() is False


# =============================================================================
# Test: CohereReranker
# =============================================================================


class TestCohereReranker:
    """Tests for Cohere-based reranking."""

    def test_create_cohere_reranker(self):
        """Should create Cohere reranker."""
        from openmemory.api.retrieval.reranker import (
            CohereReranker,
            RerankerConfig,
        )

        reranker = CohereReranker(
            api_key="test-key",
            model_name="rerank-english-v3.0",
            config=RerankerConfig(),
        )
        assert reranker is not None

    def test_cohere_rerank_calls_api(self):
        """Cohere reranker should call the API."""
        from openmemory.api.retrieval.reranker import (
            CohereReranker,
            RerankerConfig,
        )

        mock_client = MagicMock()
        mock_client.rerank.return_value = MagicMock(
            results=[
                MagicMock(index=0, relevance_score=0.9),
                MagicMock(index=1, relevance_score=0.7),
            ]
        )

        reranker = CohereReranker(
            api_key="test-key",
            model_name="rerank-english-v3.0",
            config=RerankerConfig(),
        )
        reranker._client = mock_client

        results = reranker.rerank(
            query="test query",
            documents=[
                {"id": "doc1", "content": "content 1"},
                {"id": "doc2", "content": "content 2"},
            ],
        )

        mock_client.rerank.assert_called_once()
        assert len(results) == 2

    def test_cohere_handles_api_error(self):
        """Cohere reranker should handle API errors gracefully."""
        from openmemory.api.retrieval.reranker import (
            CohereReranker,
            RerankerConfig,
            RerankerError,
        )

        mock_client = MagicMock()
        mock_client.rerank.side_effect = Exception("API error")

        reranker = CohereReranker(
            api_key="test-key",
            model_name="rerank-english-v3.0",
            config=RerankerConfig(),
        )
        reranker._client = mock_client

        with pytest.raises(RerankerError, match="API error"):
            reranker.rerank(
                query="test",
                documents=[{"id": "doc1", "content": "content"}],
            )


# =============================================================================
# Test: NoOpReranker
# =============================================================================


class TestNoOpReranker:
    """Tests for no-op reranker (passthrough)."""

    def test_noop_returns_original_order(self):
        """NoOp reranker should return documents in original order."""
        from openmemory.api.retrieval.reranker import NoOpReranker

        reranker = NoOpReranker()

        documents = [
            {"id": "doc1", "content": "first", "score": 0.9},
            {"id": "doc2", "content": "second", "score": 0.8},
            {"id": "doc3", "content": "third", "score": 0.7},
        ]

        results = reranker.rerank(query="test", documents=documents)

        assert len(results) == 3
        assert results[0].id == "doc1"
        assert results[1].id == "doc2"
        assert results[2].id == "doc3"

    def test_noop_preserves_original_scores(self):
        """NoOp reranker should preserve original scores."""
        from openmemory.api.retrieval.reranker import NoOpReranker

        reranker = NoOpReranker()

        documents = [
            {"id": "doc1", "content": "first", "score": 0.9},
        ]

        results = reranker.rerank(query="test", documents=documents)

        assert results[0].score == 0.9
        assert results[0].original_score == 0.9

    def test_noop_is_always_available(self):
        """NoOp reranker should always report available."""
        from openmemory.api.retrieval.reranker import NoOpReranker

        reranker = NoOpReranker()
        assert reranker.is_available() is True


# =============================================================================
# Test: RerankedResult
# =============================================================================


class TestRerankedResult:
    """Tests for reranked result structure."""

    def test_result_structure(self):
        """RerankedResult should have expected structure."""
        from openmemory.api.retrieval.reranker import RerankedResult

        result = RerankedResult(
            id="doc1",
            score=0.95,
            original_score=0.80,
            original_rank=3,
            new_rank=1,
            document={"content": "test content"},
        )

        assert result.id == "doc1"
        assert result.score == 0.95
        assert result.original_score == 0.80
        assert result.original_rank == 3
        assert result.new_rank == 1

    def test_result_tracks_rank_change(self):
        """RerankedResult should track rank change."""
        from openmemory.api.retrieval.reranker import RerankedResult

        result = RerankedResult(
            id="doc1",
            score=0.95,
            original_score=0.80,
            original_rank=5,
            new_rank=1,
            document={},
        )

        assert result.rank_change == 4  # Moved up 4 positions


# =============================================================================
# Test: TriHybrid Integration
# =============================================================================


class TestTriHybridIntegration:
    """Tests for integration with tri-hybrid retriever."""

    def test_rerank_trihybrid_results(self):
        """Should rerank tri-hybrid results."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            rerank_trihybrid_results,
        )
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        # Mock reranker
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.5]  # Reorder: 2nd becomes 1st

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        # Create tri-hybrid result
        trihybrid_result = TriHybridResult(
            hits=[
                TriHybridHit(
                    id="doc1",
                    score=0.9,
                    source={"content": "database pool"},
                    sources={"lexical": 0.9},
                ),
                TriHybridHit(
                    id="doc2",
                    score=0.8,
                    source={"content": "connection manager"},
                    sources={"vector": 0.8},
                ),
                TriHybridHit(
                    id="doc3",
                    score=0.7,
                    source={"content": "test helper"},
                    sources={"lexical": 0.7},
                ),
            ],
            total=3,
            timing=TriHybridTiming(total_ms=50),
        )

        # Rerank
        reranked = rerank_trihybrid_results(
            query="database connection",
            trihybrid_result=trihybrid_result,
            reranker=reranker,
        )

        # doc2 should now be first (highest reranker score)
        assert reranked.hits[0].id == "doc2"

    def test_rerank_updates_timing(self):
        """Reranking should update timing information."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            rerank_trihybrid_results,
        )
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        trihybrid_result = TriHybridResult(
            hits=[
                TriHybridHit(id="doc1", score=0.9, source={}, sources={}),
            ],
            total=1,
            timing=TriHybridTiming(total_ms=50),
        )

        reranked = rerank_trihybrid_results(
            query="test",
            trihybrid_result=trihybrid_result,
            reranker=reranker,
        )

        assert reranked.timing.rerank_ms >= 0
        assert reranked.timing.total_ms >= trihybrid_result.timing.total_ms

    def test_rerank_preserves_source_breakdown(self):
        """Reranking should preserve source breakdown."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            rerank_trihybrid_results,
        )
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        trihybrid_result = TriHybridResult(
            hits=[
                TriHybridHit(
                    id="doc1",
                    score=0.9,
                    source={},
                    sources={"lexical": 0.9, "vector": 0.8},
                ),
            ],
            total=1,
            timing=TriHybridTiming(total_ms=50),
        )

        reranked = rerank_trihybrid_results(
            query="test",
            trihybrid_result=trihybrid_result,
            reranker=reranker,
        )

        # Source breakdown should still be there
        assert "lexical" in reranked.hits[0].sources
        assert "vector" in reranked.hits[0].sources


# =============================================================================
# Test: Fallback Behavior
# =============================================================================


class TestFallbackBehavior:
    """Tests for fallback when reranker unavailable."""

    def test_fallback_when_reranker_unavailable(self):
        """Should fall back to original order when reranker unavailable."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            rerank_trihybrid_results,
        )
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
            lazy_load=True,
        )
        reranker._model = None  # Simulate unavailable

        trihybrid_result = TriHybridResult(
            hits=[
                TriHybridHit(id="doc1", score=0.9, source={}, sources={}),
                TriHybridHit(id="doc2", score=0.8, source={}, sources={}),
            ],
            total=2,
            timing=TriHybridTiming(total_ms=50),
        )

        reranked = rerank_trihybrid_results(
            query="test",
            trihybrid_result=trihybrid_result,
            reranker=reranker,
        )

        # Should return original order
        assert reranked.hits[0].id == "doc1"
        assert reranked.hits[1].id == "doc2"
        assert reranked.reranker_available is False

    def test_fallback_on_reranker_error(self):
        """Should fall back on reranker error."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            rerank_trihybrid_results,
        )
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("Model error")

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        trihybrid_result = TriHybridResult(
            hits=[
                TriHybridHit(id="doc1", score=0.9, source={}, sources={}),
            ],
            total=1,
            timing=TriHybridTiming(total_ms=50),
        )

        reranked = rerank_trihybrid_results(
            query="test",
            trihybrid_result=trihybrid_result,
            reranker=reranker,
        )

        # Should return original order
        assert reranked.hits[0].id == "doc1"
        assert reranked.reranker_error is not None

    def test_fallback_when_disabled(self):
        """Should skip reranking when disabled."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            rerank_trihybrid_results,
        )
        from openmemory.api.retrieval.trihybrid import (
            TriHybridResult,
            TriHybridHit,
            TriHybridTiming,
        )

        mock_model = MagicMock()

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(enabled=False),
        )
        reranker._model = mock_model

        trihybrid_result = TriHybridResult(
            hits=[
                TriHybridHit(id="doc1", score=0.9, source={}, sources={}),
            ],
            total=1,
            timing=TriHybridTiming(total_ms=50),
        )

        reranked = rerank_trihybrid_results(
            query="test",
            trihybrid_result=trihybrid_result,
            reranker=reranker,
        )

        # Model should not be called
        mock_model.predict.assert_not_called()


# =============================================================================
# Test: Score Normalization
# =============================================================================


class TestScoreNormalization:
    """Tests for score normalization after reranking."""

    def test_scores_normalized_to_0_1(self):
        """Reranker scores should be normalized to 0-1 range."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
        )

        mock_model = MagicMock()
        # Cross-encoders can output arbitrary scores
        mock_model.predict.return_value = [-2.5, 1.0, 5.0]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(normalize_scores=True),
        )
        reranker._model = mock_model

        results = reranker.rerank(
            query="test",
            documents=[
                {"id": "doc1", "content": "a"},
                {"id": "doc2", "content": "b"},
                {"id": "doc3", "content": "c"},
            ],
        )

        # All scores should be in 0-1 range
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_can_preserve_raw_scores(self):
        """Should be able to preserve raw scores when requested."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(normalize_scores=False),
        )
        reranker._model = mock_model

        results = reranker.rerank(
            query="test",
            documents=[{"id": "doc1", "content": "a"}],
        )

        # Raw score preserved
        assert results[0].score == 5.0


# =============================================================================
# Test: Performance
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_latency_under_50ms(self):
        """Reranking should complete under 50ms for typical workloads."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
        )

        mock_model = MagicMock()
        # Simulate fast inference
        mock_model.predict.return_value = [0.5] * 20

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(top_k=20),
        )
        reranker._model = mock_model

        documents = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(20)]

        start = time.perf_counter()
        results = reranker.rerank(query="test", documents=documents)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly with mocked model
        assert elapsed_ms < 50

    def test_timeout_handling(self):
        """Should handle timeout gracefully."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
            RerankerTimeoutError,
        )

        mock_model = MagicMock()

        def slow_predict(*args, **kwargs):
            time.sleep(0.1)  # 100ms - exceeds timeout
            return [0.5]

        mock_model.predict.side_effect = slow_predict

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(timeout_ms=10),  # 10ms timeout
        )
        reranker._model = mock_model

        # Should timeout
        with pytest.raises(RerankerTimeoutError):
            reranker.rerank(
                query="test",
                documents=[{"id": "doc1", "content": "a"}],
            )

    def test_batch_processing(self):
        """Should batch documents for efficient processing."""
        from openmemory.api.retrieval.reranker import (
            RerankerConfig,
            CrossEncoderReranker,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5] * 10

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(batch_size=10, top_k=30),  # top_k=30 to rerank all 25
        )
        reranker._model = mock_model

        documents = [{"id": f"doc{i}", "content": f"content {i}"} for i in range(25)]

        results = reranker.rerank(query="test", documents=documents)

        # Should have processed in batches (3 calls for 25 docs with batch_size=10)
        assert mock_model.predict.call_count == 3


# =============================================================================
# Test: Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_create_cross_encoder_reranker(self):
        """Factory should create cross-encoder reranker."""
        from openmemory.api.retrieval.reranker import (
            create_reranker,
            CrossEncoderReranker,
        )

        reranker = create_reranker(
            backend="cross-encoder",
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_cohere_reranker(self):
        """Factory should create Cohere reranker."""
        from openmemory.api.retrieval.reranker import (
            create_reranker,
            CohereReranker,
        )

        reranker = create_reranker(
            backend="cohere",
            api_key="test-key",
            model_name="rerank-english-v3.0",
        )

        assert isinstance(reranker, CohereReranker)

    def test_create_noop_reranker(self):
        """Factory should create no-op reranker."""
        from openmemory.api.retrieval.reranker import (
            create_reranker,
            NoOpReranker,
        )

        reranker = create_reranker(backend="noop")

        assert isinstance(reranker, NoOpReranker)

    def test_factory_with_config(self):
        """Factory should accept config."""
        from openmemory.api.retrieval.reranker import (
            create_reranker,
            RerankerConfig,
        )

        config = RerankerConfig(top_k=50)
        reranker = create_reranker(
            backend="noop",
            config=config,
        )

        # NoOp doesn't use config but should accept it
        assert reranker is not None

    def test_factory_invalid_backend(self):
        """Factory should raise on invalid backend."""
        from openmemory.api.retrieval.reranker import (
            create_reranker,
            RerankerError,
        )

        with pytest.raises(RerankerError, match="Unknown backend"):
            create_reranker(backend="invalid")


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_document(self):
        """Should handle single document."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        results = reranker.rerank(
            query="test",
            documents=[{"id": "doc1", "content": "only doc"}],
        )

        assert len(results) == 1
        assert results[0].new_rank == 1

    def test_empty_query(self):
        """Should handle empty query."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
            RerankerError,
        )

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = MagicMock()

        with pytest.raises(RerankerError, match="[Ee]mpty query"):
            reranker.rerank(query="", documents=[{"id": "doc1", "content": "a"}])

    def test_document_without_content(self):
        """Should handle document without content field."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        # Document with text field instead of content
        results = reranker.rerank(
            query="test",
            documents=[{"id": "doc1", "text": "some text"}],
        )

        # Should still work using alternative field
        assert len(results) == 1

    def test_very_long_content(self):
        """Should handle very long content by truncating."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(max_length=512),
        )
        reranker._model = mock_model

        # Very long content
        results = reranker.rerank(
            query="test",
            documents=[{"id": "doc1", "content": "x" * 10000}],
        )

        # Should succeed (content truncated internally)
        assert len(results) == 1

    def test_unicode_content(self):
        """Should handle unicode content."""
        from openmemory.api.retrieval.reranker import (
            CrossEncoderReranker,
            RerankerConfig,
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]

        reranker = CrossEncoderReranker(
            model_name="test-model",
            config=RerankerConfig(),
        )
        reranker._model = mock_model

        results = reranker.rerank(
            query="unicode test 日本語",
            documents=[{"id": "doc1", "content": "日本語テキスト 🚀"}],
        )

        assert len(results) == 1


# =============================================================================
# Test: RerankedTriHybridResult
# =============================================================================


class TestRerankedTriHybridResult:
    """Tests for the reranked tri-hybrid result type."""

    def test_includes_rerank_timing(self):
        """Result should include reranking timing."""
        from openmemory.api.retrieval.reranker import RerankedTriHybridResult
        from openmemory.api.retrieval.trihybrid import TriHybridHit

        result = RerankedTriHybridResult(
            hits=[TriHybridHit(id="doc1", score=0.9, source={}, sources={})],
            total=1,
            timing=MagicMock(rerank_ms=15.0, total_ms=65.0),
            reranker_available=True,
        )

        assert result.timing.rerank_ms == 15.0

    def test_includes_reranker_status(self):
        """Result should indicate reranker status."""
        from openmemory.api.retrieval.reranker import RerankedTriHybridResult
        from openmemory.api.retrieval.trihybrid import TriHybridHit

        result = RerankedTriHybridResult(
            hits=[TriHybridHit(id="doc1", score=0.9, source={}, sources={})],
            total=1,
            timing=MagicMock(rerank_ms=0, total_ms=50.0),
            reranker_available=False,
            reranker_error="Model not loaded",
        )

        assert result.reranker_available is False
        assert result.reranker_error == "Model not loaded"
