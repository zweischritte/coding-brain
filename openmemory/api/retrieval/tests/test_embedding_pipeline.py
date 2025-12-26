"""Tests for embedding pipeline with shadow model support (FR-004).

This module tests the embedding flexibility and shadow pipeline system:
- Content-addressed embedding storage
- Shadow embedding pipeline for A/B testing
- Multi-model support with configuration
- Embedding comparison and quality metrics
- Automatic fallback handling
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pytest

# These imports will fail until we implement the module
from openmemory.api.retrieval.embedding_pipeline import (
    # Core types
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingResult,
    EmbeddingBatch,
    ContentHash,
    # Storage
    EmbeddingStore,
    InMemoryEmbeddingStore,
    ContentAddressedEmbeddingStore,
    # Pipeline
    EmbeddingPipeline,
    ShadowPipeline,
    ShadowResult,
    ShadowComparison,
    # Providers
    EmbeddingProvider,
    OllamaProvider,
    OpenAIProvider,
    # Metrics
    EmbeddingMetrics,
    SimilarityMetric,
    # Errors
    EmbeddingError,
    ProviderError,
    StorageError,
    # Factory
    create_embedding_pipeline,
    create_shadow_pipeline,
)


# =============================================================================
# EmbeddingConfig Tests
# =============================================================================


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.model_name == "nomic-embed-text"
        assert config.dimension == 768
        assert config.max_batch_size == 32
        assert config.timeout_seconds == 30
        assert config.normalize is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="qwen3-embedding:8b",
            dimension=1024,
            max_batch_size=16,
            timeout_seconds=60,
            normalize=False,
        )
        assert config.model_name == "qwen3-embedding:8b"
        assert config.dimension == 1024
        assert config.max_batch_size == 16
        assert config.timeout_seconds == 60
        assert config.normalize is False

    def test_config_validation_dimension(self):
        """Test validation rejects invalid dimension."""
        with pytest.raises(ValueError, match="dimension"):
            EmbeddingConfig(dimension=0)

        with pytest.raises(ValueError, match="dimension"):
            EmbeddingConfig(dimension=-1)

    def test_config_validation_batch_size(self):
        """Test validation rejects invalid batch size."""
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingConfig(max_batch_size=0)

    def test_config_with_provider_options(self):
        """Test config with provider-specific options."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            provider_options={
                "api_base": "https://api.openai.com/v1",
                "model_version": "latest",
            },
        )
        assert config.provider_options["api_base"] == "https://api.openai.com/v1"


# =============================================================================
# EmbeddingModel Tests
# =============================================================================


class TestEmbeddingModel:
    """Tests for EmbeddingModel dataclass."""

    def test_model_creation(self):
        """Test creating an embedding model descriptor."""
        model = EmbeddingModel(
            model_id="nomic-embed-text",
            provider="ollama",
            dimension=768,
            description="Local embedding model via Ollama",
        )
        assert model.model_id == "nomic-embed-text"
        assert model.provider == "ollama"
        assert model.dimension == 768

    def test_model_equality(self):
        """Test model equality comparison."""
        model1 = EmbeddingModel(
            model_id="model-a",
            provider="ollama",
            dimension=768,
        )
        model2 = EmbeddingModel(
            model_id="model-a",
            provider="ollama",
            dimension=768,
        )
        model3 = EmbeddingModel(
            model_id="model-b",
            provider="ollama",
            dimension=768,
        )

        assert model1 == model2
        assert model1 != model3


# =============================================================================
# ContentHash Tests
# =============================================================================


class TestContentHash:
    """Tests for content-addressed hashing."""

    def test_hash_text(self):
        """Test hashing text content."""
        content = "def hello_world():\n    print('Hello')"
        hash_result = ContentHash.from_text(content)

        assert hash_result.hash is not None
        assert len(hash_result.hash) == 64  # SHA-256 hex
        assert hash_result.content_type == "text"

    def test_same_content_same_hash(self):
        """Test deterministic hashing."""
        content = "same content"
        hash1 = ContentHash.from_text(content)
        hash2 = ContentHash.from_text(content)

        assert hash1.hash == hash2.hash

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        hash1 = ContentHash.from_text("content A")
        hash2 = ContentHash.from_text("content B")

        assert hash1.hash != hash2.hash

    def test_hash_with_model_version(self):
        """Test hash includes model version for cache busting."""
        content = "same content"
        hash1 = ContentHash.from_text(content, model_version="v1")
        hash2 = ContentHash.from_text(content, model_version="v2")

        assert hash1.hash != hash2.hash

    def test_hash_string_representation(self):
        """Test hash has useful string representation."""
        content = "test content"
        hash_result = ContentHash.from_text(content)

        hash_str = str(hash_result)
        assert hash_result.hash[:8] in hash_str


# =============================================================================
# EmbeddingResult Tests
# =============================================================================


class TestEmbeddingResult:
    """Tests for EmbeddingResult."""

    def test_result_creation(self):
        """Test creating embedding result."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = EmbeddingResult(
            content_hash=ContentHash.from_text("test"),
            embedding=embedding,
            model_id="test-model",
            dimension=5,
        )

        assert result.embedding == embedding
        assert result.model_id == "test-model"
        assert result.dimension == 5

    def test_result_validation(self):
        """Test result validates dimension matches embedding."""
        embedding = [0.1, 0.2, 0.3]

        with pytest.raises(ValueError, match="dimension"):
            EmbeddingResult(
                content_hash=ContentHash.from_text("test"),
                embedding=embedding,
                model_id="test-model",
                dimension=5,  # Mismatch!
            )

    def test_result_normalization(self):
        """Test embedding normalization."""
        embedding = [3.0, 4.0]  # Norm = 5
        result = EmbeddingResult(
            content_hash=ContentHash.from_text("test"),
            embedding=embedding,
            model_id="test-model",
            dimension=2,
            normalized=False,
        )

        normalized = result.normalize()
        assert abs(normalized.embedding[0] - 0.6) < 0.001
        assert abs(normalized.embedding[1] - 0.8) < 0.001
        assert normalized.normalized is True

    def test_result_magnitude(self):
        """Test embedding magnitude calculation."""
        embedding = [3.0, 4.0]  # Magnitude = 5
        result = EmbeddingResult(
            content_hash=ContentHash.from_text("test"),
            embedding=embedding,
            model_id="test-model",
            dimension=2,
        )

        assert abs(result.magnitude - 5.0) < 0.001


# =============================================================================
# EmbeddingBatch Tests
# =============================================================================


class TestEmbeddingBatch:
    """Tests for EmbeddingBatch."""

    def test_batch_creation(self):
        """Test creating a batch of embeddings."""
        contents = ["text1", "text2", "text3"]
        batch = EmbeddingBatch.from_texts(contents)

        assert batch.size == 3
        assert len(batch.content_hashes) == 3

    def test_batch_iteration(self):
        """Test iterating over batch."""
        contents = ["text1", "text2"]
        batch = EmbeddingBatch.from_texts(contents)

        items = list(batch)
        assert len(items) == 2

    def test_batch_with_embeddings(self):
        """Test batch with completed embeddings."""
        contents = ["text1", "text2"]
        batch = EmbeddingBatch.from_texts(contents)

        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        completed = batch.with_embeddings(embeddings, model_id="test", dimension=2)

        assert len(completed.results) == 2
        assert completed.results[0].embedding == [0.1, 0.2]


# =============================================================================
# InMemoryEmbeddingStore Tests
# =============================================================================


class TestInMemoryEmbeddingStore:
    """Tests for in-memory embedding storage."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving embeddings."""
        store = InMemoryEmbeddingStore()

        content_hash = ContentHash.from_text("test content")
        result = EmbeddingResult(
            content_hash=content_hash,
            embedding=[0.1, 0.2, 0.3],
            model_id="test-model",
            dimension=3,
        )

        store.put(result)
        retrieved = store.get(content_hash, "test-model")

        assert retrieved is not None
        assert retrieved.embedding == [0.1, 0.2, 0.3]

    def test_retrieve_missing(self):
        """Test retrieving non-existent embedding returns None."""
        store = InMemoryEmbeddingStore()

        content_hash = ContentHash.from_text("missing")
        result = store.get(content_hash, "test-model")

        assert result is None

    def test_model_isolation(self):
        """Test embeddings are isolated per model."""
        store = InMemoryEmbeddingStore()

        content_hash = ContentHash.from_text("test content")

        result1 = EmbeddingResult(
            content_hash=content_hash,
            embedding=[0.1, 0.2],
            model_id="model-a",
            dimension=2,
        )
        result2 = EmbeddingResult(
            content_hash=content_hash,
            embedding=[0.3, 0.4],
            model_id="model-b",
            dimension=2,
        )

        store.put(result1)
        store.put(result2)

        retrieved_a = store.get(content_hash, "model-a")
        retrieved_b = store.get(content_hash, "model-b")

        assert retrieved_a.embedding == [0.1, 0.2]
        assert retrieved_b.embedding == [0.3, 0.4]

    def test_bulk_get(self):
        """Test bulk retrieval of embeddings."""
        store = InMemoryEmbeddingStore()

        # Store 3 embeddings
        hashes = []
        for i in range(3):
            content_hash = ContentHash.from_text(f"content{i}")
            hashes.append(content_hash)
            store.put(
                EmbeddingResult(
                    content_hash=content_hash,
                    embedding=[float(i)],
                    model_id="test-model",
                    dimension=1,
                )
            )

        # Bulk retrieve
        results = store.bulk_get(hashes, "test-model")

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_delete(self):
        """Test deleting embeddings."""
        store = InMemoryEmbeddingStore()

        content_hash = ContentHash.from_text("test")
        store.put(
            EmbeddingResult(
                content_hash=content_hash,
                embedding=[0.1],
                model_id="test-model",
                dimension=1,
            )
        )

        assert store.get(content_hash, "test-model") is not None

        store.delete(content_hash, "test-model")

        assert store.get(content_hash, "test-model") is None


# =============================================================================
# ContentAddressedEmbeddingStore Tests
# =============================================================================


class TestContentAddressedEmbeddingStore:
    """Tests for content-addressed embedding storage."""

    def test_deduplication(self):
        """Test identical content uses same embedding."""
        store = ContentAddressedEmbeddingStore()

        content = "identical content"
        hash1 = ContentHash.from_text(content)
        hash2 = ContentHash.from_text(content)

        assert hash1.hash == hash2.hash

        store.put(
            EmbeddingResult(
                content_hash=hash1,
                embedding=[0.1, 0.2],
                model_id="test-model",
                dimension=2,
            )
        )

        # Second put with same hash should not duplicate
        store.put(
            EmbeddingResult(
                content_hash=hash2,
                embedding=[0.1, 0.2],
                model_id="test-model",
                dimension=2,
            )
        )

        assert store.size("test-model") == 1

    def test_stats(self):
        """Test storage statistics."""
        store = ContentAddressedEmbeddingStore()

        for i in range(5):
            store.put(
                EmbeddingResult(
                    content_hash=ContentHash.from_text(f"content{i}"),
                    embedding=[float(i)] * 10,
                    model_id="test-model",
                    dimension=10,
                )
            )

        stats = store.stats("test-model")
        assert stats.entry_count == 5
        assert stats.estimated_memory_bytes > 0


# =============================================================================
# EmbeddingProvider Tests
# =============================================================================


class TestEmbeddingProvider:
    """Tests for embedding provider interface."""

    def test_provider_interface(self):
        """Test provider implements required interface."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 3

        result = mock_provider.embed(["test text"])
        assert result == [[0.1, 0.2, 0.3]]


class TestOllamaProvider:
    """Tests for Ollama embedding provider."""

    def test_create_provider(self):
        """Test creating Ollama provider."""
        config = EmbeddingConfig(
            model_name="nomic-embed-text",
            dimension=768,
        )
        provider = OllamaProvider(config)

        assert provider.model_id == "nomic-embed-text"
        assert provider.dimension == 768

    def test_embed_single_text(self):
        """Test embedding single text."""
        config = EmbeddingConfig(model_name="nomic-embed-text", dimension=768)
        provider = OllamaProvider(config)

        with patch.object(provider, "_call_ollama") as mock_call:
            mock_call.return_value = [[0.1] * 768]

            result = provider.embed(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 768

    def test_embed_batch(self):
        """Test embedding batch of texts."""
        config = EmbeddingConfig(model_name="nomic-embed-text", dimension=768)
        provider = OllamaProvider(config)

        with patch.object(provider, "_call_ollama") as mock_call:
            mock_call.return_value = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

            result = provider.embed(["text1", "text2", "text3"])

            assert len(result) == 3

    def test_provider_timeout(self):
        """Test provider respects timeout."""
        config = EmbeddingConfig(
            model_name="nomic-embed-text",
            dimension=768,
            timeout_seconds=5,
        )
        provider = OllamaProvider(config)

        assert provider.timeout == 5


class TestOpenAIProvider:
    """Tests for OpenAI embedding provider."""

    def test_create_provider(self):
        """Test creating OpenAI provider."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            dimension=1536,
            provider_options={"api_key": "test-key"},
        )
        provider = OpenAIProvider(config)

        assert provider.model_id == "text-embedding-3-small"
        assert provider.dimension == 1536

    def test_embed_with_api_call(self):
        """Test embedding calls OpenAI API."""
        config = EmbeddingConfig(
            model_name="text-embedding-3-small",
            dimension=1536,
            provider_options={"api_key": "test-key"},
        )
        provider = OpenAIProvider(config)

        with patch.object(provider, "_call_openai") as mock_call:
            mock_call.return_value = [[0.1] * 1536]

            result = provider.embed(["test text"])

            assert len(result) == 1
            assert len(result[0]) == 1536


# =============================================================================
# EmbeddingPipeline Tests
# =============================================================================


class TestEmbeddingPipeline:
    """Tests for main embedding pipeline."""

    def test_create_pipeline(self):
        """Test creating embedding pipeline."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768

        pipeline = EmbeddingPipeline(
            provider=mock_provider,
            store=InMemoryEmbeddingStore(),
        )

        assert pipeline.provider == mock_provider

    def test_embed_with_cache_miss(self):
        """Test embedding with cache miss calls provider."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768
        mock_provider.embed.return_value = [[0.1] * 768]

        pipeline = EmbeddingPipeline(
            provider=mock_provider,
            store=InMemoryEmbeddingStore(),
        )

        result = pipeline.embed("test text")

        mock_provider.embed.assert_called_once()
        assert result.dimension == 768

    def test_embed_with_cache_hit(self):
        """Test embedding with cache hit skips provider."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768
        mock_provider.embed.return_value = [[0.1] * 768]

        store = InMemoryEmbeddingStore()
        pipeline = EmbeddingPipeline(provider=mock_provider, store=store)

        # First call - cache miss
        pipeline.embed("test text")
        assert mock_provider.embed.call_count == 1

        # Second call - cache hit
        pipeline.embed("test text")
        assert mock_provider.embed.call_count == 1  # Still 1

    def test_embed_batch(self):
        """Test batch embedding."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768
        mock_provider.embed.return_value = [[0.1] * 768, [0.2] * 768, [0.3] * 768]

        pipeline = EmbeddingPipeline(
            provider=mock_provider,
            store=InMemoryEmbeddingStore(),
        )

        results = pipeline.embed_batch(["text1", "text2", "text3"])

        assert len(results) == 3

    def test_embed_batch_partial_cache(self):
        """Test batch embedding with partial cache hits."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768

        store = InMemoryEmbeddingStore()
        pipeline = EmbeddingPipeline(provider=mock_provider, store=store)

        # Pre-cache one embedding
        store.put(
            EmbeddingResult(
                content_hash=ContentHash.from_text("text1"),
                embedding=[0.1] * 768,
                model_id="test-model",
                dimension=768,
            )
        )

        mock_provider.embed.return_value = [[0.2] * 768, [0.3] * 768]

        results = pipeline.embed_batch(["text1", "text2", "text3"])

        assert len(results) == 3
        # Provider should only be called for uncached texts
        mock_provider.embed.assert_called_once()
        call_args = mock_provider.embed.call_args[0][0]
        assert "text1" not in call_args


# =============================================================================
# ShadowPipeline Tests
# =============================================================================


class TestShadowPipeline:
    """Tests for shadow embedding pipeline."""

    def test_create_shadow_pipeline(self):
        """Test creating shadow pipeline with two models."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 768

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 1024

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
        )

        assert pipeline.primary_provider == primary_provider
        assert pipeline.shadow_provider == shadow_provider

    def test_shadow_returns_primary_result(self):
        """Test shadow pipeline returns primary model result."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 768
        primary_provider.embed.return_value = [[0.1] * 768]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 768
        shadow_provider.embed.return_value = [[0.2] * 768]

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
        )

        result = pipeline.embed("test text")

        # Should return primary result
        assert result.model_id == "primary-model"

    def test_shadow_runs_both_models(self):
        """Test shadow pipeline runs both models."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 768
        primary_provider.embed.return_value = [[0.1] * 768]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 768
        shadow_provider.embed.return_value = [[0.2] * 768]

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
        )

        pipeline.embed("test text")

        primary_provider.embed.assert_called_once()
        shadow_provider.embed.assert_called_once()

    def test_shadow_comparison(self):
        """Test shadow pipeline compares results."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 3
        primary_provider.embed.return_value = [[1.0, 0.0, 0.0]]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 3
        shadow_provider.embed.return_value = [[0.0, 1.0, 0.0]]

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
            compare_results=True,
        )

        result = pipeline.embed_with_comparison("test text")

        assert isinstance(result, ShadowResult)
        assert result.primary_result is not None
        assert result.shadow_result is not None
        assert result.comparison is not None

    def test_shadow_comparison_similarity(self):
        """Test shadow comparison calculates similarity."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 3
        primary_provider.embed.return_value = [[0.6, 0.8, 0.0]]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 3
        shadow_provider.embed.return_value = [[0.6, 0.8, 0.0]]  # Same

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
            compare_results=True,
        )

        result = pipeline.embed_with_comparison("test text")

        # Should have high similarity
        assert result.comparison.cosine_similarity > 0.99

    def test_shadow_continues_on_shadow_failure(self):
        """Test shadow pipeline continues if shadow fails."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 768
        primary_provider.embed.return_value = [[0.1] * 768]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 768
        shadow_provider.embed.side_effect = ProviderError("Shadow failed")

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
        )

        # Should not raise, should return primary result
        result = pipeline.embed("test text")
        assert result.model_id == "primary-model"

    def test_shadow_async_execution(self):
        """Test shadow runs asynchronously by default."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 768
        primary_provider.embed.return_value = [[0.1] * 768]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 768
        shadow_provider.embed.return_value = [[0.2] * 768]

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
            async_shadow=True,
        )

        # Primary result should return quickly
        result = pipeline.embed("test text")
        assert result is not None

    def test_shadow_sampling_rate(self):
        """Test shadow respects sampling rate."""
        primary_provider = Mock(spec=EmbeddingProvider)
        primary_provider.model_id = "primary-model"
        primary_provider.dimension = 768
        primary_provider.embed.return_value = [[0.1] * 768]

        shadow_provider = Mock(spec=EmbeddingProvider)
        shadow_provider.model_id = "shadow-model"
        shadow_provider.dimension = 768
        shadow_provider.embed.return_value = [[0.2] * 768]

        pipeline = ShadowPipeline(
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
            sampling_rate=0.0,  # Never run shadow
        )

        pipeline.embed("test text")

        primary_provider.embed.assert_called_once()
        shadow_provider.embed.assert_not_called()


# =============================================================================
# ShadowComparison Tests
# =============================================================================


class TestShadowComparison:
    """Tests for embedding comparison metrics."""

    def test_comparison_creation(self):
        """Test creating a comparison."""
        comparison = ShadowComparison(
            cosine_similarity=0.95,
            euclidean_distance=0.5,
            primary_latency_ms=10.0,
            shadow_latency_ms=15.0,
        )

        assert comparison.cosine_similarity == 0.95
        assert comparison.euclidean_distance == 0.5

    def test_comparison_from_embeddings(self):
        """Test creating comparison from embeddings."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.707, 0.707, 0.0]  # 45 degree angle

        comparison = ShadowComparison.from_embeddings(
            emb1, emb2, primary_latency_ms=10.0, shadow_latency_ms=15.0
        )

        assert abs(comparison.cosine_similarity - 0.707) < 0.01

    def test_comparison_identical_embeddings(self):
        """Test comparison of identical embeddings."""
        emb = [0.6, 0.8, 0.0]

        comparison = ShadowComparison.from_embeddings(
            emb, emb, primary_latency_ms=10.0, shadow_latency_ms=10.0
        )

        assert abs(comparison.cosine_similarity - 1.0) < 0.001


# =============================================================================
# EmbeddingMetrics Tests
# =============================================================================


class TestEmbeddingMetrics:
    """Tests for embedding metrics tracking."""

    def test_metrics_creation(self):
        """Test creating metrics tracker."""
        metrics = EmbeddingMetrics()

        assert metrics.total_embeddings == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_metrics_recording(self):
        """Test recording metrics."""
        metrics = EmbeddingMetrics()

        metrics.record_embedding(cached=False, latency_ms=10.0)
        metrics.record_embedding(cached=True, latency_ms=1.0)
        metrics.record_embedding(cached=True, latency_ms=1.0)

        assert metrics.total_embeddings == 3
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1

    def test_metrics_cache_rate(self):
        """Test cache hit rate calculation."""
        metrics = EmbeddingMetrics()

        for _ in range(3):
            metrics.record_embedding(cached=True, latency_ms=1.0)
        metrics.record_embedding(cached=False, latency_ms=10.0)

        assert metrics.cache_hit_rate == 0.75

    def test_metrics_average_latency(self):
        """Test average latency calculation."""
        metrics = EmbeddingMetrics()

        metrics.record_embedding(cached=False, latency_ms=10.0)
        metrics.record_embedding(cached=False, latency_ms=20.0)
        metrics.record_embedding(cached=False, latency_ms=30.0)

        assert metrics.average_latency_ms == 20.0


# =============================================================================
# SimilarityMetric Tests
# =============================================================================


class TestSimilarityMetric:
    """Tests for similarity calculations."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical vectors."""
        emb = [0.6, 0.8, 0.0]
        similarity = SimilarityMetric.cosine(emb, emb)

        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]

        similarity = SimilarityMetric.cosine(emb1, emb2)

        assert abs(similarity) < 0.001

    def test_euclidean_distance(self):
        """Test euclidean distance calculation."""
        emb1 = [0.0, 0.0]
        emb2 = [3.0, 4.0]

        distance = SimilarityMetric.euclidean_distance(emb1, emb2)

        assert abs(distance - 5.0) < 0.001

    def test_dot_product(self):
        """Test dot product calculation."""
        emb1 = [1.0, 2.0, 3.0]
        emb2 = [4.0, 5.0, 6.0]

        dot = SimilarityMetric.dot_product(emb1, emb2)

        assert dot == 32.0  # 1*4 + 2*5 + 3*6


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_embedding_pipeline(self):
        """Test creating embedding pipeline via factory."""
        with patch(
            "openmemory.api.retrieval.embedding_pipeline.OllamaProvider"
        ) as mock_class:
            mock_provider = Mock(spec=EmbeddingProvider)
            mock_provider.model_id = "nomic-embed-text"
            mock_provider.dimension = 768
            mock_class.return_value = mock_provider

            pipeline = create_embedding_pipeline(
                model_name="nomic-embed-text",
                dimension=768,
                provider="ollama",
            )

            assert isinstance(pipeline, EmbeddingPipeline)

    def test_create_shadow_pipeline(self):
        """Test creating shadow pipeline via factory."""
        with patch(
            "openmemory.api.retrieval.embedding_pipeline.OllamaProvider"
        ) as mock_class:
            mock_provider = Mock(spec=EmbeddingProvider)
            mock_provider.model_id = "test"
            mock_provider.dimension = 768
            mock_class.return_value = mock_provider

            pipeline = create_shadow_pipeline(
                primary_model="nomic-embed-text",
                shadow_model="qwen3-embedding:8b",
                primary_dimension=768,
                shadow_dimension=1024,
                sampling_rate=0.5,
            )

            assert isinstance(pipeline, ShadowPipeline)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_provider_error(self):
        """Test provider error is raised correctly."""
        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768
        mock_provider.embed.side_effect = ProviderError("Connection failed")

        pipeline = EmbeddingPipeline(
            provider=mock_provider,
            store=InMemoryEmbeddingStore(),
        )

        with pytest.raises(ProviderError, match="Connection failed"):
            pipeline.embed("test text")

    def test_storage_error(self):
        """Test storage error handling."""
        mock_store = Mock(spec=EmbeddingStore)
        mock_store.get.side_effect = StorageError("Storage unavailable")

        mock_provider = Mock(spec=EmbeddingProvider)
        mock_provider.model_id = "test-model"
        mock_provider.dimension = 768
        mock_provider.embed.return_value = [[0.1] * 768]

        pipeline = EmbeddingPipeline(
            provider=mock_provider,
            store=mock_store,
        )

        # Should continue with provider even if cache fails
        result = pipeline.embed("test text")
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestEmbeddingPipelineIntegration:
    """Integration tests for embedding pipeline."""

    @pytest.mark.integration
    def test_full_embedding_flow(self):
        """Test complete embedding flow with Ollama."""
        pass

    @pytest.mark.integration
    def test_shadow_comparison_flow(self):
        """Test shadow comparison with two models."""
        pass
