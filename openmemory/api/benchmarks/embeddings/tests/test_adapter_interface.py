"""
Unit tests for embedder adapter interface and contracts.

All embedder adapters must pass these tests to ensure consistent behavior.

Tests written FIRST following TDD approach.
"""

import pytest
from typing import List
from abc import ABC

# Import will fail until implementation is written (TDD red phase)
from benchmarks.embeddings.adapters.base import EmbedderAdapter, EmbedderInfo


class TestEmbedderAdapterContract:
    """Tests that ALL embedder adapters must pass.

    These are contract tests - they define what every adapter must do.
    """

    def test_adapter_is_abstract_base_class(self):
        """EmbedderAdapter should be an ABC that cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbedderAdapter()

    def test_adapter_requires_info_property(self):
        """Adapter subclass must implement info property."""
        class IncompleteAdapter(EmbedderAdapter):
            def embed(self, text: str) -> List[float]:
                return []
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return []

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_adapter_requires_embed_method(self):
        """Adapter subclass must implement embed method."""
        class IncompleteAdapter(EmbedderAdapter):
            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo("test", "test", 128, 512)
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return []

        with pytest.raises(TypeError):
            IncompleteAdapter()

    def test_adapter_requires_embed_batch_method(self):
        """Adapter subclass must implement embed_batch method."""
        class IncompleteAdapter(EmbedderAdapter):
            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo("test", "test", 128, 512)
            def embed(self, text: str) -> List[float]:
                return []

        with pytest.raises(TypeError):
            IncompleteAdapter()


class TestEmbedderInfoDataclass:
    """Test EmbedderInfo dataclass."""

    def test_embedder_info_has_model_name(self):
        """EmbedderInfo should have model_name attribute."""
        info = EmbedderInfo(
            model_name="test-model",
            provider="local",
            dimensions=1024,
            max_sequence_length=8192
        )
        assert info.model_name == "test-model"

    def test_embedder_info_has_provider(self):
        """EmbedderInfo should have provider attribute."""
        info = EmbedderInfo("test", "local", 1024, 8192)
        assert info.provider == "local"

    def test_embedder_info_has_dimensions(self):
        """EmbedderInfo should have dimensions attribute."""
        info = EmbedderInfo("test", "local", 1024, 8192)
        assert info.dimensions == 1024

    def test_embedder_info_has_max_sequence_length(self):
        """EmbedderInfo should have max_sequence_length attribute."""
        info = EmbedderInfo("test", "local", 1024, 8192)
        assert info.max_sequence_length == 8192

    def test_embedder_info_has_is_code_optimized_default_false(self):
        """EmbedderInfo should have is_code_optimized with default False."""
        info = EmbedderInfo("test", "local", 1024, 8192)
        assert info.is_code_optimized == False

    def test_embedder_info_can_set_code_optimized(self):
        """EmbedderInfo should allow setting is_code_optimized."""
        info = EmbedderInfo("test", "local", 1024, 8192, is_code_optimized=True)
        assert info.is_code_optimized == True


class TestMockAdapter:
    """Test with a mock adapter implementation."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter for testing."""
        class MockAdapter(EmbedderAdapter):
            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo(
                    model_name="mock-model",
                    provider="test",
                    dimensions=128,
                    max_sequence_length=512,
                    is_code_optimized=False
                )

            def embed(self, text: str) -> List[float]:
                # Return a deterministic embedding based on text length
                return [float(len(text)) / 100.0] * 128

            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [self.embed(t) for t in texts]

        return MockAdapter()

    def test_embed_returns_list_of_floats(self, mock_adapter):
        """Embed should return a list of floats."""
        result = mock_adapter.embed("test text")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_embed_returns_correct_dimensions(self, mock_adapter):
        """Embedding vector should match declared dimensions."""
        result = mock_adapter.embed("test text")

        assert len(result) == mock_adapter.info.dimensions

    def test_embed_batch_returns_list_of_embeddings(self, mock_adapter):
        """Embed_batch should return list of embedding vectors."""
        texts = ["text one", "text two", "text three"]
        result = mock_adapter.embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == len(texts)
        assert all(isinstance(emb, list) for emb in result)

    def test_embed_batch_returns_same_length_as_input(self, mock_adapter):
        """Batch embed must return same number of vectors as inputs."""
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        result = mock_adapter.embed_batch(texts)

        assert len(result) == len(texts)

    def test_embed_batch_each_correct_dimensions(self, mock_adapter):
        """Each embedding in batch should have correct dimensions."""
        texts = ["text one", "text two"]
        result = mock_adapter.embed_batch(texts)

        for emb in result:
            assert len(emb) == mock_adapter.info.dimensions

    def test_embed_handles_empty_string(self, mock_adapter):
        """Adapter should handle empty string without crashing."""
        result = mock_adapter.embed("")

        assert isinstance(result, list)
        assert len(result) == mock_adapter.info.dimensions

    def test_embed_handles_unicode(self, mock_adapter):
        """Adapter should handle unicode/special characters."""
        result = mock_adapter.embed("Hello 世界 🌍 مرحبا")

        assert isinstance(result, list)
        assert len(result) == mock_adapter.info.dimensions

    def test_embed_is_deterministic(self, mock_adapter):
        """Same input should produce same embedding (for non-stochastic models)."""
        text = "test determinism"
        result1 = mock_adapter.embed(text)
        result2 = mock_adapter.embed(text)

        assert result1 == result2

    def test_warmup_method_exists(self, mock_adapter):
        """Adapter should have warmup method."""
        assert hasattr(mock_adapter, "warmup")
        # Should not raise
        mock_adapter.warmup()


class TestEmbedderAdapterPerformance:
    """Performance contract tests."""

    @pytest.fixture
    def slow_mock_adapter(self):
        """Create a mock adapter with controlled timing."""
        import time

        class SlowMockAdapter(EmbedderAdapter):
            def __init__(self, delay_ms: float = 0):
                self._delay_ms = delay_ms

            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo("slow-mock", "test", 64, 512)

            def embed(self, text: str) -> List[float]:
                time.sleep(self._delay_ms / 1000)
                return [0.1] * 64

            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                # Batch should be more efficient than N individual calls
                time.sleep(self._delay_ms / 1000)  # Only one delay
                return [[0.1] * 64 for _ in texts]

        return SlowMockAdapter

    def test_single_embed_completes(self, slow_mock_adapter):
        """Single embed call should complete in reasonable time."""
        adapter = slow_mock_adapter(delay_ms=10)
        result = adapter.embed("test")

        assert len(result) == 64

    def test_batch_more_efficient_than_loop(self, slow_mock_adapter):
        """Batch of N should be faster than N individual calls."""
        import time

        adapter = slow_mock_adapter(delay_ms=5)
        texts = ["text"] * 10

        # Time batch call
        start = time.perf_counter()
        adapter.embed_batch(texts)
        batch_time = time.perf_counter() - start

        # Time individual calls
        start = time.perf_counter()
        for t in texts:
            adapter.embed(t)
        loop_time = time.perf_counter() - start

        # Batch should be faster (at least 2x with our mock)
        assert batch_time < loop_time


class TestAdapterRegistry:
    """Test adapter registration pattern."""

    def test_adapter_can_be_identified_by_info(self):
        """Adapters should be identifiable by their info."""
        class AdapterA(EmbedderAdapter):
            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo("model-a", "local", 128, 512)
            def embed(self, text: str) -> List[float]:
                return [0.0] * 128
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.0] * 128 for _ in texts]

        class AdapterB(EmbedderAdapter):
            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo("model-b", "gemini", 256, 1024)
            def embed(self, text: str) -> List[float]:
                return [0.0] * 256
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.0] * 256 for _ in texts]

        a = AdapterA()
        b = AdapterB()

        assert a.info.model_name != b.info.model_name
        assert a.info.provider != b.info.provider
        assert a.info.dimensions != b.info.dimensions
