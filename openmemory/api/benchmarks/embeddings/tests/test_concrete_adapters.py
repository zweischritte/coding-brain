"""
Unit tests for concrete embedder adapter implementations.

Tests for:
- OllamaAdapter (Qwen3, Nomic)
- GeminiAdapter

These tests use mocking to avoid requiring actual model access.
Integration tests (marked with @pytest.mark.integration) test real models.

Tests written FIRST following TDD approach.
"""

import pytest
from typing import List
from unittest.mock import Mock, patch, MagicMock

from benchmarks.embeddings.adapters.base import EmbedderAdapter, EmbedderInfo


class TestOllamaAdapterContract:
    """Test OllamaAdapter implements EmbedderAdapter correctly."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client."""
        with patch("benchmarks.embeddings.adapters.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": [{"name": "qwen3-embedding-8b"}]}
            mock_client.embeddings.return_value = {"embedding": [0.1] * 1024}
            mock_client_class.return_value = mock_client
            yield mock_client

    def test_ollama_adapter_is_embedder_adapter(self, mock_ollama_client):
        """OllamaAdapter should be an EmbedderAdapter."""
        from benchmarks.embeddings.adapters.ollama import OllamaAdapter

        adapter = OllamaAdapter(model_name="qwen3-embedding-8b", dimensions=1024)

        assert isinstance(adapter, EmbedderAdapter)

    def test_ollama_adapter_has_info(self, mock_ollama_client):
        """OllamaAdapter should have info property."""
        from benchmarks.embeddings.adapters.ollama import OllamaAdapter

        adapter = OllamaAdapter(model_name="qwen3-embedding-8b", dimensions=1024)

        assert adapter.info.model_name == "qwen3-embedding-8b"
        assert adapter.info.provider == "ollama"
        assert adapter.info.dimensions == 1024

    def test_ollama_adapter_embed_returns_list(self, mock_ollama_client):
        """OllamaAdapter.embed should return list of floats."""
        from benchmarks.embeddings.adapters.ollama import OllamaAdapter

        adapter = OllamaAdapter(model_name="qwen3-embedding-8b", dimensions=1024)
        result = adapter.embed("test text")

        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    def test_ollama_adapter_embed_batch_returns_list_of_lists(self, mock_ollama_client):
        """OllamaAdapter.embed_batch should return list of embeddings."""
        from benchmarks.embeddings.adapters.ollama import OllamaAdapter

        adapter = OllamaAdapter(model_name="qwen3-embedding-8b", dimensions=1024)
        texts = ["text one", "text two", "text three"]
        result = adapter.embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(emb) == 1024 for emb in result)


class TestQwen3Adapter:
    """Test Qwen3-specific adapter configurations."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for Qwen3."""
        with patch("benchmarks.embeddings.adapters.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": [{"name": "qwen3-embedding-8b"}]}
            mock_client.embeddings.return_value = {"embedding": [0.1] * 1024}
            mock_client_class.return_value = mock_client
            yield mock_client

    def test_qwen3_8b_adapter_exists(self, mock_ollama_client):
        """Qwen3_8BAdapter should be importable."""
        from benchmarks.embeddings.adapters.qwen3 import Qwen3_8BAdapter

        adapter = Qwen3_8BAdapter()
        assert adapter.info.model_name == "qwen3-embedding-8b"

    def test_qwen3_8b_has_correct_dimensions(self, mock_ollama_client):
        """Qwen3-8B should have 1024 dimensions."""
        from benchmarks.embeddings.adapters.qwen3 import Qwen3_8BAdapter

        adapter = Qwen3_8BAdapter()
        assert adapter.info.dimensions == 1024

    def test_qwen3_8b_is_code_optimized(self, mock_ollama_client):
        """Qwen3-8B should be marked as code optimized."""
        from benchmarks.embeddings.adapters.qwen3 import Qwen3_8BAdapter

        adapter = Qwen3_8BAdapter()
        assert adapter.info.is_code_optimized == True

    def test_qwen3_06b_adapter_exists(self, mock_ollama_client):
        """Qwen3_06BAdapter (fallback) should be importable."""
        mock_ollama_client.list.return_value = {"models": [{"name": "qwen3-embedding-0.6b"}]}
        mock_ollama_client.embeddings.return_value = {"embedding": [0.1] * 1024}

        from benchmarks.embeddings.adapters.qwen3 import Qwen3_06BAdapter

        adapter = Qwen3_06BAdapter()
        assert adapter.info.model_name == "qwen3-embedding-0.6b"
        assert adapter.info.dimensions == 1024


class TestNomicAdapter:
    """Test Nomic Embed Code adapter."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client for Nomic."""
        with patch("benchmarks.embeddings.adapters.ollama.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.list.return_value = {"models": [{"name": "nomic-embed-text"}]}
            mock_client.embeddings.return_value = {"embedding": [0.1] * 768}
            mock_client_class.return_value = mock_client
            yield mock_client

    def test_nomic_adapter_exists(self, mock_ollama_client):
        """NomicCodeAdapter should be importable."""
        from benchmarks.embeddings.adapters.nomic import NomicCodeAdapter

        adapter = NomicCodeAdapter()
        assert adapter.info.model_name == "nomic-embed-text"

    def test_nomic_has_correct_dimensions(self, mock_ollama_client):
        """Nomic should have 768 dimensions."""
        from benchmarks.embeddings.adapters.nomic import NomicCodeAdapter

        adapter = NomicCodeAdapter()
        assert adapter.info.dimensions == 768

    def test_nomic_provider_is_ollama(self, mock_ollama_client):
        """Nomic runs via Ollama."""
        from benchmarks.embeddings.adapters.nomic import NomicCodeAdapter

        adapter = NomicCodeAdapter()
        assert adapter.info.provider == "ollama"


class TestGeminiAdapter:
    """Test Gemini embedding adapter."""

    @pytest.fixture
    def mock_gemini(self):
        """Mock Gemini client and types."""
        with patch("benchmarks.embeddings.adapters.gemini.genai") as mock_genai, \
             patch("benchmarks.embeddings.adapters.gemini.types") as mock_types:
            mock_client = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.models.embed_content.return_value = mock_response
            mock_genai.Client.return_value = mock_client

            # Mock types.EmbedContentConfig
            mock_types.EmbedContentConfig = MagicMock()

            yield mock_client

    def test_gemini_adapter_exists(self, mock_gemini):
        """GeminiAdapter should be importable."""
        from benchmarks.embeddings.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="test-key")
        assert isinstance(adapter, EmbedderAdapter)

    def test_gemini_has_correct_model(self, mock_gemini):
        """Gemini should use gemini-embedding-001."""
        from benchmarks.embeddings.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="test-key")
        assert "gemini-embedding" in adapter.info.model_name

    def test_gemini_provider_is_gemini(self, mock_gemini):
        """Provider should be 'gemini'."""
        from benchmarks.embeddings.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="test-key")
        assert adapter.info.provider == "gemini"

    def test_gemini_embed_returns_vector(self, mock_gemini):
        """Gemini embed should return embedding vector."""
        from benchmarks.embeddings.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="test-key")
        result = adapter.embed("test text")

        assert isinstance(result, list)
        assert len(result) == adapter.info.dimensions

    def test_gemini_supports_reduced_dimensions(self, mock_gemini):
        """Gemini supports Matryoshka dimension reduction."""
        # Update mock to return 768 dims
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_gemini.models.embed_content.return_value = mock_response

        from benchmarks.embeddings.adapters.gemini import GeminiAdapter

        adapter = GeminiAdapter(api_key="test-key", dimensions=768)
        assert adapter.info.dimensions == 768


class TestAdapterFactory:
    """Test adapter factory for creating adapters by name."""

    def test_factory_creates_qwen3_8b(self):
        """Factory should create Qwen3_8BAdapter."""
        with patch("benchmarks.embeddings.adapters.ollama.Client") as mock:
            mock.return_value.list.return_value = {"models": [{"name": "qwen3-embedding-8b"}]}
            mock.return_value.embeddings.return_value = {"embedding": [0.1] * 1024}

            from benchmarks.embeddings.adapters import create_adapter

            adapter = create_adapter("qwen3-8b")
            assert adapter.info.model_name == "qwen3-embedding-8b"

    def test_factory_creates_nomic(self):
        """Factory should create NomicCodeAdapter."""
        with patch("benchmarks.embeddings.adapters.ollama.Client") as mock:
            mock.return_value.list.return_value = {"models": [{"name": "nomic-embed-text"}]}
            mock.return_value.embeddings.return_value = {"embedding": [0.1] * 768}

            from benchmarks.embeddings.adapters import create_adapter

            adapter = create_adapter("nomic")
            assert "nomic" in adapter.info.model_name.lower()

    def test_factory_creates_gemini(self):
        """Factory should create GeminiAdapter."""
        with patch("benchmarks.embeddings.adapters.gemini.genai"):
            from benchmarks.embeddings.adapters import create_adapter

            adapter = create_adapter("gemini", api_key="test-key")
            assert adapter.info.provider == "gemini"

    def test_factory_unknown_adapter_raises(self):
        """Unknown adapter name should raise ValueError."""
        from benchmarks.embeddings.adapters import create_adapter

        with pytest.raises(ValueError):
            create_adapter("unknown-model")


class TestAdapterConfigOptions:
    """Test adapter configuration options."""

    def test_ollama_adapter_accepts_base_url(self):
        """OllamaAdapter should accept custom base_url."""
        with patch("benchmarks.embeddings.adapters.ollama.Client") as mock:
            mock.return_value.list.return_value = {"models": [{"name": "test"}]}

            from benchmarks.embeddings.adapters.ollama import OllamaAdapter

            adapter = OllamaAdapter(
                model_name="test",
                dimensions=128,
                base_url="http://custom:11434"
            )

            mock.assert_called_with(host="http://custom:11434")

    def test_gemini_adapter_requires_api_key(self):
        """GeminiAdapter should require API key."""
        with patch("benchmarks.embeddings.adapters.gemini.genai"):
            from benchmarks.embeddings.adapters.gemini import GeminiAdapter

            # Should work with explicit key
            adapter = GeminiAdapter(api_key="explicit-key")
            assert adapter is not None

    def test_gemini_adapter_uses_env_api_key(self):
        """GeminiAdapter should use GOOGLE_API_KEY from environment."""
        import os
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            with patch("benchmarks.embeddings.adapters.gemini.genai"):
                from benchmarks.embeddings.adapters.gemini import GeminiAdapter

                adapter = GeminiAdapter()  # No explicit key
                assert adapter is not None
