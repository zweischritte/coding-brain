"""
Ollama-based embedder adapter.

Supports local embedding models running via Ollama:
- Qwen3-Embedding-8B
- Qwen3-Embedding-0.6B
- Nomic Embed Text
"""

from typing import List, Optional

from .base import EmbedderAdapter, EmbedderInfo

try:
    from ollama import Client
except ImportError:
    Client = None


class OllamaAdapter(EmbedderAdapter):
    """
    Embedder adapter for Ollama-hosted models.

    Requires Ollama to be running locally or at a specified base_url.

    Usage:
        adapter = OllamaAdapter(
            model_name="qwen3-embedding-8b",
            dimensions=1024,
            base_url="http://localhost:11434"
        )
        embedding = adapter.embed("hello world")
    """

    def __init__(
        self,
        model_name: str,
        dimensions: int,
        base_url: str = "http://localhost:11434",
        max_sequence_length: int = 8192,
        is_code_optimized: bool = False,
    ):
        """
        Initialize Ollama adapter.

        Args:
            model_name: Ollama model name (e.g., "qwen3-embedding-8b").
            dimensions: Embedding vector dimensions.
            base_url: Ollama server URL. Defaults to localhost.
            max_sequence_length: Maximum input sequence length.
            is_code_optimized: Whether model is optimized for code.
        """
        if Client is None:
            raise ImportError(
                "ollama package is required. Install with: pip install ollama"
            )

        self._model_name = model_name
        self._dimensions = dimensions
        self._max_sequence_length = max_sequence_length
        self._is_code_optimized = is_code_optimized

        self._client = Client(host=base_url)
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Pull model if not available locally."""
        try:
            local_models = self._client.list().get("models", [])
            model_names = [
                m.get("name", m.get("model", "")) for m in local_models
            ]
            if self._model_name not in model_names:
                # Try to pull the model
                self._client.pull(self._model_name)
        except Exception:
            # If we can't list/pull, assume model exists and will fail on embed
            pass

    @property
    def info(self) -> EmbedderInfo:
        """Return model metadata."""
        return EmbedderInfo(
            model_name=self._model_name,
            provider="ollama",
            dimensions=self._dimensions,
            max_sequence_length=self._max_sequence_length,
            is_code_optimized=self._is_code_optimized,
        )

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        response = self._client.embeddings(model=self._model_name, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Note: Ollama doesn't natively support batch embeddings,
        so this iterates over texts.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]
