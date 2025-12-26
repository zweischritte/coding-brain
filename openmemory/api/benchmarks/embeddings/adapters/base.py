"""
Base class for embedding model adapters.

All embedder adapters must inherit from EmbedderAdapter and implement
the required abstract methods.

This follows the existing mem0 EmbeddingBase pattern but adds:
- Batch embedding support
- Model metadata exposure via EmbedderInfo
- Warmup method for accurate latency measurement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class EmbedderInfo:
    """
    Metadata about an embedding model.

    Attributes:
        model_name: Name/identifier of the embedding model.
        provider: Provider type ("local", "gemini", "ollama", etc.).
        dimensions: Number of dimensions in the embedding vector.
        max_sequence_length: Maximum input sequence length in tokens.
        is_code_optimized: Whether the model is optimized for code embeddings.
    """
    model_name: str
    provider: str
    dimensions: int
    max_sequence_length: int
    is_code_optimized: bool = False


class EmbedderAdapter(ABC):
    """
    Abstract base class for embedding model adapters.

    All embedder implementations must inherit from this class and implement:
    - info (property): Return model metadata
    - embed: Generate embedding for single text
    - embed_batch: Generate embeddings for multiple texts

    Example:
        class MyEmbedder(EmbedderAdapter):
            @property
            def info(self) -> EmbedderInfo:
                return EmbedderInfo("my-model", "local", 768, 8192)

            def embed(self, text: str) -> List[float]:
                # Generate embedding
                return self._model.encode(text).tolist()

            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [self.embed(t) for t in texts]
    """

    @property
    @abstractmethod
    def info(self) -> EmbedderInfo:
        """
        Return model metadata.

        Returns:
            EmbedderInfo with model name, provider, dimensions, etc.
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed. Can be empty string.

        Returns:
            Embedding vector as list of floats with length == info.dimensions.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        This method should be more efficient than calling embed() in a loop
        when the underlying model supports batch processing.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors, one per input text.
            Length of return list equals length of input list.
        """
        pass

    def warmup(self) -> None:
        """
        Pre-warm the model for accurate latency measurement.

        Called before benchmarking to ensure the model is loaded and
        any one-time initialization is complete.

        Default implementation embeds a short text.
        Override if your model needs different warmup behavior.
        """
        self.embed("warmup text for model initialization")
