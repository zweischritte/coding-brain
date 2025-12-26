"""
Gemini embedding adapter.

Cloud fallback per implementation plan v7.
Uses gemini-embedding-001 with Matryoshka dimension support.
"""

import os
from typing import List, Optional

from .base import EmbedderAdapter, EmbedderInfo

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class GeminiAdapter(EmbedderAdapter):
    """
    Gemini embedding adapter.

    Cloud fallback model per implementation plan v7.
    - Default 768 dimensions (Matryoshka: 3072, 1536, 768)
    - Supports dimension reduction via output_dimensionality

    Usage:
        adapter = GeminiAdapter(api_key="your-key")
        embedding = adapter.embed("def hello(): pass")

    Or with reduced dimensions:
        adapter = GeminiAdapter(api_key="your-key", dimensions=768)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        dimensions: int = 768,
        model_name: str = "models/gemini-embedding-001",
    ):
        """
        Initialize Gemini adapter.

        Args:
            api_key: Google API key. Falls back to GOOGLE_API_KEY env var.
            dimensions: Output embedding dimensions. Supports 768, 1536, 3072.
            model_name: Gemini embedding model name.
        """
        if genai is None:
            raise ImportError(
                "google-genai package is required. Install with: pip install google-genai"
            )

        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._dimensions = dimensions
        self._model_name = model_name

        self._client = genai.Client(api_key=self._api_key)

    @property
    def info(self) -> EmbedderInfo:
        """Return model metadata."""
        return EmbedderInfo(
            model_name=self._model_name,
            provider="gemini",
            dimensions=self._dimensions,
            max_sequence_length=2048,  # Gemini embedding limit
            is_code_optimized=False,
        )

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        # Clean text
        text = text.replace("\n", " ")

        # Create config for embedding parameters
        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)

        # Call the embed_content method
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=text,
            config=config
        )

        return list(response.embeddings[0].values)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Note: Currently iterates; could be optimized with batch API.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]
