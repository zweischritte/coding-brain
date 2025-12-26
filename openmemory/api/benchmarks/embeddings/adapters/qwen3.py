"""
Qwen3 embedding model adapters.

Primary: Qwen3-Embedding-8B (1024 dims)
Fallback: Qwen3-Embedding-0.6B (1024 dims)

Both run via Ollama.
"""

from .ollama import OllamaAdapter


class Qwen3_8BAdapter(OllamaAdapter):
    """
    Qwen3-Embedding-8B adapter.

    Primary local embedding model per implementation plan v7.
    - 1024 dimensions
    - 32K context length
    - Code optimized

    Usage:
        adapter = Qwen3_8BAdapter()
        embedding = adapter.embed("def hello(): pass")
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Qwen3-8B adapter.

        Args:
            base_url: Ollama server URL.
        """
        super().__init__(
            model_name="qwen3-embedding-8b",
            dimensions=1024,
            base_url=base_url,
            max_sequence_length=32768,
            is_code_optimized=True,
        )


class Qwen3_06BAdapter(OllamaAdapter):
    """
    Qwen3-Embedding-0.6B adapter.

    Local fallback model per implementation plan v7.
    - 1024 dimensions
    - 32K context length
    - Lighter weight for faster inference

    Usage:
        adapter = Qwen3_06BAdapter()
        embedding = adapter.embed("def hello(): pass")
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Qwen3-0.6B adapter.

        Args:
            base_url: Ollama server URL.
        """
        super().__init__(
            model_name="qwen3-embedding-0.6b",
            dimensions=1024,
            base_url=base_url,
            max_sequence_length=32768,
            is_code_optimized=True,
        )
