"""
Nomic Embed Code adapter.

Co-primary local model per implementation plan v7.
Runs via Ollama.
"""

from .ollama import OllamaAdapter


class NomicCodeAdapter(OllamaAdapter):
    """
    Nomic Embed Text adapter.

    Co-primary local embedding model per implementation plan v7.
    - 768 dimensions
    - 8192 context length
    - Good for both code and natural language

    Usage:
        adapter = NomicCodeAdapter()
        embedding = adapter.embed("def hello(): pass")
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Nomic adapter.

        Args:
            base_url: Ollama server URL.
        """
        super().__init__(
            model_name="nomic-embed-text",
            dimensions=768,
            base_url=base_url,
            max_sequence_length=8192,
            is_code_optimized=False,  # General purpose, not code-specific
        )
