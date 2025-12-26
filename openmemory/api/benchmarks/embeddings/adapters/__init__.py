"""Embedder adapter implementations."""

from typing import Optional

from .base import EmbedderAdapter, EmbedderInfo

# Lazy imports to avoid requiring all dependencies
_ADAPTERS = {
    "qwen3-8b": ("qwen3", "Qwen3_8BAdapter"),
    "qwen3-0.6b": ("qwen3", "Qwen3_06BAdapter"),
    "nomic": ("nomic", "NomicCodeAdapter"),
    "gemini": ("gemini", "GeminiAdapter"),
}


def create_adapter(name: str, **kwargs) -> EmbedderAdapter:
    """
    Factory function to create embedder adapters by name.

    Args:
        name: Adapter name. One of: qwen3-8b, qwen3-0.6b, nomic, gemini
        **kwargs: Additional arguments passed to adapter constructor.

    Returns:
        EmbedderAdapter instance.

    Raises:
        ValueError: If adapter name is unknown.

    Example:
        adapter = create_adapter("qwen3-8b")
        adapter = create_adapter("gemini", api_key="your-key")
    """
    if name not in _ADAPTERS:
        raise ValueError(
            f"Unknown adapter: {name}. "
            f"Available: {', '.join(_ADAPTERS.keys())}"
        )

    module_name, class_name = _ADAPTERS[name]

    # Dynamic import
    import importlib
    module = importlib.import_module(f".{module_name}", package=__name__)
    adapter_class = getattr(module, class_name)

    return adapter_class(**kwargs)


__all__ = [
    "EmbedderAdapter",
    "EmbedderInfo",
    "create_adapter",
]
