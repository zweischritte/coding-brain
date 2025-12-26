"""
Lexical search backend implementations.

This module provides backend adapters for lexical/BM25 search
benchmarking. Currently supports:
- TantivyBackend: Mock Tantivy adapter
- OpenSearchBackend: Mock OpenSearch adapter

Use create_backend() factory function to instantiate backends
by name.
"""

from typing import List

from .base import BackendStats, Document, LexicalBackend, SearchResult
from .opensearch import OpenSearchBackend
from .tantivy import TantivyBackend

__all__ = [
    # Base classes and dataclasses
    "LexicalBackend",
    "Document",
    "SearchResult",
    "BackendStats",
    # Concrete implementations
    "TantivyBackend",
    "OpenSearchBackend",
    # Factory functions
    "create_backend",
    "list_available_backends",
]

# Registry of available backends
_BACKENDS = {
    "tantivy": TantivyBackend,
    "opensearch": OpenSearchBackend,
}


def create_backend(name: str) -> LexicalBackend:
    """
    Create a lexical backend by name.

    Args:
        name: Backend name (case-insensitive). One of: "tantivy", "opensearch".

    Returns:
        Instance of the requested backend.

    Raises:
        ValueError: If backend name is not recognized.

    Example:
        >>> backend = create_backend("tantivy")
        >>> backend.name
        'tantivy'
    """
    normalized_name = name.lower()
    if normalized_name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown backend: '{name}'. Available backends: {available}"
        )
    return _BACKENDS[normalized_name]()


def list_available_backends() -> List[str]:
    """
    List available backend names.

    Returns:
        List of backend names that can be passed to create_backend().

    Example:
        >>> list_available_backends()
        ['opensearch', 'tantivy']
    """
    return sorted(_BACKENDS.keys())
