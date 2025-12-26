"""
Base class for lexical search backend adapters.

All lexical backends must inherit from LexicalBackend and implement
the required abstract methods.

This provides a common interface for comparing Tantivy vs OpenSearch
for lexical/BM25 search in the benchmark framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class Document:
    """
    A document to be indexed in the lexical backend.

    Attributes:
        id: Unique document identifier.
        content: Text content to index and search.
        metadata: Optional metadata dictionary.
    """
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """
    A search result from a lexical backend.

    Attributes:
        doc_id: ID of the matched document.
        score: Relevance score (e.g., BM25 score).
        content: Optional document content.
        highlights: Optional list of highlighted snippets.
    """
    doc_id: str
    score: float
    content: Optional[str] = None
    highlights: Optional[List[str]] = None


@dataclass
class BackendStats:
    """
    Statistics about a lexical backend index.

    Attributes:
        index_size_bytes: Size of the index on disk in bytes.
        document_count: Number of documents in the index.
        backend_name: Name of the backend (e.g., "tantivy", "opensearch").
        additional_info: Optional additional statistics.
    """
    index_size_bytes: int
    document_count: int
    backend_name: str
    additional_info: Optional[Dict[str, Any]] = None


class LexicalBackend(ABC):
    """
    Abstract base class for lexical search backend adapters.

    All lexical backend implementations must inherit from this class
    and implement:
    - name (property): Return backend identifier
    - index_documents: Index a list of documents
    - search: Search for documents matching a query
    - get_stats: Return index statistics
    - clear: Clear all documents from the index

    Example:
        class MyBackend(LexicalBackend):
            @property
            def name(self) -> str:
                return "my-backend"

            def index_documents(self, documents: List[Document]) -> int:
                # Index documents and return count
                return len(documents)

            def search(self, query: str, limit: int = 10) -> List[SearchResult]:
                # Search and return results
                return []

            def get_stats(self) -> BackendStats:
                return BackendStats(0, 0, "my-backend")

            def clear(self) -> None:
                # Clear all indexed documents
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return backend identifier.

        Returns:
            String name of this backend (e.g., "tantivy", "opensearch").
        """
        pass

    @abstractmethod
    def index_documents(self, documents: List[Document]) -> int:
        """
        Index a list of documents.

        Args:
            documents: List of Document objects to index.

        Returns:
            Number of documents successfully indexed.
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for documents matching a query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance (highest first).
        """
        pass

    @abstractmethod
    def get_stats(self) -> BackendStats:
        """
        Return index statistics.

        Returns:
            BackendStats with index size, document count, etc.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all documents from the index.

        After calling this method, get_stats().document_count should be 0.
        """
        pass
