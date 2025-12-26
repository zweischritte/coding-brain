"""
Tantivy mock backend adapter for benchmarking.

This is a mock implementation that simulates Tantivy's lexical search
behavior for benchmarking purposes. It uses simple in-memory search
with basic BM25-like scoring.

For production, this would be replaced with actual tantivy-py bindings.
"""

import re
from collections import Counter
from math import log
from typing import Dict, List, Set

from .base import BackendStats, Document, LexicalBackend, SearchResult


class TantivyBackend(LexicalBackend):
    """
    Mock Tantivy backend for lexical search benchmarking.

    This implementation provides basic BM25-like scoring for testing
    and benchmarking purposes. It stores documents in memory and
    performs simple text matching.

    For actual Tantivy integration, use tantivy-py bindings.
    """

    def __init__(self):
        """Initialize the mock Tantivy backend."""
        self._documents: Dict[str, Document] = {}
        self._inverted_index: Dict[str, Set[str]] = {}  # term -> doc_ids

    @property
    def name(self) -> str:
        """Return backend identifier."""
        return "tantivy"

    def index_documents(self, documents: List[Document]) -> int:
        """
        Index a list of documents.

        Args:
            documents: List of Document objects to index.

        Returns:
            Number of documents successfully indexed.
        """
        for doc in documents:
            self._documents[doc.id] = doc
            # Build inverted index
            terms = self._tokenize(doc.content)
            for term in terms:
                if term not in self._inverted_index:
                    self._inverted_index[term] = set()
                self._inverted_index[term].add(doc.id)
        return len(documents)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for documents matching a query.

        Uses simplified BM25-like scoring for ranking.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of SearchResult objects sorted by relevance (highest first).
        """
        if not query.strip():
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Find candidate documents
        candidate_doc_ids: Set[str] = set()
        for term in query_terms:
            if term in self._inverted_index:
                candidate_doc_ids.update(self._inverted_index[term])

        if not candidate_doc_ids:
            return []

        # Score documents using simplified BM25
        scored_results: List[SearchResult] = []
        for doc_id in candidate_doc_ids:
            doc = self._documents[doc_id]
            score = self._compute_bm25_score(doc.content, query_terms)
            scored_results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                content=doc.content
            ))

        # Sort by score (descending) and limit
        scored_results.sort(key=lambda r: r.score, reverse=True)
        return scored_results[:limit]

    def get_stats(self) -> BackendStats:
        """
        Return index statistics.

        Returns:
            BackendStats with index size, document count, etc.
        """
        # Estimate index size based on document content
        total_size = sum(
            len(doc.content.encode('utf-8'))
            for doc in self._documents.values()
        )
        # Add overhead for inverted index
        index_overhead = len(self._inverted_index) * 100

        return BackendStats(
            index_size_bytes=total_size + index_overhead,
            document_count=len(self._documents),
            backend_name=self.name,
            additional_info={
                "unique_terms": len(self._inverted_index),
                "implementation": "mock"
            }
        )

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents.clear()
        self._inverted_index.clear()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase terms.

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        # Simple whitespace and punctuation tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _compute_bm25_score(
        self,
        doc_content: str,
        query_terms: List[str],
        k1: float = 1.2,
        b: float = 0.75
    ) -> float:
        """
        Compute simplified BM25 score for a document.

        Args:
            doc_content: Document content to score.
            query_terms: Query terms to match.
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (length normalization).

        Returns:
            BM25-like score.
        """
        doc_terms = self._tokenize(doc_content)
        doc_len = len(doc_terms)
        term_counts = Counter(doc_terms)

        # Average document length
        if self._documents:
            avg_doc_len = sum(
                len(self._tokenize(d.content))
                for d in self._documents.values()
            ) / len(self._documents)
        else:
            avg_doc_len = doc_len

        score = 0.0
        n_docs = len(self._documents)

        for term in query_terms:
            if term not in term_counts:
                continue

            # Term frequency in document
            tf = term_counts[term]

            # Document frequency (docs containing term)
            df = len(self._inverted_index.get(term, set()))

            # IDF component
            if df > 0:
                idf = log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            else:
                idf = 0.0

            # BM25 score component
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += idf * (numerator / denominator)

        return score
