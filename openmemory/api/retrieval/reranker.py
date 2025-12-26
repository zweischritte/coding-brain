"""Reranker integration for tri-hybrid retrieval.

This module provides:
- Abstract reranker adapter interface
- Cross-encoder reranker implementation
- Cohere reranker implementation
- No-op passthrough reranker
- Integration with tri-hybrid results
- Score normalization after reranking
- Graceful fallback when reranker unavailable
- Sub-50ms p95 latency target
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class RerankerError(Exception):
    """Base exception for reranker errors."""

    pass


class RerankerTimeoutError(RerankerError):
    """Exception raised when reranking times out."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RerankerConfig:
    """Configuration for reranker.

    Args:
        top_k: Number of top results to rerank (default 20)
        enabled: Whether reranking is enabled (default True)
        timeout_ms: Timeout for reranking in milliseconds (default 50)
        batch_size: Batch size for processing (default 32)
        min_score: Minimum score threshold (default 0.0)
        normalize_scores: Whether to normalize scores to 0-1 (default True)
        max_length: Maximum content length for reranking (default 512)
    """

    top_k: int = 20
    enabled: bool = True
    timeout_ms: int = 50
    batch_size: int = 32
    min_score: float = 0.0
    normalize_scores: bool = True
    max_length: int = 512


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class RerankedResult:
    """A reranked result with score information."""

    id: str
    score: float
    original_score: Optional[float] = None
    original_rank: int = 0
    new_rank: int = 0
    document: dict[str, Any] = field(default_factory=dict)

    @property
    def rank_change(self) -> int:
        """Calculate rank change (positive = moved up)."""
        return self.original_rank - self.new_rank


@dataclass
class RerankedTriHybridTiming:
    """Timing breakdown for reranked tri-hybrid retrieval."""

    lexical_ms: float = 0.0
    vector_ms: float = 0.0
    graph_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RerankedTriHybridResult:
    """Result from reranked tri-hybrid retrieval."""

    hits: list[Any]  # List of TriHybridHit
    total: int
    timing: RerankedTriHybridTiming
    graph_available: bool = True
    graph_error: Optional[str] = None
    lexical_error: Optional[str] = None
    vector_error: Optional[str] = None
    reranker_available: bool = True
    reranker_error: Optional[str] = None


# =============================================================================
# Abstract Adapter
# =============================================================================


class RerankerAdapter(ABC):
    """Abstract base class for reranker implementations."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[RerankedResult]:
        """Rerank documents for a given query.

        Args:
            query: The search query
            documents: List of documents with id, content, and optional metadata

        Returns:
            List of RerankedResult sorted by score descending
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the reranker is available.

        Returns:
            True if reranker is ready to use
        """
        pass


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================


class CrossEncoderReranker(RerankerAdapter):
    """Cross-encoder based reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        config: Optional[RerankerConfig] = None,
        lazy_load: bool = False,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            config: Reranker configuration
            lazy_load: If True, don't load model immediately
        """
        self.model_name = model_name
        self.config = config or RerankerConfig()
        self._model: Optional[Any] = None

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, cross-encoder unavailable"
            )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[RerankedResult]:
        """Rerank documents using cross-encoder.

        Args:
            query: The search query
            documents: List of documents to rerank

        Returns:
            List of RerankedResult sorted by score descending

        Raises:
            RerankerError: If query is empty
            RerankerTimeoutError: If reranking times out
        """
        if not query:
            raise RerankerError("Empty query not allowed")

        if not documents:
            return []

        if not self.config.enabled:
            return self._passthrough(documents)

        if not self.is_available():
            return self._passthrough(documents)

        # Limit to top_k documents
        docs_to_rerank = documents[: self.config.top_k]

        # Extract content from documents
        pairs = []
        for doc in docs_to_rerank:
            content = doc.get("content") or doc.get("text") or ""
            # Truncate content if needed
            if len(content) > self.config.max_length:
                content = content[: self.config.max_length]
            pairs.append([query, content])

        # Run inference with timeout
        try:
            scores = self._predict_with_timeout(pairs)
        except FuturesTimeoutError:
            raise RerankerTimeoutError(
                f"Reranking timed out after {self.config.timeout_ms}ms"
            )

        # Normalize scores if configured
        if self.config.normalize_scores:
            scores = self._normalize_scores(scores)

        # Build results
        results = []
        for i, (doc, score) in enumerate(zip(docs_to_rerank, scores)):
            results.append(
                RerankedResult(
                    id=doc.get("id", f"doc_{i}"),
                    score=score,
                    original_score=doc.get("score"),
                    original_rank=i + 1,
                    new_rank=0,  # Will be set after sorting
                    document=doc,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Set new ranks
        for i, r in enumerate(results):
            r.new_rank = i + 1

        return results

    def _predict_with_timeout(self, pairs: list[list[str]]) -> list[float]:
        """Run model prediction with timeout.

        Args:
            pairs: List of [query, document] pairs

        Returns:
            List of scores

        Raises:
            FuturesTimeoutError: If prediction times out
        """
        # Process in batches
        all_scores = []
        batch_size = self.config.batch_size

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._model.predict, batch)
                timeout_s = self.config.timeout_ms / 1000
                batch_scores = future.result(timeout=timeout_s)
                all_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores))

        return all_scores

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalize scores to 0-1 range using min-max."""
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _passthrough(self, documents: list[dict[str, Any]]) -> list[RerankedResult]:
        """Return documents in original order as passthrough."""
        results = []
        for i, doc in enumerate(documents):
            score = doc.get("score", 1.0 - i * 0.01)
            results.append(
                RerankedResult(
                    id=doc.get("id", f"doc_{i}"),
                    score=score,
                    original_score=score,
                    original_rank=i + 1,
                    new_rank=i + 1,
                    document=doc,
                )
            )
        return results

    def is_available(self) -> bool:
        """Check if model is loaded and available."""
        return self._model is not None


# =============================================================================
# Cohere Reranker
# =============================================================================


class CohereReranker(RerankerAdapter):
    """Cohere API based reranker."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "rerank-english-v3.0",
        config: Optional[RerankerConfig] = None,
    ):
        """Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model_name: Name of the rerank model
            config: Reranker configuration
        """
        self.api_key = api_key
        self.model_name = model_name
        self.config = config or RerankerConfig()
        self._client: Optional[Any] = None

        self._init_client()

    def _init_client(self) -> None:
        """Initialize Cohere client."""
        try:
            import cohere

            self._client = cohere.Client(self.api_key)
        except ImportError:
            logger.warning("cohere not installed, Cohere reranker unavailable")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[RerankedResult]:
        """Rerank documents using Cohere API.

        Args:
            query: The search query
            documents: List of documents to rerank

        Returns:
            List of RerankedResult sorted by score descending

        Raises:
            RerankerError: If API call fails
        """
        if not query:
            raise RerankerError("Empty query not allowed")

        if not documents:
            return []

        if not self.is_available():
            return self._passthrough(documents)

        # Limit to top_k documents
        docs_to_rerank = documents[: self.config.top_k]

        # Extract content
        contents = []
        for doc in docs_to_rerank:
            content = doc.get("content") or doc.get("text") or ""
            if len(content) > self.config.max_length:
                content = content[: self.config.max_length]
            contents.append(content)

        # Call Cohere API
        try:
            response = self._client.rerank(
                model=self.model_name,
                query=query,
                documents=contents,
                top_n=len(contents),
            )
        except Exception as e:
            raise RerankerError(f"Cohere API error: {e}")

        # Build results from response
        results = []
        for rerank_result in response.results:
            idx = rerank_result.index
            doc = docs_to_rerank[idx]
            results.append(
                RerankedResult(
                    id=doc.get("id", f"doc_{idx}"),
                    score=rerank_result.relevance_score,
                    original_score=doc.get("score"),
                    original_rank=idx + 1,
                    new_rank=0,
                    document=doc,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # Set new ranks
        for i, r in enumerate(results):
            r.new_rank = i + 1

        return results

    def _passthrough(self, documents: list[dict[str, Any]]) -> list[RerankedResult]:
        """Return documents in original order as passthrough."""
        results = []
        for i, doc in enumerate(documents):
            score = doc.get("score", 1.0 - i * 0.01)
            results.append(
                RerankedResult(
                    id=doc.get("id", f"doc_{i}"),
                    score=score,
                    original_score=score,
                    original_rank=i + 1,
                    new_rank=i + 1,
                    document=doc,
                )
            )
        return results

    def is_available(self) -> bool:
        """Check if Cohere client is available."""
        return self._client is not None


# =============================================================================
# No-Op Reranker
# =============================================================================


class NoOpReranker(RerankerAdapter):
    """No-op reranker that returns documents in original order."""

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[RerankedResult]:
        """Return documents in original order.

        Args:
            query: The search query (ignored)
            documents: List of documents

        Returns:
            List of RerankedResult in original order
        """
        results = []
        for i, doc in enumerate(documents):
            score = doc.get("score", 1.0 - i * 0.01)
            results.append(
                RerankedResult(
                    id=doc.get("id", f"doc_{i}"),
                    score=score,
                    original_score=score,
                    original_rank=i + 1,
                    new_rank=i + 1,
                    document=doc,
                )
            )
        return results

    def is_available(self) -> bool:
        """No-op reranker is always available."""
        return True


# =============================================================================
# Tri-Hybrid Integration
# =============================================================================


def rerank_trihybrid_results(
    query: str,
    trihybrid_result: Any,  # TriHybridResult
    reranker: RerankerAdapter,
) -> RerankedTriHybridResult:
    """Rerank tri-hybrid retrieval results.

    Args:
        query: The search query
        trihybrid_result: Result from tri-hybrid retrieval
        reranker: Reranker to use

    Returns:
        RerankedTriHybridResult with reranked hits
    """
    start = time.perf_counter()

    # Check if reranker is available
    if not reranker.is_available():
        return _create_fallback_result(
            trihybrid_result,
            reranker_available=False,
            reranker_error="Reranker not available",
        )

    # Check if reranking is disabled
    if hasattr(reranker, "config") and not reranker.config.enabled:
        return _create_fallback_result(
            trihybrid_result,
            reranker_available=True,
            reranker_error=None,
        )

    # Convert hits to documents for reranker
    documents = []
    for i, hit in enumerate(trihybrid_result.hits):
        documents.append(
            {
                "id": hit.id,
                "content": hit.source.get("content", ""),
                "score": hit.score,
                "metadata": hit.source,
                "sources": hit.sources,
            }
        )

    # Rerank
    try:
        reranked = reranker.rerank(query=query, documents=documents)
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return _create_fallback_result(
            trihybrid_result,
            reranker_available=True,
            reranker_error=str(e),
        )

    rerank_ms = (time.perf_counter() - start) * 1000

    # Rebuild hits from reranked results
    # Import here to avoid circular dependency
    from openmemory.api.retrieval.trihybrid import TriHybridHit

    reranked_hits = []
    for result in reranked:
        hit = TriHybridHit(
            id=result.id,
            score=result.score,
            source=result.document.get("metadata", {}),
            sources=result.document.get("sources", {}),
        )
        reranked_hits.append(hit)

    # Build timing
    timing = RerankedTriHybridTiming(
        lexical_ms=trihybrid_result.timing.lexical_ms,
        vector_ms=trihybrid_result.timing.vector_ms,
        graph_ms=trihybrid_result.timing.graph_ms,
        fusion_ms=trihybrid_result.timing.fusion_ms,
        rerank_ms=rerank_ms,
        total_ms=trihybrid_result.timing.total_ms + rerank_ms,
    )

    return RerankedTriHybridResult(
        hits=reranked_hits,
        total=len(reranked_hits),
        timing=timing,
        graph_available=trihybrid_result.graph_available,
        graph_error=trihybrid_result.graph_error,
        lexical_error=trihybrid_result.lexical_error,
        vector_error=trihybrid_result.vector_error,
        reranker_available=True,
        reranker_error=None,
    )


def _create_fallback_result(
    trihybrid_result: Any,
    reranker_available: bool,
    reranker_error: Optional[str],
) -> RerankedTriHybridResult:
    """Create fallback result when reranking fails or is disabled."""
    timing = RerankedTriHybridTiming(
        lexical_ms=trihybrid_result.timing.lexical_ms,
        vector_ms=trihybrid_result.timing.vector_ms,
        graph_ms=trihybrid_result.timing.graph_ms,
        fusion_ms=trihybrid_result.timing.fusion_ms,
        rerank_ms=0.0,
        total_ms=trihybrid_result.timing.total_ms,
    )

    return RerankedTriHybridResult(
        hits=trihybrid_result.hits,
        total=trihybrid_result.total,
        timing=timing,
        graph_available=trihybrid_result.graph_available,
        graph_error=trihybrid_result.graph_error,
        lexical_error=trihybrid_result.lexical_error,
        vector_error=trihybrid_result.vector_error,
        reranker_available=reranker_available,
        reranker_error=reranker_error,
    )


# =============================================================================
# Factory Function
# =============================================================================


def create_reranker(
    backend: str = "noop",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    config: Optional[RerankerConfig] = None,
) -> RerankerAdapter:
    """Create a reranker instance.

    Args:
        backend: Reranker backend ('cross-encoder', 'cohere', 'noop')
        model_name: Model name for the reranker
        api_key: API key for cloud-based rerankers
        config: Reranker configuration

    Returns:
        Configured RerankerAdapter

    Raises:
        RerankerError: If backend is unknown
    """
    if backend == "cross-encoder":
        return CrossEncoderReranker(
            model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            config=config,
        )
    elif backend == "cohere":
        if not api_key:
            raise RerankerError("API key required for Cohere reranker")
        return CohereReranker(
            api_key=api_key,
            model_name=model_name or "rerank-english-v3.0",
            config=config,
        )
    elif backend == "noop":
        return NoOpReranker()
    else:
        raise RerankerError(f"Unknown backend: {backend}")
