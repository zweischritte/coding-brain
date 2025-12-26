"""Embedding pipeline with shadow model support (FR-004).

This module provides embedding flexibility and shadow pipeline:
- Content-addressed embedding storage
- Multi-model support with configuration
- Shadow pipeline for A/B testing
- Embedding comparison and quality metrics
- Automatic fallback handling

Key features:
- Content-addressed deduplication
- Parallel shadow model execution
- Sampling rate control for shadow traffic
- Cosine similarity comparison
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class EmbeddingError(Exception):
    """Base exception for embedding errors."""

    pass


class ProviderError(EmbeddingError):
    """Error from embedding provider."""

    pass


class StorageError(EmbeddingError):
    """Error from embedding storage."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model.

    Args:
        model_name: Model identifier
        dimension: Embedding dimension
        max_batch_size: Maximum texts per batch
        timeout_seconds: Request timeout
        normalize: Whether to normalize embeddings
        provider_options: Provider-specific options
    """

    model_name: str = "nomic-embed-text"
    dimension: int = 768
    max_batch_size: int = 32
    timeout_seconds: int = 30
    normalize: bool = True
    provider_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")


@dataclass(frozen=True)
class EmbeddingModel:
    """Descriptor for an embedding model.

    Args:
        model_id: Unique model identifier
        provider: Provider name (ollama, openai, etc.)
        dimension: Embedding dimension
        description: Model description
    """

    model_id: str
    provider: str
    dimension: int
    description: str = ""


# =============================================================================
# Content Hashing
# =============================================================================


@dataclass(frozen=True)
class ContentHash:
    """Content-addressed hash for deduplication.

    Args:
        hash: SHA-256 hash of content
        content_type: Type of content (text, code, etc.)
    """

    hash: str
    content_type: str = "text"

    @classmethod
    def from_text(
        cls,
        content: str,
        model_version: Optional[str] = None,
    ) -> ContentHash:
        """Generate hash from text content.

        Args:
            content: Text to hash
            model_version: Optional model version for cache busting

        Returns:
            ContentHash instance
        """
        data = content
        if model_version:
            data = f"{model_version}:{content}"

        hash_value = hashlib.sha256(data.encode()).hexdigest()
        return cls(hash=hash_value, content_type="text")

    def __str__(self) -> str:
        """String representation."""
        return f"ContentHash({self.hash[:8]}...)"


# =============================================================================
# Embedding Result
# =============================================================================


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Args:
        content_hash: Content hash for this embedding
        embedding: The embedding vector
        model_id: Model that generated this embedding
        dimension: Dimension of the embedding
        normalized: Whether embedding is normalized
        created_at: Creation timestamp
    """

    content_hash: ContentHash
    embedding: list[float]
    model_id: str
    dimension: int
    normalized: bool = True
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate dimension matches embedding."""
        if len(self.embedding) != self.dimension:
            raise ValueError(
                f"Embedding length {len(self.embedding)} does not match "
                f"dimension {self.dimension}"
            )

    @property
    def magnitude(self) -> float:
        """Calculate embedding magnitude."""
        return math.sqrt(sum(x * x for x in self.embedding))

    def normalize(self) -> EmbeddingResult:
        """Return normalized version of this embedding."""
        if self.normalized:
            return self

        mag = self.magnitude
        if mag == 0:
            return self

        normalized_emb = [x / mag for x in self.embedding]
        return EmbeddingResult(
            content_hash=self.content_hash,
            embedding=normalized_emb,
            model_id=self.model_id,
            dimension=self.dimension,
            normalized=True,
            created_at=self.created_at,
        )


# =============================================================================
# Embedding Batch
# =============================================================================


@dataclass
class EmbeddingBatch:
    """A batch of texts to embed.

    Args:
        texts: List of texts to embed
        content_hashes: Pre-computed content hashes
    """

    texts: list[str] = field(default_factory=list)
    content_hashes: list[ContentHash] = field(default_factory=list)
    results: list[EmbeddingResult] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.texts)

    @classmethod
    def from_texts(cls, texts: list[str]) -> EmbeddingBatch:
        """Create batch from list of texts.

        Args:
            texts: Texts to embed

        Returns:
            EmbeddingBatch instance
        """
        content_hashes = [ContentHash.from_text(text) for text in texts]
        return cls(texts=texts, content_hashes=content_hashes)

    def __iter__(self) -> Iterator[tuple[str, ContentHash]]:
        """Iterate over (text, hash) pairs."""
        return iter(zip(self.texts, self.content_hashes))

    def with_embeddings(
        self,
        embeddings: list[list[float]],
        model_id: str,
        dimension: int,
    ) -> EmbeddingBatch:
        """Create completed batch with embeddings.

        Args:
            embeddings: List of embedding vectors
            model_id: Model ID that generated embeddings
            dimension: Embedding dimension

        Returns:
            New EmbeddingBatch with results
        """
        results = []
        for content_hash, embedding in zip(self.content_hashes, embeddings):
            results.append(
                EmbeddingResult(
                    content_hash=content_hash,
                    embedding=embedding,
                    model_id=model_id,
                    dimension=dimension,
                )
            )

        return EmbeddingBatch(
            texts=self.texts,
            content_hashes=self.content_hashes,
            results=results,
        )


# =============================================================================
# Embedding Storage
# =============================================================================


class EmbeddingStore(ABC):
    """Abstract interface for embedding storage."""

    @abstractmethod
    def get(self, content_hash: ContentHash, model_id: str) -> Optional[EmbeddingResult]:
        """Get embedding by content hash and model."""
        pass

    @abstractmethod
    def put(self, result: EmbeddingResult) -> None:
        """Store an embedding result."""
        pass

    @abstractmethod
    def delete(self, content_hash: ContentHash, model_id: str) -> bool:
        """Delete an embedding."""
        pass


@dataclass
class StoreStats:
    """Statistics for embedding store."""

    entry_count: int
    estimated_memory_bytes: int


class InMemoryEmbeddingStore(EmbeddingStore):
    """In-memory embedding storage.

    Thread-safe storage with content-addressed indexing.
    """

    def __init__(self):
        """Initialize store."""
        # Key: (content_hash, model_id) -> EmbeddingResult
        self._store: dict[tuple[str, str], EmbeddingResult] = {}
        self._lock = threading.RLock()

    def get(self, content_hash: ContentHash, model_id: str) -> Optional[EmbeddingResult]:
        """Get embedding by content hash and model."""
        with self._lock:
            key = (content_hash.hash, model_id)
            return self._store.get(key)

    def put(self, result: EmbeddingResult) -> None:
        """Store an embedding result."""
        with self._lock:
            key = (result.content_hash.hash, result.model_id)
            self._store[key] = result

    def delete(self, content_hash: ContentHash, model_id: str) -> bool:
        """Delete an embedding."""
        with self._lock:
            key = (content_hash.hash, model_id)
            if key in self._store:
                del self._store[key]
                return True
            return False

    def bulk_get(
        self,
        content_hashes: list[ContentHash],
        model_id: str,
    ) -> list[Optional[EmbeddingResult]]:
        """Bulk get embeddings.

        Args:
            content_hashes: List of content hashes
            model_id: Model ID to look up

        Returns:
            List of results (None for misses)
        """
        with self._lock:
            return [self.get(ch, model_id) for ch in content_hashes]

    def size(self, model_id: Optional[str] = None) -> int:
        """Get number of stored embeddings."""
        with self._lock:
            if model_id:
                return sum(1 for k in self._store if k[1] == model_id)
            return len(self._store)


class ContentAddressedEmbeddingStore(InMemoryEmbeddingStore):
    """Content-addressed embedding storage with deduplication.

    Extends InMemoryEmbeddingStore with stats and deduplication tracking.
    """

    def stats(self, model_id: str) -> StoreStats:
        """Get storage statistics.

        Args:
            model_id: Model to get stats for

        Returns:
            StoreStats instance
        """
        with self._lock:
            entries = [v for k, v in self._store.items() if k[1] == model_id]
            entry_count = len(entries)

            # Estimate memory: embedding floats + overhead
            estimated_bytes = sum(
                len(e.embedding) * 8 + 200  # 8 bytes per float + overhead
                for e in entries
            )

            return StoreStats(
                entry_count=entry_count,
                estimated_memory_bytes=estimated_bytes,
            )


# =============================================================================
# Embedding Provider
# =============================================================================


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Get model identifier."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass


class OllamaProvider(EmbeddingProvider):
    """Ollama embedding provider.

    Uses local Ollama server for embeddings.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize provider.

        Args:
            config: Embedding configuration
        """
        self._config = config
        self._model_id = config.model_name
        self._dimension = config.dimension

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    @property
    def timeout(self) -> int:
        """Get timeout in seconds."""
        return self._config.timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via Ollama.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._call_ollama(texts)

    def _call_ollama(self, texts: list[str]) -> list[list[float]]:
        """Make API call to Ollama.

        This is a mock implementation - real impl would use httpx/requests.
        """
        # In production, this would call Ollama's /api/embeddings endpoint
        # For now, return mock embeddings
        return [[0.0] * self._dimension for _ in texts]


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider.

    Uses OpenAI API for embeddings.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize provider.

        Args:
            config: Embedding configuration
        """
        self._config = config
        self._model_id = config.model_name
        self._dimension = config.dimension
        self._api_key = config.provider_options.get("api_key", "")

    @property
    def model_id(self) -> str:
        """Get model identifier."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self._call_openai(texts)

    def _call_openai(self, texts: list[str]) -> list[list[float]]:
        """Make API call to OpenAI.

        This is a mock implementation - real impl would use openai SDK.
        """
        # In production, this would call OpenAI's embeddings endpoint
        return [[0.0] * self._dimension for _ in texts]


# =============================================================================
# Similarity Metrics
# =============================================================================


class SimilarityMetric:
    """Similarity calculations for embeddings."""

    @staticmethod
    def cosine(emb1: list[float], emb2: list[float]) -> float:
        """Calculate cosine similarity.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity (-1 to 1)
        """
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must have same dimension")

        dot = sum(a * b for a, b in zip(emb1, emb2))
        mag1 = math.sqrt(sum(x * x for x in emb1))
        mag2 = math.sqrt(sum(x * x for x in emb2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    @staticmethod
    def euclidean_distance(emb1: list[float], emb2: list[float]) -> float:
        """Calculate Euclidean distance.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Euclidean distance
        """
        if len(emb1) != len(emb2):
            raise ValueError("Embeddings must have same dimension")

        return math.sqrt(sum((a - b) ** 2 for a, b in zip(emb1, emb2)))

    @staticmethod
    def dot_product(emb1: list[float], emb2: list[float]) -> float:
        """Calculate dot product.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Dot product
        """
        return sum(a * b for a, b in zip(emb1, emb2))


# =============================================================================
# Embedding Pipeline
# =============================================================================


class EmbeddingPipeline:
    """Main embedding pipeline with caching.

    Provides:
    - Content-addressed caching
    - Batch embedding support
    - Automatic normalization
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        store: EmbeddingStore,
        normalize: bool = True,
    ):
        """Initialize pipeline.

        Args:
            provider: Embedding provider
            store: Embedding storage
            normalize: Whether to normalize embeddings
        """
        self.provider = provider
        self.store = store
        self.normalize = normalize
        self._metrics = EmbeddingMetrics()

    def embed(self, text: str) -> EmbeddingResult:
        """Embed single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult
        """
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed batch of texts.

        Args:
            texts: Texts to embed

        Returns:
            List of EmbeddingResult
        """
        batch = EmbeddingBatch.from_texts(texts)
        results: list[Optional[EmbeddingResult]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, (text, content_hash) in enumerate(batch):
            try:
                cached = self.store.get(content_hash, self.provider.model_id)
                if cached:
                    results[i] = cached
                    self._metrics.record_embedding(cached=True, latency_ms=0.0)
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
            except StorageError:
                # Continue with provider if cache fails
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts
        if uncached_texts:
            start = time.perf_counter()
            embeddings = self.provider.embed(uncached_texts)
            latency_ms = (time.perf_counter() - start) * 1000

            for idx, embedding in zip(uncached_indices, embeddings):
                content_hash = batch.content_hashes[idx]
                result = EmbeddingResult(
                    content_hash=content_hash,
                    embedding=embedding,
                    model_id=self.provider.model_id,
                    dimension=self.provider.dimension,
                    normalized=self.normalize,
                )

                if self.normalize:
                    result = result.normalize()

                results[idx] = result
                self._metrics.record_embedding(
                    cached=False,
                    latency_ms=latency_ms / len(uncached_texts),
                )

                # Store in cache
                try:
                    self.store.put(result)
                except StorageError:
                    pass  # Continue even if caching fails

        # Filter out any None values (shouldn't happen)
        return [r for r in results if r is not None]


# =============================================================================
# Shadow Pipeline
# =============================================================================


@dataclass
class ShadowComparison:
    """Comparison between primary and shadow embeddings.

    Args:
        cosine_similarity: Cosine similarity between embeddings
        euclidean_distance: Euclidean distance
        primary_latency_ms: Primary model latency
        shadow_latency_ms: Shadow model latency
    """

    cosine_similarity: float
    euclidean_distance: float
    primary_latency_ms: float
    shadow_latency_ms: float

    @classmethod
    def from_embeddings(
        cls,
        primary_emb: list[float],
        shadow_emb: list[float],
        primary_latency_ms: float,
        shadow_latency_ms: float,
    ) -> ShadowComparison:
        """Create comparison from embeddings.

        Args:
            primary_emb: Primary model embedding
            shadow_emb: Shadow model embedding
            primary_latency_ms: Primary latency
            shadow_latency_ms: Shadow latency

        Returns:
            ShadowComparison instance
        """
        cosine = SimilarityMetric.cosine(primary_emb, shadow_emb)
        euclidean = SimilarityMetric.euclidean_distance(primary_emb, shadow_emb)

        return cls(
            cosine_similarity=cosine,
            euclidean_distance=euclidean,
            primary_latency_ms=primary_latency_ms,
            shadow_latency_ms=shadow_latency_ms,
        )


@dataclass
class ShadowResult:
    """Result from shadow pipeline with comparison.

    Args:
        primary_result: Primary model result
        shadow_result: Shadow model result (may be None)
        comparison: Comparison metrics (if both available)
    """

    primary_result: EmbeddingResult
    shadow_result: Optional[EmbeddingResult] = None
    comparison: Optional[ShadowComparison] = None


class ShadowPipeline:
    """Shadow embedding pipeline for A/B testing.

    Runs primary model synchronously and optionally runs
    shadow model for comparison.
    """

    def __init__(
        self,
        primary_provider: EmbeddingProvider,
        shadow_provider: EmbeddingProvider,
        store: Optional[EmbeddingStore] = None,
        compare_results: bool = False,
        async_shadow: bool = True,
        sampling_rate: float = 1.0,
    ):
        """Initialize shadow pipeline.

        Args:
            primary_provider: Primary embedding provider
            shadow_provider: Shadow embedding provider
            store: Optional embedding storage
            compare_results: Whether to compare embeddings
            async_shadow: Whether to run shadow asynchronously
            sampling_rate: Fraction of requests to run shadow (0-1)
        """
        self.primary_provider = primary_provider
        self.shadow_provider = shadow_provider
        self.store = store or InMemoryEmbeddingStore()
        self.compare_results = compare_results
        self.async_shadow = async_shadow
        self.sampling_rate = sampling_rate
        self._executor: Optional[ThreadPoolExecutor] = None

        if async_shadow:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="shadow"
            )

    def embed(self, text: str) -> EmbeddingResult:
        """Embed text using primary model.

        Args:
            text: Text to embed

        Returns:
            Primary model result
        """
        result = self.embed_with_comparison(text)
        return result.primary_result

    def embed_with_comparison(self, text: str) -> ShadowResult:
        """Embed text and optionally compare with shadow.

        Args:
            text: Text to embed

        Returns:
            ShadowResult with comparison
        """
        content_hash = ContentHash.from_text(text)

        # Run primary model
        primary_start = time.perf_counter()
        primary_embeddings = self.primary_provider.embed([text])
        primary_latency = (time.perf_counter() - primary_start) * 1000

        primary_result = EmbeddingResult(
            content_hash=content_hash,
            embedding=primary_embeddings[0],
            model_id=self.primary_provider.model_id,
            dimension=self.primary_provider.dimension,
        )

        # Decide whether to run shadow
        should_shadow = random.random() < self.sampling_rate

        if not should_shadow:
            return ShadowResult(primary_result=primary_result)

        # Run shadow model
        shadow_result: Optional[EmbeddingResult] = None
        shadow_latency: float = 0.0
        comparison: Optional[ShadowComparison] = None

        if self.async_shadow and self._executor:
            # Run async and don't wait
            future = self._executor.submit(self._run_shadow, text, content_hash)
            # For comparison, we need to wait
            if self.compare_results:
                try:
                    shadow_result, shadow_latency = future.result(timeout=10.0)
                except Exception as e:
                    logger.debug(f"Shadow embedding failed: {e}")
        else:
            try:
                shadow_result, shadow_latency = self._run_shadow(text, content_hash)
            except Exception as e:
                logger.debug(f"Shadow embedding failed: {e}")

        # Compare if both available
        if shadow_result and self.compare_results:
            comparison = ShadowComparison.from_embeddings(
                primary_result.embedding,
                shadow_result.embedding,
                primary_latency,
                shadow_latency,
            )

        return ShadowResult(
            primary_result=primary_result,
            shadow_result=shadow_result,
            comparison=comparison,
        )

    def _run_shadow(
        self, text: str, content_hash: ContentHash
    ) -> tuple[EmbeddingResult, float]:
        """Run shadow model.

        Args:
            text: Text to embed
            content_hash: Content hash

        Returns:
            Tuple of (result, latency_ms)
        """
        start = time.perf_counter()
        embeddings = self.shadow_provider.embed([text])
        latency = (time.perf_counter() - start) * 1000

        result = EmbeddingResult(
            content_hash=content_hash,
            embedding=embeddings[0],
            model_id=self.shadow_provider.model_id,
            dimension=self.shadow_provider.dimension,
        )

        return result, latency

    def __del__(self):
        """Cleanup executor."""
        if self._executor:
            self._executor.shutdown(wait=False)


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations.

    Args:
        total_embeddings: Total embedding operations
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        total_latency_ms: Total latency in milliseconds
    """

    total_embeddings: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0

    def record_embedding(self, cached: bool, latency_ms: float) -> None:
        """Record an embedding operation.

        Args:
            cached: Whether result was cached
            latency_ms: Operation latency
        """
        self.total_embeddings += 1
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.total_latency_ms += latency_ms

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_embeddings == 0:
            return 0.0
        return self.cache_hits / self.total_embeddings

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency for uncached requests."""
        if self.cache_misses == 0:
            return 0.0
        return self.total_latency_ms / self.cache_misses


# =============================================================================
# Factory Functions
# =============================================================================


def create_embedding_pipeline(
    model_name: str = "nomic-embed-text",
    dimension: int = 768,
    provider: str = "ollama",
    normalize: bool = True,
    **provider_options: Any,
) -> EmbeddingPipeline:
    """Create an embedding pipeline.

    Args:
        model_name: Model name
        dimension: Embedding dimension
        provider: Provider type (ollama, openai)
        normalize: Whether to normalize embeddings
        **provider_options: Provider-specific options

    Returns:
        Configured EmbeddingPipeline
    """
    config = EmbeddingConfig(
        model_name=model_name,
        dimension=dimension,
        provider_options=provider_options,
    )

    if provider == "ollama":
        embedding_provider = OllamaProvider(config)
    elif provider == "openai":
        embedding_provider = OpenAIProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    store = ContentAddressedEmbeddingStore()

    return EmbeddingPipeline(
        provider=embedding_provider,
        store=store,
        normalize=normalize,
    )


def create_shadow_pipeline(
    primary_model: str,
    shadow_model: str,
    primary_dimension: int = 768,
    shadow_dimension: int = 768,
    primary_provider: str = "ollama",
    shadow_provider: str = "ollama",
    sampling_rate: float = 1.0,
    compare_results: bool = True,
    **options: Any,
) -> ShadowPipeline:
    """Create a shadow embedding pipeline.

    Args:
        primary_model: Primary model name
        shadow_model: Shadow model name
        primary_dimension: Primary embedding dimension
        shadow_dimension: Shadow embedding dimension
        primary_provider: Primary provider type
        shadow_provider: Shadow provider type
        sampling_rate: Fraction of requests to run shadow
        compare_results: Whether to compare embeddings
        **options: Additional options

    Returns:
        Configured ShadowPipeline
    """
    primary_config = EmbeddingConfig(
        model_name=primary_model,
        dimension=primary_dimension,
    )

    shadow_config = EmbeddingConfig(
        model_name=shadow_model,
        dimension=shadow_dimension,
    )

    if primary_provider == "ollama":
        primary = OllamaProvider(primary_config)
    else:
        primary = OpenAIProvider(primary_config)

    if shadow_provider == "ollama":
        shadow = OllamaProvider(shadow_config)
    else:
        shadow = OpenAIProvider(shadow_config)

    return ShadowPipeline(
        primary_provider=primary,
        shadow_provider=shadow,
        sampling_rate=sampling_rate,
        compare_results=compare_results,
    )
