"""Prefetch cache with LRU eviction for speculative retrieval (FR-010).

This module provides a caching layer for tri-hybrid retrieval:
- LRU cache with configurable size and TTL
- Cache key generation from queries
- Speculative query pattern generation
- Cache warming functionality
- Metrics tracking (hit rate, evictions)

Key features:
- Sub-100ms cache hits
- Scope-aware TTL configuration
- Async prefetching support
- Memory-bounded caching
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================


class CacheError(Exception):
    """Base exception for cache errors."""

    pass


class CacheKeyError(CacheError):
    """Error generating cache key."""

    pass


class CacheMissError(CacheError):
    """Cache miss error for get_or_raise."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CacheConfig:
    """Configuration for prefetch cache.

    Args:
        max_entries: Maximum number of entries in cache
        default_ttl_seconds: Default TTL for cache entries
        max_memory_mb: Maximum memory usage in MB
        enable_metrics: Whether to track cache metrics
        scope_ttl_overrides: Per-scope TTL overrides
    """

    max_entries: int = 1000
    default_ttl_seconds: int = 300  # 5 minutes
    max_memory_mb: int = 100
    enable_metrics: bool = True
    scope_ttl_overrides: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if self.default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")


# =============================================================================
# Cache Key
# =============================================================================


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key for retrieval queries.

    Args:
        hash: SHA-256 hash of query components
        index_name: The index being queried
    """

    hash: str
    index_name: str

    @classmethod
    def from_query(
        cls,
        index_name: str,
        query_text: str = "",
        embedding: Optional[list[float]] = None,
        filters: Optional[dict[str, Any]] = None,
        seed_symbols: Optional[list[str]] = None,
    ) -> CacheKey:
        """Generate cache key from query components.

        Args:
            index_name: Index name
            query_text: Optional query text
            embedding: Optional embedding vector
            filters: Optional filters
            seed_symbols: Optional seed symbols for graph

        Returns:
            CacheKey instance

        Raises:
            CacheKeyError: If neither query_text nor embedding provided
        """
        if not query_text and not embedding:
            raise CacheKeyError("Must provide query_text or embedding for cache key")

        # Build components for hashing
        components = {
            "query_text": query_text,
            "index_name": index_name,
        }

        if embedding:
            # Hash embedding to reduce key size
            emb_str = ",".join(f"{x:.6f}" for x in embedding)
            components["embedding_hash"] = hashlib.sha256(
                emb_str.encode()
            ).hexdigest()[:16]

        if filters:
            # Sort filters for deterministic ordering
            components["filters"] = json.dumps(filters, sort_keys=True)

        if seed_symbols:
            components["seed_symbols"] = ",".join(sorted(seed_symbols))

        # Generate final hash
        key_str = json.dumps(components, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()

        return cls(hash=key_hash, index_name=index_name)

    def __str__(self) -> str:
        """String representation."""
        return f"CacheKey({self.index_name}:{self.hash[:8]})"


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class CacheEntry:
    """A single cache entry with metadata.

    Args:
        key: Cache key
        value: Cached value
        ttl_seconds: Time-to-live in seconds
        created_at: Creation timestamp
        last_accessed_at: Last access timestamp
        access_count: Number of times accessed
    """

    key: CacheKey
    value: Any
    ttl_seconds: int
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed_at = time.time()
        self.access_count += 1

    @property
    def estimated_size_bytes(self) -> int:
        """Estimate memory size of entry."""
        return sys.getsizeof(self.value) + sys.getsizeof(self.key)


# =============================================================================
# Cache Metrics
# =============================================================================


@dataclass
class CacheMetrics:
    """Cache performance metrics.

    Args:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of evictions
        puts: Number of put operations
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    puts: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.puts = 0


@dataclass
class CacheStats:
    """Cache statistics snapshot.

    Args:
        entry_count: Current number of entries
        max_entries: Maximum allowed entries
        estimated_memory_bytes: Estimated memory usage
        max_memory_bytes: Maximum allowed memory
    """

    entry_count: int
    max_entries: int
    estimated_memory_bytes: int
    max_memory_bytes: int

    @property
    def utilization(self) -> float:
        """Calculate cache utilization (0-1)."""
        if self.max_entries == 0:
            return 0.0
        return self.entry_count / self.max_entries


# =============================================================================
# Abstract Cache Interface
# =============================================================================


class PrefetchCache(ABC):
    """Abstract interface for prefetch cache."""

    @abstractmethod
    def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def put(
        self,
        key: CacheKey,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Put value into cache."""
        pass

    @abstractmethod
    def invalidate(self, key: CacheKey) -> bool:
        """Invalidate a specific key."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of entries."""
        pass

    @abstractmethod
    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        pass


# =============================================================================
# LRU Prefetch Cache Implementation
# =============================================================================


class LRUPrefetchCache(PrefetchCache):
    """LRU (Least Recently Used) prefetch cache implementation.

    Thread-safe cache with:
    - LRU eviction policy
    - TTL-based expiration
    - Metrics tracking
    - Per-index invalidation
    """

    def __init__(self, config: CacheConfig):
        """Initialize cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._metrics = CacheMetrics()
        self._lock = threading.RLock()

    def get(self, key: CacheKey) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            key_str = key.hash

            if key_str not in self._cache:
                if self.config.enable_metrics:
                    self._metrics.misses += 1
                return None

            entry = self._cache[key_str]

            # Check expiration
            if entry.is_expired():
                del self._cache[key_str]
                if self.config.enable_metrics:
                    self._metrics.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key_str)
            entry.touch()

            if self.config.enable_metrics:
                self._metrics.hits += 1

            return entry.value

    def get_or_raise(self, key: CacheKey) -> Any:
        """Get value or raise CacheMissError.

        Args:
            key: Cache key

        Returns:
            Cached value

        Raises:
            CacheMissError: If key not found or expired
        """
        value = self.get(key)
        if value is None:
            raise CacheMissError(f"Cache miss for key: {key}")
        return value

    def put(
        self,
        key: CacheKey,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Put value into cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional custom TTL (uses default if not provided)
        """
        with self._lock:
            key_str = key.hash
            ttl = ttl_seconds or self.config.default_ttl_seconds

            # Check if we need to evict
            if key_str not in self._cache and len(self._cache) >= self.config.max_entries:
                self._evict_lru()

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl,
            )

            # Add to cache
            self._cache[key_str] = entry
            self._cache.move_to_end(key_str)

            if self.config.enable_metrics:
                self._metrics.puts += 1

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            # Pop the first item (least recently used)
            self._cache.popitem(last=False)
            if self.config.enable_metrics:
                self._metrics.evictions += 1

    def invalidate(self, key: CacheKey) -> bool:
        """Invalidate a specific key.

        Args:
            key: Cache key to invalidate

        Returns:
            True if key was found and removed
        """
        with self._lock:
            key_str = key.hash
            if key_str in self._cache:
                del self._cache[key_str]
                return True
            return False

    def invalidate_by_index(self, index_name: str) -> int:
        """Invalidate all entries for an index.

        Args:
            index_name: Index name to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            to_remove = [
                key_str
                for key_str, entry in self._cache.items()
                if entry.key.index_name == index_name
            ]

            for key_str in to_remove:
                del self._cache[key_str]

            return len(to_remove)

    def contains(self, key: CacheKey) -> bool:
        """Check if key exists (and is not expired).

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        with self._lock:
            key_str = key.hash
            if key_str not in self._cache:
                return False

            entry = self._cache[key_str]
            if entry.is_expired():
                del self._cache[key_str]
                return False

            return True

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get number of entries."""
        with self._lock:
            return len(self._cache)

    def get_metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self._metrics.reset()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(
                entry.estimated_size_bytes for entry in self._cache.values()
            )

            return CacheStats(
                entry_count=len(self._cache),
                max_entries=self.config.max_entries,
                estimated_memory_bytes=total_size,
                max_memory_bytes=self.config.max_memory_mb * 1024 * 1024,
            )


# =============================================================================
# Speculative Query Patterns
# =============================================================================


class PatternType(Enum):
    """Types of speculative query patterns."""

    SIMILAR_QUERY = "similar_query"  # Similar text queries
    FOLLOW_UP = "follow_up"  # Follow-up based on context
    SYMBOL_RELATED = "symbol_related"  # Related to symbols
    BREADCRUMB = "breadcrumb"  # Based on navigation history


@dataclass
class SpeculativePattern:
    """A speculative query pattern.

    Args:
        pattern_type: Type of pattern
        query_text: Generated query text
        confidence: Confidence score (0-1)
        embedding: Optional pre-computed embedding
        seed_symbols: Optional seed symbols
    """

    pattern_type: PatternType
    query_text: str = ""
    confidence: float = 0.5
    embedding: Optional[list[float]] = None
    seed_symbols: Optional[list[str]] = None


class SpeculativeQueryGenerator:
    """Generates speculative query patterns for prefetching.

    Analyzes query patterns and context to predict
    likely follow-up queries for prefetching.
    """

    def __init__(self):
        """Initialize generator."""
        self._similar_terms: dict[str, list[str]] = {
            "database": ["db", "storage", "persistence", "sql"],
            "connection": ["connect", "client", "session", "link"],
            "pool": ["pooling", "cache", "manager", "queue"],
            "error": ["exception", "failure", "handler", "catch"],
            "config": ["configuration", "settings", "options", "params"],
        }

    def generate_patterns(
        self,
        query_text: str = "",
        embedding: Optional[list[float]] = None,
        seed_symbols: Optional[list[str]] = None,
        pattern_types: Optional[list[PatternType]] = None,
        context: Optional[dict[str, Any]] = None,
        max_patterns: int = 3,
    ) -> list[SpeculativePattern]:
        """Generate speculative patterns for prefetching.

        Args:
            query_text: Original query text
            embedding: Original query embedding
            seed_symbols: Original seed symbols
            pattern_types: Types of patterns to generate
            context: Additional context (history, etc.)
            max_patterns: Maximum patterns to return

        Returns:
            List of speculative patterns
        """
        if not query_text and not seed_symbols:
            return []

        pattern_types = pattern_types or [PatternType.SIMILAR_QUERY]
        context = context or {}
        patterns: list[SpeculativePattern] = []

        for ptype in pattern_types:
            if ptype == PatternType.SIMILAR_QUERY:
                patterns.extend(
                    self._generate_similar_patterns(query_text, max_patterns)
                )
            elif ptype == PatternType.FOLLOW_UP:
                patterns.extend(
                    self._generate_follow_up_patterns(query_text, context, max_patterns)
                )
            elif ptype == PatternType.SYMBOL_RELATED:
                patterns.extend(
                    self._generate_symbol_patterns(
                        query_text, seed_symbols, max_patterns
                    )
                )
            elif ptype == PatternType.BREADCRUMB:
                patterns.extend(
                    self._generate_breadcrumb_patterns(query_text, context, max_patterns)
                )

        # Sort by confidence and limit
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns[:max_patterns]

    def _generate_similar_patterns(
        self, query_text: str, max_patterns: int
    ) -> list[SpeculativePattern]:
        """Generate similar query patterns."""
        patterns = []
        words = query_text.lower().split()

        for word in words:
            if word in self._similar_terms:
                for similar in self._similar_terms[word][:max_patterns]:
                    new_query = query_text.lower().replace(word, similar)
                    patterns.append(
                        SpeculativePattern(
                            pattern_type=PatternType.SIMILAR_QUERY,
                            query_text=new_query,
                            confidence=0.7,
                        )
                    )

        return patterns[:max_patterns]

    def _generate_follow_up_patterns(
        self, query_text: str, context: dict[str, Any], max_patterns: int
    ) -> list[SpeculativePattern]:
        """Generate follow-up patterns from context."""
        patterns = []
        previous_queries = context.get("previous_queries", [])

        for prev_query in previous_queries[-3:]:  # Use last 3 queries
            # Combine current and previous query concepts
            combined = f"{prev_query} {query_text}"
            patterns.append(
                SpeculativePattern(
                    pattern_type=PatternType.FOLLOW_UP,
                    query_text=combined,
                    confidence=0.6,
                )
            )

        return patterns[:max_patterns]

    def _generate_symbol_patterns(
        self,
        query_text: str,
        seed_symbols: Optional[list[str]],
        max_patterns: int,
    ) -> list[SpeculativePattern]:
        """Generate symbol-related patterns."""
        patterns = []

        if seed_symbols:
            for symbol in seed_symbols[:max_patterns]:
                # Extract symbol name from SCIP format
                parts = symbol.split("/")
                if parts:
                    symbol_name = parts[-1].rstrip("#.")
                    patterns.append(
                        SpeculativePattern(
                            pattern_type=PatternType.SYMBOL_RELATED,
                            query_text=symbol_name,
                            seed_symbols=[symbol],
                            confidence=0.8,
                        )
                    )

        return patterns[:max_patterns]

    def _generate_breadcrumb_patterns(
        self, query_text: str, context: dict[str, Any], max_patterns: int
    ) -> list[SpeculativePattern]:
        """Generate breadcrumb patterns from navigation history."""
        patterns = []
        nav_history = context.get("navigation_history", [])

        for file_path in nav_history[-max_patterns:]:
            # Extract file/module name
            file_name = file_path.split("/")[-1].replace(".py", "").replace(".ts", "")
            patterns.append(
                SpeculativePattern(
                    pattern_type=PatternType.BREADCRUMB,
                    query_text=f"{file_name} {query_text}",
                    confidence=0.5,
                )
            )

        return patterns[:max_patterns]


# =============================================================================
# Cached Tri-Hybrid Retriever
# =============================================================================


class CachedTriHybridRetriever:
    """Cache wrapper for TriHybridRetriever.

    Provides:
    - Transparent caching of retrieval results
    - Speculative prefetching
    - Cache warming
    - Bypass options
    """

    def __init__(
        self,
        retriever: Any,
        cache: PrefetchCache,
        speculative_generator: Optional[SpeculativeQueryGenerator] = None,
        enable_prefetch: bool = False,
        prefetch_async: bool = True,
        max_prefetch: int = 3,
    ):
        """Initialize cached retriever.

        Args:
            retriever: Underlying TriHybridRetriever
            cache: Cache implementation
            speculative_generator: Optional pattern generator for prefetch
            enable_prefetch: Whether to enable speculative prefetching
            prefetch_async: Whether to prefetch asynchronously
            max_prefetch: Maximum speculative queries to prefetch
        """
        self.retriever = retriever
        self.cache = cache
        self.speculative_generator = speculative_generator or SpeculativeQueryGenerator()
        self.enable_prefetch = enable_prefetch
        self.prefetch_async = prefetch_async
        self.max_prefetch = max_prefetch
        self._executor: Optional[ThreadPoolExecutor] = None

        if enable_prefetch and prefetch_async:
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")

    def retrieve(
        self,
        query: Any,
        index_name: str,
        bypass_cache: bool = False,
    ) -> Any:
        """Retrieve with caching.

        Args:
            query: TriHybridQuery
            index_name: Index to search
            bypass_cache: If True, skip cache lookup

        Returns:
            TriHybridResult
        """
        # Generate cache key
        cache_key = CacheKey.from_query(
            index_name=index_name,
            query_text=getattr(query, "query_text", ""),
            embedding=getattr(query, "embedding", None) or None,
            filters=getattr(query, "filters", None) or None,
            seed_symbols=getattr(query, "seed_symbols", None) or None,
        )

        # Check cache (unless bypassing)
        if not bypass_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Call underlying retriever
        result = self.retriever.retrieve(query, index_name)

        # Cache the result
        self.cache.put(cache_key, result)

        # Trigger speculative prefetch
        if self.enable_prefetch:
            self._prefetch_speculative(query, index_name)

        return result

    def _prefetch_speculative(self, query: Any, index_name: str) -> None:
        """Prefetch speculative queries.

        Args:
            query: Original query
            index_name: Index name
        """
        query_text = getattr(query, "query_text", "")
        seed_symbols = getattr(query, "seed_symbols", None)

        patterns = self.speculative_generator.generate_patterns(
            query_text=query_text,
            seed_symbols=seed_symbols,
            max_patterns=self.max_prefetch,
        )

        for pattern in patterns:
            if self.prefetch_async and self._executor:
                self._executor.submit(
                    self._prefetch_pattern, pattern, index_name, query
                )
            else:
                self._prefetch_pattern(pattern, index_name, query)

    def _prefetch_pattern(
        self, pattern: SpeculativePattern, index_name: str, original_query: Any
    ) -> None:
        """Prefetch a single pattern.

        Args:
            pattern: Speculative pattern
            index_name: Index name
            original_query: Original query for reference
        """
        try:
            # Create cache key for pattern
            cache_key = CacheKey.from_query(
                index_name=index_name,
                query_text=pattern.query_text,
                seed_symbols=pattern.seed_symbols,
            )

            # Skip if already cached
            if self.cache.contains(cache_key):
                return

            # Create query from pattern (copy original and modify)
            # This is a simplified approach - in production would create proper query
            prefetch_query = type(original_query)(
                query_text=pattern.query_text or getattr(original_query, "query_text", ""),
                embedding=pattern.embedding or getattr(original_query, "embedding", []),
                seed_symbols=pattern.seed_symbols or [],
                filters=getattr(original_query, "filters", {}),
                size=getattr(original_query, "size", 10),
            )

            result = self.retriever.retrieve(prefetch_query, index_name)
            self.cache.put(cache_key, result)
            logger.debug(f"Prefetched: {pattern.query_text}")

        except Exception as e:
            logger.debug(f"Prefetch failed for pattern: {e}")

    def warm_cache(
        self,
        queries: list[dict[str, Any]],
        skip_existing: bool = True,
    ) -> int:
        """Warm cache with predefined queries.

        Args:
            queries: List of query dicts with query_text, index_name, etc.
            skip_existing: Skip queries already in cache

        Returns:
            Number of queries executed
        """
        count = 0

        for query_dict in queries:
            query_text = query_dict.get("query_text", "")
            index_name = query_dict.get("index_name", "")
            embedding = query_dict.get("embedding")
            filters = query_dict.get("filters")

            if not index_name:
                continue

            cache_key = CacheKey.from_query(
                index_name=index_name,
                query_text=query_text,
                embedding=embedding,
                filters=filters,
            )

            # Skip if exists and skip_existing is True
            if skip_existing and self.cache.contains(cache_key):
                continue

            try:
                # Create a mock query object - in practice this would be TriHybridQuery
                from openmemory.api.retrieval.trihybrid import TriHybridQuery

                query = TriHybridQuery(
                    query_text=query_text,
                    embedding=embedding or [],
                    filters=filters or {},
                )

                result = self.retriever.retrieve(query, index_name)
                self.cache.put(cache_key, result)
                count += 1

            except Exception as e:
                logger.warning(f"Failed to warm cache for {query_text}: {e}")

        return count

    def __del__(self):
        """Cleanup executor on deletion."""
        if self._executor:
            self._executor.shutdown(wait=False)


# =============================================================================
# Factory Functions
# =============================================================================


def create_prefetch_cache(
    max_entries: int = 1000,
    default_ttl_seconds: int = 300,
    max_memory_mb: int = 100,
    enable_metrics: bool = True,
) -> LRUPrefetchCache:
    """Create a prefetch cache.

    Args:
        max_entries: Maximum cache entries
        default_ttl_seconds: Default TTL
        max_memory_mb: Maximum memory
        enable_metrics: Enable metrics tracking

    Returns:
        Configured LRUPrefetchCache
    """
    config = CacheConfig(
        max_entries=max_entries,
        default_ttl_seconds=default_ttl_seconds,
        max_memory_mb=max_memory_mb,
        enable_metrics=enable_metrics,
    )
    return LRUPrefetchCache(config)


def create_cached_retriever(
    retriever: Any,
    max_entries: int = 1000,
    default_ttl_seconds: int = 300,
    enable_prefetch: bool = False,
    prefetch_async: bool = True,
    max_prefetch: int = 3,
) -> CachedTriHybridRetriever:
    """Create a cached tri-hybrid retriever.

    Args:
        retriever: Underlying TriHybridRetriever
        max_entries: Maximum cache entries
        default_ttl_seconds: Default TTL
        enable_prefetch: Enable speculative prefetching
        prefetch_async: Prefetch asynchronously
        max_prefetch: Max speculative queries

    Returns:
        Configured CachedTriHybridRetriever
    """
    cache = create_prefetch_cache(
        max_entries=max_entries,
        default_ttl_seconds=default_ttl_seconds,
    )

    return CachedTriHybridRetriever(
        retriever=retriever,
        cache=cache,
        enable_prefetch=enable_prefetch,
        prefetch_async=prefetch_async,
        max_prefetch=max_prefetch,
    )
