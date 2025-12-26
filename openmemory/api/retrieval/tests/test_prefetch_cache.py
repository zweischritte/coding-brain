"""Tests for prefetch cache with LRU eviction (FR-010).

This module tests the speculative retrieval and prefetch cache system:
- LRU cache with configurable size
- Cache key generation from queries
- TTL-based expiration
- Cache warming/prefetching
- Metrics tracking (hit rate, evictions)
- Scope-aware caching
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import Mock, MagicMock
import pytest

# These imports will fail until we implement the module
from openmemory.api.retrieval.prefetch_cache import (
    # Core types
    CacheConfig,
    CacheEntry,
    CacheKey,
    CacheMetrics,
    CacheStats,
    # Main cache
    PrefetchCache,
    LRUPrefetchCache,
    # Speculative patterns
    SpeculativePattern,
    PatternType,
    SpeculativeQueryGenerator,
    # Cache wrapper for retriever
    CachedTriHybridRetriever,
    # Factory
    create_prefetch_cache,
    create_cached_retriever,
    # Errors
    CacheError,
    CacheKeyError,
    CacheMissError,
)


# =============================================================================
# CacheConfig Tests
# =============================================================================


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.max_entries == 1000
        assert config.default_ttl_seconds == 300  # 5 minutes
        assert config.max_memory_mb == 100
        assert config.enable_metrics is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            max_entries=500,
            default_ttl_seconds=600,
            max_memory_mb=200,
            enable_metrics=False,
        )
        assert config.max_entries == 500
        assert config.default_ttl_seconds == 600
        assert config.max_memory_mb == 200
        assert config.enable_metrics is False

    def test_config_validation_max_entries(self):
        """Test validation rejects invalid max_entries."""
        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=0)

        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=-1)

    def test_config_validation_ttl(self):
        """Test validation rejects invalid TTL."""
        with pytest.raises(ValueError, match="ttl"):
            CacheConfig(default_ttl_seconds=0)

        with pytest.raises(ValueError, match="ttl"):
            CacheConfig(default_ttl_seconds=-10)

    def test_config_scope_ttl_overrides(self):
        """Test per-scope TTL override configuration."""
        config = CacheConfig(
            default_ttl_seconds=300,
            scope_ttl_overrides={
                "graph": 60,  # Graph context expires faster
                "lexical": 300,
                "vector": 300,
            },
        )
        assert config.scope_ttl_overrides["graph"] == 60
        assert config.scope_ttl_overrides["lexical"] == 300


# =============================================================================
# CacheKey Tests
# =============================================================================


class TestCacheKey:
    """Tests for CacheKey generation."""

    def test_key_from_query_text(self):
        """Test key generation from query text."""
        key = CacheKey.from_query(
            query_text="database connection pool",
            index_name="code_index",
        )
        assert key.hash is not None
        assert len(key.hash) == 64  # SHA-256 hex
        assert key.index_name == "code_index"

    def test_key_from_embedding(self):
        """Test key generation from embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        key = CacheKey.from_query(
            embedding=embedding,
            index_name="code_index",
        )
        assert key.hash is not None
        assert key.index_name == "code_index"

    def test_key_from_combined(self):
        """Test key generation from text + embedding."""
        key = CacheKey.from_query(
            query_text="database",
            embedding=[0.1, 0.2, 0.3],
            index_name="code_index",
        )
        assert key.hash is not None

    def test_same_input_same_key(self):
        """Test deterministic key generation."""
        key1 = CacheKey.from_query(
            query_text="database connection",
            index_name="code_index",
        )
        key2 = CacheKey.from_query(
            query_text="database connection",
            index_name="code_index",
        )
        assert key1 == key2
        assert key1.hash == key2.hash

    def test_different_input_different_key(self):
        """Test different inputs produce different keys."""
        key1 = CacheKey.from_query(query_text="database", index_name="code_index")
        key2 = CacheKey.from_query(query_text="connection", index_name="code_index")
        assert key1 != key2

    def test_key_includes_filters(self):
        """Test filters affect cache key."""
        key1 = CacheKey.from_query(
            query_text="database",
            index_name="code_index",
            filters={"language": "python"},
        )
        key2 = CacheKey.from_query(
            query_text="database",
            index_name="code_index",
            filters={"language": "java"},
        )
        assert key1 != key2

    def test_key_includes_seed_symbols(self):
        """Test seed symbols affect cache key."""
        key1 = CacheKey.from_query(
            query_text="database",
            index_name="code_index",
            seed_symbols=["sym1", "sym2"],
        )
        key2 = CacheKey.from_query(
            query_text="database",
            index_name="code_index",
            seed_symbols=["sym3"],
        )
        assert key1 != key2

    def test_key_requires_query_or_embedding(self):
        """Test key generation requires at least query or embedding."""
        with pytest.raises(CacheKeyError, match="query_text or embedding"):
            CacheKey.from_query(index_name="code_index")

    def test_key_string_representation(self):
        """Test key has useful string representation."""
        key = CacheKey.from_query(
            query_text="database",
            index_name="code_index",
        )
        key_str = str(key)
        assert "code_index" in key_str
        assert key.hash[:8] in key_str  # Truncated hash


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_creation(self):
        """Test creating cache entry."""
        mock_result = Mock()
        entry = CacheEntry(
            key=CacheKey.from_query(query_text="test", index_name="idx"),
            value=mock_result,
            ttl_seconds=300,
        )
        assert entry.value == mock_result
        assert entry.ttl_seconds == 300
        assert entry.created_at > 0
        assert entry.access_count == 0

    def test_entry_expiration(self):
        """Test entry expiration detection."""
        entry = CacheEntry(
            key=CacheKey.from_query(query_text="test", index_name="idx"),
            value=Mock(),
            ttl_seconds=1,  # 1 second TTL
        )
        assert not entry.is_expired()

        # Simulate time passing
        entry.created_at = time.time() - 2
        assert entry.is_expired()

    def test_entry_touch_updates_access(self):
        """Test touching entry updates access time and count."""
        entry = CacheEntry(
            key=CacheKey.from_query(query_text="test", index_name="idx"),
            value=Mock(),
            ttl_seconds=300,
        )
        initial_access = entry.last_accessed_at
        initial_count = entry.access_count

        time.sleep(0.01)  # Small delay
        entry.touch()

        assert entry.last_accessed_at > initial_access
        assert entry.access_count == initial_count + 1

    def test_entry_size_estimation(self):
        """Test memory size estimation for entry."""
        entry = CacheEntry(
            key=CacheKey.from_query(query_text="test", index_name="idx"),
            value={"data": "x" * 1000},
            ttl_seconds=300,
        )
        size = entry.estimated_size_bytes
        # Size estimation is approximate - just ensure it's positive
        assert size > 0


# =============================================================================
# LRUPrefetchCache Tests
# =============================================================================


class TestLRUPrefetchCache:
    """Tests for LRU prefetch cache implementation."""

    def test_create_cache(self):
        """Test cache creation with config."""
        config = CacheConfig(max_entries=100)
        cache = LRUPrefetchCache(config)
        assert cache.config.max_entries == 100
        assert cache.size() == 0

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="test", index_name="idx")
        value = {"result": "data"}

        cache.put(key, value)
        assert cache.size() == 1

        retrieved = cache.get(key)
        assert retrieved == value

    def test_get_miss(self):
        """Test cache miss returns None."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="nonexistent", index_name="idx")

        result = cache.get(key)
        assert result is None

    def test_get_or_raise_miss(self):
        """Test get_or_raise raises on miss."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="nonexistent", index_name="idx")

        with pytest.raises(CacheMissError):
            cache.get_or_raise(key)

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=3))

        # Fill cache
        for i in range(3):
            key = CacheKey.from_query(query_text=f"query{i}", index_name="idx")
            cache.put(key, {"data": i})

        assert cache.size() == 3

        # Access first key to make it recently used
        key0 = CacheKey.from_query(query_text="query0", index_name="idx")
        cache.get(key0)

        # Add new entry - should evict query1 (least recently used)
        new_key = CacheKey.from_query(query_text="query3", index_name="idx")
        cache.put(new_key, {"data": 3})

        assert cache.size() == 3
        assert cache.get(key0) is not None  # query0 still exists
        assert cache.get(new_key) is not None  # query3 exists

        # query1 should be evicted (it was LRU)
        key1 = CacheKey.from_query(query_text="query1", index_name="idx")
        assert cache.get(key1) is None

    def test_ttl_expiration(self):
        """Test entries expire based on TTL."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10, default_ttl_seconds=1))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        cache.put(key, {"data": "value"})
        assert cache.get(key) is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should return None for expired entry
        assert cache.get(key) is None

    def test_custom_ttl_per_entry(self):
        """Test custom TTL for specific entries."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10, default_ttl_seconds=300))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        # Put with short TTL
        cache.put(key, {"data": "value"}, ttl_seconds=1)

        time.sleep(1.1)
        assert cache.get(key) is None

    def test_invalidate_key(self):
        """Test invalidating a specific key."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        cache.put(key, {"data": "value"})
        assert cache.get(key) is not None

        cache.invalidate(key)
        assert cache.get(key) is None
        assert cache.size() == 0

    def test_invalidate_by_index(self):
        """Test invalidating all entries for an index."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))

        # Add entries for different indexes
        key1 = CacheKey.from_query(query_text="q1", index_name="idx1")
        key2 = CacheKey.from_query(query_text="q2", index_name="idx1")
        key3 = CacheKey.from_query(query_text="q3", index_name="idx2")

        cache.put(key1, {"data": 1})
        cache.put(key2, {"data": 2})
        cache.put(key3, {"data": 3})

        assert cache.size() == 3

        # Invalidate idx1
        count = cache.invalidate_by_index("idx1")
        assert count == 2
        assert cache.size() == 1
        assert cache.get(key3) is not None

    def test_clear_cache(self):
        """Test clearing entire cache."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))

        for i in range(5):
            key = CacheKey.from_query(query_text=f"q{i}", index_name="idx")
            cache.put(key, {"data": i})

        assert cache.size() == 5
        cache.clear()
        assert cache.size() == 0

    def test_contains_key(self):
        """Test checking if key exists."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        assert not cache.contains(key)
        cache.put(key, {"data": "value"})
        assert cache.contains(key)


# =============================================================================
# CacheMetrics Tests
# =============================================================================


class TestCacheMetrics:
    """Tests for cache metrics tracking."""

    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        config = CacheConfig(max_entries=10, enable_metrics=True)
        cache = LRUPrefetchCache(config)
        key = CacheKey.from_query(query_text="test", index_name="idx")

        # Initial metrics
        metrics = cache.get_metrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0

        # Miss
        cache.get(key)
        metrics = cache.get_metrics()
        assert metrics.misses == 1
        assert metrics.hits == 0

        # Put and hit
        cache.put(key, {"data": "value"})
        cache.get(key)
        metrics = cache.get_metrics()
        assert metrics.hits == 1
        assert metrics.misses == 1

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        cache.put(key, {"data": "value"})

        # 3 hits
        for _ in range(3):
            cache.get(key)

        # 1 miss
        miss_key = CacheKey.from_query(query_text="missing", index_name="idx")
        cache.get(miss_key)

        metrics = cache.get_metrics()
        assert metrics.hits == 3
        assert metrics.misses == 1
        assert metrics.hit_rate == 0.75  # 3/4

    def test_eviction_tracking(self):
        """Test eviction count tracking."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=2))

        # Fill cache
        for i in range(2):
            key = CacheKey.from_query(query_text=f"q{i}", index_name="idx")
            cache.put(key, {"data": i})

        metrics = cache.get_metrics()
        assert metrics.evictions == 0

        # Add one more - should evict
        key = CacheKey.from_query(query_text="q2", index_name="idx")
        cache.put(key, {"data": 2})

        metrics = cache.get_metrics()
        assert metrics.evictions == 1

    def test_reset_metrics(self):
        """Test resetting metrics."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        cache.put(key, {"data": "value"})
        cache.get(key)
        cache.get(key)

        metrics = cache.get_metrics()
        assert metrics.hits == 2

        cache.reset_metrics()
        metrics = cache.get_metrics()
        assert metrics.hits == 0
        assert metrics.misses == 0

    def test_metrics_disabled(self):
        """Test metrics can be disabled."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=10, enable_metrics=False))
        key = CacheKey.from_query(query_text="test", index_name="idx")

        cache.put(key, {"data": "value"})
        cache.get(key)

        # Metrics should be zero/empty when disabled
        metrics = cache.get_metrics()
        assert metrics.hits == 0


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for cache statistics."""

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = LRUPrefetchCache(CacheConfig(max_entries=100))

        for i in range(10):
            key = CacheKey.from_query(query_text=f"q{i}", index_name="idx")
            cache.put(key, {"data": "x" * 100})

        stats = cache.get_stats()
        assert stats.entry_count == 10
        assert stats.max_entries == 100
        assert stats.estimated_memory_bytes > 0
        assert stats.utilization == 0.1  # 10/100


# =============================================================================
# SpeculativePattern Tests
# =============================================================================


class TestSpeculativePattern:
    """Tests for speculative query patterns."""

    def test_pattern_creation(self):
        """Test creating a speculative pattern."""
        pattern = SpeculativePattern(
            pattern_type=PatternType.SIMILAR_QUERY,
            query_text="database connection",
            confidence=0.85,
        )
        assert pattern.pattern_type == PatternType.SIMILAR_QUERY
        assert pattern.confidence == 0.85

    def test_pattern_types(self):
        """Test different pattern types."""
        assert PatternType.SIMILAR_QUERY.value == "similar_query"
        assert PatternType.FOLLOW_UP.value == "follow_up"
        assert PatternType.SYMBOL_RELATED.value == "symbol_related"
        assert PatternType.BREADCRUMB.value == "breadcrumb"


class TestSpeculativeQueryGenerator:
    """Tests for speculative query generation."""

    def test_generate_similar_queries(self):
        """Test generating similar query patterns."""
        generator = SpeculativeQueryGenerator()

        patterns = generator.generate_patterns(
            query_text="database connection pool",
            pattern_types=[PatternType.SIMILAR_QUERY],
            max_patterns=3,
        )

        assert len(patterns) <= 3
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.SIMILAR_QUERY
            assert pattern.query_text is not None
            assert 0 <= pattern.confidence <= 1

    def test_generate_follow_up_patterns(self):
        """Test generating follow-up patterns from context."""
        generator = SpeculativeQueryGenerator()

        patterns = generator.generate_patterns(
            query_text="get_connection method",
            pattern_types=[PatternType.FOLLOW_UP],
            context={"previous_queries": ["database pool"]},
            max_patterns=2,
        )

        assert len(patterns) <= 2
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.FOLLOW_UP

    def test_generate_symbol_related_patterns(self):
        """Test generating symbol-related patterns."""
        generator = SpeculativeQueryGenerator()

        patterns = generator.generate_patterns(
            query_text="ConnectionPool",
            seed_symbols=["scip-python db/ConnectionPool#"],
            pattern_types=[PatternType.SYMBOL_RELATED],
            max_patterns=3,
        )

        assert len(patterns) <= 3
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.SYMBOL_RELATED

    def test_generate_breadcrumb_patterns(self):
        """Test generating breadcrumb patterns from navigation history."""
        generator = SpeculativeQueryGenerator()

        patterns = generator.generate_patterns(
            query_text="error handling",
            pattern_types=[PatternType.BREADCRUMB],
            context={
                "navigation_history": [
                    "database.py",
                    "connection.py",
                    "pool.py",
                ]
            },
            max_patterns=2,
        )

        # Should generate patterns based on file proximity
        assert len(patterns) <= 2

    def test_no_patterns_when_empty_query(self):
        """Test empty patterns for empty query."""
        generator = SpeculativeQueryGenerator()

        patterns = generator.generate_patterns(
            query_text="",
            pattern_types=[PatternType.SIMILAR_QUERY],
        )

        assert len(patterns) == 0


# =============================================================================
# CachedTriHybridRetriever Tests
# =============================================================================


class TestCachedTriHybridRetriever:
    """Tests for cached tri-hybrid retriever wrapper."""

    def test_create_cached_retriever(self):
        """Test creating cached retriever."""
        mock_retriever = Mock()
        cache = LRUPrefetchCache(CacheConfig(max_entries=10))

        cached = CachedTriHybridRetriever(
            retriever=mock_retriever,
            cache=cache,
        )

        assert cached.retriever == mock_retriever
        assert cached.cache == cache

    def test_retrieve_cache_miss_calls_backend(self):
        """Test cache miss calls underlying retriever."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        cached = CachedTriHybridRetriever(retriever=mock_retriever, cache=cache)

        # Create mock query
        mock_query = Mock()
        mock_query.query_text = "test query"
        mock_query.embedding = []
        mock_query.filters = {}
        mock_query.seed_symbols = []

        result = cached.retrieve(mock_query, "code_index")

        assert result == mock_result
        mock_retriever.retrieve.assert_called_once()

    def test_retrieve_cache_hit_skips_backend(self):
        """Test cache hit returns cached result without calling backend."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        cached = CachedTriHybridRetriever(retriever=mock_retriever, cache=cache)

        # Create mock query
        mock_query = Mock()
        mock_query.query_text = "test query"
        mock_query.embedding = []
        mock_query.filters = {}
        mock_query.seed_symbols = []

        # First call - cache miss
        cached.retrieve(mock_query, "code_index")
        assert mock_retriever.retrieve.call_count == 1

        # Second call - cache hit
        cached.retrieve(mock_query, "code_index")
        assert mock_retriever.retrieve.call_count == 1  # Still 1

    def test_retrieve_with_prefetch(self):
        """Test speculative prefetching on retrieve."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        generator = SpeculativeQueryGenerator()

        cached = CachedTriHybridRetriever(
            retriever=mock_retriever,
            cache=cache,
            speculative_generator=generator,
            enable_prefetch=True,
            max_prefetch=2,
        )

        mock_query = Mock()
        mock_query.query_text = "database connection"
        mock_query.embedding = []
        mock_query.filters = {}
        mock_query.seed_symbols = []

        cached.retrieve(mock_query, "code_index")

        # Backend should be called for main query + speculative queries
        assert mock_retriever.retrieve.call_count >= 1

    def test_bypass_cache_option(self):
        """Test bypassing cache for specific queries."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        cached = CachedTriHybridRetriever(retriever=mock_retriever, cache=cache)

        mock_query = Mock()
        mock_query.query_text = "test query"
        mock_query.embedding = []
        mock_query.filters = {}
        mock_query.seed_symbols = []

        # First call - fills cache
        cached.retrieve(mock_query, "code_index")
        assert mock_retriever.retrieve.call_count == 1

        # Second call with bypass - should call backend
        cached.retrieve(mock_query, "code_index", bypass_cache=True)
        assert mock_retriever.retrieve.call_count == 2

    def test_prefetch_async(self):
        """Test async prefetching doesn't block main query."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        generator = SpeculativeQueryGenerator()

        cached = CachedTriHybridRetriever(
            retriever=mock_retriever,
            cache=cache,
            speculative_generator=generator,
            enable_prefetch=True,
            prefetch_async=True,
        )

        mock_query = Mock()
        mock_query.query_text = "database"
        mock_query.embedding = []
        mock_query.filters = {}
        mock_query.seed_symbols = []

        # Should return quickly without waiting for prefetch
        result = cached.retrieve(mock_query, "code_index")
        assert result == mock_result


# =============================================================================
# Cache Warming Tests
# =============================================================================


class TestCacheWarming:
    """Tests for cache warming functionality."""

    def test_warm_cache_with_queries(self):
        """Test warming cache with predefined queries."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        cached = CachedTriHybridRetriever(retriever=mock_retriever, cache=cache)

        # Create queries for warming
        warm_queries = [
            {"query_text": "database", "index_name": "code_index"},
            {"query_text": "connection", "index_name": "code_index"},
            {"query_text": "pool", "index_name": "code_index"},
        ]

        count = cached.warm_cache(warm_queries)
        assert count == 3
        assert mock_retriever.retrieve.call_count == 3

    def test_warm_cache_skips_existing(self):
        """Test cache warming skips already cached entries."""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_retriever.retrieve.return_value = mock_result

        cache = LRUPrefetchCache(CacheConfig(max_entries=10))
        cached = CachedTriHybridRetriever(retriever=mock_retriever, cache=cache)

        # Pre-populate one entry
        mock_query = Mock()
        mock_query.query_text = "database"
        mock_query.embedding = []
        mock_query.filters = {}
        mock_query.seed_symbols = []
        cached.retrieve(mock_query, "code_index")

        # Warm with 3 queries, including the pre-populated one
        warm_queries = [
            {"query_text": "database", "index_name": "code_index"},
            {"query_text": "connection", "index_name": "code_index"},
            {"query_text": "pool", "index_name": "code_index"},
        ]

        count = cached.warm_cache(warm_queries, skip_existing=True)
        assert count == 2  # Only 2 new entries
        assert mock_retriever.retrieve.call_count == 3  # Initial + 2 new


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_prefetch_cache(self):
        """Test creating prefetch cache via factory."""
        cache = create_prefetch_cache(
            max_entries=500,
            default_ttl_seconds=600,
        )
        assert isinstance(cache, LRUPrefetchCache)
        assert cache.config.max_entries == 500
        assert cache.config.default_ttl_seconds == 600

    def test_create_prefetch_cache_defaults(self):
        """Test factory with default values."""
        cache = create_prefetch_cache()
        assert isinstance(cache, LRUPrefetchCache)
        assert cache.config.max_entries == 1000
        assert cache.config.default_ttl_seconds == 300

    def test_create_cached_retriever(self):
        """Test creating cached retriever via factory."""
        mock_retriever = Mock()

        cached = create_cached_retriever(
            retriever=mock_retriever,
            max_entries=200,
            enable_prefetch=True,
        )

        assert isinstance(cached, CachedTriHybridRetriever)
        assert cached.retriever == mock_retriever
        assert cached.enable_prefetch is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestPrefetchCacheIntegration:
    """Integration tests for prefetch cache."""

    @pytest.mark.integration
    def test_full_retrieval_flow_with_cache(self):
        """Test complete retrieval flow with caching."""
        # This would test with real TriHybridRetriever
        # Marked as integration test
        pass

    @pytest.mark.integration
    def test_cache_invalidation_on_index_update(self):
        """Test cache invalidation when index is updated."""
        # This would test cache invalidation hooks
        pass
