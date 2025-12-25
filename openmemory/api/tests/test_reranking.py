"""
Unit tests for the reranking module.

Tests cover:
- DateTime parsing
- Tag normalization
- Metadata boost calculation
- Tag boost calculation
- Recency boost calculation
- Full boost calculation
- Exclusion logic
- Final score calculation

Run with: pytest openmemory/api/tests/test_reranking.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from math import exp

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.reranking import (
    BoostConfig,
    SearchContext,
    ExclusionFilters,
    parse_datetime,
    normalize_tags,
    compute_metadata_boost,
    compute_tag_boost,
    compute_recency_boost,
    compute_boost,
    should_exclude,
    calculate_final_score,
    DEFAULT_BOOST_CONFIG,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_metadata():
    """Sample memory metadata for testing."""
    return {
        "vault": "FRACTURE_LOG",
        "layer": "emotional",
        "vector": "say",
        "circuit": 2,
        "re": "BMG",
        "state": "active",
    }


@pytest.fixture
def sample_tags():
    """Sample memory tags for testing."""
    return {
        "trigger": True,
        "intensity": 7,
        "important": True,
    }


@pytest.fixture
def now():
    """Current UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# PARSE_DATETIME TESTS
# =============================================================================

class TestParseDatetime:
    """Tests for parse_datetime function."""

    def test_parse_z_suffix(self):
        """Parse ISO datetime with Z suffix."""
        result = parse_datetime("2025-12-04T11:14:00Z")
        assert result is not None
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 4
        assert result.hour == 11
        assert result.minute == 14
        assert result.tzinfo == timezone.utc

    def test_parse_explicit_offset(self):
        """Parse ISO datetime with explicit UTC offset."""
        result = parse_datetime("2025-12-04T11:14:00+00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_parse_naive_datetime(self):
        """Parse naive datetime (assumes UTC)."""
        result = parse_datetime("2025-12-04T11:14:00")
        assert result is not None
        assert result.tzinfo == timezone.utc

    def test_parse_none(self):
        """Return None for None input."""
        assert parse_datetime(None) is None

    def test_parse_empty_string(self):
        """Return None for empty string."""
        assert parse_datetime("") is None

    def test_parse_invalid_format(self):
        """Return None for invalid format."""
        assert parse_datetime("not-a-date") is None
        assert parse_datetime("12/04/2025") is None


# =============================================================================
# NORMALIZE_TAGS TESTS
# =============================================================================

class TestNormalizeTags:
    """Tests for normalize_tags function."""

    def test_dict_passthrough(self):
        """Dict tags pass through unchanged."""
        tags = {"trigger": True, "intensity": 7}
        result = normalize_tags(tags)
        assert result == tags

    def test_list_to_dict(self):
        """List tags convert to dict with True values."""
        tags = ["trigger", "important"]
        result = normalize_tags(tags)
        assert result == {"trigger": True, "important": True}

    def test_empty_list(self):
        """Empty list returns empty dict."""
        assert normalize_tags([]) == {}

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert normalize_tags({}) == {}

    def test_none_returns_empty(self):
        """None returns empty dict."""
        assert normalize_tags(None) == {}


# =============================================================================
# COMPUTE_METADATA_BOOST TESTS
# =============================================================================

class TestComputeMetadataBoost:
    """Tests for compute_metadata_boost function."""

    def test_entity_match(self, sample_metadata):
        """Entity match adds boost."""
        context = SearchContext(entity="BMG")
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == DEFAULT_BOOST_CONFIG.entity  # 0.5
        assert "entity" in breakdown

    def test_layer_match(self, sample_metadata):
        """Layer match adds boost."""
        context = SearchContext(layer="emotional")
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == DEFAULT_BOOST_CONFIG.layer  # 0.3
        assert "layer" in breakdown

    def test_vault_match(self, sample_metadata):
        """Vault match adds boost."""
        context = SearchContext(vault="FRACTURE_LOG")
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == DEFAULT_BOOST_CONFIG.vault  # 0.2
        assert "vault" in breakdown

    def test_vector_match(self, sample_metadata):
        """Vector match adds boost."""
        context = SearchContext(vector="say")
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == DEFAULT_BOOST_CONFIG.vector  # 0.15
        assert "vector" in breakdown

    def test_circuit_match(self, sample_metadata):
        """Circuit match adds boost."""
        context = SearchContext(circuit=2)
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == DEFAULT_BOOST_CONFIG.circuit  # 0.1
        assert "circuit" in breakdown

    def test_multiple_matches(self, sample_metadata):
        """Multiple matches add cumulative boost."""
        context = SearchContext(
            entity="BMG",
            layer="emotional",
            vault="FRACTURE_LOG",
        )
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        expected = (
            DEFAULT_BOOST_CONFIG.entity +
            DEFAULT_BOOST_CONFIG.layer +
            DEFAULT_BOOST_CONFIG.vault
        )
        assert boost == expected  # 0.5 + 0.3 + 0.2 = 1.0
        assert len(breakdown) == 3

    def test_no_match(self, sample_metadata):
        """No match returns zero boost."""
        context = SearchContext(entity="Other", layer="cognitive")
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == 0.0
        assert len(breakdown) == 0

    def test_empty_context(self, sample_metadata):
        """Empty context returns zero boost."""
        context = SearchContext()
        boost, breakdown = compute_metadata_boost(sample_metadata, context)
        assert boost == 0.0
        assert len(breakdown) == 0

    def test_circuit_type_coercion(self):
        """Circuit comparison handles string metadata."""
        metadata = {"circuit": "2"}  # String from JSON
        context = SearchContext(circuit=2)
        boost, breakdown = compute_metadata_boost(metadata, context)
        assert boost == DEFAULT_BOOST_CONFIG.circuit


# =============================================================================
# COMPUTE_TAG_BOOST TESTS
# =============================================================================

class TestComputeTagBoost:
    """Tests for compute_tag_boost function."""

    def test_single_tag_match(self, sample_tags):
        """Single tag match adds boost."""
        boost, count = compute_tag_boost(sample_tags, ["trigger"])
        assert boost == DEFAULT_BOOST_CONFIG.tag  # 0.1
        assert count == 1

    def test_multiple_tag_matches(self, sample_tags):
        """Multiple tag matches add cumulative boost."""
        boost, count = compute_tag_boost(sample_tags, ["trigger", "important"])
        assert boost == DEFAULT_BOOST_CONFIG.tag * 2  # 0.2
        assert count == 2

    def test_tag_boost_capped(self, sample_tags):
        """Tag boost is capped at max."""
        # Create context with many tags
        many_tags = ["trigger", "important", "tag3", "tag4", "tag5", "tag6"]
        stored = {t: True for t in many_tags}
        boost, count = compute_tag_boost(stored, many_tags)
        assert boost == DEFAULT_BOOST_CONFIG.max_tag_boost  # 0.5 cap
        assert count == 6

    def test_no_tag_match(self, sample_tags):
        """No tag match returns zero boost."""
        boost, count = compute_tag_boost(sample_tags, ["nonexistent"])
        assert boost == 0.0
        assert count == 0

    def test_empty_context_tags(self, sample_tags):
        """Empty context tags returns zero boost."""
        boost, count = compute_tag_boost(sample_tags, [])
        assert boost == 0.0
        assert count == 0

    def test_partial_match(self, sample_tags):
        """Partial match only counts matching tags."""
        boost, count = compute_tag_boost(sample_tags, ["trigger", "nonexistent"])
        assert boost == DEFAULT_BOOST_CONFIG.tag  # 0.1
        assert count == 1


# =============================================================================
# COMPUTE_RECENCY_BOOST TESTS
# =============================================================================

class TestComputeRecencyBoost:
    """Tests for compute_recency_boost function."""

    def test_today_gets_full_boost(self, now):
        """Memory from today gets full recency boost."""
        created_at = now.isoformat()
        boost, age = compute_recency_boost(created_at, recency_weight=1.0, halflife_days=45)

        # At age=0, factor is 1.0
        expected = 1.0 * 1.0 * DEFAULT_BOOST_CONFIG.max_recency_boost
        assert abs(boost - expected) < 0.01  # ~0.5
        assert age == 0

    def test_older_memory_decays(self, now):
        """Older memory has lower recency boost."""
        old_date = (now - timedelta(days=90)).isoformat()
        boost, age = compute_recency_boost(old_date, recency_weight=1.0, halflife_days=45)

        # At 90 days with halflife=45, decay_rate=90, factor = e^(-90/90) ≈ 0.37
        assert boost < DEFAULT_BOOST_CONFIG.max_recency_boost * 0.5
        assert age == 90

    def test_recency_weight_scales_boost(self, now):
        """Recency weight scales the boost."""
        created_at = now.isoformat()

        full_boost, _ = compute_recency_boost(created_at, recency_weight=1.0, halflife_days=45)
        half_boost, _ = compute_recency_boost(created_at, recency_weight=0.5, halflife_days=45)

        assert abs(half_boost - full_boost * 0.5) < 0.01

    def test_zero_weight_no_boost(self, now):
        """Zero recency weight returns no boost."""
        created_at = now.isoformat()
        boost, age = compute_recency_boost(created_at, recency_weight=0.0, halflife_days=45)
        assert boost == 0.0
        assert age is None

    def test_no_created_at(self):
        """Missing created_at returns no boost."""
        boost, age = compute_recency_boost(None, recency_weight=1.0, halflife_days=45)
        assert boost == 0.0
        assert age is None

    def test_invalid_date(self):
        """Invalid date returns no boost."""
        boost, age = compute_recency_boost("invalid", recency_weight=1.0, halflife_days=45)
        assert boost == 0.0
        assert age is None

    def test_halflife_affects_decay(self, now):
        """Shorter halflife causes faster decay."""
        old_date = (now - timedelta(days=45)).isoformat()

        short_halflife, _ = compute_recency_boost(old_date, recency_weight=1.0, halflife_days=15)
        long_halflife, _ = compute_recency_boost(old_date, recency_weight=1.0, halflife_days=90)

        assert short_halflife < long_halflife


# =============================================================================
# COMPUTE_BOOST (INTEGRATION) TESTS
# =============================================================================

class TestComputeBoost:
    """Tests for the full compute_boost function."""

    def test_all_boosts_combined(self, sample_metadata, sample_tags, now):
        """All boost sources combine correctly."""
        created_at = now.isoformat()
        context = SearchContext(
            entity="BMG",
            layer="emotional",
            tags=["trigger"],
            recency_weight=0.5,
        )

        boost, breakdown = compute_boost(
            metadata=sample_metadata,
            stored_tags=sample_tags,
            context=context,
            created_at_str=created_at,
        )

        # Should have all three boost types
        assert "metadata" in breakdown
        assert "tags" in breakdown
        assert "recency" in breakdown
        assert boost > 0

    def test_boost_capped(self, sample_metadata, sample_tags, now):
        """Total boost is capped at max."""
        created_at = now.isoformat()
        context = SearchContext(
            entity="BMG",
            layer="emotional",
            vault="FRACTURE_LOG",
            vector="say",
            circuit=2,
            tags=["trigger", "important"],
            recency_weight=1.0,
        )

        boost, breakdown = compute_boost(
            metadata=sample_metadata,
            stored_tags=sample_tags,
            context=context,
            created_at_str=created_at,
        )

        assert boost <= DEFAULT_BOOST_CONFIG.max_total_boost  # 1.5
        if boost == DEFAULT_BOOST_CONFIG.max_total_boost:
            assert breakdown.get("capped") is True

    def test_no_boost_context(self, sample_metadata, sample_tags):
        """No boost context returns zero."""
        context = SearchContext()  # Empty context

        boost, breakdown = compute_boost(
            metadata=sample_metadata,
            stored_tags=sample_tags,
            context=context,
        )

        assert boost == 0.0


# =============================================================================
# SHOULD_EXCLUDE TESTS
# =============================================================================

class TestShouldExclude:
    """Tests for should_exclude function."""

    def test_exclude_deleted_state(self, sample_tags):
        """Deleted state is excluded by default."""
        payload = {"created_at": "2025-12-04T11:14:00Z"}
        metadata = {"state": "deleted"}
        filters = ExclusionFilters()

        excluded, reason = should_exclude(payload, metadata, sample_tags, filters)
        assert excluded is True
        assert "state:deleted" in reason

    def test_exclude_by_tag(self, sample_tags):
        """Memories with excluded tags are filtered."""
        payload = {"created_at": "2025-12-04T11:14:00Z"}
        metadata = {"state": "active"}
        tags_with_rejected = {"rejected": True}
        filters = ExclusionFilters(exclude_tags=["rejected"])

        excluded, reason = should_exclude(payload, metadata, tags_with_rejected, filters)
        assert excluded is True
        assert "excluded_tag:rejected" in reason

    def test_silent_excluded_by_default(self):
        """Silent tag is excluded by default."""
        payload = {"created_at": "2025-12-04T11:14:00Z"}
        metadata = {"state": "active"}
        tags = {"silent": True}
        filters = ExclusionFilters()

        excluded, reason = should_exclude(payload, metadata, tags, filters)
        assert excluded is True
        assert "silent_default" in reason

    def test_silent_not_excluded_when_boosted(self):
        """Silent tag not excluded when in boost_tags."""
        payload = {"created_at": "2025-12-04T11:14:00Z"}
        metadata = {"state": "active"}
        tags = {"silent": True}
        filters = ExclusionFilters(boost_tags=["silent"])

        excluded, reason = should_exclude(payload, metadata, tags, filters)
        assert excluded is False

    def test_date_filter_created_after(self, now):
        """created_after filter excludes old memories."""
        yesterday = (now - timedelta(days=1)).isoformat()
        payload = {"created_at": yesterday}
        metadata = {"state": "active"}
        filters = ExclusionFilters(created_after=now)

        excluded, reason = should_exclude(payload, metadata, {}, filters)
        assert excluded is True
        assert "created_after" in reason

    def test_date_filter_created_before(self, now):
        """created_before filter excludes new memories."""
        tomorrow = (now + timedelta(days=1)).isoformat()
        payload = {"created_at": tomorrow}
        metadata = {"state": "active"}
        yesterday = now - timedelta(days=1)
        filters = ExclusionFilters(created_before=yesterday)

        excluded, reason = should_exclude(payload, metadata, {}, filters)
        assert excluded is True
        assert "created_before" in reason

    def test_not_excluded(self, sample_tags, now):
        """Active memory without exclusion tags passes."""
        payload = {"created_at": now.isoformat()}
        metadata = {"state": "active"}
        filters = ExclusionFilters()

        excluded, reason = should_exclude(payload, metadata, sample_tags, filters)
        assert excluded is False
        assert reason is None


# =============================================================================
# CALCULATE_FINAL_SCORE TESTS
# =============================================================================

class TestCalculateFinalScore:
    """Tests for calculate_final_score function."""

    def test_no_boost(self):
        """Zero boost returns semantic score unchanged."""
        score = calculate_final_score(0.8, 0.0)
        assert score == 0.8

    def test_with_boost(self):
        """Boost multiplies score."""
        score = calculate_final_score(0.8, 0.5)
        assert score == 0.8 * 1.5  # 1.2

    def test_max_boost(self):
        """Max boost gives maximum multiplier."""
        score = calculate_final_score(0.8, 1.5)
        assert score == 0.8 * 2.5  # 2.0

    def test_zero_semantic_score(self):
        """Zero semantic score stays zero."""
        score = calculate_final_score(0.0, 1.0)
        assert score == 0.0


# =============================================================================
# INTEGRATION: SEARCH SCENARIOS
# =============================================================================

class TestSearchScenarios:
    """Integration tests simulating real search scenarios."""

    def test_core_pattern_search_no_recency(self, sample_metadata, sample_tags):
        """Core pattern search: no recency bias, entity boost only."""
        context = SearchContext(entity="BMG", recency_weight=0.0)

        boost, breakdown = compute_boost(
            metadata=sample_metadata,
            stored_tags=sample_tags,
            context=context,
        )

        assert boost == DEFAULT_BOOST_CONFIG.entity  # 0.5
        assert "recency" not in breakdown

    def test_current_situation_with_recency(self, sample_metadata, sample_tags, now):
        """Current situation: moderate recency boost."""
        created_at = now.isoformat()
        context = SearchContext(recency_weight=0.4)

        boost, breakdown = compute_boost(
            metadata=sample_metadata,
            stored_tags=sample_tags,
            context=context,
            created_at_str=created_at,
        )

        assert "recency" in breakdown
        assert boost > 0

    def test_time_window_with_filters(self, now):
        """Explicit time window: date filters exclude old content."""
        last_week = now - timedelta(days=7)
        old_memory = (now - timedelta(days=30)).isoformat()

        payload = {"created_at": old_memory}
        metadata = {"state": "active"}
        filters = ExclusionFilters(created_after=last_week)

        excluded, _ = should_exclude(payload, metadata, {}, filters)
        assert excluded is True

    def test_combined_boost_and_filter(self, sample_metadata, sample_tags, now):
        """Combined: boost context + date filter work independently."""
        recent_date = now.isoformat()
        context = SearchContext(entity="BMG", layer="emotional", recency_weight=0.4)

        # Create filters that DON'T exclude this memory
        filters = ExclusionFilters(
            created_after=now - timedelta(days=1),
            created_before=now + timedelta(days=1),
        )

        # Check exclusion
        payload = {"created_at": recent_date}
        excluded, _ = should_exclude(payload, sample_metadata, sample_tags, filters)
        assert excluded is False

        # Check boost
        boost, breakdown = compute_boost(
            metadata=sample_metadata,
            stored_tags=sample_tags,
            context=context,
            created_at_str=recent_date,
        )

        # Should have entity + layer + recency boosts
        assert "metadata" in breakdown
        assert "recency" in breakdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
