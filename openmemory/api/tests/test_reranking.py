"""Tests for metadata reranking utilities with structured fields."""

from app.utils.reranking import (
    BoostConfig,
    ExclusionFilters,
    SearchContext,
    compute_boost,
    compute_metadata_boost,
    compute_recency_boost,
    compute_tag_boost,
    should_exclude,
)


def test_metadata_boost_matches_structured_fields():
    metadata = {
        "category": "architecture",
        "scope": "project",
        "artifact_type": "repo",
        "artifact_ref": "openmemory/api/app/mcp_server.py",
        "entity": "platform-team",
    }
    context = SearchContext(
        category="architecture",
        scope="project",
        artifact_type="repo",
        artifact_ref="openmemory/api/app/mcp_server.py",
        entity="platform-team",
    )

    config = BoostConfig()
    boost, breakdown = compute_metadata_boost(metadata, context, config)

    assert boost > 0
    assert breakdown["category"] == config.category
    assert breakdown["scope"] == config.scope
    assert breakdown["artifact_type"] == config.artifact_type
    assert breakdown["artifact_ref"] == config.artifact_ref
    assert breakdown["entity"] == config.entity


def test_tag_boost_counts_matches():
    stored_tags = {"important": True, "review": True}
    boost, count = compute_tag_boost(stored_tags, ["important", "missing"], BoostConfig())
    assert count == 1
    assert boost > 0


def test_recency_boost_disabled_for_zero_weight():
    boost, age = compute_recency_boost("2025-01-01T00:00:00Z", 0.0)
    assert boost == 0.0
    assert age is None


def test_compute_boost_includes_tags_and_recency():
    metadata = {"category": "workflow"}
    context = SearchContext(category="workflow", tags=["infra"], recency_weight=0.5)
    stored_tags = {"infra": True}

    boost, breakdown = compute_boost(
        metadata=metadata,
        stored_tags=stored_tags,
        context=context,
        created_at_str="2025-01-01T00:00:00Z",
    )

    assert boost > 0
    assert "metadata" in breakdown
    assert "tags" in breakdown
    assert "recency" in breakdown


def test_should_exclude_by_state():
    payload = {"state": "deleted"}
    excluded, reason = should_exclude(payload, {}, {}, ExclusionFilters())
    assert excluded is True
    assert reason == "state"


def test_should_exclude_by_tag():
    payload = {"state": "active"}
    filters = ExclusionFilters(exclude_tags=["skip"], boost_tags=[])
    excluded, reason = should_exclude(payload, {}, {"skip": True}, filters)
    assert excluded is True
    assert reason == "tag"
