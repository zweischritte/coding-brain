"""Tests for structured memory validation utilities."""

import pytest

from app.utils.structured_memory import (
    StructuredMemoryError,
    apply_metadata_updates,
    build_structured_memory,
    normalize_metadata_for_create,
    sanitize_metadata,
    validate_update_fields,
)


def test_build_structured_memory_validates():
    text, metadata = build_structured_memory(
        text="Adopt feature flags for risky deploys",
        category="decision",
        scope="org",
        artifact_type="service",
        artifact_ref="deploy-service",
        entity="platform-team",
        access_entity="org:cloudfactory",  # Required for scope=org
        source="user",
        evidence=["ADR-101"],
        tags={"rollout": True},
    )

    assert text == "Adopt feature flags for risky deploys"
    assert metadata["category"] == "decision"
    assert metadata["scope"] == "org"
    assert metadata["artifact_type"] == "service"
    assert metadata["artifact_ref"] == "deploy-service"
    assert metadata["entity"] == "platform-team"
    assert metadata["access_entity"] == "org:cloudfactory"
    assert metadata["source"] == "user"
    assert metadata["evidence"] == ["ADR-101"]
    assert metadata["tags"] == {"rollout": True}


def test_normalize_metadata_requires_category_and_scope():
    with pytest.raises(StructuredMemoryError):
        normalize_metadata_for_create({"scope": "user"})


def test_normalize_metadata_validates_and_preserves_extras():
    metadata = normalize_metadata_for_create({
        "category": "security",
        "scope": "project",
        "artifact_type": "repo",
        "artifact_ref": "coding-brain",
        "entity": "security-team",
        "access_entity": "project:cloudfactory/security/coding-brain",  # Required for scope=project
        "evidence": ["PR-99"],
        "tags": {"auth": True},
        "source": "inference",
        "custom_field": "custom",
        "src": "legacy",
    })

    assert metadata["category"] == "security"
    assert metadata["scope"] == "project"
    assert metadata["artifact_type"] == "repo"
    assert metadata["artifact_ref"] == "coding-brain"
    assert metadata["entity"] == "security-team"
    assert metadata["access_entity"] == "project:cloudfactory/security/coding-brain"
    assert metadata["evidence"] == ["PR-99"]
    assert metadata["tags"] == {"auth": True}
    assert metadata["source"] == "inference"
    assert metadata["custom_field"] == "custom"
    assert "src" not in metadata


def test_validate_update_fields_only_returns_passed_values():
    validated = validate_update_fields(
        category="workflow",
        scope="team",
        artifact_type="module",
        artifact_ref="payments/core.py",
    )
    assert validated == {
        "category": "workflow",
        "scope": "team",
        "artifact_type": "module",
        "artifact_ref": "payments/core.py",
    }


def test_apply_metadata_updates_merges_tags():
    current = {"category": "decision", "tags": {"old": True}}
    validated = {"scope": "team"}
    updated = apply_metadata_updates(
        current_metadata=current,
        validated_fields=validated,
        add_tags={"new": True},
        remove_tags=["old"],
    )
    assert updated["scope"] == "team"
    assert updated["tags"] == {"new": True}


def test_apply_metadata_updates_drops_empty_tags():
    current = {"category": "decision", "tags": {"old": True}}
    updated = apply_metadata_updates(
        current_metadata=current,
        validated_fields={},
        add_tags=None,
        remove_tags=["old"],
    )
    assert "tags" not in updated


def test_sanitize_metadata_drops_legacy_keys():
    metadata = sanitize_metadata({"from": "legacy", "category": "testing"})
    assert metadata == {"category": "testing"}
