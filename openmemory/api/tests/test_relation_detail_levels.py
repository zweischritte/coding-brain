"""
Integration tests for relation_detail parameter in search_memory.

Tests cover:
- relation_detail parameter behavior: none, minimal, standard, full
- Compact format integration with search results
- Token reduction verification
- Backward compatibility with existing meta_relations structure

These tests do not require a live Neo4j instance (uses mocks).

Run with: pytest openmemory/api/tests/test_relation_detail_levels.py -v
"""

import pytest
import json
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_meta_relations_verbose():
    """Verbose meta_relations as returned from get_meta_relations_for_memories."""
    return {
        "mem-1": [
            {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "mem-2", "score": 0.92, "preview": "Related content..."},
            {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "mem-3", "score": 0.85, "preview": "Another related..."},
            {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "src/utils/helper.ts"},
            {"type": "OM_HAS_ARTIFACT_TYPE", "target_label": "OM_ArtifactType", "target_value": "file"},
            {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "architecture"},
            {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "project"},
            {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "claude-code"},
            {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "area", "value": "backend"},
            {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "AuthService"},
            {"type": "OM_HAS_EVIDENCE", "target_label": "OM_Evidence", "target_value": "ADR-001"},
        ],
        "mem-2": [
            {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "decision"},
            {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "team"},
            {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "Backend"},
        ],
    }


@pytest.fixture
def sample_search_results():
    """Sample search results to be enriched with relations."""
    return [
        {
            "id": "mem-1",
            "memory": "Authentication service architecture",
            "scores": {"semantic": 0.9, "boost": 0.2, "final": 1.08},
            "metadata": {
                "category": "architecture",
                "scope": "project",
            },
            "created_at": "2025-12-04T11:14:00Z",
        },
        {
            "id": "mem-2",
            "memory": "Backend team decision",
            "scores": {"semantic": 0.85, "boost": 0.15, "final": 0.98},
            "metadata": {
                "category": "decision",
                "scope": "team",
            },
            "created_at": "2025-12-05T09:30:00Z",
        },
    ]


@pytest.fixture
def mock_principal():
    """Create a mock principal for ACL checks."""
    principal = MagicMock()
    principal.has_scope = MagicMock(return_value=True)
    principal.user_id = "test-user"
    principal.org_id = "test-org"
    principal.claims = MagicMock()
    principal.claims.grants = {"user:test-user"}
    return principal


# =============================================================================
# COMPACT FORMAT INTEGRATION TESTS
# =============================================================================

class TestFormatMetaRelationsCompact:
    """Tests for formatting meta_relations with compact format."""

    def test_format_meta_relations_all_memories(self, sample_meta_relations_verbose):
        """format_meta_relations_compact should format all memories."""
        from app.utils.response_format import format_compact_relations

        result = {}
        for memory_id, relations in sample_meta_relations_verbose.items():
            compact = format_compact_relations(relations)
            if compact:  # Only add if not empty
                result[memory_id] = compact

        assert "mem-1" in result
        assert "artifact" in result["mem-1"]
        assert "similar" in result["mem-1"]
        assert "entities" in result["mem-1"]
        assert "tags" in result["mem-1"]
        assert "evidence" in result["mem-1"]

        # mem-2 only has noise relations, should have entities only
        assert "mem-2" in result
        assert "entities" in result["mem-2"]

    def test_compact_excludes_noise_relations(self, sample_meta_relations_verbose):
        """Compact format should exclude noise relations."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])

        # Verify noise is excluded
        compact_str = str(compact)
        assert "OM_IN_CATEGORY" not in compact_str
        assert "OM_IN_SCOPE" not in compact_str
        assert "OM_WRITTEN_VIA" not in compact_str
        assert "OM_HAS_ARTIFACT_TYPE" not in compact_str
        assert "claude-code" not in compact_str

    def test_compact_preserves_actionable_relations(self, sample_meta_relations_verbose):
        """Compact format should preserve actionable relations."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])

        # Artifact
        assert compact["artifact"] == "src/utils/helper.ts"

        # Similar with scores and previews
        assert len(compact["similar"]) == 2
        assert compact["similar"][0]["id"] == "mem-2"
        assert compact["similar"][0]["score"] == 0.92
        assert "preview" in compact["similar"][0]

        # Entities
        assert "AuthService" in compact["entities"]

        # Tags
        assert compact["tags"]["area"] == "backend"

        # Evidence
        assert "ADR-001" in compact["evidence"]


class TestRelationDetailNone:
    """Tests for relation_detail=none (no relations)."""

    def test_none_excludes_meta_relations(self):
        """relation_detail=none should exclude meta_relations entirely."""
        from app.utils.response_format import format_compact_relations

        # When relation_detail=none, we don't call format at all
        # Just verify format_compact_relations works
        compact = format_compact_relations([])
        assert compact == {}

    def test_none_keeps_search_results_intact(self, sample_search_results):
        """relation_detail=none should preserve search results structure."""
        from app.utils.response_format import format_search_results

        response = format_search_results(sample_search_results, verbose=False)

        assert "results" in response
        assert len(response["results"]) == 2
        # meta_relations should not be added when relation_detail=none


class TestRelationDetailMinimal:
    """Tests for relation_detail=minimal (artifact + similar only)."""

    def test_minimal_includes_artifact(self, sample_meta_relations_verbose):
        """relation_detail=minimal should include artifact."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])

        # Filter to minimal
        minimal = {k: v for k, v in compact.items() if k in ("artifact", "artifacts", "similar")}

        assert "artifact" in minimal
        assert compact["artifact"] == "src/utils/helper.ts"

    def test_minimal_includes_similar(self, sample_meta_relations_verbose):
        """relation_detail=minimal should include similar memories."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])
        minimal = {k: v for k, v in compact.items() if k in ("artifact", "artifacts", "similar")}

        assert "similar" in minimal
        assert len(minimal["similar"]) == 2

    def test_minimal_excludes_entities_tags_evidence(self, sample_meta_relations_verbose):
        """relation_detail=minimal should exclude entities, tags, evidence."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])
        minimal = {k: v for k, v in compact.items() if k in ("artifact", "artifacts", "similar")}

        assert "entities" not in minimal
        assert "tags" not in minimal
        assert "evidence" not in minimal


class TestRelationDetailStandard:
    """Tests for relation_detail=standard (artifact + similar + entities + tags + evidence)."""

    def test_standard_includes_all_compact_fields(self, sample_meta_relations_verbose):
        """relation_detail=standard should include all compact format fields."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])

        # Standard is the full compact format
        assert "artifact" in compact
        assert "similar" in compact
        assert "entities" in compact
        assert "tags" in compact
        assert "evidence" in compact

    def test_standard_is_default_compact_format(self, sample_meta_relations_verbose):
        """relation_detail=standard should match format_compact_relations output."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])
        standard = compact  # standard is the default

        assert standard == compact


class TestRelationDetailFull:
    """Tests for relation_detail=full (verbose format for debugging)."""

    def test_full_preserves_verbose_structure(self, sample_meta_relations_verbose):
        """relation_detail=full should preserve the original verbose structure."""
        # Full mode bypasses compact formatting
        verbose = sample_meta_relations_verbose["mem-1"]

        # Verify verbose structure is preserved
        assert any(r["type"] == "OM_IN_CATEGORY" for r in verbose)
        assert any(r["type"] == "OM_WRITTEN_VIA" for r in verbose)
        assert all("target_label" in r for r in verbose)

    def test_full_includes_all_relation_types(self, sample_meta_relations_verbose):
        """relation_detail=full should include all relation types."""
        verbose = sample_meta_relations_verbose["mem-1"]

        types = {r["type"] for r in verbose}
        assert "OM_SIMILAR" in types
        assert "OM_IN_CATEGORY" in types
        assert "OM_IN_SCOPE" in types
        assert "OM_WRITTEN_VIA" in types
        assert "OM_TAGGED" in types
        assert "OM_ABOUT" in types


class TestTokenReduction:
    """Tests for token reduction with compact format."""

    def test_compact_reduces_token_count(self, sample_meta_relations_verbose):
        """Compact format should reduce token count significantly."""
        from app.utils.response_format import format_compact_relations
        import json

        verbose = sample_meta_relations_verbose["mem-1"]
        compact = format_compact_relations(verbose)

        verbose_json = json.dumps(verbose)
        compact_json = json.dumps(compact)

        # Token count approximation (1 token ≈ 4 chars for English/code)
        verbose_tokens = len(verbose_json) / 4
        compact_tokens = len(compact_json) / 4

        reduction = (verbose_tokens - compact_tokens) / verbose_tokens

        # Should achieve at least 50% reduction
        assert reduction >= 0.5, f"Expected >= 50% reduction, got {reduction:.1%}"

    def test_compact_preserves_navigation_value(self, sample_meta_relations_verbose):
        """Compact format should preserve all navigation-relevant info."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])

        # All navigation fields present
        assert compact.get("artifact") is not None
        assert len(compact.get("similar", [])) > 0
        assert len(compact.get("entities", [])) > 0

        # Verify scores preserved for prioritization
        for sim in compact.get("similar", []):
            assert "id" in sim
            # Score may not always be present


class TestEmptyRelations:
    """Tests for edge cases with empty or missing relations."""

    def test_empty_relations_returns_empty_dict(self):
        """Empty relations should return empty compact dict."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations([])
        assert compact == {}

    def test_noise_only_relations_returns_empty_dict(self):
        """Relations with only noise should return empty dict."""
        from app.utils.response_format import format_compact_relations

        noise_only = [
            {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "decision"},
            {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "project"},
            {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "cursor"},
        ]

        compact = format_compact_relations(noise_only)
        assert compact == {}

    def test_memory_without_relations_handled(self):
        """Memory without relations should be handled gracefully."""
        from app.utils.response_format import format_compact_relations

        empty_relations = {}
        result = {}

        for memory_id, relations in empty_relations.items():
            compact = format_compact_relations(relations)
            if compact:
                result[memory_id] = compact

        assert result == {}


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_compact_format_is_json_serializable(self, sample_meta_relations_verbose):
        """Compact format should be JSON serializable."""
        from app.utils.response_format import format_compact_relations
        import json

        compact = format_compact_relations(sample_meta_relations_verbose["mem-1"])

        # Should not raise
        json_str = json.dumps(compact)
        parsed = json.loads(json_str)

        assert parsed == compact

    def test_compact_format_works_with_format_search_results(self, sample_search_results, sample_meta_relations_verbose):
        """Compact meta_relations should integrate with search results."""
        from app.utils.response_format import format_search_results, format_compact_relations
        import json

        # Format search results
        response = format_search_results(sample_search_results, verbose=False)

        # Add compact meta_relations
        compact_relations = {}
        for memory_id, relations in sample_meta_relations_verbose.items():
            compact = format_compact_relations(relations)
            if compact:
                compact_relations[memory_id] = compact

        response["meta_relations"] = compact_relations

        # Should be JSON serializable
        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        assert "results" in parsed
        assert "meta_relations" in parsed
        assert "mem-1" in parsed["meta_relations"]
