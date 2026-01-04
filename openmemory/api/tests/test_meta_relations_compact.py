"""
Unit tests for compact meta_relations formatting.

Tests cover:
- format_compact_relations() function for converting verbose relations to compact format
- Edge cases: empty relations, multiple artifacts, missing fields
- Relation type filtering (excludes OM_WRITTEN_VIA, OM_IN_CATEGORY, OM_IN_SCOPE)
- OM_SIMILAR extraction with scores and previews
- Entity, tag, and evidence extraction

These tests do not require a live Neo4j instance.

Run with: pytest openmemory/api/tests/test_meta_relations_compact.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def verbose_relations_full():
    """Full verbose relations as returned from projector."""
    return [
        {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "def-456", "score": 0.92, "preview": "BooksReportService extends AbstractReportService..."},
        {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "ghi-789", "score": 0.87, "preview": "MoviesReportService extends AbstractReportService..."},
        {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "jkl-012", "score": 0.65, "preview": "Report pagination uses Redis cursor..."},
        {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "apps/merlin/src/reports/abstract-report.service.ts"},
        {"type": "OM_HAS_ARTIFACT_TYPE", "target_label": "OM_ArtifactType", "target_value": "file"},
        {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "architecture"},
        {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "project"},
        {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "claude-code"},
        {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "area", "value": "search"},
        {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "importance", "value": "high"},
        {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "ReportsModule"},
        {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "Redis"},
        {"type": "OM_HAS_EVIDENCE", "target_label": "OM_Evidence", "target_value": "ADR-014"},
        {"type": "OM_HAS_EVIDENCE", "target_label": "OM_Evidence", "target_value": "PR-123"},
    ]


@pytest.fixture
def verbose_relations_artifact_only():
    """Relations with only artifact reference."""
    return [
        {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "src/utils/helper.ts"},
        {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "convention"},
        {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "claude-code"},
    ]


@pytest.fixture
def verbose_relations_similar_only():
    """Relations with only similar memories."""
    return [
        {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "abc-123", "score": 0.95, "preview": "First similar memory..."},
        {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "def-456", "score": 0.88, "preview": "Second similar memory..."},
    ]


@pytest.fixture
def verbose_relations_entities_only():
    """Relations with only entity references."""
    return [
        {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "AuthService"},
        {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "UserModule"},
        {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "org"},
    ]


@pytest.fixture
def verbose_relations_tags_only():
    """Relations with only tags."""
    return [
        {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "trigger", "value": True},
        {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "priority", "value": "high"},
        {"type": "OM_TAGGED", "target_label": "OM_Tag", "target_value": "reviewed"},
    ]


@pytest.fixture
def verbose_relations_multiple_artifacts():
    """Relations with multiple artifact references."""
    return [
        {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "src/auth/login.ts"},
        {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "src/auth/logout.ts"},
        {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "src/auth/session.ts"},
    ]


@pytest.fixture
def verbose_relations_noise_only():
    """Relations with only noise (should return empty compact)."""
    return [
        {"type": "OM_IN_CATEGORY", "target_label": "OM_Category", "target_value": "decision"},
        {"type": "OM_IN_SCOPE", "target_label": "OM_Scope", "target_value": "project"},
        {"type": "OM_WRITTEN_VIA", "target_label": "OM_App", "target_value": "cursor"},
        {"type": "OM_HAS_ARTIFACT_TYPE", "target_label": "OM_ArtifactType", "target_value": "file"},
    ]


# =============================================================================
# COMPACT RELATIONS FORMATTER TESTS
# =============================================================================

class TestFormatCompactRelationsArtifact:
    """Tests for artifact extraction from relations."""

    def test_single_artifact_extracted(self, verbose_relations_artifact_only):
        """Single artifact should be extracted as 'artifact' field."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_artifact_only)

        assert "artifact" in compact
        assert compact["artifact"] == "src/utils/helper.ts"
        # Should not have 'artifacts' key for single artifact
        assert "artifacts" not in compact

    def test_multiple_artifacts_as_array(self, verbose_relations_multiple_artifacts):
        """Multiple artifacts should be extracted as 'artifacts' array."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_multiple_artifacts)

        assert "artifacts" in compact
        assert len(compact["artifacts"]) == 3
        assert "src/auth/login.ts" in compact["artifacts"]
        assert "src/auth/logout.ts" in compact["artifacts"]
        assert "src/auth/session.ts" in compact["artifacts"]
        # Should not have 'artifact' key for multiple artifacts
        assert "artifact" not in compact

    def test_artifact_from_full_relations(self, verbose_relations_full):
        """Artifact should be extracted from full relations set."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        assert compact["artifact"] == "apps/merlin/src/reports/abstract-report.service.ts"


class TestFormatCompactRelationsSimilar:
    """Tests for similar memory extraction from relations."""

    def test_similar_extracted_with_scores(self, verbose_relations_similar_only):
        """Similar memories should include ID, score, and preview."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_similar_only)

        assert "similar" in compact
        assert len(compact["similar"]) == 2

        # First similar (highest score)
        assert compact["similar"][0]["id"] == "abc-123"
        assert compact["similar"][0]["score"] == 0.95
        assert compact["similar"][0]["preview"] == "First similar memory..."

        # Second similar
        assert compact["similar"][1]["id"] == "def-456"
        assert compact["similar"][1]["score"] == 0.88

    def test_similar_from_full_relations(self, verbose_relations_full):
        """Similar memories should be extracted from full relations."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        assert "similar" in compact
        assert len(compact["similar"]) == 3

        # Verify IDs are present
        similar_ids = [s["id"] for s in compact["similar"]]
        assert "def-456" in similar_ids
        assert "ghi-789" in similar_ids
        assert "jkl-012" in similar_ids

    def test_similar_without_score_still_included(self):
        """Similar memory without score should still be included."""
        from app.utils.response_format import format_compact_relations

        relations = [
            {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "no-score-id"},
        ]

        compact = format_compact_relations(relations)

        assert "similar" in compact
        assert len(compact["similar"]) == 1
        assert compact["similar"][0]["id"] == "no-score-id"
        assert "score" not in compact["similar"][0]

    def test_similar_without_preview_still_included(self):
        """Similar memory without preview should still be included."""
        from app.utils.response_format import format_compact_relations

        relations = [
            {"type": "OM_SIMILAR", "target_label": "OM_Memory", "target_value": "abc-123", "score": 0.9},
        ]

        compact = format_compact_relations(relations)

        assert compact["similar"][0]["id"] == "abc-123"
        assert compact["similar"][0]["score"] == 0.9
        assert "preview" not in compact["similar"][0]


class TestFormatCompactRelationsEntities:
    """Tests for entity extraction from relations."""

    def test_entities_extracted_as_list(self, verbose_relations_entities_only):
        """Entities should be extracted as flat list."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_entities_only)

        assert "entities" in compact
        assert len(compact["entities"]) == 2
        assert "AuthService" in compact["entities"]
        assert "UserModule" in compact["entities"]

    def test_entities_from_full_relations(self, verbose_relations_full):
        """Entities should be extracted from full relations."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        assert "entities" in compact
        assert "ReportsModule" in compact["entities"]
        assert "Redis" in compact["entities"]


class TestFormatCompactRelationsTags:
    """Tests for tag extraction from relations."""

    def test_tags_extracted_as_dict(self, verbose_relations_tags_only):
        """Tags should be extracted as key-value dict."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_tags_only)

        assert "tags" in compact
        assert compact["tags"]["trigger"] is True
        assert compact["tags"]["priority"] == "high"
        # Tag without value should default to True
        assert compact["tags"]["reviewed"] is True

    def test_tags_from_full_relations(self, verbose_relations_full):
        """Tags should be extracted from full relations."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        assert "tags" in compact
        assert compact["tags"]["area"] == "search"
        assert compact["tags"]["importance"] == "high"


class TestFormatCompactRelationsEvidence:
    """Tests for evidence extraction from relations."""

    def test_evidence_extracted_as_list(self, verbose_relations_full):
        """Evidence should be extracted as flat list."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        assert "evidence" in compact
        assert len(compact["evidence"]) == 2
        assert "ADR-014" in compact["evidence"]
        assert "PR-123" in compact["evidence"]

    def test_evidence_single_item(self):
        """Single evidence item should still be a list."""
        from app.utils.response_format import format_compact_relations

        relations = [
            {"type": "OM_HAS_EVIDENCE", "target_label": "OM_Evidence", "target_value": "ADR-001"},
        ]

        compact = format_compact_relations(relations)

        assert "evidence" in compact
        assert compact["evidence"] == ["ADR-001"]


class TestFormatCompactRelationsEmpty:
    """Tests for empty or minimal relations."""

    def test_empty_relations_returns_empty_dict(self):
        """Empty relations list should return empty dict."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations([])

        assert compact == {}

    def test_noise_only_returns_empty_dict(self, verbose_relations_noise_only):
        """Relations with only noise types should return empty dict."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_noise_only)

        assert compact == {}

    def test_none_relations_returns_empty_dict(self):
        """None should be handled gracefully."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(None)

        assert compact == {}


class TestFormatCompactRelationsExcludes:
    """Tests for excluded relation types."""

    def test_excludes_om_written_via(self, verbose_relations_full):
        """OM_WRITTEN_VIA should be excluded from compact format."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        # Check that claude-code app name is not anywhere in output
        compact_str = str(compact)
        assert "claude-code" not in compact_str
        assert "OM_App" not in compact_str

    def test_excludes_om_in_category(self, verbose_relations_full):
        """OM_IN_CATEGORY should be excluded (already in main result)."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        # Category is in main result, not in relations
        compact_str = str(compact)
        assert "OM_Category" not in compact_str

    def test_excludes_om_in_scope(self, verbose_relations_full):
        """OM_IN_SCOPE should be excluded (already in main result)."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        compact_str = str(compact)
        assert "OM_Scope" not in compact_str

    def test_excludes_om_has_artifact_type(self, verbose_relations_full):
        """OM_HAS_ARTIFACT_TYPE should be excluded (redundant with artifact)."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        compact_str = str(compact)
        assert "OM_ArtifactType" not in compact_str


class TestFormatCompactRelationsComplete:
    """Integration tests for complete compact format."""

    def test_full_relations_produces_complete_compact(self, verbose_relations_full):
        """Full relations should produce complete compact format."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        # All expected fields should be present
        assert "artifact" in compact
        assert "similar" in compact
        assert "entities" in compact
        assert "tags" in compact
        assert "evidence" in compact

        # Verify structure
        assert compact["artifact"] == "apps/merlin/src/reports/abstract-report.service.ts"
        assert len(compact["similar"]) == 3
        assert len(compact["entities"]) == 2
        assert len(compact["tags"]) == 2
        assert len(compact["evidence"]) == 2

    def test_compact_format_has_no_verbose_fields(self, verbose_relations_full):
        """Compact format should not contain verbose field names."""
        from app.utils.response_format import format_compact_relations

        compact = format_compact_relations(verbose_relations_full)

        compact_str = str(compact)
        assert "target_label" not in compact_str
        assert "target_value" not in compact_str
        assert "type" not in compact_str or "artifact_type" in compact_str  # artifact_type is OK


class TestFormatCompactRelationsNullHandling:
    """Tests for null/missing value handling."""

    def test_handles_null_target_value(self):
        """Relations with null target_value should be skipped."""
        from app.utils.response_format import format_compact_relations

        relations = [
            {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": None},
            {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "ValidEntity"},
        ]

        compact = format_compact_relations(relations)

        assert "entities" in compact
        assert compact["entities"] == ["ValidEntity"]

    def test_handles_empty_string_target_value(self):
        """Relations with empty string target_value should be skipped."""
        from app.utils.response_format import format_compact_relations

        relations = [
            {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": ""},
            {"type": "OM_REFERENCES_ARTIFACT", "target_label": "OM_ArtifactRef", "target_value": "valid/path.ts"},
        ]

        compact = format_compact_relations(relations)

        assert "artifact" in compact
        assert compact["artifact"] == "valid/path.ts"

    def test_handles_missing_type_field(self):
        """Relations without type field should be skipped gracefully."""
        from app.utils.response_format import format_compact_relations

        relations = [
            {"target_label": "OM_Entity", "target_value": "InvalidRelation"},
            {"type": "OM_ABOUT", "target_label": "OM_Entity", "target_value": "ValidEntity"},
        ]

        compact = format_compact_relations(relations)

        assert "entities" in compact
        assert compact["entities"] == ["ValidEntity"]
