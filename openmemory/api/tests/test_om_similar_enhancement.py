"""
Tests for OM_SIMILAR relations enhancement in MCP tool outputs.

TDD: These tests define the expected behavior for the OM_SIMILAR enhancement.

The enhancement adds:
1. target_value = memory UUID (instead of null)
2. score = similarity score from edge
3. preview = first 80 chars of target memory content
4. Limit to top 5 similar memories per result (sorted by score)
"""

import pytest
from typing import Dict, List
from unittest.mock import MagicMock, patch


class TestCypherQueryOMSimilar:
    """Test the Cypher query correctly handles OM_SIMILAR edges."""

    def test_query_returns_memory_id_for_om_similar(self):
        """OM_SIMILAR edges should return target memory ID as targetValue."""
        from app.graph.metadata_projector import CypherBuilder

        query = CypherBuilder.get_memory_relations_query()

        # Verify the query contains handling for OM_Memory targets
        assert "WHEN target:OM_Memory THEN target.id" in query, \
            "Query should extract memory ID for OM_Memory targets"

    def test_query_returns_similarity_score(self):
        """Query should return similarity score for OM_SIMILAR relations."""
        from app.graph.metadata_projector import CypherBuilder

        query = CypherBuilder.get_memory_relations_query()

        # Verify score is extracted for OM_SIMILAR relations
        assert "similarityScore" in query, \
            "Query should return similarityScore field"
        assert "r.score" in query, \
            "Query should extract score from relation"

    def test_query_returns_preview(self):
        """Query should return content preview for OM_Memory targets."""
        from app.graph.metadata_projector import CypherBuilder

        query = CypherBuilder.get_memory_relations_query()

        # Verify preview is extracted (first 80 chars of content)
        assert "targetPreview" in query, \
            "Query should return targetPreview field"
        assert "target.content" in query or "left(" in query.lower(), \
            "Query should extract content preview"


class TestResponseProcessingOMSimilar:
    """Test response processing includes OM_SIMILAR enhancements."""

    def test_response_includes_score_field(self):
        """Processed relations should include score for OM_SIMILAR."""
        from app.graph.metadata_projector import MetadataProjector

        # Create mock session factory
        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "memoryId": "mem-123",
            "relationType": "OM_SIMILAR",
            "targetLabel": "OM_Memory",
            "targetValue": "similar-456",
            "relationValue": None,
            "similarityScore": 0.8765,
            "targetPreview": "This is a test memory content that should be truncated...",
        }.get(key)

        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result

        def session_factory():
            return MagicMock(__enter__=lambda s: mock_session, __exit__=lambda s, *args: None)

        projector = MetadataProjector(session_factory)
        relations = projector.get_relations_for_memories(["mem-123"])

        assert "mem-123" in relations
        assert len(relations["mem-123"]) == 1
        rel = relations["mem-123"][0]

        assert rel["type"] == "OM_SIMILAR"
        assert rel["target_value"] == "similar-456"
        assert "score" in rel, "OM_SIMILAR should include score"
        assert rel["score"] == 0.88, "Score should be rounded to 2 decimals"

    def test_response_includes_preview_field(self):
        """Processed relations should include preview for OM_SIMILAR."""
        from app.graph.metadata_projector import MetadataProjector

        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "memoryId": "mem-123",
            "relationType": "OM_SIMILAR",
            "targetLabel": "OM_Memory",
            "targetValue": "similar-456",
            "relationValue": None,
            "similarityScore": 0.85,
            "targetPreview": "Cache auth tokens for 5 minutes to reduce API calls.",
        }.get(key)

        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result

        def session_factory():
            return MagicMock(__enter__=lambda s: mock_session, __exit__=lambda s, *args: None)

        projector = MetadataProjector(session_factory)
        relations = projector.get_relations_for_memories(["mem-123"])

        rel = relations["mem-123"][0]
        assert "preview" in rel, "OM_SIMILAR should include preview"
        assert rel["preview"] == "Cache auth tokens for 5 minutes to reduce API calls."

    def test_response_handles_null_score(self):
        """Non-OM_SIMILAR relations should not have score field."""
        from app.graph.metadata_projector import MetadataProjector

        mock_session = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "memoryId": "mem-123",
            "relationType": "OM_IN_CATEGORY",
            "targetLabel": "OM_Category",
            "targetValue": "workflow",
            "relationValue": None,
            "similarityScore": None,
            "targetPreview": None,
        }.get(key)

        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([mock_record])
        mock_session.run.return_value = mock_result

        def session_factory():
            return MagicMock(__enter__=lambda s: mock_session, __exit__=lambda s, *args: None)

        projector = MetadataProjector(session_factory)
        relations = projector.get_relations_for_memories(["mem-123"])

        rel = relations["mem-123"][0]
        assert rel["type"] == "OM_IN_CATEGORY"
        assert "score" not in rel, "Non-OM_SIMILAR should not have score"
        assert "preview" not in rel, "Non-OM_SIMILAR should not have preview"


class TestOMSimilarLimiting:
    """Test OM_SIMILAR relations are limited to top N by score."""

    def test_similar_relations_limited_to_top_5(self):
        """Only top 5 OM_SIMILAR relations should be kept."""
        from app.graph.metadata_projector import _limit_similar_relations

        # Create 10 OM_SIMILAR relations with different scores
        relations = [
            {"type": "OM_SIMILAR", "target_value": f"mem-{i}", "score": 0.5 + i * 0.05}
            for i in range(10)
        ]
        # Add some non-OM_SIMILAR relations
        relations.append({"type": "OM_IN_CATEGORY", "target_value": "workflow"})
        relations.append({"type": "OM_IN_SCOPE", "target_value": "project"})

        limited = _limit_similar_relations(relations)

        # Should have 5 OM_SIMILAR + 2 other = 7 total
        assert len(limited) == 7

        # Count OM_SIMILAR
        similar_count = sum(1 for r in limited if r["type"] == "OM_SIMILAR")
        assert similar_count == 5, "Should limit to 5 OM_SIMILAR"

    def test_similar_relations_sorted_by_score(self):
        """OM_SIMILAR relations should be sorted by score descending."""
        from app.graph.metadata_projector import _limit_similar_relations

        relations = [
            {"type": "OM_SIMILAR", "target_value": "low", "score": 0.3},
            {"type": "OM_SIMILAR", "target_value": "high", "score": 0.9},
            {"type": "OM_SIMILAR", "target_value": "mid", "score": 0.6},
        ]

        limited = _limit_similar_relations(relations)

        similar = [r for r in limited if r["type"] == "OM_SIMILAR"]
        scores = [r["score"] for r in similar]

        assert scores == sorted(scores, reverse=True), \
            "OM_SIMILAR should be sorted by score descending"
        assert similar[0]["target_value"] == "high"
        assert similar[-1]["target_value"] == "low"

    def test_other_relations_not_affected(self):
        """Non-OM_SIMILAR relations should not be filtered or sorted."""
        from app.graph.metadata_projector import _limit_similar_relations

        relations = [
            {"type": "OM_IN_CATEGORY", "target_value": "workflow"},
            {"type": "OM_IN_SCOPE", "target_value": "project"},
            {"type": "OM_HAS_ENTITY", "target_value": "AuthService"},
            {"type": "OM_SIMILAR", "target_value": "sim1", "score": 0.9},
            {"type": "OM_SIMILAR", "target_value": "sim2", "score": 0.8},
        ]

        limited = _limit_similar_relations(relations)

        # All non-OM_SIMILAR should be preserved
        other = [r for r in limited if r["type"] != "OM_SIMILAR"]
        assert len(other) == 3
        assert {"type": "OM_IN_CATEGORY", "target_value": "workflow"} in other
        assert {"type": "OM_IN_SCOPE", "target_value": "project"} in other

    def test_handles_missing_score(self):
        """OM_SIMILAR without score should be treated as score=0."""
        from app.graph.metadata_projector import _limit_similar_relations

        relations = [
            {"type": "OM_SIMILAR", "target_value": "no-score"},  # No score
            {"type": "OM_SIMILAR", "target_value": "with-score", "score": 0.5},
            {"type": "OM_SIMILAR", "target_value": "none-score", "score": None},  # Explicit None
        ]

        limited = _limit_similar_relations(relations)

        # Should not crash, missing scores treated as 0
        similar = [r for r in limited if r["type"] == "OM_SIMILAR"]
        assert similar[0]["target_value"] == "with-score", \
            "Items with score should come first"

    def test_fewer_than_limit_unchanged(self):
        """If fewer than 5 OM_SIMILAR, all should be kept."""
        from app.graph.metadata_projector import _limit_similar_relations

        relations = [
            {"type": "OM_SIMILAR", "target_value": "sim1", "score": 0.9},
            {"type": "OM_SIMILAR", "target_value": "sim2", "score": 0.8},
            {"type": "OM_IN_CATEGORY", "target_value": "workflow"},
        ]

        limited = _limit_similar_relations(relations)

        similar = [r for r in limited if r["type"] == "OM_SIMILAR"]
        assert len(similar) == 2, "Should keep all if fewer than limit"


class TestOMSimilarConstant:
    """Test the OM_SIMILAR_LIMIT constant exists."""

    def test_constant_defined(self):
        """OM_SIMILAR_LIMIT constant should be defined."""
        from app.graph.metadata_projector import OM_SIMILAR_LIMIT

        assert OM_SIMILAR_LIMIT == 5, "Limit should be 5"


class TestIntegrationOMSimilar:
    """Integration tests for OM_SIMILAR in search results."""

    def test_get_relations_applies_limit(self):
        """get_relations_for_memories should apply OM_SIMILAR limit."""
        from app.graph.metadata_projector import MetadataProjector

        # Create mock with 10 OM_SIMILAR records
        mock_session = MagicMock()
        mock_records = []
        for i in range(10):
            record = MagicMock()
            record.__getitem__ = lambda self, key, i=i: {
                "memoryId": "mem-123",
                "relationType": "OM_SIMILAR",
                "targetLabel": "OM_Memory",
                "targetValue": f"similar-{i}",
                "relationValue": None,
                "similarityScore": 0.5 + i * 0.05,
                "targetPreview": f"Preview {i}",
            }.get(key)
            mock_records.append(record)

        # Add one non-OM_SIMILAR record
        category_record = MagicMock()
        category_record.__getitem__ = lambda self, key: {
            "memoryId": "mem-123",
            "relationType": "OM_IN_CATEGORY",
            "targetLabel": "OM_Category",
            "targetValue": "workflow",
            "relationValue": None,
            "similarityScore": None,
            "targetPreview": None,
        }.get(key)
        mock_records.append(category_record)

        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter(mock_records)
        mock_session.run.return_value = mock_result

        def session_factory():
            return MagicMock(__enter__=lambda s: mock_session, __exit__=lambda s, *args: None)

        projector = MetadataProjector(session_factory)
        relations = projector.get_relations_for_memories(["mem-123"])

        # Should have 5 OM_SIMILAR + 1 OM_IN_CATEGORY = 6 total
        assert len(relations["mem-123"]) == 6

        similar = [r for r in relations["mem-123"] if r["type"] == "OM_SIMILAR"]
        assert len(similar) == 5, "Should limit OM_SIMILAR to 5"

        # Verify sorted by score (highest first)
        scores = [r["score"] for r in similar]
        assert scores == sorted(scores, reverse=True)
