"""
Unit tests for the Business Concepts system.

Tests cover:
- ConceptProjector: Node creation, relationships, constraints
- ConceptCypherBuilder: Query generation
- ConceptOps: CRUD operations (mocked)
- ContradictionDetector: Pattern matching, detection logic
- Config: Feature flags

These tests do not require a live Neo4j instance.

Run with: pytest openmemory/api/tests/test_business_concepts.py -v
"""

import pytest
import json
import os
from unittest.mock import MagicMock, patch, call
from datetime import datetime

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import BusinessConceptsConfig
from app.graph.concept_projector import (
    ConceptCypherBuilder,
    ConceptProjector,
    CONCEPT_TYPES,
    ENTITY_TYPES,
    SOURCE_TYPES,
)
from app.graph.convergence_detector import (
    ConvergenceEvidence,
    ConvergenceResult,
    ContradictionResult,
    ContradictionDetector,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock Neo4j session."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=None)
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    """Create a mock session factory that returns the mock session."""
    def factory():
        return mock_session
    return factory


@pytest.fixture
def projector(mock_session_factory):
    """Create a ConceptProjector with mocked session."""
    return ConceptProjector(session_factory=mock_session_factory)


@pytest.fixture
def sample_concept():
    """Sample concept data."""
    return {
        "id": "concept-123",
        "user_id": "test_user",
        "name": "Revenue is growing fast",
        "type": "trend",
        "confidence": 0.8,
        "category": "decision",
        "summary": "Company revenue doubled in Q4",
        "source_type": "stated_fact",
        "evidence_count": 3,
    }


@pytest.fixture
def sample_entity():
    """Sample business entity data."""
    return {
        "id": "entity-456",
        "user_id": "test_user",
        "name": "Stripe",
        "type": "company",
        "importance": 0.9,
        "context": "Payment processing company",
        "mention_count": 5,
    }


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestBusinessConceptsConfig:
    """Tests for BusinessConceptsConfig."""

    def test_is_enabled_default_false(self):
        """Default value should be false."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing env vars
            os.environ.pop("BUSINESS_CONCEPTS_ENABLED", None)
            assert BusinessConceptsConfig.is_enabled() is False

    def test_is_enabled_true(self):
        """Should return True when env var is set."""
        with patch.dict(os.environ, {"BUSINESS_CONCEPTS_ENABLED": "true"}):
            assert BusinessConceptsConfig.is_enabled() is True

    def test_is_auto_extract_requires_enabled(self):
        """Auto extract requires concepts to be enabled."""
        with patch.dict(os.environ, {
            "BUSINESS_CONCEPTS_ENABLED": "false",
            "BUSINESS_CONCEPTS_AUTO_EXTRACT": "true"
        }):
            assert BusinessConceptsConfig.is_auto_extract_enabled() is False

        with patch.dict(os.environ, {
            "BUSINESS_CONCEPTS_ENABLED": "true",
            "BUSINESS_CONCEPTS_AUTO_EXTRACT": "true"
        }):
            assert BusinessConceptsConfig.is_auto_extract_enabled() is True

    def test_get_min_confidence_default(self):
        """Default min confidence should be 0.5."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BUSINESS_CONCEPTS_MIN_CONFIDENCE", None)
            assert BusinessConceptsConfig.get_min_confidence() == 0.5

    def test_get_min_confidence_custom(self):
        """Should parse custom min confidence."""
        with patch.dict(os.environ, {"BUSINESS_CONCEPTS_MIN_CONFIDENCE": "0.7"}):
            assert BusinessConceptsConfig.get_min_confidence() == 0.7

    def test_get_min_confidence_invalid(self):
        """Should return default on invalid value."""
        with patch.dict(os.environ, {"BUSINESS_CONCEPTS_MIN_CONFIDENCE": "not_a_number"}):
            assert BusinessConceptsConfig.get_min_confidence() == 0.5


# =============================================================================
# CYPHER BUILDER TESTS
# =============================================================================


class TestConceptCypherBuilder:
    """Tests for ConceptCypherBuilder query generation."""

    def test_constraint_queries(self):
        """Should generate valid constraint queries."""
        queries = ConceptCypherBuilder.constraint_queries()

        assert len(queries) > 0

        # Check for key constraints
        assert any("om_concept_id" in q.lower() for q in queries)
        assert any("om_bizentity_id" in q.lower() for q in queries)

        # All should be CREATE queries
        for q in queries:
            assert q.startswith("CREATE")

    def test_fulltext_index_queries(self):
        """Should generate fulltext index queries."""
        queries = ConceptCypherBuilder.fulltext_index_queries()

        assert len(queries) >= 2
        assert any("fulltext" in q.lower() for q in queries)

    def test_upsert_concept_query(self):
        """Should generate valid concept upsert query."""
        query = ConceptCypherBuilder.upsert_concept_query()

        assert "MERGE" in query
        assert "OM_Concept" in query
        assert "ON CREATE SET" in query
        assert "ON MATCH SET" in query
        assert "$userId" in query
        assert "$name" in query

    def test_upsert_bizentity_query(self):
        """Should generate valid entity upsert query."""
        query = ConceptCypherBuilder.upsert_bizentity_query()

        assert "MERGE" in query
        assert "OM_BizEntity" in query
        assert "ON CREATE SET" in query
        assert "$type" in query
        assert "$importance" in query

    def test_link_memory_to_concept_query(self):
        """Should generate valid memory-concept link query."""
        query = ConceptCypherBuilder.link_memory_to_concept_query()

        assert "MATCH" in query
        assert "OM_Memory" in query
        assert "OM_Concept" in query
        assert "MERGE" in query
        assert "SUPPORTS" in query

    def test_create_contradiction_query(self):
        """Should generate valid contradiction query."""
        query = ConceptCypherBuilder.create_contradiction_query()

        assert "CONTRADICTS" in query
        assert "$severity" in query
        assert "$evidence" in query
        assert "resolved = false" in query


# =============================================================================
# CONCEPT PROJECTOR TESTS
# =============================================================================


class TestConceptProjector:
    """Tests for ConceptProjector operations."""

    def test_upsert_concept_valid_type(self, projector, mock_session, sample_concept):
        """Should upsert concept with valid type."""
        # Setup mock response
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _: iter([{"id": "concept-123", "name": "Test"}])
        mock_session.run.return_value = mock_result

        result = projector.upsert_concept(
            user_id=sample_concept["user_id"],
            name=sample_concept["name"],
            concept_type=sample_concept["type"],
            confidence=sample_concept["confidence"],
            category=sample_concept["category"],
        )

        assert result is not None
        assert result["id"] == "concept-123"
        mock_session.run.assert_called_once()

    def test_upsert_concept_invalid_type(self, projector, mock_session):
        """Should reject invalid concept type."""
        result = projector.upsert_concept(
            user_id="test",
            name="Test Concept",
            concept_type="invalid_type",  # Not in CONCEPT_TYPES
            confidence=0.5,
        )

        assert result is None
        mock_session.run.assert_not_called()

    def test_upsert_bizentity_valid_type(self, projector, mock_session, sample_entity):
        """Should upsert entity with valid type."""
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _: iter([{"id": "entity-456", "name": "Stripe"}])
        mock_session.run.return_value = mock_result

        result = projector.upsert_bizentity(
            user_id=sample_entity["user_id"],
            name=sample_entity["name"],
            entity_type=sample_entity["type"],
            importance=sample_entity["importance"],
        )

        assert result is not None
        assert result["name"] == "Stripe"

    def test_upsert_bizentity_invalid_type(self, projector, mock_session):
        """Should reject invalid entity type."""
        result = projector.upsert_bizentity(
            user_id="test",
            name="Invalid Entity",
            entity_type="not_a_valid_type",
            importance=0.5,
        )

        assert result is None

    def test_link_memory_to_concept(self, projector, mock_session):
        """Should create SUPPORTS relationship."""
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _: iter([{"relType": "SUPPORTS"}])
        mock_session.run.return_value = mock_result

        result = projector.link_memory_to_concept(
            memory_id="mem-123",
            user_id="test",
            concept_name="Test Concept",
            confidence=0.7,
        )

        assert result is True

    def test_create_contradiction(self, projector, mock_session):
        """Should create CONTRADICTS relationship."""
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _: iter([{"relType": "CONTRADICTS"}])
        mock_session.run.return_value = mock_result

        result = projector.create_contradiction(
            user_id="test",
            concept_name1="Revenue growing",
            concept_name2="Revenue shrinking",
            severity=0.8,
            evidence=["Quote 1", "Quote 2"],
        )

        assert result is True

    def test_confidence_clamped(self, projector, mock_session):
        """Should clamp confidence to 0.0-1.0 range."""
        mock_result = MagicMock()
        mock_result.__iter__ = lambda _: iter([{"id": "test", "name": "Test"}])
        mock_session.run.return_value = mock_result

        # Test with value > 1.0
        projector.upsert_concept(
            user_id="test",
            name="Test",
            concept_type="fact",
            confidence=1.5,  # Should be clamped to 1.0
        )

        # Check the parameters passed to run
        call_args = mock_session.run.call_args
        params = call_args[0][1]
        assert params["confidence"] == 1.0


# =============================================================================
# CONTRADICTION DETECTOR TESTS
# =============================================================================


class TestContradictionPatterns:
    """Tests for contradiction pattern matching."""

    @pytest.fixture
    def detector(self):
        return ContradictionDetector(user_id="test")

    def test_check_opposition_growing_shrinking(self, detector):
        """Should detect growing vs shrinking contradiction."""
        result = detector._check_opposition_patterns(
            "Revenue is growing fast",
            "Revenue is shrinking significantly"
        )
        assert result == "contradicts"

    def test_check_opposition_increasing_decreasing(self, detector):
        """Should detect increasing vs decreasing contradiction."""
        result = detector._check_opposition_patterns(
            "Costs are increasing",
            "Costs are decreasing"
        )
        assert result == "contradicts"

    def test_check_opposition_success_failure(self, detector):
        """Should detect success vs failure contradiction."""
        result = detector._check_opposition_patterns(
            "Project is a success",
            "Project is a failure"
        )
        assert result == "contradicts"

    def test_check_opposition_german(self, detector):
        """Should detect German language contradictions."""
        result = detector._check_opposition_patterns(
            "Umsatz steigend",
            "Umsatz sinkend"
        )
        assert result == "contradicts"

    def test_check_opposition_no_match(self, detector):
        """Should return None when no contradiction."""
        result = detector._check_opposition_patterns(
            "Revenue is growing",
            "Customers are happy"
        )
        assert result is None

    def test_check_opposition_negation(self, detector):
        """Should detect negation-based conflicts."""
        result = detector._check_opposition_patterns(
            "Product is ready for launch",
            "Product is not ready for market"
        )
        assert result == "partially_conflicts"


class TestContradictionResult:
    """Tests for ContradictionResult dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary correctly."""
        result = ContradictionResult(
            concept_a="Revenue growing",
            concept_b="Revenue shrinking",
            similarity_score=0.75,
            semantic_relation="contradicts",
            severity=0.85,
            evidence_a=["Quote 1", "Quote 2", "Quote 3", "Quote 4"],
            evidence_b=["Quote A", "Quote B"],
            detection_method="semantic_opposition",
        )

        d = result.to_dict()

        assert d["concept_a"] == "Revenue growing"
        assert d["concept_b"] == "Revenue shrinking"
        assert d["severity"] == 0.85
        assert d["detection_method"] == "semantic_opposition"
        # Evidence should be limited to 3 items
        assert len(d["evidence_a"]) <= 3


# =============================================================================
# CONVERGENCE RESULT TESTS
# =============================================================================


class TestConvergenceResult:
    """Tests for ConvergenceResult."""

    def test_is_strong_convergence_true(self):
        """Should return True for strong convergence."""
        evidence = [
            ConvergenceEvidence(
                memory_id="m1", content="test", category="decision",
                created_at=datetime(2024, 1, 1), entities=[]
            ),
            ConvergenceEvidence(
                memory_id="m2", content="test", category="architecture",
                created_at=datetime(2024, 1, 15), entities=[]
            ),
            ConvergenceEvidence(
                memory_id="m3", content="test", category="workflow",
                created_at=datetime(2024, 1, 20), entities=[]
            ),
        ]

        result = ConvergenceResult(
            concept_name="Test Concept",
            evidence=evidence,
            convergence_score=0.75,
            temporal_spread_days=20,
            category_diversity=0.67,
            source_diversity=0.5,
            entity_path_diversity=0.4,
            recommended_confidence=0.85,
        )

        assert result.is_strong_convergence() is True

    def test_is_strong_convergence_false_low_score(self):
        """Should return False for low convergence score."""
        evidence = [
            ConvergenceEvidence(
                memory_id="m1", content="test", category="decision",
                created_at=datetime(2024, 1, 1), entities=[]
            ),
            ConvergenceEvidence(
                memory_id="m2", content="test", category="architecture",
                created_at=datetime(2024, 1, 15), entities=[]
            ),
            ConvergenceEvidence(
                memory_id="m3", content="test", category="workflow",
                created_at=datetime(2024, 1, 20), entities=[]
            ),
        ]

        result = ConvergenceResult(
            concept_name="Test Concept",
            evidence=evidence,
            convergence_score=0.5,  # Below 0.7 threshold
            temporal_spread_days=20,
            category_diversity=0.67,
            source_diversity=0.5,
            entity_path_diversity=0.4,
            recommended_confidence=0.7,
        )

        assert result.is_strong_convergence() is False

    def test_to_dict(self):
        """Should serialize correctly."""
        result = ConvergenceResult(
            concept_name="Test Concept",
            evidence=[],
            convergence_score=0.756,
            temporal_spread_days=20,
            category_diversity=0.678,
            source_diversity=0.512,
            entity_path_diversity=0.433,
            recommended_confidence=0.854,
        )

        d = result.to_dict()

        # Check rounding
        assert d["convergence_score"] == 0.756
        assert d["category_diversity"] == 0.678
        assert d["recommended_confidence"] == 0.854


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_concept_types(self):
        """Should have all required concept types."""
        expected = {"causal", "pattern", "comparison", "trend",
                    "contradiction", "hypothesis", "fact"}
        assert CONCEPT_TYPES == expected

    def test_entity_types(self):
        """Should have all required entity types."""
        expected = {"company", "person", "product", "market",
                    "metric", "business_model", "technology", "strategy"}
        assert ENTITY_TYPES == expected

    def test_source_types(self):
        """Should have all required source types."""
        expected = {"stated_fact", "inference", "opinion"}
        assert SOURCE_TYPES == expected


# =============================================================================
# INTEGRATION HELPER TESTS
# =============================================================================


class TestExtractFromMemory:
    """Tests for extract_from_memory helper function."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True)
    def test_extract_without_api_key(self):
        """Should return error when API key not configured."""
        from app.utils.concept_extractor import extract_from_memory

        result = extract_from_memory(
            memory_id="test-123",
            user_id="test",
            content="Test content",
        )

        assert "error" in result
        assert "API key" in result["error"]

    @patch("app.utils.concept_extractor.ConceptExtractor")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_extract_returns_results(self, mock_extractor_class):
        """Should return extraction results."""
        from app.utils.concept_extractor import extract_from_memory, TranscriptExtraction

        # Mock the extractor
        mock_extractor = MagicMock()
        mock_extractor.extract_full.return_value = TranscriptExtraction(
            entities=[],
            concepts=[],
            summary="Test summary",
            language="en"
        )
        mock_extractor_class.return_value = mock_extractor

        result = extract_from_memory(
            memory_id="test-123",
            user_id="test",
            content="Test content",
            store_in_graph=False,  # Skip graph storage
        )

        assert "error" not in result or result.get("error") is None
        assert result.get("summary") == "Test summary"
        assert result.get("language") == "en"
