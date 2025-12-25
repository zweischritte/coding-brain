"""
Unit tests for Query Router module.

Tests cover:
- Route type determination based on entity count and keywords
- Relationship keyword detection
- RRF alpha selection for routes
- Configuration handling
"""

import pytest
from app.utils.query_router import (
    RouteType,
    RoutingConfig,
    QueryAnalysis,
    determine_route,
    detect_relationship_keywords,
    get_rrf_alpha_for_route,
)


class TestRouteTypeDetermination:
    """Test the routing decision matrix."""

    def test_no_entities_no_keywords_is_vector_only(self):
        """0 entities, no keywords -> VECTOR_ONLY"""
        route, confidence = determine_route(
            entity_count=0,
            has_relationship_keywords=False
        )
        assert route == RouteType.VECTOR_ONLY
        assert confidence == 0.9

    def test_no_entities_with_keywords_is_hybrid(self):
        """0 entities, keywords -> HYBRID (might be generic relationship query)"""
        route, confidence = determine_route(
            entity_count=0,
            has_relationship_keywords=True
        )
        assert route == RouteType.HYBRID
        assert confidence == 0.6

    def test_one_entity_no_keywords_is_hybrid(self):
        """1 entity, no keywords -> HYBRID"""
        route, confidence = determine_route(
            entity_count=1,
            has_relationship_keywords=False
        )
        assert route == RouteType.HYBRID
        assert confidence == 0.8

    def test_one_entity_with_keywords_is_graph_primary(self):
        """1 entity, keywords -> GRAPH_PRIMARY"""
        route, confidence = determine_route(
            entity_count=1,
            has_relationship_keywords=True
        )
        assert route == RouteType.GRAPH_PRIMARY
        assert confidence == 0.85

    def test_multiple_entities_is_graph_primary(self):
        """2+ entities -> GRAPH_PRIMARY regardless of keywords"""
        route, confidence = determine_route(
            entity_count=2,
            has_relationship_keywords=False
        )
        assert route == RouteType.GRAPH_PRIMARY
        assert confidence == 0.9

        route2, confidence2 = determine_route(
            entity_count=5,
            has_relationship_keywords=True
        )
        assert route2 == RouteType.GRAPH_PRIMARY
        assert confidence2 == 0.9


class TestRelationshipKeywordDetection:
    """Test detection of relationship-indicating keywords."""

    def test_english_relationship_keywords(self):
        """Test detection of English relationship keywords."""
        config = RoutingConfig()

        # Test various English patterns
        assert len(detect_relationship_keywords("connected to Bob", config)) > 0
        assert len(detect_relationship_keywords("related to the project", config)) > 0
        assert len(detect_relationship_keywords("between Julia and Mike", config)) > 0
        assert len(detect_relationship_keywords("find the path", config)) > 0
        assert len(detect_relationship_keywords("show me the network", config)) > 0
        assert len(detect_relationship_keywords("what is their relationship", config)) > 0
        assert len(detect_relationship_keywords("who knows about this", config)) > 0

    def test_german_relationship_keywords(self):
        """Test detection of German relationship keywords."""
        config = RoutingConfig()

        assert len(detect_relationship_keywords("verbunden mit Julia", config)) > 0
        assert len(detect_relationship_keywords("die Beziehung zu Bob", config)) > 0
        assert len(detect_relationship_keywords("zwischen den Projekten", config)) > 0
        assert len(detect_relationship_keywords("wer kennt das Thema", config)) > 0

    def test_no_keywords_found(self):
        """Test queries without relationship keywords."""
        config = RoutingConfig()

        assert len(detect_relationship_keywords("what are my goals", config)) == 0
        assert len(detect_relationship_keywords("find memories about work", config)) == 0
        assert len(detect_relationship_keywords("list all projects", config)) == 0

    def test_case_insensitive(self):
        """Test that keyword detection is case insensitive."""
        config = RoutingConfig()

        assert len(detect_relationship_keywords("CONNECTED TO Bob", config)) > 0
        assert len(detect_relationship_keywords("Connected To Bob", config)) > 0
        assert len(detect_relationship_keywords("BETWEEN Julia AND Mike", config)) > 0

    def test_multiple_keywords(self):
        """Test queries with multiple relationship keywords."""
        config = RoutingConfig()

        keywords = detect_relationship_keywords(
            "how is Julia connected to Bob and what is their relationship", config
        )
        # Should find multiple patterns
        assert len(keywords) >= 2


class TestRRFAlphaForRoute:
    """Test RRF alpha value selection for routes."""

    def test_vector_only_alpha(self):
        """VECTOR_ONLY should return alpha=1.0 (pure vector)"""
        alpha = get_rrf_alpha_for_route(RouteType.VECTOR_ONLY)
        assert alpha == 1.0

    def test_hybrid_alpha(self):
        """HYBRID should return alpha=0.6 (slight vector preference)"""
        alpha = get_rrf_alpha_for_route(RouteType.HYBRID)
        assert alpha == 0.6

    def test_graph_primary_alpha(self):
        """GRAPH_PRIMARY should return alpha=0.4 (graph preference)"""
        alpha = get_rrf_alpha_for_route(RouteType.GRAPH_PRIMARY)
        assert alpha == 0.4


class TestRoutingConfig:
    """Test routing configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RoutingConfig()

        assert config.enabled == True
        assert config.min_entity_score == 0.7
        assert config.fallback_min_results == 3
        assert len(config.relationship_keywords) > 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RoutingConfig(
            enabled=False,
            min_entity_score=0.5,
            fallback_min_results=5,
        )

        assert config.enabled == False
        assert config.min_entity_score == 0.5
        assert config.fallback_min_results == 5


class TestQueryAnalysis:
    """Test QueryAnalysis dataclass."""

    def test_query_analysis_creation(self):
        """Test creating a QueryAnalysis instance."""
        analysis = QueryAnalysis(
            detected_entities=[("julia", 0.95), ("bob", 0.88)],
            relationship_keywords=["connected to"],
            route=RouteType.GRAPH_PRIMARY,
            confidence=0.9,
            analysis_time_ms=15.5,
        )

        assert len(analysis.detected_entities) == 2
        assert analysis.detected_entities[0] == ("julia", 0.95)
        assert analysis.route == RouteType.GRAPH_PRIMARY
        assert analysis.confidence == 0.9
        assert analysis.analysis_time_ms == 15.5

    def test_query_analysis_empty_entities(self):
        """Test QueryAnalysis with no entities."""
        analysis = QueryAnalysis(
            detected_entities=[],
            relationship_keywords=[],
            route=RouteType.VECTOR_ONLY,
            confidence=0.9,
        )

        assert len(analysis.detected_entities) == 0
        assert analysis.route == RouteType.VECTOR_ONLY


class TestRouteTypeEnum:
    """Test RouteType enum values."""

    def test_route_type_values(self):
        """Test that route types have correct string values."""
        assert RouteType.VECTOR_ONLY.value == "vector"
        assert RouteType.HYBRID.value == "hybrid"
        assert RouteType.GRAPH_PRIMARY.value == "graph"

    def test_route_type_from_string(self):
        """Test creating RouteType from string value."""
        assert RouteType("vector") == RouteType.VECTOR_ONLY
        assert RouteType("hybrid") == RouteType.HYBRID
        assert RouteType("graph") == RouteType.GRAPH_PRIMARY
