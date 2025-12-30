"""
Tests for REST Search API with Embedding Support (TDD).

These tests define the expected behavior for the search endpoint's embedding
integration, including:
- mode parameter (auto, lexical, semantic)
- Embedding generation via memory client
- Fallback to lexical on embedding failure
- Response meta field with degraded_mode indicator

Refactored to test router functions and models directly without requiring
TestClient or database connections.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

# Import from the router directly - no need for main.py
from app.routers.search import (
    SearchRequest,
    SearchResponse,
    SearchMode,
    SearchMeta,
    SearchFilters,
    SearchResult,
    SemanticSearchRequest,
    hybrid_search,
    lexical_search,
    semantic_search,
    _filters_to_dict,
    _access_entity_filters,
    _format_results,
)
from app.security.types import Principal, TokenClaims, Scope


# Test embedding dimensions
TEST_EMBEDDING_DIM = 1536


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_vector():
    """Create a mock embedding vector."""
    return [0.1] * TEST_EMBEDDING_DIM


@pytest.fixture
def mock_principal():
    """Create a mock Principal for testing."""
    claims = TokenClaims(
        sub="test-user-123",
        iss="https://auth.test.example.com",
        aud="https://api.test.example.com",
        exp=datetime(2099, 1, 1, tzinfo=timezone.utc),
        iat=datetime.now(timezone.utc),
        jti="test-jti-123",
        org_id="test-org-456",
        scopes={"search:read"},
        grants={"user:test-user-123"},
    )
    return Principal(
        user_id="test-user-123",
        org_id="test-org-456",
        claims=claims,
    )


@pytest.fixture
def mock_principal_with_grants():
    """Create a mock Principal with team grants."""
    claims = TokenClaims(
        sub="grischa",
        iss="https://auth.test.example.com",
        aud="https://api.test.example.com",
        exp=datetime(2099, 1, 1, tzinfo=timezone.utc),
        iat=datetime.now(timezone.utc),
        jti="test-jti-456",
        org_id="test-org",
        scopes={"search:read"},
        grants={"user:grischa", "team:test-org/backend"},
    )
    return Principal(
        user_id="grischa",
        org_id="test-org",
        claims=claims,
    )


@pytest.fixture
def mock_memory_client(mock_embedding_vector):
    """
    Create a mock memory client with embedding_model.

    The memory client is obtained via get_memory_client() and has an
    embedding_model attribute with an embed() method.
    """
    mock_client = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed.return_value = mock_embedding_vector
    mock_client.embedding_model = mock_embedding_model
    return mock_client


@pytest.fixture
def mock_memory_client_with_failure():
    """
    Create a mock memory client where embedding fails.
    """
    mock_client = MagicMock()
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed.side_effect = Exception("Embedding service unavailable")
    mock_client.embedding_model = mock_embedding_model
    return mock_client


@pytest.fixture
def mock_opensearch_store():
    """Create a mock OpenSearch store for search operations."""
    mock_store = MagicMock()

    # Default return empty results
    mock_store.search_with_access_control.return_value = []
    mock_store.hybrid_search_with_access_control.return_value = []

    return mock_store


@pytest.fixture
def mock_opensearch_store_with_results():
    """Create a mock OpenSearch store that returns sample results."""
    mock_store = MagicMock()

    sample_results = [
        {
            "_id": "mem-123",
            "_score": 0.95,
            "_source": {
                "content": "Test memory content about Python",
                "category": "workflow",
                "scope": "project",
                "access_entity": "user:test-user-123",
            },
            "highlight": {"content": ["<em>Python</em> programming"]},
        },
        {
            "_id": "mem-456",
            "_score": 0.85,
            "_source": {
                "content": "Another test memory",
                "category": "decision",
                "scope": "team",
                "access_entity": "team:test-org/backend",
            },
            "highlight": {},
        },
    ]

    mock_store.search_with_access_control.return_value = sample_results
    mock_store.hybrid_search_with_access_control.return_value = sample_results

    return mock_store


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return MagicMock()


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestSearchModeEnum:
    """Tests for SearchMode enum values."""

    def test_search_mode_values(self):
        """SearchMode should have auto, lexical, semantic values."""
        assert SearchMode.auto == "auto"
        assert SearchMode.lexical == "lexical"
        assert SearchMode.semantic == "semantic"

    def test_search_mode_is_string_enum(self):
        """SearchMode values should be strings."""
        assert isinstance(SearchMode.auto.value, str)
        assert isinstance(SearchMode.lexical.value, str)
        assert isinstance(SearchMode.semantic.value, str)


class TestSearchMetaModel:
    """Tests for SearchMeta Pydantic model."""

    def test_search_meta_defaults(self):
        """SearchMeta should have correct defaults."""
        meta = SearchMeta()
        assert meta.degraded_mode is False
        assert meta.missing_sources == []

    def test_search_meta_with_values(self):
        """SearchMeta should accept custom values."""
        meta = SearchMeta(degraded_mode=True, missing_sources=["embedding"])
        assert meta.degraded_mode is True
        assert meta.missing_sources == ["embedding"]


class TestSearchRequestModel:
    """Tests for SearchRequest Pydantic model."""

    def test_search_request_query_required(self):
        """query field should be required."""
        with pytest.raises(Exception):  # ValidationError
            SearchRequest()

    def test_search_request_mode_defaults_to_auto(self):
        """mode field should default to 'auto'."""
        request = SearchRequest(query="test")
        assert request.mode == SearchMode.auto

    def test_search_request_accepts_all_modes(self):
        """mode field should accept all SearchMode values."""
        for mode in [SearchMode.auto, SearchMode.lexical, SearchMode.semantic]:
            request = SearchRequest(query="test", mode=mode)
            assert request.mode == mode

    def test_search_request_mode_from_string(self):
        """mode field should accept string values."""
        request = SearchRequest(query="test", mode="lexical")
        assert request.mode == SearchMode.lexical

    def test_search_request_rejects_invalid_mode(self):
        """mode field should reject invalid values."""
        with pytest.raises(Exception):  # ValidationError
            SearchRequest(query="test", mode="invalid_mode")

    def test_search_request_limit_defaults_to_10(self):
        """limit should default to 10."""
        request = SearchRequest(query="test")
        assert request.limit == 10

    def test_search_request_limit_validation(self):
        """limit should be between 1 and 100."""
        # Valid limits
        SearchRequest(query="test", limit=1)
        SearchRequest(query="test", limit=100)

        # Invalid limits
        with pytest.raises(Exception):
            SearchRequest(query="test", limit=0)
        with pytest.raises(Exception):
            SearchRequest(query="test", limit=101)


class TestSearchResponseModel:
    """Tests for SearchResponse Pydantic model."""

    def test_search_response_structure(self):
        """SearchResponse should have all required fields."""
        response = SearchResponse(
            results=[],
            total_count=0,
            took_ms=10,
        )
        assert response.results == []
        assert response.total_count == 0
        assert response.took_ms == 10
        assert isinstance(response.meta, SearchMeta)

    def test_search_response_meta_defaults(self):
        """SearchResponse meta should default to non-degraded."""
        response = SearchResponse(results=[], total_count=0, took_ms=10)
        assert response.meta.degraded_mode is False
        assert response.meta.missing_sources == []

    def test_search_response_with_results(self):
        """SearchResponse should hold SearchResult objects."""
        result = SearchResult(
            memory_id="mem-123",
            score=0.95,
            content="Test content",
            highlights=["<em>test</em>"],
            metadata={"category": "workflow"},
        )
        response = SearchResponse(
            results=[result],
            total_count=1,
            took_ms=15,
        )
        assert len(response.results) == 1
        assert response.results[0].memory_id == "mem-123"


class TestSemanticSearchRequestModel:
    """Tests for SemanticSearchRequest Pydantic model."""

    def test_semantic_search_request_requires_query_vector(self):
        """query_vector field should be required."""
        with pytest.raises(Exception):  # ValidationError
            SemanticSearchRequest(query="test")

    def test_semantic_search_request_with_vector(self):
        """SemanticSearchRequest should accept query_vector."""
        vector = [0.1] * 768
        request = SemanticSearchRequest(query="test", query_vector=vector)
        assert len(request.query_vector) == 768


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFiltersToDictHelper:
    """Tests for _filters_to_dict helper function."""

    def test_filters_to_dict_none(self):
        """Should return None for None input."""
        assert _filters_to_dict(None) is None

    def test_filters_to_dict_empty(self):
        """Should return None for empty filters."""
        filters = SearchFilters()
        assert _filters_to_dict(filters) is None

    def test_filters_to_dict_with_values(self):
        """Should convert filters to dict."""
        filters = SearchFilters(category="workflow", scope="project")
        result = _filters_to_dict(filters)
        assert result == {"category": "workflow", "scope": "project"}


class TestAccessEntityFiltersHelper:
    """Tests for _access_entity_filters helper function."""

    def test_access_entity_filters_basic(self, mock_principal):
        """Should build access entity patterns from principal."""
        exact, prefixes = _access_entity_filters(mock_principal)
        assert "user:test-user-123" in exact


class TestFormatResultsHelper:
    """Tests for _format_results helper function."""

    def test_format_results_empty(self):
        """Should handle empty results."""
        assert _format_results([]) == []

    def test_format_results_with_hits(self):
        """Should format OpenSearch hits."""
        hits = [
            {
                "_id": "mem-123",
                "_score": 0.95,
                "_source": {
                    "content": "Test content",
                    "category": "workflow",
                },
                "highlight": {"content": ["<em>Test</em>"]},
            }
        ]
        results = _format_results(hits)
        assert len(results) == 1
        assert results[0].memory_id == "mem-123"
        assert results[0].score == 0.95
        assert results[0].content == "Test content"
        assert results[0].highlights == ["<em>Test</em>"]
        assert results[0].metadata["category"] == "workflow"


# =============================================================================
# Hybrid Search Endpoint Tests
# =============================================================================


class TestHybridSearchModeAuto:
    """Tests for hybrid_search with mode=auto."""

    @pytest.mark.asyncio
    async def test_mode_auto_with_embedding_success(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_embedding_vector,
        mock_db,
    ):
        """
        mode=auto should:
        1. Generate embedding via memory_client.embedding_model.embed()
        2. Use hybrid_search_with_access_control with the embedding
        3. Return results without degraded_mode
        """
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should have results
            assert len(response.results) == 2

            # degraded_mode should be False (embedding succeeded)
            assert response.meta.degraded_mode is False
            assert response.meta.missing_sources == []

            # Verify embedding was called with the query
            mock_memory_client.embedding_model.embed.assert_called_once()
            call_args = mock_memory_client.embedding_model.embed.call_args
            assert call_args[0][0] == "test query"

            # Verify hybrid_search was used (not lexical-only)
            mock_opensearch_store_with_results.hybrid_search_with_access_control.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_auto_passes_embedding_to_hybrid_search(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store,
        mock_embedding_vector,
        mock_db,
    ):
        """mode=auto should pass the generated embedding to hybrid search."""
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store):

            await hybrid_search(request, mock_principal, mock_db)

            # Verify hybrid_search received the embedding vector
            call_kwargs = mock_opensearch_store.hybrid_search_with_access_control.call_args[1]
            assert "query_vector" in call_kwargs
            assert call_kwargs["query_vector"] == mock_embedding_vector


class TestHybridSearchModeAutoWithEmbeddingFailure:
    """Tests for mode=auto when embedding generation fails."""

    @pytest.mark.asyncio
    async def test_mode_auto_with_embedding_failure(
        self,
        mock_principal,
        mock_memory_client_with_failure,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """
        mode=auto with embedding failure should:
        1. Attempt to generate embedding
        2. Fall back to lexical-only search
        3. Return results with meta.degraded_mode=true
        """
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client_with_failure), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should have results (from lexical fallback)
            assert len(response.results) > 0

            # degraded_mode should be True (embedding failed)
            assert response.meta.degraded_mode is True

            # missing_sources should indicate what failed
            assert "embedding" in response.meta.missing_sources

            # Verify lexical search was used (not hybrid)
            mock_opensearch_store_with_results.search_with_access_control.assert_called_once()
            # hybrid_search should NOT have been called
            mock_opensearch_store_with_results.hybrid_search_with_access_control.assert_not_called()

    @pytest.mark.asyncio
    async def test_mode_auto_with_null_memory_client(
        self,
        mock_principal,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """mode=auto should fall back gracefully when memory client is None."""
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=None), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should indicate degraded mode
            assert response.meta.degraded_mode is True
            assert "embedding" in response.meta.missing_sources


class TestHybridSearchModeLexical:
    """Tests for mode=lexical which skips embedding."""

    @pytest.mark.asyncio
    async def test_mode_lexical_skips_embedding(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """
        mode=lexical should:
        1. NOT call the embedding model
        2. Use search_with_access_control (lexical only)
        3. Return results without degraded_mode
        """
        request = SearchRequest(query="test query", mode=SearchMode.lexical)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should have results
            assert len(response.results) == 2

            # degraded_mode should be False (lexical is expected behavior)
            assert response.meta.degraded_mode is False

            # Embedding should NOT have been called
            mock_memory_client.embedding_model.embed.assert_not_called()

            # Lexical search should have been used
            mock_opensearch_store_with_results.search_with_access_control.assert_called_once()

            # Hybrid search should NOT have been called
            mock_opensearch_store_with_results.hybrid_search_with_access_control.assert_not_called()


class TestHybridSearchModeSemantic:
    """Tests for mode=semantic which requires embedding."""

    @pytest.mark.asyncio
    async def test_mode_semantic_unavailable_raises_503(
        self,
        mock_principal,
        mock_memory_client_with_failure,
        mock_opensearch_store,
        mock_db,
    ):
        """
        mode=semantic with unavailable embedding should:
        1. Attempt to generate embedding
        2. Fail and raise HTTPException 503 (not fall back)
        """
        from fastapi import HTTPException

        request = SearchRequest(query="test query", mode=SearchMode.semantic)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client_with_failure), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store):

            with pytest.raises(HTTPException) as exc_info:
                await hybrid_search(request, mock_principal, mock_db)

            assert exc_info.value.status_code == 503
            assert "embedding" in exc_info.value.detail.lower() or "unavailable" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_mode_semantic_with_null_memory_client_raises_503(
        self,
        mock_principal,
        mock_opensearch_store,
        mock_db,
    ):
        """mode=semantic should raise 503 when memory client is None."""
        from fastapi import HTTPException

        request = SearchRequest(query="test query", mode=SearchMode.semantic)

        with patch("app.routers.search.get_memory_client", return_value=None), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store):

            with pytest.raises(HTTPException) as exc_info:
                await hybrid_search(request, mock_principal, mock_db)

            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_mode_semantic_success(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """mode=semantic with working embedding should use hybrid search."""
        request = SearchRequest(query="test query", mode=SearchMode.semantic)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            assert response.meta.degraded_mode is False

            # Embedding should have been called
            mock_memory_client.embedding_model.embed.assert_called_once()

            # Hybrid search should have been used
            mock_opensearch_store_with_results.hybrid_search_with_access_control.assert_called_once()


class TestHybridSearchBackwardCompatibility:
    """Tests for backward compatibility when mode is not specified."""

    @pytest.mark.asyncio
    async def test_backward_compat_no_mode(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """
        Request without mode field should:
        1. Default to mode=auto behavior
        2. Work normally with embedding
        """
        # Request WITHOUT explicit mode (defaults to auto)
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should have results
            assert len(response.results) > 0

            # Should behave like mode=auto (try embedding)
            mock_memory_client.embedding_model.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_backward_compat_no_mode_with_failure(
        self,
        mock_principal,
        mock_memory_client_with_failure,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """
        Request without mode field when embedding fails should:
        1. Default to mode=auto behavior
        2. Fall back to lexical gracefully
        """
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client_with_failure), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should succeed with fallback
            assert response.meta.degraded_mode is True


class TestHybridSearchAccessControl:
    """Tests for access control enforcement regardless of search mode."""

    @pytest.mark.asyncio
    async def test_access_control_enforced_mode_auto(
        self,
        mock_principal_with_grants,
        mock_memory_client,
        mock_opensearch_store,
        mock_db,
    ):
        """
        Access control should be enforced in mode=auto:
        - access_entities and access_entity_prefixes should be passed to search
        """
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store):

            await hybrid_search(request, mock_principal_with_grants, mock_db)

            # Verify access control parameters were passed to hybrid search
            call_kwargs = mock_opensearch_store.hybrid_search_with_access_control.call_args[1]

            assert "access_entities" in call_kwargs
            assert "access_entity_prefixes" in call_kwargs

            # Should include user's access entity
            assert "user:grischa" in call_kwargs["access_entities"]

    @pytest.mark.asyncio
    async def test_access_control_enforced_mode_lexical(
        self,
        mock_principal_with_grants,
        mock_memory_client,
        mock_opensearch_store,
        mock_db,
    ):
        """
        Access control should be enforced in mode=lexical:
        - access_entities and access_entity_prefixes should be passed to search
        """
        request = SearchRequest(query="test query", mode=SearchMode.lexical)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store):

            await hybrid_search(request, mock_principal_with_grants, mock_db)

            # Verify access control parameters were passed to lexical search
            call_kwargs = mock_opensearch_store.search_with_access_control.call_args[1]

            assert "access_entities" in call_kwargs
            assert "access_entity_prefixes" in call_kwargs


class TestHybridSearchResponseMeta:
    """Tests for the meta field in search responses."""

    @pytest.mark.asyncio
    async def test_response_includes_meta(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """
        Search response should include meta field with:
        - degraded_mode: boolean
        - missing_sources: list of strings
        """
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Verify meta structure
            assert isinstance(response.meta, SearchMeta)
            assert isinstance(response.meta.degraded_mode, bool)
            assert isinstance(response.meta.missing_sources, list)

    @pytest.mark.asyncio
    async def test_response_meta_empty_missing_sources_on_success(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """missing_sources should be empty when all sources are available."""
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            assert response.meta.missing_sources == []

    @pytest.mark.asyncio
    async def test_response_meta_lists_failed_sources(
        self,
        mock_principal,
        mock_memory_client_with_failure,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """missing_sources should list sources that failed."""
        request = SearchRequest(query="test query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client_with_failure), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            assert "embedding" in response.meta.missing_sources


class TestHybridSearchResponseStructure:
    """Tests for the complete response structure."""

    @pytest.mark.asyncio
    async def test_response_structure(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """Response should include results, total_count, took_ms, and meta."""
        request = SearchRequest(query="test")

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Required fields
            assert isinstance(response.results, list)
            assert isinstance(response.total_count, int)
            assert isinstance(response.took_ms, int)
            assert isinstance(response.meta, SearchMeta)


class TestEmbeddingIntegration:
    """Tests for the embedding model integration details."""

    @pytest.mark.asyncio
    async def test_embed_called_with_search_mode(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store,
        mock_db,
    ):
        """
        embed() should be called with (query, "search") parameters.
        The second parameter indicates embedding type for potential optimization.
        """
        request = SearchRequest(query="my search query", mode=SearchMode.auto)

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store):

            await hybrid_search(request, mock_principal, mock_db)

            # Verify embed was called with correct arguments
            mock_memory_client.embedding_model.embed.assert_called_once()
            call_args = mock_memory_client.embedding_model.embed.call_args

            # First arg should be the query
            assert call_args[0][0] == "my search query"

            # Second arg should indicate search mode
            assert call_args[0][1] == "search"

    @pytest.mark.asyncio
    async def test_embed_handles_missing_embedding_model(
        self,
        mock_principal,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """Should handle case where memory_client exists but has no embedding_model."""
        request = SearchRequest(query="test", mode=SearchMode.auto)

        mock_client = MagicMock()
        mock_client.embedding_model = None  # No embedding model

        with patch("app.routers.search.get_memory_client", return_value=mock_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await hybrid_search(request, mock_principal, mock_db)

            # Should fall back gracefully
            assert response.meta.degraded_mode is True


# =============================================================================
# Lexical Search Endpoint Tests
# =============================================================================


class TestLexicalSearchEndpoint:
    """Tests for the /search/lexical endpoint."""

    @pytest.mark.asyncio
    async def test_lexical_endpoint_never_uses_embedding(
        self,
        mock_principal,
        mock_memory_client,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """
        POST /search/lexical should never call embedding,
        regardless of whether mode parameter is passed.
        """
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_memory_client", return_value=mock_memory_client), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await lexical_search(request, mock_principal, mock_db)

            # Embedding should never be called for lexical endpoint
            mock_memory_client.embedding_model.embed.assert_not_called()

            # Only lexical search should be used
            mock_opensearch_store_with_results.search_with_access_control.assert_called_once()

    @pytest.mark.asyncio
    async def test_lexical_endpoint_returns_non_degraded(
        self,
        mock_principal,
        mock_opensearch_store_with_results,
        mock_db,
    ):
        """Lexical endpoint should return degraded_mode=False."""
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await lexical_search(request, mock_principal, mock_db)

            assert response.meta.degraded_mode is False
            assert response.meta.missing_sources == []


# =============================================================================
# Semantic Search Endpoint Tests
# =============================================================================


class TestSemanticSearchEndpoint:
    """Tests for the /search/semantic endpoint behavior."""

    @pytest.mark.asyncio
    async def test_semantic_endpoint_uses_provided_vector(
        self,
        mock_principal,
        mock_opensearch_store_with_results,
        mock_embedding_vector,
        mock_db,
    ):
        """
        POST /search/semantic should use the provided query_vector.
        """
        request = SemanticSearchRequest(
            query="test query",
            query_vector=mock_embedding_vector,
        )

        with patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await semantic_search(request, mock_principal, mock_db)

            # Hybrid search should be used with the provided vector
            call_kwargs = mock_opensearch_store_with_results.hybrid_search_with_access_control.call_args[1]
            assert call_kwargs["query_vector"] == mock_embedding_vector

    @pytest.mark.asyncio
    async def test_semantic_endpoint_returns_non_degraded(
        self,
        mock_principal,
        mock_opensearch_store_with_results,
        mock_embedding_vector,
        mock_db,
    ):
        """Semantic endpoint should return degraded_mode=False."""
        request = SemanticSearchRequest(
            query="test query",
            query_vector=mock_embedding_vector,
        )

        with patch("app.routers.search.get_tenant_opensearch_store", return_value=mock_opensearch_store_with_results):

            response = await semantic_search(request, mock_principal, mock_db)

            assert response.meta.degraded_mode is False
            assert response.meta.missing_sources == []


# =============================================================================
# OpenSearch Unavailability Tests
# =============================================================================


class TestOpenSearchUnavailable:
    """Tests for behavior when OpenSearch is unavailable."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_empty_when_store_unavailable(
        self,
        mock_principal,
        mock_db,
    ):
        """hybrid_search should return empty results when store is None."""
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_memory_client", return_value=None), \
             patch("app.routers.search.get_tenant_opensearch_store", return_value=None):

            response = await hybrid_search(request, mock_principal, mock_db)

            assert response.results == []
            assert response.total_count == 0
            assert response.meta.degraded_mode is False

    @pytest.mark.asyncio
    async def test_lexical_search_returns_empty_when_store_unavailable(
        self,
        mock_principal,
        mock_db,
    ):
        """lexical_search should return empty results when store is None."""
        request = SearchRequest(query="test query")

        with patch("app.routers.search.get_tenant_opensearch_store", return_value=None):

            response = await lexical_search(request, mock_principal, mock_db)

            assert response.results == []
            assert response.total_count == 0

    @pytest.mark.asyncio
    async def test_semantic_search_returns_empty_when_store_unavailable(
        self,
        mock_principal,
        mock_embedding_vector,
        mock_db,
    ):
        """semantic_search should return empty results when store is None."""
        request = SemanticSearchRequest(
            query="test query",
            query_vector=mock_embedding_vector,
        )

        with patch("app.routers.search.get_tenant_opensearch_store", return_value=None):

            response = await semantic_search(request, mock_principal, mock_db)

            assert response.results == []
            assert response.total_count == 0
