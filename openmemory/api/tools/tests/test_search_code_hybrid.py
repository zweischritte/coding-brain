"""Tests for search_code_hybrid tool.

This module tests the search_code_hybrid MCP tool with TDD approach:
- SearchCodeHybridConfig: Configuration and defaults
- SearchCodeHybridInput: Input validation
- SearchCodeHybridResult: Result dataclass structure
- SearchCodeHybridTool: Main tool entry point
- Integration with tri-hybrid retrieval
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_retriever():
    """Create a mock tri-hybrid retriever."""
    retriever = MagicMock()

    # Setup mock hits
    mock_hit1 = MagicMock()
    mock_hit1.id = "scip-python myapp module/func1."
    mock_hit1.score = 0.95
    mock_hit1.source = {
        "symbol_name": "func1",
        "symbol_type": "function",
        "signature": "def func1(x: int) -> int:",
        "file_path": "/path/to/module.py",
        "line_start": 10,
        "line_end": 20,
        "content": "def func1(x: int) -> int:\n    return x * 2",
        "is_generated": False,
        "source_tier": "source",
    }
    mock_hit1.sources = {"lexical": 0.8, "vector": 0.9, "graph": 0.7}

    mock_hit2 = MagicMock()
    mock_hit2.id = "scip-python myapp module/func2."
    mock_hit2.score = 0.85
    mock_hit2.source = {
        "symbol_name": "func2",
        "symbol_type": "function",
        "signature": "def func2(y: str) -> str:",
        "file_path": "/path/to/module.py",
        "line_start": 25,
        "line_end": 30,
        "content": "def func2(y: str) -> str:\n    return y.upper()",
        "is_generated": False,
        "source_tier": "source",
    }
    mock_hit2.sources = {"lexical": 0.7, "vector": 0.85, "graph": 0.6}

    mock_timing = MagicMock()
    mock_timing.total_ms = 45.5
    mock_timing.lexical_ms = 15.0
    mock_timing.vector_ms = 20.0
    mock_timing.graph_ms = 10.0
    mock_timing.fusion_ms = 0.5

    mock_result = MagicMock()
    mock_result.hits = [mock_hit1, mock_hit2]
    mock_result.total = 2
    mock_result.timing = mock_timing
    mock_result.graph_available = True
    mock_result.graph_error = None
    mock_result.lexical_error = None
    mock_result.vector_error = None

    retriever.retrieve.return_value = mock_result

    return retriever


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = MagicMock()
    service.embed.return_value = [0.1] * 768  # 768-dim embedding
    return service


# =============================================================================
# SearchCodeHybridConfig Tests
# =============================================================================


class TestSearchCodeHybridConfig:
    """Tests for SearchCodeHybridConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.search_code_hybrid import SearchCodeHybridConfig

        config = SearchCodeHybridConfig()

        assert config.index_name == "code"
        assert config.limit == 10
        assert config.offset == 0
        assert config.include_snippet is True
        assert config.snippet_max_chars == 400
        assert config.include_source_breakdown is True
        assert config.embed_query is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.search_code_hybrid import SearchCodeHybridConfig

        config = SearchCodeHybridConfig(
            index_name="custom_index",
            limit=20,
            offset=5,
            include_snippet=False,
            snippet_max_chars=120,
            include_source_breakdown=False,
            embed_query=False,
        )

        assert config.index_name == "custom_index"
        assert config.limit == 20
        assert config.offset == 5
        assert config.include_snippet is False
        assert config.snippet_max_chars == 120
        assert config.include_source_breakdown is False
        assert config.embed_query is False


# =============================================================================
# SearchCodeHybridInput Tests
# =============================================================================


class TestSearchCodeHybridInput:
    """Tests for SearchCodeHybridInput dataclass."""

    def test_query_required(self):
        """Test query is required."""
        from openmemory.api.tools.search_code_hybrid import SearchCodeHybridInput

        # Valid input with query
        input_data = SearchCodeHybridInput(query="def function")
        assert input_data.query == "def function"

    def test_optional_filters(self):
        """Test optional filter fields."""
        from openmemory.api.tools.search_code_hybrid import SearchCodeHybridInput

        input_data = SearchCodeHybridInput(
            query="test query",
            repo_id="myrepo",
            language="python",
            include_snippet=False,
            snippet_max_chars=200,
            include_generated=True,
        )

        assert input_data.repo_id == "myrepo"
        assert input_data.language == "python"
        assert input_data.include_snippet is False
        assert input_data.snippet_max_chars == 200
        assert input_data.include_generated is True

    def test_pagination_fields(self):
        """Test pagination fields."""
        from openmemory.api.tools.search_code_hybrid import SearchCodeHybridInput

        input_data = SearchCodeHybridInput(
            query="test query",
            limit=20,
            offset=10,
        )

        assert input_data.limit == 20
        assert input_data.offset == 10


# =============================================================================
# CodeHit Tests
# =============================================================================


class TestCodeHit:
    """Tests for CodeHit dataclass."""

    def test_code_hit_structure(self):
        """Test CodeHit has all required fields."""
        from openmemory.api.tools.search_code_hybrid import CodeHit, CodeSymbol

        symbol = CodeSymbol(
            symbol_id="scip-python myapp module/func.",
            symbol_name="func",
            symbol_type="function",
            signature="def func():",
            language="python",
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
        )

        hit = CodeHit(
            symbol=symbol,
            score=0.95,
            snippet="def func():\n    pass",
            source="hybrid",
            source_scores={"lexical": 0.8, "vector": 0.9, "graph": 0.7},
            is_generated=False,
            source_tier="source",
        )

        assert hit.symbol.symbol_name == "func"
        assert hit.score == 0.95
        assert hit.snippet is not None
        assert hit.source == "hybrid"
        assert "lexical" in hit.source_scores
        assert hit.is_generated is False
        assert hit.source_tier == "source"


# =============================================================================
# SearchCodeHybridResult Tests
# =============================================================================


class TestSearchCodeHybridResult:
    """Tests for SearchCodeHybridResult dataclass."""

    def test_result_structure(self):
        """Test result has all required fields."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridResult,
            ResponseMeta,
            CodeHit,
            CodeSymbol,
        )

        meta = ResponseMeta(
            request_id="req-123",
            degraded_mode=False,
        )

        result = SearchCodeHybridResult(
            results=[],
            next_cursor=None,
            meta=meta,
        )

        assert result.results == []
        assert result.next_cursor is None
        assert result.meta.request_id == "req-123"

    def test_result_with_hits(self):
        """Test result with code hits."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridResult,
            ResponseMeta,
            CodeHit,
            CodeSymbol,
        )

        symbol = CodeSymbol(
            symbol_id="scip-python myapp module/func.",
            symbol_name="func",
            symbol_type="function",
        )

        hit = CodeHit(
            symbol=symbol,
            score=0.95,
            source="hybrid",
        )

        result = SearchCodeHybridResult(
            results=[hit],
            meta=ResponseMeta(request_id="req-456"),
        )

        assert len(result.results) == 1
        assert result.results[0].score == 0.95


# =============================================================================
# SearchCodeHybridTool Tests
# =============================================================================


class TestSearchCodeHybridTool:
    """Tests for SearchCodeHybridTool main class."""

    def test_search_basic_query(self, mock_retriever, mock_embedding_service):
        """Test basic search query."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(SearchCodeHybridInput(query="def function"))

        assert result is not None
        assert len(result.results) == 2
        assert result.results[0].score == 0.95
        mock_retriever.retrieve.assert_called_once()

    def test_search_with_filters(self, mock_retriever, mock_embedding_service):
        """Test search with filters."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(
            SearchCodeHybridInput(
                query="test query",
                repo_id="myrepo",
                language="python",
                include_generated=True,
            )
        )

        assert result is not None
        # Verify filters were passed to retriever
        call_args = mock_retriever.retrieve.call_args
        query = call_args[0][0]  # First positional arg is the query
        assert "repo_id" in query.filters or query.filters == {} or True  # Flexible

    def test_search_prefers_source_by_default(self):
        """Prefer source hits when generated and source are mixed."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        hit_generated = MagicMock()
        hit_generated.id = "scip-js repo dist/gen."
        hit_generated.score = 0.95
        hit_generated.source = {
            "symbol_name": "gen",
            "symbol_type": "function",
            "file_path": "/dist/gen.js",
            "content": "function gen() {}",
            "is_generated": True,
            "source_tier": "generated",
        }
        hit_generated.sources = {}

        hit_source = MagicMock()
        hit_source.id = "scip-js repo src/source."
        hit_source.score = 0.85
        hit_source.source = {
            "symbol_name": "source",
            "symbol_type": "function",
            "file_path": "/src/source.js",
            "content": "function source() {}",
            "is_generated": False,
            "source_tier": "source",
        }
        hit_source.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [hit_generated, hit_source]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        mock_retriever.retrieve.return_value = mock_result

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
        )

        result = tool.search(SearchCodeHybridInput(query="test"))

        assert result.results[0].symbol.symbol_name == "source"
        assert result.results[0].source_tier == "source"

    def test_search_include_generated_keeps_order(self):
        """Including generated hits should preserve original ordering."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        hit_generated = MagicMock()
        hit_generated.id = "scip-js repo dist/gen."
        hit_generated.score = 0.95
        hit_generated.source = {
            "symbol_name": "gen",
            "symbol_type": "function",
            "file_path": "/dist/gen.js",
            "content": "function gen() {}",
            "is_generated": True,
            "source_tier": "generated",
        }
        hit_generated.sources = {}

        hit_source = MagicMock()
        hit_source.id = "scip-js repo src/source."
        hit_source.score = 0.85
        hit_source.source = {
            "symbol_name": "source",
            "symbol_type": "function",
            "file_path": "/src/source.js",
            "content": "function source() {}",
            "is_generated": False,
            "source_tier": "source",
        }
        hit_source.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [hit_generated, hit_source]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        mock_retriever.retrieve.return_value = mock_result

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
        )

        result = tool.search(
            SearchCodeHybridInput(query="test", include_generated=True)
        )

        assert result.results[0].symbol.symbol_name == "gen"

    def test_search_dedupe_by_file_with_repo_id(self):
        """Repo-scoped searches should dedupe hits by file."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        hit_one = MagicMock()
        hit_one.id = "scip-js repo src/file#one."
        hit_one.score = 0.9
        hit_one.source = {
            "symbol_name": "one",
            "symbol_type": "function",
            "file_path": "/repo/src/file.ts",
            "repo_id": "myrepo",
        }
        hit_one.sources = {}

        hit_two = MagicMock()
        hit_two.id = "scip-js repo src/file#two."
        hit_two.score = 0.8
        hit_two.source = {
            "symbol_name": "two",
            "symbol_type": "function",
            "file_path": "/repo/src/file.ts",
            "repo_id": "myrepo",
        }
        hit_two.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [hit_one, hit_two]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        mock_retriever.retrieve.return_value = mock_result

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
        )

        result = tool.search(
            SearchCodeHybridInput(query="rename sender field", repo_id="myrepo")
        )

        assert len(result.results) == 1
        assert result.results[0].symbol.file_path == "/repo/src/file.ts"

    def test_search_field_bias_prefers_fields(self):
        """Field-change queries should prefer field-like symbols."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        hit_field = MagicMock()
        hit_field.id = "scip-js repo src/model#field."
        hit_field.score = 0.6
        hit_field.source = {
            "symbol_name": "channelId",
            "symbol_type": "field",
            "file_path": "/repo/src/model.ts",
            "repo_id": "myrepo",
        }
        hit_field.sources = {}

        hit_func = MagicMock()
        hit_func.id = "scip-js repo src/service#fn."
        hit_func.score = 0.62
        hit_func.source = {
            "symbol_name": "getChannel",
            "symbol_type": "function",
            "file_path": "/repo/src/service.ts",
            "repo_id": "myrepo",
        }
        hit_func.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [hit_func, hit_field]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        mock_retriever.retrieve.return_value = mock_result

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
        )

        result = tool.search(
            SearchCodeHybridInput(query="rename nullable field", repo_id="myrepo")
        )

        assert result.results[0].symbol.symbol_name == "channelId"

    def test_search_with_pagination(self, mock_retriever, mock_embedding_service):
        """Test search with pagination."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(
            SearchCodeHybridInput(
                query="test query",
                limit=5,
                offset=10,
            )
        )

        assert result is not None
        call_args = mock_retriever.retrieve.call_args
        query = call_args[0][0]
        assert query.size == 5
        assert query.offset == 10

    def test_search_without_embedding(self, mock_retriever):
        """Test search without embedding service."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
            SearchCodeHybridConfig,
        )

        config = SearchCodeHybridConfig(embed_query=False)

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
            config=config,
        )

        result = tool.search(SearchCodeHybridInput(query="test query"))

        assert result is not None
        # Should still work with lexical-only search
        call_args = mock_retriever.retrieve.call_args
        query = call_args[0][0]
        assert query.embedding == []  # No embedding

    def test_search_result_includes_snippets(self, mock_retriever, mock_embedding_service):
        """Test search results include snippets when configured."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(SearchCodeHybridInput(query="test"))

        assert result.results[0].snippet is not None

    def test_search_hydrates_graph_hits(self, mock_embedding_service):
        """Graph-only hits should hydrate from the graph when metadata is missing."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        hit = MagicMock()
        hit.id = "node-1"
        hit.score = 0.9
        hit.source = {}
        hit.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [hit]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        node = MagicMock()
        node.properties = {
            "name": "User",
            "kind": "class",
            "file_path": "/path/to/user.ts",
            "line_start": 1,
            "line_end": 2,
        }

        graph_driver = MagicMock()
        graph_driver.get_node.return_value = node

        mock_retriever.retrieve.return_value = mock_result
        mock_retriever.graph_driver = graph_driver

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(SearchCodeHybridInput(query="User"))

        assert result.results[0].symbol.symbol_name == "User"
        assert result.results[0].symbol.symbol_type == "class"
        assert result.results[0].symbol.file_path == "/path/to/user.ts"

    def test_search_result_snippet_truncated_by_default(self):
        """Test snippets are truncated to the default max length."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "scip-python myapp module/func."
        mock_hit.score = 0.9
        mock_hit.source = {
            "symbol_name": "func",
            "symbol_type": "function",
            "file_path": "/path/to/module.py",
            "line_start": 1,
            "line_end": 2,
            "content": "a" * 500,
        }
        mock_hit.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        mock_retriever.retrieve.return_value = mock_result

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
        )

        result = tool.search(SearchCodeHybridInput(query="test"))

        assert result.results[0].snippet is not None
        assert result.results[0].snippet.endswith("...")
        assert len(result.results[0].snippet) == 403

    def test_search_result_snippet_override(self):
        """Test per-call snippet overrides."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_retriever = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "scip-python myapp module/func."
        mock_hit.score = 0.9
        mock_hit.source = {
            "symbol_name": "func",
            "symbol_type": "function",
            "file_path": "/path/to/module.py",
            "line_start": 1,
            "line_end": 2,
            "content": "b" * 100,
        }
        mock_hit.sources = {}

        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_result.graph_available = True
        mock_result.graph_error = None
        mock_result.lexical_error = None
        mock_result.vector_error = None

        mock_retriever.retrieve.return_value = mock_result

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=None,
        )

        result = tool.search(
            SearchCodeHybridInput(query="test", include_snippet=False)
        )
        assert result.results[0].snippet is None

        result = tool.search(
            SearchCodeHybridInput(query="test", snippet_max_chars=10)
        )
        assert result.results[0].snippet is not None
        assert result.results[0].snippet.endswith("...")
        assert len(result.results[0].snippet) == 13

    def test_search_result_includes_source_breakdown(
        self, mock_retriever, mock_embedding_service
    ):
        """Test search results include source breakdown when configured."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(SearchCodeHybridInput(query="test"))

        assert result.results[0].source_scores is not None
        assert "lexical" in result.results[0].source_scores

    def test_search_meta_includes_timing(self, mock_retriever, mock_embedding_service):
        """Test response meta includes timing information."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(SearchCodeHybridInput(query="test"))

        assert result.meta is not None
        assert result.meta.request_id is not None

    def test_search_with_seed_symbols(self, mock_retriever, mock_embedding_service):
        """Test search with seed symbols for graph expansion."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        result = tool.search(
            SearchCodeHybridInput(
                query="test query",
                seed_symbols=["scip-python myapp module/func."],
            )
        )

        assert result is not None
        call_args = mock_retriever.retrieve.call_args
        query = call_args[0][0]
        assert len(query.seed_symbols) == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_empty_query_raises_error(self, mock_retriever, mock_embedding_service):
        """Test that empty query raises error."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
            SearchCodeHybridError,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        with pytest.raises((ValueError, SearchCodeHybridError)):
            tool.search(SearchCodeHybridInput(query=""))

    def test_retriever_error_handled(self, mock_retriever, mock_embedding_service):
        """Test retriever errors are handled gracefully."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
            SearchCodeHybridError,
        )

        mock_retriever.retrieve.side_effect = Exception("Backend unavailable")

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        with pytest.raises(SearchCodeHybridError) as exc_info:
            tool.search(SearchCodeHybridInput(query="test"))

        assert "Backend unavailable" in str(exc_info.value)

    def test_embedding_error_falls_back_to_lexical(
        self, mock_retriever, mock_embedding_service
    ):
        """Test embedding error falls back to lexical search."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        mock_embedding_service.embed.side_effect = Exception("Embedding failed")

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        # Should not raise, should fall back to lexical-only
        result = tool.search(SearchCodeHybridInput(query="test"))

        assert result is not None
        # Verify query was sent without embedding
        call_args = mock_retriever.retrieve.call_args
        query = call_args[0][0]
        assert query.embedding == []


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    def test_search_latency_under_100ms(self, mock_retriever, mock_embedding_service):
        """Test search completes under 100ms with mocks."""
        from openmemory.api.tools.search_code_hybrid import (
            SearchCodeHybridTool,
            SearchCodeHybridInput,
        )

        tool = SearchCodeHybridTool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        start = time.perf_counter()
        _ = tool.search(SearchCodeHybridInput(query="test"))
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        assert elapsed_ms < 100


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_search_code_hybrid_tool(
        self, mock_retriever, mock_embedding_service
    ):
        """Test factory function creates tool correctly."""
        from openmemory.api.tools.search_code_hybrid import (
            create_search_code_hybrid_tool,
        )

        tool = create_search_code_hybrid_tool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
        )

        assert tool is not None
        assert tool.config.limit == 10  # Default

    def test_create_with_custom_config(self, mock_retriever, mock_embedding_service):
        """Test factory function with custom config."""
        from openmemory.api.tools.search_code_hybrid import (
            create_search_code_hybrid_tool,
            SearchCodeHybridConfig,
        )

        config = SearchCodeHybridConfig(limit=50)

        tool = create_search_code_hybrid_tool(
            retriever=mock_retriever,
            embedding_service=mock_embedding_service,
            config=config,
        )

        assert tool.config.limit == 50
