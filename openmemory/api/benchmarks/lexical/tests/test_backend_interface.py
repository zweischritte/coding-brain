"""
Tests for lexical backend interface.

Tests cover:
- LexicalBackend abstract base class
- Document dataclass
- SearchResult dataclass
- BackendStats dataclass
- TantivyBackend mock adapter
- OpenSearchBackend mock adapter
- Backend factory function
"""

import pytest
from abc import ABC
from dataclasses import FrozenInstanceError
from typing import List


class TestDocumentDataclass:
    """Tests for the Document dataclass."""

    def test_document_creation(self):
        """Document can be created with required fields."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        doc = Document(id="doc1", content="Hello world")
        assert doc.id == "doc1"
        assert doc.content == "Hello world"

    def test_document_with_metadata(self):
        """Document can be created with optional metadata."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        doc = Document(id="doc1", content="Hello", metadata={"lang": "en"})
        assert doc.metadata == {"lang": "en"}

    def test_document_default_metadata_is_none(self):
        """Document metadata defaults to None."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        doc = Document(id="doc1", content="Hello")
        assert doc.metadata is None

    def test_document_equality(self):
        """Documents with same fields are equal."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        doc1 = Document(id="doc1", content="Hello")
        doc2 = Document(id="doc1", content="Hello")
        assert doc1 == doc2

    def test_document_with_empty_content(self):
        """Document can have empty content."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        doc = Document(id="doc1", content="")
        assert doc.content == ""


class TestSearchResultDataclass:
    """Tests for the SearchResult dataclass."""

    def test_search_result_creation(self):
        """SearchResult can be created with required fields."""
        from openmemory.api.benchmarks.lexical.backends.base import SearchResult

        result = SearchResult(doc_id="doc1", score=0.95)
        assert result.doc_id == "doc1"
        assert result.score == 0.95

    def test_search_result_with_content(self):
        """SearchResult can include document content."""
        from openmemory.api.benchmarks.lexical.backends.base import SearchResult

        result = SearchResult(doc_id="doc1", score=0.95, content="Hello world")
        assert result.content == "Hello world"

    def test_search_result_default_content_is_none(self):
        """SearchResult content defaults to None."""
        from openmemory.api.benchmarks.lexical.backends.base import SearchResult

        result = SearchResult(doc_id="doc1", score=0.95)
        assert result.content is None

    def test_search_result_with_highlights(self):
        """SearchResult can include highlights."""
        from openmemory.api.benchmarks.lexical.backends.base import SearchResult

        result = SearchResult(
            doc_id="doc1",
            score=0.95,
            highlights=["<em>Hello</em> world"]
        )
        assert result.highlights == ["<em>Hello</em> world"]

    def test_search_result_sorting_by_score(self):
        """SearchResults can be sorted by score."""
        from openmemory.api.benchmarks.lexical.backends.base import SearchResult

        results = [
            SearchResult(doc_id="doc1", score=0.5),
            SearchResult(doc_id="doc2", score=0.9),
            SearchResult(doc_id="doc3", score=0.7),
        ]
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        assert sorted_results[0].doc_id == "doc2"
        assert sorted_results[1].doc_id == "doc3"
        assert sorted_results[2].doc_id == "doc1"


class TestBackendStatsDataclass:
    """Tests for the BackendStats dataclass."""

    def test_backend_stats_creation(self):
        """BackendStats can be created with all fields."""
        from openmemory.api.benchmarks.lexical.backends.base import BackendStats

        stats = BackendStats(
            index_size_bytes=1024,
            document_count=100,
            backend_name="tantivy"
        )
        assert stats.index_size_bytes == 1024
        assert stats.document_count == 100
        assert stats.backend_name == "tantivy"

    def test_backend_stats_with_zero_values(self):
        """BackendStats can have zero values for empty index."""
        from openmemory.api.benchmarks.lexical.backends.base import BackendStats

        stats = BackendStats(
            index_size_bytes=0,
            document_count=0,
            backend_name="opensearch"
        )
        assert stats.index_size_bytes == 0
        assert stats.document_count == 0

    def test_backend_stats_with_additional_metadata(self):
        """BackendStats can include additional metadata."""
        from openmemory.api.benchmarks.lexical.backends.base import BackendStats

        stats = BackendStats(
            index_size_bytes=1024,
            document_count=100,
            backend_name="tantivy",
            additional_info={"version": "0.22.0"}
        )
        assert stats.additional_info == {"version": "0.22.0"}

    def test_backend_stats_default_additional_info_is_none(self):
        """BackendStats additional_info defaults to None."""
        from openmemory.api.benchmarks.lexical.backends.base import BackendStats

        stats = BackendStats(
            index_size_bytes=1024,
            document_count=100,
            backend_name="tantivy"
        )
        assert stats.additional_info is None


class TestLexicalBackendAbstractClass:
    """Tests for the LexicalBackend abstract base class."""

    def test_lexical_backend_is_abstract(self):
        """LexicalBackend is an abstract base class."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        assert issubclass(LexicalBackend, ABC)

    def test_cannot_instantiate_lexical_backend_directly(self):
        """Cannot instantiate LexicalBackend directly."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        with pytest.raises(TypeError, match="abstract"):
            LexicalBackend()

    def test_lexical_backend_has_name_property(self):
        """LexicalBackend has abstract name property."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        # Create a concrete implementation to verify the interface
        class TestBackend(LexicalBackend):
            @property
            def name(self) -> str:
                return "test"

            def index_documents(self, documents):
                pass

            def search(self, query, limit=10):
                return []

            def get_stats(self):
                from openmemory.api.benchmarks.lexical.backends.base import BackendStats
                return BackendStats(0, 0, "test")

            def clear(self):
                pass

        backend = TestBackend()
        assert backend.name == "test"

    def test_lexical_backend_requires_index_documents_method(self):
        """LexicalBackend requires index_documents method."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        # Missing index_documents - should fail
        class IncompleteBackend(LexicalBackend):
            @property
            def name(self) -> str:
                return "incomplete"

            def search(self, query, limit=10):
                return []

            def get_stats(self):
                pass

            def clear(self):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBackend()

    def test_lexical_backend_requires_search_method(self):
        """LexicalBackend requires search method."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        class IncompleteBackend(LexicalBackend):
            @property
            def name(self) -> str:
                return "incomplete"

            def index_documents(self, documents):
                pass

            def get_stats(self):
                pass

            def clear(self):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBackend()

    def test_lexical_backend_requires_get_stats_method(self):
        """LexicalBackend requires get_stats method."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        class IncompleteBackend(LexicalBackend):
            @property
            def name(self) -> str:
                return "incomplete"

            def index_documents(self, documents):
                pass

            def search(self, query, limit=10):
                return []

            def clear(self):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBackend()

    def test_lexical_backend_requires_clear_method(self):
        """LexicalBackend requires clear method."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend

        class IncompleteBackend(LexicalBackend):
            @property
            def name(self) -> str:
                return "incomplete"

            def index_documents(self, documents):
                pass

            def search(self, query, limit=10):
                return []

            def get_stats(self):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBackend()


class TestTantivyBackend:
    """Tests for the TantivyBackend mock adapter."""

    def test_tantivy_backend_is_lexical_backend(self):
        """TantivyBackend inherits from LexicalBackend."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend

        assert issubclass(TantivyBackend, LexicalBackend)

    def test_tantivy_backend_instantiation(self):
        """TantivyBackend can be instantiated."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend

        backend = TantivyBackend()
        assert backend is not None

    def test_tantivy_backend_name(self):
        """TantivyBackend has correct name."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend

        backend = TantivyBackend()
        assert backend.name == "tantivy"

    def test_tantivy_backend_index_documents(self):
        """TantivyBackend can index documents."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = TantivyBackend()
        docs = [
            Document(id="doc1", content="Hello world"),
            Document(id="doc2", content="Goodbye world"),
        ]
        backend.index_documents(docs)

        stats = backend.get_stats()
        assert stats.document_count == 2

    def test_tantivy_backend_index_returns_count(self):
        """TantivyBackend.index_documents returns count of indexed docs."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = TantivyBackend()
        docs = [
            Document(id="doc1", content="Hello world"),
            Document(id="doc2", content="Goodbye world"),
        ]
        count = backend.index_documents(docs)
        assert count == 2

    def test_tantivy_backend_search_returns_results(self):
        """TantivyBackend.search returns SearchResult list."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document, SearchResult

        backend = TantivyBackend()
        docs = [
            Document(id="doc1", content="Python programming language"),
            Document(id="doc2", content="Java programming language"),
            Document(id="doc3", content="Hello world"),
        ]
        backend.index_documents(docs)

        results = backend.search("programming", limit=10)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
        # Should find docs with "programming"
        assert len(results) >= 1

    def test_tantivy_backend_search_respects_limit(self):
        """TantivyBackend.search respects limit parameter."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = TantivyBackend()
        docs = [Document(id=f"doc{i}", content=f"test document {i}") for i in range(20)]
        backend.index_documents(docs)

        results = backend.search("test", limit=5)
        assert len(results) <= 5

    def test_tantivy_backend_search_empty_query(self):
        """TantivyBackend.search handles empty query."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = TantivyBackend()
        docs = [Document(id="doc1", content="Hello world")]
        backend.index_documents(docs)

        results = backend.search("", limit=10)
        assert isinstance(results, list)

    def test_tantivy_backend_search_no_matches(self):
        """TantivyBackend.search returns empty list for no matches."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = TantivyBackend()
        docs = [Document(id="doc1", content="Hello world")]
        backend.index_documents(docs)

        results = backend.search("zzzznonexistent", limit=10)
        assert results == []

    def test_tantivy_backend_get_stats(self):
        """TantivyBackend.get_stats returns BackendStats."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document, BackendStats

        backend = TantivyBackend()
        docs = [
            Document(id="doc1", content="Hello world"),
            Document(id="doc2", content="Goodbye world"),
        ]
        backend.index_documents(docs)

        stats = backend.get_stats()
        assert isinstance(stats, BackendStats)
        assert stats.backend_name == "tantivy"
        assert stats.document_count == 2
        assert stats.index_size_bytes >= 0

    def test_tantivy_backend_clear(self):
        """TantivyBackend.clear removes all documents."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = TantivyBackend()
        docs = [Document(id="doc1", content="Hello world")]
        backend.index_documents(docs)

        assert backend.get_stats().document_count == 1

        backend.clear()
        assert backend.get_stats().document_count == 0

    def test_tantivy_backend_empty_index_stats(self):
        """TantivyBackend.get_stats works on empty index."""
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.base import BackendStats

        backend = TantivyBackend()
        stats = backend.get_stats()

        assert isinstance(stats, BackendStats)
        assert stats.document_count == 0


class TestOpenSearchBackend:
    """Tests for the OpenSearchBackend mock adapter."""

    def test_opensearch_backend_is_lexical_backend(self):
        """OpenSearchBackend inherits from LexicalBackend."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend

        assert issubclass(OpenSearchBackend, LexicalBackend)

    def test_opensearch_backend_instantiation(self):
        """OpenSearchBackend can be instantiated."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend

        backend = OpenSearchBackend()
        assert backend is not None

    def test_opensearch_backend_name(self):
        """OpenSearchBackend has correct name."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend

        backend = OpenSearchBackend()
        assert backend.name == "opensearch"

    def test_opensearch_backend_index_documents(self):
        """OpenSearchBackend can index documents."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = OpenSearchBackend()
        docs = [
            Document(id="doc1", content="Hello world"),
            Document(id="doc2", content="Goodbye world"),
        ]
        backend.index_documents(docs)

        stats = backend.get_stats()
        assert stats.document_count == 2

    def test_opensearch_backend_index_returns_count(self):
        """OpenSearchBackend.index_documents returns count of indexed docs."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = OpenSearchBackend()
        docs = [
            Document(id="doc1", content="Hello world"),
            Document(id="doc2", content="Goodbye world"),
        ]
        count = backend.index_documents(docs)
        assert count == 2

    def test_opensearch_backend_search_returns_results(self):
        """OpenSearchBackend.search returns SearchResult list."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document, SearchResult

        backend = OpenSearchBackend()
        docs = [
            Document(id="doc1", content="Python programming language"),
            Document(id="doc2", content="Java programming language"),
            Document(id="doc3", content="Hello world"),
        ]
        backend.index_documents(docs)

        results = backend.search("programming", limit=10)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
        assert len(results) >= 1

    def test_opensearch_backend_search_respects_limit(self):
        """OpenSearchBackend.search respects limit parameter."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = OpenSearchBackend()
        docs = [Document(id=f"doc{i}", content=f"test document {i}") for i in range(20)]
        backend.index_documents(docs)

        results = backend.search("test", limit=5)
        assert len(results) <= 5

    def test_opensearch_backend_search_empty_query(self):
        """OpenSearchBackend.search handles empty query."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = OpenSearchBackend()
        docs = [Document(id="doc1", content="Hello world")]
        backend.index_documents(docs)

        results = backend.search("", limit=10)
        assert isinstance(results, list)

    def test_opensearch_backend_search_no_matches(self):
        """OpenSearchBackend.search returns empty list for no matches."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = OpenSearchBackend()
        docs = [Document(id="doc1", content="Hello world")]
        backend.index_documents(docs)

        results = backend.search("zzzznonexistent", limit=10)
        assert results == []

    def test_opensearch_backend_get_stats(self):
        """OpenSearchBackend.get_stats returns BackendStats."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document, BackendStats

        backend = OpenSearchBackend()
        docs = [
            Document(id="doc1", content="Hello world"),
            Document(id="doc2", content="Goodbye world"),
        ]
        backend.index_documents(docs)

        stats = backend.get_stats()
        assert isinstance(stats, BackendStats)
        assert stats.backend_name == "opensearch"
        assert stats.document_count == 2
        assert stats.index_size_bytes >= 0

    def test_opensearch_backend_clear(self):
        """OpenSearchBackend.clear removes all documents."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend = OpenSearchBackend()
        docs = [Document(id="doc1", content="Hello world")]
        backend.index_documents(docs)

        assert backend.get_stats().document_count == 1

        backend.clear()
        assert backend.get_stats().document_count == 0

    def test_opensearch_backend_empty_index_stats(self):
        """OpenSearchBackend.get_stats works on empty index."""
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend
        from openmemory.api.benchmarks.lexical.backends.base import BackendStats

        backend = OpenSearchBackend()
        stats = backend.get_stats()

        assert isinstance(stats, BackendStats)
        assert stats.document_count == 0


class TestBackendFactory:
    """Tests for the backend factory function."""

    def test_create_backend_tantivy(self):
        """Factory creates TantivyBackend."""
        from openmemory.api.benchmarks.lexical.backends import create_backend
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend

        backend = create_backend("tantivy")
        assert isinstance(backend, TantivyBackend)

    def test_create_backend_opensearch(self):
        """Factory creates OpenSearchBackend."""
        from openmemory.api.benchmarks.lexical.backends import create_backend
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend

        backend = create_backend("opensearch")
        assert isinstance(backend, OpenSearchBackend)

    def test_create_backend_case_insensitive(self):
        """Factory handles case-insensitive backend names."""
        from openmemory.api.benchmarks.lexical.backends import create_backend
        from openmemory.api.benchmarks.lexical.backends.tantivy import TantivyBackend
        from openmemory.api.benchmarks.lexical.backends.opensearch import OpenSearchBackend

        assert isinstance(create_backend("TANTIVY"), TantivyBackend)
        assert isinstance(create_backend("Tantivy"), TantivyBackend)
        assert isinstance(create_backend("OpenSearch"), OpenSearchBackend)
        assert isinstance(create_backend("OPENSEARCH"), OpenSearchBackend)

    def test_create_backend_unknown_raises_error(self):
        """Factory raises ValueError for unknown backend."""
        from openmemory.api.benchmarks.lexical.backends import create_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            create_backend("unknown_backend")

    def test_list_available_backends(self):
        """list_available_backends returns available backend names."""
        from openmemory.api.benchmarks.lexical.backends import list_available_backends

        backends = list_available_backends()
        assert "tantivy" in backends
        assert "opensearch" in backends
        assert len(backends) == 2


class TestBackendInterfaceContract:
    """Tests for backend interface contract across implementations."""

    @pytest.fixture(params=["tantivy", "opensearch"])
    def backend(self, request):
        """Fixture providing each backend implementation."""
        from openmemory.api.benchmarks.lexical.backends import create_backend
        return create_backend(request.param)

    def test_backend_implements_lexical_backend(self, backend):
        """All backends implement LexicalBackend interface."""
        from openmemory.api.benchmarks.lexical.backends.base import LexicalBackend
        assert isinstance(backend, LexicalBackend)

    def test_backend_has_name_property(self, backend):
        """All backends have name property returning string."""
        assert isinstance(backend.name, str)
        assert len(backend.name) > 0

    def test_backend_index_and_search_consistency(self, backend):
        """Index then search returns consistent results."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        docs = [
            Document(id="unique_test_doc", content="unique_content_xyz123"),
        ]
        backend.index_documents(docs)

        results = backend.search("unique_content_xyz123", limit=10)
        assert len(results) == 1
        assert results[0].doc_id == "unique_test_doc"

    def test_backend_clear_empties_index(self, backend):
        """Clear removes all documents from index."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        docs = [Document(id="doc1", content="test content")]
        backend.index_documents(docs)
        assert backend.get_stats().document_count == 1

        backend.clear()
        assert backend.get_stats().document_count == 0

    def test_backend_stats_reflect_document_count(self, backend):
        """Stats accurately reflect number of indexed documents."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        assert backend.get_stats().document_count == 0

        docs = [Document(id=f"doc{i}", content=f"content {i}") for i in range(5)]
        backend.index_documents(docs)
        assert backend.get_stats().document_count == 5

    def test_backend_search_results_have_scores(self, backend):
        """Search results include scores."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        docs = [Document(id="doc1", content="test content")]
        backend.index_documents(docs)

        results = backend.search("test", limit=10)
        if results:  # Only check if there are results
            assert all(isinstance(r.score, (int, float)) for r in results)
            assert all(r.score >= 0 for r in results)

    def test_backend_multiple_index_calls_add_documents(self, backend):
        """Multiple index calls accumulate documents."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        backend.index_documents([Document(id="doc1", content="first")])
        assert backend.get_stats().document_count == 1

        backend.index_documents([Document(id="doc2", content="second")])
        assert backend.get_stats().document_count == 2

    def test_backend_search_returns_list(self, backend):
        """Search always returns a list (possibly empty)."""
        from openmemory.api.benchmarks.lexical.backends.base import Document

        # Empty index
        results = backend.search("anything", limit=10)
        assert isinstance(results, list)

        # With documents
        backend.index_documents([Document(id="doc1", content="test")])
        results = backend.search("test", limit=10)
        assert isinstance(results, list)
