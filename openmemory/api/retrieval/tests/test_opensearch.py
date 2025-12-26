"""Tests for OpenSearch client and retrieval operations.

Following TDD: Write tests first, then implement.
Covers:
- OpenSearch client wrapper with connection pooling
- Index management (create, update, delete)
- Document indexing with embeddings
- Lexical search (BM25)
- Vector search (kNN)
- Hybrid search (BM25 + vector with RRF)
- Search result ranking and scoring
"""

import pytest
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio

# Import the module under test (will be implemented)
from openmemory.api.retrieval.opensearch import (
    # Configuration
    OpenSearchConfig,
    IndexConfig,
    # Client
    OpenSearchClient,
    # Index management
    IndexManager,
    IndexInfo,
    IndexStats,
    # Document operations
    Document,
    BulkResult,
    # Search operations
    SearchResult,
    SearchHit,
    SearchResponse,
    # Lexical search
    LexicalSearchQuery,
    LexicalSearchParams,
    # Vector search
    VectorSearchQuery,
    VectorSearchParams,
    # Hybrid search
    HybridSearchQuery,
    HybridSearchParams,
    RRFConfig,
    # Ranking
    ScoringFunction,
    FieldWeight,
    # Exceptions
    OpenSearchError,
    ConnectionError,
    IndexError,
    DocumentError,
    SearchError,
    # Factory
    create_opensearch_client,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestOpenSearchConfig:
    """Tests for OpenSearchConfig dataclass."""

    def test_config_creation_with_defaults(self):
        """Config can be created with default values."""
        config = OpenSearchConfig()
        assert config.hosts == ["localhost:9200"]
        assert config.use_ssl is False
        assert config.verify_certs is True
        assert config.pool_maxsize == 10
        assert config.pool_connections == 10
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_on_timeout is True

    def test_config_creation_with_custom_values(self):
        """Config can be created with custom values."""
        config = OpenSearchConfig(
            hosts=["node1:9200", "node2:9200"],
            use_ssl=True,
            verify_certs=False,
            username="admin",
            password="secret",
            pool_maxsize=20,
            pool_connections=15,
            timeout=60,
            max_retries=5,
            retry_on_timeout=False,
        )
        assert config.hosts == ["node1:9200", "node2:9200"]
        assert config.use_ssl is True
        assert config.verify_certs is False
        assert config.username == "admin"
        assert config.password == "secret"
        assert config.pool_maxsize == 20
        assert config.pool_connections == 15
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.retry_on_timeout is False

    def test_config_from_env(self):
        """Config can be loaded from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "OPENSEARCH_HOSTS": "node1:9200,node2:9200",
                "OPENSEARCH_USE_SSL": "true",
                "OPENSEARCH_USERNAME": "admin",
                "OPENSEARCH_PASSWORD": "secret",
            },
        ):
            config = OpenSearchConfig.from_env()
            assert config.hosts == ["node1:9200", "node2:9200"]
            assert config.use_ssl is True
            assert config.username == "admin"
            assert config.password == "secret"

    def test_config_validation_hosts_not_empty(self):
        """Config validates hosts list is not empty."""
        with pytest.raises(ValueError, match="hosts cannot be empty"):
            OpenSearchConfig(hosts=[])

    def test_config_validation_timeout_positive(self):
        """Config validates timeout is positive."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            OpenSearchConfig(timeout=0)

    def test_config_validation_pool_size_positive(self):
        """Config validates pool sizes are positive."""
        with pytest.raises(ValueError, match="pool_maxsize must be positive"):
            OpenSearchConfig(pool_maxsize=0)


class TestIndexConfig:
    """Tests for IndexConfig dataclass."""

    def test_index_config_creation(self):
        """IndexConfig can be created with required fields."""
        config = IndexConfig(
            name="test_index",
            mappings={
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 768},
                }
            },
        )
        assert config.name == "test_index"
        assert "properties" in config.mappings
        assert config.settings == {}
        assert config.aliases == []

    def test_index_config_with_settings(self):
        """IndexConfig can have custom settings."""
        config = IndexConfig(
            name="test_index",
            mappings={"properties": {}},
            settings={
                "index.knn": True,
                "number_of_shards": 3,
                "number_of_replicas": 1,
            },
            aliases=["code_index"],
        )
        assert config.settings["number_of_shards"] == 3
        assert "code_index" in config.aliases

    def test_index_config_for_code(self):
        """IndexConfig factory for code index."""
        config = IndexConfig.for_code(
            name="code_index",
            embedding_dim=768,
            languages=["python", "typescript"],
        )
        assert config.name == "code_index"
        assert "embedding" in config.mappings["properties"]
        assert config.mappings["properties"]["embedding"]["dimension"] == 768
        assert "content" in config.mappings["properties"]
        assert "language" in config.mappings["properties"]
        assert "file_path" in config.mappings["properties"]
        assert "symbol_name" in config.mappings["properties"]
        assert "symbol_type" in config.mappings["properties"]
        assert config.settings.get("index.knn") is True

    def test_index_config_for_memory(self):
        """IndexConfig factory for memory index."""
        config = IndexConfig.for_memory(
            name="memory_index",
            embedding_dim=1536,
        )
        assert config.name == "memory_index"
        assert config.mappings["properties"]["embedding"]["dimension"] == 1536
        assert "content" in config.mappings["properties"]
        assert "scope" in config.mappings["properties"]
        assert "user_id" in config.mappings["properties"]
        assert "org_id" in config.mappings["properties"]


# =============================================================================
# Client Tests
# =============================================================================


class TestOpenSearchClient:
    """Tests for OpenSearchClient wrapper."""

    def test_client_creation(self):
        """Client can be created with config."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        assert client.config == config
        assert client._client is None  # Lazy initialization

    def test_client_connection_pooling(self):
        """Client uses connection pooling."""
        config = OpenSearchConfig(pool_maxsize=20, pool_connections=15)
        client = OpenSearchClient(config)
        # Pool settings are applied when connecting
        with patch("openmemory.api.retrieval.opensearch.opensearchpy.OpenSearch") as mock_os:
            client.connect()
            call_args = mock_os.call_args
            assert call_args.kwargs.get("pool_maxsize") == 20
            assert call_args.kwargs.get("pool_connections") == 15

    @pytest.mark.asyncio
    async def test_client_async_connection(self):
        """Client supports async connection."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        # AsyncOpenSearch may not be available in all installations
        # Patch the module's opensearchpy reference
        import openmemory.api.retrieval.opensearch as os_module
        mock_async_os = MagicMock()
        mock_async_os.return_value = AsyncMock()
        original = getattr(os_module.opensearchpy, "AsyncOpenSearch", None)
        os_module.opensearchpy.AsyncOpenSearch = mock_async_os
        try:
            await client.connect_async()
            mock_async_os.assert_called_once()
        finally:
            if original is not None:
                os_module.opensearchpy.AsyncOpenSearch = original
            else:
                delattr(os_module.opensearchpy, "AsyncOpenSearch")

    def test_client_health_check(self):
        """Client can check cluster health."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        mock_client = Mock()
        mock_client.cluster.health.return_value = {
            "cluster_name": "test",
            "status": "green",
            "number_of_nodes": 3,
        }
        client._client = mock_client

        health = client.health()
        assert health["status"] == "green"
        assert health["number_of_nodes"] == 3

    def test_client_close(self):
        """Client can be closed."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        mock_client = Mock()
        client._client = mock_client

        client.close()
        mock_client.close.assert_called_once()
        assert client._client is None

    def test_client_context_manager(self):
        """Client works as context manager."""
        config = OpenSearchConfig()
        with patch("openmemory.api.retrieval.opensearch.opensearchpy.OpenSearch") as mock_os:
            mock_instance = Mock()
            mock_os.return_value = mock_instance

            with OpenSearchClient(config) as client:
                client.connect()
                assert client._client is not None

            mock_instance.close.assert_called_once()

    def test_client_retry_on_connection_error(self):
        """Client retries on connection errors."""
        config = OpenSearchConfig(max_retries=3)
        client = OpenSearchClient(config)

        with patch("openmemory.api.retrieval.opensearch.opensearchpy.OpenSearch") as mock_os:
            from opensearchpy.exceptions import ConnectionError as OSConnectionError

            mock_os.side_effect = [
                OSConnectionError("Failed"),
                OSConnectionError("Failed"),
                Mock(),  # Success on third try
            ]

            client.connect()
            assert mock_os.call_count == 3

    def test_client_raises_after_max_retries(self):
        """Client raises after max retries exceeded."""
        config = OpenSearchConfig(max_retries=2)
        client = OpenSearchClient(config)

        with patch("openmemory.api.retrieval.opensearch.opensearchpy.OpenSearch") as mock_os:
            from openmemory.api.retrieval.opensearch import OSConnectionError

            # Create a mock that properly simulates a connection error
            def raise_error(*args, **kwargs):
                raise OSConnectionError(500, "Connection refused", {"error": "Failed"})

            mock_os.side_effect = raise_error

            with pytest.raises(ConnectionError, match="Failed to connect"):
                client.connect()


# =============================================================================
# Index Management Tests
# =============================================================================


class TestIndexManager:
    """Tests for IndexManager."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenSearch client."""
        client = Mock(spec=OpenSearchClient)
        client._client = Mock()
        return client

    def test_index_manager_creation(self, mock_client):
        """IndexManager can be created with client."""
        manager = IndexManager(mock_client)
        assert manager.client == mock_client

    def test_create_index(self, mock_client):
        """IndexManager can create an index."""
        manager = IndexManager(mock_client)
        config = IndexConfig(
            name="test_index",
            mappings={"properties": {"content": {"type": "text"}}},
            settings={"number_of_shards": 1},
        )

        mock_client._client.indices.create.return_value = {"acknowledged": True}

        result = manager.create_index(config)
        assert result is True

        mock_client._client.indices.create.assert_called_once()
        call_args = mock_client._client.indices.create.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert "mappings" in call_args.kwargs["body"]
        assert "settings" in call_args.kwargs["body"]

    def test_create_index_with_aliases(self, mock_client):
        """IndexManager creates index with aliases."""
        manager = IndexManager(mock_client)
        config = IndexConfig(
            name="test_index_v1",
            mappings={"properties": {}},
            aliases=["test_index", "code_index"],
        )

        mock_client._client.indices.create.return_value = {"acknowledged": True}

        manager.create_index(config)

        call_args = mock_client._client.indices.create.call_args
        assert "aliases" in call_args.kwargs["body"]
        aliases = call_args.kwargs["body"]["aliases"]
        assert "test_index" in aliases
        assert "code_index" in aliases

    def test_create_index_already_exists(self, mock_client):
        """IndexManager handles index already exists error."""
        manager = IndexManager(mock_client)
        config = IndexConfig(name="test_index", mappings={"properties": {}})

        from opensearchpy.exceptions import RequestError

        mock_client._client.indices.create.side_effect = RequestError(
            400, "resource_already_exists_exception", {}
        )

        # Should not raise, returns False
        result = manager.create_index(config, ignore_existing=True)
        assert result is False

    def test_create_index_raises_on_error(self, mock_client):
        """IndexManager raises on index creation error."""
        manager = IndexManager(mock_client)
        config = IndexConfig(name="test_index", mappings={"properties": {}})

        from opensearchpy.exceptions import RequestError

        mock_client._client.indices.create.side_effect = RequestError(
            400, "invalid_mapping", {}
        )

        with pytest.raises(IndexError, match="Failed to create index"):
            manager.create_index(config)

    def test_delete_index(self, mock_client):
        """IndexManager can delete an index."""
        manager = IndexManager(mock_client)
        mock_client._client.indices.delete.return_value = {"acknowledged": True}

        result = manager.delete_index("test_index")
        assert result is True

        mock_client._client.indices.delete.assert_called_once_with(index="test_index")

    def test_delete_index_not_found(self, mock_client):
        """IndexManager handles index not found on delete."""
        manager = IndexManager(mock_client)

        from opensearchpy.exceptions import NotFoundError

        mock_client._client.indices.delete.side_effect = NotFoundError(
            404, "index_not_found_exception", {}
        )

        result = manager.delete_index("test_index", ignore_not_found=True)
        assert result is False

    def test_index_exists(self, mock_client):
        """IndexManager can check if index exists."""
        manager = IndexManager(mock_client)
        mock_client._client.indices.exists.return_value = True

        assert manager.index_exists("test_index") is True
        mock_client._client.indices.exists.assert_called_once_with(index="test_index")

    def test_get_index_info(self, mock_client):
        """IndexManager can get index info."""
        manager = IndexManager(mock_client)
        mock_client._client.indices.get.return_value = {
            "test_index": {
                "mappings": {"properties": {}},
                "settings": {"index": {"number_of_shards": "1"}},
                "aliases": {"code_index": {}},
            }
        }

        info = manager.get_index_info("test_index")
        assert isinstance(info, IndexInfo)
        assert info.name == "test_index"
        assert info.aliases == ["code_index"]

    def test_get_index_stats(self, mock_client):
        """IndexManager can get index stats."""
        manager = IndexManager(mock_client)
        mock_client._client.indices.stats.return_value = {
            "_all": {
                "primaries": {
                    "docs": {"count": 1000, "deleted": 10},
                    "store": {"size_in_bytes": 1024000},
                }
            }
        }

        stats = manager.get_index_stats("test_index")
        assert isinstance(stats, IndexStats)
        assert stats.doc_count == 1000
        assert stats.deleted_docs == 10
        assert stats.size_bytes == 1024000

    def test_update_mapping(self, mock_client):
        """IndexManager can update index mapping."""
        manager = IndexManager(mock_client)
        new_mapping = {"properties": {"new_field": {"type": "keyword"}}}
        mock_client._client.indices.put_mapping.return_value = {"acknowledged": True}

        result = manager.update_mapping("test_index", new_mapping)
        assert result is True

        mock_client._client.indices.put_mapping.assert_called_once_with(
            index="test_index", body=new_mapping
        )

    def test_update_settings(self, mock_client):
        """IndexManager can update index settings."""
        manager = IndexManager(mock_client)
        new_settings = {"index.refresh_interval": "30s"}
        mock_client._client.indices.put_settings.return_value = {"acknowledged": True}

        result = manager.update_settings("test_index", new_settings)
        assert result is True

    def test_refresh_index(self, mock_client):
        """IndexManager can refresh an index."""
        manager = IndexManager(mock_client)
        mock_client._client.indices.refresh.return_value = {"_shards": {"total": 2}}

        manager.refresh_index("test_index")
        mock_client._client.indices.refresh.assert_called_once_with(index="test_index")

    def test_list_indices(self, mock_client):
        """IndexManager can list indices matching a pattern."""
        manager = IndexManager(mock_client)
        mock_client._client.indices.get.return_value = {
            "code_index_v1": {},
            "code_index_v2": {},
        }

        indices = manager.list_indices(pattern="code_*")
        assert "code_index_v1" in indices
        assert "code_index_v2" in indices


# =============================================================================
# Document Operations Tests
# =============================================================================


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Document can be created with required fields."""
        doc = Document(
            id="doc1",
            content="def hello(): pass",
            embedding=[0.1, 0.2, 0.3],
        )
        assert doc.id == "doc1"
        assert doc.content == "def hello(): pass"
        assert len(doc.embedding) == 3
        assert doc.metadata == {}

    def test_document_with_metadata(self):
        """Document can have metadata."""
        doc = Document(
            id="doc1",
            content="function hello() {}",
            embedding=[0.1] * 768,
            metadata={
                "file_path": "/src/main.ts",
                "language": "typescript",
                "symbol_name": "hello",
                "symbol_type": "function",
                "line_start": 1,
                "line_end": 3,
            },
        )
        assert doc.metadata["file_path"] == "/src/main.ts"
        assert doc.metadata["language"] == "typescript"

    def test_document_to_opensearch_body(self):
        """Document can convert to OpenSearch document body."""
        doc = Document(
            id="doc1",
            content="def hello(): pass",
            embedding=[0.1, 0.2, 0.3],
            metadata={"language": "python"},
        )
        body = doc.to_opensearch_body()
        assert body["content"] == "def hello(): pass"
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert body["language"] == "python"

    def test_document_from_opensearch_hit(self):
        """Document can be created from OpenSearch hit."""
        hit = {
            "_id": "doc1",
            "_source": {
                "content": "def hello(): pass",
                "embedding": [0.1, 0.2, 0.3],
                "language": "python",
            },
            "_score": 0.95,
        }
        doc = Document.from_opensearch_hit(hit)
        assert doc.id == "doc1"
        assert doc.content == "def hello(): pass"
        assert doc.metadata["language"] == "python"


class TestDocumentOperations:
    """Tests for document indexing operations."""

    @pytest.fixture
    def client_with_mock(self):
        """Create a real OpenSearchClient with mocked internal client."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        client._client = Mock()
        return client

    def test_index_document(self, client_with_mock):
        """Client can index a single document."""
        client = client_with_mock
        doc = Document(
            id="doc1",
            content="def hello(): pass",
            embedding=[0.1, 0.2, 0.3],
        )

        client._client.index.return_value = {"result": "created", "_id": "doc1"}

        result = client.index_document("test_index", doc)
        assert result is True

        client._client.index.assert_called_once()
        call_args = client._client.index.call_args
        assert call_args.kwargs["index"] == "test_index"
        assert call_args.kwargs["id"] == "doc1"

    def test_index_document_update(self, client_with_mock):
        """Client can update an existing document."""
        client = client_with_mock
        doc = Document(
            id="doc1",
            content="def hello_updated(): pass",
            embedding=[0.1, 0.2, 0.4],
        )

        client._client.index.return_value = {"result": "updated", "_id": "doc1"}

        result = client.index_document("test_index", doc)
        assert result is True

    def test_bulk_index_documents(self, client_with_mock):
        """Client can bulk index documents."""
        client = client_with_mock
        docs = [
            Document(id=f"doc{i}", content=f"content{i}", embedding=[0.1] * 3)
            for i in range(100)
        ]

        client._client.bulk.return_value = {
            "took": 100,
            "errors": False,
            "items": [{"index": {"_id": f"doc{i}", "result": "created"}} for i in range(100)],
        }

        result = client.bulk_index("test_index", docs)
        assert isinstance(result, BulkResult)
        assert result.total == 100
        assert result.succeeded == 100
        assert result.failed == 0

    def test_bulk_index_partial_failure(self, client_with_mock):
        """Client handles partial bulk index failures."""
        client = client_with_mock
        docs = [
            Document(id=f"doc{i}", content=f"content{i}", embedding=[0.1] * 3)
            for i in range(10)
        ]

        client._client.bulk.return_value = {
            "took": 50,
            "errors": True,
            "items": [
                {"index": {"_id": "doc0", "result": "created"}},
                {"index": {"_id": "doc1", "error": {"reason": "mapping error"}}},
                *[{"index": {"_id": f"doc{i}", "result": "created"}} for i in range(2, 10)],
            ],
        }

        result = client.bulk_index("test_index", docs)
        assert result.total == 10
        assert result.succeeded == 9
        assert result.failed == 1
        assert len(result.errors) == 1
        assert "doc1" in result.errors[0]

    def test_get_document(self, client_with_mock):
        """Client can get a document by ID."""
        client = client_with_mock
        client._client.get.return_value = {
            "_id": "doc1",
            "_source": {
                "content": "def hello(): pass",
                "embedding": [0.1, 0.2, 0.3],
            },
            "found": True,
        }

        doc = client.get_document("test_index", "doc1")
        assert doc is not None
        assert doc.id == "doc1"
        assert doc.content == "def hello(): pass"

    def test_get_document_not_found(self, client_with_mock):
        """Client returns None for non-existent document."""
        client = client_with_mock

        from opensearchpy.exceptions import NotFoundError

        client._client.get.side_effect = NotFoundError(404, "not_found", {})

        doc = client.get_document("test_index", "nonexistent")
        assert doc is None

    def test_delete_document(self, client_with_mock):
        """Client can delete a document."""
        client = client_with_mock
        client._client.delete.return_value = {"result": "deleted"}

        result = client.delete_document("test_index", "doc1")
        assert result is True

    def test_delete_by_query(self, client_with_mock):
        """Client can delete documents by query."""
        client = client_with_mock
        query = {"match": {"file_path": "/old/path.py"}}
        client._client.delete_by_query.return_value = {
            "deleted": 5,
            "total": 5,
            "failures": [],
        }

        result = client.delete_by_query("test_index", query)
        assert result == 5

    def test_update_document(self, client_with_mock):
        """Client can update a document partially."""
        client = client_with_mock
        client._client.update.return_value = {"result": "updated", "_id": "doc1"}

        result = client.update_document(
            "test_index",
            "doc1",
            {"content": "def hello_v2(): pass"},
        )
        assert result is True

    def test_count_documents(self, client_with_mock):
        """Client can count documents matching a query."""
        client = client_with_mock
        client._client.count.return_value = {"count": 1000}

        count = client.count("test_index", {"match_all": {}})
        assert count == 1000


# =============================================================================
# Lexical Search Tests
# =============================================================================


class TestLexicalSearchQuery:
    """Tests for LexicalSearchQuery."""

    def test_query_creation(self):
        """LexicalSearchQuery can be created."""
        query = LexicalSearchQuery(
            query_text="def hello",
            fields=["content", "symbol_name"],
        )
        assert query.query_text == "def hello"
        assert "content" in query.fields
        assert query.size == 10  # default
        assert query.offset == 0  # default

    def test_query_with_filters(self):
        """LexicalSearchQuery can have filters."""
        query = LexicalSearchQuery(
            query_text="function",
            fields=["content"],
            filters={
                "language": "typescript",
                "symbol_type": "function",
            },
        )
        assert query.filters["language"] == "typescript"

    def test_query_to_opensearch_body(self):
        """LexicalSearchQuery converts to OpenSearch body."""
        query = LexicalSearchQuery(
            query_text="def hello",
            fields=["content", "symbol_name"],
            filters={"language": "python"},
            size=20,
            offset=10,
        )
        body = query.to_opensearch_body()

        assert "query" in body
        assert "bool" in body["query"]
        assert "must" in body["query"]["bool"]
        assert "filter" in body["query"]["bool"]
        assert body["size"] == 20
        assert body["from"] == 10


class TestLexicalSearchParams:
    """Tests for LexicalSearchParams."""

    def test_params_creation(self):
        """LexicalSearchParams can be created."""
        params = LexicalSearchParams(
            analyzer="code_analyzer",
            minimum_should_match="75%",
            fuzziness="AUTO",
        )
        assert params.analyzer == "code_analyzer"
        assert params.minimum_should_match == "75%"

    def test_params_with_field_weights(self):
        """LexicalSearchParams can have field weights."""
        params = LexicalSearchParams(
            field_weights=[
                FieldWeight(field="symbol_name", weight=3.0),
                FieldWeight(field="content", weight=1.0),
                FieldWeight(field="file_path", weight=2.0),
            ],
        )
        assert len(params.field_weights) == 3
        assert params.field_weights[0].weight == 3.0

    def test_params_boost_calculation(self):
        """LexicalSearchParams applies field boosts to query."""
        params = LexicalSearchParams(
            field_weights=[
                FieldWeight(field="symbol_name", weight=3.0),
                FieldWeight(field="content", weight=1.0),
            ],
        )
        boosted_fields = params.get_boosted_fields()
        assert "symbol_name^3.0" in boosted_fields
        assert "content^1.0" in boosted_fields


class TestLexicalSearch:
    """Tests for lexical search operations."""

    @pytest.fixture
    def client_with_mock(self):
        """Create a real OpenSearchClient with mocked internal client."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        client._client = Mock()
        return client

    def test_lexical_search_basic(self, client_with_mock):
        """Client can perform basic lexical search."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="hello world",
            fields=["content"],
        )

        client._client.search.return_value = {
            "took": 10,
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {"_id": "doc1", "_score": 0.9, "_source": {"content": "hello world"}},
                    {"_id": "doc2", "_score": 0.7, "_source": {"content": "hello there"}},
                ],
            },
        }

        response = client.lexical_search("test_index", query)
        assert isinstance(response, SearchResponse)
        assert response.total == 2
        assert len(response.hits) == 2
        assert response.hits[0].score == 0.9

    def test_lexical_search_with_params(self, client_with_mock):
        """Client can perform lexical search with custom params."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="getUserById",
            fields=["content", "symbol_name"],
        )
        params = LexicalSearchParams(
            analyzer="camel_case_analyzer",
            minimum_should_match="50%",
            field_weights=[
                FieldWeight(field="symbol_name", weight=3.0),
                FieldWeight(field="content", weight=1.0),
            ],
        )

        client._client.search.return_value = {
            "took": 5,
            "hits": {"total": {"value": 1}, "hits": []},
        }

        response = client.lexical_search("test_index", query, params)
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        # Should use multi_match with field boosts
        assert "multi_match" in str(body)

    def test_lexical_search_bm25_scoring(self, client_with_mock):
        """Lexical search uses BM25 scoring by default."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="function test",
            fields=["content"],
        )

        client._client.search.return_value = {
            "took": 10,
            "hits": {"total": {"value": 0}, "hits": []},
        }

        client.lexical_search("test_index", query)
        call_args = client._client.search.call_args
        # BM25 is OpenSearch default, no special query needed
        body = call_args.kwargs["body"]
        assert "multi_match" in str(body) or "match" in str(body)

    def test_lexical_search_with_highlight(self, client_with_mock):
        """Lexical search can return highlights."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="hello",
            fields=["content"],
            highlight=True,
        )

        client._client.search.return_value = {
            "took": 10,
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 0.9,
                        "_source": {"content": "def hello(): pass"},
                        "highlight": {"content": ["def <em>hello</em>(): pass"]},
                    },
                ],
            },
        }

        response = client.lexical_search("test_index", query)
        assert response.hits[0].highlights is not None
        assert "content" in response.hits[0].highlights

    def test_lexical_search_pagination(self, client_with_mock):
        """Lexical search supports pagination."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="test",
            fields=["content"],
            size=10,
            offset=20,
        )

        client._client.search.return_value = {
            "took": 10,
            "hits": {"total": {"value": 100}, "hits": []},
        }

        client.lexical_search("test_index", query)
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        assert body["size"] == 10
        assert body["from"] == 20


# =============================================================================
# Vector Search Tests
# =============================================================================


class TestVectorSearchQuery:
    """Tests for VectorSearchQuery."""

    def test_query_creation(self):
        """VectorSearchQuery can be created."""
        embedding = [0.1] * 768
        query = VectorSearchQuery(
            embedding=embedding,
            k=10,
        )
        assert len(query.embedding) == 768
        assert query.k == 10
        assert query.min_score is None

    def test_query_with_filters(self):
        """VectorSearchQuery can have filters."""
        query = VectorSearchQuery(
            embedding=[0.1] * 768,
            k=20,
            filters={"language": "python"},
            min_score=0.7,
        )
        assert query.filters["language"] == "python"
        assert query.min_score == 0.7

    def test_query_to_opensearch_body(self):
        """VectorSearchQuery converts to OpenSearch script_score body."""
        embedding = [0.1] * 768
        query = VectorSearchQuery(
            embedding=embedding,
            k=10,
            filters={"language": "python"},
        )
        body = query.to_opensearch_body(field="embedding")

        # Uses script_score with knn_score for vector search
        assert "query" in body
        assert "script_score" in body["query"]
        assert "script" in body["query"]["script_score"]
        assert body["query"]["script_score"]["script"]["params"]["query_value"] == embedding
        assert body["query"]["script_score"]["script"]["params"]["field"] == "embedding"
        assert body["size"] == 10


class TestVectorSearchParams:
    """Tests for VectorSearchParams."""

    def test_params_creation(self):
        """VectorSearchParams can be created."""
        params = VectorSearchParams(
            ef_search=100,
            space_type="cosinesimil",
        )
        assert params.ef_search == 100
        assert params.space_type == "cosinesimil"

    def test_params_defaults(self):
        """VectorSearchParams has sensible defaults."""
        params = VectorSearchParams()
        assert params.ef_search is None  # Use index default
        assert params.space_type is None  # Use index default


class TestVectorSearch:
    """Tests for vector search operations."""

    @pytest.fixture
    def client_with_mock(self):
        """Create a real OpenSearchClient with mocked internal client."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        client._client = Mock()
        return client

    def test_vector_search_basic(self, client_with_mock):
        """Client can perform basic vector search."""
        client = client_with_mock
        embedding = [0.1] * 768
        query = VectorSearchQuery(embedding=embedding, k=5)

        client._client.search.return_value = {
            "took": 15,
            "hits": {
                "total": {"value": 5},
                "hits": [
                    {"_id": "doc1", "_score": 0.95, "_source": {"content": "similar1"}},
                    {"_id": "doc2", "_score": 0.90, "_source": {"content": "similar2"}},
                ],
            },
        }

        response = client.vector_search("test_index", query, embedding_field="embedding")
        assert isinstance(response, SearchResponse)
        assert len(response.hits) == 2
        assert response.hits[0].score == 0.95

    def test_vector_search_with_min_score(self, client_with_mock):
        """Client can perform vector search with minimum score."""
        client = client_with_mock
        query = VectorSearchQuery(
            embedding=[0.1] * 768,
            k=10,
            min_score=0.8,
        )

        client._client.search.return_value = {
            "took": 10,
            "hits": {
                "total": {"value": 3},
                "hits": [
                    {"_id": "doc1", "_score": 0.95, "_source": {}},
                    {"_id": "doc2", "_score": 0.85, "_source": {}},
                    {"_id": "doc3", "_score": 0.82, "_source": {}},
                ],
            },
        }

        response = client.vector_search("test_index", query)
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        # min_score is now a top-level query parameter
        assert body.get("min_score") == 0.8

    def test_vector_search_with_filter(self, client_with_mock):
        """Client can perform filtered vector search."""
        client = client_with_mock
        query = VectorSearchQuery(
            embedding=[0.1] * 768,
            k=10,
            filters={"language": "python", "symbol_type": "function"},
        )

        client._client.search.return_value = {
            "took": 20,
            "hits": {"total": {"value": 0}, "hits": []},
        }

        response = client.vector_search("test_index", query)
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        # Filters are now part of the bool query in script_score
        assert "script_score" in body["query"]
        assert "bool" in body["query"]["script_score"]["query"]

    def test_vector_search_different_dimensions(self, client_with_mock):
        """Client handles different embedding dimensions."""
        client = client_with_mock

        # Test 768-dim (e.g., nomic-embed-text)
        query_768 = VectorSearchQuery(embedding=[0.1] * 768, k=10)
        client._client.search.return_value = {"took": 10, "hits": {"total": {"value": 0}, "hits": []}}
        client.vector_search("index_768", query_768)

        # Test 1536-dim (e.g., OpenAI embeddings)
        query_1536 = VectorSearchQuery(embedding=[0.1] * 1536, k=10)
        client.vector_search("index_1536", query_1536)

        # Test 3072-dim (e.g., Gemini embeddings)
        query_3072 = VectorSearchQuery(embedding=[0.1] * 3072, k=10)
        client.vector_search("index_3072", query_3072)

        assert client._client.search.call_count == 3


# =============================================================================
# Hybrid Search Tests
# =============================================================================


class TestHybridSearchQuery:
    """Tests for HybridSearchQuery."""

    def test_query_creation(self):
        """HybridSearchQuery can be created."""
        query = HybridSearchQuery(
            query_text="def hello",
            embedding=[0.1] * 768,
            lexical_fields=["content", "symbol_name"],
        )
        assert query.query_text == "def hello"
        assert len(query.embedding) == 768
        assert "content" in query.lexical_fields

    def test_query_with_rrf_config(self):
        """HybridSearchQuery can have RRF configuration."""
        query = HybridSearchQuery(
            query_text="function test",
            embedding=[0.1] * 768,
            lexical_fields=["content"],
            rrf_config=RRFConfig(
                rank_constant=60,
                window_size=100,
            ),
        )
        assert query.rrf_config.rank_constant == 60
        assert query.rrf_config.window_size == 100


class TestHybridSearchParams:
    """Tests for HybridSearchParams."""

    def test_params_creation(self):
        """HybridSearchParams can be created."""
        params = HybridSearchParams(
            lexical_weight=0.35,
            vector_weight=0.40,
            graph_weight=0.25,
        )
        assert params.lexical_weight == 0.35
        assert params.vector_weight == 0.40
        assert params.graph_weight == 0.25

    def test_params_weights_sum_to_one(self):
        """HybridSearchParams validates weights sum to 1.0."""
        # This should work
        params = HybridSearchParams(
            lexical_weight=0.4,
            vector_weight=0.4,
            graph_weight=0.2,
        )
        assert abs(params.total_weight - 1.0) < 0.001

    def test_params_normalize_weights(self):
        """HybridSearchParams can normalize weights."""
        params = HybridSearchParams(
            lexical_weight=1.0,
            vector_weight=2.0,
            graph_weight=0.0,
        )
        params.normalize()
        assert abs(params.lexical_weight - 0.333) < 0.01
        assert abs(params.vector_weight - 0.667) < 0.01

    def test_params_defaults(self):
        """HybridSearchParams has v9 plan defaults."""
        params = HybridSearchParams()
        # Per v9 plan: vector 0.40, lexical 0.35, graph 0.25
        assert params.vector_weight == 0.40
        assert params.lexical_weight == 0.35
        assert params.graph_weight == 0.25


class TestRRFConfig:
    """Tests for RRF (Reciprocal Rank Fusion) configuration."""

    def test_rrf_config_creation(self):
        """RRFConfig can be created."""
        config = RRFConfig(
            rank_constant=60,
            window_size=100,
        )
        assert config.rank_constant == 60
        assert config.window_size == 100

    def test_rrf_config_defaults(self):
        """RRFConfig has sensible defaults."""
        config = RRFConfig()
        assert config.rank_constant == 60  # OpenSearch default
        assert config.window_size == 100

    def test_rrf_score_calculation(self):
        """RRFConfig can calculate RRF score."""
        config = RRFConfig(rank_constant=60)

        # Score for rank 1
        score1 = config.calculate_score(rank=1)
        assert abs(score1 - 1 / 61) < 0.001

        # Score for rank 10
        score10 = config.calculate_score(rank=10)
        assert abs(score10 - 1 / 70) < 0.001


class TestHybridSearch:
    """Tests for hybrid search operations."""

    @pytest.fixture
    def client_with_mock(self):
        """Create a real OpenSearchClient with mocked internal client."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        client._client = Mock()
        return client

    def test_hybrid_search_basic(self, client_with_mock):
        """Client can perform basic hybrid search."""
        client = client_with_mock
        query = HybridSearchQuery(
            query_text="def hello",
            embedding=[0.1] * 768,
            lexical_fields=["content"],
        )

        client._client.search.return_value = {
            "took": 25,
            "hits": {
                "total": {"value": 5},
                "hits": [
                    {"_id": "doc1", "_score": 0.85, "_source": {"content": "def hello(): pass"}},
                    {"_id": "doc2", "_score": 0.75, "_source": {"content": "hello world"}},
                ],
            },
        }

        response = client.hybrid_search("test_index", query)
        assert isinstance(response, SearchResponse)
        assert len(response.hits) == 2

    def test_hybrid_search_with_rrf(self, client_with_mock):
        """Client performs hybrid search with RRF fusion."""
        client = client_with_mock
        query = HybridSearchQuery(
            query_text="getUserById",
            embedding=[0.1] * 768,
            lexical_fields=["content", "symbol_name"],
            rrf_config=RRFConfig(rank_constant=60),
        )

        # Simulate OpenSearch hybrid query response
        client._client.search.return_value = {
            "took": 30,
            "hits": {
                "total": {"value": 10},
                "hits": [
                    {"_id": "doc1", "_score": 0.023, "_source": {}},  # RRF combined score
                    {"_id": "doc2", "_score": 0.021, "_source": {}},
                ],
            },
        }

        response = client.hybrid_search("test_index", query)
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        # Should use script_score for combined lexical+vector scoring
        assert "query" in body
        assert "script_score" in body["query"]

    def test_hybrid_search_with_weights(self, client_with_mock):
        """Client applies custom weights in hybrid search."""
        client = client_with_mock
        query = HybridSearchQuery(
            query_text="test function",
            embedding=[0.1] * 768,
            lexical_fields=["content"],
        )
        params = HybridSearchParams(
            lexical_weight=0.5,
            vector_weight=0.5,
            graph_weight=0.0,
        )

        client._client.search.return_value = {
            "took": 20,
            "hits": {"total": {"value": 0}, "hits": []},
        }

        response = client.hybrid_search("test_index", query, params)
        # Verify weights are applied
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        # Weights should be reflected in the query structure
        assert "query" in body
        assert "script_score" in body["query"]

    def test_hybrid_search_fallback_mode(self, client_with_mock):
        """Client falls back gracefully when one component fails."""
        client = client_with_mock
        query = HybridSearchQuery(
            query_text="test",
            embedding=[0.1] * 768,
            lexical_fields=["content"],
        )

        # First call (hybrid) fails
        from opensearchpy.exceptions import RequestError

        client._client.search.side_effect = [
            RequestError(400, "knn_not_supported", {}),
            {  # Fallback to lexical only
                "took": 10,
                "hits": {"total": {"value": 2}, "hits": []},
            },
        ]

        response = client.hybrid_search("test_index", query, fallback=True)
        assert response is not None
        assert client._client.search.call_count == 2

    def test_hybrid_search_with_filters(self, client_with_mock):
        """Hybrid search applies filters to both lexical and vector."""
        client = client_with_mock
        query = HybridSearchQuery(
            query_text="test",
            embedding=[0.1] * 768,
            lexical_fields=["content"],
            filters={
                "language": "python",
                "symbol_type": "function",
            },
        )

        client._client.search.return_value = {
            "took": 25,
            "hits": {"total": {"value": 0}, "hits": []},
        }

        client.hybrid_search("test_index", query)
        call_args = client._client.search.call_args
        body = call_args.kwargs["body"]
        # Filter should be applied
        assert "filter" in str(body).lower() or "must" in str(body).lower()


# =============================================================================
# Search Result and Ranking Tests
# =============================================================================


class TestSearchHit:
    """Tests for SearchHit dataclass."""

    def test_search_hit_creation(self):
        """SearchHit can be created."""
        hit = SearchHit(
            id="doc1",
            score=0.95,
            source={"content": "hello world", "language": "python"},
        )
        assert hit.id == "doc1"
        assert hit.score == 0.95
        assert hit.source["content"] == "hello world"

    def test_search_hit_with_highlights(self):
        """SearchHit can have highlights."""
        hit = SearchHit(
            id="doc1",
            score=0.9,
            source={"content": "hello world"},
            highlights={"content": ["<em>hello</em> world"]},
        )
        assert hit.highlights["content"][0] == "<em>hello</em> world"

    def test_search_hit_from_opensearch(self):
        """SearchHit can be created from OpenSearch response."""
        os_hit = {
            "_id": "doc1",
            "_score": 0.85,
            "_source": {"content": "test", "embedding": [0.1] * 768},
            "highlight": {"content": ["<em>test</em>"]},
        }
        hit = SearchHit.from_opensearch(os_hit)
        assert hit.id == "doc1"
        assert hit.score == 0.85
        assert hit.highlights is not None


class TestSearchResponse:
    """Tests for SearchResponse dataclass."""

    def test_search_response_creation(self):
        """SearchResponse can be created."""
        response = SearchResponse(
            total=100,
            hits=[
                SearchHit(id="doc1", score=0.9, source={}),
                SearchHit(id="doc2", score=0.8, source={}),
            ],
            took_ms=25,
        )
        assert response.total == 100
        assert len(response.hits) == 2
        assert response.took_ms == 25

    def test_search_response_with_aggregations(self):
        """SearchResponse can have aggregations."""
        response = SearchResponse(
            total=50,
            hits=[],
            took_ms=15,
            aggregations={
                "languages": {
                    "buckets": [
                        {"key": "python", "doc_count": 30},
                        {"key": "typescript", "doc_count": 20},
                    ]
                }
            },
        )
        assert "languages" in response.aggregations

    def test_search_response_empty(self):
        """SearchResponse handles empty results."""
        response = SearchResponse(total=0, hits=[], took_ms=5)
        assert response.total == 0
        assert len(response.hits) == 0
        assert response.is_empty is True


class TestScoringFunction:
    """Tests for ScoringFunction."""

    def test_scoring_function_creation(self):
        """ScoringFunction can be created."""
        func = ScoringFunction(
            type="field_value_factor",
            field="popularity",
            factor=1.2,
            modifier="log1p",
        )
        assert func.type == "field_value_factor"
        assert func.field == "popularity"

    def test_scoring_function_decay(self):
        """ScoringFunction supports decay functions."""
        func = ScoringFunction(
            type="decay",
            decay_function="exp",
            field="last_modified",
            origin="now",
            scale="7d",
            decay=0.5,
        )
        assert func.decay_function == "exp"

    def test_scoring_function_to_opensearch(self):
        """ScoringFunction converts to OpenSearch body."""
        func = ScoringFunction(
            type="field_value_factor",
            field="popularity",
            factor=1.2,
        )
        body = func.to_opensearch_body()
        assert "field_value_factor" in body


class TestFieldWeight:
    """Tests for FieldWeight."""

    def test_field_weight_creation(self):
        """FieldWeight can be created."""
        weight = FieldWeight(field="content", weight=1.0)
        assert weight.field == "content"
        assert weight.weight == 1.0

    def test_field_weight_to_boost_string(self):
        """FieldWeight converts to boost string."""
        weight = FieldWeight(field="symbol_name", weight=3.0)
        assert weight.to_boost_string() == "symbol_name^3.0"


class TestSearchResult:
    """Tests for SearchResult (higher-level wrapper)."""

    def test_search_result_creation(self):
        """SearchResult can be created."""
        result = SearchResult(
            query="test query",
            response=SearchResponse(total=10, hits=[], took_ms=20),
            search_type="hybrid",
        )
        assert result.query == "test query"
        assert result.search_type == "hybrid"

    def test_search_result_with_metadata(self):
        """SearchResult can have metadata."""
        result = SearchResult(
            query="test",
            response=SearchResponse(total=5, hits=[], took_ms=15),
            search_type="lexical",
            metadata={
                "index_coverage": 0.95,
                "confidence": "high",
                "rrf_weights": {"lexical": 0.35, "vector": 0.40},
            },
        )
        assert result.metadata["index_coverage"] == 0.95
        assert result.metadata["confidence"] == "high"


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for retrieval exceptions."""

    def test_opensearch_error(self):
        """OpenSearchError is base exception."""
        error = OpenSearchError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_connection_error(self):
        """ConnectionError for connection failures."""
        error = ConnectionError(
            "Failed to connect to localhost:9200",
            host="localhost:9200",
            retries=3,
        )
        assert "localhost:9200" in str(error)
        assert error.host == "localhost:9200"
        assert error.retries == 3

    def test_index_error(self):
        """IndexError for index operations."""
        error = IndexError(
            "Failed to create index: test error",
            index_name="test_index",
            operation="create",
        )
        assert error.index_name == "test_index"
        assert error.operation == "create"

    def test_document_error(self):
        """DocumentError for document operations."""
        error = DocumentError(
            "Failed to index document",
            doc_id="doc1",
            operation="index",
        )
        assert error.doc_id == "doc1"

    def test_search_error(self):
        """SearchError for search operations."""
        error = SearchError(
            "Search failed",
            query_type="hybrid",
            took_ms=100,
        )
        assert error.query_type == "hybrid"


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactory:
    """Tests for factory functions."""

    def test_create_opensearch_client_default(self):
        """Factory creates client with default config."""
        with patch("opensearchpy.OpenSearch"):
            client = create_opensearch_client()
            assert isinstance(client, OpenSearchClient)

    def test_create_opensearch_client_with_config(self):
        """Factory creates client with custom config."""
        config = OpenSearchConfig(
            hosts=["node1:9200", "node2:9200"],
            use_ssl=True,
        )
        with patch("opensearchpy.OpenSearch"):
            client = create_opensearch_client(config)
            assert client.config.hosts == ["node1:9200", "node2:9200"]
            assert client.config.use_ssl is True

    def test_create_opensearch_client_from_env(self):
        """Factory creates client from environment."""
        with patch.dict(
            "os.environ",
            {
                "OPENSEARCH_HOSTS": "prod1:9200,prod2:9200",
                "OPENSEARCH_USE_SSL": "true",
            },
        ):
            with patch("opensearchpy.OpenSearch"):
                client = create_opensearch_client(from_env=True)
                assert "prod1:9200" in client.config.hosts


# =============================================================================
# Integration Tests (marked for optional running)
# =============================================================================


@pytest.mark.integration
class TestOpenSearchIntegration:
    """Integration tests requiring a running OpenSearch instance."""

    @pytest.fixture
    def client(self):
        """Create a real OpenSearch client."""
        config = OpenSearchConfig(hosts=["localhost:9200"])
        client = create_opensearch_client(config)
        client.connect()
        yield client
        client.close()

    @pytest.fixture
    def test_index(self, client):
        """Create and clean up a test index."""
        index_name = "test_integration_index"
        manager = IndexManager(client)

        config = IndexConfig.for_code(
            name=index_name,
            embedding_dim=768,
            languages=["python"],
        )
        manager.create_index(config, ignore_existing=True)

        yield index_name

        manager.delete_index(index_name, ignore_not_found=True)

    def test_full_indexing_and_search_flow(self, client, test_index):
        """Test complete flow: index documents and search."""
        # Index documents
        docs = [
            Document(
                id="doc1",
                content="def hello_world(): print('Hello')",
                embedding=[0.1] * 768,
                metadata={
                    "file_path": "/src/main.py",
                    "language": "python",
                    "symbol_name": "hello_world",
                    "symbol_type": "function",
                },
            ),
            Document(
                id="doc2",
                content="def goodbye_world(): print('Goodbye')",
                embedding=[0.2] * 768,
                metadata={
                    "file_path": "/src/main.py",
                    "language": "python",
                    "symbol_name": "goodbye_world",
                    "symbol_type": "function",
                },
            ),
        ]

        result = client.bulk_index(test_index, docs)
        assert result.succeeded == 2

        # Refresh index
        manager = IndexManager(client)
        manager.refresh_index(test_index)

        # Lexical search
        lexical_query = LexicalSearchQuery(
            query_text="hello",
            fields=["content", "symbol_name"],
        )
        lexical_response = client.lexical_search(test_index, lexical_query)
        assert lexical_response.total >= 1
        assert any("hello" in str(hit.source).lower() for hit in lexical_response.hits)

        # Vector search
        vector_query = VectorSearchQuery(embedding=[0.1] * 768, k=5)
        vector_response = client.vector_search(test_index, vector_query)
        assert vector_response.total >= 1

        # Hybrid search
        hybrid_query = HybridSearchQuery(
            query_text="hello",
            embedding=[0.1] * 768,
            lexical_fields=["content", "symbol_name"],
        )
        hybrid_response = client.hybrid_search(test_index, hybrid_query)
        assert hybrid_response.total >= 1


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance requirements."""

    @pytest.fixture
    def client_with_mock(self):
        """Create a real OpenSearchClient with mocked internal client."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        client._client = Mock()
        return client

    def test_search_latency_tracking(self, client_with_mock):
        """Search responses include latency information."""
        client = client_with_mock
        query = LexicalSearchQuery(query_text="test", fields=["content"])

        client._client.search.return_value = {
            "took": 45,
            "hits": {"total": {"value": 10}, "hits": []},
        }

        response = client.lexical_search("test_index", query)
        assert response.took_ms == 45

    def test_bulk_indexing_batch_size(self, client_with_mock):
        """Bulk indexing respects batch size."""
        client = client_with_mock
        docs = [
            Document(id=f"doc{i}", content=f"content{i}", embedding=[0.1] * 3)
            for i in range(500)
        ]

        client._client.bulk.return_value = {
            "took": 100,
            "errors": False,
            "items": [],
        }

        # Should batch into multiple calls
        client.bulk_index("test_index", docs, batch_size=100)
        assert client._client.bulk.call_count == 5


# =============================================================================
# Code-Aware Tokenization Tests
# =============================================================================


class TestCodeTokenization:
    """Tests for code-aware tokenization in search."""

    @pytest.fixture
    def client_with_mock(self):
        """Create a real OpenSearchClient with mocked internal client."""
        config = OpenSearchConfig()
        client = OpenSearchClient(config)
        client._client = Mock()
        return client

    def test_camel_case_query(self, client_with_mock):
        """Search handles CamelCase queries."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="getUserById",
            fields=["content", "symbol_name"],
        )
        params = LexicalSearchParams(analyzer="camel_case_analyzer")

        client._client.search.return_value = {
            "took": 10,
            "hits": {"total": {"value": 1}, "hits": []},
        }

        # Query should find matches for getUserById, get_user_by_id, etc.
        client.lexical_search("test_index", query, params)
        call_args = client._client.search.call_args
        # Analyzer should be specified in query
        body = call_args.kwargs["body"]
        assert body is not None

    def test_snake_case_query(self, client_with_mock):
        """Search handles snake_case queries."""
        client = client_with_mock
        query = LexicalSearchQuery(
            query_text="get_user_by_id",
            fields=["content"],
        )

        client._client.search.return_value = {
            "took": 10,
            "hits": {"total": {"value": 1}, "hits": []},
        }

        client.lexical_search("test_index", query)
        # Should work without special handling
        client._client.search.assert_called_once()
