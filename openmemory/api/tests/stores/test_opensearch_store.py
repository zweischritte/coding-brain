"""
TDD tests for TenantOpenSearchStore.

Tests are written FIRST per strict TDD methodology.
These tests verify the OpenSearch-backed store with:
- Tenant alias strategy for index isolation
- Org-scoped queries via alias routing
- Hybrid search (lexical + vector) with tenant filtering
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest


# Test fixtures
TEST_ORG_A_ID = "11111111-1111-1111-1111-111111111111"
TEST_ORG_B_ID = "22222222-2222-2222-2222-222222222222"
TEST_USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_EMBEDDING_DIM = 1536


@pytest.fixture
def mock_opensearch_client():
    """Create a mock OpenSearch client for unit tests."""
    client = MagicMock()

    # In-memory storage for documents
    indices = {}
    aliases = {}

    def mock_indices_exists(index):
        return index in indices or index in aliases

    def mock_indices_create(index, body=None):
        if index not in indices:
            indices[index] = {"docs": {}, "settings": body or {}}
            # Handle aliases in body
            if body and "aliases" in body:
                for alias_name in body["aliases"]:
                    aliases[alias_name] = index
        return {"acknowledged": True}

    def mock_indices_delete(index):
        if index in indices:
            # Remove aliases pointing to this index
            to_remove = [a for a, i in aliases.items() if i == index]
            for a in to_remove:
                del aliases[a]
            del indices[index]
        return {"acknowledged": True}

    def mock_indices_get_alias(index=None, name=None):
        result = {}
        for alias, idx in aliases.items():
            if name is None or alias == name:
                if idx not in result:
                    result[idx] = {"aliases": {}}
                result[idx]["aliases"][alias] = {}
        return result

    def mock_indices_put_alias(index, name):
        aliases[name] = index
        return {"acknowledged": True}

    def mock_indices_delete_alias(index, name):
        if name in aliases:
            del aliases[name]
        return {"acknowledged": True}

    def mock_index(index, body, id=None, refresh=None):
        # Resolve alias to real index
        real_index = aliases.get(index, index)
        if real_index not in indices:
            indices[real_index] = {"docs": {}, "settings": {}}
        doc_id = id or str(uuid.uuid4())
        indices[real_index]["docs"][doc_id] = body
        return {"_id": doc_id, "result": "created"}

    def mock_get(index, id):
        real_index = aliases.get(index, index)
        if real_index in indices and id in indices[real_index]["docs"]:
            return {"_id": id, "_source": indices[real_index]["docs"][id], "found": True}
        raise Exception("Document not found")

    def mock_delete(index, id, refresh=None):
        real_index = aliases.get(index, index)
        if real_index in indices and id in indices[real_index]["docs"]:
            del indices[real_index]["docs"][id]
            return {"result": "deleted"}
        return {"result": "not_found"}

    def mock_search(index=None, body=None):
        # Resolve alias to real index
        real_index = aliases.get(index, index) if index else None

        results = []
        search_indices = [real_index] if real_index else list(indices.keys())

        for idx in search_indices:
            if idx not in indices:
                continue
            for doc_id, doc in indices[idx]["docs"].items():
                # Apply filter if present
                matches = True
                if body and "query" in body:
                    query = body["query"]
                    # Handle bool query with filter
                    if "bool" in query and "filter" in query["bool"]:
                        for filter_clause in query["bool"]["filter"]:
                            if "term" in filter_clause:
                                for field, value in filter_clause["term"].items():
                                    if doc.get(field) != value:
                                        matches = False
                                        break
                if matches:
                    results.append({
                        "_id": doc_id,
                        "_source": doc,
                        "_score": 1.0
                    })

        return {
            "hits": {
                "total": {"value": len(results)},
                "hits": results[:body.get("size", 10)] if body else results[:10]
            }
        }

    def mock_count(index=None, body=None):
        real_index = aliases.get(index, index) if index else None
        count = 0
        search_indices = [real_index] if real_index else list(indices.keys())
        for idx in search_indices:
            if idx in indices:
                count += len(indices[idx]["docs"])
        return {"count": count}

    # Set up mock methods
    client.indices = MagicMock()
    client.indices.exists = MagicMock(side_effect=mock_indices_exists)
    client.indices.create = MagicMock(side_effect=mock_indices_create)
    client.indices.delete = MagicMock(side_effect=mock_indices_delete)
    client.indices.get_alias = MagicMock(side_effect=mock_indices_get_alias)
    client.indices.put_alias = MagicMock(side_effect=mock_indices_put_alias)
    client.indices.delete_alias = MagicMock(side_effect=mock_indices_delete_alias)

    client.index = MagicMock(side_effect=mock_index)
    client.get = MagicMock(side_effect=mock_get)
    client.delete = MagicMock(side_effect=mock_delete)
    client.search = MagicMock(side_effect=mock_search)
    client.count = MagicMock(side_effect=mock_count)
    client.ping = MagicMock(return_value=True)

    # Expose storage for assertions
    client._indices = indices
    client._aliases = aliases

    return client


@pytest.fixture
def sample_document():
    """Create a sample document for indexing."""
    return {
        "content": "This is a test memory about Python programming",
        "memory_id": str(uuid.uuid4()),
        "vault": "WLT",
        "layer": "cognitive",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return [0.1] * TEST_EMBEDDING_DIM


class TestTenantOpenSearchStoreAliasCreation:
    """Tests for TenantOpenSearchStore alias creation."""

    def test_creates_tenant_alias_on_init(self, mock_opensearch_client):
        """Store should create a tenant-specific alias on initialization."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        expected_alias = f"tenant_{TEST_ORG_A_ID}"
        assert expected_alias in mock_opensearch_client._aliases

    def test_creates_shared_index_for_small_tenants(self, mock_opensearch_client):
        """Small tenants should share a common index via alias."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
            use_dedicated_index=False,
        )

        # Should point to shared index
        assert "memories_shared" in mock_opensearch_client._indices

    def test_creates_dedicated_index_when_configured(self, mock_opensearch_client):
        """Large tenants should get dedicated indices."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
            use_dedicated_index=True,
        )

        expected_index = f"memories_{TEST_ORG_A_ID}"
        assert expected_index in mock_opensearch_client._indices

    def test_reuses_existing_alias(self, mock_opensearch_client):
        """Should reuse existing alias if already created."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Pre-create alias
        alias_name = f"tenant_{TEST_ORG_A_ID}"
        mock_opensearch_client._aliases[alias_name] = "memories_shared"

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        # Should not fail, reuse existing
        assert alias_name in mock_opensearch_client._aliases


class TestTenantOpenSearchStoreIndex:
    """Tests for TenantOpenSearchStore.index() method."""

    def test_index_adds_org_id_to_document(self, mock_opensearch_client, sample_document):
        """index() should add org_id to the document automatically."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        doc_id = str(uuid.uuid4())
        store.index(doc_id, sample_document)

        # Find the document in storage
        for idx in mock_opensearch_client._indices.values():
            if doc_id in idx["docs"]:
                assert idx["docs"][doc_id]["org_id"] == TEST_ORG_A_ID
                break
        else:
            pytest.fail("Document not found in any index")

    def test_index_preserves_existing_fields(self, mock_opensearch_client, sample_document):
        """index() should preserve all existing document fields."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        doc_id = str(uuid.uuid4())
        store.index(doc_id, sample_document)

        # Verify fields preserved
        for idx in mock_opensearch_client._indices.values():
            if doc_id in idx["docs"]:
                doc = idx["docs"][doc_id]
                assert doc["content"] == sample_document["content"]
                assert doc["vault"] == sample_document["vault"]
                break

    def test_index_uses_tenant_alias(self, mock_opensearch_client, sample_document):
        """index() should route through tenant alias."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        doc_id = str(uuid.uuid4())
        store.index(doc_id, sample_document)

        # The mock.index should have been called
        assert mock_opensearch_client.index.called


class TestTenantOpenSearchStoreSearch:
    """Tests for TenantOpenSearchStore.search() method."""

    def test_search_filters_by_org_id(self, mock_opensearch_client, sample_document):
        """search() should only return results for the current org."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create stores for two orgs
        store_a = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )
        store_b = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_B_ID,
            index_prefix="memories",
        )

        # Index documents for each org
        doc_a = sample_document.copy()
        doc_a["content"] = "Org A document"
        store_a.index("doc-a", doc_a)

        doc_b = sample_document.copy()
        doc_b["content"] = "Org B document"
        store_b.index("doc-b", doc_b)

        # Search as org A
        results = store_a.search("document")

        # Should only find org A's document
        assert len(results) == 1
        assert results[0]["_source"]["content"] == "Org A document"

    def test_search_returns_empty_for_no_matches(self, mock_opensearch_client):
        """search() should return empty list when no matches found."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        results = store.search("nonexistent query")

        assert results == []

    def test_search_respects_limit(self, mock_opensearch_client, sample_document):
        """search() should respect the limit parameter."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        # Index multiple documents
        for i in range(5):
            doc = sample_document.copy()
            doc["content"] = f"Document {i}"
            store.index(f"doc-{i}", doc)

        results = store.search("Document", limit=2)

        assert len(results) <= 2

    def test_hybrid_search_with_embedding(self, mock_opensearch_client, sample_document, sample_embedding):
        """search() should support hybrid search with embeddings."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        doc = sample_document.copy()
        doc["embedding"] = sample_embedding
        store.index("doc-1", doc)

        results = store.hybrid_search(
            query_text="test",
            query_vector=sample_embedding,
        )

        assert isinstance(results, list)


class TestTenantOpenSearchStoreGet:
    """Tests for TenantOpenSearchStore.get() method."""

    def test_get_returns_document(self, mock_opensearch_client, sample_document):
        """get() should return the document if it exists and is owned."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        doc_id = "test-doc"
        store.index(doc_id, sample_document)

        result = store.get(doc_id)

        assert result is not None
        assert result["_id"] == doc_id

    def test_get_returns_none_for_nonexistent(self, mock_opensearch_client):
        """get() should return None for non-existent document."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        result = store.get("nonexistent")

        assert result is None

    def test_get_returns_none_for_other_orgs_document(
        self, mock_opensearch_client, sample_document
    ):
        """get() should return None when trying to access another org's document."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create stores for two orgs
        store_a = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )
        store_b = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_B_ID,
            index_prefix="memories",
        )

        # Index as org A
        store_a.index("org-a-doc", sample_document)

        # Try to get as org B
        result = store_b.get("org-a-doc")

        assert result is None


class TestTenantOpenSearchStoreDelete:
    """Tests for TenantOpenSearchStore.delete() method."""

    def test_delete_removes_document(self, mock_opensearch_client, sample_document):
        """delete() should remove the document."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        doc_id = "to-delete"
        store.index(doc_id, sample_document)
        store.delete(doc_id)

        result = store.get(doc_id)
        assert result is None

    def test_delete_only_deletes_owned_documents(self, mock_opensearch_client, sample_document):
        """delete() should verify ownership before deleting."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        # Create stores for two orgs
        store_a = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )
        store_b = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_B_ID,
            index_prefix="memories",
        )

        # Index as org A
        store_a.index("org-a-doc", sample_document)

        # Try to delete as org B
        result = store_b.delete("org-a-doc")

        # Should fail
        assert result is False

        # Document should still exist for org A
        assert store_a.get("org-a-doc") is not None


class TestTenantOpenSearchStoreHealth:
    """Tests for TenantOpenSearchStore health check."""

    def test_health_check_returns_true_when_connected(self, mock_opensearch_client):
        """Health check should return True when OpenSearch is connected."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
        )

        assert store.health_check() is True

    def test_health_check_returns_false_on_error(self, mock_opensearch_client):
        """Health check should return False when ping fails."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        mock_opensearch_client.ping.return_value = False

        store = TenantOpenSearchStore.__new__(TenantOpenSearchStore)
        store.client = mock_opensearch_client
        store.org_id = TEST_ORG_A_ID

        assert store.health_check() is False


class TestTenantOpenSearchStoreAliasStrategies:
    """Tests for alias routing strategies."""

    def test_alias_points_to_correct_index(self, mock_opensearch_client):
        """Alias should point to the correct underlying index."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
            use_dedicated_index=True,
        )

        alias_name = f"tenant_{TEST_ORG_A_ID}"
        expected_index = f"memories_{TEST_ORG_A_ID}"

        assert mock_opensearch_client._aliases.get(alias_name) == expected_index

    def test_multiple_tenants_can_share_index(self, mock_opensearch_client):
        """Multiple small tenants should be able to share an index."""
        from app.stores.opensearch_store import TenantOpenSearchStore

        store_a = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_A_ID,
            index_prefix="memories",
            use_dedicated_index=False,
        )
        store_b = TenantOpenSearchStore(
            client=mock_opensearch_client,
            org_id=TEST_ORG_B_ID,
            index_prefix="memories",
            use_dedicated_index=False,
        )

        # Both should point to shared index
        alias_a = f"tenant_{TEST_ORG_A_ID}"
        alias_b = f"tenant_{TEST_ORG_B_ID}"

        assert mock_opensearch_client._aliases.get(alias_a) == "memories_shared"
        assert mock_opensearch_client._aliases.get(alias_b) == "memories_shared"
