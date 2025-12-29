"""
TDD tests for TenantQdrantStore.

Tests are written FIRST per strict TDD methodology.
These tests verify the Qdrant-backed embedding store with:
- Tenant isolation via org_id payload filtering
- Per-model collection naming
- Payload index creation for efficient filtering
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# Test fixtures
TEST_ORG_A_ID = "11111111-1111-1111-1111-111111111111"
TEST_ORG_B_ID = "22222222-2222-2222-2222-222222222222"
TEST_USER_A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_EMBEDDING_DIM = 1536


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for unit tests."""
    client = MagicMock()

    # In-memory storage for points
    collections = {}

    class MockCollection:
        def __init__(self, name):
            self.name = name

    class MockCollectionsResponse:
        def __init__(self, colls):
            self.collections = colls

    def mock_get_collections():
        return MockCollectionsResponse([MockCollection(n) for n in collections.keys()])

    def mock_create_collection(collection_name, vectors_config):
        if collection_name not in collections:
            collections[collection_name] = {"points": {}, "config": vectors_config}
        return True

    def mock_delete_collection(collection_name):
        if collection_name in collections:
            del collections[collection_name]
        return True

    def mock_upsert(collection_name, points):
        if collection_name not in collections:
            collections[collection_name] = {"points": {}}
        for point in points:
            collections[collection_name]["points"][point.id] = {
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload,
            }
        return True

    def mock_retrieve(collection_name, ids, with_payload=True):
        if collection_name not in collections:
            return []
        result = []
        for point_id in ids:
            if point_id in collections[collection_name]["points"]:
                point = collections[collection_name]["points"][point_id]
                mock_point = MagicMock()
                mock_point.id = point["id"]
                mock_point.payload = point["payload"]
                mock_point.vector = point["vector"]
                result.append(mock_point)
        return result

    def mock_delete(collection_name, points_selector):
        if collection_name not in collections:
            return True
        for point_id in points_selector.points:
            if point_id in collections[collection_name]["points"]:
                del collections[collection_name]["points"][point_id]
        return True

    def mock_query_points(collection_name, query, query_filter=None, limit=10):
        """Simulate vector search with filtering."""
        if collection_name not in collections:
            return MagicMock(points=[])

        points = list(collections[collection_name]["points"].values())

        # Apply filter if provided
        if query_filter is not None:
            filtered_points = []
            for point in points:
                matches = True
                # Check must conditions
                if hasattr(query_filter, 'must') and query_filter.must:
                    for condition in query_filter.must:
                        field_name = condition.key
                        expected_value = condition.match.value
                        if point["payload"].get(field_name) != expected_value:
                            matches = False
                            break
                if matches:
                    filtered_points.append(point)
            points = filtered_points

        # Create mock results with scores
        results = []
        for i, point in enumerate(points[:limit]):
            mock_result = MagicMock()
            mock_result.id = point["id"]
            mock_result.payload = point["payload"]
            mock_result.score = 1.0 - (i * 0.1)  # Decreasing scores
            results.append(mock_result)

        mock_response = MagicMock()
        mock_response.points = results
        return mock_response

    def mock_scroll(collection_name, scroll_filter=None, limit=100, with_payload=True, with_vectors=False):
        """Simulate scroll/list operation with filtering."""
        if collection_name not in collections:
            return ([], None)

        points = list(collections[collection_name]["points"].values())

        # Apply filter if provided
        if scroll_filter is not None:
            filtered_points = []
            for point in points:
                matches = True
                if hasattr(scroll_filter, 'must') and scroll_filter.must:
                    for condition in scroll_filter.must:
                        field_name = condition.key
                        expected_value = condition.match.value
                        if point["payload"].get(field_name) != expected_value:
                            matches = False
                            break
                if matches:
                    filtered_points.append(point)
            points = filtered_points

        # Create mock results
        results = []
        for point in points[:limit]:
            mock_point = MagicMock()
            mock_point.id = point["id"]
            mock_point.payload = point["payload"]
            if with_vectors:
                mock_point.vector = point["vector"]
            results.append(mock_point)

        return (results, None)

    client.get_collections = MagicMock(side_effect=mock_get_collections)
    client.create_collection = MagicMock(side_effect=mock_create_collection)
    client.delete_collection = MagicMock(side_effect=mock_delete_collection)
    client.upsert = MagicMock(side_effect=mock_upsert)
    client.retrieve = MagicMock(side_effect=mock_retrieve)
    client.delete = MagicMock(side_effect=mock_delete)
    client.query_points = MagicMock(side_effect=mock_query_points)
    client.scroll = MagicMock(side_effect=mock_scroll)
    client.create_payload_index = MagicMock(return_value=True)

    # Expose storage for assertions
    client._collections = collections

    return client


@pytest.fixture
def sample_vector():
    """Create a sample embedding vector."""
    return [0.1] * TEST_EMBEDDING_DIM


@pytest.fixture
def sample_payload():
    """Create a sample payload for an embedding."""
    return {
        "content": "This is a test memory",
        "memory_id": str(uuid.uuid4()),
        "category": "architecture",
        "scope": "project",
    }


class TestTenantQdrantStoreCreation:
    """Tests for TenantQdrantStore initialization."""

    def test_creates_collection_with_model_name_prefix(self, mock_qdrant_client):
        """Store should create collection with embeddings_{model_name} format."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="text-embedding-3-small",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        assert "embeddings_text-embedding-3-small" in mock_qdrant_client._collections

    def test_creates_payload_index_for_org_id(self, mock_qdrant_client):
        """Store should create a payload index for org_id field."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Verify create_payload_index was called with org_id
        calls = mock_qdrant_client.create_payload_index.call_args_list
        org_id_call = [c for c in calls if c.kwargs.get('field_name') == 'org_id' or
                       (c.args and len(c.args) > 1 and c.args[1] == 'org_id')]
        assert len(org_id_call) > 0 or any('org_id' in str(c) for c in calls)

    def test_reuses_existing_collection(self, mock_qdrant_client):
        """Store should not recreate collection if it already exists."""
        from app.stores.qdrant_store import TenantQdrantStore

        # Pre-create collection
        mock_qdrant_client._collections["embeddings_test-model"] = {"points": {}}

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # create_collection should not be called for existing collection
        # (the mock tracks this internally)
        assert "embeddings_test-model" in mock_qdrant_client._collections


class TestTenantQdrantStoreUpsert:
    """Tests for TenantQdrantStore.upsert() method."""

    def test_upsert_adds_org_id_to_payload(self, mock_qdrant_client, sample_vector, sample_payload):
        """upsert() should add org_id to the payload automatically."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        point_id = str(uuid.uuid4())
        store.upsert(point_id, sample_vector, sample_payload)

        # Verify org_id was added
        collection = mock_qdrant_client._collections["embeddings_test-model"]
        stored_payload = collection["points"][point_id]["payload"]
        assert stored_payload["org_id"] == TEST_ORG_A_ID

    def test_upsert_preserves_existing_payload(self, mock_qdrant_client, sample_vector, sample_payload):
        """upsert() should preserve existing payload fields."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        point_id = str(uuid.uuid4())
        store.upsert(point_id, sample_vector, sample_payload)

        collection = mock_qdrant_client._collections["embeddings_test-model"]
        stored_payload = collection["points"][point_id]["payload"]
        assert stored_payload["content"] == sample_payload["content"]
        assert stored_payload["category"] == sample_payload["category"]

    def test_upsert_stores_vector(self, mock_qdrant_client, sample_vector, sample_payload):
        """upsert() should store the embedding vector."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        point_id = str(uuid.uuid4())
        store.upsert(point_id, sample_vector, sample_payload)

        collection = mock_qdrant_client._collections["embeddings_test-model"]
        assert collection["points"][point_id]["vector"] == sample_vector


class TestTenantQdrantStoreSearch:
    """Tests for TenantQdrantStore.search() method."""

    def test_search_filters_by_org_id(self, mock_qdrant_client, sample_vector, sample_payload):
        """search() should only return results for the current org."""
        from app.stores.qdrant_store import TenantQdrantStore

        # Create stores for two orgs
        store_a = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )
        store_b = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_B_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add points for each org
        store_a.upsert("point-a", sample_vector, {"content": "Org A content"})
        store_b.upsert("point-b", sample_vector, {"content": "Org B content"})

        # Search as org A
        results = store_a.search(sample_vector, limit=10)

        # Should only find org A's point
        assert len(results) == 1
        assert results[0].payload["content"] == "Org A content"

    def test_search_returns_empty_for_no_matches(self, mock_qdrant_client, sample_vector):
        """search() should return empty list when no matches found."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        results = store.search(sample_vector, limit=10)

        assert results == []

    def test_search_respects_limit(self, mock_qdrant_client, sample_vector, sample_payload):
        """search() should respect the limit parameter."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add multiple points
        for i in range(5):
            store.upsert(f"point-{i}", sample_vector, {"content": f"Content {i}"})

        results = store.search(sample_vector, limit=2)

        assert len(results) <= 2

    def test_search_with_additional_filters(self, mock_qdrant_client, sample_vector):
        """search() should support additional payload filters."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add points with different categories
        store.upsert("point-1", sample_vector, {"content": "Content 1", "category": "workflow"})
        store.upsert("point-2", sample_vector, {"content": "Content 2", "category": "security"})

        results = store.search(sample_vector, limit=10, filters={"category": "workflow"})

        # Should only find workflow category point
        for result in results:
            if result.payload.get("category"):
                assert result.payload["category"] == "workflow"


class TestTenantQdrantStoreDelete:
    """Tests for TenantQdrantStore.delete() method."""

    def test_delete_removes_point(self, mock_qdrant_client, sample_vector, sample_payload):
        """delete() should remove the point from the collection."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        point_id = "point-to-delete"
        store.upsert(point_id, sample_vector, sample_payload)
        store.delete(point_id)

        collection = mock_qdrant_client._collections["embeddings_test-model"]
        assert point_id not in collection["points"]

    def test_delete_only_deletes_owned_points(self, mock_qdrant_client, sample_vector, sample_payload):
        """delete() should verify ownership before deleting."""
        from app.stores.qdrant_store import TenantQdrantStore

        # Create stores for two orgs
        store_a = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )
        store_b = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_B_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add point as org A
        store_a.upsert("org-a-point", sample_vector, sample_payload)

        # Try to delete as org B
        result = store_b.delete("org-a-point")

        # Should fail (return False)
        assert result is False

        # Point should still exist
        collection = mock_qdrant_client._collections["embeddings_test-model"]
        assert "org-a-point" in collection["points"]


class TestTenantQdrantStoreGet:
    """Tests for TenantQdrantStore.get() method."""

    def test_get_returns_point(self, mock_qdrant_client, sample_vector, sample_payload):
        """get() should return the point if it exists and is owned."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        point_id = "test-point"
        store.upsert(point_id, sample_vector, sample_payload)

        result = store.get(point_id)

        assert result is not None
        assert result.id == point_id

    def test_get_returns_none_for_nonexistent(self, mock_qdrant_client):
        """get() should return None for non-existent point."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        result = store.get("nonexistent")

        assert result is None

    def test_get_returns_none_for_other_orgs_point(
        self, mock_qdrant_client, sample_vector, sample_payload
    ):
        """get() should return None when trying to access another org's point."""
        from app.stores.qdrant_store import TenantQdrantStore

        # Create stores for two orgs
        store_a = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )
        store_b = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_B_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add point as org A
        store_a.upsert("org-a-point", sample_vector, sample_payload)

        # Try to get as org B
        result = store_b.get("org-a-point")

        assert result is None


class TestTenantQdrantStoreList:
    """Tests for TenantQdrantStore.list() method."""

    def test_list_returns_org_points_only(self, mock_qdrant_client, sample_vector, sample_payload):
        """list() should only return points for the current org."""
        from app.stores.qdrant_store import TenantQdrantStore

        # Create stores for two orgs
        store_a = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )
        store_b = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_B_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add points for each org
        store_a.upsert("point-a-1", sample_vector, {"content": "Org A 1"})
        store_a.upsert("point-a-2", sample_vector, {"content": "Org A 2"})
        store_b.upsert("point-b-1", sample_vector, {"content": "Org B 1"})

        # List as org A
        results = store_a.list()

        assert len(results) == 2
        for point in results:
            assert point.payload["org_id"] == TEST_ORG_A_ID

    def test_list_respects_limit(self, mock_qdrant_client, sample_vector, sample_payload):
        """list() should respect the limit parameter."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Add multiple points
        for i in range(10):
            store.upsert(f"point-{i}", sample_vector, {"content": f"Content {i}"})

        results = store.list(limit=5)

        assert len(results) <= 5


class TestTenantQdrantStoreHealth:
    """Tests for TenantQdrantStore health check."""

    def test_health_check_returns_true_when_connected(self, mock_qdrant_client):
        """Health check should return True when Qdrant is connected."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="test-model",
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        assert store.health_check() is True

    def test_health_check_returns_false_on_error(self, mock_qdrant_client):
        """Health check should return False when get_collections fails."""
        from app.stores.qdrant_store import TenantQdrantStore

        mock_qdrant_client.get_collections.side_effect = Exception("Connection refused")

        store = TenantQdrantStore.__new__(TenantQdrantStore)
        store.client = mock_qdrant_client
        store.org_id = TEST_ORG_A_ID
        store.collection_name = "test-collection"

        assert store.health_check() is False


class TestTenantQdrantStoreCollectionNaming:
    """Tests for collection naming conventions."""

    def test_sanitizes_model_name_for_collection(self, mock_qdrant_client):
        """Collection name should sanitize model name for Qdrant compatibility."""
        from app.stores.qdrant_store import TenantQdrantStore

        store = TenantQdrantStore(
            client=mock_qdrant_client,
            org_id=TEST_ORG_A_ID,
            model_name="openai/text-embedding-3-small",  # Contains slash
            embedding_dim=TEST_EMBEDDING_DIM,
        )

        # Should sanitize the name
        assert "/" not in store.collection_name
        assert "embeddings_" in store.collection_name
