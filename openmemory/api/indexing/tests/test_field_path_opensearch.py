"""Tests for indexing field path nodes into OpenSearch."""

from pathlib import Path
from unittest.mock import MagicMock

from openmemory.api.indexing.code_indexer import CodeIndexingService
from openmemory.api.indexing.graph_projection import MemoryGraphStore
from openmemory.api.retrieval.opensearch import BulkResult


def _collect_bulk_documents(opensearch_client: MagicMock) -> list:
    documents = []
    for call in opensearch_client.bulk_index.call_args_list:
        docs = call.kwargs.get("documents")
        if docs is None and len(call.args) > 1:
            docs = call.args[1]
        if docs:
            documents.extend(docs)
    return documents


def test_field_path_indexed_in_opensearch(tmp_path: Path) -> None:
    store = MemoryGraphStore()
    opensearch_client = MagicMock()
    opensearch_client._client = MagicMock()
    opensearch_client._client.indices.create.return_value = None
    opensearch_client._client.indices.refresh.return_value = None
    opensearch_client.bulk_index.return_value = BulkResult(
        total=0,
        succeeded=0,
        failed=0,
    )

    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "paths.ts").write_text(
        'const value = get(obj, "order.items[0].price");'
    )

    indexer = CodeIndexingService(
        root_path=tmp_path,
        repo_id="test-repo",
        graph_driver=store,
        opensearch_client=opensearch_client,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts"],
    )

    indexer.index_repository()

    documents = _collect_bulk_documents(opensearch_client)

    field_path_doc = next(
        (doc for doc in documents if doc.metadata.get("symbol_type") == "field_path"),
        None,
    )
    assert field_path_doc is not None
    assert field_path_doc.metadata.get("symbol_name") == "price"
    assert field_path_doc.metadata.get("confidence") is not None
