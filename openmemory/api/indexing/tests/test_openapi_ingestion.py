"""Tests for OpenAPI spec ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from openmemory.api.indexing.code_indexer import CodeIndexingService
from openmemory.api.indexing.graph_projection import CodeEdgeType, CodeNodeType, MemoryGraphStore


@pytest.fixture
def indexer(tmp_path: Path) -> tuple[CodeIndexingService, MemoryGraphStore, Path, Path]:
    store = MemoryGraphStore()
    root = tmp_path
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    return (
        CodeIndexingService(
            root_path=root,
            repo_id="test-repo",
            graph_driver=store,
            opensearch_client=None,
            embedding_service=None,
            include_api_boundaries=False,
            extensions=[".ts"],
        ),
        store,
        src,
        root,
    )


def _find_symbol(nodes, name: str, kind: str, parent_name: Optional[str] = None):
    for node in nodes:
        if node.properties.get("name") != name:
            continue
        if node.properties.get("kind") != kind:
            continue
        if parent_name and node.properties.get("parent_name") != parent_name:
            continue
        return node
    return None


def test_openapi_ingestion_creates_nodes_and_edges(indexer):
    indexer, store, src, root = indexer

    code = """
class User {
  name: string;
  age: number;
}
"""
    (src / "user.ts").write_text(code)

    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Demo"},
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer", "nullable": True},
                    },
                }
            }
        },
    }
    (root / "openapi.json").write_text(json.dumps(spec))

    indexer.index_repository()

    openapi_defs = store.query_nodes_by_type(CodeNodeType.OPENAPI_DEF)
    openapi_user = next(
        (node for node in openapi_defs if node.properties.get("name") == "User"),
        None,
    )
    assert openapi_user is not None
    assert openapi_user.properties.get("title") == "Demo"

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    name_schema = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "openapi"
            and node.properties.get("name") == "name"
        ),
        None,
    )
    age_schema = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "openapi"
            and node.properties.get("name") == "age"
        ),
        None,
    )

    assert name_schema is not None
    assert age_schema is not None
    assert name_schema.properties.get("field_type") == "string"
    assert name_schema.properties.get("nullable") is False
    assert age_schema.properties.get("field_type") == "integer"
    assert age_schema.properties.get("nullable") is True

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    name_field = _find_symbol(symbols, "name", "field", parent_name="User")
    assert name_field is not None
    assert store.has_edge(name_field.id, name_schema.id, CodeEdgeType.SCHEMA_EXPOSES)
