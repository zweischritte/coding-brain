"""Tests for allowlist support in indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from openmemory.api.indexing.code_indexer import CodeIndexingService
from openmemory.api.indexing.graph_projection import CodeEdgeType, CodeNodeType, MemoryGraphStore


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


def test_allow_patterns_include_dist_schema(tmp_path: Path):
    store = MemoryGraphStore()
    root = tmp_path
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)

    schema_path = root / "dist/schema.gql"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        """
type Query {
  movie: Movie
}

type Movie {
  producers: [Producer!]!
}

type Producer {
  firstname: String
}
"""
    )

    (src / "producer.ts").write_text(
        """
class Producer {
  firstname: string;
}
"""
    )

    doc_path = src / "movie.graphql"
    doc_path.write_text(
        """
query MovieDetails {
  movie {
    producers {
      firstname
    }
  }
}
"""
    )

    indexer = CodeIndexingService(
        root_path=root,
        repo_id="test-repo",
        graph_driver=store,
        opensearch_client=None,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts"],
        ignore_patterns=["dist"],
        allow_patterns=["dist/schema.gql"],
    )
    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    field_node = _find_symbol(symbols, "firstname", "field", parent_name="Producer")

    assert field_node is not None
    assert store.has_edge(str(doc_path), field_node.id, CodeEdgeType.READS)
