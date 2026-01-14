"""Tests for generated source classification heuristics."""

from pathlib import Path

import pytest

from openmemory.api.indexing.code_indexer import CodeIndexingService
from openmemory.api.indexing.graph_projection import MemoryGraphStore


@pytest.fixture
def indexer(tmp_path: Path) -> CodeIndexingService:
    return CodeIndexingService(
        root_path=tmp_path,
        repo_id="test-repo",
        graph_driver=MemoryGraphStore(),
        opensearch_client=None,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts", ".js"],
    )


def test_generated_by_path(indexer, tmp_path: Path) -> None:
    file_path = tmp_path / "dist" / "bundle.js"
    is_generated, reason, tier = indexer._classify_source_tier(file_path, ["console.log('hi');"])

    assert is_generated is True
    assert tier == "generated"
    assert reason == "path:dist"


def test_vendor_by_path(indexer, tmp_path: Path) -> None:
    file_path = tmp_path / "node_modules" / "pkg" / "index.js"
    is_generated, reason, tier = indexer._classify_source_tier(file_path, ["export const x = 1;"])

    assert is_generated is True
    assert tier == "vendor"
    assert reason == "path:node_modules"


def test_generated_by_extension(indexer, tmp_path: Path) -> None:
    file_path = tmp_path / "types.d.ts"
    is_generated, reason, tier = indexer._classify_source_tier(file_path, ["export interface Foo {}"])

    assert is_generated is True
    assert tier == "generated"
    assert reason == "extension:.d.ts"


def test_generated_by_header(indexer, tmp_path: Path) -> None:
    file_path = tmp_path / "src" / "schema.ts"
    lines = ["// @generated", "export const schema = {};"]
    is_generated, reason, tier = indexer._classify_source_tier(file_path, lines)

    assert is_generated is True
    assert tier == "generated"
    assert reason == "header:@generated"


def test_source_default(indexer, tmp_path: Path) -> None:
    file_path = tmp_path / "src" / "main.ts"
    is_generated, reason, tier = indexer._classify_source_tier(file_path, ["export function main() {}"])

    assert is_generated is False
    assert tier == "source"
    assert reason is None
