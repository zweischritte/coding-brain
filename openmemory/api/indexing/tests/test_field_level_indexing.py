"""Integration tests for field-level symbols and schema edges."""

from pathlib import Path
from typing import Optional

import pytest

from openmemory.api.indexing.code_indexer import CodeIndexingService
from openmemory.api.indexing.graph_projection import CodeEdgeType, CodeNodeType, MemoryGraphStore


@pytest.fixture
def indexer(tmp_path: Path) -> tuple[CodeIndexingService, MemoryGraphStore, Path]:
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


def test_has_field_and_field_access_edges(indexer):
    indexer, store, src = indexer

    code = """
import { Field } from '@nestjs/graphql';

class User {
  @Field()
  name: string;
  count = 0;

  constructor(public id: string) {}

  setName(value: string) {
    this.name = value;
  }

  getName() {
    return this.name;
  }
}
"""
    file_path = src / "user.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    class_node = _find_symbol(symbols, "User", "class")
    field_node = _find_symbol(symbols, "name", "field", parent_name="User")
    set_node = _find_symbol(symbols, "setName", "method", parent_name="User")
    get_node = _find_symbol(symbols, "getName", "method", parent_name="User")

    assert class_node is not None
    assert field_node is not None
    assert set_node is not None
    assert get_node is not None

    assert store.has_edge(class_node.id, field_node.id, CodeEdgeType.HAS_FIELD)
    assert store.has_edge(set_node.id, field_node.id, CodeEdgeType.WRITES)
    assert store.has_edge(get_node.id, field_node.id, CodeEdgeType.READS)


def test_schema_exposure_edges_graphql_and_zod(indexer):
    indexer, store, src = indexer

    code = """
import { Field } from '@nestjs/graphql';
import { z } from 'zod';

class User {
  @Field()
  name: string;
}

const UserSchema = z.object({
  name: z.string(),
});
"""
    file_path = src / "schema.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    field_node = _find_symbol(symbols, "name", "field", parent_name="User")
    assert field_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    graphql_node = next(
        (node for node in schema_nodes if node.properties.get("schema_type") == "graphql"),
        None,
    )
    zod_node = next(
        (node for node in schema_nodes if node.properties.get("schema_type") == "zod"),
        None,
    )

    assert graphql_node is not None
    assert zod_node is not None

    assert store.has_edge(field_node.id, graphql_node.id, CodeEdgeType.SCHEMA_EXPOSES)
    assert store.has_edge(field_node.id, zod_node.id, CodeEdgeType.SCHEMA_EXPOSES)


def test_zod_schema_field_metadata(indexer):
    indexer, store, src = indexer

    code = """
import { z } from 'zod';

class User {
  name: string;
  age: number;
}

const UserSchema = z.object({
  name: z.string().optional(),
  age: z.number().nullable(),
});
"""
    file_path = src / "zod.ts"
    file_path.write_text(code)

    indexer.index_repository()

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    name_node = next(
        (node for node in schema_nodes if node.properties.get("schema_type") == "zod" and node.properties.get("name") == "name"),
        None,
    )
    age_node = next(
        (node for node in schema_nodes if node.properties.get("schema_type") == "zod" and node.properties.get("name") == "age"),
        None,
    )

    assert name_node is not None
    assert age_node is not None
    assert name_node.properties.get("field_type") == "string"
    assert age_node.properties.get("field_type") == "number"
    assert name_node.properties.get("nullable") is True
    assert age_node.properties.get("nullable") is True


def test_dto_schema_exposure_edges(indexer):
    indexer, store, src = indexer

    code = """
import { IsOptional, IsString, IsInt } from 'class-validator';

class UserDto {
  @IsOptional()
  @IsString()
  name: string;

  @IsInt()
  age: number;
}
"""
    file_path = src / "dto.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    name_field = _find_symbol(symbols, "name", "field", parent_name="UserDto")
    age_field = _find_symbol(symbols, "age", "field", parent_name="UserDto")
    assert name_field is not None
    assert age_field is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    name_schema = next(
        (node for node in schema_nodes if node.properties.get("schema_type") == "dto" and node.properties.get("name") == "name"),
        None,
    )
    age_schema = next(
        (node for node in schema_nodes if node.properties.get("schema_type") == "dto" and node.properties.get("name") == "age"),
        None,
    )

    assert name_schema is not None
    assert age_schema is not None
    assert name_schema.properties.get("field_type") == "string"
    assert name_schema.properties.get("nullable") is True
    assert age_schema.properties.get("field_type") == "number"

    assert store.has_edge(name_field.id, name_schema.id, CodeEdgeType.SCHEMA_EXPOSES)
    assert store.has_edge(age_field.id, age_schema.id, CodeEdgeType.SCHEMA_EXPOSES)
