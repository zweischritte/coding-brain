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


def _find_path_node(nodes, path_value: str):
    for node in nodes:
        if node.properties.get("path") == path_value:
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
    assert store.has_edge(str(file_path), graphql_node.id, CodeEdgeType.CONTAINS)
    assert store.has_edge(str(file_path), zod_node.id, CodeEdgeType.CONTAINS)


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


def test_zod_schema_cross_file_links_to_class_field(indexer):
    indexer, store, src = indexer

    class_code = """
class Producer {
  firstname: string;
}
"""
    schema_code = """
import { z } from 'zod';

const ProducerSchema = z.object({
  firstname: z.string(),
});
"""
    class_path = src / "producer.ts"
    schema_path = src / "producer.schema.ts"
    class_path.write_text(class_code)
    schema_path.write_text(schema_code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    field_node = _find_symbol(symbols, "firstname", "field", parent_name="Producer")
    assert field_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    zod_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
        ),
        None,
    )
    assert zod_node is not None
    assert store.has_edge(field_node.id, zod_node.id, CodeEdgeType.SCHEMA_EXPOSES)
    assert store.has_edge(str(schema_path), zod_node.id, CodeEdgeType.CONTAINS)


def test_zod_infer_reads_and_writes_edges(indexer):
    indexer, store, src = indexer

    code = """
import { z } from 'zod';

const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
});

function useProducer(producer: z.infer<typeof ProducerSchema>) {
  return producer.firstname;
}

function makeProducer(): z.infer<typeof ProducerSchema> {
  return {
    firstname: "Alice",
    lastname: "Smith",
  };
}
"""
    file_path = src / "producer.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    use_node = _find_symbol(symbols, "useProducer", "function")
    make_node = _find_symbol(symbols, "makeProducer", "function")

    assert use_node is not None
    assert make_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    firstname_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
        ),
        None,
    )
    assert firstname_node is not None
    assert store.has_edge(use_node.id, firstname_node.id, CodeEdgeType.READS)
    assert store.has_edge(make_node.id, firstname_node.id, CodeEdgeType.WRITES)


def test_zod_infer_arrow_function_writes_edges(indexer):
    indexer, store, src = indexer

    code = """
import { z } from 'zod';

const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
});

function build() {
  return [1].map(
    (): z.infer<typeof ProducerSchema> => ({
      firstname: "Alice",
      lastname: "Smith",
    }),
  );
}
"""
    file_path = src / "producer-arrow.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    build_node = _find_symbol(symbols, "build", "function")
    assert build_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    firstname_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
        ),
        None,
    )
    assert firstname_node is not None
    assert store.has_edge(build_node.id, firstname_node.id, CodeEdgeType.WRITES)


def test_zod_satisfies_object_literal_writes_edges(indexer):
    indexer, store, src = indexer

    code = """
import { z } from 'zod';

const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
});

function build() {
  const value = { firstname: "Ada", lastname: "Lovelace" } satisfies z.output<typeof ProducerSchema>;
  return value;
}
"""
    file_path = src / "producer-satisfies.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    build_node = _find_symbol(symbols, "build", "function")
    assert build_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    firstname_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
        ),
        None,
    )
    assert firstname_node is not None
    assert store.has_edge(build_node.id, firstname_node.id, CodeEdgeType.WRITES)


def test_zod_parse_object_literal_writes_edges(indexer):
    indexer, store, src = indexer

    code = """
import { z } from 'zod';

const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
});

function build() {
  ProducerSchema.parse({ firstname: "Ada", lastname: "Lovelace" });
}
"""
    file_path = src / "producer-parse.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    build_node = _find_symbol(symbols, "build", "function")
    assert build_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    firstname_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
        ),
        None,
    )
    assert firstname_node is not None
    assert store.has_edge(build_node.id, firstname_node.id, CodeEdgeType.WRITES)


def test_zod_infer_resolves_workspace_package(tmp_path: Path):
    store = MemoryGraphStore()
    indexer = CodeIndexingService(
        root_path=tmp_path,
        repo_id="test-repo",
        graph_driver=store,
        opensearch_client=None,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts"],
    )

    package_root = tmp_path / "packages" / "shared"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "package.json").write_text('{"name": "@repo/shared"}')

    schema_path = package_root / "src" / "schema.ts"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        """
import { z } from 'zod';

export const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
});
"""
    )

    consumer_path = tmp_path / "apps" / "app" / "src" / "consumer.ts"
    consumer_path.parent.mkdir(parents=True, exist_ok=True)
    consumer_path.write_text(
        """
import { z } from 'zod';
import { ProducerSchema } from '@repo/shared';

function build(): z.infer<typeof ProducerSchema> {
  return { firstname: "Alice", lastname: "Smith" };
}
"""
    )

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    build_node = _find_symbol(symbols, "build", "function")
    assert build_node is not None

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    firstname_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
            and node.properties.get("file_path") == str(schema_path)
        ),
        None,
    )
    assert firstname_node is not None
    assert store.has_edge(build_node.id, firstname_node.id, CodeEdgeType.WRITES)


def test_nested_zod_schema_fields(indexer):
    indexer, store, src = indexer

    code = """
import { z } from 'zod';

const schema = z
  .object({
    movie: z
      .object({
        producers: z.array(
          z.object({
            firstname: z.string(),
            lastname: z.string(),
          }),
        ).min(1),
      })
      .superRefine(() => {}),
  })
  .superRefine(() => {});
"""
    file_path = src / "nested.ts"
    file_path.write_text(code)

    indexer.index_repository()

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    nested_node = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "movie.producers.firstname"
        ),
        None,
    )
    assert nested_node is not None


def test_zod_schema_alias_edges(tmp_path: Path):
    store = MemoryGraphStore()
    indexer = CodeIndexingService(
        root_path=tmp_path,
        repo_id="test-repo",
        graph_driver=store,
        opensearch_client=None,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts"],
        enable_zod_schema_aliases=True,
    )

    canonical_path = tmp_path / "packages" / "core" / "entities" / "producer.schema.ts"
    local_path = tmp_path / "apps" / "frontend" / "routes" / "producer.schema.ts"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_path.write_text(
        """
import { z } from 'zod';

export const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
  company: z.string(),
});
"""
    )
    local_path.write_text(
        """
import { z } from 'zod';

const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
  company: z.string(),
});
"""
    )

    indexer.index_repository()

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    canonical_firstname = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
            and node.properties.get("file_path") == str(canonical_path)
        ),
        None,
    )
    local_firstname = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
            and node.properties.get("file_path") == str(local_path)
        ),
        None,
    )

    assert canonical_firstname is not None
    assert local_firstname is not None
    assert store.has_edge(
        local_firstname.id,
        canonical_firstname.id,
        CodeEdgeType.SCHEMA_ALIASES,
    )


def test_zod_schema_alias_edges_generic_name(tmp_path: Path):
    store = MemoryGraphStore()
    indexer = CodeIndexingService(
        root_path=tmp_path,
        repo_id="test-repo",
        graph_driver=store,
        opensearch_client=None,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts"],
        enable_zod_schema_aliases=True,
    )

    canonical_path = tmp_path / "packages" / "core" / "entities" / "movie.schema.ts"
    local_path = tmp_path / "apps" / "frontend" / "routes" / "movie-report.state.ts"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_path.write_text(
        """
import { z } from 'zod';

export const MovieSchema = z.object({
  title: z.string(),
  year: z.number(),
  producers: z.array(z.string()),
});
"""
    )
    local_path.write_text(
        """
import { z } from 'zod';

const schema = z.object({
  title: z.string(),
  year: z.number(),
  producers: z.array(z.string()),
});
"""
    )

    indexer.index_repository()

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    canonical_title = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "title"
            and node.properties.get("file_path") == str(canonical_path)
        ),
        None,
    )
    local_title = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "title"
            and node.properties.get("file_path") == str(local_path)
        ),
        None,
    )

    assert canonical_title is not None
    assert local_title is not None
    assert store.has_edge(
        local_title.id,
        canonical_title.id,
        CodeEdgeType.SCHEMA_ALIASES,
    )


def test_zod_subschema_alias_edges(tmp_path: Path):
    store = MemoryGraphStore()
    indexer = CodeIndexingService(
        root_path=tmp_path,
        repo_id="test-repo",
        graph_driver=store,
        opensearch_client=None,
        embedding_service=None,
        include_api_boundaries=False,
        extensions=[".ts"],
        enable_zod_schema_aliases=True,
    )

    canonical_path = tmp_path / "packages" / "core" / "producer.schema.ts"
    local_path = tmp_path / "apps" / "frontend" / "routes" / "report.state.ts"
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    canonical_path.write_text(
        """
import { z } from 'zod';

export const ProducerSchema = z.object({
  firstname: z.string(),
  lastname: z.string(),
});
"""
    )
    local_path.write_text(
        """
import { z } from 'zod';

const schema = z.object({
  movie: z.object({
    producers: z.array(
      z.object({
        firstname: z.string(),
        lastname: z.string(),
      }),
    ),
  }),
});
"""
    )

    indexer.index_repository()

    schema_nodes = store.query_nodes_by_type(CodeNodeType.SCHEMA_FIELD)
    canonical_firstname = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "firstname"
            and node.properties.get("file_path") == str(canonical_path)
        ),
        None,
    )
    local_firstname = next(
        (
            node
            for node in schema_nodes
            if node.properties.get("schema_type") == "zod"
            and node.properties.get("name") == "movie.producers.firstname"
            and node.properties.get("file_path") == str(local_path)
        ),
        None,
    )

    assert canonical_firstname is not None
    assert local_firstname is not None
    assert store.has_edge(
        local_firstname.id,
        canonical_firstname.id,
        CodeEdgeType.SCHEMA_ALIASES,
    )

def test_field_reads_from_typed_variable(indexer):
    indexer, store, src = indexer

    producer_code = """
export class Producer {
  firstname: string;
}
"""
    consumer_code = """
import { Producer } from "./producer";

function useProducer(producer: Producer) {
  return producer.firstname;
}
"""
    producer_path = src / "producer.ts"
    consumer_path = src / "consumer.ts"
    producer_path.write_text(producer_code)
    consumer_path.write_text(consumer_code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    field_node = _find_symbol(symbols, "firstname", "field", parent_name="Producer")
    use_node = _find_symbol(symbols, "useProducer", "function")

    assert field_node is not None
    assert use_node is not None
    assert store.has_edge(use_node.id, field_node.id, CodeEdgeType.READS)


def test_graphql_document_links_to_field(indexer):
    indexer, store, src = indexer

    schema_code = """
type Query {
  movie: Movie
}

type Movie {
  producers: [Producer!]!
}

type Producer {
  firstname: String!
}
"""
    class_code = """
class Producer {
  firstname: string;
}
"""
    doc_code = """
query MovieDetails {
  movie {
    producers {
      firstname
    }
  }
}
"""
    schema_path = src / "schema.graphql"
    class_path = src / "producer.ts"
    doc_path = src / "movie.graphql"
    schema_path.write_text(schema_code)
    class_path.write_text(class_code)
    doc_path.write_text(doc_code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    field_node = _find_symbol(symbols, "firstname", "field", parent_name="Producer")
    assert field_node is not None
    assert store.has_edge(str(doc_path), field_node.id, CodeEdgeType.READS)


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


def test_path_literal_indexing(indexer):
    indexer, store, src = indexer

    code = """
const mapping = {
  "user.address.city": "x",
};

const value = get(obj, "order.items[0].price");
register(`movie.producers.${i}.firstname`);
const jsonPath = "$.movie.producers[*].firstname";
const standalone = "profile.contact.email";
const data = obj["account.settings.theme"];
"""
    file_path = src / "paths.ts"
    file_path.write_text(code)

    indexer.index_repository()

    path_nodes = store.query_nodes_by_type(CodeNodeType.FIELD_PATH)

    high_node = _find_path_node(path_nodes, "user.address.city")
    assert high_node is not None
    assert high_node.properties.get("confidence") == "high"
    assert store.has_edge(str(file_path), high_node.id, CodeEdgeType.CONTAINS)

    bracket_node = _find_path_node(path_nodes, "account.settings.theme")
    assert bracket_node is not None
    assert bracket_node.properties.get("confidence") == "high"

    medium_node = _find_path_node(path_nodes, "order.items[0].price")
    assert medium_node is not None
    assert medium_node.properties.get("confidence") == "medium"

    template_node = _find_path_node(path_nodes, "movie.producers.${i}.firstname")
    assert template_node is not None
    assert template_node.properties.get("confidence") == "medium"
    assert "*" in template_node.properties.get("segments", [])

    json_node = _find_path_node(path_nodes, "$.movie.producers[*].firstname")
    assert json_node is not None
    assert json_node.properties.get("leaf") == "firstname"
    assert "*" not in json_node.properties.get("segments", [])

    low_node = _find_path_node(path_nodes, "profile.contact.email")
    assert low_node is not None
    assert low_node.properties.get("confidence") == "low"


def test_member_expression_path_literals(indexer):
    indexer, store, src = indexer

    code = """
const data = { channelId: "x" };
const id = data.channelId;
const nested = item.meldung.senderId;
"""
    file_path = src / "member-paths.ts"
    file_path.write_text(code)

    indexer.index_repository()

    path_nodes = store.query_nodes_by_type(CodeNodeType.FIELD_PATH)

    data_node = _find_path_node(path_nodes, "data.channelId")
    assert data_node is not None
    assert data_node.properties.get("confidence") == "medium"

    nested_node = _find_path_node(path_nodes, "item.meldung.senderId")
    assert nested_node is not None
    assert nested_node.properties.get("confidence") == "medium"


def test_path_literal_edges(indexer):
    indexer, store, src = indexer

    code = """
class Producer {
  firstname: string;
  lastname: string;
}

const value = "movie.producers.firstname";
"""
    file_path = src / "path-edges.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    firstname_field = _find_symbol(symbols, "firstname", "field", parent_name="Producer")
    assert firstname_field is not None

    path_nodes = store.query_nodes_by_type(CodeNodeType.FIELD_PATH)
    path_node = _find_path_node(path_nodes, "movie.producers.firstname")
    assert path_node is not None

    assert store.has_edge(path_node.id, firstname_field.id, CodeEdgeType.PATH_READS)


def test_object_literal_field_writes_with_property_hint(indexer):
    indexer, store, src = indexer

    code = """
class Producer {
  firstname: string;
  lastname: string;
}

class Movie {
  producers: Producer[];
}

function makeMovie() {
  return {
    producers: [
      { firstname: "Ada", lastname: "Lovelace" },
    ],
  };
}
"""
    file_path = src / "movie.ts"
    file_path.write_text(code)

    indexer.index_repository()

    symbols = store.query_nodes_by_type(CodeNodeType.SYMBOL)
    firstname_field = _find_symbol(symbols, "firstname", "field", parent_name="Producer")
    make_movie = _find_symbol(symbols, "makeMovie", "function")

    assert firstname_field is not None
    assert make_movie is not None

    assert store.has_edge(make_movie.id, firstname_field.id, CodeEdgeType.WRITES)
