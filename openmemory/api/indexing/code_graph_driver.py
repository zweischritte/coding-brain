"""Neo4j-backed driver for CODE_* graph operations."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from app.graph.neo4j_client import get_neo4j_config
from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
    GraphProjectionError,
    Neo4jDriver,
)

logger = logging.getLogger(__name__)

_EDGE_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PROPERTY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class CodeGraphDriver(Neo4jDriver):
    """Neo4j driver implementation for CODE_* nodes and edges."""

    def __init__(self, driver: Any, database: Optional[str] = None):
        self._driver = driver
        self._database = database

    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def _run(self, query: str, parameters: Optional[dict[str, Any]] = None):
        with self._session() as session:
            return list(session.run(query, parameters or {}))

    def _label(self, node_type: CodeNodeType) -> str:
        return node_type.label if hasattr(node_type, "label") else str(node_type)

    def _edge_type(self, edge_type: CodeEdgeType | str) -> str:
        value = edge_type.value if hasattr(edge_type, "value") else str(edge_type)
        if not _EDGE_TYPE_RE.match(value):
            raise ValueError(f"Invalid edge type: {value}")
        return value

    def _property_name(self, property_name: str) -> str:
        if not _PROPERTY_RE.match(property_name):
            raise ValueError(f"Invalid property name: {property_name}")
        return property_name

    def _node_type_from_labels(self, labels: list[str]) -> CodeNodeType:
        for label in labels:
            try:
                return CodeNodeType(label)
            except ValueError:
                continue
        return CodeNodeType.SYMBOL

    def add_node(self, node: CodeNode) -> None:
        label = self._label(node.node_type)
        props = dict(node.properties)
        props["id"] = node.id
        query = f"MERGE (n:{label} {{id: $id}}) SET n += $props"
        self._run(query, {"id": node.id, "props": props})

    def update_node(self, node: CodeNode) -> None:
        label = self._label(node.node_type)
        props = dict(node.properties)
        props["id"] = node.id
        query = f"MATCH (n:{label} {{id: $id}}) SET n += $props RETURN count(n) AS count"
        records = self._run(query, {"id": node.id, "props": props})
        if not records or records[0]["count"] == 0:
            raise GraphProjectionError(f"Node {node.id} not found")

    def delete_node(self, node_id: str) -> None:
        query = (
            "MATCH (n {id: $id}) "
            "WHERE any(label IN labels(n) WHERE label STARTS WITH 'CODE_') "
            "DETACH DELETE n"
        )
        self._run(query, {"id": node_id})

    def get_node(self, node_id: str) -> Optional[CodeNode]:
        query = (
            "MATCH (n {id: $id}) "
            "WHERE any(label IN labels(n) WHERE label STARTS WITH 'CODE_') "
            "RETURN labels(n) AS labels, properties(n) AS props "
            "LIMIT 1"
        )
        records = self._run(query, {"id": node_id})
        if not records:
            return None

        labels = records[0]["labels"]
        props = dict(records[0]["props"])
        props.pop("id", None)
        node_type = self._node_type_from_labels(labels)
        return CodeNode(node_type=node_type, id=node_id, properties=props)

    def add_edge(self, edge: CodeEdge) -> None:
        edge_type = self._edge_type(edge.edge_type)
        query = (
            f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
            f"MERGE (a)-[r:{edge_type}]->(b) "
            "SET r += $props"
        )
        self._run(
            query,
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "props": edge.properties or {},
            },
        )

    def delete_edge(self, source_id: str, target_id: str, edge_type: CodeEdgeType) -> None:
        edge_label = self._edge_type(edge_type)
        query = (
            f"MATCH (a {{id: $source_id}})-[r:{edge_label}]->(b {{id: $target_id}}) "
            "DELETE r"
        )
        self._run(query, {"source_id": source_id, "target_id": target_id})

    def has_edge(self, source_id: str, target_id: str, edge_type: CodeEdgeType) -> bool:
        edge_label = self._edge_type(edge_type)
        query = (
            f"MATCH (a {{id: $source_id}})-[r:{edge_label}]->(b {{id: $target_id}}) "
            "RETURN count(r) AS count"
        )
        records = self._run(query, {"source_id": source_id, "target_id": target_id})
        return bool(records and records[0]["count"] > 0)

    def get_outgoing_edges(self, node_id: str) -> list[CodeEdge]:
        query = (
            "MATCH (a {id: $node_id})-[r]->(b) "
            "RETURN type(r) AS type, properties(r) AS props, b.id AS target_id"
        )
        records = self._run(query, {"node_id": node_id})
        edges: list[CodeEdge] = []
        for record in records:
            edge_type_value = record["type"]
            edge_type = (
                CodeEdgeType(edge_type_value)
                if edge_type_value in CodeEdgeType._value2member_map_
                else edge_type_value
            )
            edges.append(
                CodeEdge(
                    edge_type=edge_type,
                    source_id=node_id,
                    target_id=record["target_id"],
                    properties=dict(record["props"] or {}),
                )
            )
        return edges

    def get_incoming_edges(self, node_id: str) -> list[CodeEdge]:
        query = (
            "MATCH (a)-[r]->(b {id: $node_id}) "
            "RETURN type(r) AS type, properties(r) AS props, a.id AS source_id"
        )
        records = self._run(query, {"node_id": node_id})
        edges: list[CodeEdge] = []
        for record in records:
            edge_type_value = record["type"]
            edge_type = (
                CodeEdgeType(edge_type_value)
                if edge_type_value in CodeEdgeType._value2member_map_
                else edge_type_value
            )
            edges.append(
                CodeEdge(
                    edge_type=edge_type,
                    source_id=record["source_id"],
                    target_id=node_id,
                    properties=dict(record["props"] or {}),
                )
            )
        return edges

    def query_nodes_by_type(self, node_type: CodeNodeType) -> list[CodeNode]:
        label = self._label(node_type)
        query = f"MATCH (n:{label}) RETURN n.id AS id, properties(n) AS props"
        records = self._run(query)
        nodes: list[CodeNode] = []
        for record in records:
            props = dict(record["props"])
            node_id = record["id"]
            props.pop("id", None)
            nodes.append(CodeNode(node_type=node_type, id=node_id, properties=props))
        return nodes

    def create_constraint(
        self,
        name: str,
        node_type: CodeNodeType,
        property_name: str,
        constraint_type: str,
    ) -> None:
        label = self._label(node_type)
        prop = self._property_name(property_name)
        constraint = constraint_type.upper()

        if constraint == "UNIQUE":
            clause = f"REQUIRE n.{prop} IS UNIQUE"
        elif constraint == "EXISTS":
            clause = f"REQUIRE n.{prop} IS NOT NULL"
        else:
            raise ValueError(f"Unsupported constraint type: {constraint_type}")

        query = f"CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) {clause}"
        self._run(query)

    def has_constraint(self, name: str) -> bool:
        query = "SHOW CONSTRAINTS YIELD name WHERE name = $name RETURN count(*) AS count"
        records = self._run(query, {"name": name})
        return bool(records and records[0]["count"] > 0)

    @property
    def node_count(self) -> int:
        query = (
            "MATCH (n) "
            "WHERE any(label IN labels(n) WHERE label STARTS WITH 'CODE_') "
            "RETURN count(n) AS count"
        )
        records = self._run(query)
        return int(records[0]["count"]) if records else 0

    @property
    def edge_count(self) -> int:
        query = (
            "MATCH (a)-[r]->(b) "
            "WHERE any(label IN labels(a) WHERE label STARTS WITH 'CODE_') "
            "   OR any(label IN labels(b) WHERE label STARTS WITH 'CODE_') "
            "RETURN count(r) AS count"
        )
        records = self._run(query)
        return int(records[0]["count"]) if records else 0

    def find_symbol_id(self, symbol_name: str, repo_id: Optional[str] = None) -> Optional[str]:
        query = "MATCH (n:CODE_SYMBOL {name: $name})"
        params: dict[str, Any] = {"name": symbol_name}
        if repo_id:
            query += " WHERE n.repo_id = $repo_id"
            params["repo_id"] = repo_id
        query += " RETURN n.id AS id ORDER BY n.file_path, n.line_start LIMIT 1"
        records = self._run(query, params)
        if records:
            return records[0]["id"]
        return None

    def find_file_id(self, file_path: str, repo_id: Optional[str] = None) -> Optional[str]:
        query = "MATCH (n:CODE_FILE) WHERE n.path ENDS WITH $path"
        params: dict[str, Any] = {"path": file_path}
        if repo_id:
            query += " AND n.repo_id = $repo_id"
            params["repo_id"] = repo_id
        query += " RETURN n.id AS id ORDER BY n.path LIMIT 1"
        records = self._run(query, params)
        if records:
            return records[0]["id"]
        return None

    def delete_repo_nodes(self, repo_id: str) -> int:
        query = (
            "MATCH (n) "
            "WHERE n.repo_id = $repo_id "
            "  AND any(label IN labels(n) WHERE label STARTS WITH 'CODE_') "
            "WITH collect(n) AS nodes, count(n) AS count "
            "FOREACH (n IN nodes | DETACH DELETE n) "
            "RETURN count AS deleted"
        )
        records = self._run(query, {"repo_id": repo_id})
        if records:
            return int(records[0]["deleted"])
        return 0


def create_code_graph_driver(driver: Any) -> Optional[CodeGraphDriver]:
    """Create a code graph driver from a Neo4j driver."""
    if driver is None:
        return None
    config = get_neo4j_config()
    return CodeGraphDriver(driver=driver, database=config.get("database"))
