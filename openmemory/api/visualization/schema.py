"""Hierarchical code graph JSON schema.

This module provides:
- HierarchicalNode for nested graph representation
- HierarchicalGraph for complete graph structure
- SchemaValidator for validating exported graphs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openmemory.api.indexing.graph_projection import (
    CodeEdge,
    CodeEdgeType,
    CodeNode,
    CodeNodeType,
)


# =============================================================================
# Exceptions
# =============================================================================


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    pass


# =============================================================================
# Valid Types
# =============================================================================


VALID_NODE_TYPES = {"CODE_FILE", "CODE_SYMBOL", "CODE_PACKAGE"}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HierarchicalNode:
    """A node in a hierarchical graph representation.

    Attributes:
        id: Unique identifier for the node.
        type: Node type (CODE_FILE, CODE_SYMBOL, CODE_PACKAGE).
        name: Display name for the node.
        children: Child nodes in the hierarchy.
        properties: Additional node properties.
        parent_id: Optional reference to parent node ID.
    """

    id: str
    type: str
    name: str
    children: list["HierarchicalNode"] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def to_dict(self, exclude_empty: bool = False) -> dict[str, Any]:
        """Convert node to dictionary representation.

        Args:
            exclude_empty: If True, exclude empty optional fields.

        Returns:
            Dictionary representation of the node.
        """
        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
        }

        if self.children or not exclude_empty:
            result["children"] = [child.to_dict(exclude_empty) for child in self.children]

        if self.properties or not exclude_empty:
            result["properties"] = self.properties

        if self.parent_id:
            result["parent_id"] = self.parent_id

        return result

    @classmethod
    def from_code_node(
        cls,
        node: CodeNode,
        parent_id: Optional[str] = None,
    ) -> "HierarchicalNode":
        """Create HierarchicalNode from CodeNode.

        Args:
            node: The CodeNode to convert.
            parent_id: Optional parent node ID.

        Returns:
            HierarchicalNode instance.
        """
        name = node.properties.get("name", node.id)
        return cls(
            id=node.id,
            type=node.node_type.value,
            name=name,
            properties=dict(node.properties),
            parent_id=parent_id,
        )


@dataclass
class HierarchicalGraph:
    """A hierarchical graph representation.

    Attributes:
        version: Schema version.
        roots: Root nodes of the hierarchy.
        metadata: Optional graph metadata.
        edge_list: Non-containment edges (e.g., CALLS).
    """

    version: str
    roots: list[HierarchicalNode]
    metadata: dict[str, Any] = field(default_factory=dict)
    edge_list: list[tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary representation.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "version": self.version,
            "roots": [root.to_dict() for root in self.roots],
            "metadata": self.metadata,
            "edges": [
                {"source": s, "target": t, "type": e}
                for s, t, e in self.edge_list
            ],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert graph to JSON string.

        Args:
            indent: Indentation level for pretty printing.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_code_nodes(
        cls,
        nodes: list[CodeNode],
        edges: list[CodeEdge],
        version: str = "1.0",
    ) -> "HierarchicalGraph":
        """Build hierarchical graph from code nodes and edges.

        Args:
            nodes: List of CodeNodes.
            edges: List of CodeEdges.
            version: Schema version.

        Returns:
            HierarchicalGraph instance.
        """
        # Build node lookup
        node_map: dict[str, CodeNode] = {n.id: n for n in nodes}

        # Separate containment edges from other edges
        containment: dict[str, list[str]] = {}  # parent -> [children]
        other_edges: list[tuple[str, str, str]] = []
        contained_ids: set[str] = set()

        for edge in edges:
            if edge.edge_type == CodeEdgeType.CONTAINS:
                if edge.source_id not in containment:
                    containment[edge.source_id] = []
                containment[edge.source_id].append(edge.target_id)
                contained_ids.add(edge.target_id)
            else:
                other_edges.append((
                    edge.source_id,
                    edge.target_id,
                    edge.edge_type.value,
                ))

        # Build hierarchical nodes recursively
        def build_hier_node(
            node_id: str,
            parent_id: Optional[str] = None,
        ) -> Optional[HierarchicalNode]:
            if node_id not in node_map:
                return None
            code_node = node_map[node_id]
            hier_node = HierarchicalNode.from_code_node(code_node, parent_id)

            # Add children
            if node_id in containment:
                for child_id in containment[node_id]:
                    child = build_hier_node(child_id, node_id)
                    if child:
                        hier_node.children.append(child)

            return hier_node

        # Find roots (nodes not contained by anything)
        root_ids = [n.id for n in nodes if n.id not in contained_ids]
        roots = []
        for root_id in root_ids:
            root_node = build_hier_node(root_id)
            if root_node:
                roots.append(root_node)

        # Build metadata
        metadata = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        return cls(
            version=version,
            roots=roots,
            metadata=metadata,
            edge_list=other_edges,
        )

    @classmethod
    def from_flat_export(
        cls,
        flat_data: dict[str, Any],
        version: str = "1.0",
    ) -> "HierarchicalGraph":
        """Build hierarchical graph from flat JSON export.

        Args:
            flat_data: Flat export data with nodes and edges.
            version: Schema version.

        Returns:
            HierarchicalGraph instance.
        """
        # Convert flat nodes back to CodeNodes
        nodes: list[CodeNode] = []
        for node_dict in flat_data.get("nodes", []):
            node_type = CodeNodeType(node_dict["type"])
            nodes.append(CodeNode(
                node_type=node_type,
                id=node_dict["id"],
                properties=node_dict.get("properties", {}),
            ))

        # Convert flat edges back to CodeEdges
        edges: list[CodeEdge] = []
        for edge_dict in flat_data.get("edges", []):
            edge_type = CodeEdgeType(edge_dict["type"])
            edges.append(CodeEdge(
                edge_type=edge_type,
                source_id=edge_dict["source"],
                target_id=edge_dict["target"],
                properties=edge_dict.get("properties", {}),
            ))

        return cls.from_code_nodes(nodes, edges, version)


# =============================================================================
# Schema Validator
# =============================================================================


class SchemaValidator:
    """Validate hierarchical graph JSON schemas.

    Validates that exported graphs conform to the expected schema:
    - Required fields present
    - Valid node types
    - Valid property types
    - Proper nesting structure
    """

    def __init__(self):
        """Initialize schema validator."""
        self.valid_node_types = VALID_NODE_TYPES

    def validate_node(self, node_dict: dict[str, Any]) -> bool:
        """Validate a node dictionary.

        Args:
            node_dict: Node to validate.

        Returns:
            True if valid.

        Raises:
            SchemaValidationError: If validation fails.
        """
        # Check required fields
        if "id" not in node_dict:
            raise SchemaValidationError("Node missing required field: id")
        if "type" not in node_dict:
            raise SchemaValidationError("Node missing required field: type")
        if "name" not in node_dict:
            raise SchemaValidationError("Node missing required field: name")

        # Validate type
        if node_dict["type"] not in self.valid_node_types:
            raise SchemaValidationError(
                f"Invalid node type: {node_dict['type']}. "
                f"Valid types: {self.valid_node_types}"
            )

        # Validate children recursively
        children = node_dict.get("children", [])
        if not isinstance(children, list):
            raise SchemaValidationError("children must be a list")
        for child in children:
            self.validate_node(child)

        # Validate properties if present
        properties = node_dict.get("properties", {})
        if not isinstance(properties, dict):
            raise SchemaValidationError("properties must be a dict")

        return True

    def validate_graph(self, graph_dict: dict[str, Any]) -> bool:
        """Validate a complete graph dictionary.

        Args:
            graph_dict: Graph to validate.

        Returns:
            True if valid.

        Raises:
            SchemaValidationError: If validation fails.
        """
        # Check required fields
        if "version" not in graph_dict:
            raise SchemaValidationError("Graph missing required field: version")
        if "roots" not in graph_dict:
            raise SchemaValidationError("Graph missing required field: roots")

        # Validate roots
        roots = graph_dict["roots"]
        if not isinstance(roots, list):
            raise SchemaValidationError("roots must be a list")

        for root in roots:
            self.validate_node(root)

        return True
