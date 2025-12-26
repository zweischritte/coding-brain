"""CODE_* Graph Projection for Neo4j.

This module provides graph projection for code indexing:
- Node types: CODE_FILE, CODE_SYMBOL, CODE_PACKAGE
- Edge types: CONTAINS, DEFINES, IMPORTS, CALLS, READS, WRITES, DATA_FLOWS_TO
- Incremental updates (add/modify/delete)
- Transaction support (atomic operations)
- Batch operations for performance
- Neo4j constraint management
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType
from openmemory.api.indexing.scip_symbols import SCIPSymbolID

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class GraphProjectionError(Exception):
    """Base exception for graph projection errors."""

    pass


class ConstraintViolationError(GraphProjectionError):
    """Raised when a constraint is violated."""

    pass


class TransactionError(GraphProjectionError):
    """Raised for transaction-related errors."""

    pass


# =============================================================================
# Enums
# =============================================================================


class CodeNodeType(Enum):
    """Types of code nodes in the graph."""

    FILE = "CODE_FILE"
    SYMBOL = "CODE_SYMBOL"
    PACKAGE = "CODE_PACKAGE"

    @property
    def label(self) -> str:
        """Get Neo4j label for this node type."""
        return self.value


class CodeEdgeType(Enum):
    """Types of edges in the code graph."""

    CONTAINS = "CONTAINS"  # File contains symbol
    DEFINES = "DEFINES"  # Package defines symbol
    IMPORTS = "IMPORTS"  # File/symbol imports another
    CALLS = "CALLS"  # Symbol calls another symbol
    READS = "READS"  # Symbol reads a variable
    WRITES = "WRITES"  # Symbol writes a variable
    DATA_FLOWS_TO = "DATA_FLOWS_TO"  # Data flows from source to sink


class BatchOperationType(Enum):
    """Types of batch operations."""

    ADD_NODES = "add_nodes"
    UPDATE_NODES = "update_nodes"
    DELETE_NODES = "delete_nodes"
    ADD_EDGES = "add_edges"
    DELETE_EDGES = "delete_edges"


class TransactionState(Enum):
    """Transaction state."""

    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class CodeNode:
    """A node in the code graph."""

    node_type: CodeNodeType
    id: str
    properties: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeNode):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class CodeEdge:
    """An edge in the code graph."""

    edge_type: CodeEdgeType
    source_id: str
    target_id: str
    properties: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeEdge):
            return False
        return (
            self.edge_type == other.edge_type
            and self.source_id == other.source_id
            and self.target_id == other.target_id
        )

    def __hash__(self) -> int:
        return hash((self.edge_type, self.source_id, self.target_id))


@dataclass
class BatchOperation:
    """A batch operation on the graph."""

    operation_type: BatchOperationType
    items: list[Any]


@dataclass
class BatchResult:
    """Result of a batch operation."""

    success: bool
    processed: int
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class GraphProjectionConfig:
    """Configuration for graph projection."""

    create_constraints: bool = True
    batch_size: int = 1000
    auto_commit: bool = True


@dataclass
class Constraint:
    """Graph constraint definition."""

    name: str
    node_type: CodeNodeType
    property_name: str
    constraint_type: str  # UNIQUE, EXISTS, etc.


# =============================================================================
# Node Builders
# =============================================================================


class FileNodeBuilder:
    """Builder for CODE_FILE nodes."""

    def __init__(self):
        self._path: Optional[Path] = None
        self._language: Optional[Language] = None
        self._size: Optional[int] = None
        self._content_hash: Optional[str] = None

    def path(self, path: Path) -> "FileNodeBuilder":
        """Set file path."""
        self._path = path
        return self

    def language(self, language: Language) -> "FileNodeBuilder":
        """Set programming language."""
        self._language = language
        return self

    def size(self, size: int) -> "FileNodeBuilder":
        """Set file size."""
        self._size = size
        return self

    def content_hash(self, hash_value: str) -> "FileNodeBuilder":
        """Set content hash."""
        self._content_hash = hash_value
        return self

    def build(self) -> CodeNode:
        """Build the file node."""
        if self._path is None:
            raise ValueError("path is required")

        properties: dict[str, Any] = {
            "path": str(self._path),
            "name": self._path.name,
        }

        if self._language:
            properties["language"] = self._language.value

        if self._size is not None:
            properties["size"] = self._size

        if self._content_hash:
            properties["content_hash"] = self._content_hash

        return CodeNode(
            node_type=CodeNodeType.FILE,
            id=str(self._path),
            properties=properties,
        )


class SymbolNodeBuilder:
    """Builder for CODE_SYMBOL nodes."""

    def __init__(self):
        self._symbol: Optional[Symbol] = None
        self._scip_id: Optional[SCIPSymbolID] = None
        self._file_path: Optional[Path] = None

    def from_symbol(self, symbol: Symbol) -> "SymbolNodeBuilder":
        """Initialize from parsed symbol."""
        self._symbol = symbol
        return self

    def scip_id(self, scip_id: SCIPSymbolID) -> "SymbolNodeBuilder":
        """Set SCIP symbol ID."""
        self._scip_id = scip_id
        return self

    def file_path(self, path: Path) -> "SymbolNodeBuilder":
        """Set source file path."""
        self._file_path = path
        return self

    def build(self) -> CodeNode:
        """Build the symbol node."""
        if self._scip_id is None:
            raise ValueError("scip_id is required")

        properties: dict[str, Any] = {
            "scip_id": str(self._scip_id),
        }

        if self._symbol:
            properties["name"] = self._symbol.name
            properties["kind"] = self._symbol.symbol_type.value
            properties["line_start"] = self._symbol.line_start
            properties["line_end"] = self._symbol.line_end
            properties["language"] = self._symbol.language.value

            if self._symbol.signature:
                properties["signature"] = self._symbol.signature
            if self._symbol.docstring:
                properties["docstring"] = self._symbol.docstring
            if self._symbol.parent_name:
                properties["parent_name"] = self._symbol.parent_name

        if self._file_path:
            properties["file_path"] = str(self._file_path)

        return CodeNode(
            node_type=CodeNodeType.SYMBOL,
            id=str(self._scip_id),
            properties=properties,
        )


class PackageNodeBuilder:
    """Builder for CODE_PACKAGE nodes."""

    def __init__(self):
        self._name: Optional[str] = None
        self._full_name: Optional[str] = None
        self._language: Optional[Language] = None

    def name(self, name: str) -> "PackageNodeBuilder":
        """Set package short name."""
        self._name = name
        return self

    def full_name(self, full_name: str) -> "PackageNodeBuilder":
        """Set fully qualified package name."""
        self._full_name = full_name
        return self

    def language(self, language: Language) -> "PackageNodeBuilder":
        """Set programming language."""
        self._language = language
        return self

    def build(self) -> CodeNode:
        """Build the package node."""
        if self._full_name is None:
            raise ValueError("full_name is required")

        properties: dict[str, Any] = {
            "full_name": self._full_name,
        }

        if self._name:
            properties["name"] = self._name

        if self._language:
            properties["language"] = self._language.value

        return CodeNode(
            node_type=CodeNodeType.PACKAGE,
            id=self._full_name,
            properties=properties,
        )


# =============================================================================
# Edge Builder
# =============================================================================


class EdgeBuilder:
    """Builder for graph edges."""

    def __init__(self):
        self._edge_type: Optional[CodeEdgeType] = None
        self._source_id: Optional[str] = None
        self._target_id: Optional[str] = None
        self._properties: dict[str, Any] = {}

    def contains(self) -> "EdgeBuilder":
        """Set edge type to CONTAINS."""
        self._edge_type = CodeEdgeType.CONTAINS
        return self

    def defines(self) -> "EdgeBuilder":
        """Set edge type to DEFINES."""
        self._edge_type = CodeEdgeType.DEFINES
        return self

    def imports(self) -> "EdgeBuilder":
        """Set edge type to IMPORTS."""
        self._edge_type = CodeEdgeType.IMPORTS
        return self

    def calls(self) -> "EdgeBuilder":
        """Set edge type to CALLS."""
        self._edge_type = CodeEdgeType.CALLS
        return self

    def reads(self) -> "EdgeBuilder":
        """Set edge type to READS."""
        self._edge_type = CodeEdgeType.READS
        return self

    def writes(self) -> "EdgeBuilder":
        """Set edge type to WRITES."""
        self._edge_type = CodeEdgeType.WRITES
        return self

    def data_flows_to(self) -> "EdgeBuilder":
        """Set edge type to DATA_FLOWS_TO."""
        self._edge_type = CodeEdgeType.DATA_FLOWS_TO
        return self

    def from_node(self, source_id: str) -> "EdgeBuilder":
        """Set source node ID."""
        self._source_id = source_id
        return self

    def to_node(self, target_id: str) -> "EdgeBuilder":
        """Set target node ID."""
        self._target_id = target_id
        return self

    def at_line(self, line: int) -> "EdgeBuilder":
        """Set call line number."""
        self._properties["call_line"] = line
        return self

    def at_column(self, column: int) -> "EdgeBuilder":
        """Set call column number."""
        self._properties["call_col"] = column
        return self

    def with_property(self, key: str, value: Any) -> "EdgeBuilder":
        """Add custom property."""
        self._properties[key] = value
        return self

    def build(self) -> CodeEdge:
        """Build the edge."""
        if self._edge_type is None:
            raise ValueError("edge_type is required")
        if self._source_id is None:
            raise ValueError("source_id is required")
        if self._target_id is None:
            raise ValueError("target_id is required")

        return CodeEdge(
            edge_type=self._edge_type,
            source_id=self._source_id,
            target_id=self._target_id,
            properties=self._properties,
        )


# =============================================================================
# Graph Store Interface
# =============================================================================


class Neo4jDriver(ABC):
    """Abstract interface for Neo4j driver operations."""

    @abstractmethod
    def add_node(self, node: CodeNode) -> None:
        """Add a node to the graph."""
        pass

    @abstractmethod
    def update_node(self, node: CodeNode) -> None:
        """Update an existing node."""
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[CodeNode]:
        """Get a node by ID."""
        pass

    @abstractmethod
    def add_edge(self, edge: CodeEdge) -> None:
        """Add an edge to the graph."""
        pass

    @abstractmethod
    def delete_edge(self, source_id: str, target_id: str, edge_type: CodeEdgeType) -> None:
        """Delete an edge from the graph."""
        pass

    @abstractmethod
    def has_edge(self, source_id: str, target_id: str, edge_type: CodeEdgeType) -> bool:
        """Check if an edge exists."""
        pass

    @abstractmethod
    def get_outgoing_edges(self, node_id: str) -> list[CodeEdge]:
        """Get all outgoing edges from a node."""
        pass

    @abstractmethod
    def query_nodes_by_type(self, node_type: CodeNodeType) -> list[CodeNode]:
        """Query nodes by type."""
        pass

    @abstractmethod
    def create_constraint(
        self,
        name: str,
        node_type: CodeNodeType,
        property_name: str,
        constraint_type: str,
    ) -> None:
        """Create a constraint on the graph."""
        pass

    @abstractmethod
    def has_constraint(self, name: str) -> bool:
        """Check if a constraint exists."""
        pass

    @property
    @abstractmethod
    def node_count(self) -> int:
        """Get total number of nodes."""
        pass

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Get total number of edges."""
        pass


# =============================================================================
# Memory Graph Store (for testing)
# =============================================================================


class MemoryGraphStore(Neo4jDriver):
    """In-memory graph store for testing."""

    def __init__(self):
        self._nodes: dict[str, CodeNode] = {}
        self._edges: dict[tuple[str, str, CodeEdgeType], CodeEdge] = {}
        self._constraints: dict[str, Constraint] = {}
        self._constraint_values: dict[str, set[Any]] = {}  # property -> values

    def add_node(self, node: CodeNode) -> None:
        """Add a node to the graph."""
        if node.id in self._nodes:
            raise ConstraintViolationError(f"Node {node.id} already exists")

        # Check unique constraints
        for constraint in self._constraints.values():
            if constraint.node_type == node.node_type:
                prop_value = node.properties.get(constraint.property_name)
                if prop_value is not None:
                    key = f"{constraint.name}:{constraint.property_name}"
                    if key not in self._constraint_values:
                        self._constraint_values[key] = set()

                    if prop_value in self._constraint_values[key]:
                        raise ConstraintViolationError(
                            f"Constraint {constraint.name} violated: "
                            f"duplicate value {prop_value} for {constraint.property_name}"
                        )
                    self._constraint_values[key].add(prop_value)

        self._nodes[node.id] = node

    def update_node(self, node: CodeNode) -> None:
        """Update an existing node."""
        if node.id not in self._nodes:
            raise GraphProjectionError(f"Node {node.id} not found")

        # Remove old constraint values
        old_node = self._nodes[node.id]
        for constraint in self._constraints.values():
            if constraint.node_type == old_node.node_type:
                key = f"{constraint.name}:{constraint.property_name}"
                if key in self._constraint_values:
                    old_value = old_node.properties.get(constraint.property_name)
                    if old_value in self._constraint_values[key]:
                        self._constraint_values[key].discard(old_value)

        # Add new constraint values
        for constraint in self._constraints.values():
            if constraint.node_type == node.node_type:
                prop_value = node.properties.get(constraint.property_name)
                if prop_value is not None:
                    key = f"{constraint.name}:{constraint.property_name}"
                    if key not in self._constraint_values:
                        self._constraint_values[key] = set()
                    self._constraint_values[key].add(prop_value)

        self._nodes[node.id] = node

    def delete_node(self, node_id: str) -> None:
        """Delete a node and its edges."""
        if node_id not in self._nodes:
            return

        # Remove constraint values
        node = self._nodes[node_id]
        for constraint in self._constraints.values():
            if constraint.node_type == node.node_type:
                key = f"{constraint.name}:{constraint.property_name}"
                if key in self._constraint_values:
                    prop_value = node.properties.get(constraint.property_name)
                    if prop_value in self._constraint_values[key]:
                        self._constraint_values[key].discard(prop_value)

        del self._nodes[node_id]

        # Remove related edges
        edges_to_remove = [
            key
            for key in self._edges
            if key[0] == node_id or key[1] == node_id
        ]
        for key in edges_to_remove:
            del self._edges[key]

    def get_node(self, node_id: str) -> Optional[CodeNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def add_edge(self, edge: CodeEdge) -> None:
        """Add an edge to the graph."""
        key = (edge.source_id, edge.target_id, edge.edge_type)
        self._edges[key] = edge

    def delete_edge(self, source_id: str, target_id: str, edge_type: CodeEdgeType) -> None:
        """Delete an edge from the graph."""
        key = (source_id, target_id, edge_type)
        if key in self._edges:
            del self._edges[key]

    def has_edge(self, source_id: str, target_id: str, edge_type: CodeEdgeType) -> bool:
        """Check if an edge exists."""
        key = (source_id, target_id, edge_type)
        return key in self._edges

    def get_outgoing_edges(self, node_id: str) -> list[CodeEdge]:
        """Get all outgoing edges from a node."""
        return [
            edge
            for key, edge in self._edges.items()
            if key[0] == node_id
        ]

    def query_nodes_by_type(self, node_type: CodeNodeType) -> list[CodeNode]:
        """Query nodes by type."""
        return [
            node
            for node in self._nodes.values()
            if node.node_type == node_type
        ]

    def create_constraint(
        self,
        name: str,
        node_type: CodeNodeType,
        property_name: str,
        constraint_type: str,
    ) -> None:
        """Create a constraint on the graph."""
        self._constraints[name] = Constraint(
            name=name,
            node_type=node_type,
            property_name=property_name,
            constraint_type=constraint_type,
        )

    def has_constraint(self, name: str) -> bool:
        """Check if a constraint exists."""
        return name in self._constraints

    @property
    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Get total number of edges."""
        return len(self._edges)


# =============================================================================
# Transaction
# =============================================================================


class ProjectionTransaction:
    """Transaction for atomic graph operations."""

    def __init__(self, projection: "GraphProjection"):
        self._projection = projection
        self._pending_nodes: list[CodeNode] = []
        self._pending_edges: list[CodeEdge] = []
        self._pending_deletes: list[str] = []
        self.state = TransactionState.PENDING

    def add_node(self, node: CodeNode) -> None:
        """Add node to pending operations."""
        self._pending_nodes.append(node)

    def add_edge(self, edge: CodeEdge) -> None:
        """Add edge to pending operations."""
        self._pending_edges.append(edge)

    def delete_node(self, node_id: str) -> None:
        """Add delete to pending operations."""
        self._pending_deletes.append(node_id)

    def commit(self) -> None:
        """Commit all pending operations."""
        if self.state != TransactionState.PENDING:
            raise TransactionError("Transaction already finalized")

        try:
            # Apply all operations
            for node in self._pending_nodes:
                self._projection.driver.add_node(node)
            for edge in self._pending_edges:
                self._projection.driver.add_edge(edge)
            for node_id in self._pending_deletes:
                self._projection.driver.delete_node(node_id)

            self.state = TransactionState.COMMITTED
        except Exception as e:
            self.rollback()
            raise TransactionError(f"Commit failed: {e}") from e

    def rollback(self) -> None:
        """Rollback pending operations."""
        self._pending_nodes.clear()
        self._pending_edges.clear()
        self._pending_deletes.clear()
        self.state = TransactionState.ROLLED_BACK


# =============================================================================
# Graph Projection Service
# =============================================================================


class GraphProjection:
    """Main graph projection service."""

    def __init__(
        self,
        driver: Neo4jDriver,
        config: Optional[GraphProjectionConfig] = None,
    ):
        self.driver = driver
        self.config = config or GraphProjectionConfig()
        self._in_transaction = False
        self._current_transaction: Optional[ProjectionTransaction] = None

    # -------------------------------------------------------------------------
    # Node Creation
    # -------------------------------------------------------------------------

    def create_file_node(
        self,
        path: Path,
        language: Language,
        size: Optional[int] = None,
        content_hash: Optional[str] = None,
    ) -> CodeNode:
        """Create a CODE_FILE node."""
        builder = FileNodeBuilder().path(path).language(language)

        if size is not None:
            builder.size(size)
        if content_hash:
            builder.content_hash(content_hash)

        node = builder.build()
        self.driver.add_node(node)
        return node

    def create_symbol_node(
        self,
        symbol: Symbol,
        scip_id: SCIPSymbolID,
        file_path: Path,
    ) -> CodeNode:
        """Create a CODE_SYMBOL node."""
        node = (
            SymbolNodeBuilder()
            .from_symbol(symbol)
            .scip_id(scip_id)
            .file_path(file_path)
            .build()
        )
        self.driver.add_node(node)
        return node

    def create_package_node(
        self,
        name: str,
        full_name: str,
        language: Language,
    ) -> CodeNode:
        """Create a CODE_PACKAGE node."""
        node = (
            PackageNodeBuilder()
            .name(name)
            .full_name(full_name)
            .language(language)
            .build()
        )
        self.driver.add_node(node)
        return node

    # -------------------------------------------------------------------------
    # Edge Creation
    # -------------------------------------------------------------------------

    def create_edge(
        self,
        edge_type: CodeEdgeType,
        source_id: str,
        target_id: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> CodeEdge:
        """Create an edge between nodes."""
        edge = CodeEdge(
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            properties=properties or {},
        )
        self.driver.add_edge(edge)
        return edge

    # -------------------------------------------------------------------------
    # Node Updates
    # -------------------------------------------------------------------------

    def update_symbol_node(
        self,
        scip_id: SCIPSymbolID,
        symbol: Symbol,
    ) -> CodeNode:
        """Update an existing symbol node."""
        node = (
            SymbolNodeBuilder()
            .from_symbol(symbol)
            .scip_id(scip_id)
            .build()
        )
        self.driver.update_node(node)
        return node

    def update_file_node(
        self,
        path: Path,
        content_hash: Optional[str] = None,
        size: Optional[int] = None,
    ) -> CodeNode:
        """Update an existing file node."""
        existing = self.driver.get_node(str(path))
        if not existing:
            raise GraphProjectionError(f"File node not found: {path}")

        properties = dict(existing.properties)
        if content_hash:
            properties["content_hash"] = content_hash
        if size is not None:
            properties["size"] = size

        updated = CodeNode(
            node_type=CodeNodeType.FILE,
            id=str(path),
            properties=properties,
        )
        self.driver.update_node(updated)
        return updated

    # -------------------------------------------------------------------------
    # Node Deletion
    # -------------------------------------------------------------------------

    def delete_node(self, node_id: str) -> None:
        """Delete a node by ID."""
        self.driver.delete_node(node_id)

    def delete_file_with_contents(self, path: Path) -> None:
        """Delete a file node and all contained symbols."""
        file_id = str(path)

        # Find all contained symbols
        edges = self.driver.get_outgoing_edges(file_id)
        symbol_ids = [
            edge.target_id
            for edge in edges
            if edge.edge_type == CodeEdgeType.CONTAINS
        ]

        # Delete symbols first
        for symbol_id in symbol_ids:
            self.driver.delete_node(symbol_id)

        # Delete file
        self.driver.delete_node(file_id)

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def execute_batch(
        self,
        batch: BatchOperation,
        continue_on_error: bool = False,
    ) -> BatchResult:
        """Execute a batch operation."""
        processed = 0
        failed = 0
        errors: list[str] = []

        if batch.operation_type == BatchOperationType.ADD_NODES:
            for item in batch.items:
                try:
                    self.driver.add_node(item)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(str(e))
                    if not continue_on_error:
                        raise

        elif batch.operation_type == BatchOperationType.ADD_EDGES:
            for item in batch.items:
                try:
                    self.driver.add_edge(item)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(str(e))
                    if not continue_on_error:
                        raise

        elif batch.operation_type == BatchOperationType.DELETE_NODES:
            for item in batch.items:
                try:
                    self.driver.delete_node(item)
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(str(e))
                    if not continue_on_error:
                        raise

        return BatchResult(
            success=failed == 0,
            processed=processed,
            failed=failed,
            errors=errors,
        )

    # -------------------------------------------------------------------------
    # Transactions
    # -------------------------------------------------------------------------

    def begin_transaction(self) -> ProjectionTransaction:
        """Begin a new transaction."""
        if self._in_transaction:
            raise TransactionError("already in transaction")

        self._in_transaction = True
        self._current_transaction = ProjectionTransaction(self)
        return self._current_transaction

    @contextmanager
    def transaction(self) -> Iterator[ProjectionTransaction]:
        """Context manager for transactions."""
        tx = self.begin_transaction()
        try:
            yield tx
            if tx.state == TransactionState.PENDING:
                tx.commit()
        except Exception:
            if tx.state == TransactionState.PENDING:
                tx.rollback()
            raise
        finally:
            self._in_transaction = False
            self._current_transaction = None


# =============================================================================
# Factory Function
# =============================================================================


def create_graph_projection(
    driver_type: str = "memory",
    config: Optional[GraphProjectionConfig] = None,
    **driver_kwargs: Any,
) -> GraphProjection:
    """Create a graph projection instance.

    Args:
        driver_type: Type of driver ("memory" or "neo4j")
        config: Optional projection configuration
        **driver_kwargs: Additional arguments for driver initialization

    Returns:
        GraphProjection instance
    """
    if driver_type == "memory":
        driver = MemoryGraphStore()
    else:
        raise ValueError(f"Unknown driver type: {driver_type}")

    projection = GraphProjection(driver=driver, config=config)

    # Create default constraints if configured
    if config and config.create_constraints:
        driver.create_constraint(
            name="code_symbol_scip_id_unique",
            node_type=CodeNodeType.SYMBOL,
            property_name="scip_id",
            constraint_type="UNIQUE",
        )

    return projection
