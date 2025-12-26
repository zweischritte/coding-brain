"""Cross-language API boundary detection (FR-003).

This module provides:
- REST endpoint detection in Python (FastAPI, Flask)
- API client detection in TypeScript (fetch, axios)
- EXPOSES edges from handler symbols to API endpoints
- CONSUMES edges from client code to API endpoints
- HTTP method and path extraction
- Path parameter detection
- Integration with SCIP symbol IDs and graph projection
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import tree_sitter_python as ts_python
import tree_sitter_typescript as ts_typescript
from tree_sitter import Language as TSLanguage, Parser, Tree, Node

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType
from openmemory.api.indexing.graph_projection import (
    CodeNode,
    CodeNodeType,
    CodeEdge,
    CodeEdgeType,
    GraphProjection,
)
from openmemory.api.indexing.scip_symbols import SCIPSymbolID

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class HTTPMethod(Enum):
    """HTTP methods for REST APIs."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

    @classmethod
    def from_string(cls, method: str) -> Optional["HTTPMethod"]:
        """Parse HTTP method from string."""
        method_upper = method.upper()
        for m in cls:
            if m.value == method_upper:
                return m
        return None


class APIBoundaryEdgeType(Enum):
    """Edge types for API boundaries."""

    EXPOSES = "EXPOSES"  # Handler symbol exposes an API endpoint
    CONSUMES = "CONSUMES"  # Client code consumes an API endpoint


class APIBoundaryNodeType(Enum):
    """Node types for API boundaries."""

    API_ENDPOINT = "CODE_APIEndpoint"
    API_CLIENT = "CODE_APIClient"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class APIEndpoint:
    """Represents a detected REST API endpoint."""

    method: HTTPMethod
    path: str
    handler_name: str
    handler_scip_id: Optional[str]
    file_path: Path
    line_number: int
    framework: str = "unknown"

    @property
    def path_params(self) -> list[str]:
        """Extract path parameters from the path."""
        # Match {param} or <param> or <type:param>
        params = re.findall(r"\{([^}]+)\}|<(?:[^:>]+:)?([^>]+)>", self.path)
        return [p[0] or p[1] for p in params if p[0] or p[1]]

    @property
    def unique_id(self) -> str:
        """Generate unique ID for this endpoint."""
        return f"{self.method.value}:{self.path}"


@dataclass
class APIClient:
    """Represents a detected API client call."""

    method: HTTPMethod
    path: str
    caller_name: str
    caller_scip_id: Optional[str]
    file_path: Path
    line_number: int
    client_type: str  # "fetch", "axios", etc.

    @property
    def target_endpoint_id(self) -> str:
        """Generate target endpoint ID for matching."""
        return f"{self.method.value}:{self.path}"


@dataclass
class APIBoundaryLink:
    """Represents a link between an endpoint and a client."""

    endpoint: APIEndpoint
    client: APIClient


@dataclass
class APIBoundaryAnalysisResult:
    """Result of API boundary analysis."""

    endpoints_found: int
    clients_found: int
    links_created: int
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Custom Node Type for API Boundaries
# =============================================================================


@dataclass(frozen=True)
class APIBoundaryNode:
    """A node in the API boundary graph."""

    node_type: APIBoundaryNodeType
    id: str
    properties: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, APIBoundaryNode):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class APIBoundaryEdge:
    """An edge in the API boundary graph."""

    edge_type: APIBoundaryEdgeType
    source_id: str
    target_id: str
    properties: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, APIBoundaryEdge):
            return False
        return (
            self.edge_type == other.edge_type
            and self.source_id == other.source_id
            and self.target_id == other.target_id
        )

    def __hash__(self) -> int:
        return hash((self.edge_type, self.source_id, self.target_id))


# =============================================================================
# Node Builders
# =============================================================================


class APIEndpointNodeBuilder:
    """Builder for CODE_APIEndpoint nodes."""

    def __init__(self):
        self._method: Optional[HTTPMethod] = None
        self._path: Optional[str] = None
        self._handler_name: Optional[str] = None
        self._handler_scip_id: Optional[str] = None
        self._file_path: Optional[Path] = None
        self._line_number: Optional[int] = None
        self._framework: str = "unknown"

    def method(self, method: HTTPMethod) -> "APIEndpointNodeBuilder":
        """Set HTTP method."""
        self._method = method
        return self

    def path(self, path: str) -> "APIEndpointNodeBuilder":
        """Set endpoint path."""
        self._path = path
        return self

    def handler_name(self, name: str) -> "APIEndpointNodeBuilder":
        """Set handler function name."""
        self._handler_name = name
        return self

    def handler_scip_id(self, scip_id: Union[str, SCIPSymbolID]) -> "APIEndpointNodeBuilder":
        """Set handler SCIP ID."""
        self._handler_scip_id = str(scip_id)
        return self

    def file_path(self, path: Path) -> "APIEndpointNodeBuilder":
        """Set source file path."""
        self._file_path = path
        return self

    def line_number(self, line: int) -> "APIEndpointNodeBuilder":
        """Set line number."""
        self._line_number = line
        return self

    def framework(self, framework: str) -> "APIEndpointNodeBuilder":
        """Set framework name."""
        self._framework = framework
        return self

    def build(self) -> APIBoundaryNode:
        """Build the endpoint node."""
        if self._method is None:
            raise ValueError("method is required")
        if self._path is None:
            raise ValueError("path is required")

        node_id = f"{self._method.value}:{self._path}"

        properties: dict[str, Any] = {
            "method": self._method.value,
            "path": self._path,
            "framework": self._framework,
        }

        if self._handler_name:
            properties["handler_name"] = self._handler_name
        if self._handler_scip_id:
            properties["handler_scip_id"] = self._handler_scip_id
        if self._file_path:
            properties["file_path"] = str(self._file_path)
        if self._line_number is not None:
            properties["line_number"] = self._line_number

        return APIBoundaryNode(
            node_type=APIBoundaryNodeType.API_ENDPOINT,
            id=node_id,
            properties=properties,
        )


class APIClientNodeBuilder:
    """Builder for CODE_APIClient nodes."""

    def __init__(self):
        self._method: Optional[HTTPMethod] = None
        self._path: Optional[str] = None
        self._caller_name: Optional[str] = None
        self._caller_scip_id: Optional[str] = None
        self._file_path: Optional[Path] = None
        self._line_number: Optional[int] = None
        self._client_type: str = "unknown"

    def method(self, method: HTTPMethod) -> "APIClientNodeBuilder":
        """Set HTTP method."""
        self._method = method
        return self

    def path(self, path: str) -> "APIClientNodeBuilder":
        """Set target path."""
        self._path = path
        return self

    def caller_name(self, name: str) -> "APIClientNodeBuilder":
        """Set caller function name."""
        self._caller_name = name
        return self

    def caller_scip_id(self, scip_id: Union[str, SCIPSymbolID]) -> "APIClientNodeBuilder":
        """Set caller SCIP ID."""
        self._caller_scip_id = str(scip_id)
        return self

    def file_path(self, path: Path) -> "APIClientNodeBuilder":
        """Set source file path."""
        self._file_path = path
        return self

    def line_number(self, line: int) -> "APIClientNodeBuilder":
        """Set line number."""
        self._line_number = line
        return self

    def client_type(self, client_type: str) -> "APIClientNodeBuilder":
        """Set client type (fetch, axios, etc)."""
        self._client_type = client_type
        return self

    def build(self) -> APIBoundaryNode:
        """Build the client node."""
        if self._method is None:
            raise ValueError("method is required")
        if self._path is None:
            raise ValueError("path is required")

        # Include line number and file for uniqueness
        node_id = f"{self._client_type}:{self._method.value}:{self._path}:{self._file_path}:{self._line_number}"

        properties: dict[str, Any] = {
            "method": self._method.value,
            "path": self._path,
            "client_type": self._client_type,
        }

        if self._caller_name:
            properties["caller_name"] = self._caller_name
        if self._caller_scip_id:
            properties["caller_scip_id"] = self._caller_scip_id
        if self._file_path:
            properties["file_path"] = str(self._file_path)
        if self._line_number is not None:
            properties["line_number"] = self._line_number

        return APIBoundaryNode(
            node_type=APIBoundaryNodeType.API_CLIENT,
            id=node_id,
            properties=properties,
        )


# =============================================================================
# Edge Builder
# =============================================================================


class APIBoundaryEdgeBuilder:
    """Builder for API boundary edges."""

    def __init__(self):
        self._edge_type: Optional[APIBoundaryEdgeType] = None
        self._source_id: Optional[str] = None
        self._target_id: Optional[str] = None
        self._properties: dict[str, Any] = {}

    def exposes(self) -> "APIBoundaryEdgeBuilder":
        """Set edge type to EXPOSES."""
        self._edge_type = APIBoundaryEdgeType.EXPOSES
        return self

    def consumes(self) -> "APIBoundaryEdgeBuilder":
        """Set edge type to CONSUMES."""
        self._edge_type = APIBoundaryEdgeType.CONSUMES
        return self

    def from_handler(self, handler_id: str) -> "APIBoundaryEdgeBuilder":
        """Set source as handler ID."""
        self._source_id = handler_id
        return self

    def from_client(self, client_id: str) -> "APIBoundaryEdgeBuilder":
        """Set source as client ID."""
        self._source_id = client_id
        return self

    def to_endpoint(self, endpoint_id: str) -> "APIBoundaryEdgeBuilder":
        """Set target as endpoint ID."""
        self._target_id = endpoint_id
        return self

    def with_property(self, key: str, value: Any) -> "APIBoundaryEdgeBuilder":
        """Add custom property."""
        self._properties[key] = value
        return self

    def build(self) -> APIBoundaryEdge:
        """Build the edge."""
        if self._edge_type is None:
            raise ValueError("edge_type is required")
        if self._source_id is None:
            raise ValueError("source_id is required")
        if self._target_id is None:
            raise ValueError("target_id is required")

        return APIBoundaryEdge(
            edge_type=self._edge_type,
            source_id=self._source_id,
            target_id=self._target_id,
            properties=self._properties,
        )


# =============================================================================
# REST Endpoint Detection
# =============================================================================


class RestEndpointDetector:
    """Detects REST endpoints in Python code."""

    def __init__(self):
        self._parser = Parser(TSLanguage(ts_python.language()))
        # Patterns for FastAPI and Flask decorators
        self._fastapi_pattern = re.compile(
            r"@(?:app|router)\.(?P<method>get|post|put|delete|patch)\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
        )
        self._flask_route_pattern = re.compile(
            r"@(?:app|bp|blueprint)\.route\s*\(\s*['\"](?P<path>[^'\"]+)['\"]"
        )
        self._flask_methods_pattern = re.compile(
            r"methods\s*=\s*\[([^\]]+)\]"
        )

    def detect_python(self, code: str, file_path: Path) -> list[APIEndpoint]:
        """Detect REST endpoints in Python code."""
        if not code.strip():
            return []

        endpoints: list[APIEndpoint] = []

        try:
            source = code.encode("utf-8")
            tree = self._parser.parse(source)
            self._extract_endpoints(tree.root_node, source, file_path, endpoints)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return endpoints

    def _extract_endpoints(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        endpoints: list[APIEndpoint],
    ) -> None:
        """Extract endpoints from AST node."""
        if node.type == "decorated_definition":
            self._process_decorated_definition(node, source, file_path, endpoints)
        else:
            for child in node.children:
                self._extract_endpoints(child, source, file_path, endpoints)

    def _process_decorated_definition(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        endpoints: list[APIEndpoint],
    ) -> None:
        """Process a decorated function definition."""
        decorators: list[Node] = []
        function_node: Optional[Node] = None

        for child in node.children:
            if child.type == "decorator":
                decorators.append(child)
            elif child.type in ("function_definition", "async_function_definition"):
                function_node = child

        if not function_node:
            return

        # Get function name
        name_node = function_node.child_by_field_name("name")
        if not name_node:
            return
        func_name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")

        # Process decorators
        for decorator in decorators:
            decorator_text = source[decorator.start_byte : decorator.end_byte].decode("utf-8")
            line_number = decorator.start_point[0] + 1

            # Try FastAPI pattern
            fastapi_match = self._fastapi_pattern.search(decorator_text)
            if fastapi_match:
                method = HTTPMethod.from_string(fastapi_match.group("method"))
                path = fastapi_match.group("path")
                if method:
                    endpoints.append(
                        APIEndpoint(
                            method=method,
                            path=path,
                            handler_name=func_name,
                            handler_scip_id=None,
                            file_path=file_path,
                            line_number=line_number,
                            framework="fastapi",
                        )
                    )
                continue

            # Try Flask route pattern
            flask_match = self._flask_route_pattern.search(decorator_text)
            if flask_match:
                path = flask_match.group("path")
                # Check for methods parameter
                methods_match = self._flask_methods_pattern.search(decorator_text)
                if methods_match:
                    methods_str = methods_match.group(1)
                    # Extract methods from list
                    method_strs = re.findall(r"['\"](\w+)['\"]", methods_str)
                    for method_str in method_strs:
                        method = HTTPMethod.from_string(method_str)
                        if method:
                            endpoints.append(
                                APIEndpoint(
                                    method=method,
                                    path=path,
                                    handler_name=func_name,
                                    handler_scip_id=None,
                                    file_path=file_path,
                                    line_number=line_number,
                                    framework="flask",
                                )
                            )
                else:
                    # Default to GET for Flask
                    endpoints.append(
                        APIEndpoint(
                            method=HTTPMethod.GET,
                            path=path,
                            handler_name=func_name,
                            handler_scip_id=None,
                            file_path=file_path,
                            line_number=line_number,
                            framework="flask",
                        )
                    )


# =============================================================================
# API Client Detection
# =============================================================================


class APIClientDetector:
    """Detects API client calls in TypeScript code."""

    def __init__(self):
        self._ts_parser = Parser(TSLanguage(ts_typescript.language_typescript()))
        self._tsx_parser = Parser(TSLanguage(ts_typescript.language_tsx()))

    def detect_typescript(self, code: str, file_path: Path) -> list[APIClient]:
        """Detect API client calls in TypeScript code."""
        if not code.strip():
            return []

        clients: list[APIClient] = []

        try:
            source = code.encode("utf-8")
            is_tsx = file_path.suffix.lower() == ".tsx"
            parser = self._tsx_parser if is_tsx else self._ts_parser
            tree = parser.parse(source)
            self._extract_clients(tree.root_node, source, file_path, clients)
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")

        return clients

    def _extract_clients(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        clients: list[APIClient],
    ) -> None:
        """Extract API clients from AST node."""
        if node.type == "call_expression":
            self._process_call_expression(node, source, file_path, clients)

        for child in node.children:
            self._extract_clients(child, source, file_path, clients)

    def _process_call_expression(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        clients: list[APIClient],
    ) -> None:
        """Process a call expression to detect fetch/axios calls."""
        func_node = node.child_by_field_name("function")
        if not func_node:
            return

        func_text = source[func_node.start_byte : func_node.end_byte].decode("utf-8")
        args_node = node.child_by_field_name("arguments")
        line_number = node.start_point[0] + 1

        # Get containing function name
        caller_name = self._find_containing_function(node, source)

        # Check for fetch() calls
        if func_text == "fetch":
            client = self._extract_fetch_client(args_node, source, file_path, line_number, caller_name)
            if client:
                clients.append(client)
            return

        # Check for axios method calls
        if "." in func_text:
            parts = func_text.split(".")
            method_name = parts[-1].lower()
            if method_name in ("get", "post", "put", "delete", "patch"):
                http_method = HTTPMethod.from_string(method_name)
                if http_method and args_node:
                    path = self._extract_first_string_arg(args_node, source)
                    if path:
                        clients.append(
                            APIClient(
                                method=http_method,
                                path=path,
                                caller_name=caller_name or "anonymous",
                                caller_scip_id=None,
                                file_path=file_path,
                                line_number=line_number,
                                client_type="axios",
                            )
                        )

    def _extract_fetch_client(
        self,
        args_node: Optional[Node],
        source: bytes,
        file_path: Path,
        line_number: int,
        caller_name: Optional[str],
    ) -> Optional[APIClient]:
        """Extract fetch client call details."""
        if not args_node:
            return None

        # Get URL from first argument
        path = self._extract_first_string_arg(args_node, source)
        if not path:
            return None

        # Default to GET
        method = HTTPMethod.GET

        # Check for options object with method
        for child in args_node.children:
            if child.type == "object":
                method = self._extract_method_from_object(child, source) or method
                break

        return APIClient(
            method=method,
            path=path,
            caller_name=caller_name or "anonymous",
            caller_scip_id=None,
            file_path=file_path,
            line_number=line_number,
            client_type="fetch",
        )

    def _extract_first_string_arg(self, args_node: Node, source: bytes) -> Optional[str]:
        """Extract the first string argument from arguments node."""
        for child in args_node.children:
            if child.type == "string":
                # Remove quotes
                text = source[child.start_byte : child.end_byte].decode("utf-8")
                return text.strip("'\"")
            elif child.type == "template_string":
                # Handle template literals like `/api/users/${id}`
                text = source[child.start_byte : child.end_byte].decode("utf-8")
                # Remove backticks and simplify template expressions
                text = text.strip("`")
                # Replace ${...} with {param} for matching
                text = re.sub(r"\$\{[^}]+\}", "{param}", text)
                return text
        return None

    def _extract_method_from_object(self, obj_node: Node, source: bytes) -> Optional[HTTPMethod]:
        """Extract HTTP method from options object."""
        for child in obj_node.children:
            if child.type == "pair":
                key_node = child.child_by_field_name("key")
                value_node = child.child_by_field_name("value")
                if key_node and value_node:
                    key = source[key_node.start_byte : key_node.end_byte].decode("utf-8").strip("'\"")
                    if key.lower() == "method":
                        value = source[value_node.start_byte : value_node.end_byte].decode("utf-8").strip("'\"")
                        return HTTPMethod.from_string(value)
        return None

    def _find_containing_function(self, node: Node, source: bytes) -> Optional[str]:
        """Find the name of the containing function."""
        current = node.parent
        while current:
            if current.type in (
                "function_declaration",
                "method_definition",
                "arrow_function",
            ):
                name_node = current.child_by_field_name("name")
                if name_node:
                    return source[name_node.start_byte : name_node.end_byte].decode("utf-8")
                # For arrow functions, check parent variable declarator
                if current.type == "arrow_function" and current.parent:
                    if current.parent.type == "variable_declarator":
                        var_name = current.parent.child_by_field_name("name")
                        if var_name:
                            return source[var_name.start_byte : var_name.end_byte].decode("utf-8")
            current = current.parent
        return None


# =============================================================================
# Path Matching
# =============================================================================


class PathMatcher:
    """Matches and normalizes API paths."""

    def __init__(self):
        # Pattern to normalize path parameters
        self._param_pattern = re.compile(r"\{[^}]+\}|<(?:[^:>]+:)?[^>]+>|\{param\}")

    def normalize(self, path: str) -> str:
        """Normalize a path for comparison."""
        # Remove trailing slash
        path = path.rstrip("/")
        if not path:
            path = "/"
        # Normalize all parameter syntaxes to {*}
        path = self._param_pattern.sub("{*}", path)
        return path

    def match(self, pattern_path: str, concrete_path: str) -> bool:
        """Check if a concrete path matches a pattern path."""
        # Normalize both paths
        pattern = self.normalize(pattern_path)
        concrete = self.normalize(concrete_path)

        # Split into segments
        pattern_parts = pattern.split("/")
        concrete_parts = concrete.split("/")

        if len(pattern_parts) != len(concrete_parts):
            return False

        for pattern_part, concrete_part in zip(pattern_parts, concrete_parts):
            if pattern_part == "{*}":
                continue  # Parameter matches anything
            if pattern_part != concrete_part:
                return False

        return True


# =============================================================================
# API Boundary Linker
# =============================================================================


class APIBoundaryLinker:
    """Links API endpoints with their clients."""

    def __init__(self):
        self._path_matcher = PathMatcher()

    def link(
        self,
        endpoints: list[APIEndpoint],
        clients: list[APIClient],
    ) -> list[APIBoundaryLink]:
        """Link endpoints with matching clients."""
        links: list[APIBoundaryLink] = []

        for endpoint in endpoints:
            endpoint_norm = self._path_matcher.normalize(endpoint.path)

            for client in clients:
                # Method must match
                if endpoint.method != client.method:
                    continue

                # Path must match
                client_norm = self._path_matcher.normalize(client.path)
                if endpoint_norm == client_norm or self._path_matcher.match(
                    endpoint.path, client.path
                ):
                    links.append(APIBoundaryLink(endpoint=endpoint, client=client))

        return links


# =============================================================================
# API Boundary Projection
# =============================================================================


class APIBoundaryProjection:
    """Projects API boundaries to graph."""

    def __init__(self, projection: GraphProjection):
        self._projection = projection

    def create_endpoint_node(self, endpoint: APIEndpoint) -> APIBoundaryNode:
        """Create CODE_APIEndpoint node in graph."""
        builder = (
            APIEndpointNodeBuilder()
            .method(endpoint.method)
            .path(endpoint.path)
            .handler_name(endpoint.handler_name)
            .file_path(endpoint.file_path)
            .line_number(endpoint.line_number)
            .framework(endpoint.framework)
        )

        if endpoint.handler_scip_id:
            builder.handler_scip_id(endpoint.handler_scip_id)

        node = builder.build()

        # Add to graph using the underlying driver
        code_node = CodeNode(
            node_type=CodeNodeType.SYMBOL,  # Use SYMBOL as base type
            id=node.id,
            properties={
                **node.properties,
                "kind": "api_endpoint",
            },
        )
        self._projection.driver.add_node(code_node)

        return node

    def create_client_node(self, client: APIClient) -> APIBoundaryNode:
        """Create CODE_APIClient node in graph."""
        builder = (
            APIClientNodeBuilder()
            .method(client.method)
            .path(client.path)
            .caller_name(client.caller_name)
            .file_path(client.file_path)
            .line_number(client.line_number)
            .client_type(client.client_type)
        )

        if client.caller_scip_id:
            builder.caller_scip_id(client.caller_scip_id)

        node = builder.build()

        # Add to graph using the underlying driver
        code_node = CodeNode(
            node_type=CodeNodeType.SYMBOL,  # Use SYMBOL as base type
            id=node.id,
            properties={
                **node.properties,
                "kind": "api_client",
            },
        )
        self._projection.driver.add_node(code_node)

        return node

    def create_exposes_edge(
        self,
        handler_id: str,
        endpoint_id: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> APIBoundaryEdge:
        """Create EXPOSES edge from handler to endpoint."""
        builder = (
            APIBoundaryEdgeBuilder()
            .exposes()
            .from_handler(handler_id)
            .to_endpoint(endpoint_id)
        )

        if properties:
            for key, value in properties.items():
                builder.with_property(key, value)

        edge = builder.build()

        # Add to graph - use custom edge type handling
        code_edge = CodeEdge(
            edge_type=CodeEdgeType.DEFINES,  # Use DEFINES as closest semantic match
            source_id=handler_id,
            target_id=endpoint_id,
            properties={
                **(properties or {}),
                "boundary_type": "EXPOSES",
            },
        )
        self._projection.driver.add_edge(code_edge)

        return edge

    def create_consumes_edge(
        self,
        client_id: str,
        endpoint_id: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> APIBoundaryEdge:
        """Create CONSUMES edge from client to endpoint."""
        builder = (
            APIBoundaryEdgeBuilder()
            .consumes()
            .from_client(client_id)
            .to_endpoint(endpoint_id)
        )

        if properties:
            for key, value in properties.items():
                builder.with_property(key, value)

        edge = builder.build()

        # Add to graph - use custom edge type handling
        code_edge = CodeEdge(
            edge_type=CodeEdgeType.CALLS,  # Use CALLS as closest semantic match
            source_id=client_id,
            target_id=endpoint_id,
            properties={
                **(properties or {}),
                "boundary_type": "CONSUMES",
            },
        )
        self._projection.driver.add_edge(code_edge)

        return edge


# =============================================================================
# API Boundary Analyzer
# =============================================================================


class APIBoundaryAnalyzer:
    """Full API boundary analysis pipeline."""

    def __init__(self, projection: GraphProjection):
        self._projection = projection
        self._api_projection = APIBoundaryProjection(projection)
        self._endpoint_detector = RestEndpointDetector()
        self._client_detector = APIClientDetector()
        self._linker = APIBoundaryLinker()
        self._endpoints: list[APIEndpoint] = []
        self._clients: list[APIClient] = []

    def analyze_python_file(
        self,
        code: str,
        file_path: Path,
    ) -> APIBoundaryAnalysisResult:
        """Analyze Python file for REST endpoints."""
        errors: list[str] = []

        try:
            endpoints = self._endpoint_detector.detect_python(code, file_path)
            self._endpoints.extend(endpoints)

            # Project endpoints to graph
            for endpoint in endpoints:
                try:
                    self._api_projection.create_endpoint_node(endpoint)
                except Exception as e:
                    logger.warning(f"Failed to project endpoint {endpoint.unique_id}: {e}")

        except Exception as e:
            errors.append(f"Failed to analyze {file_path}: {e}")

        return APIBoundaryAnalysisResult(
            endpoints_found=len(endpoints) if "endpoints" in dir() else 0,
            clients_found=0,
            links_created=0,
            errors=errors,
        )

    def analyze_typescript_file(
        self,
        code: str,
        file_path: Path,
    ) -> APIBoundaryAnalysisResult:
        """Analyze TypeScript file for API clients."""
        errors: list[str] = []

        try:
            clients = self._client_detector.detect_typescript(code, file_path)
            self._clients.extend(clients)

            # Project clients to graph
            for client in clients:
                try:
                    self._api_projection.create_client_node(client)
                except Exception as e:
                    logger.warning(f"Failed to project client: {e}")

        except Exception as e:
            errors.append(f"Failed to analyze {file_path}: {e}")

        return APIBoundaryAnalysisResult(
            endpoints_found=0,
            clients_found=len(clients) if "clients" in dir() else 0,
            links_created=0,
            errors=errors,
        )

    def link_boundaries(self) -> list[APIBoundaryLink]:
        """Link detected endpoints and clients."""
        links = self._linker.link(self._endpoints, self._clients)

        # Create edges for links
        for link in links:
            try:
                endpoint_id = link.endpoint.unique_id
                client_id = f"{link.client.client_type}:{link.client.method.value}:{link.client.path}:{link.client.file_path}:{link.client.line_number}"

                # If we have handler SCIP ID, create EXPOSES edge
                if link.endpoint.handler_scip_id:
                    self._api_projection.create_exposes_edge(
                        link.endpoint.handler_scip_id,
                        endpoint_id,
                    )

                # Create CONSUMES edge
                self._api_projection.create_consumes_edge(
                    client_id,
                    endpoint_id,
                )
            except Exception as e:
                logger.warning(f"Failed to create link edge: {e}")

        return links


# =============================================================================
# Factory Functions
# =============================================================================


def create_api_boundary_analyzer(projection: GraphProjection) -> APIBoundaryAnalyzer:
    """Create an API boundary analyzer."""
    return APIBoundaryAnalyzer(projection)


def create_rest_endpoint_detector() -> RestEndpointDetector:
    """Create a REST endpoint detector."""
    return RestEndpointDetector()


def create_api_client_detector() -> APIClientDetector:
    """Create an API client detector."""
    return APIClientDetector()
