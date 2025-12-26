"""Tests for cross-language API boundary detection (FR-003).

This module tests:
- REST endpoint detection in Python (FastAPI, Flask decorators)
- API client detection in TypeScript (fetch, axios)
- EXPOSES edges from handler symbols to API endpoints
- CONSUMES edges from client code to API endpoints
- HTTP method and path extraction
- Path parameter detection (/users/{id})
- Integration with SCIP symbol IDs
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

from openmemory.api.indexing.ast_parser import Language, Symbol, SymbolType
from openmemory.api.indexing.scip_symbols import (
    SCIPScheme,
    SCIPSymbolID,
    SCIPDescriptor,
)
from openmemory.api.indexing.graph_projection import (
    CodeNode,
    CodeNodeType,
    CodeEdge,
    CodeEdgeType,
    MemoryGraphStore,
    GraphProjection,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_store() -> MemoryGraphStore:
    """Create in-memory graph store for testing."""
    return MemoryGraphStore()


@pytest.fixture
def projection(memory_store: MemoryGraphStore) -> GraphProjection:
    """Create graph projection with memory store."""
    return GraphProjection(driver=memory_store)


# =============================================================================
# Test HTTP Method Enum
# =============================================================================


class TestHTTPMethod:
    """Tests for HTTPMethod enum."""

    def test_get_method(self):
        """Test GET method."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.GET.value == "GET"

    def test_post_method(self):
        """Test POST method."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.POST.value == "POST"

    def test_put_method(self):
        """Test PUT method."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.PUT.value == "PUT"

    def test_delete_method(self):
        """Test DELETE method."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.DELETE.value == "DELETE"

    def test_patch_method(self):
        """Test PATCH method."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.PATCH.value == "PATCH"

    def test_from_string(self):
        """Test parsing from string."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.from_string("get") == HTTPMethod.GET
        assert HTTPMethod.from_string("POST") == HTTPMethod.POST
        assert HTTPMethod.from_string("Delete") == HTTPMethod.DELETE

    def test_from_string_unknown(self):
        """Test parsing unknown method."""
        from openmemory.api.indexing.api_boundaries import HTTPMethod

        assert HTTPMethod.from_string("OPTIONS") is None


# =============================================================================
# Test APIEndpoint Data Class
# =============================================================================


class TestAPIEndpoint:
    """Tests for APIEndpoint data class."""

    def test_create_endpoint(self):
        """Test creating an API endpoint."""
        from openmemory.api.indexing.api_boundaries import APIEndpoint, HTTPMethod

        endpoint = APIEndpoint(
            method=HTTPMethod.GET,
            path="/users/{id}",
            handler_name="get_user",
            handler_scip_id=None,
            file_path=Path("/api/users.py"),
            line_number=10,
        )

        assert endpoint.method == HTTPMethod.GET
        assert endpoint.path == "/users/{id}"
        assert endpoint.handler_name == "get_user"
        assert endpoint.line_number == 10

    def test_endpoint_path_params(self):
        """Test extracting path parameters."""
        from openmemory.api.indexing.api_boundaries import APIEndpoint, HTTPMethod

        endpoint = APIEndpoint(
            method=HTTPMethod.GET,
            path="/users/{user_id}/posts/{post_id}",
            handler_name="get_user_post",
            handler_scip_id=None,
            file_path=Path("/api/posts.py"),
            line_number=20,
        )

        params = endpoint.path_params
        assert params == ["user_id", "post_id"]

    def test_endpoint_no_path_params(self):
        """Test endpoint without path parameters."""
        from openmemory.api.indexing.api_boundaries import APIEndpoint, HTTPMethod

        endpoint = APIEndpoint(
            method=HTTPMethod.GET,
            path="/health",
            handler_name="health_check",
            handler_scip_id=None,
            file_path=Path("/api/health.py"),
            line_number=5,
        )

        assert endpoint.path_params == []

    def test_endpoint_unique_id(self):
        """Test endpoint unique ID generation."""
        from openmemory.api.indexing.api_boundaries import APIEndpoint, HTTPMethod

        endpoint = APIEndpoint(
            method=HTTPMethod.POST,
            path="/users",
            handler_name="create_user",
            handler_scip_id=None,
            file_path=Path("/api/users.py"),
            line_number=30,
        )

        assert endpoint.unique_id == "POST:/users"


class TestAPIClient:
    """Tests for APIClient data class."""

    def test_create_client(self):
        """Test creating an API client."""
        from openmemory.api.indexing.api_boundaries import APIClient, HTTPMethod

        client = APIClient(
            method=HTTPMethod.GET,
            path="/api/users/{id}",
            caller_name="fetchUser",
            caller_scip_id=None,
            file_path=Path("/src/api.ts"),
            line_number=15,
            client_type="fetch",
        )

        assert client.method == HTTPMethod.GET
        assert client.path == "/api/users/{id}"
        assert client.caller_name == "fetchUser"
        assert client.client_type == "fetch"

    def test_client_target_endpoint_id(self):
        """Test client target endpoint ID."""
        from openmemory.api.indexing.api_boundaries import APIClient, HTTPMethod

        client = APIClient(
            method=HTTPMethod.POST,
            path="/api/users",
            caller_name="createUser",
            caller_scip_id=None,
            file_path=Path("/src/api.ts"),
            line_number=25,
            client_type="axios",
        )

        assert client.target_endpoint_id == "POST:/api/users"


# =============================================================================
# Test REST Endpoint Detection - FastAPI
# =============================================================================


class TestFastAPIEndpointDetection:
    """Tests for FastAPI endpoint detection."""

    def test_detect_fastapi_get_endpoint(self):
        """Test detecting FastAPI GET endpoint."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/main.py"))

        assert len(endpoints) == 1
        assert endpoints[0].method == HTTPMethod.GET
        assert endpoints[0].path == "/users/{user_id}"
        assert endpoints[0].handler_name == "get_user"

    def test_detect_fastapi_post_endpoint(self):
        """Test detecting FastAPI POST endpoint."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from fastapi import FastAPI

app = FastAPI()

@app.post("/users")
async def create_user(user: UserCreate):
    return {"id": 1, "name": user.name}
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/main.py"))

        assert len(endpoints) == 1
        assert endpoints[0].method == HTTPMethod.POST
        assert endpoints[0].path == "/users"
        assert endpoints[0].handler_name == "create_user"

    def test_detect_fastapi_router_endpoint(self):
        """Test detecting FastAPI APIRouter endpoint."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from fastapi import APIRouter

router = APIRouter()

@router.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}

@router.delete("/items/{item_id}")
def delete_item(item_id: int):
    return {"deleted": True}
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/items.py"))

        assert len(endpoints) == 2
        assert endpoints[0].method == HTTPMethod.GET
        assert endpoints[0].path == "/items/{item_id}"
        assert endpoints[1].method == HTTPMethod.DELETE
        assert endpoints[1].path == "/items/{item_id}"

    def test_detect_fastapi_all_http_methods(self):
        """Test detecting all HTTP methods in FastAPI."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from fastapi import APIRouter

router = APIRouter()

@router.get("/resource")
def get_resource(): pass

@router.post("/resource")
def create_resource(): pass

@router.put("/resource/{id}")
def update_resource(id: int): pass

@router.patch("/resource/{id}")
def patch_resource(id: int): pass

@router.delete("/resource/{id}")
def delete_resource(id: int): pass
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/resource.py"))

        assert len(endpoints) == 5
        methods = [e.method for e in endpoints]
        assert HTTPMethod.GET in methods
        assert HTTPMethod.POST in methods
        assert HTTPMethod.PUT in methods
        assert HTTPMethod.PATCH in methods
        assert HTTPMethod.DELETE in methods


class TestFlaskEndpointDetection:
    """Tests for Flask endpoint detection."""

    def test_detect_flask_route_with_methods(self):
        """Test detecting Flask route with methods."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from flask import Flask

app = Flask(__name__)

@app.route("/users", methods=["GET"])
def list_users():
    return {"users": []}

@app.route("/users", methods=["POST"])
def create_user():
    return {"id": 1}
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/flask_app.py"))

        assert len(endpoints) == 2
        assert endpoints[0].method == HTTPMethod.GET
        assert endpoints[0].path == "/users"
        assert endpoints[1].method == HTTPMethod.POST
        assert endpoints[1].path == "/users"

    def test_detect_flask_route_default_get(self):
        """Test detecting Flask route with default GET method."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from flask import Flask

app = Flask(__name__)

@app.route("/health")
def health_check():
    return {"status": "ok"}
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/flask_app.py"))

        assert len(endpoints) == 1
        assert endpoints[0].method == HTTPMethod.GET
        assert endpoints[0].path == "/health"

    def test_detect_flask_blueprint_route(self):
        """Test detecting Flask Blueprint routes."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            HTTPMethod,
        )

        code = '''
from flask import Blueprint

bp = Blueprint("users", __name__)

@bp.route("/users/<int:user_id>", methods=["GET"])
def get_user(user_id):
    return {"user_id": user_id}
'''

        detector = RestEndpointDetector()
        endpoints = detector.detect_python(code, Path("/api/users_bp.py"))

        assert len(endpoints) == 1
        assert endpoints[0].method == HTTPMethod.GET
        # Flask uses <type:name> syntax
        assert "/users/" in endpoints[0].path


# =============================================================================
# Test API Client Detection - TypeScript
# =============================================================================


class TestFetchAPIClientDetection:
    """Tests for fetch() API client detection."""

    def test_detect_fetch_get_call(self):
        """Test detecting fetch GET call."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
async function getUser(id: number) {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) == 1
        assert clients[0].method == HTTPMethod.GET
        assert "/api/users/" in clients[0].path
        assert clients[0].client_type == "fetch"

    def test_detect_fetch_post_call(self):
        """Test detecting fetch POST call."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
async function createUser(data: UserData) {
    const response = await fetch('/api/users', {
        method: 'POST',
        body: JSON.stringify(data)
    });
    return response.json();
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) == 1
        assert clients[0].method == HTTPMethod.POST
        assert clients[0].path == "/api/users"

    def test_detect_fetch_with_variable_method(self):
        """Test detecting fetch with method in options."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
async function deleteUser(id: number) {
    await fetch(`/api/users/${id}`, {
        method: 'DELETE'
    });
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) == 1
        assert clients[0].method == HTTPMethod.DELETE


class TestAxiosAPIClientDetection:
    """Tests for axios API client detection."""

    def test_detect_axios_get_call(self):
        """Test detecting axios.get() call."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
import axios from 'axios';

async function getUsers() {
    const response = await axios.get('/api/users');
    return response.data;
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) == 1
        assert clients[0].method == HTTPMethod.GET
        assert clients[0].path == "/api/users"
        assert clients[0].client_type == "axios"

    def test_detect_axios_post_call(self):
        """Test detecting axios.post() call."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
import axios from 'axios';

async function createPost(data: PostData) {
    const response = await axios.post('/api/posts', data);
    return response.data;
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) == 1
        assert clients[0].method == HTTPMethod.POST
        assert clients[0].path == "/api/posts"

    def test_detect_axios_put_and_delete(self):
        """Test detecting axios.put() and axios.delete() calls."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
import axios from 'axios';

async function updateUser(id: number, data: UserData) {
    await axios.put(`/api/users/${id}`, data);
}

async function deleteUser(id: number) {
    await axios.delete(`/api/users/${id}`);
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) == 2
        methods = [c.method for c in clients]
        assert HTTPMethod.PUT in methods
        assert HTTPMethod.DELETE in methods

    def test_detect_axios_instance_call(self):
        """Test detecting axios instance method calls."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientDetector,
            HTTPMethod,
        )

        code = '''
const api = axios.create({ baseURL: '/api' });

async function getItems() {
    const response = await api.get('/items');
    return response.data;
}
'''

        detector = APIClientDetector()
        clients = detector.detect_typescript(code, Path("/src/api.ts"))

        assert len(clients) >= 1
        # Should detect the api.get call


# =============================================================================
# Test Edge Types for API Boundaries
# =============================================================================


class TestAPIBoundaryEdgeTypes:
    """Tests for API boundary edge types in CodeEdgeType."""

    def test_exposes_edge_type(self):
        """Test EXPOSES edge type exists."""
        from openmemory.api.indexing.api_boundaries import APIBoundaryEdgeType

        assert APIBoundaryEdgeType.EXPOSES.value == "EXPOSES"

    def test_consumes_edge_type(self):
        """Test CONSUMES edge type exists."""
        from openmemory.api.indexing.api_boundaries import APIBoundaryEdgeType

        assert APIBoundaryEdgeType.CONSUMES.value == "CONSUMES"


# =============================================================================
# Test API Boundary Node Types
# =============================================================================


class TestAPIBoundaryNodeTypes:
    """Tests for API boundary node types."""

    def test_api_endpoint_node_type(self):
        """Test CODE_APIEndpoint node type."""
        from openmemory.api.indexing.api_boundaries import APIBoundaryNodeType

        assert APIBoundaryNodeType.API_ENDPOINT.value == "CODE_APIEndpoint"

    def test_api_client_node_type(self):
        """Test CODE_APIClient node type."""
        from openmemory.api.indexing.api_boundaries import APIBoundaryNodeType

        assert APIBoundaryNodeType.API_CLIENT.value == "CODE_APIClient"


# =============================================================================
# Test APIEndpoint Node Builder
# =============================================================================


class TestAPIEndpointNodeBuilder:
    """Tests for building CODE_APIEndpoint nodes."""

    def test_build_endpoint_node(self):
        """Test building an API endpoint node."""
        from openmemory.api.indexing.api_boundaries import (
            APIEndpointNodeBuilder,
            HTTPMethod,
            APIBoundaryNodeType,
        )

        builder = APIEndpointNodeBuilder()
        node = (
            builder.method(HTTPMethod.GET)
            .path("/users/{id}")
            .handler_name("get_user")
            .file_path(Path("/api/users.py"))
            .line_number(10)
            .build()
        )

        assert node.node_type.value == "CODE_APIEndpoint"
        assert node.properties["method"] == "GET"
        assert node.properties["path"] == "/users/{id}"
        assert node.properties["handler_name"] == "get_user"
        assert node.id == "GET:/users/{id}"

    def test_endpoint_node_with_scip_id(self):
        """Test building endpoint node with handler SCIP ID."""
        from openmemory.api.indexing.api_boundaries import (
            APIEndpointNodeBuilder,
            HTTPMethod,
        )

        scip_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON,
            "api.users",
            [SCIPDescriptor.term("get_user")],
        )

        builder = APIEndpointNodeBuilder()
        node = (
            builder.method(HTTPMethod.GET)
            .path("/users/{id}")
            .handler_name("get_user")
            .handler_scip_id(scip_id)
            .file_path(Path("/api/users.py"))
            .build()
        )

        assert node.properties["handler_scip_id"] == str(scip_id)


# =============================================================================
# Test APIClient Node Builder
# =============================================================================


class TestAPIClientNodeBuilder:
    """Tests for building CODE_APIClient nodes."""

    def test_build_client_node(self):
        """Test building an API client node."""
        from openmemory.api.indexing.api_boundaries import (
            APIClientNodeBuilder,
            HTTPMethod,
            APIBoundaryNodeType,
        )

        builder = APIClientNodeBuilder()
        node = (
            builder.method(HTTPMethod.GET)
            .path("/api/users/{id}")
            .caller_name("fetchUser")
            .file_path(Path("/src/api.ts"))
            .line_number(15)
            .client_type("fetch")
            .build()
        )

        assert node.node_type.value == "CODE_APIClient"
        assert node.properties["method"] == "GET"
        assert node.properties["path"] == "/api/users/{id}"
        assert node.properties["caller_name"] == "fetchUser"
        assert node.properties["client_type"] == "fetch"


# =============================================================================
# Test EXPOSES Edge Creation
# =============================================================================


class TestExposesEdgeCreation:
    """Tests for creating EXPOSES edges."""

    def test_create_exposes_edge(self):
        """Test creating EXPOSES edge from handler to endpoint."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryEdgeBuilder,
            APIBoundaryEdgeType,
        )

        edge = (
            APIBoundaryEdgeBuilder()
            .exposes()
            .from_handler("scip-python api.users get_user.")
            .to_endpoint("GET:/users/{id}")
            .build()
        )

        assert edge.edge_type == APIBoundaryEdgeType.EXPOSES
        assert edge.source_id == "scip-python api.users get_user."
        assert edge.target_id == "GET:/users/{id}"

    def test_exposes_edge_with_properties(self):
        """Test EXPOSES edge with additional properties."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryEdgeBuilder,
            APIBoundaryEdgeType,
        )

        edge = (
            APIBoundaryEdgeBuilder()
            .exposes()
            .from_handler("handler_id")
            .to_endpoint("POST:/users")
            .with_property("framework", "fastapi")
            .with_property("line", 42)
            .build()
        )

        assert edge.properties["framework"] == "fastapi"
        assert edge.properties["line"] == 42


# =============================================================================
# Test CONSUMES Edge Creation
# =============================================================================


class TestConsumesEdgeCreation:
    """Tests for creating CONSUMES edges."""

    def test_create_consumes_edge(self):
        """Test creating CONSUMES edge from client to endpoint."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryEdgeBuilder,
            APIBoundaryEdgeType,
        )

        edge = (
            APIBoundaryEdgeBuilder()
            .consumes()
            .from_client("scip-typescript src.api fetchUser.")
            .to_endpoint("GET:/api/users/{id}")
            .build()
        )

        assert edge.edge_type == APIBoundaryEdgeType.CONSUMES
        assert edge.source_id == "scip-typescript src.api fetchUser."
        assert edge.target_id == "GET:/api/users/{id}"

    def test_consumes_edge_with_client_type(self):
        """Test CONSUMES edge with client type property."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryEdgeBuilder,
            APIBoundaryEdgeType,
        )

        edge = (
            APIBoundaryEdgeBuilder()
            .consumes()
            .from_client("client_id")
            .to_endpoint("GET:/api/data")
            .with_property("client_type", "axios")
            .build()
        )

        assert edge.properties["client_type"] == "axios"


# =============================================================================
# Test API Boundary Linker
# =============================================================================


class TestAPIBoundaryLinker:
    """Tests for linking API endpoints and clients."""

    def test_link_matching_endpoints_and_clients(self):
        """Test linking endpoints with matching clients."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryLinker,
            APIEndpoint,
            APIClient,
            HTTPMethod,
        )

        endpoints = [
            APIEndpoint(
                method=HTTPMethod.GET,
                path="/api/users/{id}",
                handler_name="get_user",
                handler_scip_id=None,
                file_path=Path("/api/users.py"),
                line_number=10,
            ),
            APIEndpoint(
                method=HTTPMethod.POST,
                path="/api/users",
                handler_name="create_user",
                handler_scip_id=None,
                file_path=Path("/api/users.py"),
                line_number=20,
            ),
        ]

        clients = [
            APIClient(
                method=HTTPMethod.GET,
                path="/api/users/{id}",
                caller_name="fetchUser",
                caller_scip_id=None,
                file_path=Path("/src/api.ts"),
                line_number=15,
                client_type="fetch",
            ),
        ]

        linker = APIBoundaryLinker()
        links = linker.link(endpoints, clients)

        assert len(links) == 1
        assert links[0].endpoint.handler_name == "get_user"
        assert links[0].client.caller_name == "fetchUser"

    def test_link_with_path_normalization(self):
        """Test linking with path normalization (trailing slash, etc)."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryLinker,
            APIEndpoint,
            APIClient,
            HTTPMethod,
        )

        endpoints = [
            APIEndpoint(
                method=HTTPMethod.GET,
                path="/api/users/",
                handler_name="list_users",
                handler_scip_id=None,
                file_path=Path("/api/users.py"),
                line_number=10,
            ),
        ]

        clients = [
            APIClient(
                method=HTTPMethod.GET,
                path="/api/users",  # No trailing slash
                caller_name="fetchUsers",
                caller_scip_id=None,
                file_path=Path("/src/api.ts"),
                line_number=15,
                client_type="fetch",
            ),
        ]

        linker = APIBoundaryLinker()
        links = linker.link(endpoints, clients)

        assert len(links) == 1

    def test_no_link_for_mismatched_methods(self):
        """Test that mismatched methods don't link."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryLinker,
            APIEndpoint,
            APIClient,
            HTTPMethod,
        )

        endpoints = [
            APIEndpoint(
                method=HTTPMethod.GET,
                path="/api/users",
                handler_name="list_users",
                handler_scip_id=None,
                file_path=Path("/api/users.py"),
                line_number=10,
            ),
        ]

        clients = [
            APIClient(
                method=HTTPMethod.POST,  # Different method
                path="/api/users",
                caller_name="createUser",
                caller_scip_id=None,
                file_path=Path("/src/api.ts"),
                line_number=15,
                client_type="fetch",
            ),
        ]

        linker = APIBoundaryLinker()
        links = linker.link(endpoints, clients)

        assert len(links) == 0


# =============================================================================
# Test Path Parameter Matching
# =============================================================================


class TestPathParameterMatching:
    """Tests for path parameter matching."""

    def test_match_path_with_parameters(self):
        """Test matching paths with parameters."""
        from openmemory.api.indexing.api_boundaries import PathMatcher

        matcher = PathMatcher()

        # FastAPI style
        assert matcher.match("/users/{id}", "/users/123")
        assert matcher.match("/users/{user_id}/posts/{post_id}", "/users/1/posts/2")

        # Should not match different paths
        assert not matcher.match("/users/{id}", "/posts/123")

    def test_match_path_with_flask_syntax(self):
        """Test matching paths with Flask-style parameters."""
        from openmemory.api.indexing.api_boundaries import PathMatcher

        matcher = PathMatcher()

        # Flask uses <type:name> syntax
        assert matcher.match("/users/<int:id>", "/users/123")
        assert matcher.match("/users/<id>", "/users/abc")

    def test_normalize_path_parameters(self):
        """Test normalizing path parameters."""
        from openmemory.api.indexing.api_boundaries import PathMatcher

        matcher = PathMatcher()

        # Both should normalize to the same pattern
        assert matcher.normalize("/users/{id}") == matcher.normalize("/users/<id>")
        assert matcher.normalize("/users/{user_id}") == matcher.normalize(
            "/users/<int:user_id>"
        )


# =============================================================================
# Test Integration with Graph Projection
# =============================================================================


class TestAPIBoundaryGraphIntegration:
    """Tests for integrating API boundaries with graph projection."""

    def test_project_api_endpoint_to_graph(self, projection: GraphProjection):
        """Test projecting API endpoint to graph."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryProjection,
            APIEndpoint,
            HTTPMethod,
        )

        endpoint = APIEndpoint(
            method=HTTPMethod.GET,
            path="/users/{id}",
            handler_name="get_user",
            handler_scip_id=None,
            file_path=Path("/api/users.py"),
            line_number=10,
        )

        api_projection = APIBoundaryProjection(projection)
        node = api_projection.create_endpoint_node(endpoint)

        assert projection.driver.node_count == 1
        assert node.id == "GET:/users/{id}"

    def test_project_api_client_to_graph(self, projection: GraphProjection):
        """Test projecting API client to graph."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryProjection,
            APIClient,
            HTTPMethod,
        )

        client = APIClient(
            method=HTTPMethod.GET,
            path="/api/users/{id}",
            caller_name="fetchUser",
            caller_scip_id=None,
            file_path=Path("/src/api.ts"),
            line_number=15,
            client_type="fetch",
        )

        api_projection = APIBoundaryProjection(projection)
        node = api_projection.create_client_node(client)

        assert projection.driver.node_count == 1
        assert "fetch" in str(node.id) or "fetchUser" in str(node.id)

    def test_create_exposes_edge_in_graph(self, projection: GraphProjection):
        """Test creating EXPOSES edge in graph."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryProjection,
            APIBoundaryEdgeType,
            APIEndpoint,
            HTTPMethod,
        )

        # Create handler symbol node first
        handler_scip_id = SCIPSymbolID(
            SCIPScheme.SCIP_PYTHON,
            "api.users",
            [SCIPDescriptor.term("get_user")],
        )
        handler_symbol = Symbol(
            name="get_user",
            symbol_type=SymbolType.FUNCTION,
            line_start=10,
            line_end=20,
            language=Language.PYTHON,
        )
        projection.create_symbol_node(handler_symbol, handler_scip_id, Path("/api/users.py"))

        endpoint = APIEndpoint(
            method=HTTPMethod.GET,
            path="/users/{id}",
            handler_name="get_user",
            handler_scip_id=str(handler_scip_id),
            file_path=Path("/api/users.py"),
            line_number=10,
        )

        api_projection = APIBoundaryProjection(projection)
        endpoint_node = api_projection.create_endpoint_node(endpoint)
        edge = api_projection.create_exposes_edge(str(handler_scip_id), endpoint_node.id)

        assert projection.driver.edge_count == 1
        # The edge is stored as CodeEdgeType.DEFINES in the underlying graph
        assert projection.driver.has_edge(
            str(handler_scip_id),
            endpoint_node.id,
            CodeEdgeType.DEFINES,  # Underlying edge type used for storage
        )
        # But our edge object has the logical type
        assert edge.edge_type == APIBoundaryEdgeType.EXPOSES

    def test_create_consumes_edge_in_graph(self, projection: GraphProjection):
        """Test creating CONSUMES edge in graph."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryProjection,
            APIClient,
            APIEndpoint,
            HTTPMethod,
        )

        # Create endpoint node
        endpoint = APIEndpoint(
            method=HTTPMethod.GET,
            path="/api/users/{id}",
            handler_name="get_user",
            handler_scip_id=None,
            file_path=Path("/api/users.py"),
            line_number=10,
        )

        # Create client symbol node
        client_scip_id = SCIPSymbolID(
            SCIPScheme.SCIP_TYPESCRIPT,
            "src.api",
            [SCIPDescriptor.term("fetchUser")],
        )
        client_symbol = Symbol(
            name="fetchUser",
            symbol_type=SymbolType.FUNCTION,
            line_start=15,
            line_end=20,
            language=Language.TYPESCRIPT,
        )
        projection.create_symbol_node(client_symbol, client_scip_id, Path("/src/api.ts"))

        client = APIClient(
            method=HTTPMethod.GET,
            path="/api/users/{id}",
            caller_name="fetchUser",
            caller_scip_id=str(client_scip_id),
            file_path=Path("/src/api.ts"),
            line_number=15,
            client_type="fetch",
        )

        api_projection = APIBoundaryProjection(projection)
        endpoint_node = api_projection.create_endpoint_node(endpoint)
        client_node = api_projection.create_client_node(client)
        edge = api_projection.create_consumes_edge(client_node.id, endpoint_node.id)

        assert projection.driver.edge_count == 1


# =============================================================================
# Test Full Pipeline
# =============================================================================


class TestAPIBoundaryPipeline:
    """Tests for full API boundary detection pipeline."""

    def test_detect_and_project_python_endpoints(self, projection: GraphProjection):
        """Test detecting Python endpoints and projecting to graph."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryAnalyzer,
        )

        python_code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}

@app.post("/users")
def create_user(user: UserCreate):
    return {"id": 1}
'''

        analyzer = APIBoundaryAnalyzer(projection)
        result = analyzer.analyze_python_file(
            python_code,
            Path("/api/users.py"),
        )

        assert result.endpoints_found == 2
        assert result.clients_found == 0

    def test_detect_and_project_typescript_clients(self, projection: GraphProjection):
        """Test detecting TypeScript clients and projecting to graph."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryAnalyzer,
        )

        ts_code = '''
async function fetchUser(id: number) {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}

async function createUser(data: UserData) {
    const response = await fetch('/api/users', {
        method: 'POST',
        body: JSON.stringify(data)
    });
    return response.json();
}
'''

        analyzer = APIBoundaryAnalyzer(projection)
        result = analyzer.analyze_typescript_file(
            ts_code,
            Path("/src/api.ts"),
        )

        assert result.endpoints_found == 0
        assert result.clients_found == 2

    def test_cross_language_boundary_detection(self, projection: GraphProjection):
        """Test full cross-language API boundary detection."""
        from openmemory.api.indexing.api_boundaries import (
            APIBoundaryAnalyzer,
        )

        python_code = '''
from fastapi import FastAPI

app = FastAPI()

@app.get("/api/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
'''

        ts_code = '''
async function fetchUser(id: number) {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
}
'''

        analyzer = APIBoundaryAnalyzer(projection)

        # Analyze both files
        analyzer.analyze_python_file(python_code, Path("/api/users.py"))
        analyzer.analyze_typescript_file(ts_code, Path("/src/api.ts"))

        # Link endpoints and clients
        links = analyzer.link_boundaries()

        assert len(links) >= 1


# =============================================================================
# Test Error Handling
# =============================================================================


class TestAPIBoundaryErrorHandling:
    """Tests for error handling in API boundary detection."""

    def test_handle_malformed_python_code(self):
        """Test handling malformed Python code."""
        from openmemory.api.indexing.api_boundaries import RestEndpointDetector

        code = '''
@app.get("/users"
def broken_syntax():
    pass
'''

        detector = RestEndpointDetector()
        # Should not raise, return empty or partial results
        endpoints = detector.detect_python(code, Path("/api/broken.py"))
        # May return empty list due to syntax error
        assert isinstance(endpoints, list)

    def test_handle_malformed_typescript_code(self):
        """Test handling malformed TypeScript code."""
        from openmemory.api.indexing.api_boundaries import APIClientDetector

        code = '''
async function broken(
    const response = await fetch('/api/data'
'''

        detector = APIClientDetector()
        # Should not raise, return empty or partial results
        clients = detector.detect_typescript(code, Path("/src/broken.ts"))
        assert isinstance(clients, list)

    def test_handle_empty_code(self):
        """Test handling empty code."""
        from openmemory.api.indexing.api_boundaries import (
            RestEndpointDetector,
            APIClientDetector,
        )

        detector = RestEndpointDetector()
        assert detector.detect_python("", Path("/empty.py")) == []

        client_detector = APIClientDetector()
        assert client_detector.detect_typescript("", Path("/empty.ts")) == []


# =============================================================================
# Test Statistics and Results
# =============================================================================


class TestAPIBoundaryStatistics:
    """Tests for API boundary detection statistics."""

    def test_analysis_result_structure(self):
        """Test analysis result structure."""
        from openmemory.api.indexing.api_boundaries import APIBoundaryAnalysisResult

        result = APIBoundaryAnalysisResult(
            endpoints_found=5,
            clients_found=3,
            links_created=2,
            errors=[],
        )

        assert result.endpoints_found == 5
        assert result.clients_found == 3
        assert result.links_created == 2
        assert result.errors == []

    def test_analysis_result_with_errors(self):
        """Test analysis result with errors."""
        from openmemory.api.indexing.api_boundaries import APIBoundaryAnalysisResult

        result = APIBoundaryAnalysisResult(
            endpoints_found=1,
            clients_found=0,
            links_created=0,
            errors=["Failed to parse /broken.py: syntax error"],
        )

        assert len(result.errors) == 1
        assert "syntax error" in result.errors[0]


# =============================================================================
# Test Factory Function
# =============================================================================


class TestAPIBoundaryFactory:
    """Tests for API boundary factory functions."""

    def test_create_analyzer(self, projection: GraphProjection):
        """Test creating API boundary analyzer."""
        from openmemory.api.indexing.api_boundaries import create_api_boundary_analyzer

        analyzer = create_api_boundary_analyzer(projection)
        assert analyzer is not None

    def test_create_detector(self):
        """Test creating REST endpoint detector."""
        from openmemory.api.indexing.api_boundaries import create_rest_endpoint_detector

        detector = create_rest_endpoint_detector()
        assert detector is not None

    def test_create_client_detector(self):
        """Test creating API client detector."""
        from openmemory.api.indexing.api_boundaries import create_api_client_detector

        detector = create_api_client_detector()
        assert detector is not None
