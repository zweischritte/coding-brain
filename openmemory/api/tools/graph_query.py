"""Read-only graph query validation and serialization helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.security.types import Principal

try:  # Optional dependency for serialization
    from neo4j.graph import Node, Relationship, Path
except Exception:  # pragma: no cover - optional dependency
    Node = None
    Relationship = None
    Path = None


_FORBIDDEN_KEYWORDS = {
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "DETACH",
    "REMOVE",
    "DROP",
    "CALL",
    "LOAD",
    "FOREACH",
    "APOC",
    "DBMS",
    "SHOW",
    "PROFILE",
    "EXPLAIN",
    "COMMIT",
    "ROLLBACK",
    "TRANSACTION",
    "CONSTRAINT",
    "INDEX",
}

_LABEL_RE = re.compile(r":`([^`]+)`|:([A-Za-z_][A-Za-z0-9_]*)")
_LIMIT_LITERAL_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)
_LIMIT_PARAM_RE = re.compile(r"\blimit\s+\$(\w+)\b", re.IGNORECASE)


class GraphQueryError(ValueError):
    """Raised when a graph query fails validation."""

    def __init__(self, message: str, code: str = "INVALID_QUERY"):
        super().__init__(message)
        self.code = code


@dataclass
class GraphQueryInput:
    """Input parameters for graph_query."""

    scope: str
    query: str
    params: dict[str, Any] = field(default_factory=dict)
    repo_id: Optional[str] = None
    access_entity: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 50


@dataclass
class GraphQueryPrepared:
    """Validated graph query with safe parameters."""

    query: str
    params: dict[str, Any]
    limit: int


def prepare_graph_query(
    input_data: GraphQueryInput,
    principal: Principal,
    max_limit: int = 200,
) -> GraphQueryPrepared:
    """Validate and normalize a graph query."""
    scope = (input_data.scope or "").strip().lower()
    if scope not in {"code", "memory", "mem0"}:
        raise GraphQueryError("scope must be one of: code, memory, mem0", code="INVALID_SCOPE")

    query = _normalize_query(input_data.query)
    _validate_read_only(query)

    limit = _normalize_limit(input_data.limit, max_limit)
    params = dict(input_data.params or {})

    labels = _extract_labels(query)
    _validate_labels(scope, labels)

    if scope == "code":
        if not input_data.repo_id:
            raise GraphQueryError("repo_id is required for scope=code", code="MISSING_REPO_ID")
        _require_param(params, "repo_id", input_data.repo_id)
        _require_param(params, "repoId", input_data.repo_id)
        _require_repo_filter(query)

    if scope == "memory":
        if not input_data.access_entity:
            raise GraphQueryError(
                "access_entity is required for scope=memory",
                code="MISSING_ACCESS_ENTITY",
            )
        if not principal.can_access(input_data.access_entity):
            raise GraphQueryError(
                "access_entity is not permitted for this principal",
                code="FORBIDDEN_ACCESS_ENTITY",
            )
        _require_param(params, "access_entity", input_data.access_entity)
        _require_param(params, "accessEntity", input_data.access_entity)
        _require_access_entity_filter(query)

    if scope == "mem0":
        user_id = input_data.user_id or principal.user_id
        if user_id != principal.user_id and not principal.has_scope("admin:read"):
            raise GraphQueryError("user_id is not permitted for this principal", code="FORBIDDEN_USER_ID")
        _require_param(params, "user_id", user_id)
        _require_param(params, "userId", user_id)
        _require_user_id_filter(query)

    query = _ensure_limit(query, params, limit)
    return GraphQueryPrepared(query=query, params=params, limit=limit)


def serialize_record(record: Any) -> dict[str, Any]:
    """Serialize a Neo4j record into JSON-friendly primitives."""
    if hasattr(record, "data"):
        data = record.data()
    else:
        data = dict(record)
    return {key: _serialize_value(value) for key, value in data.items()}


def _serialize_value(value: Any) -> Any:
    if Node is not None and isinstance(value, Node):
        return {
            "type": "node",
            "id": value.element_id if hasattr(value, "element_id") else value.id,
            "labels": list(value.labels),
            "properties": dict(value),
        }
    if Relationship is not None and isinstance(value, Relationship):
        return {
            "type": "relationship",
            "id": value.element_id if hasattr(value, "element_id") else value.id,
            "rel_type": value.type,
            "start_id": value.start_node.element_id if hasattr(value.start_node, "element_id") else value.start_node.id,
            "end_id": value.end_node.element_id if hasattr(value.end_node, "element_id") else value.end_node.id,
            "properties": dict(value),
        }
    if Path is not None and isinstance(value, Path):
        return {
            "type": "path",
            "nodes": [_serialize_value(node) for node in value.nodes],
            "relationships": [_serialize_value(rel) for rel in value.relationships],
        }
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _normalize_query(query: str) -> str:
    if not isinstance(query, str):
        raise GraphQueryError("query must be a string", code="INVALID_QUERY")
    stripped = query.strip()
    if not stripped:
        raise GraphQueryError("query cannot be empty", code="INVALID_QUERY")
    # Remove line and block comments for keyword scanning.
    stripped = re.sub(r"//.*", "", stripped)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S)
    return stripped.strip()


def _validate_read_only(query: str) -> None:
    if ";" in query:
        raise GraphQueryError("only single-statement queries are allowed", code="INVALID_QUERY")

    upper = query.upper()
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise GraphQueryError(f"read-only queries only (found {keyword})", code="READ_ONLY_REQUIRED")

    if not re.search(r"\bRETURN\b", upper):
        raise GraphQueryError("query must include RETURN", code="INVALID_QUERY")

    if not re.match(r"\s*(MATCH|OPTIONAL\s+MATCH|WITH)\b", query, flags=re.IGNORECASE):
        raise GraphQueryError("query must start with MATCH/OPTIONAL MATCH/WITH", code="INVALID_QUERY")


def _normalize_limit(limit: Optional[int], max_limit: int) -> int:
    if limit is None:
        return min(50, max_limit)
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        raise GraphQueryError("limit must be an integer", code="INVALID_LIMIT")
    if limit_value <= 0:
        raise GraphQueryError("limit must be >= 1", code="INVALID_LIMIT")
    return min(limit_value, max_limit)


def _extract_labels(query: str) -> set[str]:
    labels: set[str] = set()
    for match in _LABEL_RE.finditer(query):
        label = match.group(1) or match.group(2)
        if label:
            labels.add(label)
    return labels


def _validate_labels(scope: str, labels: set[str]) -> None:
    if not labels:
        raise GraphQueryError("query must include explicit labels", code="MISSING_LABELS")

    if scope == "code":
        if not any(label.startswith("CODE_") for label in labels):
            raise GraphQueryError("code scope requires CODE_* labels", code="INVALID_LABELS")
        if any(label.startswith("OM_") or label.startswith("__") for label in labels):
            raise GraphQueryError("code scope cannot query OM_* or __Entity__ labels", code="INVALID_LABELS")

    if scope == "memory":
        if not any(label.startswith("OM_") for label in labels):
            raise GraphQueryError("memory scope requires OM_* labels", code="INVALID_LABELS")
        if any(label.startswith("CODE_") or label.startswith("__") for label in labels):
            raise GraphQueryError("memory scope cannot query CODE_* or __Entity__ labels", code="INVALID_LABELS")

    if scope == "mem0":
        if "__Entity__" not in labels:
            raise GraphQueryError("mem0 scope requires __Entity__ label", code="INVALID_LABELS")
        if any(label.startswith("CODE_") or label.startswith("OM_") for label in labels):
            raise GraphQueryError("mem0 scope cannot query CODE_* or OM_* labels", code="INVALID_LABELS")


def _require_param(params: dict[str, Any], key: str, value: Any) -> None:
    if key in params and params[key] != value:
        raise GraphQueryError(f"parameter {key} does not match enforced value", code="INVALID_PARAMS")
    params[key] = value


def _require_repo_filter(query: str) -> None:
    if not re.search(r"repo_id\s*[:=]\s*\$(repo_id|repoId)\b", query):
        raise GraphQueryError("query must filter by repo_id using $repo_id", code="MISSING_REPO_FILTER")


def _require_access_entity_filter(query: str) -> None:
    if not re.search(r"accessEntity\s*[:=]\s*\$(access_entity|accessEntity)\b", query):
        raise GraphQueryError(
            "query must filter by accessEntity using $access_entity",
            code="MISSING_ACCESS_ENTITY_FILTER",
        )


def _require_user_id_filter(query: str) -> None:
    if not re.search(r"user_id\s*[:=]\s*\$(user_id|userId)\b", query):
        raise GraphQueryError(
            "query must filter by user_id using $user_id",
            code="MISSING_USER_ID_FILTER",
        )


def _ensure_limit(query: str, params: dict[str, Any], limit: int) -> str:
    literal_match = _LIMIT_LITERAL_RE.search(query)
    if literal_match:
        literal_value = int(literal_match.group(1))
        if literal_value > limit:
            raise GraphQueryError("limit exceeds maximum allowed", code="INVALID_LIMIT")
        return query

    param_match = _LIMIT_PARAM_RE.search(query)
    if param_match:
        param_name = param_match.group(1)
        if param_name != "limit":
            raise GraphQueryError("limit parameter must be $limit", code="INVALID_LIMIT")
        params["limit"] = min(int(params.get("limit", limit)), limit)
        return query

    return f"{query.rstrip()} LIMIT {limit}"
