"""OpenAPI schema extraction for contract nodes and edges."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from openmemory.api.indexing.deterministic_edges import RepoSymbolIndex, SchemaFieldDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAPIDefinition:
    """OpenAPI schema definition node."""

    openapi_id: str
    name: str
    file_path: Path
    title: Optional[str] = None


@dataclass
class OpenAPIExtraction:
    """Extracted OpenAPI schema data."""

    definitions: list[OpenAPIDefinition] = field(default_factory=list)
    schema_fields: list[SchemaFieldDefinition] = field(default_factory=list)
    schema_expose_edges: list[tuple[str, str]] = field(default_factory=list)


class OpenAPISpecExtractor:
    """Extract schema field nodes and edges from OpenAPI specs."""

    def __init__(self, root_path: Path, symbol_index: RepoSymbolIndex):
        self.root_path = Path(root_path)
        self.symbol_index = symbol_index

    def extract_from_file(self, file_path: Path) -> OpenAPIExtraction:
        doc = self._load_document(file_path)
        if not isinstance(doc, dict):
            return OpenAPIExtraction()
        if "openapi" not in doc and "swagger" not in doc:
            return OpenAPIExtraction()

        title = self._spec_title(doc)
        schemas = self._component_schemas(doc)
        extraction = OpenAPIExtraction()

        for schema_name, schema_body in schemas.items():
            if not isinstance(schema_body, dict):
                continue
            openapi_id = self._openapi_id(file_path, schema_name)
            extraction.definitions.append(
                OpenAPIDefinition(
                    openapi_id=openapi_id,
                    name=schema_name,
                    file_path=file_path,
                    title=title,
                )
            )

            properties = self._schema_properties(schema_body)
            required = set(schema_body.get("required") or [])
            for field_name, field_schema in properties.items():
                field_type, nullable = self._field_metadata(field_schema, required, field_name)
                schema_id = self._schema_field_id(
                    file_path=file_path,
                    schema_name=schema_name,
                    field_name=field_name,
                )
                extraction.schema_fields.append(
                    SchemaFieldDefinition(
                        schema_id=schema_id,
                        name=field_name,
                        schema_type="openapi",
                        schema_name=schema_name,
                        file_path=file_path,
                        line_start=None,
                        line_end=None,
                        nullable=nullable,
                        field_type=field_type,
                    )
                )

                field_symbol_id = self._resolve_field_symbol(schema_name, field_name)
                if field_symbol_id:
                    extraction.schema_expose_edges.append((field_symbol_id, schema_id))

        return extraction

    def _load_document(self, file_path: Path) -> Optional[dict[str, Any]]:
        try:
            content = file_path.read_text(errors="ignore")
        except Exception as exc:
            logger.debug(f"Failed to read OpenAPI spec {file_path}: {exc}")
            return None

        if file_path.suffix.lower() == ".json":
            try:
                return json.loads(content)
            except Exception as exc:
                logger.debug(f"Failed to parse OpenAPI JSON {file_path}: {exc}")
                return None

        if yaml is None:
            logger.debug("PyYAML not available; skipping OpenAPI YAML parsing")
            return None

        try:
            return yaml.safe_load(content)
        except Exception as exc:
            logger.debug(f"Failed to parse OpenAPI YAML {file_path}: {exc}")
            return None

    def _spec_title(self, doc: dict[str, Any]) -> Optional[str]:
        info = doc.get("info")
        if isinstance(info, dict):
            title = info.get("title")
            if isinstance(title, str) and title:
                return title
        return None

    def _component_schemas(self, doc: dict[str, Any]) -> dict[str, Any]:
        components = doc.get("components")
        if not isinstance(components, dict):
            return {}
        schemas = components.get("schemas")
        if not isinstance(schemas, dict):
            return {}
        return schemas

    def _schema_properties(self, schema_body: dict[str, Any]) -> dict[str, Any]:
        props = schema_body.get("properties")
        if isinstance(props, dict):
            return props
        return {}

    def _field_metadata(
        self,
        field_schema: Any,
        required_fields: set[str],
        field_name: str,
    ) -> tuple[Optional[str], Optional[bool]]:
        if not isinstance(field_schema, dict):
            return None, None

        nullable = self._nullable_flag(field_schema, required_fields, field_name)
        field_type = self._schema_type(field_schema)
        return field_type, nullable

    def _nullable_flag(
        self,
        field_schema: dict[str, Any],
        required_fields: set[str],
        field_name: str,
    ) -> Optional[bool]:
        if "nullable" in field_schema:
            return bool(field_schema.get("nullable"))
        field_type = field_schema.get("type")
        if isinstance(field_type, list) and "null" in field_type:
            return True
        return field_name not in required_fields

    def _schema_type(self, field_schema: dict[str, Any]) -> Optional[str]:
        ref = field_schema.get("$ref")
        if isinstance(ref, str):
            return self._ref_name(ref)

        if "enum" in field_schema:
            return "enum"

        field_type = field_schema.get("type")
        if isinstance(field_type, list):
            field_type = next((t for t in field_type if t != "null"), None)

        if field_type == "array":
            item_type = self._array_item_type(field_schema.get("items"))
            if item_type:
                return f"{item_type}[]"
            return "array"

        if isinstance(field_type, str):
            return field_type
        return None

    def _array_item_type(self, items: Any) -> Optional[str]:
        if not isinstance(items, dict):
            return None
        ref = items.get("$ref")
        if isinstance(ref, str):
            return self._ref_name(ref)
        item_type = items.get("type")
        if isinstance(item_type, str):
            return item_type
        return None

    def _ref_name(self, ref: str) -> str:
        return ref.split("/")[-1]

    def _resolve_field_symbol(self, schema_name: str, field_name: str) -> Optional[str]:
        candidates = self._schema_name_candidates(schema_name)
        for candidate in candidates:
            field_id = self.symbol_index.resolve_field_global(candidate, field_name)
            if field_id:
                return field_id
        return None

    def _schema_name_candidates(self, schema_name: str) -> list[str]:
        candidates = [schema_name]
        if schema_name.endswith("Dto"):
            candidates.append(schema_name[: -len("Dto")])
        if schema_name.endswith("DTO"):
            candidates.append(schema_name[: -len("DTO")])
        if not schema_name.endswith(("Dto", "DTO")):
            candidates.append(f"{schema_name}Dto")
            candidates.append(f"{schema_name}DTO")
        return candidates

    def _schema_field_id(
        self,
        file_path: Path,
        schema_name: str,
        field_name: str,
    ) -> str:
        return f"schema::openapi:{file_path}:{schema_name}:{field_name}"

    def _openapi_id(self, file_path: Path, schema_name: str) -> str:
        return f"openapi::{file_path}:{schema_name}"
