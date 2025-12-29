"""
Structured Memory API Validation (Dev Assistant).

Validates and builds structured memory metadata for the coding-brain system.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# VALIDATION ERROR
# =============================================================================


class StructuredMemoryError(Exception):
    """Raised when structured memory parameters are invalid."""


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_CATEGORIES = [
    "decision",
    "convention",
    "architecture",
    "dependency",
    "workflow",
    "testing",
    "security",
    "performance",
    "runbook",
    "glossary",
]

VALID_SCOPES = [
    "session",
    "user",
    "team",
    "project",
    "org",
    "enterprise",
]

VALID_ARTIFACT_TYPES = [
    "repo",
    "service",
    "module",
    "component",
    "api",
    "db",
    "infra",
    "file",
]

VALID_SOURCES = ["user", "inference"]

LEGACY_KEYS = {
    "re",
    "src",
    "ev",
    "from",
    "was",
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def _normalize_value(value: str) -> str:
    if not isinstance(value, str):
        raise StructuredMemoryError("Expected string value")
    return value.strip()


def validate_text(text: str) -> str:
    """Validate memory text content."""
    text = _normalize_value(text)
    if not text:
        raise StructuredMemoryError("Text cannot be empty")
    return text


def validate_category(category: str) -> str:
    """Validate category name."""
    category = _normalize_value(category).lower()
    if category not in VALID_CATEGORIES:
        raise StructuredMemoryError(
            f"Invalid category '{category}'. Must be one of: {', '.join(VALID_CATEGORIES)}"
        )
    return category


def validate_scope(scope: str) -> str:
    """Validate scope name."""
    scope = _normalize_value(scope).lower()
    if scope not in VALID_SCOPES:
        raise StructuredMemoryError(
            f"Invalid scope '{scope}'. Must be one of: {', '.join(VALID_SCOPES)}"
        )
    return scope


def validate_artifact_type(artifact_type: str) -> str:
    """Validate artifact type."""
    artifact_type = _normalize_value(artifact_type).lower()
    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise StructuredMemoryError(
            f"Invalid artifact_type '{artifact_type}'. Must be one of: "
            f"{', '.join(VALID_ARTIFACT_TYPES)}"
        )
    return artifact_type


def validate_source(source: str) -> str:
    """Validate source type."""
    source = _normalize_value(source).lower()
    if source not in VALID_SOURCES:
        raise StructuredMemoryError(
            f"Invalid source '{source}'. Must be one of: {', '.join(VALID_SOURCES)}"
        )
    return source


def validate_artifact_ref(artifact_ref: str) -> str:
    """Validate artifact ref."""
    artifact_ref = _normalize_value(artifact_ref)
    if not artifact_ref:
        raise StructuredMemoryError("artifact_ref cannot be empty")
    return artifact_ref


def validate_entity(entity: str) -> str:
    """Validate entity string."""
    entity = _normalize_value(entity)
    if not entity:
        raise StructuredMemoryError("entity cannot be empty")
    return entity


def validate_evidence(evidence: List[str]) -> List[str]:
    """Validate evidence list."""
    if not isinstance(evidence, list):
        raise StructuredMemoryError("Evidence must be a list")
    cleaned = []
    for item in evidence:
        if not isinstance(item, str):
            raise StructuredMemoryError(
                f"Evidence item must be string, got {type(item).__name__}"
            )
        cleaned_item = item.strip()
        if cleaned_item:
            cleaned.append(cleaned_item)
    return cleaned


def validate_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    """Validate tags dictionary."""
    if not isinstance(tags, dict):
        raise StructuredMemoryError("Tags must be a dictionary")
    for key in tags.keys():
        if not isinstance(key, str):
            raise StructuredMemoryError(
                f"Tag key must be string, got {type(key).__name__}"
            )
    return tags


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Drop legacy keys and keep unknown extras."""
    if not isinstance(metadata, dict):
        raise StructuredMemoryError("metadata must be a dictionary")
    return {k: v for k, v in metadata.items() if k not in LEGACY_KEYS}


# =============================================================================
# STRUCTURED MEMORY BUILDER
# =============================================================================


@dataclass
class StructuredMemoryInput:
    """Validated structured memory input."""

    text: str
    category: str
    scope: str

    artifact_type: Optional[str] = None
    artifact_ref: Optional[str] = None
    entity: Optional[str] = None
    source: str = "user"
    evidence: Optional[List[str]] = None
    tags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.text = validate_text(self.text)
        self.category = validate_category(self.category)
        self.scope = validate_scope(self.scope)

        if self.artifact_type is not None:
            self.artifact_type = validate_artifact_type(self.artifact_type)

        if self.artifact_ref is not None:
            self.artifact_ref = validate_artifact_ref(self.artifact_ref)

        if self.entity is not None:
            self.entity = validate_entity(self.entity)

        self.source = validate_source(self.source)

        if self.evidence is not None:
            self.evidence = validate_evidence(self.evidence)

        if self.tags is not None:
            self.tags = validate_tags(self.tags)

    def to_metadata_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category,
            "scope": self.scope,
            "source": self.source,
        }

        if self.artifact_type is not None:
            result["artifact_type"] = self.artifact_type
        if self.artifact_ref is not None:
            result["artifact_ref"] = self.artifact_ref
        if self.entity is not None:
            result["entity"] = self.entity
        if self.evidence is not None:
            result["evidence"] = self.evidence
        if self.tags:
            result["tags"] = self.tags

        return result


# =============================================================================
# API HELPERS
# =============================================================================


def build_structured_memory(
    text: str,
    category: str,
    scope: str,
    artifact_type: Optional[str] = None,
    artifact_ref: Optional[str] = None,
    entity: Optional[str] = None,
    source: str = "user",
    evidence: Optional[List[str]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build and validate structured memory input."""
    memory = StructuredMemoryInput(
        text=text,
        category=category,
        scope=scope,
        artifact_type=artifact_type,
        artifact_ref=artifact_ref,
        entity=entity,
        source=source,
        evidence=evidence,
        tags=tags,
    )

    return memory.text, memory.to_metadata_dict()


def normalize_metadata_for_create(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate metadata dict for create requests."""
    metadata = sanitize_metadata(metadata)
    category = metadata.get("category")
    scope = metadata.get("scope")
    if category is None or scope is None:
        raise StructuredMemoryError("metadata must include 'category' and 'scope'")

    validated = {
        "category": validate_category(category),
        "scope": validate_scope(scope),
        "source": validate_source(metadata.get("source", "user")),
    }

    if metadata.get("artifact_type") is not None:
        validated["artifact_type"] = validate_artifact_type(metadata["artifact_type"])

    if metadata.get("artifact_ref") is not None:
        validated["artifact_ref"] = validate_artifact_ref(metadata["artifact_ref"])

    if metadata.get("entity") is not None:
        validated["entity"] = validate_entity(metadata["entity"])

    if metadata.get("evidence") is not None:
        validated["evidence"] = validate_evidence(metadata["evidence"])

    if metadata.get("tags") is not None:
        validated["tags"] = validate_tags(metadata["tags"])

    # Preserve extra metadata keys (minus legacy) to avoid data loss
    extras = {
        key: value
        for key, value in metadata.items()
        if key not in validated and key not in {"category", "scope", "source"}
    }

    return {**validated, **extras}


def validate_update_fields(
    category: Optional[str] = None,
    scope: Optional[str] = None,
    artifact_type: Optional[str] = None,
    artifact_ref: Optional[str] = None,
    entity: Optional[str] = None,
    source: Optional[str] = None,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate fields for memory update."""
    validated: Dict[str, Any] = {}

    if category is not None:
        validated["category"] = validate_category(category)

    if scope is not None:
        validated["scope"] = validate_scope(scope)

    if artifact_type is not None:
        validated["artifact_type"] = validate_artifact_type(artifact_type)

    if artifact_ref is not None:
        validated["artifact_ref"] = validate_artifact_ref(artifact_ref)

    if entity is not None:
        validated["entity"] = validate_entity(entity)

    if source is not None:
        validated["source"] = validate_source(source)

    if evidence is not None:
        validated["evidence"] = validate_evidence(evidence)

    return validated


def apply_metadata_updates(
    current_metadata: Dict[str, Any],
    validated_fields: Dict[str, Any],
    add_tags: Optional[Dict[str, Any]] = None,
    remove_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Apply validated updates to existing metadata."""
    metadata = sanitize_metadata(current_metadata)

    for key, value in validated_fields.items():
        metadata[key] = value

    current_tags = metadata.get("tags", {})
    if not isinstance(current_tags, dict):
        current_tags = {}

    if remove_tags:
        for tag in remove_tags:
            current_tags.pop(tag, None)

    if add_tags:
        current_tags.update(add_tags)

    if current_tags:
        metadata["tags"] = current_tags
    elif "tags" in metadata and not current_tags:
        del metadata["tags"]

    return metadata
