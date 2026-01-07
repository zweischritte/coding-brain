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

# Valid prefixes for access_entity field (access control)
VALID_ACCESS_ENTITY_PREFIXES = [
    "user",
    "team",
    "project",
    "org",
]

# Scopes that require access_entity (shared scopes)
SHARED_SCOPES = ["team", "project", "org", "enterprise"]

# Mapping from scope to expected access_entity prefix
SCOPE_TO_ACCESS_ENTITY_PREFIX = {
    "user": "user",
    "session": "user",  # Session is personal like user
    "team": "team",
    "project": "project",
    "org": "org",
    "enterprise": "org",  # Enterprise uses org prefix
}

# Reverse mapping: access_entity prefix to scope (for derivation)
# Note: session and enterprise are special cases handled separately
ACCESS_ENTITY_PREFIX_TO_SCOPE = {
    "user": "user",
    "team": "team",
    "project": "project",
    "org": "org",
}

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


def _split_comma_separated(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_evidence_input(evidence: Any) -> List[str]:
    """Normalize evidence input to a list of strings."""
    if isinstance(evidence, str):
        evidence = _split_comma_separated(evidence)
    elif isinstance(evidence, tuple):
        evidence = list(evidence)

    if not isinstance(evidence, list):
        raise StructuredMemoryError("Evidence must be a list or comma-separated string")

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


def normalize_tags_input(tags: Any) -> Dict[str, Any]:
    """Normalize tags input to a dictionary."""
    if isinstance(tags, dict):
        normalized = tags
    elif isinstance(tags, (list, tuple)):
        normalized = {}
        for item in tags:
            if not isinstance(item, str):
                raise StructuredMemoryError(
                    f"Tag item must be string, got {type(item).__name__}"
                )
            key = item.strip()
            if key:
                normalized[key] = True
    else:
        raise StructuredMemoryError("Tags must be a dictionary or list of strings")

    for key in normalized.keys():
        if not isinstance(key, str):
            raise StructuredMemoryError(
                f"Tag key must be string, got {type(key).__name__}"
            )
    return normalized


def normalize_tag_list_input(tags: Any) -> List[str]:
    """Normalize tag list input to a list of strings."""
    if isinstance(tags, str):
        tags = _split_comma_separated(tags)
    elif isinstance(tags, tuple):
        tags = list(tags)

    if not isinstance(tags, list):
        raise StructuredMemoryError("remove_tags must be a list or comma-separated string")

    cleaned = []
    for item in tags:
        if not isinstance(item, str):
            raise StructuredMemoryError(
                f"remove_tags item must be string, got {type(item).__name__}"
            )
        cleaned_item = item.strip()
        if cleaned_item:
            cleaned.append(cleaned_item)
    return cleaned


def validate_evidence(evidence: Any) -> List[str]:
    """Validate evidence list."""
    return normalize_evidence_input(evidence)


def validate_tags(tags: Any) -> Dict[str, Any]:
    """Validate tags dictionary."""
    return normalize_tags_input(tags)


def validate_access_entity(access_entity: str) -> str:
    """Validate access_entity format.

    access_entity must have format: <prefix>:<value>
    where prefix is one of: user, team, project, org
    """
    access_entity = _normalize_value(access_entity)
    if not access_entity:
        raise StructuredMemoryError("access_entity cannot be empty")

    if ":" not in access_entity:
        raise StructuredMemoryError(
            f"Invalid access_entity format '{access_entity}'. "
            f"Must be <prefix>:<value> where prefix is one of: "
            f"{', '.join(VALID_ACCESS_ENTITY_PREFIXES)}"
        )

    prefix, value = access_entity.split(":", 1)
    prefix = prefix.lower()

    if prefix not in VALID_ACCESS_ENTITY_PREFIXES:
        raise StructuredMemoryError(
            f"Invalid access_entity prefix '{prefix}'. "
            f"Must be one of: {', '.join(VALID_ACCESS_ENTITY_PREFIXES)}"
        )

    if not value.strip():
        raise StructuredMemoryError(
            f"access_entity value cannot be empty after prefix '{prefix}:'"
        )

    return access_entity


def validate_scope_access_entity_consistency(scope: str, access_entity: str) -> None:
    """Validate that scope and access_entity prefix are consistent.

    Rules:
    - scope=user/session -> access_entity must be user:*
    - scope=team -> access_entity must be team:*
    - scope=project -> access_entity must be project:*
    - scope=org/enterprise -> access_entity must be org:*
    """
    if ":" not in access_entity:
        return  # Will be caught by validate_access_entity

    prefix = access_entity.split(":", 1)[0].lower()
    expected_prefix = SCOPE_TO_ACCESS_ENTITY_PREFIX.get(scope)

    if expected_prefix and prefix != expected_prefix:
        raise StructuredMemoryError(
            f"Scope/access_entity mismatch: scope='{scope}' requires "
            f"access_entity with prefix '{expected_prefix}:', but got '{prefix}:'"
        )


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Drop legacy keys and keep unknown extras."""
    if not isinstance(metadata, dict):
        raise StructuredMemoryError("metadata must be a dictionary")
    return {k: v for k, v in metadata.items() if k not in LEGACY_KEYS}


def derive_scope(
    access_entity: Optional[str],
    explicit_scope: Optional[str],
) -> str:
    """
    Derive scope from access_entity, unless explicitly provided.

    PRD-13: access_entity is the Single Source of Truth for Access Control.
    scope is derived from access_entity prefix (unless explicitly provided).

    Rules:
    - If explicit_scope is 'session' or 'enterprise', always use it (special cases)
    - If explicit_scope is provided and not empty, use it
    - If access_entity is None or invalid format, default to 'user'
    - Otherwise, derive from access_entity prefix

    Args:
        access_entity: The access control entity (e.g., 'user:grischa', 'team:org/path')
        explicit_scope: Explicitly provided scope (may be None)

    Returns:
        The derived or explicit scope string

    Raises:
        ValueError: If access_entity has an unknown prefix
    """
    # Handle explicit scope for special cases (session, enterprise)
    if explicit_scope in ("session", "enterprise"):
        return explicit_scope

    # If explicit scope is provided (and not empty string), use it
    if explicit_scope is not None and explicit_scope.strip():
        return explicit_scope

    # If no access_entity or empty/whitespace, default to user
    if not access_entity or not access_entity.strip():
        return "user"

    access_entity = access_entity.strip()

    # Check for valid format (must contain colon)
    if ":" not in access_entity:
        return "user"

    # Extract prefix (case-insensitive)
    prefix = access_entity.split(":", 1)[0].lower()

    # Look up scope from prefix
    scope = ACCESS_ENTITY_PREFIX_TO_SCOPE.get(prefix)
    if scope is None:
        raise ValueError(f"Unknown access_entity prefix: {prefix}")

    return scope


# =============================================================================
# STRUCTURED MEMORY BUILDER
# =============================================================================


@dataclass
class StructuredMemoryInput:
    """Validated structured memory input."""

    text: str
    category: str
    scope: Optional[str] = None  # Now optional - derived from access_entity if not provided

    artifact_type: Optional[str] = None
    artifact_ref: Optional[str] = None
    entity: Optional[str] = None
    access_entity: Optional[str] = None  # Access control field
    source: str = "user"
    evidence: Optional[List[str]] = None
    tags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.text = validate_text(self.text)
        self.category = validate_category(self.category)

        # Normalize empty access_entity to None
        if self.access_entity is not None and not self.access_entity.strip():
            self.access_entity = None

        # PRD-13: Derive scope from access_entity if not explicitly provided
        # This makes scope optional while keeping it as a persisted field
        if self.scope is None:
            # Derive scope from access_entity
            try:
                self.scope = derive_scope(self.access_entity, explicit_scope=None)
            except ValueError:
                # Unknown prefix - will be caught by validate_access_entity later
                self.scope = "user"
        else:
            # Explicit scope provided - validate it
            self.scope = validate_scope(self.scope)

        if self.artifact_type is not None:
            self.artifact_type = validate_artifact_type(self.artifact_type)

        if self.artifact_ref is not None:
            self.artifact_ref = validate_artifact_ref(self.artifact_ref)

        if self.entity is not None:
            self.entity = validate_entity(self.entity)

        # Validate access_entity requirement based on scope
        # Only require access_entity for shared scopes when scope is EXPLICITLY provided
        # If scope was derived, we already have a valid access_entity
        if self.scope in SHARED_SCOPES:
            if self.access_entity is None:
                raise StructuredMemoryError(
                    f"access_entity is required for scope='{self.scope}'. "
                    f"Shared scopes ({', '.join(SHARED_SCOPES)}) require access_entity."
                )

        if self.access_entity is not None:
            self.access_entity = validate_access_entity(self.access_entity)
            # Validate scope/access_entity consistency
            validate_scope_access_entity_consistency(self.scope, self.access_entity)

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
        if self.access_entity is not None:
            result["access_entity"] = self.access_entity
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
    scope: Optional[str] = None,  # Now optional - derived from access_entity if not provided
    artifact_type: Optional[str] = None,
    artifact_ref: Optional[str] = None,
    entity: Optional[str] = None,
    access_entity: Optional[str] = None,
    source: str = "user",
    evidence: Optional[List[str]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build and validate structured memory input.

    PRD-13: scope is now optional. If not provided, it will be derived from
    access_entity prefix. If neither is provided, defaults to 'user' scope.
    """
    memory = StructuredMemoryInput(
        text=text,
        category=category,
        scope=scope,
        artifact_type=artifact_type,
        artifact_ref=artifact_ref,
        entity=entity,
        access_entity=access_entity,
        source=source,
        evidence=evidence,
        tags=tags,
    )

    return memory.text, memory.to_metadata_dict()


def normalize_metadata_for_create(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate metadata dict for create requests.

    PRD-13: scope is now optional. If not provided, it will be derived from
    access_entity prefix. If neither is provided, defaults to 'user' scope.
    """
    metadata = sanitize_metadata(metadata)
    category = metadata.get("category")
    scope = metadata.get("scope")
    access_entity = metadata.get("access_entity")

    # Category is always required
    if category is None:
        raise StructuredMemoryError("metadata must include 'category'")

    # PRD-13: scope is now optional - derive from access_entity if not provided
    if scope is None:
        # Derive scope from access_entity
        try:
            validated_scope = derive_scope(access_entity, explicit_scope=None)
        except ValueError:
            # Unknown prefix - will be caught by validate_access_entity later
            validated_scope = "user"
    else:
        validated_scope = validate_scope(scope)

    validated = {
        "category": validate_category(category),
        "scope": validated_scope,
        "source": validate_source(metadata.get("source", "user")),
    }

    if metadata.get("artifact_type") is not None:
        validated["artifact_type"] = validate_artifact_type(metadata["artifact_type"])

    if metadata.get("artifact_ref") is not None:
        validated["artifact_ref"] = validate_artifact_ref(metadata["artifact_ref"])

    if metadata.get("entity") is not None:
        validated["entity"] = validate_entity(metadata["entity"])

    # Validate access_entity requirement based on scope
    # Only require access_entity for shared scopes when scope is EXPLICITLY provided
    if scope is not None and validated_scope in SHARED_SCOPES:
        if access_entity is None:
            raise StructuredMemoryError(
                f"access_entity is required for scope='{validated_scope}'. "
                f"Shared scopes ({', '.join(SHARED_SCOPES)}) require access_entity."
            )

    if access_entity is not None:
        validated["access_entity"] = validate_access_entity(access_entity)
        validate_scope_access_entity_consistency(validated_scope, access_entity)

    if metadata.get("evidence") is not None:
        validated["evidence"] = validate_evidence(metadata["evidence"])

    if metadata.get("tags") is not None:
        validated["tags"] = validate_tags(metadata["tags"])

    # Preserve extra metadata keys (minus legacy) to avoid data loss
    extras = {
        key: value
        for key, value in metadata.items()
        if key not in validated and key not in {"category", "scope", "source", "access_entity"}
    }

    return {**validated, **extras}


def validate_update_fields(
    category: Optional[str] = None,
    scope: Optional[str] = None,
    artifact_type: Optional[str] = None,
    artifact_ref: Optional[str] = None,
    entity: Optional[str] = None,
    access_entity: Optional[str] = None,
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

    if access_entity is not None:
        validated["access_entity"] = validate_access_entity(access_entity)

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
