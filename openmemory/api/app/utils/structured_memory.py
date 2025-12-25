"""
Structured Memory API Validation

Provides validation and building of structured memory parameters
for the AXIS 3.4 protocol without embedded string parsing.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from app.utils.axis_tags import (
    VAULT_CODES,
    VAULT_TO_CATEGORY,
    VALID_LAYERS,
    VALID_VECTORS,
    VALID_CIRCUITS,
)


# =============================================================================
# VALIDATION ERROR
# =============================================================================

class StructuredMemoryError(Exception):
    """Raised when structured memory parameters are invalid."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_VAULTS = list(VAULT_CODES.keys())
VALID_SOURCES = ["user", "inference"]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_vault(vault: str) -> str:
    """
    Validate vault code.

    Args:
        vault: Vault short code (SOV, WLT, SIG, FRC, DIR, FGP, Q)

    Returns:
        Validated vault code (uppercase)

    Raises:
        StructuredMemoryError: If vault is invalid
    """
    vault = vault.upper().strip()
    if vault not in VALID_VAULTS:
        raise StructuredMemoryError(
            f"Invalid vault '{vault}'. Must be one of: {', '.join(VALID_VAULTS)}"
        )
    return vault


def validate_layer(layer: str) -> str:
    """
    Validate layer name.

    Args:
        layer: Layer name (somatic, emotional, narrative, etc.)

    Returns:
        Validated layer name (lowercase)

    Raises:
        StructuredMemoryError: If layer is invalid
    """
    layer = layer.lower().strip()
    if layer not in VALID_LAYERS:
        raise StructuredMemoryError(
            f"Invalid layer '{layer}'. Must be one of: {', '.join(VALID_LAYERS)}"
        )
    return layer


def validate_circuit(circuit: int) -> int:
    """
    Validate circuit number.

    Args:
        circuit: Circuit number (1-8)

    Returns:
        Validated circuit number

    Raises:
        StructuredMemoryError: If circuit is invalid
    """
    if not isinstance(circuit, int) or circuit not in VALID_CIRCUITS:
        raise StructuredMemoryError(
            f"Invalid circuit {circuit}. Must be integer 1-8"
        )
    return circuit


def validate_vector(vector: str) -> str:
    """
    Validate vector type.

    Args:
        vector: Vector type (say, want, do)

    Returns:
        Validated vector (lowercase)

    Raises:
        StructuredMemoryError: If vector is invalid
    """
    vector = vector.lower().strip()
    if vector not in VALID_VECTORS:
        raise StructuredMemoryError(
            f"Invalid vector '{vector}'. Must be one of: {', '.join(VALID_VECTORS)}"
        )
    return vector


def validate_source(source: str) -> str:
    """
    Validate source type.

    Args:
        source: Source type (user, inference)

    Returns:
        Validated source (lowercase)

    Raises:
        StructuredMemoryError: If source is invalid
    """
    source = source.lower().strip()
    if source not in VALID_SOURCES:
        raise StructuredMemoryError(
            f"Invalid source '{source}'. Must be one of: {', '.join(VALID_SOURCES)}"
        )
    return source


def validate_text(text: str) -> str:
    """
    Validate memory text content.

    Args:
        text: Memory content

    Returns:
        Validated text (stripped)

    Raises:
        StructuredMemoryError: If text is empty
    """
    text = text.strip()
    if not text:
        raise StructuredMemoryError("Text cannot be empty")
    return text


def validate_tags(tags: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate tags dictionary.

    Args:
        tags: Dictionary of tags

    Returns:
        Validated tags dictionary

    Raises:
        StructuredMemoryError: If tags format is invalid
    """
    if not isinstance(tags, dict):
        raise StructuredMemoryError("Tags must be a dictionary")

    # Validate all keys are strings
    for key in tags.keys():
        if not isinstance(key, str):
            raise StructuredMemoryError(f"Tag key must be string, got {type(key).__name__}")

    return tags


def validate_evidence(evidence: List[str]) -> List[str]:
    """
    Validate evidence list.

    Args:
        evidence: List of evidence strings

    Returns:
        Validated evidence list

    Raises:
        StructuredMemoryError: If evidence format is invalid
    """
    if not isinstance(evidence, list):
        raise StructuredMemoryError("Evidence must be a list")

    for item in evidence:
        if not isinstance(item, str):
            raise StructuredMemoryError(f"Evidence item must be string, got {type(item).__name__}")

    return evidence


# =============================================================================
# STRUCTURED MEMORY BUILDER
# =============================================================================

@dataclass
class StructuredMemoryInput:
    """
    Validated structured memory input.

    All fields are validated upon creation.
    """
    text: str
    vault: str
    layer: str
    vault_full: str = field(init=False)
    axis_category: str = field(init=False)

    # Optional structure
    circuit: Optional[int] = None
    vector: Optional[str] = None

    # Optional metadata
    entity: Optional[str] = None
    source: str = "user"
    was: Optional[str] = None
    origin: Optional[str] = None
    evidence: Optional[List[str]] = None

    # Optional tags
    tags: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate all fields after initialization."""
        # Required fields
        self.text = validate_text(self.text)
        self.vault = validate_vault(self.vault)
        self.layer = validate_layer(self.layer)

        # Computed fields
        self.vault_full = VAULT_CODES[self.vault]
        self.axis_category = VAULT_TO_CATEGORY.get(self.vault_full, "personal")

        # Optional fields
        if self.circuit is not None:
            self.circuit = validate_circuit(self.circuit)

        if self.vector is not None:
            self.vector = validate_vector(self.vector)

        self.source = validate_source(self.source)

        if self.evidence is not None:
            self.evidence = validate_evidence(self.evidence)

        if self.tags is not None:
            self.tags = validate_tags(self.tags)

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Convert to flat metadata dictionary for storage.

        Returns:
            Dictionary suitable for database/vector store metadata
        """
        result = {
            "source": "axis_protocol",
            "vault": self.vault_full,
            "layer": self.layer,
            "axis_category": self.axis_category,
            "src": self.source,
        }

        if self.circuit is not None:
            result["circuit"] = self.circuit

        if self.vector is not None:
            result["vector"] = self.vector

        if self.entity is not None:
            result["re"] = self.entity

        if self.was is not None:
            result["was"] = self.was

        if self.origin is not None:
            result["from"] = self.origin

        if self.evidence is not None:
            result["ev"] = self.evidence

        if self.tags:
            result["tags"] = self.tags

        return result


def build_structured_memory(
    text: str,
    vault: str,
    layer: str,
    circuit: Optional[int] = None,
    vector: Optional[str] = None,
    entity: Optional[str] = None,
    source: str = "user",
    was: Optional[str] = None,
    origin: Optional[str] = None,
    evidence: Optional[List[str]] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build and validate structured memory input.

    This is the main entry point for the structured API.

    Args:
        text: Pure content, no markers
        vault: SOV|WLT|SIG|FRC|DIR|FGP|Q
        layer: somatic|emotional|narrative|cognitive|values|identity|relational|goals|resources|context|temporal|meta
        circuit: 1-8 (optional)
        vector: say|want|do (optional)
        entity: Reference entity (optional)
        source: user|inference (default: user)
        was: Previous state (optional)
        origin: Origin reference (optional)
        evidence: Evidence list (optional)
        tags: Dict with string keys (optional)

    Returns:
        Tuple of (clean_text, metadata_dict)

    Raises:
        StructuredMemoryError: If any parameter is invalid
    """
    memory = StructuredMemoryInput(
        text=text,
        vault=vault,
        layer=layer,
        circuit=circuit,
        vector=vector,
        entity=entity,
        source=source,
        was=was,
        origin=origin,
        evidence=evidence,
        tags=tags,
    )

    return memory.text, memory.to_metadata_dict()


# =============================================================================
# UPDATE HELPERS
# =============================================================================

def validate_update_fields(
    text: Optional[str] = None,
    vault: Optional[str] = None,
    layer: Optional[str] = None,
    circuit: Optional[int] = None,
    vector: Optional[str] = None,
    entity: Optional[str] = None,
    source: Optional[str] = None,
    was: Optional[str] = None,
    origin: Optional[str] = None,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate fields for memory update.

    Only validates fields that are provided (not None).

    Returns:
        Dictionary of validated fields

    Raises:
        StructuredMemoryError: If any provided field is invalid
    """
    validated = {}

    if text is not None:
        validated["text"] = validate_text(text)

    if vault is not None:
        vault = validate_vault(vault)
        validated["vault"] = vault
        validated["vault_full"] = VAULT_CODES[vault]
        validated["axis_category"] = VAULT_TO_CATEGORY.get(
            validated["vault_full"], "personal"
        )

    if layer is not None:
        validated["layer"] = validate_layer(layer)

    if circuit is not None:
        validated["circuit"] = validate_circuit(circuit)

    if vector is not None:
        validated["vector"] = validate_vector(vector)

    if entity is not None:
        validated["entity"] = entity.strip()

    if source is not None:
        validated["source"] = validate_source(source)

    if was is not None:
        validated["was"] = was.strip()

    if origin is not None:
        validated["origin"] = origin.strip()

    if evidence is not None:
        validated["evidence"] = validate_evidence(evidence)

    return validated


def apply_metadata_updates(
    current_metadata: Dict[str, Any],
    validated_fields: Dict[str, Any],
    add_tags: Optional[Dict[str, Any]] = None,
    remove_tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Apply validated updates to existing metadata.

    Args:
        current_metadata: Existing metadata dict
        validated_fields: Validated fields to update
        add_tags: Tags to add/update
        remove_tags: Tag keys to remove

    Returns:
        Updated metadata dictionary
    """
    metadata = dict(current_metadata)

    # Apply field updates
    field_mapping = {
        "vault_full": "vault",
        "layer": "layer",
        "circuit": "circuit",
        "vector": "vector",
        "entity": "re",
        "source": "src",
        "was": "was",
        "origin": "from",
        "evidence": "ev",
        "axis_category": "axis_category",
    }

    for field_key, meta_key in field_mapping.items():
        if field_key in validated_fields:
            metadata[meta_key] = validated_fields[field_key]

    # Handle tags
    current_tags = metadata.get("tags", {})
    if not isinstance(current_tags, dict):
        current_tags = {}

    # Remove tags
    if remove_tags:
        for tag in remove_tags:
            current_tags.pop(tag, None)

    # Add/update tags
    if add_tags:
        current_tags.update(add_tags)

    if current_tags:
        metadata["tags"] = current_tags
    elif "tags" in metadata and not current_tags:
        del metadata["tags"]

    return metadata
