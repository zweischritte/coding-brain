"""
AXIS 3.4 Protocol Memory Parser

Parses memory entries in AXIS 3.4 format:
[V:vault] [L:layer] [C:circuit] [vec:vector] Content {tags} [metadata]

Supports backward compatibility with legacy format:
[VAULT: VAULT_NAME] [LAYER: layer_name] Content
"""

import re
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# CONSTANTS & MAPPINGS
# =============================================================================

# Vault short codes to full names
VAULT_CODES: Dict[str, str] = {
    "SOV": "SOVEREIGNTY_CORE",
    "WLT": "WEALTH_MATRIX",
    "SIG": "SIGNAL_LIBRARY",
    "FRC": "FRACTURE_LOG",
    "DIR": "SOURCE_DIRECTIVES",
    "FGP": "FINGERPRINT",
    "Q": "QUESTIONS_QUEUE",
}

# Full vault names to category (for filtering/grouping)
VAULT_TO_CATEGORY: Dict[str, str] = {
    "SOVEREIGNTY_CORE": "identity",
    "WEALTH_MATRIX": "business",
    "SIGNAL_LIBRARY": "pattern",
    "FRACTURE_LOG": "health",
    "SOURCE_DIRECTIVES": "system",
    "FINGERPRINT": "evolution",
    "QUESTIONS_QUEUE": "meta",
}

# Valid layer names (12 layers)
VALID_LAYERS: list = [
    "somatic",      # Body knowing, energy, intuition, dreams
    "emotional",    # Feelings, triggers, unprocessed material
    "narrative",    # How SOURCE speaks about self
    "cognitive",    # Thinking patterns, biases, decision modes
    "values",       # Stated AND lived ethics
    "identity",     # Roles, masks, authentic vs performed
    "relational",   # People + their energy cost/gift
    "goals",        # Active AND abandoned
    "resources",    # Time, money, energy, network, limits
    "context",      # Environment, life phase, external modulators
    "temporal",     # Biography, turning points, future anticipation
    "meta",         # Self-knowledge quality, blind spots
]

# Valid vector types (Say-Want-Do triangulation)
VALID_VECTORS: list = ["say", "want", "do"]

# Valid circuit numbers (1-8)
VALID_CIRCUITS: list = list(range(1, 9))

# Tags that expect typed values
VALUE_TAGS: Dict[str, type] = {
    "intensity": int,       # {intensity:7} - emotional intensity 1-10
    "energy": int,          # {energy:+5} or {energy:-3} - relational energy
    "conf": float,          # {conf:0.8} - AI observation confidence 0-1
    "prio": str,            # {prio:high} - priority: high/med/low
    "trigger": str,         # {trigger:projekt_stockt} - deployment trigger
    "gap": str,             # {gap:say_do} - gap type
    "symbols": list,        # {symbols:Wasser|Feuer} - dream symbols
    "reason": str,          # {reason:completion_pattern} - question reason
    "tension": str,         # {tension:security_freedom} - productive tension
}

# Boolean flag tags (presence = True)
FLAG_TAGS: list = [
    "abandoned",    # Given-up goal
    "phrase",       # Recurring phrase
    "silence",      # Notable absence
    "dream",        # Dream content
    "person",       # About a person
    "project",      # About a project
    "ai",           # AI observation
    "silent",       # Don't surface yet
    "confirmed",    # User confirmed observation
    "rejected",     # User rejected observation
    "state",        # State that can change
    "health",       # Health-related
    "meaning",      # Existential/meaning
    "bypass",       # Spiritual bypass
    "loop",         # Thought loop
    "shadow",       # Shadow material
    "dilemma",      # Ethical dilemma
    "batch",        # Batch processing
]

# Valid inline metadata keys
VALID_METADATA_KEYS: list = ["src", "re", "was", "from", "ev"]

# Default values
DEFAULT_VAULT = "SOV"
DEFAULT_VAULT_FULL = "SOVEREIGNTY_CORE"
DEFAULT_LAYER = "cognitive"
DEFAULT_CATEGORY = "identity"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ParsedMemory:
    """Structured representation of a parsed AXIS 3.4 memory."""
    vault: str = DEFAULT_VAULT
    vault_full: str = DEFAULT_VAULT_FULL
    layer: str = DEFAULT_LAYER
    circuit: Optional[int] = None
    vector: Optional[str] = None
    content: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    axis_category: str = DEFAULT_CATEGORY

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Convert to flat metadata dictionary for storage.
        This is what gets saved to the database and vector store.
        """
        result = {
            "source": "axis_protocol",
            "vault": self.vault_full,
            "layer": self.layer,
            "axis_category": self.axis_category,
        }

        if self.circuit is not None:
            result["circuit"] = self.circuit

        if self.vector is not None:
            result["vector"] = self.vector

        if self.tags:
            result["tags"] = self.tags

        # Inline metadata (re, src, was, from, ev)
        if self.metadata:
            result.update(self.metadata)

        return result


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_tags(tag_string: str) -> Dict[str, Any]:
    """
    Parse {tag1,tag2:value,tag3:a|b|c} format.

    Examples:
        "{trigger,intensity:7}" -> {"trigger": True, "intensity": 7}
        "{ai,conf:0.8,silent}" -> {"ai": True, "conf": 0.8, "silent": True}
        "{symbols:Wasser|Feuer}" -> {"symbols": ["Wasser", "Feuer"]}
        "{gap:say_do}" -> {"gap": "say_do"}

    Returns:
        Dict with tag names as keys, values (typed) or True for boolean flags.
    """
    if not tag_string:
        return {}

    tags = {}

    # Remove braces
    inner = tag_string.strip("{}")
    if not inner:
        return {}

    # Split by comma - pipes are inside values (like symbols:A|B), not spanning commas
    parts = inner.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            # Tag with value
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Type conversion based on expected type
            if key in VALUE_TAGS:
                expected_type = VALUE_TAGS[key]
                try:
                    if expected_type == int:
                        # Handle +/- prefix for energy
                        value = int(value.replace("+", ""))
                    elif expected_type == float:
                        value = float(value)
                    elif expected_type == list:
                        value = value.split("|")
                    # str stays as-is
                except (ValueError, TypeError):
                    pass  # Keep as string if conversion fails

            tags[key] = value
        else:
            # Boolean flag tag
            tags[part] = True

    return tags


def parse_inline_metadata(content: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract inline metadata like [re:BMG] [src:user] from content.
    Only extracts known metadata keys to avoid stripping content brackets.

    Args:
        content: Text that may contain inline metadata

    Returns:
        Tuple of (clean_content, metadata_dict)
    """
    metadata = {}

    # Pattern for inline metadata: [key:value] where key is known
    # Build pattern from valid keys
    keys_pattern = "|".join(VALID_METADATA_KEYS)
    meta_pattern = rf"\[({keys_pattern}):([^\]]+)\]"

    # Find all matches
    matches = list(re.finditer(meta_pattern, content))

    if not matches:
        return content, metadata

    # Extract metadata and remove from content
    clean_content = content
    for match in reversed(matches):  # Reverse to preserve indices
        key, value = match.groups()
        metadata[key] = value.strip()
        clean_content = clean_content[:match.start()] + clean_content[match.end():]

    return clean_content.strip(), metadata


def process_memory_input(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse AXIS 3.4 format and return clean text + metadata.

    Input formats supported:
        "[V:FRC] [L:emotional] [C:2] [vec:say] Content {tags} [metadata]"

    Output:
        Tuple of (clean_content, metadata_dict)

    Example:
        Input: "[V:FRC] [L:emotional] [C:2] [vec:say] Kritik triggert Wut {trigger,intensity:7} [re:BMG]"
        Output: ("Kritik triggert Wut", {
            "source": "axis_protocol",
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
            "circuit": 2,
            "vector": "say",
            "axis_category": "health",
            "tags": {"trigger": True, "intensity": 7},
            "re": "BMG"
        })
    """
    parsed = ParsedMemory()
    text = raw_text.strip()

    # === PARSE VAULT [V:XXX] ===
    vault_match = re.search(r"\[V:([A-Z]+)\]", text)
    if vault_match:
        vault_code = vault_match.group(1)
        if vault_code in VAULT_CODES:
            parsed.vault = vault_code
            parsed.vault_full = VAULT_CODES[vault_code]
            parsed.axis_category = VAULT_TO_CATEGORY.get(
                parsed.vault_full, "personal"
            )

    # === PARSE LAYER [L:xxx] ===
    layer_match = re.search(r"\[L:([a-z]+)\]", text)
    if layer_match:
        layer = layer_match.group(1)
        if layer in VALID_LAYERS:
            parsed.layer = layer

    # === PARSE CIRCUIT [C:N] ===
    circuit_match = re.search(r"\[C:(\d)\]", text)
    if circuit_match:
        circuit = int(circuit_match.group(1))
        if circuit in VALID_CIRCUITS:
            parsed.circuit = circuit

    # === PARSE VECTOR [vec:xxx] ===
    vector_match = re.search(r"\[vec:([a-z]+)\]", text)
    if vector_match:
        vector = vector_match.group(1)
        if vector in VALID_VECTORS:
            parsed.vector = vector

    # === PARSE TAGS {xxx} ===
    tags_match = re.search(r"\{([^}]+)\}", text)
    if tags_match:
        parsed.tags = parse_tags(tags_match.group(0))

    # === EXTRACT CONTENT ===
    # Remove structural brackets: [V:X], [L:X], [C:N], [vec:X]
    clean_text = re.sub(r"\[V:[A-Z]+\]", "", text)
    clean_text = re.sub(r"\[L:[a-z]+\]", "", clean_text)
    clean_text = re.sub(r"\[C:\d\]", "", clean_text)
    clean_text = re.sub(r"\[vec:[a-z]+\]", "", clean_text)

    # Remove tags block
    clean_text = re.sub(r"\{[^}]+\}", "", clean_text)

    # Clean up whitespace
    clean_text = clean_text.strip()

    # === PARSE INLINE METADATA [re:X] [src:X] etc ===
    clean_text, inline_meta = parse_inline_metadata(clean_text)
    parsed.metadata = inline_meta
    parsed.content = clean_text

    return clean_text, parsed.to_metadata_dict()


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def parse_legacy_format(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse old/legacy format: [VAULT: X] [LAYER: Y] Content

    This maintains backward compatibility with the previous format.
    """
    metadata = {
        "source": "axis_protocol",
        "vault": DEFAULT_VAULT_FULL,
        "layer": DEFAULT_LAYER,
        "axis_category": DEFAULT_CATEGORY,
    }

    # Parse [VAULT: VAULT_NAME]
    vault_match = re.search(r"\[VAULT:\s*([A-Z_]+)\]", raw_text)
    if vault_match:
        metadata["vault"] = vault_match.group(1)
        metadata["axis_category"] = VAULT_TO_CATEGORY.get(
            metadata["vault"], "personal"
        )

    # Parse [LAYER: layer_name]
    layer_match = re.search(r"\[LAYER:\s*([a-z]+)\]", raw_text)
    if layer_match:
        metadata["layer"] = layer_match.group(1)

    # Clean text - remove all bracket blocks
    clean_text = re.sub(r"\[.*?\]", "", raw_text).strip()
    # Remove leading pipe character (legacy artifact)
    clean_text = clean_text.lstrip("| ").strip()

    return clean_text, metadata


def process_memory_input_auto(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Auto-detect format version and parse accordingly.

    Detects:
    - AXIS 3.4: Contains [V: pattern
    - Legacy: Contains [VAULT: pattern
    - Plain text: No recognized patterns

    Returns:
        Tuple of (clean_content, metadata_dict)
    """
    if "[V:" in raw_text:
        # New AXIS 3.4 format
        return process_memory_input(raw_text)
    elif "[VAULT:" in raw_text:
        # Legacy format
        return parse_legacy_format(raw_text)
    else:
        # Plain text - use defaults
        return raw_text.strip(), {
            "source": "axis_protocol",
            "vault": DEFAULT_VAULT_FULL,
            "layer": DEFAULT_LAYER,
            "axis_category": DEFAULT_CATEGORY,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_vault_full_name(code: str) -> str:
    """Convert vault short code to full name."""
    return VAULT_CODES.get(code, code)


def get_vault_category(vault_name: str) -> str:
    """Get the category for a vault name."""
    return VAULT_TO_CATEGORY.get(vault_name, "personal")


def is_ai_observation(metadata: Dict[str, Any]) -> bool:
    """Check if memory is an AI observation."""
    tags = metadata.get("tags", {})
    return tags.get("ai", False) is True


def get_confidence(metadata: Dict[str, Any]) -> Optional[float]:
    """Extract confidence value from metadata."""
    tags = metadata.get("tags", {})
    conf = tags.get("conf")
    if isinstance(conf, (int, float)):
        return float(conf)
    return None


def is_silent(metadata: Dict[str, Any]) -> bool:
    """Check if memory should not be surfaced yet."""
    tags = metadata.get("tags", {})
    return tags.get("silent", False) is True


def is_question(metadata: Dict[str, Any]) -> bool:
    """Check if memory is a queued question."""
    return metadata.get("vault") == "QUESTIONS_QUEUE"
