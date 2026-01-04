"""
Code Reference API for Coding Brain.

Provides structured code references for linking memories to specific code locations.
Supports file paths, line ranges, SCIP symbol IDs, and git versioning.

Phase 1 implements tags-based storage without schema migration.
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# =============================================================================
# VALIDATION ERROR
# =============================================================================


class CodeReferenceError(Exception):
    """Raised when code reference parameters are invalid."""


# =============================================================================
# LINE RANGE
# =============================================================================


@dataclass
class LineRange:
    """Start- and end-line of a code reference (1-indexed, inclusive)."""

    start: int
    end: int

    def __post_init__(self):
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise CodeReferenceError(
                f"LineRange start and end must be integers, "
                f"got {type(self.start).__name__} and {type(self.end).__name__}"
            )
        if self.start < 1:
            raise CodeReferenceError(
                f"LineRange start must be >= 1, got {self.start}"
            )
        if self.end < self.start:
            raise CodeReferenceError(
                f"LineRange end ({self.end}) must be >= start ({self.start})"
            )

    def to_fragment(self) -> str:
        """Convert to URI fragment: #L42-L87"""
        return f"#L{self.start}-L{self.end}"

    @classmethod
    def from_fragment(cls, fragment: str) -> "LineRange":
        """Parse URI fragment: #L42-L87 -> LineRange(42, 87)

        Also supports single line: #L42 -> LineRange(42, 42)
        """
        # Match #L42-L87 or #L42
        match = re.match(r"#L(\d+)(?:-L(\d+))?", fragment)
        if not match:
            raise CodeReferenceError(f"Invalid line fragment: {fragment}")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        return cls(start=start, end=end)

    @classmethod
    def from_string(cls, lines_str: str) -> "LineRange":
        """Parse 'start-end' or 'start' string -> LineRange(start, end)"""
        if "-" in lines_str:
            parts = lines_str.split("-", 1)
            return cls(start=int(parts[0]), end=int(parts[1]))
        else:
            line = int(lines_str)
            return cls(start=line, end=line)

    def to_string(self) -> str:
        """Convert to 'start-end' format for storage."""
        return f"{self.start}-{self.end}"


# =============================================================================
# CODE REFERENCE
# =============================================================================


@dataclass
class CodeReference:
    """Structured code reference for a memory.

    A code reference points to a specific location in source code:
    - file_path: absolute or repo-relative path to the file
    - line_range: start and end lines (1-indexed, inclusive)
    - symbol_id: SCIP-compatible symbol identifier

    It also tracks versioning information:
    - git_commit: commit SHA when the reference was created
    - code_hash: SHA256 of the referenced code block

    And temporal tracking:
    - created_at: when this reference was created
    - last_verified_at: when the code was last verified current
    - stale_since: when the code was detected as changed

    And confidence:
    - confidence_score: 0.0-1.0 how reliable is this reference
    - confidence_reason: why this score (e.g., 'user_provided', 'inferred_from_code_index')
    """

    # Localization (at least file_path recommended)
    file_path: Optional[str] = None
    line_range: Optional[LineRange] = None
    symbol_id: Optional[str] = None

    # Versioning
    git_commit: Optional[str] = None
    code_hash: Optional[str] = None

    # Temporal tracking
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_verified_at: Optional[datetime] = None
    stale_since: Optional[datetime] = None

    # Confidence
    confidence_score: float = 0.8
    confidence_reason: Optional[str] = None

    def __post_init__(self):
        # Validate at least one localization field is set
        if not any([self.file_path, self.symbol_id]):
            raise CodeReferenceError(
                "CodeReference requires at least file_path or symbol_id"
            )

        # Validate confidence_score range
        if not isinstance(self.confidence_score, (int, float)):
            raise CodeReferenceError(
                f"confidence_score must be a number, got {type(self.confidence_score).__name__}"
            )
        if not 0.0 <= self.confidence_score <= 1.0:
            raise CodeReferenceError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )

        # Validate code_hash format if provided
        if self.code_hash and not self.code_hash.startswith("sha256:"):
            raise CodeReferenceError(
                f"code_hash must start with 'sha256:', got {self.code_hash}"
            )

    def to_uri(self) -> str:
        """Convert to file: URI with fragment.

        Example: file:/apps/merlin/src/storage.ts#L42-L87
        """
        if not self.file_path:
            raise CodeReferenceError("Cannot create URI without file_path")

        uri = f"file:{self.file_path}"
        if self.line_range:
            uri += self.line_range.to_fragment()
        return uri

    @classmethod
    def from_uri(cls, uri: str, **kwargs) -> "CodeReference":
        """Parse file: URI: file:/path/to/file.ts#L42-L87

        Additional kwargs are passed to the CodeReference constructor.
        """
        match = re.match(r"file:([^#]+)(#L\d+(?:-L\d+)?)?", uri)
        if not match:
            raise CodeReferenceError(f"Invalid file URI: {uri}")

        file_path = match.group(1)
        line_range = None
        if match.group(2):
            line_range = LineRange.from_fragment(match.group(2))

        return cls(file_path=file_path, line_range=line_range, **kwargs)

    def is_stale(self) -> bool:
        """Check if this reference is marked as stale."""
        return self.stale_since is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/tags storage (Phase 1 format)."""
        result: Dict[str, Any] = {}

        if self.file_path:
            result["file_path"] = self.file_path
        if self.line_range:
            result["line_start"] = self.line_range.start
            result["line_end"] = self.line_range.end
        if self.symbol_id:
            result["symbol_id"] = self.symbol_id
        if self.git_commit:
            result["git_commit"] = self.git_commit
        if self.code_hash:
            result["code_hash"] = self.code_hash

        result["confidence_score"] = round(self.confidence_score, 2)
        if self.confidence_reason:
            result["confidence_reason"] = self.confidence_reason

        if self.stale_since:
            result["stale_since"] = self.stale_since.isoformat()
        if self.last_verified_at:
            result["last_verified_at"] = self.last_verified_at.isoformat()

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeReference":
        """Deserialize from JSON/tags storage."""
        line_range = None
        if data.get("line_start") is not None and data.get("line_end") is not None:
            line_range = LineRange(
                start=int(data["line_start"]),
                end=int(data["line_end"])
            )
        elif data.get("lines"):
            # Support "42-87" string format
            line_range = LineRange.from_string(data["lines"])

        stale_since = None
        if data.get("stale_since"):
            stale_since = datetime.fromisoformat(data["stale_since"])

        last_verified_at = None
        if data.get("last_verified_at"):
            last_verified_at = datetime.fromisoformat(data["last_verified_at"])

        return cls(
            file_path=data.get("file_path"),
            line_range=line_range,
            symbol_id=data.get("symbol_id"),
            git_commit=data.get("git_commit"),
            code_hash=data.get("code_hash"),
            confidence_score=float(data.get("confidence_score", 0.8)),
            confidence_reason=data.get("confidence_reason"),
            stale_since=stale_since,
            last_verified_at=last_verified_at,
        )


# =============================================================================
# CODE HASH UTILITIES
# =============================================================================


def compute_code_hash(content: str) -> str:
    """Compute SHA256 hash of code content with sha256: prefix."""
    hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{hash_value}"


def compute_code_hash_from_file(
    file_path: str, start_line: int, end_line: int
) -> str:
    """Compute SHA256 of a code block from a file.

    Args:
        file_path: Path to the source file
        start_line: Start line (1-indexed, inclusive)
        end_line: End line (1-indexed, inclusive)

    Returns:
        Hash string with sha256: prefix

    Raises:
        FileNotFoundError: If file does not exist
        CodeReferenceError: If line range is invalid
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if start_line < 1 or end_line > len(lines):
        raise CodeReferenceError(
            f"Line range {start_line}-{end_line} out of bounds "
            f"(file has {len(lines)} lines)"
        )

    block = "".join(lines[start_line - 1 : end_line])
    return compute_code_hash(block)


# =============================================================================
# TAGS-BASED SERIALIZATION (Phase 1)
# =============================================================================


def serialize_code_refs_to_tags(code_refs: List[CodeReference]) -> Dict[str, Any]:
    """Serialize code references to tags dict for Phase 1 storage.

    Format:
        code_ref_count: 2
        code_ref_0_path: /apps/merlin/src/storage.ts
        code_ref_0_lines: 42-87
        code_ref_0_symbol: StorageService#createFileUploads
        code_ref_0_hash: sha256:e3b0c44...
        code_ref_0_commit: abc123def
        code_ref_0_confidence: 0.95
        code_ref_1_path: ...
    """
    tags: Dict[str, Any] = {}

    if not code_refs:
        return tags

    tags["code_ref_count"] = len(code_refs)

    for idx, ref in enumerate(code_refs):
        prefix = f"code_ref_{idx}_"

        if ref.file_path:
            tags[f"{prefix}path"] = ref.file_path
        if ref.line_range:
            tags[f"{prefix}lines"] = ref.line_range.to_string()
        if ref.symbol_id:
            tags[f"{prefix}symbol"] = ref.symbol_id
        if ref.code_hash:
            tags[f"{prefix}hash"] = ref.code_hash
        if ref.git_commit:
            tags[f"{prefix}commit"] = ref.git_commit
        tags[f"{prefix}confidence"] = round(ref.confidence_score, 2)
        if ref.confidence_reason:
            tags[f"{prefix}reason"] = ref.confidence_reason
        if ref.stale_since:
            tags[f"{prefix}stale_since"] = ref.stale_since.isoformat()

    return tags


def deserialize_code_refs_from_tags(tags: Dict[str, Any]) -> List[CodeReference]:
    """Deserialize code references from tags dict.

    Returns list of CodeReference objects extracted from tags.
    """
    code_refs: List[CodeReference] = []

    count = tags.get("code_ref_count", 0)
    if not count:
        # Try to detect code_refs by looking for code_ref_0_path
        if "code_ref_0_path" in tags or "code_ref_0_symbol" in tags:
            # Count manually
            idx = 0
            while f"code_ref_{idx}_path" in tags or f"code_ref_{idx}_symbol" in tags:
                idx += 1
            count = idx

    for idx in range(count):
        prefix = f"code_ref_{idx}_"

        file_path = tags.get(f"{prefix}path")
        symbol_id = tags.get(f"{prefix}symbol")

        if not file_path and not symbol_id:
            continue

        line_range = None
        lines_str = tags.get(f"{prefix}lines")
        if lines_str:
            try:
                line_range = LineRange.from_string(lines_str)
            except (ValueError, CodeReferenceError):
                pass

        stale_since = None
        stale_str = tags.get(f"{prefix}stale_since")
        if stale_str:
            try:
                stale_since = datetime.fromisoformat(stale_str)
            except ValueError:
                pass

        code_refs.append(
            CodeReference(
                file_path=file_path,
                line_range=line_range,
                symbol_id=symbol_id,
                code_hash=tags.get(f"{prefix}hash"),
                git_commit=tags.get(f"{prefix}commit"),
                confidence_score=float(tags.get(f"{prefix}confidence", 0.8)),
                confidence_reason=tags.get(f"{prefix}reason"),
                stale_since=stale_since,
            )
        )

    return code_refs


def has_code_refs_in_tags(tags: Dict[str, Any]) -> bool:
    """Check if tags contain code references."""
    if not tags:
        return False
    return (
        tags.get("code_ref_count", 0) > 0
        or "code_ref_0_path" in tags
        or "code_ref_0_symbol" in tags
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_code_refs_input(code_refs: Any) -> List[CodeReference]:
    """Validate and normalize code_refs input.

    Accepts:
        - List of CodeReference objects
        - List of dicts with code_ref fields
        - Single dict (converted to list of one)

    Returns:
        List of validated CodeReference objects
    """
    if code_refs is None:
        return []

    if isinstance(code_refs, CodeReference):
        return [code_refs]

    if isinstance(code_refs, dict):
        code_refs = [code_refs]

    if not isinstance(code_refs, list):
        raise CodeReferenceError(
            f"code_refs must be a list, got {type(code_refs).__name__}"
        )

    validated: List[CodeReference] = []
    for idx, ref in enumerate(code_refs):
        if isinstance(ref, CodeReference):
            validated.append(ref)
        elif isinstance(ref, dict):
            try:
                validated.append(CodeReference.from_dict(ref))
            except (CodeReferenceError, ValueError, TypeError) as e:
                raise CodeReferenceError(
                    f"Invalid code_ref at index {idx}: {e}"
                ) from e
        else:
            raise CodeReferenceError(
                f"code_ref at index {idx} must be dict or CodeReference, "
                f"got {type(ref).__name__}"
            )

    return validated


# =============================================================================
# RESPONSE FORMATTING
# =============================================================================


def format_code_refs_for_response(code_refs: List[CodeReference]) -> List[Dict[str, Any]]:
    """Format code references for API response (lean format).

    Returns list of dicts with:
        - file_path, line_range (as {start, end}), symbol_id
        - code_link: file:/path#L42-L87 URI
        - is_stale: boolean
        - confidence_score: float
    """
    result = []
    for ref in code_refs:
        formatted: Dict[str, Any] = {}

        if ref.file_path:
            formatted["file_path"] = ref.file_path
        if ref.line_range:
            formatted["line_range"] = {
                "start": ref.line_range.start,
                "end": ref.line_range.end,
            }
        if ref.symbol_id:
            formatted["symbol_id"] = ref.symbol_id

        # Generate code_link URI
        if ref.file_path:
            formatted["code_link"] = ref.to_uri()

        formatted["is_stale"] = ref.is_stale()
        formatted["confidence_score"] = round(ref.confidence_score, 2)

        result.append(formatted)

    return result
