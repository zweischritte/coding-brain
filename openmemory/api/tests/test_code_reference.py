"""
Tests for Code Reference API.

Tests LineRange, CodeReference dataclasses and serialization/deserialization
for Phase 1 tags-based storage.
"""

import pytest
from datetime import datetime

from app.utils.code_reference import (
    LineRange,
    CodeReference,
    CodeReferenceError,
    compute_code_hash,
    serialize_code_refs_to_tags,
    deserialize_code_refs_from_tags,
    has_code_refs_in_tags,
    validate_code_refs_input,
    format_code_refs_for_response,
)


# =============================================================================
# LineRange Tests
# =============================================================================


class TestLineRange:
    """Tests for LineRange dataclass."""

    def test_valid_range(self):
        """Test creating a valid line range."""
        lr = LineRange(start=1, end=10)
        assert lr.start == 1
        assert lr.end == 10

    def test_single_line(self):
        """Test creating a single line range."""
        lr = LineRange(start=42, end=42)
        assert lr.start == 42
        assert lr.end == 42

    def test_invalid_start_zero(self):
        """Test that start < 1 raises error."""
        with pytest.raises(CodeReferenceError, match="start must be >= 1"):
            LineRange(start=0, end=10)

    def test_invalid_start_negative(self):
        """Test that negative start raises error."""
        with pytest.raises(CodeReferenceError, match="start must be >= 1"):
            LineRange(start=-1, end=10)

    def test_invalid_end_less_than_start(self):
        """Test that end < start raises error."""
        with pytest.raises(CodeReferenceError, match="end.*must be >= start"):
            LineRange(start=10, end=5)

    def test_non_integer_values(self):
        """Test that non-integer values raise error."""
        with pytest.raises(CodeReferenceError, match="must be integers"):
            LineRange(start="1", end=10)

    def test_to_fragment(self):
        """Test converting to URI fragment."""
        lr = LineRange(start=42, end=87)
        assert lr.to_fragment() == "#L42-L87"

    def test_to_fragment_single_line(self):
        """Test fragment for single line range."""
        lr = LineRange(start=100, end=100)
        assert lr.to_fragment() == "#L100-L100"

    def test_from_fragment(self):
        """Test parsing URI fragment."""
        lr = LineRange.from_fragment("#L42-L87")
        assert lr.start == 42
        assert lr.end == 87

    def test_from_fragment_single_line(self):
        """Test parsing single line fragment."""
        lr = LineRange.from_fragment("#L42")
        assert lr.start == 42
        assert lr.end == 42

    def test_from_fragment_invalid(self):
        """Test parsing invalid fragment."""
        with pytest.raises(CodeReferenceError, match="Invalid line fragment"):
            LineRange.from_fragment("L42-L87")  # Missing #

        with pytest.raises(CodeReferenceError, match="Invalid line fragment"):
            LineRange.from_fragment("#42-87")  # Missing L

    def test_from_string(self):
        """Test parsing 'start-end' string."""
        lr = LineRange.from_string("42-87")
        assert lr.start == 42
        assert lr.end == 87

    def test_from_string_single(self):
        """Test parsing single line string."""
        lr = LineRange.from_string("42")
        assert lr.start == 42
        assert lr.end == 42

    def test_to_string(self):
        """Test converting to 'start-end' format."""
        lr = LineRange(start=42, end=87)
        assert lr.to_string() == "42-87"


# =============================================================================
# CodeReference Tests
# =============================================================================


class TestCodeReference:
    """Tests for CodeReference dataclass."""

    def test_minimal_with_file_path(self):
        """Test creating reference with just file_path."""
        ref = CodeReference(file_path="/src/app.ts")
        assert ref.file_path == "/src/app.ts"
        assert ref.line_range is None
        assert ref.symbol_id is None
        assert ref.confidence_score == 0.8

    def test_minimal_with_symbol_id(self):
        """Test creating reference with just symbol_id."""
        ref = CodeReference(symbol_id="scip-typescript npm myapp app.ts/App#render().")
        assert ref.file_path is None
        assert ref.symbol_id == "scip-typescript npm myapp app.ts/App#render()."

    def test_requires_localization(self):
        """Test that at least one localization field is required."""
        with pytest.raises(CodeReferenceError, match="requires at least file_path or symbol_id"):
            CodeReference()

    def test_full_reference(self):
        """Test creating a full code reference."""
        ref = CodeReference(
            file_path="/apps/merlin/src/storage.ts",
            line_range=LineRange(start=42, end=87),
            symbol_id="scip-typescript npm merlin storage.ts/StorageService#createFileUploads().",
            git_commit="abc123def",
            code_hash="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            confidence_score=0.95,
            confidence_reason="user_provided",
        )
        assert ref.file_path == "/apps/merlin/src/storage.ts"
        assert ref.line_range.start == 42
        assert ref.line_range.end == 87
        assert ref.git_commit == "abc123def"
        assert ref.confidence_score == 0.95

    def test_invalid_confidence_score_too_high(self):
        """Test that confidence > 1.0 raises error."""
        with pytest.raises(CodeReferenceError, match="between 0.0 and 1.0"):
            CodeReference(file_path="/src/app.ts", confidence_score=1.5)

    def test_invalid_confidence_score_negative(self):
        """Test that negative confidence raises error."""
        with pytest.raises(CodeReferenceError, match="between 0.0 and 1.0"):
            CodeReference(file_path="/src/app.ts", confidence_score=-0.1)

    def test_invalid_code_hash_format(self):
        """Test that code_hash without sha256: prefix raises error."""
        with pytest.raises(CodeReferenceError, match="must start with 'sha256:'"):
            CodeReference(
                file_path="/src/app.ts",
                code_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            )

    def test_to_uri(self):
        """Test generating file: URI."""
        ref = CodeReference(
            file_path="/apps/merlin/src/storage.ts",
            line_range=LineRange(start=42, end=87),
        )
        assert ref.to_uri() == "file:/apps/merlin/src/storage.ts#L42-L87"

    def test_to_uri_no_lines(self):
        """Test URI without line range."""
        ref = CodeReference(file_path="/apps/merlin/src/storage.ts")
        assert ref.to_uri() == "file:/apps/merlin/src/storage.ts"

    def test_to_uri_requires_file_path(self):
        """Test that to_uri raises without file_path."""
        ref = CodeReference(symbol_id="scip-typescript npm myapp app.ts/App#render().")
        with pytest.raises(CodeReferenceError, match="Cannot create URI without file_path"):
            ref.to_uri()

    def test_from_uri(self):
        """Test parsing file: URI."""
        ref = CodeReference.from_uri("file:/apps/merlin/src/storage.ts#L42-L87")
        assert ref.file_path == "/apps/merlin/src/storage.ts"
        assert ref.line_range.start == 42
        assert ref.line_range.end == 87

    def test_from_uri_no_fragment(self):
        """Test parsing URI without fragment."""
        ref = CodeReference.from_uri("file:/apps/merlin/src/storage.ts")
        assert ref.file_path == "/apps/merlin/src/storage.ts"
        assert ref.line_range is None

    def test_from_uri_invalid(self):
        """Test parsing invalid URI."""
        with pytest.raises(CodeReferenceError, match="Invalid file URI"):
            CodeReference.from_uri("/apps/merlin/src/storage.ts")  # Missing file:

    def test_is_stale(self):
        """Test staleness check."""
        ref = CodeReference(file_path="/src/app.ts")
        assert not ref.is_stale()

        ref.stale_since = datetime.utcnow()
        assert ref.is_stale()

    def test_to_dict(self):
        """Test serializing to dict."""
        ref = CodeReference(
            file_path="/apps/merlin/src/storage.ts",
            line_range=LineRange(start=42, end=87),
            symbol_id="StorageService#createFileUploads",
            code_hash="sha256:abc123",
            confidence_score=0.95,
            confidence_reason="user_provided",
        )
        d = ref.to_dict()
        assert d["file_path"] == "/apps/merlin/src/storage.ts"
        assert d["line_start"] == 42
        assert d["line_end"] == 87
        assert d["symbol_id"] == "StorageService#createFileUploads"
        assert d["code_hash"] == "sha256:abc123"
        assert d["confidence_score"] == 0.95
        assert d["confidence_reason"] == "user_provided"
        assert d.get("stale_since") is None

    def test_from_dict(self):
        """Test deserializing from dict."""
        d = {
            "file_path": "/apps/merlin/src/storage.ts",
            "line_start": 42,
            "line_end": 87,
            "symbol_id": "StorageService#createFileUploads",
            "code_hash": "sha256:abc123",
            "confidence_score": 0.95,
            "confidence_reason": "user_provided",
        }
        ref = CodeReference.from_dict(d)
        assert ref.file_path == "/apps/merlin/src/storage.ts"
        assert ref.line_range.start == 42
        assert ref.line_range.end == 87
        assert ref.symbol_id == "StorageService#createFileUploads"
        assert ref.code_hash == "sha256:abc123"
        assert ref.confidence_score == 0.95

    def test_from_dict_with_lines_string(self):
        """Test deserializing with 'lines' format."""
        d = {
            "file_path": "/src/app.ts",
            "lines": "42-87",
        }
        ref = CodeReference.from_dict(d)
        assert ref.line_range.start == 42
        assert ref.line_range.end == 87


# =============================================================================
# Code Hash Tests
# =============================================================================


class TestCodeHash:
    """Tests for code hash computation."""

    def test_compute_code_hash(self):
        """Test computing hash of content."""
        content = "function test() { return 42; }"
        h = compute_code_hash(content)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # sha256: prefix + 64 hex chars

    def test_hash_consistency(self):
        """Test that same content produces same hash."""
        content = "function test() { return 42; }"
        h1 = compute_code_hash(content)
        h2 = compute_code_hash(content)
        assert h1 == h2

    def test_hash_different_content(self):
        """Test that different content produces different hash."""
        h1 = compute_code_hash("function test() { return 42; }")
        h2 = compute_code_hash("function test() { return 43; }")
        assert h1 != h2


# =============================================================================
# Tags Serialization Tests
# =============================================================================


class TestTagsSerialization:
    """Tests for tags-based serialization (Phase 1)."""

    def test_serialize_single_ref(self):
        """Test serializing a single code reference."""
        refs = [
            CodeReference(
                file_path="/src/app.ts",
                line_range=LineRange(start=42, end=87),
                symbol_id="App#render",
                code_hash="sha256:abc123",
                git_commit="def456",
                confidence_score=0.95,
            )
        ]
        tags = serialize_code_refs_to_tags(refs)

        assert tags["code_ref_count"] == 1
        assert tags["code_ref_0_path"] == "/src/app.ts"
        assert tags["code_ref_0_lines"] == "42-87"
        assert tags["code_ref_0_symbol"] == "App#render"
        assert tags["code_ref_0_hash"] == "sha256:abc123"
        assert tags["code_ref_0_commit"] == "def456"
        assert tags["code_ref_0_confidence"] == 0.95

    def test_serialize_multiple_refs(self):
        """Test serializing multiple code references."""
        refs = [
            CodeReference(file_path="/src/a.ts", line_range=LineRange(1, 10)),
            CodeReference(file_path="/src/b.ts", line_range=LineRange(20, 30)),
        ]
        tags = serialize_code_refs_to_tags(refs)

        assert tags["code_ref_count"] == 2
        assert tags["code_ref_0_path"] == "/src/a.ts"
        assert tags["code_ref_1_path"] == "/src/b.ts"

    def test_serialize_empty_list(self):
        """Test serializing empty list returns empty dict."""
        tags = serialize_code_refs_to_tags([])
        assert tags == {}

    def test_deserialize_single_ref(self):
        """Test deserializing a single code reference."""
        tags = {
            "code_ref_count": 1,
            "code_ref_0_path": "/src/app.ts",
            "code_ref_0_lines": "42-87",
            "code_ref_0_symbol": "App#render",
            "code_ref_0_hash": "sha256:abc123",
            "code_ref_0_commit": "def456",
            "code_ref_0_confidence": 0.95,
        }
        refs = deserialize_code_refs_from_tags(tags)

        assert len(refs) == 1
        assert refs[0].file_path == "/src/app.ts"
        assert refs[0].line_range.start == 42
        assert refs[0].line_range.end == 87
        assert refs[0].symbol_id == "App#render"
        assert refs[0].code_hash == "sha256:abc123"
        assert refs[0].git_commit == "def456"
        assert refs[0].confidence_score == 0.95

    def test_deserialize_without_count(self):
        """Test deserializing without explicit code_ref_count."""
        tags = {
            "code_ref_0_path": "/src/app.ts",
            "code_ref_0_lines": "42-87",
        }
        refs = deserialize_code_refs_from_tags(tags)

        assert len(refs) == 1
        assert refs[0].file_path == "/src/app.ts"

    def test_deserialize_multiple_refs(self):
        """Test deserializing multiple code references."""
        tags = {
            "code_ref_count": 2,
            "code_ref_0_path": "/src/a.ts",
            "code_ref_0_lines": "1-10",
            "code_ref_1_path": "/src/b.ts",
            "code_ref_1_lines": "20-30",
        }
        refs = deserialize_code_refs_from_tags(tags)

        assert len(refs) == 2
        assert refs[0].file_path == "/src/a.ts"
        assert refs[1].file_path == "/src/b.ts"

    def test_deserialize_empty(self):
        """Test deserializing empty tags returns empty list."""
        refs = deserialize_code_refs_from_tags({})
        assert refs == []

    def test_has_code_refs_in_tags(self):
        """Test detecting code_refs in tags."""
        # With count
        assert has_code_refs_in_tags({"code_ref_count": 1, "code_ref_0_path": "/src/app.ts"})

        # Without count but with path
        assert has_code_refs_in_tags({"code_ref_0_path": "/src/app.ts"})

        # Without count but with symbol
        assert has_code_refs_in_tags({"code_ref_0_symbol": "App#render"})

        # No code_refs
        assert not has_code_refs_in_tags({"decision": True})
        assert not has_code_refs_in_tags({})
        assert not has_code_refs_in_tags(None)

    def test_roundtrip(self):
        """Test serialize/deserialize roundtrip."""
        original = [
            CodeReference(
                file_path="/apps/merlin/src/storage.ts",
                line_range=LineRange(start=42, end=87),
                symbol_id="StorageService#createFileUploads",
                code_hash="sha256:abc123",
                git_commit="def456",
                confidence_score=0.95,
                confidence_reason="user_provided",
            ),
            CodeReference(
                file_path="/apps/merlin/src/utils.ts",
                line_range=LineRange(start=10, end=20),
                confidence_score=0.8,
            ),
        ]

        tags = serialize_code_refs_to_tags(original)
        restored = deserialize_code_refs_from_tags(tags)

        assert len(restored) == len(original)
        assert restored[0].file_path == original[0].file_path
        assert restored[0].line_range.start == original[0].line_range.start
        assert restored[0].symbol_id == original[0].symbol_id
        assert restored[0].code_hash == original[0].code_hash
        assert restored[0].git_commit == original[0].git_commit
        assert restored[0].confidence_score == original[0].confidence_score

        assert restored[1].file_path == original[1].file_path
        assert restored[1].line_range.start == original[1].line_range.start


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestValidateCodeRefsInput:
    """Tests for validate_code_refs_input function."""

    def test_none_returns_empty_list(self):
        """Test that None input returns empty list."""
        result = validate_code_refs_input(None)
        assert result == []

    def test_single_code_reference(self):
        """Test passing a single CodeReference object."""
        ref = CodeReference(file_path="/src/app.ts")
        result = validate_code_refs_input(ref)
        assert len(result) == 1
        assert result[0].file_path == "/src/app.ts"

    def test_single_dict(self):
        """Test passing a single dict (converted to list)."""
        result = validate_code_refs_input({
            "file_path": "/src/app.ts",
            "line_start": 42,
            "line_end": 87,
        })
        assert len(result) == 1
        assert result[0].file_path == "/src/app.ts"

    def test_list_of_dicts(self):
        """Test passing list of dicts."""
        result = validate_code_refs_input([
            {"file_path": "/src/a.ts", "line_start": 1, "line_end": 10},
            {"file_path": "/src/b.ts", "line_start": 20, "line_end": 30},
        ])
        assert len(result) == 2

    def test_list_of_code_references(self):
        """Test passing list of CodeReference objects."""
        refs = [
            CodeReference(file_path="/src/a.ts"),
            CodeReference(file_path="/src/b.ts"),
        ]
        result = validate_code_refs_input(refs)
        assert len(result) == 2

    def test_mixed_list(self):
        """Test passing mixed list of dicts and CodeReferences."""
        refs = [
            CodeReference(file_path="/src/a.ts"),
            {"file_path": "/src/b.ts", "line_start": 20, "line_end": 30},
        ]
        result = validate_code_refs_input(refs)
        assert len(result) == 2

    def test_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(CodeReferenceError, match="must be a list"):
            validate_code_refs_input("not a list")

    def test_invalid_item_type(self):
        """Test that invalid item type raises error."""
        with pytest.raises(CodeReferenceError, match="must be dict or CodeReference"):
            validate_code_refs_input([123])

    def test_invalid_dict_content(self):
        """Test that invalid dict content raises error."""
        with pytest.raises(CodeReferenceError, match="Invalid code_ref at index 0"):
            validate_code_refs_input([{"invalid_field": "no file_path or symbol_id"}])


# =============================================================================
# Response Formatting Tests
# =============================================================================


class TestFormatCodeRefsForResponse:
    """Tests for format_code_refs_for_response function."""

    def test_format_single_ref(self):
        """Test formatting a single reference."""
        refs = [
            CodeReference(
                file_path="/apps/merlin/src/storage.ts",
                line_range=LineRange(start=42, end=87),
                symbol_id="StorageService#createFileUploads",
                confidence_score=0.95,
            )
        ]
        result = format_code_refs_for_response(refs)

        assert len(result) == 1
        assert result[0]["file_path"] == "/apps/merlin/src/storage.ts"
        assert result[0]["line_range"] == {"start": 42, "end": 87}
        assert result[0]["symbol_id"] == "StorageService#createFileUploads"
        assert result[0]["code_link"] == "file:/apps/merlin/src/storage.ts#L42-L87"
        assert result[0]["is_stale"] is False
        assert result[0]["confidence_score"] == 0.95

    def test_format_stale_ref(self):
        """Test formatting a stale reference."""
        ref = CodeReference(file_path="/src/app.ts")
        ref.stale_since = datetime.utcnow()

        result = format_code_refs_for_response([ref])
        assert result[0]["is_stale"] is True

    def test_format_minimal_ref(self):
        """Test formatting minimal reference."""
        refs = [CodeReference(file_path="/src/app.ts")]
        result = format_code_refs_for_response(refs)

        assert result[0]["file_path"] == "/src/app.ts"
        assert result[0]["code_link"] == "file:/src/app.ts"
        assert "line_range" not in result[0]
        assert "symbol_id" not in result[0]

    def test_format_symbol_only(self):
        """Test formatting symbol-only reference."""
        refs = [CodeReference(symbol_id="App#render")]
        result = format_code_refs_for_response(refs)

        assert result[0]["symbol_id"] == "App#render"
        assert "file_path" not in result[0]
        assert "code_link" not in result[0]  # No code_link without file_path
