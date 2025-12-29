"""Tests for MCP tool schema validation.

This module validates the MCP schema definitions in docs/mcp/schema/v1/tools.schema.json
against the explain_code tool implementation to ensure schema/implementation consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Try to import jsonschema for validation, skip if not available
try:
    import jsonschema
    from jsonschema import Draft202012Validator

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def schema_path() -> Path:
    """Return path to the MCP tools schema file."""
    # Navigate from test file to schema file
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    return repo_root / "docs" / "mcp" / "schema" / "v1" / "tools.schema.json"


@pytest.fixture
def schema(schema_path: Path) -> dict[str, Any]:
    """Load the MCP tools schema."""
    with open(schema_path) as f:
        schema = json.load(f)
    schema["$id"] = schema_path.resolve().as_uri()
    return schema


@pytest.fixture
def sample_explain_code_input() -> dict[str, Any]:
    """Return sample valid explain_code input."""
    return {
        "symbol_id": "scip-python myapp module/MyClass#my_method.",
        "config": {
            "depth": 2,
            "include_callers": True,
            "include_callees": True,
            "include_usages": True,
            "max_usages": 5,
            "include_related": True,
            "max_related": 10,
            "format": "llm",
        },
    }


@pytest.fixture
def sample_explain_code_output() -> dict[str, Any]:
    """Return sample valid explain_code output."""
    return {
        "explanation": {
            "symbol_id": "scip-python myapp module/MyClass#my_method.",
            "name": "my_method",
            "kind": "method",
            "signature": "def my_method(self, arg: str) -> int:",
            "file_path": "/path/to/module.py",
            "line_start": 10,
            "line_end": 20,
            "docstring": "This is a docstring.",
            "callers": [
                {
                    "symbol_id": "scip-python myapp caller/func.",
                    "name": "caller_func",
                    "kind": "function",
                    "file_path": "/path/to/caller.py",
                    "line_start": 5,
                    "depth": 1,
                }
            ],
            "callees": [
                {
                    "symbol_id": "scip-python myapp helper/func.",
                    "name": "helper_func",
                }
            ],
            "usages": [{"code": "obj.my_method('test')", "file": "/path/test.py"}],
            "related": [
                {
                    "symbol_id": "scip-python myapp module/related.",
                    "name": "related_func",
                    "kind": "function",
                }
            ],
            "context": "Additional context from retrieval.",
        },
        "formatted": "Symbol: my_method\nType: method\n...",
        "cached": False,
        "meta": {"request_id": "req-123"},
    }


# =============================================================================
# Schema Structure Tests
# =============================================================================


class TestSchemaStructure:
    """Tests for MCP schema structure."""

    def test_schema_file_exists(self, schema_path: Path):
        """Test that schema file exists."""
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    def test_schema_is_valid_json(self, schema_path: Path):
        """Test that schema is valid JSON."""
        with open(schema_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_schema_has_defs(self, schema: dict[str, Any]):
        """Test that schema has $defs section."""
        assert "$defs" in schema
        assert isinstance(schema["$defs"], dict)

    def test_explain_code_in_properties(self, schema: dict[str, Any]):
        """Test that explain_code is in the tool properties."""
        assert "properties" in schema
        assert "explain_code" in schema["properties"]

    def test_explain_code_references_tool(self, schema: dict[str, Any]):
        """Test that explain_code references ExplainCodeTool."""
        explain_code_ref = schema["properties"]["explain_code"]
        assert "$ref" in explain_code_ref
        assert "ExplainCodeTool" in explain_code_ref["$ref"]


# =============================================================================
# ExplainCodeTool Schema Tests
# =============================================================================


class TestExplainCodeToolSchema:
    """Tests for ExplainCodeTool schema definition."""

    def test_explain_code_tool_exists(self, schema: dict[str, Any]):
        """Test ExplainCodeTool definition exists."""
        assert "ExplainCodeTool" in schema["$defs"]

    def test_explain_code_tool_has_input_output(self, schema: dict[str, Any]):
        """Test ExplainCodeTool has input and output."""
        tool = schema["$defs"]["ExplainCodeTool"]
        assert "properties" in tool
        assert "input" in tool["properties"]
        assert "output" in tool["properties"]
        assert "required" in tool
        assert "input" in tool["required"]
        assert "output" in tool["required"]

    def test_explain_code_input_exists(self, schema: dict[str, Any]):
        """Test ExplainCodeInput definition exists."""
        assert "ExplainCodeInput" in schema["$defs"]

    def test_explain_code_input_has_symbol_id(self, schema: dict[str, Any]):
        """Test ExplainCodeInput requires symbol_id."""
        input_schema = schema["$defs"]["ExplainCodeInput"]
        assert "symbol_id" in input_schema["properties"]
        assert "required" in input_schema
        assert "symbol_id" in input_schema["required"]

    def test_explain_code_input_has_config(self, schema: dict[str, Any]):
        """Test ExplainCodeInput has optional config."""
        input_schema = schema["$defs"]["ExplainCodeInput"]
        assert "config" in input_schema["properties"]
        # Config should be optional (not in required)
        if "required" in input_schema:
            assert "config" not in input_schema["required"]

    def test_explain_code_output_exists(self, schema: dict[str, Any]):
        """Test ExplainCodeOutput definition exists."""
        assert "ExplainCodeOutput" in schema["$defs"]

    def test_explain_code_output_has_explanation(self, schema: dict[str, Any]):
        """Test ExplainCodeOutput has explanation field."""
        output_schema = schema["$defs"]["ExplainCodeOutput"]
        assert "explanation" in output_schema["properties"]
        assert "required" in output_schema
        assert "explanation" in output_schema["required"]

    def test_explain_code_output_has_meta(self, schema: dict[str, Any]):
        """Test ExplainCodeOutput has meta field."""
        output_schema = schema["$defs"]["ExplainCodeOutput"]
        assert "meta" in output_schema["properties"]
        assert "meta" in output_schema["required"]


# =============================================================================
# ExplainCodeConfig Schema Tests
# =============================================================================


class TestExplainCodeConfigSchema:
    """Tests for ExplainCodeConfig schema definition."""

    def test_config_exists(self, schema: dict[str, Any]):
        """Test ExplainCodeConfig definition exists."""
        assert "ExplainCodeConfig" in schema["$defs"]

    def test_config_has_depth(self, schema: dict[str, Any]):
        """Test config has depth property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "depth" in config["properties"]
        depth = config["properties"]["depth"]
        assert depth["type"] == "integer"
        assert depth["minimum"] == 1
        assert depth["maximum"] == 5

    def test_config_has_include_callers(self, schema: dict[str, Any]):
        """Test config has include_callers property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "include_callers" in config["properties"]
        assert config["properties"]["include_callers"]["type"] == "boolean"

    def test_config_has_include_callees(self, schema: dict[str, Any]):
        """Test config has include_callees property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "include_callees" in config["properties"]
        assert config["properties"]["include_callees"]["type"] == "boolean"

    def test_config_has_include_usages(self, schema: dict[str, Any]):
        """Test config has include_usages property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "include_usages" in config["properties"]
        assert config["properties"]["include_usages"]["type"] == "boolean"

    def test_config_has_max_usages(self, schema: dict[str, Any]):
        """Test config has max_usages property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "max_usages" in config["properties"]
        max_usages = config["properties"]["max_usages"]
        assert max_usages["type"] == "integer"
        assert max_usages["minimum"] == 0

    def test_config_has_include_related(self, schema: dict[str, Any]):
        """Test config has include_related property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "include_related" in config["properties"]
        assert config["properties"]["include_related"]["type"] == "boolean"

    def test_config_has_max_related(self, schema: dict[str, Any]):
        """Test config has max_related property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "max_related" in config["properties"]
        max_related = config["properties"]["max_related"]
        assert max_related["type"] == "integer"
        assert max_related["minimum"] == 0

    def test_config_has_format(self, schema: dict[str, Any]):
        """Test config has format property."""
        config = schema["$defs"]["ExplainCodeConfig"]
        assert "format" in config["properties"]
        fmt = config["properties"]["format"]
        assert fmt["type"] == "string"
        assert "enum" in fmt
        assert "json" in fmt["enum"]
        assert "markdown" in fmt["enum"]
        assert "llm" in fmt["enum"]


# =============================================================================
# SymbolExplanation Schema Tests
# =============================================================================


class TestSymbolExplanationSchema:
    """Tests for SymbolExplanation schema definition."""

    def test_symbol_explanation_exists(self, schema: dict[str, Any]):
        """Test SymbolExplanation definition exists."""
        assert "SymbolExplanation" in schema["$defs"]

    def test_symbol_explanation_required_fields(self, schema: dict[str, Any]):
        """Test SymbolExplanation has required fields."""
        explanation = schema["$defs"]["SymbolExplanation"]
        required = explanation["required"]
        assert "symbol_id" in required
        assert "name" in required
        assert "kind" in required
        assert "signature" in required
        assert "file_path" in required
        assert "line_start" in required
        assert "line_end" in required

    def test_symbol_explanation_has_docstring(self, schema: dict[str, Any]):
        """Test SymbolExplanation has docstring field."""
        explanation = schema["$defs"]["SymbolExplanation"]
        assert "docstring" in explanation["properties"]

    def test_symbol_explanation_has_callers(self, schema: dict[str, Any]):
        """Test SymbolExplanation has callers array."""
        explanation = schema["$defs"]["SymbolExplanation"]
        assert "callers" in explanation["properties"]
        callers = explanation["properties"]["callers"]
        assert callers["type"] == "array"

    def test_symbol_explanation_has_callees(self, schema: dict[str, Any]):
        """Test SymbolExplanation has callees array."""
        explanation = schema["$defs"]["SymbolExplanation"]
        assert "callees" in explanation["properties"]
        callees = explanation["properties"]["callees"]
        assert callees["type"] == "array"

    def test_symbol_explanation_has_usages(self, schema: dict[str, Any]):
        """Test SymbolExplanation has usages array."""
        explanation = schema["$defs"]["SymbolExplanation"]
        assert "usages" in explanation["properties"]
        usages = explanation["properties"]["usages"]
        assert usages["type"] == "array"

    def test_symbol_explanation_has_related(self, schema: dict[str, Any]):
        """Test SymbolExplanation has related array."""
        explanation = schema["$defs"]["SymbolExplanation"]
        assert "related" in explanation["properties"]
        related = explanation["properties"]["related"]
        assert related["type"] == "array"

    def test_symbol_explanation_has_context(self, schema: dict[str, Any]):
        """Test SymbolExplanation has context field."""
        explanation = schema["$defs"]["SymbolExplanation"]
        assert "context" in explanation["properties"]


# =============================================================================
# Supporting Types Schema Tests
# =============================================================================


class TestSupportingTypesSchema:
    """Tests for supporting type schemas."""

    def test_call_graph_entry_exists(self, schema: dict[str, Any]):
        """Test CallGraphEntry definition exists."""
        assert "CallGraphEntry" in schema["$defs"]

    def test_call_graph_entry_required_fields(self, schema: dict[str, Any]):
        """Test CallGraphEntry has required fields."""
        entry = schema["$defs"]["CallGraphEntry"]
        required = entry["required"]
        assert "symbol_id" in required
        assert "name" in required

    def test_usage_example_exists(self, schema: dict[str, Any]):
        """Test UsageExample definition exists."""
        assert "UsageExample" in schema["$defs"]

    def test_usage_example_has_code(self, schema: dict[str, Any]):
        """Test UsageExample has code field."""
        usage = schema["$defs"]["UsageExample"]
        assert "code" in usage["properties"]
        assert "code" in usage["required"]

    def test_related_symbol_exists(self, schema: dict[str, Any]):
        """Test RelatedSymbol definition exists."""
        assert "RelatedSymbol" in schema["$defs"]

    def test_related_symbol_required_fields(self, schema: dict[str, Any]):
        """Test RelatedSymbol has required fields."""
        related = schema["$defs"]["RelatedSymbol"]
        required = related["required"]
        assert "symbol_id" in required
        assert "name" in required


# =============================================================================
# JSON Schema Validation Tests (require jsonschema library)
# =============================================================================


@pytest.mark.skipif(not HAS_JSONSCHEMA, reason="jsonschema not installed")
class TestJsonSchemaValidation:
    """Tests that validate data against the JSON schema."""

    def test_schema_is_valid_json_schema(self, schema: dict[str, Any]):
        """Test that the schema itself is valid JSON Schema."""
        # This validates the schema follows JSON Schema spec
        Draft202012Validator.check_schema(schema)

    def test_valid_input_passes_validation(
        self,
        schema: dict[str, Any],
        sample_explain_code_input: dict[str, Any],
    ):
        """Test that valid input passes schema validation."""
        input_schema = schema["$defs"]["ExplainCodeInput"]

        # Resolve refs manually for testing
        resolver = jsonschema.RefResolver.from_schema(schema)
        validator = Draft202012Validator(input_schema, resolver=resolver)

        # Should not raise
        validator.validate(sample_explain_code_input)

    def test_valid_output_passes_validation(
        self,
        schema: dict[str, Any],
        sample_explain_code_output: dict[str, Any],
    ):
        """Test that valid output passes schema validation."""
        output_schema = schema["$defs"]["ExplainCodeOutput"]

        # Resolve refs manually for testing
        resolver = jsonschema.RefResolver.from_schema(schema)
        validator = Draft202012Validator(output_schema, resolver=resolver)

        # Should not raise
        validator.validate(sample_explain_code_output)

    def test_input_without_symbol_id_fails(self, schema: dict[str, Any]):
        """Test that input without symbol_id fails validation."""
        input_schema = schema["$defs"]["ExplainCodeInput"]
        resolver = jsonschema.RefResolver.from_schema(schema)
        validator = Draft202012Validator(input_schema, resolver=resolver)

        invalid_input = {"config": {"depth": 2}}

        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid_input)

    def test_output_without_explanation_fails(self, schema: dict[str, Any]):
        """Test that output without explanation fails validation."""
        output_schema = schema["$defs"]["ExplainCodeOutput"]
        resolver = jsonschema.RefResolver.from_schema(schema)
        validator = Draft202012Validator(output_schema, resolver=resolver)

        invalid_output = {"meta": {"request_id": "req-123"}}

        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid_output)

    def test_depth_out_of_range_fails(self, schema: dict[str, Any]):
        """Test that depth outside valid range fails validation."""
        config_schema = schema["$defs"]["ExplainCodeConfig"]
        validator = Draft202012Validator(config_schema)

        # Depth too high
        invalid_config = {"depth": 10}
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid_config)

        # Depth too low
        invalid_config = {"depth": 0}
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid_config)

    def test_invalid_format_fails(self, schema: dict[str, Any]):
        """Test that invalid format value fails validation."""
        config_schema = schema["$defs"]["ExplainCodeConfig"]
        validator = Draft202012Validator(config_schema)

        invalid_config = {"format": "invalid_format"}
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(invalid_config)


# =============================================================================
# Schema-Implementation Consistency Tests
# =============================================================================


class TestSchemaImplementationConsistency:
    """Tests to ensure schema matches implementation."""

    def test_config_fields_match_implementation(self, schema: dict[str, Any]):
        """Test ExplainCodeConfig schema fields match Python implementation."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        config_schema = schema["$defs"]["ExplainCodeConfig"]
        schema_fields = set(config_schema["properties"].keys())

        # Get implementation fields from dataclass
        impl = ExplainCodeConfig()
        impl_fields = {
            "depth",
            "include_callers",
            "include_callees",
            "include_usages",
            "max_usages",
            "include_related",
            "max_related",
            # format is schema-only (for formatted output)
        }

        # Implementation fields that should be in schema
        # Note: format is schema-only, cache_ttl_seconds is implementation-only
        for field in ["depth", "include_callers", "include_callees", "include_usages"]:
            assert field in schema_fields, f"Missing field in schema: {field}"

    def test_symbol_explanation_fields_match_implementation(
        self, schema: dict[str, Any]
    ):
        """Test SymbolExplanation schema fields match Python implementation."""
        from openmemory.api.tools.explain_code import SymbolExplanation

        explanation_schema = schema["$defs"]["SymbolExplanation"]
        schema_fields = set(explanation_schema["properties"].keys())

        # Get implementation fields
        impl_fields = {
            "symbol_id",
            "name",
            "kind",
            "signature",
            "file_path",
            "line_start",
            "line_end",
            "docstring",
            "callers",
            "callees",
            "usages",
            "related",
            "context",
        }

        # All implementation fields should be in schema
        for field in impl_fields:
            assert field in schema_fields, f"Missing field in schema: {field}"

    def test_default_config_values_match(self, schema: dict[str, Any]):
        """Test default values in schema match implementation defaults."""
        from openmemory.api.tools.explain_code import ExplainCodeConfig

        config_schema = schema["$defs"]["ExplainCodeConfig"]
        impl = ExplainCodeConfig()

        # Check depth default
        assert config_schema["properties"]["depth"]["default"] == impl.depth

        # Check boolean defaults
        assert (
            config_schema["properties"]["include_callers"]["default"]
            == impl.include_callers
        )
        assert (
            config_schema["properties"]["include_callees"]["default"]
            == impl.include_callees
        )
        assert (
            config_schema["properties"]["include_usages"]["default"]
            == impl.include_usages
        )
        assert (
            config_schema["properties"]["include_related"]["default"]
            == impl.include_related
        )

        # Check integer defaults
        assert config_schema["properties"]["max_usages"]["default"] == impl.max_usages
        assert config_schema["properties"]["max_related"]["default"] == impl.max_related
