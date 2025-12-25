"""
Tests for Structured Memory API

Tests validation, building, and update operations for the AXIS 3.4
structured memory protocol.
"""

import pytest
from app.utils.structured_memory import (
    validate_vault,
    validate_layer,
    validate_circuit,
    validate_vector,
    validate_source,
    validate_text,
    validate_tags,
    validate_evidence,
    build_structured_memory,
    validate_update_fields,
    apply_metadata_updates,
    StructuredMemoryError,
    StructuredMemoryInput,
    VALID_VAULTS,
)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidateVault:
    """Tests for vault validation."""

    def test_valid_vaults(self):
        """All valid vault codes should pass."""
        for vault in VALID_VAULTS:
            assert validate_vault(vault) == vault

    def test_case_insensitive(self):
        """Vault codes should be case-insensitive."""
        assert validate_vault("sov") == "SOV"
        assert validate_vault("Frc") == "FRC"
        assert validate_vault("wlt") == "WLT"

    def test_strips_whitespace(self):
        """Whitespace should be stripped."""
        assert validate_vault("  SOV  ") == "SOV"

    def test_invalid_vault(self):
        """Invalid vault should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_vault("XYZ")
        assert "Invalid vault 'XYZ'" in str(exc_info.value)
        assert "SOV" in str(exc_info.value)


class TestValidateLayer:
    """Tests for layer validation."""

    def test_valid_layers(self):
        """All valid layers should pass."""
        valid_layers = [
            "somatic", "emotional", "narrative", "cognitive",
            "values", "identity", "relational", "goals",
            "resources", "context", "temporal", "meta"
        ]
        for layer in valid_layers:
            assert validate_layer(layer) == layer

    def test_case_insensitive(self):
        """Layer names should be case-insensitive."""
        assert validate_layer("EMOTIONAL") == "emotional"
        assert validate_layer("Cognitive") == "cognitive"

    def test_invalid_layer(self):
        """Invalid layer should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_layer("invalid")
        assert "Invalid layer 'invalid'" in str(exc_info.value)


class TestValidateCircuit:
    """Tests for circuit validation."""

    def test_valid_circuits(self):
        """Circuits 1-8 should pass."""
        for circuit in range(1, 9):
            assert validate_circuit(circuit) == circuit

    def test_circuit_zero(self):
        """Circuit 0 should fail."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_circuit(0)
        assert "Must be integer 1-8" in str(exc_info.value)

    def test_circuit_nine(self):
        """Circuit 9 should fail."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_circuit(9)
        assert "Must be integer 1-8" in str(exc_info.value)

    def test_circuit_string(self):
        """Circuit as string should fail."""
        with pytest.raises(StructuredMemoryError):
            validate_circuit("5")  # type: ignore


class TestValidateVector:
    """Tests for vector validation."""

    def test_valid_vectors(self):
        """Valid vectors should pass."""
        assert validate_vector("say") == "say"
        assert validate_vector("want") == "want"
        assert validate_vector("do") == "do"

    def test_case_insensitive(self):
        """Vector should be case-insensitive."""
        assert validate_vector("SAY") == "say"
        assert validate_vector("Want") == "want"

    def test_invalid_vector(self):
        """Invalid vector should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_vector("think")
        assert "Invalid vector 'think'" in str(exc_info.value)


class TestValidateSource:
    """Tests for source validation."""

    def test_valid_sources(self):
        """Valid sources should pass."""
        assert validate_source("user") == "user"
        assert validate_source("inference") == "inference"

    def test_case_insensitive(self):
        """Source should be case-insensitive."""
        assert validate_source("USER") == "user"
        assert validate_source("Inference") == "inference"

    def test_invalid_source(self):
        """Invalid source should raise error."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_source("system")
        assert "Invalid source 'system'" in str(exc_info.value)


class TestValidateText:
    """Tests for text validation."""

    def test_valid_text(self):
        """Non-empty text should pass."""
        assert validate_text("Hello world") == "Hello world"

    def test_strips_whitespace(self):
        """Whitespace should be stripped."""
        assert validate_text("  Hello  ") == "Hello"

    def test_empty_text(self):
        """Empty text should fail."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_text("")
        assert "cannot be empty" in str(exc_info.value)

    def test_whitespace_only(self):
        """Whitespace-only text should fail."""
        with pytest.raises(StructuredMemoryError):
            validate_text("   ")


class TestValidateTags:
    """Tests for tags validation."""

    def test_valid_tags(self):
        """Valid tags should pass."""
        tags = {"trigger": True, "intensity": 7, "note": "test"}
        assert validate_tags(tags) == tags

    def test_empty_tags(self):
        """Empty tags dict should pass."""
        assert validate_tags({}) == {}

    def test_non_dict_tags(self):
        """Non-dict should fail."""
        with pytest.raises(StructuredMemoryError):
            validate_tags("invalid")  # type: ignore

    def test_non_string_key(self):
        """Non-string key should fail."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_tags({1: "value"})  # type: ignore
        assert "must be string" in str(exc_info.value)


class TestValidateEvidence:
    """Tests for evidence validation."""

    def test_valid_evidence(self):
        """Valid evidence list should pass."""
        evidence = ["project-a", "project-b"]
        assert validate_evidence(evidence) == evidence

    def test_empty_evidence(self):
        """Empty list should pass."""
        assert validate_evidence([]) == []

    def test_non_list(self):
        """Non-list should fail."""
        with pytest.raises(StructuredMemoryError):
            validate_evidence("invalid")  # type: ignore

    def test_non_string_item(self):
        """Non-string item should fail."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_evidence(["valid", 123])  # type: ignore
        assert "must be string" in str(exc_info.value)


# =============================================================================
# BUILD STRUCTURED MEMORY TESTS
# =============================================================================

class TestBuildStructuredMemory:
    """Tests for build_structured_memory function."""

    def test_minimal_required_fields(self):
        """Minimal required fields should work."""
        text, metadata = build_structured_memory(
            text="Test content",
            vault="SOV",
            layer="cognitive"
        )
        assert text == "Test content"
        assert metadata["vault"] == "SOVEREIGNTY_CORE"
        assert metadata["layer"] == "cognitive"
        assert metadata["axis_category"] == "identity"
        assert metadata["source"] == "axis_protocol"
        assert metadata["src"] == "user"  # default

    def test_all_fields(self):
        """All fields should be included in metadata."""
        text, metadata = build_structured_memory(
            text="Kritik triggert Wut",
            vault="FRC",
            layer="emotional",
            circuit=2,
            vector="say",
            entity="BMG",
            source="user",
            was="neutral",
            origin="session-123",
            evidence=["projekt-a", "projekt-b"],
            tags={"trigger": True, "intensity": 7}
        )

        assert text == "Kritik triggert Wut"
        assert metadata["vault"] == "FRACTURE_LOG"
        assert metadata["layer"] == "emotional"
        assert metadata["circuit"] == 2
        assert metadata["vector"] == "say"
        assert metadata["re"] == "BMG"
        assert metadata["src"] == "user"
        assert metadata["was"] == "neutral"
        assert metadata["from"] == "session-123"
        assert metadata["ev"] == ["projekt-a", "projekt-b"]
        assert metadata["tags"] == {"trigger": True, "intensity": 7}
        assert metadata["axis_category"] == "health"  # FRC -> health

    def test_inference_source(self):
        """Inference source should be stored correctly."""
        _, metadata = build_structured_memory(
            text="AI observation",
            vault="FGP",
            layer="meta",
            source="inference"
        )
        assert metadata["src"] == "inference"

    def test_validation_error_propagates(self):
        """Validation errors should propagate."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            build_structured_memory(
                text="Test",
                vault="INVALID",
                layer="cognitive"
            )
        assert "Invalid vault" in str(exc_info.value)


class TestStructuredMemoryInput:
    """Tests for StructuredMemoryInput dataclass."""

    def test_computed_fields(self):
        """Computed fields should be set correctly."""
        memory = StructuredMemoryInput(
            text="Test",
            vault="WLT",
            layer="goals"
        )
        assert memory.vault_full == "WEALTH_MATRIX"
        assert memory.axis_category == "business"

    def test_to_metadata_dict(self):
        """to_metadata_dict should produce correct output."""
        memory = StructuredMemoryInput(
            text="Test",
            vault="SIG",
            layer="cognitive",
            circuit=5,
            vector="want"
        )
        metadata = memory.to_metadata_dict()

        assert metadata["vault"] == "SIGNAL_LIBRARY"
        assert metadata["layer"] == "cognitive"
        assert metadata["circuit"] == 5
        assert metadata["vector"] == "want"
        assert "re" not in metadata  # entity not provided


# =============================================================================
# UPDATE FIELDS TESTS
# =============================================================================

class TestValidateUpdateFields:
    """Tests for validate_update_fields function."""

    def test_empty_update(self):
        """No fields should produce empty dict."""
        result = validate_update_fields()
        assert result == {}

    def test_partial_update(self):
        """Only provided fields should be validated."""
        result = validate_update_fields(
            vault="FRC",
            layer="emotional"
        )
        assert result["vault"] == "FRC"
        assert result["vault_full"] == "FRACTURE_LOG"
        assert result["layer"] == "emotional"
        assert "text" not in result
        assert "circuit" not in result

    def test_text_update(self):
        """Text update should be validated."""
        result = validate_update_fields(text="New content")
        assert result["text"] == "New content"

    def test_validation_error(self):
        """Invalid field should raise error."""
        with pytest.raises(StructuredMemoryError):
            validate_update_fields(circuit=10)


class TestApplyMetadataUpdates:
    """Tests for apply_metadata_updates function."""

    def test_apply_field_updates(self):
        """Field updates should be applied."""
        current = {"vault": "OLD", "layer": "narrative", "tags": {}}
        validated = {"vault_full": "SOVEREIGNTY_CORE", "layer": "cognitive"}

        result = apply_metadata_updates(current, validated)

        assert result["vault"] == "SOVEREIGNTY_CORE"
        assert result["layer"] == "cognitive"

    def test_add_tags(self):
        """Tags should be added."""
        current = {"tags": {"existing": True}}
        result = apply_metadata_updates(
            current,
            {},
            add_tags={"new": "value"}
        )
        assert result["tags"] == {"existing": True, "new": "value"}

    def test_remove_tags(self):
        """Tags should be removed."""
        current = {"tags": {"keep": True, "remove": True}}
        result = apply_metadata_updates(
            current,
            {},
            remove_tags=["remove"]
        )
        assert result["tags"] == {"keep": True}

    def test_add_and_remove_tags(self):
        """Adding and removing tags simultaneously."""
        current = {"tags": {"old": True}}
        result = apply_metadata_updates(
            current,
            {},
            add_tags={"new": True},
            remove_tags=["old"]
        )
        assert result["tags"] == {"new": True}

    def test_preserves_other_metadata(self):
        """Other metadata fields should be preserved."""
        current = {"vault": "OLD", "custom_field": "value"}
        result = apply_metadata_updates(current, {})
        assert result["custom_field"] == "value"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestStructuredMemoryIntegration:
    """Integration tests for the full structured memory workflow."""

    def test_add_memory_flow(self):
        """Full add memory workflow."""
        text, metadata = build_structured_memory(
            text="Kritik triggert Wut",
            vault="FRC",
            layer="emotional",
            circuit=2,
            vector="say",
            entity="BMG",
            tags={"trigger": True, "intensity": 7}
        )

        # Verify clean text
        assert text == "Kritik triggert Wut"

        # Verify all metadata
        assert metadata["vault"] == "FRACTURE_LOG"
        assert metadata["layer"] == "emotional"
        assert metadata["circuit"] == 2
        assert metadata["vector"] == "say"
        assert metadata["re"] == "BMG"
        assert metadata["tags"]["trigger"] is True
        assert metadata["tags"]["intensity"] == 7

    def test_update_memory_flow(self):
        """Full update memory workflow."""
        # Initial metadata (as would be stored)
        current = {
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
            "circuit": 2,
            "tags": {"trigger": True, "intensity": 7}
        }

        # Validate update fields
        validated = validate_update_fields(
            text="Kritik triggert jetzt Neugier",
            vault="SOV",
            layer="identity"
        )

        # Apply updates with new tags
        updated = apply_metadata_updates(
            current,
            validated,
            add_tags={"evolved": True}
        )

        # Verify updates
        assert validated["text"] == "Kritik triggert jetzt Neugier"
        assert updated["vault"] == "SOVEREIGNTY_CORE"
        assert updated["layer"] == "identity"
        assert updated["tags"]["evolved"] is True
        assert updated["tags"]["trigger"] is True  # Preserved
        assert updated["circuit"] == 2  # Preserved

    def test_maintenance_mode_fields(self):
        """Maintenance mode should not modify validated fields."""
        # Validation doesn't care about preserve_timestamps
        # That's handled at the MCP level
        validated = validate_update_fields(
            vault="SOV",
            layer="identity"
        )
        assert "vault" in validated
        assert "layer" in validated


# =============================================================================
# ERROR MESSAGE TESTS
# =============================================================================

class TestErrorMessages:
    """Tests for error message quality."""

    def test_vault_error_shows_valid_options(self):
        """Vault error should list valid options."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_vault("INVALID")
        error = str(exc_info.value)
        assert "SOV" in error
        assert "WLT" in error
        assert "FRC" in error

    def test_layer_error_shows_valid_options(self):
        """Layer error should list valid options."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_layer("invalid")
        error = str(exc_info.value)
        assert "somatic" in error
        assert "emotional" in error

    def test_circuit_error_shows_range(self):
        """Circuit error should show valid range."""
        with pytest.raises(StructuredMemoryError) as exc_info:
            validate_circuit(10)
        assert "1-8" in str(exc_info.value)
