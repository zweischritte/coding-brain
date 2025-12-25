"""
Tests for Lean Response Formatting Module

Tests the response_format.py utilities for:
- Timestamp formatting to Europe/Berlin timezone
- Vault short code conversion
- Memory result formatting
- Search result formatting
- Memory list formatting
- Conditional field inclusion
"""

import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.utils.response_format import (
    format_timestamp,
    get_vault_short,
    format_memory_result,
    format_search_results,
    format_memory_list,
    format_add_memory_result,
    format_add_memories_response,
    VAULT_SHORT_CODES,
    BERLIN_TZ,
)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_converts_utc_to_berlin(self):
        """UTC timestamp should be converted to Berlin time."""
        # UTC winter time (+1 hour)
        utc_str = "2025-12-05T09:30:00+00:00"
        result = format_timestamp(utc_str)
        assert result == "2025-12-05T10:30:00+01:00"

    def test_converts_pacific_to_berlin(self):
        """Pacific timezone (-8 hours) should be converted to Berlin."""
        # Pacific winter time
        pacific_str = "2025-12-05T01:39:18.187996-08:00"
        result = format_timestamp(pacific_str)
        assert result == "2025-12-05T10:39:18+01:00"

    def test_removes_microseconds(self):
        """Microseconds should be removed from output."""
        utc_str = "2025-12-05T09:30:00.123456+00:00"
        result = format_timestamp(utc_str)
        assert ".123456" not in result
        assert result == "2025-12-05T10:30:00+01:00"

    def test_handles_z_suffix(self):
        """Z suffix (common in JSON) should be handled."""
        z_str = "2025-12-05T09:30:00Z"
        result = format_timestamp(z_str)
        assert result == "2025-12-05T10:30:00+01:00"

    def test_handles_none(self):
        """None input should return None."""
        assert format_timestamp(None) is None

    def test_handles_empty_string(self):
        """Empty string should return None."""
        assert format_timestamp("") is None

    def test_handles_invalid_format(self):
        """Invalid format should return None (not raise)."""
        assert format_timestamp("not-a-date") is None
        assert format_timestamp("2025-13-45") is None

    def test_summer_time(self):
        """Berlin summer time (+2 hours) should be correct."""
        # June is summer time in Berlin
        utc_str = "2025-06-15T08:00:00+00:00"
        result = format_timestamp(utc_str)
        assert result == "2025-06-15T10:00:00+02:00"


class TestGetVaultShort:
    """Tests for get_vault_short function."""

    def test_all_vault_codes(self):
        """All vault full names should map to short codes."""
        assert get_vault_short("SOVEREIGNTY_CORE") == "SOV"
        assert get_vault_short("WEALTH_MATRIX") == "WLT"
        assert get_vault_short("SIGNAL_LIBRARY") == "SIG"
        assert get_vault_short("FRACTURE_LOG") == "FRC"
        assert get_vault_short("SOURCE_DIRECTIVES") == "DIR"
        assert get_vault_short("FINGERPRINT") == "FGP"
        assert get_vault_short("QUESTIONS_QUEUE") == "Q"

    def test_returns_original_if_unknown(self):
        """Unknown vault names should be returned as-is."""
        assert get_vault_short("UNKNOWN_VAULT") == "UNKNOWN_VAULT"
        assert get_vault_short("custom") == "custom"

    def test_handles_none(self):
        """None input should return None."""
        assert get_vault_short(None) is None

    def test_handles_empty_string(self):
        """Empty string should return None."""
        assert get_vault_short("") is None


class TestFormatMemoryResult:
    """Tests for format_memory_result function."""

    def test_basic_fields(self):
        """Basic required fields should be present."""
        result = {
            "id": "af9ad6a0-3518-4678-b9e7-f52d5f8d68da",
            "memory": "Test memory content",
            "scores": {"semantic": 0.6, "boost": 0.3, "final": 0.78},
            "metadata": {"vault": "FRACTURE_LOG", "layer": "emotional"},
            "created_at": "2025-12-05T09:30:00+00:00",
        }
        formatted = format_memory_result(result, include_score=True)

        assert formatted["id"] == "af9ad6a0-3518-4678-b9e7-f52d5f8d68da"
        assert formatted["memory"] == "Test memory content"
        assert formatted["score"] == 0.78  # Rounded to 2 decimals
        assert formatted["vault"] == "FRC"  # Short code
        assert formatted["layer"] == "emotional"
        assert formatted["created_at"] == "2025-12-05T10:30:00+01:00"

    def test_score_rounding(self):
        """Score should be rounded to 2 decimal places."""
        result = {
            "id": "test-id",
            "memory": "content",
            "scores": {"final": 1.2741123456},
            "created_at": "2025-12-05T09:30:00+00:00",
        }
        formatted = format_memory_result(result, include_score=True)
        assert formatted["score"] == 1.27

    def test_no_score_for_list(self):
        """Score should not be included when include_score=False."""
        result = {
            "id": "test-id",
            "memory": "content",
            "scores": {"final": 0.78},
            "created_at": "2025-12-05T09:30:00+00:00",
        }
        formatted = format_memory_result(result, include_score=False)
        assert "score" not in formatted

    def test_conditional_circuit(self):
        """Circuit should only be included if present."""
        result_with = {"id": "1", "memory": "x", "metadata": {"circuit": 2}}
        result_without = {"id": "2", "memory": "y", "metadata": {}}

        assert format_memory_result(result_with)["circuit"] == 2
        assert "circuit" not in format_memory_result(result_without)

    def test_conditional_vector(self):
        """Vector should only be included if present."""
        result_with = {"id": "1", "memory": "x", "metadata": {"vector": "say"}}
        result_without = {"id": "2", "memory": "y", "metadata": {}}

        assert format_memory_result(result_with)["vector"] == "say"
        assert "vector" not in format_memory_result(result_without)

    def test_conditional_entity(self):
        """Entity should only be included if present."""
        result_with = {"id": "1", "memory": "x", "metadata": {"re": "Arbeitsmuster"}}
        result_without = {"id": "2", "memory": "y", "metadata": {}}

        assert format_memory_result(result_with)["entity"] == "Arbeitsmuster"
        assert "entity" not in format_memory_result(result_without)

    def test_conditional_tags(self):
        """Tags should only be included if present and non-empty."""
        result_with = {"id": "1", "memory": "x", "metadata": {"tags": {"trigger": True}}}
        result_empty = {"id": "2", "memory": "y", "metadata": {"tags": {}}}
        result_without = {"id": "3", "memory": "z", "metadata": {}}

        assert format_memory_result(result_with)["tags"] == {"trigger": True}
        assert "tags" not in format_memory_result(result_empty)
        assert "tags" not in format_memory_result(result_without)

    def test_conditional_updated_at(self):
        """updated_at should only be included if present."""
        result_with = {
            "id": "1", "memory": "x",
            "created_at": "2025-12-05T09:00:00+00:00",
            "updated_at": "2025-12-05T10:00:00+00:00",
        }
        result_without = {
            "id": "2", "memory": "y",
            "created_at": "2025-12-05T09:00:00+00:00",
            "updated_at": None,
        }

        assert "updated_at" in format_memory_result(result_with)
        assert "updated_at" not in format_memory_result(result_without)

    def test_flat_metadata_fallback(self):
        """Fields should be extracted from top-level if not in metadata."""
        result = {
            "id": "1",
            "memory": "x",
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
            "circuit": 3,
        }
        formatted = format_memory_result(result)
        assert formatted["vault"] == "FRC"
        assert formatted["layer"] == "emotional"
        assert formatted["circuit"] == 3


class TestFormatSearchResults:
    """Tests for format_search_results function."""

    def test_lean_format_default(self):
        """Default format should be lean (no debug info)."""
        results = [
            {
                "id": "1",
                "memory": "Memory one",
                "scores": {"semantic": 0.6, "boost": 0.3, "final": 0.78},
                "metadata": {"vault": "FRACTURE_LOG"},
                "created_at": "2025-12-05T09:30:00+00:00",
            }
        ]
        response = format_search_results(results, verbose=False)

        # Should only have results key
        assert "results" in response
        assert "query" not in response
        assert "context_applied" not in response
        assert "filters_applied" not in response
        assert "total_candidates" not in response
        assert "returned" not in response

        # Results should be lean formatted
        assert response["results"][0]["vault"] == "FRC"
        assert response["results"][0]["score"] == 0.78

    def test_verbose_format(self):
        """Verbose format should include all debug info."""
        results = [{"id": "1", "memory": "x", "scores": {"final": 0.5}}]
        response = format_search_results(
            results,
            verbose=True,
            query="test query",
            context_applied={"entity": "Test"},
            filters_applied={"created_after": "2025-01-01"},
            total_candidates=100,
        )

        assert response["query"] == "test query"
        assert response["context_applied"] == {"entity": "Test"}
        assert response["filters_applied"] == {"created_after": "2025-01-01"}
        assert response["total_candidates"] == 100
        assert response["returned"] == 1

    def test_empty_results(self):
        """Empty results should work correctly."""
        response = format_search_results([], verbose=False)
        assert response == {"results": []}


class TestFormatMemoryList:
    """Tests for format_memory_list function."""

    def test_formats_list_correctly(self):
        """List format should not include scores."""
        memories = [
            {
                "id": "1",
                "memory": "Memory one",
                "vault": "FRACTURE_LOG",
                "layer": "emotional",
                "created_at": "2025-12-05T09:30:00+00:00",
            },
            {
                "id": "2",
                "memory": "Memory two",
                "vault": "SOVEREIGNTY_CORE",
                "layer": "identity",
                "created_at": "2025-12-05T10:30:00+00:00",
            },
        ]
        formatted = format_memory_list(memories)

        assert len(formatted) == 2
        assert formatted[0]["vault"] == "FRC"
        assert formatted[1]["vault"] == "SOV"
        assert "score" not in formatted[0]
        assert "score" not in formatted[1]

    def test_includes_axis_fields(self):
        """AXIS fields should be included when present."""
        memories = [
            {
                "id": "1",
                "memory": "x",
                "vault": "FRACTURE_LOG",
                "layer": "emotional",
                "circuit": 2,
                "vector": "say",
                "entity": "Arbeitsmuster",
                "tags": {"trigger": True, "intensity": 7},
                "created_at": "2025-12-05T09:30:00+00:00",
            }
        ]
        formatted = format_memory_list(memories)

        assert formatted[0]["circuit"] == 2
        assert formatted[0]["vector"] == "say"
        assert formatted[0]["entity"] == "Arbeitsmuster"
        assert formatted[0]["tags"] == {"trigger": True, "intensity": 7}

    def test_empty_list(self):
        """Empty list should work correctly."""
        assert format_memory_list([]) == []


class TestTokenReduction:
    """Tests to verify token reduction is achieved."""

    def test_lean_format_smaller_than_verbose(self):
        """Lean format should produce smaller output than verbose."""
        import json

        results = [
            {
                "id": "af9ad6a0-3518-4678-b9e7-f52d5f8d68da",
                "memory": "Frustration bei technischen Blockaden triggert Perfektionismus",
                "scores": {"semantic": 0.6067, "boost": 1.1, "final": 1.2741},
                "metadata": {
                    "vault": "FRACTURE_LOG",
                    "layer": "emotional",
                    "vector": "say",
                    "circuit": 2,
                    "tags": {"trigger": True, "intensity": 6},
                    "re": "Arbeitsmuster",
                },
                "created_at": "2025-12-05T01:39:18.187996-08:00",
                "updated_at": None,
            }
        ]

        lean_response = format_search_results(results, verbose=False)
        verbose_response = format_search_results(
            results,
            verbose=True,
            query="Frustration Trigger Arbeit",
            context_applied={"entity": "Arbeitsmuster", "layer": "emotional", "vault": "FRACTURE_LOG", "circuit": 2},
            filters_applied=None,
            total_candidates=30,
        )

        lean_json = json.dumps(lean_response, default=str)
        verbose_json = json.dumps(verbose_response, default=str)

        # Lean should be significantly smaller
        assert len(lean_json) < len(verbose_json)

        # Estimate token reduction (rough: 4 chars per token)
        lean_tokens = len(lean_json) / 4
        verbose_tokens = len(verbose_json) / 4
        reduction = (verbose_tokens - lean_tokens) / verbose_tokens

        # Should achieve at least 20% reduction
        assert reduction >= 0.20, f"Expected at least 20% reduction, got {reduction:.1%}"


class TestFormatAddMemoryResult:
    """Tests for format_add_memory_result function."""

    def test_basic_fields(self):
        """Basic required fields should be present."""
        result = {
            "id": "d615437f-390d-42cb-8975-cdb324d14671",
            "memory": "Kritik triggert Schutzreaktion",
            "event": "ADD",
        }
        axis_metadata = {
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
            "circuit": 2,
            "vector": "say",
            "re": "Test-Entity",
            "tags": {"trigger": True, "intensity": 8, "shadow": True},
            "src": "user",
            "from": "Kindheit-Dynamik",
        }

        formatted = format_add_memory_result(result, axis_metadata)

        assert formatted["id"] == "d615437f-390d-42cb-8975-cdb324d14671"
        assert formatted["memory"] == "Kritik triggert Schutzreaktion"
        assert formatted["vault"] == "FRC"  # Short code
        assert formatted["layer"] == "emotional"
        assert formatted["circuit"] == 2
        assert formatted["vector"] == "say"
        assert formatted["entity"] == "Test-Entity"  # From 're'
        assert formatted["tags"] == {"trigger": True, "intensity": 8, "shadow": True}
        assert formatted["meta"] == {"src": "user", "from": "Kindheit-Dynamik"}
        assert "created_at" in formatted  # Always present

    def test_vault_short_code_conversion(self):
        """Full vault names should be converted to short codes."""
        result = {"id": "1", "memory": "x"}

        # Test all vault codes
        vault_mappings = {
            "FRACTURE_LOG": "FRC",
            "SOVEREIGNTY_CORE": "SOV",
            "WEALTH_MATRIX": "WLT",
            "SIGNAL_LIBRARY": "SIG",
            "SOURCE_DIRECTIVES": "DIR",
            "FINGERPRINT": "FGP",
            "QUESTIONS_QUEUE": "Q",
        }

        for full_name, short_code in vault_mappings.items():
            formatted = format_add_memory_result(result, {"vault": full_name})
            assert formatted["vault"] == short_code

    def test_entity_from_re(self):
        """Entity field should come from 're' in axis_metadata."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {"re": "BMG"}

        formatted = format_add_memory_result(result, axis_metadata)
        assert formatted["entity"] == "BMG"

    def test_meta_object_construction(self):
        """Meta object should contain src, from, was, ev when present."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {
            "src": "inference",
            "from": "Kindheit",
            "was": "Previous-Version",
            "ev": "event-a,event-b",
        }

        formatted = format_add_memory_result(result, axis_metadata)
        assert formatted["meta"] == {
            "src": "inference",
            "from": "Kindheit",
            "was": "Previous-Version",
            "ev": "event-a,event-b",
        }

    def test_meta_object_partial(self):
        """Meta object should only include present fields."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {"src": "user"}  # Only src

        formatted = format_add_memory_result(result, axis_metadata)
        assert formatted["meta"] == {"src": "user"}

    def test_no_meta_when_empty(self):
        """Meta object should not be included if no inline metadata."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {"vault": "FRACTURE_LOG"}  # No inline metadata

        formatted = format_add_memory_result(result, axis_metadata)
        assert "meta" not in formatted

    def test_created_at_always_present(self):
        """created_at should always be present (confirms storage)."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {}

        formatted = format_add_memory_result(result, axis_metadata)
        assert "created_at" in formatted
        assert formatted["created_at"] is not None

    def test_conditional_fields_omitted_when_none(self):
        """Optional fields should be omitted when not present."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {}  # No axis metadata

        formatted = format_add_memory_result(result, axis_metadata)

        assert "vault" not in formatted
        assert "layer" not in formatted
        assert "circuit" not in formatted
        assert "vector" not in formatted
        assert "entity" not in formatted
        assert "tags" not in formatted
        assert "meta" not in formatted

    def test_empty_tags_omitted(self):
        """Empty tags dict should be omitted."""
        result = {"id": "1", "memory": "x"}
        axis_metadata = {"tags": {}}

        formatted = format_add_memory_result(result, axis_metadata)
        assert "tags" not in formatted


class TestFormatAddMemoriesResponse:
    """Tests for format_add_memories_response function."""

    def test_single_result_returns_flat_object(self):
        """Single result should return flat object, not wrapped in results array."""
        results = [
            {"id": "d615437f-390d-42cb-8975-cdb324d14671", "memory": "Test memory", "event": "ADD"}
        ]
        axis_metadata = {
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
        }

        response = format_add_memories_response(results, axis_metadata)

        # Should be flat, not wrapped
        assert "results" not in response
        assert "summary" not in response
        assert response["id"] == "d615437f-390d-42cb-8975-cdb324d14671"
        assert response["memory"] == "Test memory"
        assert response["vault"] == "FRC"

    def test_batch_results_returns_array_with_summary(self):
        """Multiple results should return array with summary."""
        results = [
            {"id": "1", "memory": "Memory 1", "event": "ADD"},
            {"id": "2", "memory": "Memory 2", "event": "ADD"},
        ]
        axis_metadata = {"vault": "FRACTURE_LOG"}

        response = format_add_memories_response(results, axis_metadata)

        assert "results" in response
        assert "summary" in response
        assert len(response["results"]) == 2
        assert response["summary"]["total"] == 2
        assert response["summary"]["success"] == 2
        assert response["summary"]["failed"] == 0

    def test_empty_results_returns_parsed_metadata(self):
        """Empty results (Q-vault bug) should still return parsed metadata."""
        results = []
        axis_metadata = {
            "vault": "QUESTIONS_QUEUE",
            "layer": "meta",
            "circuit": 7,
        }

        response = format_add_memories_response(results, axis_metadata)

        # Should still show parsing worked
        assert response["id"] is None
        assert response["memory"] is None
        assert response["vault"] == "Q"
        assert response["layer"] == "meta"
        assert response["circuit"] == 7
        assert "created_at" in response

    def test_no_redundancy_with_axis_metadata(self):
        """Response should NOT contain separate axis_metadata field."""
        results = [{"id": "1", "memory": "x", "event": "ADD"}]
        axis_metadata = {
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
            "tags": {"trigger": True},
        }

        response = format_add_memories_response(results, axis_metadata)

        # Old format had axis_metadata - new format should NOT
        assert "axis_metadata" not in response

    def test_matches_list_memories_structure(self):
        """Response structure should match list_memories output."""
        results = [
            {
                "id": "d615437f-390d-42cb-8975-cdb324d14671",
                "memory": "Test memory",
                "event": "ADD",
            }
        ]
        axis_metadata = {
            "vault": "FRACTURE_LOG",
            "layer": "emotional",
            "circuit": 2,
            "vector": "say",
            "re": "Test-Entity",
            "tags": {"trigger": True},
        }

        response = format_add_memories_response(results, axis_metadata)

        # Expected fields from list_memories format
        expected_fields = {"id", "memory", "vault", "layer", "circuit", "vector", "entity", "tags", "created_at"}
        actual_fields = set(response.keys())

        # All expected fields should be present
        for field in expected_fields:
            assert field in actual_fields, f"Missing expected field: {field}"

    def test_batch_with_partial_success(self):
        """Batch with some failures should report correctly."""
        results = [
            {"id": "1", "memory": "Success 1", "event": "ADD"},
            {"id": None, "memory": None, "event": "ADD"},  # Failed
            {"id": "3", "memory": "Success 2", "event": "ADD"},
        ]
        axis_metadata = {"vault": "FRACTURE_LOG"}

        response = format_add_memories_response(results, axis_metadata)

        assert response["summary"]["total"] == 3
        assert response["summary"]["success"] == 2
        assert response["summary"]["failed"] == 1
