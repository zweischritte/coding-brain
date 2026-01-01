"""Tests for access_entity visibility in MCP tool outputs."""

from app.utils.response_format import (
    format_add_memory_result,
    format_memory_list,
    format_memory_result,
    format_search_results,
)


class TestAccessEntityInMemoryResult:
    """Test access_entity inclusion in formatted memory results."""

    def test_format_memory_result_includes_access_entity(self):
        result = {
            "id": "test-123",
            "memory": "Test memory content",
            "metadata": {
                "category": "decision",
                "scope": "team",
                "access_entity": "team:default_org/backend",
            },
            "created_at": "2025-12-05T10:00:00+00:00",
        }

        formatted = format_memory_result(result, include_score=False)

        assert "access_entity" in formatted
        assert formatted["access_entity"] == "team:default_org/backend"

    def test_format_memory_result_handles_null_access_entity(self):
        result = {
            "id": "legacy-123",
            "memory": "Legacy memory",
            "metadata": {
                "category": "decision",
                "scope": "user",
            },
            "created_at": "2025-12-05T10:00:00+00:00",
        }

        formatted = format_memory_result(result, include_score=False)

        assert "access_entity" in formatted
        assert formatted["access_entity"] is None

    def test_format_memory_result_extracts_access_entity_from_flat(self):
        result = {
            "id": "flat-123",
            "memory": "Flat structure memory",
            "access_entity": "org:cloudfactory",
            "category": "architecture",
            "scope": "org",
            "created_at": "2025-12-05T10:00:00+00:00",
        }

        formatted = format_memory_result(result, include_score=False)

        assert formatted["access_entity"] == "org:cloudfactory"


class TestAccessEntityInSearchResults:
    """Test access_entity in search results."""

    def test_search_results_include_access_entity(self):
        results = [
            {
                "id": "mem-1",
                "memory": "First memory",
                "scores": {"final": 0.95},
                "metadata": {
                    "category": "decision",
                    "scope": "project",
                    "access_entity": "project:default_org/coding-brain",
                },
                "created_at": "2025-12-05T10:00:00+00:00",
            }
        ]

        response = format_search_results(results, verbose=False)

        assert "access_entity" in response["results"][0]
        assert response["results"][0]["access_entity"] == "project:default_org/coding-brain"


class TestVisibilityReasonInVerboseMode:
    """Test visibility_reason in verbose search results."""

    def test_visibility_reason_present_in_verbose(self):
        results = [
            {
                "id": "mem-1",
                "memory": "Memory with visibility",
                "scores": {"final": 0.9},
                "metadata": {"access_entity": "team:acme/backend"},
                "visibility_reason": {
                    "access_entity": "team:acme/backend",
                    "matched_grants": ["team:acme/backend"],
                },
                "created_at": "2025-12-05T10:00:00+00:00",
            }
        ]

        response = format_search_results(results, verbose=True)

        assert "visibility_reason" in response["results"][0]

    def test_visibility_reason_absent_in_lean(self):
        results = [
            {
                "id": "mem-1",
                "memory": "Memory without visibility reason",
                "scores": {"final": 0.9},
                "metadata": {"access_entity": "team:acme/backend"},
                "visibility_reason": {
                    "access_entity": "team:acme/backend",
                    "matched_grants": ["team:acme/backend"],
                },
                "created_at": "2025-12-05T10:00:00+00:00",
            }
        ]

        response = format_search_results(results, verbose=False)

        assert "visibility_reason" not in response["results"][0]


class TestAccessEntityInListMemories:
    """Test access_entity in list_memories output."""

    def test_list_memories_includes_access_entity(self):
        memories = [
            {
                "id": "list-1",
                "memory": "Listed memory",
                "category": "workflow",
                "scope": "team",
                "access_entity": "team:default_org/dev",
                "created_at": "2025-12-05T10:00:00+00:00",
            }
        ]

        formatted = format_memory_list(memories)

        assert len(formatted) == 1
        assert "access_entity" in formatted[0]
        assert formatted[0]["access_entity"] == "team:default_org/dev"


class TestAccessEntityInAddMemories:
    """Test access_entity in add_memories response."""

    def test_add_memory_result_includes_access_entity(self):
        result = {
            "id": "new-123",
            "memory": "Newly created memory",
        }
        structured_metadata = {
            "category": "decision",
            "scope": "project",
            "access_entity": "project:default_org/coding-brain",
        }

        formatted = format_add_memory_result(
            result=result,
            structured_metadata=structured_metadata,
        )

        assert "access_entity" in formatted
        assert formatted["access_entity"] == "project:default_org/coding-brain"
