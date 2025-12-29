"""Tests for response formatting utilities."""

from app.utils.response_format import (
    format_add_memories_response,
    format_add_memory_result,
    format_memory_result,
    format_search_results,
    format_timestamp,
)


def test_format_timestamp_handles_none():
    assert format_timestamp(None) is None


def test_format_timestamp_converts_to_berlin():
    assert format_timestamp("2025-12-05T01:39:18.187996-08:00") == "2025-12-05T10:39:18+01:00"


def test_format_memory_result_includes_structured_metadata():
    result = {
        "id": "1",
        "memory": "Use Alembic for migrations",
        "scores": {"final": 0.81234},
        "metadata": {
            "category": "architecture",
            "scope": "project",
            "artifact_type": "db",
            "artifact_ref": "migrations/2025_01_01.sql",
            "entity": "billing-service",
            "source": "user",
            "evidence": ["ADR-012"],
            "tags": {"migration": True},
        },
        "created_at": "2025-12-05T01:39:18.187996-08:00",
        "updated_at": "2025-12-06T01:39:18.187996-08:00",
    }

    formatted = format_memory_result(result, include_score=True)

    assert formatted["category"] == "architecture"
    assert formatted["scope"] == "project"
    assert formatted["artifact_type"] == "db"
    assert formatted["artifact_ref"] == "migrations/2025_01_01.sql"
    assert formatted["entity"] == "billing-service"
    assert formatted["source"] == "user"
    assert formatted["evidence"] == ["ADR-012"]
    assert formatted["tags"] == {"migration": True}
    assert formatted["score"] == 0.81
    assert formatted["created_at"] == "2025-12-05T10:39:18+01:00"
    assert formatted["updated_at"] == "2025-12-06T10:39:18+01:00"


def test_format_memory_result_converts_tag_list():
    result = {
        "id": "1",
        "memory": "Tags list",
        "metadata": {"tags": ["one", "two"]},
    }
    formatted = format_memory_result(result, include_score=False)
    assert formatted["tags"] == {"one": True, "two": True}


def test_format_search_results_verbose():
    results = [{"id": "1", "memory": "A", "scores": {"final": 0.5}}]
    response = format_search_results(
        results=results,
        verbose=True,
        query="test",
        context_applied={"category": "testing"},
        filters_applied={"exclude_tags": ["skip"]},
        total_candidates=1,
    )
    assert response["results"] == results
    assert response["query"] == "test"
    assert response["total_candidates"] == 1


def test_format_search_results_lean():
    results = [{"id": "1", "memory": "A", "scores": {"final": 0.5}}]
    response = format_search_results(results=results, verbose=False)
    assert response["results"][0]["id"] == "1"
    assert response["results"][0]["score"] == 0.5


def test_format_add_memory_result():
    result = {"id": "mem-1", "memory": "Add memory"}
    structured_metadata = {
        "category": "decision",
        "scope": "team",
        "artifact_type": "service",
        "artifact_ref": "auth-service",
        "entity": "auth-team",
        "source": "user",
        "evidence": ["PR-456"],
        "tags": {"auth": True},
    }

    formatted = format_add_memory_result(result, structured_metadata)
    assert formatted["category"] == "decision"
    assert formatted["scope"] == "team"
    assert formatted["artifact_type"] == "service"
    assert formatted["artifact_ref"] == "auth-service"
    assert formatted["entity"] == "auth-team"
    assert formatted["source"] == "user"
    assert formatted["evidence"] == ["PR-456"]
    assert formatted["tags"] == {"auth": True}
    assert "created_at" in formatted


def test_format_add_memories_response_single():
    results = [{"id": "mem-1", "memory": "Single"}]
    structured_metadata = {"category": "workflow", "scope": "project"}
    response = format_add_memories_response(results, structured_metadata)
    assert response["id"] == "mem-1"
    assert response["category"] == "workflow"


def test_format_add_memories_response_batch():
    results = [{"id": "mem-1", "memory": "One"}, {"id": "mem-2", "memory": "Two"}]
    structured_metadata = {"category": "workflow", "scope": "project"}
    response = format_add_memories_response(results, structured_metadata)
    assert response["summary"]["total"] == 2
    assert len(response["results"]) == 2


def test_format_add_memories_response_empty():
    structured_metadata = {"category": "glossary", "scope": "org"}
    response = format_add_memories_response([], structured_metadata)
    assert response["category"] == "glossary"
    assert response["scope"] == "org"
