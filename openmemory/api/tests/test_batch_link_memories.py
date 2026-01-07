"""
Tests for Batch Link Memories script.

Tests the nightly batch job for automatically linking memories to code
via entity name matching.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.scripts.batch_link_memories import (
    get_memories_without_code_refs,
    batch_link_memories,
)
from app.graph.evidence_linker import CodeLink


# =============================================================================
# Mock Memory Model
# =============================================================================


class MockMemory:
    """Mock SQLAlchemy Memory model."""

    def __init__(self, id, memory, metadata_, state="active"):
        self.id = id
        self.memory = memory
        self.metadata_ = metadata_
        self.state = state


# =============================================================================
# get_memories_without_code_refs Tests
# =============================================================================


class TestGetMemoriesWithoutCodeRefs:
    """Tests for get_memories_without_code_refs function."""

    @pytest.mark.asyncio
    async def test_finds_memories_with_entity_but_no_code_refs(self):
        """Test finding memories that have entity but no code_refs."""
        mock_memories = [
            MockMemory(
                id="mem_1",
                memory="AuthService handles authentication",
                metadata_={"entity": "AuthService"},
            ),
            MockMemory(
                id="mem_2",
                memory="PaymentService processes payments",
                metadata_={"entity": "PaymentService"},
            ),
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_memories
        mock_query.__iter__ = lambda self: iter(mock_memories)

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        result = await get_memories_without_code_refs(mock_db, limit=100)

        assert len(result) == 2
        assert result[0]["metadata"]["entity"] == "AuthService"
        assert result[1]["metadata"]["entity"] == "PaymentService"

    @pytest.mark.asyncio
    async def test_excludes_memories_with_code_refs(self):
        """Test that memories with existing code_refs are excluded."""
        mock_memories = [
            MockMemory(
                id="mem_1",
                memory="Already linked",
                metadata_={"entity": "Service", "code_refs": [{"file_path": "/src/s.ts"}]},
            ),
            MockMemory(
                id="mem_2",
                memory="Not linked",
                metadata_={"entity": "OtherService"},
            ),
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_memories
        mock_query.__iter__ = lambda self: iter(mock_memories)

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        result = await get_memories_without_code_refs(mock_db, limit=100)

        # Only the one without code_refs should be included
        assert len(result) == 1
        assert result[0]["id"] == "mem_2"

    @pytest.mark.asyncio
    async def test_excludes_memories_without_entity(self):
        """Test that memories without entity are excluded."""
        mock_memories = [
            MockMemory(
                id="mem_1",
                memory="No entity here",
                metadata_={},
            ),
            MockMemory(
                id="mem_2",
                memory="Has entity",
                metadata_={"entity": "SomeService"},
            ),
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_memories
        mock_query.__iter__ = lambda self: iter(mock_memories)

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        result = await get_memories_without_code_refs(mock_db, limit=100)

        # Only the one with entity should be included
        assert len(result) == 1
        assert result[0]["metadata"]["entity"] == "SomeService"

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Test that limit parameter is respected."""
        mock_memories = [
            MockMemory(
                id=f"mem_{i}",
                memory=f"Memory {i}",
                metadata_={"entity": f"Service{i}"},
            )
            for i in range(10)
        ]

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = mock_memories
        mock_query.__iter__ = lambda self: iter(mock_memories)

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        result = await get_memories_without_code_refs(mock_db, limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_all_linked(self):
        """Test returning empty list when all memories are already linked."""
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.limit.return_value = []
        mock_query.__iter__ = lambda self: iter([])

        mock_db = MagicMock()
        mock_db.query.return_value = mock_query

        result = await get_memories_without_code_refs(mock_db, limit=100)

        assert result == []


# =============================================================================
# batch_link_memories Tests
# =============================================================================


class TestBatchLinkMemories:
    """Tests for batch_link_memories function."""

    @pytest.mark.asyncio
    async def test_links_memories_with_matching_entities(self):
        """Test that memories are linked when entity matches code symbols."""
        mock_memory_obj = MockMemory(
            id="mem_1",
            memory="AuthService handles auth",
            metadata_={"entity": "AuthService"},
        )

        mock_links = [
            CodeLink(
                file_path="/src/auth/auth_service.ts",
                line_start=10,
                line_end=100,
                symbol_id="scip-ts AuthService#",
                symbol_name="AuthService",
                link_source="entity_match",
            )
        ]

        with patch("app.scripts.batch_link_memories.is_neo4j_configured") as mock_configured:
            with patch("app.scripts.batch_link_memories.get_neo4j_driver") as mock_driver:
                with patch("app.scripts.batch_link_memories.SessionLocal") as mock_session_cls:
                    with patch("app.scripts.batch_link_memories.get_memories_without_code_refs") as mock_get:
                        with patch("app.scripts.batch_link_memories.find_code_links_for_memory") as mock_find:
                            mock_configured.return_value = True
                            mock_driver.return_value = MagicMock()
                            mock_get.return_value = [{
                                "id": "mem_1",
                                "text": "AuthService handles auth",
                                "metadata": {"entity": "AuthService"},
                            }]
                            mock_find.return_value = mock_links

                            mock_db = MagicMock()
                            mock_query = MagicMock()
                            mock_query.filter.return_value.first.return_value = mock_memory_obj
                            mock_db.query.return_value = mock_query
                            mock_session_cls.return_value = mock_db

                            linked_count = await batch_link_memories(limit=100, dry_run=False)

                            assert linked_count == 1
                            mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_dry_run_does_not_update(self):
        """Test that dry_run mode does not actually update memories."""
        mock_links = [
            CodeLink(
                file_path="/src/auth/auth_service.ts",
                line_start=10,
                symbol_name="AuthService",
                link_source="entity_match",
            )
        ]

        with patch("app.scripts.batch_link_memories.is_neo4j_configured") as mock_configured:
            with patch("app.scripts.batch_link_memories.get_neo4j_driver") as mock_driver:
                with patch("app.scripts.batch_link_memories.SessionLocal") as mock_session_cls:
                    with patch("app.scripts.batch_link_memories.get_memories_without_code_refs") as mock_get:
                        with patch("app.scripts.batch_link_memories.find_code_links_for_memory") as mock_find:
                            mock_configured.return_value = True
                            mock_driver.return_value = MagicMock()
                            mock_get.return_value = [{
                                "id": "mem_1",
                                "text": "AuthService handles auth",
                                "metadata": {"entity": "AuthService"},
                            }]
                            mock_find.return_value = mock_links

                            mock_db = MagicMock()
                            mock_session_cls.return_value = mock_db

                            linked_count = await batch_link_memories(limit=100, dry_run=True)

                            # Dry run should not commit
                            mock_db.commit.assert_not_called()
                            assert linked_count == 0

    @pytest.mark.asyncio
    async def test_handles_no_neo4j(self):
        """Test graceful handling when Neo4j is not configured."""
        with patch("app.scripts.batch_link_memories.is_neo4j_configured") as mock_configured:
            mock_configured.return_value = False

            linked_count = await batch_link_memories(limit=100, dry_run=False)

            assert linked_count == 0

    @pytest.mark.asyncio
    async def test_filters_to_entity_match_links_only(self):
        """Test that only entity_match links are stored (not explicit)."""
        mock_memory_obj = MockMemory(
            id="mem_1",
            memory="Memory",
            metadata_={"entity": "Service"},
        )

        # Return both explicit and entity_match links
        mock_links = [
            CodeLink(
                file_path="/explicit.ts",
                symbol_name="Explicit",
                link_source="explicit",  # Should be filtered out
            ),
            CodeLink(
                file_path="/matched.ts",
                symbol_name="Matched",
                link_source="entity_match",  # Should be kept
            ),
        ]

        with patch("app.scripts.batch_link_memories.is_neo4j_configured") as mock_configured:
            with patch("app.scripts.batch_link_memories.get_neo4j_driver") as mock_driver:
                with patch("app.scripts.batch_link_memories.SessionLocal") as mock_session_cls:
                    with patch("app.scripts.batch_link_memories.get_memories_without_code_refs") as mock_get:
                        with patch("app.scripts.batch_link_memories.find_code_links_for_memory") as mock_find:
                            mock_configured.return_value = True
                            mock_driver.return_value = MagicMock()
                            mock_get.return_value = [{
                                "id": "mem_1",
                                "text": "Memory",
                                "metadata": {"entity": "Service"},
                            }]
                            mock_find.return_value = mock_links

                            mock_db = MagicMock()
                            mock_query = MagicMock()
                            mock_query.filter.return_value.first.return_value = mock_memory_obj
                            mock_db.query.return_value = mock_query
                            mock_session_cls.return_value = mock_db

                            await batch_link_memories(limit=100, dry_run=False)

                            # Check that metadata was updated with only entity_match link
                            assert mock_memory_obj.metadata_["code_refs"][0]["link_source"] == "entity_match"
                            assert len(mock_memory_obj.metadata_["code_refs"]) == 1

    @pytest.mark.asyncio
    async def test_handles_no_memories_to_process(self):
        """Test graceful handling when no memories need linking."""
        with patch("app.scripts.batch_link_memories.is_neo4j_configured") as mock_configured:
            with patch("app.scripts.batch_link_memories.get_neo4j_driver") as mock_driver:
                with patch("app.scripts.batch_link_memories.SessionLocal") as mock_session_cls:
                    with patch("app.scripts.batch_link_memories.get_memories_without_code_refs") as mock_get:
                        mock_configured.return_value = True
                        mock_driver.return_value = MagicMock()
                        mock_get.return_value = []

                        mock_db = MagicMock()
                        mock_session_cls.return_value = mock_db

                        linked_count = await batch_link_memories(limit=100, dry_run=False)

                        assert linked_count == 0

    @pytest.mark.asyncio
    async def test_converts_code_links_to_code_refs_format(self):
        """Test that CodeLink objects are converted to code_refs dict format."""
        mock_memory_obj = MockMemory(
            id="mem_1",
            memory="Memory",
            metadata_={"entity": "TestService"},
        )

        mock_links = [
            CodeLink(
                file_path="/src/test_service.ts",
                line_start=10,
                line_end=50,
                symbol_id="scip-ts TestService#",
                symbol_name="TestService",
                link_source="entity_match",
            )
        ]

        with patch("app.scripts.batch_link_memories.is_neo4j_configured") as mock_configured:
            with patch("app.scripts.batch_link_memories.get_neo4j_driver") as mock_driver:
                with patch("app.scripts.batch_link_memories.SessionLocal") as mock_session_cls:
                    with patch("app.scripts.batch_link_memories.get_memories_without_code_refs") as mock_get:
                        with patch("app.scripts.batch_link_memories.find_code_links_for_memory") as mock_find:
                            mock_configured.return_value = True
                            mock_driver.return_value = MagicMock()
                            mock_get.return_value = [{
                                "id": "mem_1",
                                "text": "Memory",
                                "metadata": {"entity": "TestService"},
                            }]
                            mock_find.return_value = mock_links

                            mock_db = MagicMock()
                            mock_query = MagicMock()
                            mock_query.filter.return_value.first.return_value = mock_memory_obj
                            mock_db.query.return_value = mock_query
                            mock_session_cls.return_value = mock_db

                            await batch_link_memories(limit=100, dry_run=False)

                            code_refs = mock_memory_obj.metadata_["code_refs"]
                            assert len(code_refs) == 1
                            ref = code_refs[0]
                            assert ref["file_path"] == "/src/test_service.ts"
                            assert ref["line_start"] == 10
                            assert ref["line_end"] == 50
                            assert ref["symbol_id"] == "scip-ts TestService#"
                            assert ref["link_source"] == "entity_match"
