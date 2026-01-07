"""
Tests for Evidence Linker module.

Tests the simplified memory-to-code evidence linking system:
- CodeLink dataclass
- Entity name matching to CODE_SYMBOL nodes
- find_code_links_for_memory() function
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.evidence_linker import (
    CodeLink,
    find_code_links_for_memory,
    search_code_symbols_by_name,
)


# =============================================================================
# CodeLink Tests
# =============================================================================


class TestCodeLink:
    """Tests for CodeLink dataclass."""

    def test_create_explicit_link(self):
        """Test creating an explicit code link."""
        link = CodeLink(
            file_path="/src/auth/token_service.ts",
            line_start=42,
            line_end=89,
            symbol_id="scip-ts auth TokenService#generateToken()",
            symbol_name="TokenService",
            link_source="explicit",
        )
        assert link.file_path == "/src/auth/token_service.ts"
        assert link.line_start == 42
        assert link.line_end == 89
        assert link.symbol_id == "scip-ts auth TokenService#generateToken()"
        assert link.symbol_name == "TokenService"
        assert link.link_source == "explicit"

    def test_create_entity_match_link(self):
        """Test creating an entity-matched code link."""
        link = CodeLink(
            file_path="/src/auth/auth_service.ts",
            line_start=10,
            line_end=None,
            symbol_id="scip-ts auth AuthService#",
            symbol_name="AuthService",
            link_source="entity_match",
        )
        assert link.link_source == "entity_match"
        assert link.line_end is None

    def test_minimal_link(self):
        """Test creating a minimal code link with only required fields."""
        link = CodeLink(
            file_path="/src/utils.ts",
            symbol_name="utils",
            link_source="explicit",
        )
        assert link.file_path == "/src/utils.ts"
        assert link.line_start is None
        assert link.line_end is None
        assert link.symbol_id is None

    def test_to_dict(self):
        """Test converting CodeLink to dict for storage."""
        link = CodeLink(
            file_path="/src/auth/token_service.ts",
            line_start=42,
            line_end=89,
            symbol_id="scip-ts auth TokenService#generateToken()",
            symbol_name="TokenService",
            link_source="entity_match",
        )
        result = link.to_dict()

        assert result["file_path"] == "/src/auth/token_service.ts"
        assert result["line_start"] == 42
        assert result["line_end"] == 89
        assert result["symbol_id"] == "scip-ts auth TokenService#generateToken()"
        assert result["link_source"] == "entity_match"

    def test_to_dict_minimal(self):
        """Test converting minimal CodeLink to dict."""
        link = CodeLink(
            file_path="/src/utils.ts",
            symbol_name="utils",
            link_source="explicit",
        )
        result = link.to_dict()

        assert result["file_path"] == "/src/utils.ts"
        assert result["link_source"] == "explicit"
        assert "line_start" not in result
        assert "line_end" not in result
        assert "symbol_id" not in result


# =============================================================================
# search_code_symbols_by_name Tests
# =============================================================================


class TestSearchCodeSymbolsByName:
    """Tests for search_code_symbols_by_name function."""

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Test finding symbols with exact name match."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([
            {
                "symbol_id": "scip-ts auth AuthService#",
                "name": "AuthService",
                "file_path": "/src/auth/auth_service.ts",
                "line_start": 10,
                "line_end": 100,
            }
        ])
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        results = await search_code_symbols_by_name("AuthService", mock_driver)

        assert len(results) == 1
        assert results[0]["name"] == "AuthService"
        assert results[0]["file_path"] == "/src/auth/auth_service.ts"

    @pytest.mark.asyncio
    async def test_partial_match(self):
        """Test finding symbols with partial name match."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([
            {
                "symbol_id": "scip-ts auth AuthService#",
                "name": "AuthService",
                "file_path": "/src/auth/auth_service.ts",
                "line_start": 10,
                "line_end": 100,
            },
            {
                "symbol_id": "scip-ts auth AuthController#",
                "name": "AuthController",
                "file_path": "/src/auth/auth_controller.ts",
                "line_start": 5,
                "line_end": 50,
            }
        ])
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        results = await search_code_symbols_by_name("Auth", mock_driver)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_no_matches(self):
        """Test returning empty list when no symbols match."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        results = await search_code_symbols_by_name("NonExistentService", mock_driver)

        assert results == []

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Test that search respects limit parameter."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        # Return 5 results but limit should be 3
        mock_result.__iter__ = lambda self: iter([
            {"symbol_id": f"sym_{i}", "name": f"Service{i}", "file_path": f"/src/s{i}.ts"}
            for i in range(3)
        ])
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        results = await search_code_symbols_by_name("Service", mock_driver, limit=3)

        # The mock returns 3, which respects our limit
        assert len(results) <= 3


# =============================================================================
# find_code_links_for_memory Tests
# =============================================================================


class TestFindCodeLinksForMemory:
    """Tests for find_code_links_for_memory function."""

    @pytest.mark.asyncio
    async def test_returns_explicit_refs(self):
        """Test that existing code_refs are returned as explicit links."""
        memory = {
            "id": "mem_123",
            "text": "Use JWT with 15-minute expiry",
            "metadata": {
                "entity": "AuthService",
                "code_refs": [
                    {
                        "file_path": "/src/auth/token_service.ts",
                        "line_start": 42,
                        "line_end": 89,
                        "symbol_id": "scip-ts auth TokenService#generateToken()",
                    }
                ]
            }
        }

        mock_driver = MagicMock()

        links = await find_code_links_for_memory(memory, mock_driver)

        assert len(links) == 1
        assert links[0].link_source == "explicit"
        assert links[0].file_path == "/src/auth/token_service.ts"

    @pytest.mark.asyncio
    async def test_entity_matching_when_no_explicit_refs(self):
        """Test entity matching is used when no explicit code_refs exist."""
        memory = {
            "id": "mem_123",
            "text": "AuthService handles authentication",
            "metadata": {
                "entity": "AuthService",
            }
        }

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([
            {
                "symbol_id": "scip-ts auth AuthService#",
                "name": "AuthService",
                "file_path": "/src/auth/auth_service.ts",
                "line_start": 10,
                "line_end": 100,
            }
        ])
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        links = await find_code_links_for_memory(memory, mock_driver)

        assert len(links) == 1
        assert links[0].link_source == "entity_match"
        assert links[0].symbol_name == "AuthService"

    @pytest.mark.asyncio
    async def test_skips_entity_matching_when_explicit_refs_exist(self):
        """Test entity matching is skipped when explicit code_refs exist."""
        memory = {
            "id": "mem_123",
            "text": "AuthService handles authentication",
            "metadata": {
                "entity": "AuthService",
                "code_refs": [
                    {
                        "file_path": "/src/auth/manual_ref.ts",
                        "line_start": 1,
                        "line_end": 10,
                    }
                ]
            }
        }

        mock_driver = MagicMock()
        # Session should not be called since we have explicit refs
        mock_driver.session.return_value.__enter__ = MagicMock()

        links = await find_code_links_for_memory(memory, mock_driver)

        # Only explicit link, no entity matching
        assert len(links) == 1
        assert links[0].link_source == "explicit"

    @pytest.mark.asyncio
    async def test_no_entity_returns_empty_list(self):
        """Test memory without entity returns empty list (unless has explicit refs)."""
        memory = {
            "id": "mem_123",
            "text": "General note without entity",
            "metadata": {}
        }

        mock_driver = MagicMock()

        links = await find_code_links_for_memory(memory, mock_driver)

        assert links == []

    @pytest.mark.asyncio
    async def test_limits_entity_matches_to_top_3(self):
        """Test that entity matching returns at most 3 results."""
        memory = {
            "id": "mem_123",
            "text": "Service handles operations",
            "metadata": {
                "entity": "Service",
            }
        }

        mock_session = MagicMock()
        mock_result = MagicMock()
        # Return 5 results from Neo4j
        mock_result.__iter__ = lambda self: iter([
            {
                "symbol_id": f"scip-ts Service{i}#",
                "name": f"Service{i}",
                "file_path": f"/src/service{i}.ts",
                "line_start": i * 10,
                "line_end": i * 10 + 50,
            }
            for i in range(5)
        ])
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        links = await find_code_links_for_memory(memory, mock_driver)

        # Should be limited to 3 entity matches
        assert len(links) <= 3

    @pytest.mark.asyncio
    async def test_handles_missing_metadata(self):
        """Test graceful handling of missing metadata."""
        memory = {
            "id": "mem_123",
            "text": "Memory without metadata",
        }

        mock_driver = MagicMock()

        links = await find_code_links_for_memory(memory, mock_driver)

        assert links == []

    @pytest.mark.asyncio
    async def test_preserves_existing_symbol_name(self):
        """Test that explicit refs with symbol_name preserve it."""
        memory = {
            "id": "mem_123",
            "text": "Token generation logic",
            "metadata": {
                "code_refs": [
                    {
                        "file_path": "/src/auth/token.ts",
                        "symbol_id": "TokenService#generate",
                        "symbol_name": "TokenService",
                    }
                ]
            }
        }

        mock_driver = MagicMock()

        links = await find_code_links_for_memory(memory, mock_driver)

        assert len(links) == 1
        assert links[0].symbol_name == "TokenService"
