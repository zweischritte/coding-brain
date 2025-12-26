"""Tests for scope hierarchy and multi-scope retrieval.

Per v9 plan section 4.2:
- Scope hierarchy: session > user > team > project > org > enterprise
- De-duplicate by content hash; keep highest-precedence result
- Multi-team users include all team_ids unless request narrows scope
"""

import hashlib
from datetime import datetime, timezone
from typing import Any

import pytest


class TestMemoryScope:
    """Tests for MemoryScope enum."""

    def test_memory_scope_enum_values(self):
        """MemoryScope should have all required scope levels."""
        from openmemory.api.memory import MemoryScope

        assert MemoryScope.SESSION.value == "session"
        assert MemoryScope.USER.value == "user"
        assert MemoryScope.TEAM.value == "team"
        assert MemoryScope.PROJECT.value == "project"
        assert MemoryScope.ORG.value == "org"
        assert MemoryScope.ENTERPRISE.value == "enterprise"

    def test_memory_scope_precedence_order(self):
        """Scopes should have correct precedence ordering."""
        from openmemory.api.memory import MemoryScope

        # Precedence: session > user > team > project > org > enterprise
        assert MemoryScope.SESSION.precedence > MemoryScope.USER.precedence
        assert MemoryScope.USER.precedence > MemoryScope.TEAM.precedence
        assert MemoryScope.TEAM.precedence > MemoryScope.PROJECT.precedence
        assert MemoryScope.PROJECT.precedence > MemoryScope.ORG.precedence
        assert MemoryScope.ORG.precedence > MemoryScope.ENTERPRISE.precedence

    def test_memory_scope_from_string(self):
        """Should parse scope from string."""
        from openmemory.api.memory import MemoryScope

        assert MemoryScope.from_string("session") == MemoryScope.SESSION
        assert MemoryScope.from_string("user") == MemoryScope.USER
        assert MemoryScope.from_string("SESSION") == MemoryScope.SESSION

    def test_memory_scope_invalid_string(self):
        """Should raise ValueError for invalid scope string."""
        from openmemory.api.memory import MemoryScope

        with pytest.raises(ValueError, match="Unknown scope"):
            MemoryScope.from_string("invalid")


class TestScopedMemory:
    """Tests for ScopedMemory model."""

    def test_scoped_memory_creation(self):
        """ScopedMemory should store all scope fields."""
        from openmemory.api.memory import ScopedMemory, MemoryScope

        memory = ScopedMemory(
            memory_id="mem_123",
            content="Test memory content",
            content_hash="abc123",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )

        assert memory.memory_id == "mem_123"
        assert memory.content == "Test memory content"
        assert memory.content_hash == "abc123"
        assert memory.scope == MemoryScope.USER
        assert memory.user_id == "user_456"
        assert memory.org_id == "org_789"

    def test_scoped_memory_optional_fields(self):
        """ScopedMemory should support optional scope fields."""
        from openmemory.api.memory import ScopedMemory, MemoryScope

        memory = ScopedMemory(
            memory_id="mem_123",
            content="Test",
            content_hash="abc",
            scope=MemoryScope.SESSION,
            user_id="user_456",
            org_id="org_789",
            session_id="sess_001",
            team_id="team_abc",
            project_id="proj_xyz",
            enterprise_id="ent_000",
            geo_scope="us-west-2",
        )

        assert memory.session_id == "sess_001"
        assert memory.team_id == "team_abc"
        assert memory.project_id == "proj_xyz"
        assert memory.enterprise_id == "ent_000"
        assert memory.geo_scope == "us-west-2"

    def test_scoped_memory_auto_content_hash(self):
        """ScopedMemory should auto-compute content hash if not provided."""
        from openmemory.api.memory import ScopedMemory, MemoryScope

        content = "Test memory content"
        memory = ScopedMemory(
            memory_id="mem_123",
            content=content,
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )

        expected_hash = hashlib.sha256(content.encode()).hexdigest()
        assert memory.content_hash == expected_hash

    def test_scoped_memory_created_at_default(self):
        """ScopedMemory should default created_at to now."""
        from openmemory.api.memory import ScopedMemory, MemoryScope

        before = datetime.now(timezone.utc)
        memory = ScopedMemory(
            memory_id="mem_123",
            content="Test",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )
        after = datetime.now(timezone.utc)

        assert before <= memory.created_at <= after

    def test_scoped_memory_metadata(self):
        """ScopedMemory should support arbitrary metadata."""
        from openmemory.api.memory import ScopedMemory, MemoryScope

        memory = ScopedMemory(
            memory_id="mem_123",
            content="Test",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
            metadata={"source": "conversation", "tool": "search_code"},
        )

        assert memory.metadata["source"] == "conversation"
        assert memory.metadata["tool"] == "search_code"


class TestScopeContext:
    """Tests for ScopeContext (defines accessor's permitted scopes)."""

    def test_scope_context_creation(self):
        """ScopeContext should capture accessor's identity."""
        from openmemory.api.memory import ScopeContext

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
        )

        assert ctx.user_id == "user_456"
        assert ctx.org_id == "org_789"

    def test_scope_context_multi_team(self):
        """ScopeContext should support multiple team memberships."""
        from openmemory.api.memory import ScopeContext

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            team_ids=["team_a", "team_b", "team_c"],
        )

        assert ctx.team_ids == ["team_a", "team_b", "team_c"]

    def test_scope_context_full_hierarchy(self):
        """ScopeContext should support full scope hierarchy."""
        from openmemory.api.memory import ScopeContext

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            enterprise_id="ent_000",
            session_id="sess_001",
            team_ids=["team_a"],
            project_ids=["proj_x", "proj_y"],
        )

        assert ctx.enterprise_id == "ent_000"
        assert ctx.session_id == "sess_001"
        assert ctx.project_ids == ["proj_x", "proj_y"]

    def test_scope_context_from_principal(self):
        """ScopeContext should be constructible from RBAC Principal."""
        from openmemory.api.memory import ScopeContext
        from openmemory.api.security.rbac import Principal, Role, Scope

        principal = Principal(
            user_id="user_456",
            org_id="org_789",
            enterprise_id="ent_000",
            session_id="sess_001",
            team_ids=["team_a", "team_b"],
            project_ids=["proj_x"],
            roles=[Role.MAINTAINER],
            scopes=[Scope.MEMORY_WRITE],
        )

        ctx = ScopeContext.from_principal(principal)

        assert ctx.user_id == "user_456"
        assert ctx.org_id == "org_789"
        assert ctx.enterprise_id == "ent_000"
        assert ctx.session_id == "sess_001"
        assert ctx.team_ids == ["team_a", "team_b"]
        assert ctx.project_ids == ["proj_x"]

    def test_scope_context_accessible_scopes(self):
        """ScopeContext should compute all accessible scopes."""
        from openmemory.api.memory import ScopeContext, MemoryScope

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            enterprise_id="ent_000",
            session_id="sess_001",
            team_ids=["team_a"],
            project_ids=["proj_x"],
        )

        accessible = ctx.get_accessible_scopes()

        # Should include all scopes user has access to
        assert MemoryScope.SESSION in accessible
        assert MemoryScope.USER in accessible
        assert MemoryScope.TEAM in accessible
        assert MemoryScope.PROJECT in accessible
        assert MemoryScope.ORG in accessible
        assert MemoryScope.ENTERPRISE in accessible


class TestScopeFilter:
    """Tests for ScopeFilter (query filters for retrieval)."""

    def test_scope_filter_single_scope(self):
        """ScopeFilter should filter to a single scope level."""
        from openmemory.api.memory import ScopeFilter, MemoryScope

        filter = ScopeFilter(scopes=[MemoryScope.USER])

        assert filter.scopes == [MemoryScope.USER]

    def test_scope_filter_multiple_scopes(self):
        """ScopeFilter should support multiple scope levels."""
        from openmemory.api.memory import ScopeFilter, MemoryScope

        filter = ScopeFilter(scopes=[MemoryScope.USER, MemoryScope.TEAM])

        assert MemoryScope.USER in filter.scopes
        assert MemoryScope.TEAM in filter.scopes

    def test_scope_filter_scope_ids(self):
        """ScopeFilter should filter by specific scope IDs."""
        from openmemory.api.memory import ScopeFilter, MemoryScope

        filter = ScopeFilter(
            scopes=[MemoryScope.TEAM],
            team_ids=["team_a", "team_b"],
        )

        assert filter.team_ids == ["team_a", "team_b"]

    def test_scope_filter_from_context_default(self):
        """ScopeFilter.from_context should include all accessible scopes."""
        from openmemory.api.memory import ScopeContext, ScopeFilter, MemoryScope

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            session_id="sess_001",
            team_ids=["team_a"],
        )

        filter = ScopeFilter.from_context(ctx)

        # By default, include all accessible scopes
        assert MemoryScope.SESSION in filter.scopes
        assert MemoryScope.USER in filter.scopes
        assert MemoryScope.TEAM in filter.scopes
        assert MemoryScope.ORG in filter.scopes

    def test_scope_filter_from_context_narrowed(self):
        """ScopeFilter.from_context should support narrowing to specific scopes."""
        from openmemory.api.memory import ScopeContext, ScopeFilter, MemoryScope

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            team_ids=["team_a", "team_b"],
        )

        # Only want team-scoped memories from team_a
        filter = ScopeFilter.from_context(
            ctx,
            scopes=[MemoryScope.TEAM],
            team_ids=["team_a"],
        )

        assert filter.scopes == [MemoryScope.TEAM]
        assert filter.team_ids == ["team_a"]


class TestScopedMemoryStore:
    """Tests for ScopedMemoryStore (abstract storage interface)."""

    def test_in_memory_store_add_memory(self):
        """InMemoryScopedMemoryStore should store memories."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
        )

        store = InMemoryScopedMemoryStore()
        memory = ScopedMemory(
            memory_id="mem_123",
            content="Test content",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )

        store.add(memory)

        assert store.get("mem_123") == memory

    def test_in_memory_store_get_not_found(self):
        """InMemoryScopedMemoryStore should return None for missing IDs."""
        from openmemory.api.memory import InMemoryScopedMemoryStore

        store = InMemoryScopedMemoryStore()

        assert store.get("nonexistent") is None

    def test_in_memory_store_update_memory(self):
        """InMemoryScopedMemoryStore should update existing memories."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
        )

        store = InMemoryScopedMemoryStore()
        memory = ScopedMemory(
            memory_id="mem_123",
            content="Original",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )
        store.add(memory)

        updated = ScopedMemory(
            memory_id="mem_123",
            content="Updated",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )
        store.update(updated)

        result = store.get("mem_123")
        assert result.content == "Updated"

    def test_in_memory_store_delete_memory(self):
        """InMemoryScopedMemoryStore should delete memories."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
        )

        store = InMemoryScopedMemoryStore()
        memory = ScopedMemory(
            memory_id="mem_123",
            content="Test",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )
        store.add(memory)
        store.delete("mem_123")

        assert store.get("mem_123") is None

    def test_in_memory_store_query_by_scope(self):
        """InMemoryScopedMemoryStore should query by scope."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopeFilter,
        )

        store = InMemoryScopedMemoryStore()

        # Add memories at different scopes
        store.add(ScopedMemory(
            memory_id="mem_user",
            content="User memory",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        ))
        store.add(ScopedMemory(
            memory_id="mem_team",
            content="Team memory",
            scope=MemoryScope.TEAM,
            user_id="user_456",
            org_id="org_789",
            team_id="team_a",
        ))

        ctx = ScopeContext(user_id="user_456", org_id="org_789")
        filter = ScopeFilter(scopes=[MemoryScope.USER])

        results = store.query(ctx, filter)

        assert len(results) == 1
        assert results[0].memory_id == "mem_user"


class TestScopedRetriever:
    """Tests for ScopedRetriever with de-duplication."""

    def test_retriever_respects_scope_precedence(self):
        """Retriever should prefer higher-precedence scope for duplicates."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        # Same content at different scopes
        content = "Important information"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        store.add(ScopedMemory(
            memory_id="mem_org",
            content=content,
            content_hash=content_hash,
            scope=MemoryScope.ORG,
            user_id="user_456",
            org_id="org_789",
        ))
        store.add(ScopedMemory(
            memory_id="mem_user",
            content=content,
            content_hash=content_hash,
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        ))
        store.add(ScopedMemory(
            memory_id="mem_session",
            content=content,
            content_hash=content_hash,
            scope=MemoryScope.SESSION,
            user_id="user_456",
            org_id="org_789",
            session_id="sess_001",
        ))

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            session_id="sess_001",
        )
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        # Should de-duplicate and keep session (highest precedence)
        assert len(results.memories) == 1
        assert results.memories[0].memory_id == "mem_session"
        assert results.memories[0].scope == MemoryScope.SESSION

    def test_retriever_dedup_by_content_hash(self):
        """Retriever should de-duplicate by content hash."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        content_a = "Content A"
        content_b = "Content B"

        store.add(ScopedMemory(
            memory_id="mem_1",
            content=content_a,
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        ))
        store.add(ScopedMemory(
            memory_id="mem_2",
            content=content_b,
            scope=MemoryScope.ORG,
            user_id="user_456",
            org_id="org_789",
        ))

        ctx = ScopeContext(user_id="user_456", org_id="org_789")
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        # Different content hashes, so both should appear
        assert len(results.memories) == 2

    def test_retriever_multi_team_access(self):
        """Retriever should access all user's teams by default."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        store.add(ScopedMemory(
            memory_id="mem_team_a",
            content="Team A memory",
            scope=MemoryScope.TEAM,
            user_id="user_456",
            org_id="org_789",
            team_id="team_a",
        ))
        store.add(ScopedMemory(
            memory_id="mem_team_b",
            content="Team B memory",
            scope=MemoryScope.TEAM,
            user_id="user_456",
            org_id="org_789",
            team_id="team_b",
        ))
        store.add(ScopedMemory(
            memory_id="mem_team_c",
            content="Team C memory",
            scope=MemoryScope.TEAM,
            user_id="other_user",
            org_id="org_789",
            team_id="team_c",  # User not in this team
        ))

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            team_ids=["team_a", "team_b"],
        )
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        # Should get both teams user belongs to
        memory_ids = [m.memory_id for m in results.memories]
        assert "mem_team_a" in memory_ids
        assert "mem_team_b" in memory_ids
        assert "mem_team_c" not in memory_ids

    def test_retriever_narrow_to_single_team(self):
        """Retriever should support narrowing to specific team."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopeFilter,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        store.add(ScopedMemory(
            memory_id="mem_team_a",
            content="Team A memory",
            scope=MemoryScope.TEAM,
            user_id="user_456",
            org_id="org_789",
            team_id="team_a",
        ))
        store.add(ScopedMemory(
            memory_id="mem_team_b",
            content="Team B memory",
            scope=MemoryScope.TEAM,
            user_id="user_456",
            org_id="org_789",
            team_id="team_b",
        ))

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            team_ids=["team_a", "team_b"],
        )
        filter = ScopeFilter(
            scopes=[MemoryScope.TEAM],
            team_ids=["team_a"],  # Narrow to single team
        )
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx, filter)

        assert len(results.memories) == 1
        assert results.memories[0].memory_id == "mem_team_a"

    def test_retriever_org_isolation(self):
        """Retriever should isolate memories by org."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        store.add(ScopedMemory(
            memory_id="mem_org_a",
            content="Org A memory",
            scope=MemoryScope.ORG,
            user_id="user_456",
            org_id="org_a",
        ))
        store.add(ScopedMemory(
            memory_id="mem_org_b",
            content="Org B memory",
            scope=MemoryScope.ORG,
            user_id="user_456",
            org_id="org_b",
        ))

        ctx = ScopeContext(user_id="user_456", org_id="org_a")
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        assert len(results.memories) == 1
        assert results.memories[0].memory_id == "mem_org_a"

    def test_retriever_enterprise_scope(self):
        """Retriever should support enterprise-wide memories."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        store.add(ScopedMemory(
            memory_id="mem_enterprise",
            content="Enterprise-wide memory",
            scope=MemoryScope.ENTERPRISE,
            user_id="admin_user",
            org_id="org_a",
            enterprise_id="ent_000",
        ))
        store.add(ScopedMemory(
            memory_id="mem_org",
            content="Org-level memory",
            scope=MemoryScope.ORG,
            user_id="user_456",
            org_id="org_a",
        ))

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_a",
            enterprise_id="ent_000",
        )
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        memory_ids = [m.memory_id for m in results.memories]
        assert "mem_enterprise" in memory_ids
        assert "mem_org" in memory_ids

    def test_retriever_session_isolation(self):
        """Retriever should isolate session memories."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        store.add(ScopedMemory(
            memory_id="mem_sess_1",
            content="Session 1 memory",
            scope=MemoryScope.SESSION,
            user_id="user_456",
            org_id="org_789",
            session_id="sess_001",
        ))
        store.add(ScopedMemory(
            memory_id="mem_sess_2",
            content="Session 2 memory",
            scope=MemoryScope.SESSION,
            user_id="user_456",
            org_id="org_789",
            session_id="sess_002",
        ))

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            session_id="sess_001",
        )
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        session_memories = [
            m for m in results.memories
            if m.scope == MemoryScope.SESSION
        ]
        assert len(session_memories) == 1
        assert session_memories[0].memory_id == "mem_sess_1"


class TestScopedRetrievalResult:
    """Tests for ScopedRetrievalResult."""

    def test_result_metadata(self):
        """Result should include retrieval metadata."""
        from openmemory.api.memory import ScopedRetrievalResult, ScopedMemory, MemoryScope

        memories = [
            ScopedMemory(
                memory_id="mem_1",
                content="Test",
                scope=MemoryScope.USER,
                user_id="user_456",
                org_id="org_789",
            )
        ]

        result = ScopedRetrievalResult(
            memories=memories,
            total_before_dedup=3,
            total_after_dedup=1,
        )

        assert result.total_before_dedup == 3
        assert result.total_after_dedup == 1
        assert len(result.memories) == 1

    def test_result_dedup_stats(self):
        """Result should track de-duplication statistics."""
        from openmemory.api.memory import ScopedRetrievalResult, ScopedMemory, MemoryScope

        result = ScopedRetrievalResult(
            memories=[],
            total_before_dedup=10,
            total_after_dedup=4,
            dedup_by_scope={
                MemoryScope.SESSION: 2,
                MemoryScope.USER: 3,
                MemoryScope.ORG: 1,
            },
        )

        assert result.dedup_by_scope[MemoryScope.SESSION] == 2
        assert result.dedup_by_scope[MemoryScope.USER] == 3


class TestScopeBackfill:
    """Tests for scope backfill utilities."""

    def test_backfill_legacy_memory(self):
        """Should backfill legacy memories with default scope."""
        from openmemory.api.memory.scope import backfill_legacy_memory
        from openmemory.api.memory import ScopedMemory, MemoryScope

        # Legacy memory without scope fields
        legacy = {
            "memory_id": "mem_123",
            "content": "Legacy content",
            "user_id": "user_456",
            "app_id": "app_xyz",
        }

        memory = backfill_legacy_memory(
            legacy,
            default_org_id="org_default",
        )

        assert memory.memory_id == "mem_123"
        assert memory.scope == MemoryScope.USER  # Default scope
        assert memory.org_id == "org_default"
        assert memory.user_id == "user_456"

    def test_backfill_with_org_lookup(self):
        """Should use org lookup function for backfill."""
        from openmemory.api.memory.scope import backfill_legacy_memory

        def lookup_org(user_id: str) -> str:
            return f"org_for_{user_id}"

        legacy = {
            "memory_id": "mem_123",
            "content": "Legacy content",
            "user_id": "user_456",
        }

        memory = backfill_legacy_memory(legacy, org_lookup=lookup_org)

        assert memory.org_id == "org_for_user_456"


class TestGeoScopeInheritance:
    """Tests for geo_scope inheritance per v9 plan."""

    def test_geo_scope_inherits_from_org(self):
        """geo_scope should inherit from org by default."""
        from openmemory.api.memory import ScopeContext

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            org_geo_scope="eu-west-1",  # Org's geo scope
        )

        assert ctx.effective_geo_scope == "eu-west-1"

    def test_geo_scope_override_at_team(self):
        """team geo_scope should override org."""
        from openmemory.api.memory import ScopeContext

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            org_geo_scope="eu-west-1",
            geo_scope="us-west-2",  # Explicit override
        )

        assert ctx.effective_geo_scope == "us-west-2"


class TestScopeFilterMatching:
    """Tests for scope filter matching logic."""

    def test_filter_matches_user_memory(self):
        """Filter should match user-scoped memory for correct user."""
        from openmemory.api.memory import (
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopeFilter,
        )
        from openmemory.api.memory.scope import memory_matches_filter

        memory = ScopedMemory(
            memory_id="mem_1",
            content="User memory",
            scope=MemoryScope.USER,
            user_id="user_456",
            org_id="org_789",
        )

        ctx = ScopeContext(user_id="user_456", org_id="org_789")
        filter = ScopeFilter.from_context(ctx)

        assert memory_matches_filter(memory, ctx, filter)

    def test_filter_rejects_other_user_memory(self):
        """Filter should reject memories from other users at USER scope."""
        from openmemory.api.memory import (
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopeFilter,
        )
        from openmemory.api.memory.scope import memory_matches_filter

        memory = ScopedMemory(
            memory_id="mem_1",
            content="Other user memory",
            scope=MemoryScope.USER,
            user_id="other_user",
            org_id="org_789",
        )

        ctx = ScopeContext(user_id="user_456", org_id="org_789")
        filter = ScopeFilter.from_context(ctx)

        assert not memory_matches_filter(memory, ctx, filter)

    def test_filter_accepts_shared_team_memory(self):
        """Filter should accept team memories user belongs to."""
        from openmemory.api.memory import (
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopeFilter,
        )
        from openmemory.api.memory.scope import memory_matches_filter

        memory = ScopedMemory(
            memory_id="mem_1",
            content="Team memory",
            scope=MemoryScope.TEAM,
            user_id="other_user",  # Created by another team member
            org_id="org_789",
            team_id="team_a",
        )

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            team_ids=["team_a", "team_b"],
        )
        filter = ScopeFilter.from_context(ctx)

        assert memory_matches_filter(memory, ctx, filter)

    def test_filter_rejects_other_team_memory(self):
        """Filter should reject team memories user doesn't belong to."""
        from openmemory.api.memory import (
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopeFilter,
        )
        from openmemory.api.memory.scope import memory_matches_filter

        memory = ScopedMemory(
            memory_id="mem_1",
            content="Other team memory",
            scope=MemoryScope.TEAM,
            user_id="other_user",
            org_id="org_789",
            team_id="team_c",  # User not in this team
        )

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            team_ids=["team_a", "team_b"],
        )
        filter = ScopeFilter.from_context(ctx)

        assert not memory_matches_filter(memory, ctx, filter)


class TestProjectScopeRetrieval:
    """Tests for project-scoped memory retrieval."""

    def test_project_scope_access(self):
        """User should access project memories they belong to."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        store.add(ScopedMemory(
            memory_id="mem_proj_x",
            content="Project X memory",
            scope=MemoryScope.PROJECT,
            user_id="user_456",
            org_id="org_789",
            project_id="proj_x",
        ))
        store.add(ScopedMemory(
            memory_id="mem_proj_y",
            content="Project Y memory",
            scope=MemoryScope.PROJECT,
            user_id="other_user",
            org_id="org_789",
            project_id="proj_y",  # User not in this project
        ))

        ctx = ScopeContext(
            user_id="user_456",
            org_id="org_789",
            project_ids=["proj_x"],
        )
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx)

        project_memories = [
            m for m in results.memories
            if m.scope == MemoryScope.PROJECT
        ]
        assert len(project_memories) == 1
        assert project_memories[0].memory_id == "mem_proj_x"


class TestLimitAndPagination:
    """Tests for retrieval limit and pagination."""

    def test_retriever_respects_limit(self):
        """Retriever should respect limit parameter."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        for i in range(10):
            store.add(ScopedMemory(
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                scope=MemoryScope.USER,
                user_id="user_456",
                org_id="org_789",
            ))

        ctx = ScopeContext(user_id="user_456", org_id="org_789")
        retriever = ScopedRetriever(store)

        results = retriever.retrieve(ctx, limit=5)

        assert len(results.memories) == 5

    def test_retriever_offset_pagination(self):
        """Retriever should support offset pagination."""
        from openmemory.api.memory import (
            InMemoryScopedMemoryStore,
            ScopedMemory,
            MemoryScope,
            ScopeContext,
            ScopedRetriever,
        )

        store = InMemoryScopedMemoryStore()

        for i in range(10):
            store.add(ScopedMemory(
                memory_id=f"mem_{i}",
                content=f"Memory {i}",
                scope=MemoryScope.USER,
                user_id="user_456",
                org_id="org_789",
            ))

        ctx = ScopeContext(user_id="user_456", org_id="org_789")
        retriever = ScopedRetriever(store)

        page1 = retriever.retrieve(ctx, limit=3, offset=0)
        page2 = retriever.retrieve(ctx, limit=3, offset=3)

        # Pages should have different memories
        page1_ids = {m.memory_id for m in page1.memories}
        page2_ids = {m.memory_id for m in page2.memories}
        assert page1_ids.isdisjoint(page2_ids)
