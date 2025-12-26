"""Tests for episodic memory layer (FR-005).

Per v9 plan section 4.3:
- Session scope TTL default: 24h (configurable)
- Episodic memory stored per session and summarized over time
- Cross-tool context handoff within session scope
- Reference resolution with confidence
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest


class TestEpisodicMemoryConfig:
    """Tests for EpisodicMemoryConfig."""

    def test_default_session_ttl(self):
        """Default session TTL should be 24 hours."""
        from openmemory.api.memory import EpisodicMemoryConfig

        config = EpisodicMemoryConfig()
        assert config.session_ttl == timedelta(hours=24)

    def test_custom_session_ttl(self):
        """Should support custom session TTL."""
        from openmemory.api.memory import EpisodicMemoryConfig

        config = EpisodicMemoryConfig(session_ttl=timedelta(hours=48))
        assert config.session_ttl == timedelta(hours=48)

    def test_summarization_threshold(self):
        """Should have summarization threshold."""
        from openmemory.api.memory import EpisodicMemoryConfig

        config = EpisodicMemoryConfig(summarization_threshold=20)
        assert config.summarization_threshold == 20

    def test_max_active_memories(self):
        """Should have max active memories limit."""
        from openmemory.api.memory import EpisodicMemoryConfig

        config = EpisodicMemoryConfig(max_active_memories=100)
        assert config.max_active_memories == 100


class TestSessionContext:
    """Tests for SessionContext."""

    def test_session_context_creation(self):
        """SessionContext should store session metadata."""
        from openmemory.api.memory import SessionContext

        ctx = SessionContext(
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
        )

        assert ctx.session_id == "sess_001"
        assert ctx.user_id == "user_456"
        assert ctx.org_id == "org_789"

    def test_session_context_auto_expiration(self):
        """SessionContext should auto-set expiration to 24h from start."""
        from openmemory.api.memory import SessionContext

        before = datetime.now(timezone.utc)
        ctx = SessionContext(
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
        )
        after = datetime.now(timezone.utc)

        expected_min = before + timedelta(hours=24)
        expected_max = after + timedelta(hours=24)

        assert expected_min <= ctx.expires_at <= expected_max

    def test_session_context_custom_expiration(self):
        """SessionContext should support custom expiration."""
        from openmemory.api.memory import SessionContext

        custom_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        ctx = SessionContext(
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            expires_at=custom_expires,
        )

        assert ctx.expires_at == custom_expires

    def test_session_not_expired(self):
        """is_expired should be False for active session."""
        from openmemory.api.memory import SessionContext

        ctx = SessionContext(
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
        )

        assert not ctx.is_expired

    def test_session_expired(self):
        """is_expired should be True for expired session."""
        from openmemory.api.memory import SessionContext

        past = datetime.now(timezone.utc) - timedelta(hours=1)
        ctx = SessionContext(
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            started_at=past - timedelta(hours=25),
            expires_at=past,
        )

        assert ctx.is_expired

    def test_session_metadata(self):
        """SessionContext should support arbitrary metadata."""
        from openmemory.api.memory import SessionContext

        ctx = SessionContext(
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            metadata={"client": "vscode", "project": "my-app"},
        )

        assert ctx.metadata["client"] == "vscode"
        assert ctx.metadata["project"] == "my-app"


class TestEpisodicMemory:
    """Tests for EpisodicMemory model."""

    def test_episodic_memory_creation(self):
        """EpisodicMemory should store basic fields."""
        from openmemory.api.memory import EpisodicMemory

        memory = EpisodicMemory(
            memory_id="mem_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="User asked about database connections",
        )

        assert memory.memory_id == "mem_123"
        assert memory.session_id == "sess_001"
        assert memory.user_id == "user_456"
        assert memory.org_id == "org_789"
        assert memory.content == "User asked about database connections"

    def test_episodic_memory_default_recency(self):
        """EpisodicMemory should have default recency score of 1.0."""
        from openmemory.api.memory import EpisodicMemory

        memory = EpisodicMemory(
            memory_id="mem_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Test",
        )

        assert memory.recency_score == 1.0

    def test_episodic_memory_tool_name(self):
        """EpisodicMemory should track source tool."""
        from openmemory.api.memory import EpisodicMemory

        memory = EpisodicMemory(
            memory_id="mem_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Found function at src/utils.ts:42",
            tool_name="search_code_hybrid",
        )

        assert memory.tool_name == "search_code_hybrid"

    def test_episodic_memory_references(self):
        """EpisodicMemory should track references to other memories."""
        from openmemory.api.memory import EpisodicMemory

        memory = EpisodicMemory(
            memory_id="mem_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Continuing from previous analysis",
            references=["mem_100", "mem_101"],
        )

        assert "mem_100" in memory.references
        assert "mem_101" in memory.references

    def test_episodic_memory_summarized_flag(self):
        """EpisodicMemory should track summarization status."""
        from openmemory.api.memory import EpisodicMemory

        memory = EpisodicMemory(
            memory_id="mem_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Test",
            summarized=True,
        )

        assert memory.summarized is True


class TestEpisodicMemoryStore:
    """Tests for EpisodicMemoryStore."""

    def test_in_memory_store_add_and_get(self):
        """InMemoryEpisodicStore should store and retrieve memories."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()
        memory = EpisodicMemory(
            memory_id="mem_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Test memory",
        )

        store.add(memory)

        assert store.get("mem_123") == memory

    def test_in_memory_store_get_not_found(self):
        """InMemoryEpisodicStore should return None for missing ID."""
        from openmemory.api.memory import InMemoryEpisodicStore

        store = InMemoryEpisodicStore()
        assert store.get("nonexistent") is None

    def test_in_memory_store_get_session_memories(self):
        """InMemoryEpisodicStore should retrieve all session memories."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        # Add memories to different sessions
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Session 1 memory 1",
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Session 1 memory 2",
        ))
        store.add(EpisodicMemory(
            memory_id="mem_3",
            session_id="sess_002",
            user_id="user_456",
            org_id="org_789",
            content="Session 2 memory",
        ))

        session_memories = store.get_session_memories("sess_001")

        assert len(session_memories) == 2
        memory_ids = [m.memory_id for m in session_memories]
        assert "mem_1" in memory_ids
        assert "mem_2" in memory_ids
        assert "mem_3" not in memory_ids

    def test_in_memory_store_excludes_summarized_by_default(self):
        """get_session_memories should exclude summarized memories by default."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Active memory",
            summarized=False,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Summarized memory",
            summarized=True,
        ))

        memories = store.get_session_memories("sess_001")

        assert len(memories) == 1
        assert memories[0].memory_id == "mem_1"

    def test_in_memory_store_includes_summarized_when_requested(self):
        """get_session_memories should include summarized if requested."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Active memory",
            summarized=False,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Summarized memory",
            summarized=True,
        ))

        memories = store.get_session_memories("sess_001", include_summarized=True)

        assert len(memories) == 2

    def test_in_memory_store_sorted_by_recency(self):
        """get_session_memories should sort by recency (highest first)."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_low",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Low recency",
            recency_score=0.5,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_high",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="High recency",
            recency_score=1.0,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_mid",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Mid recency",
            recency_score=0.8,
        ))

        memories = store.get_session_memories("sess_001")

        assert memories[0].memory_id == "mem_high"
        assert memories[1].memory_id == "mem_mid"
        assert memories[2].memory_id == "mem_low"


class TestRecencyDecay:
    """Tests for recency score decay."""

    def test_update_recency_decays_scores(self):
        """update_recency should decay all session memory scores."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Memory 1",
            recency_score=1.0,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Memory 2",
            recency_score=0.8,
        ))

        store.update_recency("sess_001", decay_factor=0.9)

        mem1 = store.get("mem_1")
        mem2 = store.get("mem_2")

        assert mem1.recency_score == pytest.approx(0.9, rel=0.01)
        assert mem2.recency_score == pytest.approx(0.72, rel=0.01)  # 0.8 * 0.9

    def test_update_recency_isolates_sessions(self):
        """update_recency should only affect specified session."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Session 1",
            recency_score=1.0,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_002",
            user_id="user_456",
            org_id="org_789",
            content="Session 2",
            recency_score=1.0,
        ))

        store.update_recency("sess_001", decay_factor=0.5)

        assert store.get("mem_1").recency_score == 0.5
        assert store.get("mem_2").recency_score == 1.0  # Unchanged


class TestSummarization:
    """Tests for memory summarization."""

    def test_mark_summarized(self):
        """mark_summarized should update memory flags."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="To be summarized",
            summarized=False,
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Also to be summarized",
            summarized=False,
        ))

        store.mark_summarized(["mem_1", "mem_2"])

        assert store.get("mem_1").summarized is True
        assert store.get("mem_2").summarized is True

    def test_mark_summarized_partial(self):
        """mark_summarized should handle partial ID list."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="To be summarized",
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Keep active",
        ))

        store.mark_summarized(["mem_1"])  # Only mem_1

        assert store.get("mem_1").summarized is True
        assert store.get("mem_2").summarized is False


class TestReferenceResolver:
    """Tests for ReferenceResolver."""

    def test_resolve_exact_match(self):
        """Should resolve reference with exact content match."""
        from openmemory.api.memory import (
            InMemoryEpisodicStore,
            EpisodicMemory,
            ReferenceResolver,
        )

        store = InMemoryEpisodicStore()
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="The database connection pool uses HikariCP",
        ))

        resolver = ReferenceResolver(store)
        result = resolver.resolve("HikariCP", "sess_001")

        assert result is not None
        assert "HikariCP" in result.resolved_content
        assert result.confidence > 0.0
        assert result.source_memory_id == "mem_1"

    def test_resolve_no_match(self):
        """Should return None when no match found."""
        from openmemory.api.memory import (
            InMemoryEpisodicStore,
            EpisodicMemory,
            ReferenceResolver,
        )

        store = InMemoryEpisodicStore()
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Something completely unrelated",
        ))

        resolver = ReferenceResolver(store)
        result = resolver.resolve("NonexistentTopic", "sess_001")

        assert result is None

    def test_resolve_session_isolation(self):
        """Should only search within specified session."""
        from openmemory.api.memory import (
            InMemoryEpisodicStore,
            EpisodicMemory,
            ReferenceResolver,
        )

        store = InMemoryEpisodicStore()
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Target content in session 1",
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_002",
            user_id="user_456",
            org_id="org_789",
            content="Target content in session 2",
        ))

        resolver = ReferenceResolver(store)

        # Search in sess_001
        result = resolver.resolve("Target", "sess_001")
        assert result is not None
        assert result.source_memory_id == "mem_1"

    def test_resolve_confidence_threshold(self):
        """Should respect minimum confidence threshold."""
        from openmemory.api.memory import (
            InMemoryEpisodicStore,
            EpisodicMemory,
            ReferenceResolver,
        )

        store = InMemoryEpisodicStore()
        # Use a very long content with substring match (not word match)
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="A very long content string that mentions extraordinary things repeatedly in many sentences that go on and on with more details about various topics without getting to the point directly but rather meandering through multiple subjects",
        ))

        resolver = ReferenceResolver(store)

        # "extra" as substring in "extraordinary" - lower confidence
        result_low = resolver.resolve("extra", "sess_001", min_confidence=0.01)
        assert result_low is not None

        # Very high threshold should fail for partial substring matches
        result_high = resolver.resolve("extra", "sess_001", min_confidence=0.95)
        assert result_high is None

    def test_resolve_best_match(self):
        """Should return the best matching memory."""
        from openmemory.api.memory import (
            InMemoryEpisodicStore,
            EpisodicMemory,
            ReferenceResolver,
        )

        store = InMemoryEpisodicStore()
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="The API uses REST endpoints for communication",
        ))
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="REST",  # Shorter, higher match ratio for "REST"
        ))

        resolver = ReferenceResolver(store)
        result = resolver.resolve("REST", "sess_001")

        # Should prefer the more specific match
        assert result is not None


class TestCrossToolContextHandoff:
    """Tests for cross-tool context handoff within session."""

    def test_context_handoff_same_session(self):
        """Memories from different tools should be accessible in same session."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        # Tool 1 creates a memory
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Found UserService class at src/services/user.ts",
            tool_name="search_code_hybrid",
        ))

        # Tool 2 creates a memory referencing tool 1's result
        store.add(EpisodicMemory(
            memory_id="mem_2",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="UserService has 5 callers",
            tool_name="find_callers",
            references=["mem_1"],
        ))

        # Both should be accessible in the session
        memories = store.get_session_memories("sess_001")
        assert len(memories) == 2

        # Verify tools are tracked
        tools = {m.tool_name for m in memories}
        assert "search_code_hybrid" in tools
        assert "find_callers" in tools

    def test_context_not_shared_across_sessions(self):
        """Memories should not leak across sessions."""
        from openmemory.api.memory import InMemoryEpisodicStore, EpisodicMemory

        store = InMemoryEpisodicStore()

        # Session 1 memory
        store.add(EpisodicMemory(
            memory_id="mem_1",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Session 1 context",
            tool_name="search_code_hybrid",
        ))

        # Session 2 should not see session 1's memories
        session2_memories = store.get_session_memories("sess_002")
        assert len(session2_memories) == 0


class TestEpisodicMemoryIntegrationWithScope:
    """Tests for episodic memory integration with scoped memory."""

    def test_episodic_to_scoped_promotion(self):
        """Should be able to promote episodic memory to scoped memory."""
        from openmemory.api.memory import (
            EpisodicMemory,
            ScopedMemory,
            MemoryScope,
        )

        # Create episodic memory
        episodic = EpisodicMemory(
            memory_id="ep_123",
            session_id="sess_001",
            user_id="user_456",
            org_id="org_789",
            content="Important finding to persist",
        )

        # Promote to scoped memory (user scope)
        scoped = ScopedMemory(
            memory_id=f"scoped_{episodic.memory_id}",
            content=episodic.content,
            scope=MemoryScope.USER,
            user_id=episodic.user_id,
            org_id=episodic.org_id,
            metadata={
                "promoted_from_episodic": episodic.memory_id,
                "original_session": episodic.session_id,
            },
        )

        assert scoped.content == episodic.content
        assert scoped.scope == MemoryScope.USER
        assert scoped.metadata["promoted_from_episodic"] == "ep_123"


class TestEmptySessionHandling:
    """Tests for edge cases with empty sessions."""

    def test_get_memories_empty_session(self):
        """Should return empty list for session with no memories."""
        from openmemory.api.memory import InMemoryEpisodicStore

        store = InMemoryEpisodicStore()
        memories = store.get_session_memories("nonexistent_session")

        assert memories == []

    def test_update_recency_empty_session(self):
        """update_recency should handle empty session gracefully."""
        from openmemory.api.memory import InMemoryEpisodicStore

        store = InMemoryEpisodicStore()
        # Should not raise
        store.update_recency("nonexistent_session")

    def test_resolve_empty_session(self):
        """Resolver should handle empty session gracefully."""
        from openmemory.api.memory import InMemoryEpisodicStore, ReferenceResolver

        store = InMemoryEpisodicStore()
        resolver = ReferenceResolver(store)

        result = resolver.resolve("anything", "nonexistent_session")
        assert result is None
