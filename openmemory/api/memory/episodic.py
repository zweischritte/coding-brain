"""Episodic memory layer for Phase 3 (FR-005).

Per v9 plan section 4.3:
- Session scope TTL default: 24h (configurable)
- Episodic memory stored per session and summarized over time
- Cross-tool context handoff within session scope
- Separate from persistent scoped memory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory behavior."""

    session_ttl: timedelta = field(default_factory=lambda: timedelta(hours=24))
    summarization_threshold: int = 10  # Summarize after N memories
    max_active_memories: int = 50  # Max before forced summarization


@dataclass
class SessionContext:
    """Context for an episodic memory session."""

    session_id: str
    user_id: str
    org_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set expiration if not provided."""
        if self.expires_at is None:
            self.expires_at = self.started_at + timedelta(hours=24)

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class EpisodicMemory:
    """An episodic memory within a session.

    Episodic memories are temporary, session-scoped memories that:
    - Track recency for prioritization
    - Support reference resolution
    - Get summarized over time
    """

    memory_id: str
    session_id: str
    user_id: str
    org_id: str
    content: str
    recency_score: float = 1.0  # Decays over time
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str | None = None  # Tool that created this memory
    references: list[str] = field(default_factory=list)  # Referenced memory IDs
    summarized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedReference:
    """A resolved reference from episodic memory."""

    reference_id: str
    resolved_content: str
    confidence: float
    source_memory_id: str


class EpisodicMemoryStore(ABC):
    """Abstract interface for episodic memory storage."""

    @abstractmethod
    def add(self, memory: EpisodicMemory) -> None:
        """Add an episodic memory."""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> EpisodicMemory | None:
        """Get an episodic memory by ID."""
        pass

    @abstractmethod
    def get_session_memories(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[EpisodicMemory]:
        """Get all memories for a session."""
        pass

    @abstractmethod
    def update_recency(
        self,
        session_id: str,
        decay_factor: float = 0.9,
    ) -> None:
        """Decay recency scores for all session memories."""
        pass

    @abstractmethod
    def mark_summarized(self, memory_ids: list[str]) -> None:
        """Mark memories as summarized."""
        pass


class InMemoryEpisodicStore(EpisodicMemoryStore):
    """In-memory implementation for testing."""

    def __init__(self):
        self._memories: dict[str, EpisodicMemory] = {}
        self._by_session: dict[str, list[str]] = {}

    def add(self, memory: EpisodicMemory) -> None:
        """Add an episodic memory."""
        self._memories[memory.memory_id] = memory
        if memory.session_id not in self._by_session:
            self._by_session[memory.session_id] = []
        self._by_session[memory.session_id].append(memory.memory_id)

    def get(self, memory_id: str) -> EpisodicMemory | None:
        """Get an episodic memory by ID."""
        return self._memories.get(memory_id)

    def get_session_memories(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[EpisodicMemory]:
        """Get all memories for a session."""
        memory_ids = self._by_session.get(session_id, [])
        memories = [self._memories[mid] for mid in memory_ids if mid in self._memories]

        if not include_summarized:
            memories = [m for m in memories if not m.summarized]

        # Sort by recency score (highest first)
        memories.sort(key=lambda m: m.recency_score, reverse=True)
        return memories

    def update_recency(
        self,
        session_id: str,
        decay_factor: float = 0.9,
    ) -> None:
        """Decay recency scores for all session memories."""
        for memory_id in self._by_session.get(session_id, []):
            if memory_id in self._memories:
                memory = self._memories[memory_id]
                memory.recency_score *= decay_factor

    def mark_summarized(self, memory_ids: list[str]) -> None:
        """Mark memories as summarized."""
        for memory_id in memory_ids:
            if memory_id in self._memories:
                self._memories[memory_id].summarized = True


class ReferenceResolver:
    """Resolves references from episodic memory to persistent memory."""

    def __init__(self, episodic_store: EpisodicMemoryStore):
        self.store = episodic_store

    def resolve(
        self,
        reference: str,
        session_id: str,
        min_confidence: float = 0.5,
    ) -> ResolvedReference | None:
        """Resolve a reference within a session context.

        Args:
            reference: The reference string to resolve
            session_id: The session context
            min_confidence: Minimum confidence threshold

        Returns:
            ResolvedReference if found with sufficient confidence
        """
        # Get all session memories
        memories = self.store.get_session_memories(session_id)

        # Simple matching for now - could be enhanced with embeddings
        best_match: EpisodicMemory | None = None
        best_score = 0.0

        for memory in memories:
            # Check if reference matches any part of content
            ref_lower = reference.lower()
            content_lower = memory.content.lower()

            if ref_lower in content_lower:
                # Score based on how much of the content the reference covers
                # and how specific the match is
                coverage = len(reference) / len(memory.content)

                # Boost score for exact word matches (not partial)
                words = content_lower.split()
                if ref_lower in words:
                    coverage = max(coverage, 0.7)  # Minimum 70% for word match

                # Boost for shorter content (more specific match)
                specificity_boost = min(1.0, 50 / len(memory.content))
                score = min(1.0, coverage + specificity_boost * 0.3)

                if score > best_score:
                    best_score = score
                    best_match = memory

        if best_match is None or best_score < min_confidence:
            return None

        return ResolvedReference(
            reference_id=reference,
            resolved_content=best_match.content,
            confidence=best_score,
            source_memory_id=best_match.memory_id,
        )
