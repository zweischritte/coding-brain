"""Scope hierarchy and multi-scope retrieval for Phase 3.

Per v9 plan section 4.2:
- Scope hierarchy: session > user > team > project > org > enterprise
- De-duplicate by content hash; keep highest-precedence result
- Multi-team users include all team_ids unless request narrows scope
- geo_scope inherits from org unless explicitly overridden
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from openmemory.api.security.rbac import Principal


class MemoryScope(str, Enum):
    """Scope levels for memory with precedence ordering.

    Higher precedence = more specific/local scope.
    """

    SESSION = "session"
    USER = "user"
    TEAM = "team"
    PROJECT = "project"
    ORG = "org"
    ENTERPRISE = "enterprise"

    @property
    def precedence(self) -> int:
        """Return precedence value (higher = more specific)."""
        precedence_map = {
            MemoryScope.SESSION: 60,
            MemoryScope.USER: 50,
            MemoryScope.TEAM: 40,
            MemoryScope.PROJECT: 30,
            MemoryScope.ORG: 20,
            MemoryScope.ENTERPRISE: 10,
        }
        return precedence_map[self]

    @classmethod
    def from_string(cls, value: str) -> MemoryScope:
        """Parse scope from string (case-insensitive)."""
        normalized = value.lower()
        for scope in cls:
            if scope.value == normalized:
                return scope
        raise ValueError(f"Unknown scope: {value}")


@dataclass
class ScopedMemory:
    """A memory item with explicit scope fields.

    All items carry explicit scope fields per v9 plan section 4.2.
    """

    memory_id: str
    content: str
    scope: MemoryScope
    user_id: str
    org_id: str
    content_hash: str | None = None
    session_id: str | None = None
    team_id: str | None = None
    project_id: str | None = None
    enterprise_id: str | None = None
    geo_scope: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-compute content hash if not provided."""
        if self.content_hash is None:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class ScopeContext:
    """Defines the accessor's identity and accessible scopes.

    Represents the context from which retrieval is being performed,
    determining which scopes the user can access.
    """

    user_id: str
    org_id: str
    enterprise_id: str | None = None
    session_id: str | None = None
    team_ids: list[str] = field(default_factory=list)
    project_ids: list[str] = field(default_factory=list)
    geo_scope: str | None = None
    org_geo_scope: str | None = None  # Org's default geo scope for inheritance

    @classmethod
    def from_principal(cls, principal: Principal) -> ScopeContext:
        """Create ScopeContext from RBAC Principal."""
        return cls(
            user_id=principal.user_id,
            org_id=principal.org_id,
            enterprise_id=principal.enterprise_id,
            session_id=principal.session_id,
            team_ids=list(principal.team_ids),
            project_ids=list(principal.project_ids),
            geo_scope=principal.geo_scope,
        )

    @property
    def effective_geo_scope(self) -> str | None:
        """Get effective geo scope with inheritance from org."""
        # Explicit geo_scope overrides org default
        return self.geo_scope if self.geo_scope else self.org_geo_scope

    def get_accessible_scopes(self) -> set[MemoryScope]:
        """Compute all scope levels this context can access."""
        scopes = set()

        # Session scope if session_id present
        if self.session_id:
            scopes.add(MemoryScope.SESSION)

        # User scope always available
        scopes.add(MemoryScope.USER)

        # Team scope if user belongs to teams
        if self.team_ids:
            scopes.add(MemoryScope.TEAM)

        # Project scope if user belongs to projects
        if self.project_ids:
            scopes.add(MemoryScope.PROJECT)

        # Org scope always available (user must be in an org)
        scopes.add(MemoryScope.ORG)

        # Enterprise scope if enterprise_id present
        if self.enterprise_id:
            scopes.add(MemoryScope.ENTERPRISE)

        return scopes


@dataclass
class ScopeFilter:
    """Query filter for scoped memory retrieval."""

    scopes: list[MemoryScope] = field(default_factory=list)
    team_ids: list[str] | None = None
    project_ids: list[str] | None = None
    session_id: str | None = None

    @classmethod
    def from_context(
        cls,
        ctx: ScopeContext,
        scopes: list[MemoryScope] | None = None,
        team_ids: list[str] | None = None,
        project_ids: list[str] | None = None,
    ) -> ScopeFilter:
        """Create filter from context, optionally narrowing scope.

        Args:
            ctx: The scope context
            scopes: Specific scopes to filter to (defaults to all accessible)
            team_ids: Specific teams to filter to (defaults to all user's teams)
            project_ids: Specific projects to filter to
        """
        if scopes is None:
            scopes = list(ctx.get_accessible_scopes())

        return cls(
            scopes=scopes,
            team_ids=team_ids if team_ids is not None else ctx.team_ids,
            project_ids=project_ids if project_ids is not None else ctx.project_ids,
            session_id=ctx.session_id,
        )


@dataclass
class ScopedRetrievalResult:
    """Result of scoped memory retrieval with metadata."""

    memories: list[ScopedMemory]
    total_before_dedup: int
    total_after_dedup: int
    dedup_by_scope: dict[MemoryScope, int] = field(default_factory=dict)


def memory_matches_filter(
    memory: ScopedMemory,
    ctx: ScopeContext,
    filter: ScopeFilter,
) -> bool:
    """Check if a memory matches the filter for the given context.

    Implements scope-based access control:
    - SESSION: must match session_id
    - USER: must match user_id
    - TEAM: must be in user's team_ids
    - PROJECT: must be in user's project_ids
    - ORG: must match org_id
    - ENTERPRISE: must match enterprise_id
    """
    # Must match org
    if memory.org_id != ctx.org_id:
        return False

    # Scope must be in filter
    if memory.scope not in filter.scopes:
        return False

    # Scope-specific checks
    if memory.scope == MemoryScope.SESSION:
        # Session memories only visible to same session
        if memory.session_id != ctx.session_id:
            return False

    elif memory.scope == MemoryScope.USER:
        # User memories only visible to same user
        if memory.user_id != ctx.user_id:
            return False

    elif memory.scope == MemoryScope.TEAM:
        # Team memories visible to team members
        effective_teams = filter.team_ids if filter.team_ids else ctx.team_ids
        if memory.team_id not in effective_teams:
            return False

    elif memory.scope == MemoryScope.PROJECT:
        # Project memories visible to project members
        effective_projects = filter.project_ids if filter.project_ids else ctx.project_ids
        if memory.project_id not in effective_projects:
            return False

    elif memory.scope == MemoryScope.ENTERPRISE:
        # Enterprise memories visible to enterprise members
        if memory.enterprise_id != ctx.enterprise_id:
            return False

    # ORG scope: org_id match already checked above

    return True


class ScopedMemoryStore(ABC):
    """Abstract interface for scoped memory storage."""

    @abstractmethod
    def add(self, memory: ScopedMemory) -> None:
        """Add a memory to the store."""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> ScopedMemory | None:
        """Get a memory by ID."""
        pass

    @abstractmethod
    def update(self, memory: ScopedMemory) -> None:
        """Update an existing memory."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        pass

    @abstractmethod
    def query(
        self,
        ctx: ScopeContext,
        filter: ScopeFilter,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScopedMemory]:
        """Query memories matching the filter."""
        pass


class InMemoryScopedMemoryStore(ScopedMemoryStore):
    """In-memory implementation of ScopedMemoryStore for testing."""

    def __init__(self):
        self._memories: dict[str, ScopedMemory] = {}

    def add(self, memory: ScopedMemory) -> None:
        """Add a memory to the store."""
        self._memories[memory.memory_id] = memory

    def get(self, memory_id: str) -> ScopedMemory | None:
        """Get a memory by ID."""
        return self._memories.get(memory_id)

    def update(self, memory: ScopedMemory) -> None:
        """Update an existing memory."""
        if memory.memory_id in self._memories:
            self._memories[memory.memory_id] = memory

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._memories.pop(memory_id, None)

    def query(
        self,
        ctx: ScopeContext,
        filter: ScopeFilter,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[ScopedMemory]:
        """Query memories matching the filter."""
        results = [
            m for m in self._memories.values()
            if memory_matches_filter(m, ctx, filter)
        ]

        # Sort by created_at descending (newest first)
        results.sort(key=lambda m: m.created_at, reverse=True)

        # Apply pagination
        if offset:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]

        return results


class ScopedRetriever:
    """Retriever with scope-based de-duplication.

    Implements precedence-based de-duplication:
    - Union of all permitted scopes
    - De-duplicate by content hash
    - Keep highest-precedence result
    """

    def __init__(self, store: ScopedMemoryStore):
        self.store = store

    def retrieve(
        self,
        ctx: ScopeContext,
        filter: ScopeFilter | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> ScopedRetrievalResult:
        """Retrieve memories with scope-based de-duplication.

        Args:
            ctx: The scope context
            filter: Optional filter (defaults to all accessible scopes)
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            ScopedRetrievalResult with de-duplicated memories
        """
        if filter is None:
            filter = ScopeFilter.from_context(ctx)

        # Get all matching memories (before de-dup)
        all_memories = self.store.query(ctx, filter)
        total_before = len(all_memories)

        # De-duplicate by content hash, keeping highest precedence
        seen_hashes: dict[str, ScopedMemory] = {}
        dedup_counts: dict[MemoryScope, int] = {}

        for memory in all_memories:
            existing = seen_hashes.get(memory.content_hash)
            if existing is None:
                seen_hashes[memory.content_hash] = memory
            elif memory.scope.precedence > existing.scope.precedence:
                # Replace with higher-precedence version
                dedup_counts[memory.scope] = dedup_counts.get(memory.scope, 0) + 1
                seen_hashes[memory.content_hash] = memory

        deduplicated = list(seen_hashes.values())

        # Sort by scope precedence (highest first), then by created_at
        deduplicated.sort(
            key=lambda m: (m.scope.precedence, m.created_at.timestamp()),
            reverse=True,
        )

        total_after = len(deduplicated)

        # Apply pagination after de-dup
        if offset:
            deduplicated = deduplicated[offset:]
        if limit is not None:
            deduplicated = deduplicated[:limit]

        return ScopedRetrievalResult(
            memories=deduplicated,
            total_before_dedup=total_before,
            total_after_dedup=total_after,
            dedup_by_scope=dedup_counts,
        )


def backfill_legacy_memory(
    legacy: dict[str, Any],
    default_org_id: str | None = None,
    org_lookup: Callable[[str], str] | None = None,
) -> ScopedMemory:
    """Backfill a legacy memory with scope fields.

    Args:
        legacy: Legacy memory dict (memory_id, content, user_id, etc.)
        default_org_id: Default org_id to use if none can be determined
        org_lookup: Function to look up org_id from user_id

    Returns:
        ScopedMemory with backfilled scope fields
    """
    user_id = legacy.get("user_id", "")

    # Determine org_id
    org_id = legacy.get("org_id")
    if org_id is None and org_lookup is not None:
        org_id = org_lookup(user_id)
    if org_id is None:
        org_id = default_org_id or ""

    return ScopedMemory(
        memory_id=legacy.get("memory_id", ""),
        content=legacy.get("content", ""),
        scope=MemoryScope.USER,  # Default scope for legacy memories
        user_id=user_id,
        org_id=org_id,
        content_hash=legacy.get("content_hash"),
        session_id=legacy.get("session_id"),
        team_id=legacy.get("team_id"),
        project_id=legacy.get("project_id"),
        enterprise_id=legacy.get("enterprise_id"),
        metadata=legacy.get("metadata", {}),
    )
