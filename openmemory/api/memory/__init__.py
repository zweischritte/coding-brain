"""Phase 3: Scoped Memory + Conversation Memory.

This module implements:
- Scope hierarchy models (session > user > team > project > org > enterprise)
- Multi-scope retrieval with de-duplication
- Episodic memory layer (FR-005)
- SCIM orphan data handling (FR-012)
"""

from .scope import (
    # Core enums
    MemoryScope,
    # Scope models
    ScopedMemory,
    ScopeContext,
    ScopeFilter,
    # Retrieval
    ScopedMemoryStore,
    InMemoryScopedMemoryStore,
    ScopedRetriever,
    # Results
    ScopedRetrievalResult,
)

from .episodic import (
    # Core types
    EpisodicMemory,
    EpisodicMemoryConfig,
    SessionContext,
    # Storage
    EpisodicMemoryStore,
    InMemoryEpisodicStore,
    # Reference resolution
    ReferenceResolver,
    ResolvedReference,
)

from .scim import (
    # User states
    SCIMUserState,
    OrphanDataPolicy,
    # Workflows
    OrphanDataHandler,
    SuspensionRecord,
    DeletionSchedule,
    OwnershipTransfer,
)

__all__ = [
    # Scope
    "MemoryScope",
    "ScopedMemory",
    "ScopeContext",
    "ScopeFilter",
    "ScopedMemoryStore",
    "InMemoryScopedMemoryStore",
    "ScopedRetriever",
    "ScopedRetrievalResult",
    # Episodic
    "EpisodicMemory",
    "EpisodicMemoryConfig",
    "SessionContext",
    "EpisodicMemoryStore",
    "InMemoryEpisodicStore",
    "ReferenceResolver",
    "ResolvedReference",
    # SCIM
    "SCIMUserState",
    "OrphanDataPolicy",
    "OrphanDataHandler",
    "SuspensionRecord",
    "DeletionSchedule",
    "OwnershipTransfer",
]
