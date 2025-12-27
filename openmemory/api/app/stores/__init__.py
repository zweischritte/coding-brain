"""
Multi-tenant data plane stores.

This package provides persistent store implementations with tenant isolation
via PostgreSQL Row Level Security (RLS) or application-level filtering.

Stores:
- ScopedMemoryStore: CRUD for memories with user-level isolation
- FeedbackStore: Feedback events with retention queries
- ExperimentStore: A/B experiments with status history
- EpisodicMemoryStore: Session-scoped ephemeral memory (Valkey)

All stores inherit from BaseStore ABC and enforce tenant isolation.
"""

from app.stores.base import BaseStore
from app.stores.memory_store import ScopedMemoryStore
from app.stores.feedback_store import PostgresFeedbackStore, FeedbackEventModel
from app.stores.experiment_store import (
    PostgresExperimentStore,
    ExperimentModel,
    ExperimentStatusHistoryModel,
    VariantAssignmentModel,
)

__all__ = [
    "BaseStore",
    "ScopedMemoryStore",
    "PostgresFeedbackStore",
    "FeedbackEventModel",
    "PostgresExperimentStore",
    "ExperimentModel",
    "ExperimentStatusHistoryModel",
    "VariantAssignmentModel",
]
