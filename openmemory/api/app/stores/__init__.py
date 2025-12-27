"""
Multi-tenant data plane stores.

This package provides persistent store implementations with tenant isolation
via PostgreSQL Row Level Security (RLS) or application-level filtering.

Stores:
- ScopedMemoryStore: CRUD for memories with user-level isolation (PostgreSQL)
- FeedbackStore: Feedback events with retention queries (PostgreSQL)
- ExperimentStore: A/B experiments with status history (PostgreSQL)
- ValkeyEpisodicStore: Session-scoped ephemeral memory with TTL (Valkey)

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
from app.stores.episodic_store import ValkeyEpisodicStore, get_valkey_episodic_store
from app.stores.qdrant_store import TenantQdrantStore, get_tenant_qdrant_store
from app.stores.opensearch_store import TenantOpenSearchStore, get_tenant_opensearch_store

__all__ = [
    "BaseStore",
    "ScopedMemoryStore",
    "PostgresFeedbackStore",
    "FeedbackEventModel",
    "PostgresExperimentStore",
    "ExperimentModel",
    "ExperimentStatusHistoryModel",
    "VariantAssignmentModel",
    "ValkeyEpisodicStore",
    "get_valkey_episodic_store",
    "TenantQdrantStore",
    "get_tenant_qdrant_store",
    "TenantOpenSearchStore",
    "get_tenant_opensearch_store",
]
