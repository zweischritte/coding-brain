"""
Scoped memory store with tenant isolation.

This module provides the ScopedMemoryStore class for CRUD operations
on memories with automatic tenant isolation. On PostgreSQL with RLS,
isolation is enforced at the database level. On other databases,
application-level filtering is used.
"""

from typing import List, Optional, Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Memory, MemoryState
from app.stores.base import BaseStore


class ScopedMemoryStore(BaseStore[Memory]):
    """
    Memory store with tenant isolation.

    This store provides CRUD operations for Memory entities, automatically
    filtering all operations by the current user's ID.

    Isolation behavior:
    - get(): Returns None if memory belongs to different user
    - list(): Only returns memories owned by current user
    - create(): Forces user_id to current user
    - update(): Returns None if memory belongs to different user
    - delete(): Returns False if memory belongs to different user

    Args:
        db: SQLAlchemy session
        user_id: UUID of the current user (tenant)

    Example:
        store = ScopedMemoryStore(db, principal.user_id)
        memory = store.get(memory_id)  # Only returns if owned by user
    """

    def __init__(self, db: Session, user_id: UUID):
        """
        Initialize the memory store.

        Args:
            db: SQLAlchemy session to use for database operations
            user_id: UUID of the current user for tenant isolation
        """
        self._db = db
        self._user_id = user_id

    def get(self, id: UUID) -> Optional[Memory]:
        """
        Get a memory by ID if owned by current user.

        Args:
            id: The memory's unique identifier

        Returns:
            The Memory if found and owned by current user, None otherwise
        """
        return (
            self._db.query(Memory)
            .filter(Memory.id == id)
            .filter(Memory.user_id == self._user_id)
            .first()
        )

    def list(
        self,
        state: Optional[MemoryState] = None,
        app_id: Optional[UUID] = None,
        **filters: Any
    ) -> List[Memory]:
        """
        List memories owned by current user with optional filters.

        Args:
            state: Optional filter by memory state
            app_id: Optional filter by application ID
            **filters: Additional filters (reserved for future use)

        Returns:
            List of memories matching filters, owned by current user
        """
        query = self._db.query(Memory).filter(Memory.user_id == self._user_id)

        if state is not None:
            query = query.filter(Memory.state == state)

        if app_id is not None:
            query = query.filter(Memory.app_id == app_id)

        return query.all()

    def create(self, entity: Memory) -> Memory:
        """
        Create a new memory.

        The memory's user_id will be set to the current user's ID,
        regardless of what was passed in the entity.

        Args:
            entity: Memory to create

        Returns:
            The created Memory with ID assigned
        """
        # Force user_id to current user for security
        entity.user_id = self._user_id

        self._db.add(entity)
        self._db.commit()
        self._db.refresh(entity)

        return entity

    def update(self, entity: Memory) -> Optional[Memory]:
        """
        Update an existing memory if owned by current user.

        Args:
            entity: Memory with updated values

        Returns:
            The updated Memory, or None if not found/not owned
        """
        # Verify ownership before updating
        existing = self.get(entity.id)
        if existing is None:
            return None

        # Apply updates
        existing.content = entity.content
        existing.state = entity.state
        existing.metadata_ = entity.metadata_
        existing.vault = entity.vault
        existing.layer = entity.layer
        existing.axis_vector = entity.axis_vector

        self._db.commit()
        self._db.refresh(existing)

        return existing

    def delete(self, id: UUID) -> bool:
        """
        Soft-delete a memory by setting state to deleted.

        Args:
            id: The memory's unique identifier

        Returns:
            True if deleted, False if not found or not owned
        """
        memory = self.get(id)
        if memory is None:
            return False

        memory.state = MemoryState.deleted
        self._db.commit()

        return True
