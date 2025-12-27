"""
Abstract base class for all data plane stores.

This module defines the BaseStore interface that all store implementations
must satisfy. The interface provides consistent CRUD semantics with
automatic tenant isolation.

Implementations may use:
- PostgreSQL Row Level Security (RLS) for database-level isolation
- Application-level filtering for non-RLS databases
- External service APIs with tenant context injection
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Any
from uuid import UUID

# Type variable for entity type
T = TypeVar('T')


class BaseStore(ABC, Generic[T]):
    """
    Abstract base class for all tenant-scoped stores.

    All store implementations must:
    1. Accept a tenant identifier (user_id or org_id) at construction
    2. Filter all operations by the tenant identifier
    3. Return None/empty for cross-tenant access (not errors)

    The store interface uses soft delete semantics where applicable.

    Type Parameters:
        T: The entity type this store manages

    Example:
        class MemoryStore(BaseStore[Memory]):
            def get(self, id: UUID) -> Optional[Memory]:
                # Implementation with tenant filtering
                ...
    """

    @abstractmethod
    def get(self, id: UUID) -> Optional[T]:
        """
        Retrieve an entity by ID within the current tenant scope.

        Args:
            id: The unique identifier of the entity

        Returns:
            The entity if found and owned by current tenant, None otherwise

        Note:
            Cross-tenant access attempts return None to prevent
            information leakage via error messages.
        """
        ...

    @abstractmethod
    def list(self, **filters: Any) -> List[T]:
        """
        List entities within the current tenant scope.

        Args:
            **filters: Optional filters (e.g., state, app_id)

        Returns:
            List of matching entities owned by current tenant.
            Returns empty list if no matches.
        """
        ...

    @abstractmethod
    def create(self, entity: T) -> T:
        """
        Create a new entity within the current tenant scope.

        The entity's tenant identifier (user_id/org_id) will be
        set to the current tenant, regardless of what was passed.

        Args:
            entity: The entity to create

        Returns:
            The created entity with ID assigned

        Raises:
            ValueError: If entity is invalid
        """
        ...

    @abstractmethod
    def update(self, entity: T) -> Optional[T]:
        """
        Update an existing entity within the current tenant scope.

        Args:
            entity: The entity with updated values

        Returns:
            The updated entity, or None if not found/not owned

        Note:
            Cross-tenant update attempts return None and do not modify data.
        """
        ...

    @abstractmethod
    def delete(self, id: UUID) -> bool:
        """
        Delete (or soft-delete) an entity by ID.

        Args:
            id: The unique identifier of the entity

        Returns:
            True if deleted, False if not found or not owned

        Note:
            Most implementations use soft delete (setting state to deleted).
            Cross-tenant delete attempts return False and do not modify data.
        """
        ...
