"""
Tests for ScopedMemoryStore.

The ScopedMemoryStore provides CRUD operations for memories with
automatic tenant isolation via RLS or application-level filtering.

TDD: These tests are written BEFORE the implementation.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session

from app.models import Memory, MemoryState


# Test UUIDs
USER_A_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_B_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
APP_A_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
MEMORY_A_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
MEMORY_B_ID = uuid.UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")


class TestScopedMemoryStoreGet:
    """Test ScopedMemoryStore.get() method."""

    def test_get_returns_memory_for_owner(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_app_a,
        sample_memory_a
    ):
        """
        get() should return a memory when called by the owner.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        result = store.get(sample_memory_a.id)

        assert result is not None
        assert result.id == sample_memory_a.id
        assert result.content == sample_memory_a.content

    def test_get_returns_none_for_other_user(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_user_b,
        sample_memory_a
    ):
        """
        get() should return None when called by a different user.

        This is the tenant isolation behavior - cross-tenant access
        returns None, not an error (to prevent information leakage).
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_b.id)
        result = store.get(sample_memory_a.id)

        assert result is None

    def test_get_returns_none_for_nonexistent_id(
        self,
        sqlite_test_db: Session,
        sample_user_a
    ):
        """
        get() should return None for non-existent memory IDs.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        result = store.get(uuid.uuid4())

        assert result is None


class TestScopedMemoryStoreList:
    """Test ScopedMemoryStore.list() method."""

    def test_list_returns_only_user_memories(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_user_b,
        sample_memory_a,
        sample_memory_b
    ):
        """
        list() should only return memories owned by the current user.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        results = store.list()

        assert len(results) >= 1
        for memory in results:
            assert memory.user_id == sample_user_a.id

    def test_list_returns_empty_for_user_with_no_memories(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_user_b
    ):
        """
        list() should return empty list when user has no memories.
        """
        from app.stores.memory_store import ScopedMemoryStore

        # User B has no memories in this test
        store = ScopedMemoryStore(sqlite_test_db, sample_user_b.id)
        results = store.list()

        assert results == []

    def test_list_filters_by_state(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_app_a
    ):
        """
        list(state=...) should filter by memory state.
        """
        from app.stores.memory_store import ScopedMemoryStore

        # Create additional memories with different states
        active_memory = Memory(
            id=uuid.uuid4(),
            user_id=sample_user_a.id,
            app_id=sample_app_a.id,
            content="Active memory",
            state=MemoryState.active
        )
        archived_memory = Memory(
            id=uuid.uuid4(),
            user_id=sample_user_a.id,
            app_id=sample_app_a.id,
            content="Archived memory",
            state=MemoryState.archived
        )
        sqlite_test_db.add_all([active_memory, archived_memory])
        sqlite_test_db.commit()

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)

        active_results = store.list(state=MemoryState.active)
        archived_results = store.list(state=MemoryState.archived)

        assert all(m.state == MemoryState.active for m in active_results)
        assert all(m.state == MemoryState.archived for m in archived_results)

    def test_list_filters_by_app_id(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_app_a,
        sample_memory_a
    ):
        """
        list(app_id=...) should filter by application.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        results = store.list(app_id=sample_app_a.id)

        assert len(results) >= 1
        for memory in results:
            assert memory.app_id == sample_app_a.id


class TestScopedMemoryStoreCreate:
    """Test ScopedMemoryStore.create() method."""

    def test_create_persists_memory(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_app_a
    ):
        """
        create() should persist a new memory with an ID.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)

        new_memory = Memory(
            user_id=sample_user_a.id,
            app_id=sample_app_a.id,
            content="New test memory",
            state=MemoryState.active
        )
        result = store.create(new_memory)

        assert result.id is not None
        assert result.content == "New test memory"

        # Verify it's in the database
        fetched = sqlite_test_db.query(Memory).filter(Memory.id == result.id).first()
        assert fetched is not None

    def test_create_sets_user_id_to_current_user(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_app_a
    ):
        """
        create() should ensure the user_id matches the current user.

        Even if a different user_id is provided, the store should
        use the authenticated user's ID.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)

        # Try to create memory with different user_id
        new_memory = Memory(
            user_id=USER_B_ID,  # Different user
            app_id=sample_app_a.id,
            content="Attempted cross-tenant create",
            state=MemoryState.active
        )
        result = store.create(new_memory)

        # Should be created with the current user's ID
        assert result.user_id == sample_user_a.id


class TestScopedMemoryStoreUpdate:
    """Test ScopedMemoryStore.update() method."""

    def test_update_modifies_memory(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_memory_a
    ):
        """
        update() should modify an existing memory.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)

        sample_memory_a.content = "Updated content"
        result = store.update(sample_memory_a)

        assert result.content == "Updated content"

        # Verify in database
        sqlite_test_db.refresh(sample_memory_a)
        assert sample_memory_a.content == "Updated content"

    def test_update_returns_none_for_other_user_memory(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_user_b,
        sample_memory_a
    ):
        """
        update() should return None when trying to update another user's memory.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_b.id)

        sample_memory_a.content = "Hacked content"
        result = store.update(sample_memory_a)

        assert result is None

        # Verify original content unchanged
        sqlite_test_db.refresh(sample_memory_a)
        assert sample_memory_a.content != "Hacked content"


class TestScopedMemoryStoreDelete:
    """Test ScopedMemoryStore.delete() method."""

    def test_delete_soft_deletes_memory(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_memory_a
    ):
        """
        delete() should set state to deleted (soft delete).
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        result = store.delete(sample_memory_a.id)

        assert result is True

        # Verify state changed
        sqlite_test_db.refresh(sample_memory_a)
        assert sample_memory_a.state == MemoryState.deleted

    def test_delete_returns_false_for_nonexistent(
        self,
        sqlite_test_db: Session,
        sample_user_a
    ):
        """
        delete() should return False for non-existent memory.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        result = store.delete(uuid.uuid4())

        assert result is False

    def test_delete_returns_false_for_other_user_memory(
        self,
        sqlite_test_db: Session,
        sample_user_a,
        sample_user_b,
        sample_memory_a
    ):
        """
        delete() should return False when trying to delete another user's memory.
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_b.id)
        result = store.delete(sample_memory_a.id)

        assert result is False

        # Verify not deleted
        sqlite_test_db.refresh(sample_memory_a)
        assert sample_memory_a.state == MemoryState.active


class TestBaseStoreContract:
    """Test that ScopedMemoryStore satisfies the BaseStore interface."""

    def test_store_has_all_required_methods(self):
        """
        ScopedMemoryStore should implement all BaseStore abstract methods.
        """
        from app.stores.memory_store import ScopedMemoryStore
        from app.stores.base import BaseStore

        # Verify ScopedMemoryStore is a subclass of BaseStore
        assert issubclass(ScopedMemoryStore, BaseStore)

        # Verify all abstract methods are implemented
        required_methods = ['get', 'list', 'create', 'update', 'delete']
        for method in required_methods:
            assert hasattr(ScopedMemoryStore, method)
            assert callable(getattr(ScopedMemoryStore, method))

    def test_store_can_be_instantiated(
        self,
        sqlite_test_db: Session,
        sample_user_a
    ):
        """
        ScopedMemoryStore should be instantiable (not abstract).
        """
        from app.stores.memory_store import ScopedMemoryStore

        store = ScopedMemoryStore(sqlite_test_db, sample_user_a.id)
        assert store is not None
