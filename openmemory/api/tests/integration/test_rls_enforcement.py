"""
Integration tests for PostgreSQL Row Level Security (RLS) enforcement.

These tests verify that RLS policies correctly isolate tenant data
at the database level.

TDD: These tests are written BEFORE the RLS migration and implementation.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import Memory, User, App, MemoryState


# Test user UUIDs
USER_A_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
USER_B_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


class TestRLSPolicyExists:
    """Tests to verify RLS policies are properly configured."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,  # Enable when RLS migration is applied
        reason="Requires RLS migration to be applied"
    )
    def test_rls_enabled_on_memories_table(self, postgres_test_db: Session):
        """
        Verify that RLS is enabled on the memories table.
        """
        result = postgres_test_db.execute(
            text("""
                SELECT relrowsecurity, relforcerowsecurity
                FROM pg_class
                WHERE relname = 'memories'
            """)
        )
        row = result.fetchone()

        assert row is not None, "memories table not found"
        assert row[0] is True, "RLS should be enabled (relrowsecurity)"

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration to be applied"
    )
    def test_tenant_isolation_policy_exists(self, postgres_test_db: Session):
        """
        Verify that the tenant_isolation policy exists on memories table.
        """
        result = postgres_test_db.execute(
            text("""
                SELECT polname, polcmd, polroles
                FROM pg_policy
                WHERE polrelid = 'memories'::regclass
            """)
        )
        policies = result.fetchall()

        policy_names = [p[0] for p in policies]
        assert "tenant_isolation" in policy_names or \
               any("tenant" in name.lower() for name in policy_names), \
               f"Expected tenant isolation policy, found: {policy_names}"


class TestRLSEnforcement:
    """Tests for RLS enforcement behavior."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_query_without_session_var_returns_empty(
        self,
        rls_enabled_session: Session
    ):
        """
        Queries without app.current_user_id set should return empty results.

        RLS should fail closed (no data) rather than fail open (all data).
        """
        # Query without setting tenant context
        result = rls_enabled_session.execute(
            text("SELECT COUNT(*) FROM memories")
        )
        count = result.scalar()

        assert count == 0, \
            "Queries without session var should return no rows (fail closed)"

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_query_returns_only_user_memories(
        self,
        rls_enabled_session: Session
    ):
        """
        Queries with session var should only return that user's memories.
        """
        from app.database import tenant_session

        # Set tenant context to User A
        with tenant_session(rls_enabled_session, USER_A_ID) as session:
            result = session.execute(
                text("SELECT user_id FROM memories")
            )
            rows = result.fetchall()

        # All returned rows should belong to User A
        for row in rows:
            assert str(row[0]) == str(USER_A_ID), \
                f"RLS should filter to User A only, got {row[0]}"

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_user_a_cannot_see_user_b_memories(
        self,
        rls_enabled_session: Session
    ):
        """
        User A should not be able to see any of User B's memories.
        """
        from app.database import tenant_session

        with tenant_session(rls_enabled_session, USER_A_ID) as session:
            result = session.execute(
                text("SELECT id FROM memories WHERE user_id = :user_b"),
                {"user_b": str(USER_B_ID)}
            )
            rows = result.fetchall()

        assert len(rows) == 0, \
            "User A should not see User B's memories"

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_user_b_cannot_see_user_a_memories(
        self,
        rls_enabled_session: Session
    ):
        """
        User B should not be able to see any of User A's memories.
        """
        from app.database import tenant_session

        with tenant_session(rls_enabled_session, USER_B_ID) as session:
            result = session.execute(
                text("SELECT id FROM memories WHERE user_id = :user_a"),
                {"user_a": str(USER_A_ID)}
            )
            rows = result.fetchall()

        assert len(rows) == 0, \
            "User B should not see User A's memories"

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_rls_prevents_cross_tenant_insert(
        self,
        rls_enabled_session: Session
    ):
        """
        Users should not be able to insert memories for other users.

        Even with a crafted INSERT statement, RLS should enforce that
        the user_id matches the session variable.
        """
        from app.database import tenant_session

        # User A tries to insert a memory for User B
        with tenant_session(rls_enabled_session, USER_A_ID) as session:
            # This should either:
            # 1. Fail with a policy violation
            # 2. Silently insert with User A's ID (policy overrides)
            # We expect option 1 for strict security
            with pytest.raises(Exception):  # Policy violation
                session.execute(
                    text("""
                        INSERT INTO memories (id, user_id, app_id, content, state)
                        VALUES (:id, :user_id, :app_id, :content, :state)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "user_id": str(USER_B_ID),  # Trying to insert as User B
                        "app_id": str(uuid.uuid4()),
                        "content": "Malicious content",
                        "state": "active"
                    }
                )

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_rls_prevents_cross_tenant_update(
        self,
        rls_enabled_session: Session
    ):
        """
        Users should not be able to update other users' memories.
        """
        from app.database import tenant_session

        # Assume User B has a memory with known ID
        user_b_memory_id = uuid.uuid4()

        # User A tries to update User B's memory
        with tenant_session(rls_enabled_session, USER_A_ID) as session:
            result = session.execute(
                text("""
                    UPDATE memories
                    SET content = 'Hacked content'
                    WHERE id = :memory_id
                """),
                {"memory_id": str(user_b_memory_id)}
            )
            # Should affect 0 rows due to RLS filtering
            assert result.rowcount == 0, \
                "Cross-tenant UPDATE should affect 0 rows"

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and test data setup"
    )
    def test_rls_prevents_cross_tenant_delete(
        self,
        rls_enabled_session: Session
    ):
        """
        Users should not be able to delete other users' memories.
        """
        from app.database import tenant_session

        # Assume User B has a memory with known ID
        user_b_memory_id = uuid.uuid4()

        # User A tries to delete User B's memory
        with tenant_session(rls_enabled_session, USER_A_ID) as session:
            result = session.execute(
                text("""
                    DELETE FROM memories
                    WHERE id = :memory_id
                """),
                {"memory_id": str(user_b_memory_id)}
            )
            # Should affect 0 rows due to RLS filtering
            assert result.rowcount == 0, \
                "Cross-tenant DELETE should affect 0 rows"


class TestRLSWithConnectionPooling:
    """Tests for RLS behavior with connection pooling."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and connection pool setup"
    )
    def test_pooled_connections_isolated(self, postgres_test_db: Session):
        """
        Different requests using pooled connections should be isolated.

        Even if the same connection is reused, the tenant context
        should be properly set/reset.
        """
        from app.database import tenant_session

        # Simulate User A's request
        with tenant_session(postgres_test_db, USER_A_ID) as session:
            result_a = session.execute(
                text("SELECT current_setting('app.current_user_id', true)")
            )
            assert result_a.scalar() == str(USER_A_ID)

        # After exiting, verify context is cleared
        result_cleared = postgres_test_db.execute(
            text("SELECT current_setting('app.current_user_id', true)")
        )
        cleared_value = result_cleared.scalar()
        assert cleared_value is None or cleared_value == "", \
            "Context should be cleared between requests"

        # Simulate User B's request on same connection
        with tenant_session(postgres_test_db, USER_B_ID) as session:
            result_b = session.execute(
                text("SELECT current_setting('app.current_user_id', true)")
            )
            assert result_b.scalar() == str(USER_B_ID)

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration and connection pool setup"
    )
    def test_context_not_leaked_on_error(self, postgres_test_db: Session):
        """
        Tenant context should be cleared even if an error occurs.
        """
        from app.database import tenant_session

        try:
            with tenant_session(postgres_test_db, USER_A_ID):
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        # Verify context is cleared despite the error
        result = postgres_test_db.execute(
            text("SELECT current_setting('app.current_user_id', true)")
        )
        value = result.scalar()
        assert value is None or value == "", \
            "Context should be cleared even after error"


class TestRLSEdgeCases:
    """Tests for RLS edge cases and security boundaries."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration"
    )
    def test_malformed_uuid_rejected(self, postgres_test_db: Session):
        """
        Malformed UUIDs in session variable should be rejected.
        """
        from app.database import tenant_session

        with pytest.raises((ValueError, TypeError)):
            with tenant_session(postgres_test_db, "'; DROP TABLE memories; --"):
                pass

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration"
    )
    def test_empty_user_id_rejected(self, postgres_test_db: Session):
        """
        Empty user_id should be rejected.
        """
        from app.database import tenant_session

        with pytest.raises((ValueError, TypeError)):
            with tenant_session(postgres_test_db, ""):
                pass

    @pytest.mark.integration
    @pytest.mark.skipif(
        True,
        reason="Requires RLS migration"
    )
    def test_none_user_id_rejected(self, postgres_test_db: Session):
        """
        None user_id should be rejected.
        """
        from app.database import tenant_session

        with pytest.raises((ValueError, TypeError)):
            with tenant_session(postgres_test_db, None):
                pass
