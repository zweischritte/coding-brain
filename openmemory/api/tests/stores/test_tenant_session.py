"""
Tests for tenant_session() context manager.

The tenant_session context manager sets PostgreSQL session variables
for Row Level Security (RLS) enforcement.

TDD: These tests are written BEFORE the implementation.
"""
import uuid
from unittest.mock import MagicMock, call, patch

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session


class TestTenantSessionContextManager:
    """Test suite for tenant_session() context manager."""

    def test_tenant_session_sets_user_id_variable(self, mock_session: MagicMock):
        """
        tenant_session() should set app.current_user_id session variable.

        The PostgreSQL session variable is used by RLS policies to
        filter rows by the current tenant.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with tenant_session(mock_session, user_id):
            pass

        # Verify execute was called at least twice (SET and RESET)
        assert mock_session.execute.call_count >= 2, \
            f"Expected at least 2 execute calls, got {mock_session.execute.call_count}"

        # Get the first call (should be SET)
        first_call = mock_session.execute.call_args_list[0]
        first_stmt = first_call[0][0]  # First positional arg is the statement

        # The statement should contain SET and current_user_id
        stmt_text = str(first_stmt)
        assert "SET" in stmt_text, f"First call should be SET, got: {stmt_text}"
        assert "current_user_id" in stmt_text, f"Should set current_user_id, got: {stmt_text}"

    def test_tenant_session_resets_variable_on_normal_exit(self, mock_session: MagicMock):
        """
        tenant_session() should reset the session variable on normal exit.

        This prevents tenant context leaking to subsequent queries
        in connection pools.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with tenant_session(mock_session, user_id):
            pass

        # Verify execute was called at least twice (SET and RESET)
        assert mock_session.execute.call_count >= 2, \
            f"Expected at least 2 execute calls, got {mock_session.execute.call_count}"

        # Get the last call (should be RESET)
        last_call = mock_session.execute.call_args_list[-1]
        last_stmt = last_call[0][0]

        stmt_text = str(last_stmt)
        assert "RESET" in stmt_text, f"Last call should be RESET, got: {stmt_text}"

    def test_tenant_session_resets_variable_on_exception(self, mock_session: MagicMock):
        """
        tenant_session() should reset the session variable even on exception.

        The finally block must ensure cleanup regardless of errors.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with pytest.raises(ValueError):
            with tenant_session(mock_session, user_id):
                raise ValueError("Simulated error")

        # Verify execute was called at least twice (SET and RESET)
        assert mock_session.execute.call_count >= 2, \
            f"Expected at least 2 execute calls, got {mock_session.execute.call_count}"

        # Get the last call (should be RESET)
        last_call = mock_session.execute.call_args_list[-1]
        last_stmt = last_call[0][0]

        stmt_text = str(last_stmt)
        assert "RESET" in stmt_text, f"Last call should be RESET even on exception, got: {stmt_text}"

    def test_tenant_session_yields_session(self, mock_session: MagicMock):
        """
        tenant_session() should yield the session for use within the block.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with tenant_session(mock_session, user_id) as session:
            assert session is mock_session

    def test_tenant_session_accepts_string_user_id(self, mock_session: MagicMock):
        """
        tenant_session() should accept string user_id for convenience.
        """
        from app.database import tenant_session

        user_id_str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

        # Should not raise
        with tenant_session(mock_session, user_id_str):
            pass

        # Verify SET was called (first execute call)
        assert mock_session.execute.call_count >= 2

        first_call = mock_session.execute.call_args_list[0]
        first_stmt = first_call[0][0]
        stmt_text = str(first_stmt)
        assert "SET" in stmt_text

    def test_tenant_session_rejects_invalid_uuid(self, mock_session: MagicMock):
        """
        tenant_session() should reject invalid UUIDs.

        This prevents injection attacks through malformed user IDs.
        """
        from app.database import tenant_session

        with pytest.raises((ValueError, TypeError)):
            with tenant_session(mock_session, "not-a-valid-uuid"):
                pass

    def test_tenant_session_order_of_operations(self, mock_session: MagicMock):
        """
        Verify SET is called before the block, RESET after.

        Order matters for RLS enforcement and cleanup.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        operations = []

        def track_execute(stmt, params=None):
            stmt_text = str(stmt)
            if "SET" in stmt_text and "RESET" not in stmt_text:
                operations.append("SET")
            elif "RESET" in stmt_text:
                operations.append("RESET")
            return MagicMock()

        mock_session.execute = MagicMock(side_effect=track_execute)

        with tenant_session(mock_session, user_id):
            operations.append("BODY")

        assert operations == ["SET", "BODY", "RESET"], \
            f"Expected ['SET', 'BODY', 'RESET'], got {operations}"


class TestTenantSessionIntegration:
    """Integration tests for tenant_session with real PostgreSQL."""

    @pytest.mark.skipif(
        True,  # Will be enabled when PostgreSQL is available
        reason="Requires PostgreSQL for RLS testing"
    )
    def test_tenant_session_sets_postgres_variable(self, postgres_test_db):
        """
        Verify the session variable is actually set in PostgreSQL.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with tenant_session(postgres_test_db, user_id) as session:
            # Query the current setting
            result = session.execute(
                text("SELECT current_setting('app.current_user_id', true)")
            )
            value = result.scalar()
            assert value == str(user_id)

    @pytest.mark.skipif(
        True,  # Will be enabled when PostgreSQL is available
        reason="Requires PostgreSQL for RLS testing"
    )
    def test_tenant_session_variable_cleared_after_exit(self, postgres_test_db):
        """
        Verify the session variable is cleared after exiting the context.
        """
        from app.database import tenant_session

        user_id = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")

        with tenant_session(postgres_test_db, user_id):
            pass

        # Query the current setting after exiting
        result = postgres_test_db.execute(
            text("SELECT current_setting('app.current_user_id', true)")
        )
        value = result.scalar()
        assert value is None or value == ""
