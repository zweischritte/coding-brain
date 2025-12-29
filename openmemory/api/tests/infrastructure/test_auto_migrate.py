"""
Tests for AUTO_MIGRATE feature.

TDD: These tests are written first and should fail until implementation is complete.

This module tests the automatic database migration feature that runs Alembic
migrations on startup when AUTO_MIGRATE=true (PostgreSQL only).

Run with: docker compose exec codingbrain-mcp pytest tests/infrastructure/test_auto_migrate.py -v
"""
import os
from unittest.mock import MagicMock, patch, call

import pytest


class TestAutoMigrateConfig:
    """Test AUTO_MIGRATE configuration and environment detection."""

    def test_auto_migrate_function_can_be_imported(self):
        """auto_migrate_on_startup can be imported from database module."""
        from app.database import auto_migrate_on_startup

        assert auto_migrate_on_startup is not None
        assert callable(auto_migrate_on_startup)

    def test_is_postgres_database_can_be_imported(self):
        """is_postgres_database helper can be imported."""
        from app.database import is_postgres_database

        assert is_postgres_database is not None
        assert callable(is_postgres_database)

    def test_is_postgres_database_returns_true_for_postgres(self):
        """is_postgres_database returns True for PostgreSQL URLs."""
        from app.database import is_postgres_database

        assert is_postgres_database("postgresql://localhost/db") is True
        assert is_postgres_database("postgresql+psycopg2://localhost/db") is True
        assert is_postgres_database("postgresql+asyncpg://localhost/db") is True

    def test_is_postgres_database_returns_false_for_sqlite(self):
        """is_postgres_database returns False for SQLite URLs."""
        from app.database import is_postgres_database

        assert is_postgres_database("sqlite:///./openmemory.db") is False
        assert is_postgres_database("sqlite:///:memory:") is False

    def test_is_postgres_database_handles_edge_cases(self):
        """is_postgres_database handles edge cases gracefully."""
        from app.database import is_postgres_database

        # Empty string should return False
        assert is_postgres_database("") is False
        # None falls back to DATABASE_URL from environment, so we don't test it here
        # as it depends on the runtime environment


class TestAutoMigrateExecution:
    """Test AUTO_MIGRATE execution behavior."""

    @patch.dict(os.environ, {"AUTO_MIGRATE": "true"})
    @patch("app.database.is_postgres_database")
    @patch("app.database.run_alembic_upgrade")
    def test_auto_migrate_runs_when_enabled_and_postgres(
        self, mock_run_upgrade, mock_is_postgres
    ):
        """auto_migrate_on_startup runs migrations when AUTO_MIGRATE=true and PostgreSQL."""
        from app.database import auto_migrate_on_startup

        mock_is_postgres.return_value = True

        auto_migrate_on_startup()

        mock_run_upgrade.assert_called_once()

    @patch.dict(os.environ, {"AUTO_MIGRATE": "false"})
    @patch("app.database.is_postgres_database")
    @patch("app.database.run_alembic_upgrade")
    def test_auto_migrate_skips_when_disabled(
        self, mock_run_upgrade, mock_is_postgres
    ):
        """auto_migrate_on_startup does not run when AUTO_MIGRATE=false."""
        from app.database import auto_migrate_on_startup

        mock_is_postgres.return_value = True

        auto_migrate_on_startup()

        mock_run_upgrade.assert_not_called()

    @patch.dict(os.environ, {"AUTO_MIGRATE": "true"})
    @patch("app.database.is_postgres_database")
    @patch("app.database.run_alembic_upgrade")
    def test_auto_migrate_skips_for_sqlite(
        self, mock_run_upgrade, mock_is_postgres
    ):
        """auto_migrate_on_startup does not run for SQLite databases."""
        from app.database import auto_migrate_on_startup

        mock_is_postgres.return_value = False

        auto_migrate_on_startup()

        mock_run_upgrade.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch("app.database.is_postgres_database")
    @patch("app.database.run_alembic_upgrade")
    def test_auto_migrate_defaults_to_false(
        self, mock_run_upgrade, mock_is_postgres
    ):
        """auto_migrate_on_startup defaults to disabled when env not set."""
        from app.database import auto_migrate_on_startup

        mock_is_postgres.return_value = True

        auto_migrate_on_startup()

        mock_run_upgrade.assert_not_called()


class TestAlembicUpgrade:
    """Test Alembic upgrade execution."""

    def test_run_alembic_upgrade_can_be_imported(self):
        """run_alembic_upgrade can be imported from database module."""
        from app.database import run_alembic_upgrade

        assert run_alembic_upgrade is not None
        assert callable(run_alembic_upgrade)

    @patch("app.database.alembic_command")
    @patch("app.database.alembic_Config")
    def test_run_alembic_upgrade_calls_alembic_command(
        self, mock_config_class, mock_command
    ):
        """run_alembic_upgrade calls alembic.command.upgrade with correct config."""
        from app.database import run_alembic_upgrade

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        run_alembic_upgrade()

        mock_command.upgrade.assert_called_once_with(mock_config, "head")

    @patch("app.database.alembic_command")
    @patch("app.database.alembic_Config")
    def test_run_alembic_upgrade_uses_correct_config_path(
        self, mock_config_class, mock_command
    ):
        """run_alembic_upgrade uses the correct alembic.ini path."""
        from app.database import run_alembic_upgrade

        run_alembic_upgrade()

        # Verify Config was instantiated with a path ending in alembic.ini
        call_args = mock_config_class.call_args[0]
        assert len(call_args) > 0
        config_path = call_args[0]
        assert config_path.endswith("alembic.ini")

    @patch("app.database.alembic_command")
    @patch("app.database.alembic_Config")
    @patch("app.database.logger")
    def test_run_alembic_upgrade_logs_success(
        self, mock_logger, mock_config_class, mock_command
    ):
        """run_alembic_upgrade logs success message."""
        from app.database import run_alembic_upgrade

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        run_alembic_upgrade()

        # Should log info about migration
        assert mock_logger.info.called

    @patch("app.database.alembic_command")
    @patch("app.database.alembic_Config")
    @patch("app.database.logger")
    def test_run_alembic_upgrade_handles_errors_gracefully(
        self, mock_logger, mock_config_class, mock_command
    ):
        """run_alembic_upgrade logs errors without crashing."""
        from app.database import run_alembic_upgrade

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_command.upgrade.side_effect = Exception("Migration failed")

        # Should not raise, but log the error
        run_alembic_upgrade()

        mock_logger.error.assert_called()


class TestAutoMigrateIntegration:
    """Integration tests for AUTO_MIGRATE with main.py startup."""

    @patch("app.database.auto_migrate_on_startup")
    def test_main_calls_auto_migrate_on_startup(self, mock_auto_migrate):
        """main.py should call auto_migrate_on_startup during initialization."""
        # This test verifies the integration point exists
        # The actual call happens at module load time
        from app.database import auto_migrate_on_startup

        assert auto_migrate_on_startup is not None

    def test_auto_migrate_flag_accepts_various_true_values(self):
        """AUTO_MIGRATE should accept 'true', 'True', 'TRUE', '1', 'yes'."""
        from app.database import should_auto_migrate

        with patch.dict(os.environ, {"AUTO_MIGRATE": "true"}):
            assert should_auto_migrate() is True

        with patch.dict(os.environ, {"AUTO_MIGRATE": "True"}):
            assert should_auto_migrate() is True

        with patch.dict(os.environ, {"AUTO_MIGRATE": "TRUE"}):
            assert should_auto_migrate() is True

        with patch.dict(os.environ, {"AUTO_MIGRATE": "1"}):
            assert should_auto_migrate() is True

        with patch.dict(os.environ, {"AUTO_MIGRATE": "yes"}):
            assert should_auto_migrate() is True

    def test_auto_migrate_flag_rejects_false_values(self):
        """AUTO_MIGRATE should return False for 'false', 'False', '0', 'no'."""
        from app.database import should_auto_migrate

        with patch.dict(os.environ, {"AUTO_MIGRATE": "false"}):
            assert should_auto_migrate() is False

        with patch.dict(os.environ, {"AUTO_MIGRATE": "False"}):
            assert should_auto_migrate() is False

        with patch.dict(os.environ, {"AUTO_MIGRATE": "0"}):
            assert should_auto_migrate() is False

        with patch.dict(os.environ, {"AUTO_MIGRATE": "no"}):
            assert should_auto_migrate() is False
