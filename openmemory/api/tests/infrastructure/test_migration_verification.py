"""
Tests for Phase 3: PostgreSQL Migration - Verification Utilities.

TDD: These tests are written first and should fail until implementation is complete.

This module tests the migration verification infrastructure required by the
production readiness plan:
- Pre-migration backup validation
- Row count verification
- Checksum calculation for data integrity
- Rollback procedures

Run with: docker compose exec codingbrain-mcp pytest tests/infrastructure/test_migration_verification.py -v
"""
import hashlib
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


class TestMigrationVerifier:
    """Test the MigrationVerifier class for pre/post migration checks."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        return session

    @pytest.fixture
    def mock_engine(self, mock_session):
        """Create a mock SQLAlchemy engine."""
        engine = MagicMock()
        engine.connect.return_value = mock_session
        return engine

    def test_verifier_can_be_imported(self):
        """MigrationVerifier can be imported from alembic utils."""
        from app.alembic.utils import MigrationVerifier

        assert MigrationVerifier is not None

    def test_verifier_initializes_with_engine(self, mock_engine):
        """MigrationVerifier initializes with a SQLAlchemy engine."""
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(mock_engine)
        assert verifier.engine is mock_engine

    def test_get_table_row_counts_returns_dict(self, mock_engine, mock_session):
        """get_table_row_counts returns a dict of table_name -> count."""
        from app.alembic.utils import MigrationVerifier

        # Setup mock to return row counts
        mock_session.execute.return_value.scalar.side_effect = [100, 50, 25]

        verifier = MigrationVerifier(mock_engine)
        counts = verifier.get_table_row_counts(["users", "apps", "memories"])

        assert isinstance(counts, dict)
        assert counts == {"users": 100, "apps": 50, "memories": 25}

    def test_get_table_row_counts_handles_empty_tables(self, mock_engine, mock_session):
        """get_table_row_counts handles tables with zero rows."""
        from app.alembic.utils import MigrationVerifier

        mock_session.execute.return_value.scalar.return_value = 0

        verifier = MigrationVerifier(mock_engine)
        counts = verifier.get_table_row_counts(["empty_table"])

        assert counts == {"empty_table": 0}

    def test_calculate_table_checksum_returns_hex_string(self, mock_engine, mock_session):
        """calculate_table_checksum returns a hex digest string."""
        from app.alembic.utils import MigrationVerifier

        # Mock query result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (uuid4(), "user1", "test@example.com"),
            (uuid4(), "user2", "other@example.com"),
        ]
        mock_session.execute.return_value = mock_result

        verifier = MigrationVerifier(mock_engine)
        checksum = verifier.calculate_table_checksum("users", ["id", "user_id", "email"])

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length

    def test_calculate_table_checksum_is_deterministic(self, mock_engine, mock_session):
        """Same data produces same checksum."""
        from app.alembic.utils import MigrationVerifier

        fixed_uuid = uuid4()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (fixed_uuid, "user1", "test@example.com"),
        ]
        mock_session.execute.return_value = mock_result

        verifier = MigrationVerifier(mock_engine)
        checksum1 = verifier.calculate_table_checksum("users", ["id", "user_id", "email"])
        checksum2 = verifier.calculate_table_checksum("users", ["id", "user_id", "email"])

        assert checksum1 == checksum2

    def test_verify_row_counts_passes_when_equal(self, mock_engine):
        """verify_row_counts returns True when pre and post counts match."""
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(mock_engine)
        pre_counts = {"users": 100, "apps": 50}
        post_counts = {"users": 100, "apps": 50}

        result = verifier.verify_row_counts(pre_counts, post_counts)

        assert result.success is True
        assert result.mismatches == []

    def test_verify_row_counts_fails_when_different(self, mock_engine):
        """verify_row_counts returns False with mismatches when counts differ."""
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(mock_engine)
        pre_counts = {"users": 100, "apps": 50}
        post_counts = {"users": 100, "apps": 45}  # apps lost 5 rows

        result = verifier.verify_row_counts(pre_counts, post_counts)

        assert result.success is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0]["table"] == "apps"
        assert result.mismatches[0]["pre"] == 50
        assert result.mismatches[0]["post"] == 45

    def test_verify_checksums_passes_when_equal(self, mock_engine):
        """verify_checksums returns True when pre and post checksums match."""
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(mock_engine)
        checksum = hashlib.sha256(b"test data").hexdigest()
        pre_checksums = {"users": checksum}
        post_checksums = {"users": checksum}

        result = verifier.verify_checksums(pre_checksums, post_checksums)

        assert result.success is True
        assert result.mismatches == []

    def test_verify_checksums_fails_when_different(self, mock_engine):
        """verify_checksums returns False with mismatches when checksums differ."""
        from app.alembic.utils import MigrationVerifier

        verifier = MigrationVerifier(mock_engine)
        pre_checksums = {"users": hashlib.sha256(b"before").hexdigest()}
        post_checksums = {"users": hashlib.sha256(b"after").hexdigest()}

        result = verifier.verify_checksums(pre_checksums, post_checksums)

        assert result.success is False
        assert len(result.mismatches) == 1
        assert result.mismatches[0]["table"] == "users"


class TestVerificationResult:
    """Test the VerificationResult dataclass."""

    def test_verification_result_can_be_imported(self):
        """VerificationResult can be imported from alembic utils."""
        from app.alembic.utils import VerificationResult

        assert VerificationResult is not None

    def test_verification_result_has_required_fields(self):
        """VerificationResult has success and mismatches fields."""
        from app.alembic.utils import VerificationResult

        result = VerificationResult(success=True, mismatches=[])

        assert hasattr(result, "success")
        assert hasattr(result, "mismatches")

    def test_verification_result_with_mismatches(self):
        """VerificationResult can store mismatch details."""
        from app.alembic.utils import VerificationResult

        mismatches = [{"table": "users", "pre": 100, "post": 90}]
        result = VerificationResult(success=False, mismatches=mismatches)

        assert result.success is False
        assert len(result.mismatches) == 1


class TestBackupValidator:
    """Test the BackupValidator class for pre-migration backup checks."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SQLAlchemy engine."""
        engine = MagicMock()
        return engine

    def test_backup_validator_can_be_imported(self):
        """BackupValidator can be imported from alembic utils."""
        from app.alembic.utils import BackupValidator

        assert BackupValidator is not None

    def test_backup_validator_initializes_with_backup_path(self):
        """BackupValidator initializes with a backup directory path."""
        from app.alembic.utils import BackupValidator

        validator = BackupValidator("/backups")
        assert validator.backup_path == "/backups"

    def test_validate_backup_exists_returns_true_when_present(self):
        """validate_backup_exists returns True when backup file exists."""
        from app.alembic.utils import BackupValidator

        validator = BackupValidator("/backups")

        with patch("os.path.exists", return_value=True):
            result = validator.validate_backup_exists("migration_123_backup.sql")

        assert result is True

    def test_validate_backup_exists_returns_false_when_missing(self):
        """validate_backup_exists returns False when backup file is missing."""
        from app.alembic.utils import BackupValidator

        validator = BackupValidator("/backups")

        with patch("os.path.exists", return_value=False):
            result = validator.validate_backup_exists("missing_backup.sql")

        assert result is False

    def test_validate_backup_integrity_checks_file_size(self):
        """validate_backup_integrity ensures backup file is not empty."""
        from app.alembic.utils import BackupValidator

        validator = BackupValidator("/backups")

        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=0):
                result = validator.validate_backup_integrity("empty_backup.sql")

        assert result.success is False
        assert "empty" in result.error.lower()

    def test_validate_backup_integrity_passes_for_valid_backup(self):
        """validate_backup_integrity returns True for valid backup."""
        from app.alembic.utils import BackupValidator

        validator = BackupValidator("/backups")

        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=1024 * 1024):  # 1MB file
                result = validator.validate_backup_integrity("valid_backup.sql")

        assert result.success is True

    def test_get_latest_backup_returns_most_recent(self):
        """get_latest_backup returns the most recently created backup file."""
        from app.alembic.utils import BackupValidator

        validator = BackupValidator("/backups")

        mock_files = [
            "backup_20250101_120000.sql",
            "backup_20250102_120000.sql",
            "backup_20250103_120000.sql",
        ]

        with patch("app.alembic.utils.os.path.exists", return_value=True):
            with patch("app.alembic.utils.os.listdir", return_value=mock_files):
                with patch("app.alembic.utils.os.path.getmtime") as mock_mtime:
                    # Return timestamps in order (oldest to newest)
                    mock_mtime.side_effect = [1000, 2000, 3000]
                    result = validator.get_latest_backup()

        assert result == "backup_20250103_120000.sql"


class TestBatchMigrator:
    """Test the BatchMigrator class for large data migrations."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        return session

    @pytest.fixture
    def mock_engine(self, mock_session):
        """Create a mock SQLAlchemy engine."""
        engine = MagicMock()
        engine.connect.return_value = mock_session
        return engine

    def test_batch_migrator_can_be_imported(self):
        """BatchMigrator can be imported from alembic utils."""
        from app.alembic.utils import BatchMigrator

        assert BatchMigrator is not None

    def test_batch_migrator_initializes_with_defaults(self, mock_engine):
        """BatchMigrator initializes with default batch size."""
        from app.alembic.utils import BatchMigrator

        migrator = BatchMigrator(mock_engine)

        assert migrator.engine is mock_engine
        assert migrator.batch_size == 1000  # Default batch size

    def test_batch_migrator_accepts_custom_batch_size(self, mock_engine):
        """BatchMigrator accepts custom batch size."""
        from app.alembic.utils import BatchMigrator

        migrator = BatchMigrator(mock_engine, batch_size=500)

        assert migrator.batch_size == 500

    def test_migrate_in_batches_processes_all_rows(self, mock_engine, mock_session):
        """migrate_in_batches processes all rows in batches."""
        from app.alembic.utils import BatchMigrator

        # Mock: first execute returns count, second returns result for SELECT *
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 2500

        mock_data_result = MagicMock()
        # Return 3 batches: 1000, 1000, 500, then empty to stop
        batch1 = [{"id": i} for i in range(1000)]
        batch2 = [{"id": i} for i in range(1000, 2000)]
        batch3 = [{"id": i} for i in range(2000, 2500)]
        mock_data_result.fetchmany.side_effect = [batch1, batch2, batch3, []]

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        migrator = BatchMigrator(mock_engine, batch_size=1000)

        progress_callback = MagicMock()
        migrator.migrate_in_batches(
            source_table="old_table",
            target_table="new_table",
            transform_fn=lambda x: x,
            progress_callback=progress_callback,
        )

        # Should process 3 batches (1000 + 1000 + 500)
        assert progress_callback.call_count == 3

    def test_migrate_in_batches_calls_transform_fn(self, mock_engine, mock_session):
        """migrate_in_batches applies transform function to each row."""
        from app.alembic.utils import BatchMigrator

        # Mock: first execute returns count, second returns result for SELECT *
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 2

        mock_data_result = MagicMock()
        # Return 2 rows with proper row-like objects
        mock_row1 = MagicMock()
        mock_row1._mapping = {"id": 1, "value": "old1"}
        mock_row2 = MagicMock()
        mock_row2._mapping = {"id": 2, "value": "old2"}
        mock_data_result.fetchmany.side_effect = [[mock_row1, mock_row2], []]

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        migrator = BatchMigrator(mock_engine, batch_size=1000)

        transform_fn = MagicMock(side_effect=lambda x: {**x, "value": x["value"].upper()})
        migrator.migrate_in_batches(
            source_table="old_table",
            target_table="new_table",
            transform_fn=transform_fn,
        )

        # Transform should be called for each row
        assert transform_fn.call_count == 2


class TestRollbackManager:
    """Test the RollbackManager class for migration rollback procedures."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=None)
        return session

    @pytest.fixture
    def mock_engine(self, mock_session):
        """Create a mock SQLAlchemy engine."""
        engine = MagicMock()
        engine.connect.return_value = mock_session
        return engine

    def test_rollback_manager_can_be_imported(self):
        """RollbackManager can be imported from alembic utils."""
        from app.alembic.utils import RollbackManager

        assert RollbackManager is not None

    def test_rollback_manager_initializes_with_engine(self, mock_engine):
        """RollbackManager initializes with a SQLAlchemy engine."""
        from app.alembic.utils import RollbackManager

        manager = RollbackManager(mock_engine)
        assert manager.engine is mock_engine

    def test_create_savepoint_returns_savepoint_id(self, mock_engine, mock_session):
        """create_savepoint creates a database savepoint and returns its ID."""
        from app.alembic.utils import RollbackManager

        manager = RollbackManager(mock_engine)
        savepoint_id = manager.create_savepoint("migration_v1_2_3")

        assert savepoint_id is not None
        assert "migration_v1_2_3" in savepoint_id

    def test_rollback_to_savepoint_executes_rollback(self, mock_engine, mock_session):
        """rollback_to_savepoint executes ROLLBACK TO SAVEPOINT."""
        from app.alembic.utils import RollbackManager

        manager = RollbackManager(mock_engine)
        manager.rollback_to_savepoint("sp_migration_v1_2_3")

        # Verify rollback was executed - check the text clause content
        mock_session.execute.assert_called()
        # Get the TextClause object passed to execute
        text_clause = mock_session.execute.call_args[0][0]
        sql_text = str(text_clause)
        assert "ROLLBACK" in sql_text and "sp_migration_v1_2_3" in sql_text

    def test_release_savepoint_removes_savepoint(self, mock_engine, mock_session):
        """release_savepoint releases the savepoint after successful migration."""
        from app.alembic.utils import RollbackManager

        manager = RollbackManager(mock_engine)
        manager.release_savepoint("sp_migration_v1_2_3")

        # Verify release was executed - check the text clause content
        mock_session.execute.assert_called()
        # Get the TextClause object passed to execute
        text_clause = mock_session.execute.call_args[0][0]
        sql_text = str(text_clause)
        assert "RELEASE" in sql_text and "sp_migration_v1_2_3" in sql_text

    def test_get_current_revision_returns_alembic_version(self, mock_engine, mock_session):
        """get_current_revision returns the current Alembic version."""
        from app.alembic.utils import RollbackManager

        mock_session.execute.return_value.scalar.return_value = "abc123def"

        manager = RollbackManager(mock_engine)
        revision = manager.get_current_revision()

        assert revision == "abc123def"

    def test_can_safely_rollback_checks_dependencies(self, mock_engine, mock_session):
        """can_safely_rollback checks if rollback is safe based on data dependencies."""
        from app.alembic.utils import RollbackManager

        manager = RollbackManager(mock_engine)

        # Mock: no new data added that would be lost
        mock_session.execute.return_value.scalar.return_value = 0

        result = manager.can_safely_rollback("new_column_migration")

        assert result.safe is True

    def test_can_safely_rollback_warns_about_data_loss(self, mock_engine, mock_session):
        """can_safely_rollback warns when rollback would cause data loss."""
        from app.alembic.utils import RollbackManager

        manager = RollbackManager(mock_engine)

        # Mock: new data exists that would be lost
        mock_session.execute.return_value.scalar.return_value = 50

        result = manager.can_safely_rollback("new_column_migration")

        assert result.safe is False
        assert result.data_at_risk > 0
