"""
Tests for Phase 7: Backup Verification Script.

TDD: These tests are written first and should fail until implementation is complete.

This module tests the nightly backup verification infrastructure:
- Backup existence checks
- Backup freshness validation
- Backup size validation
- Backup integrity checks (format validation)
- Alert triggering on verification failure

Run with: docker compose exec codingbrain-mcp pytest tests/infrastructure/test_backup_verification.py -v
"""
import gzip
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBackupExistenceCheck:
    """Test backup file existence verification."""

    def test_backup_verifier_can_be_imported(self):
        """BackupVerifier can be imported from scripts module."""
        from app.backup.verifier import BackupVerifier

        assert BackupVerifier is not None

    def test_check_exists_returns_true_for_existing_file(self):
        """check_exists returns True when backup file exists."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"backup content")
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.check_exists(temp_path)
            assert result is True
        finally:
            Path(temp_path).unlink()

    def test_check_exists_returns_false_for_missing_file(self):
        """check_exists returns False when backup file is missing."""
        from app.backup.verifier import BackupVerifier

        verifier = BackupVerifier()
        result = verifier.check_exists("/nonexistent/path/backup.dump")
        assert result is False

    def test_check_exists_returns_false_for_directory(self):
        """check_exists returns False when path is a directory, not a file."""
        from app.backup.verifier import BackupVerifier

        with tempfile.TemporaryDirectory() as tmpdir:
            verifier = BackupVerifier()
            result = verifier.check_exists(tmpdir)
            assert result is False


class TestBackupFreshnessCheck:
    """Test backup freshness (age) validation."""

    def test_check_freshness_passes_for_recent_backup(self):
        """check_freshness returns True for backup within max_age_hours."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"recent backup")
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            # File just created, should be within 24 hours
            result = verifier.check_freshness(temp_path, max_age_hours=24)
            assert result is True
        finally:
            Path(temp_path).unlink()

    def test_check_freshness_fails_for_stale_backup(self):
        """check_freshness returns False for backup older than max_age_hours."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"old backup")
            temp_path = f.name

        try:
            # Set modification time to 48 hours ago
            old_time = time.time() - (48 * 3600)
            import os

            os.utime(temp_path, (old_time, old_time))

            verifier = BackupVerifier()
            result = verifier.check_freshness(temp_path, max_age_hours=24)
            assert result is False
        finally:
            Path(temp_path).unlink()

    def test_check_freshness_raises_for_missing_file(self):
        """check_freshness raises FileNotFoundError for missing file."""
        from app.backup.verifier import BackupVerifier

        verifier = BackupVerifier()
        with pytest.raises(FileNotFoundError):
            verifier.check_freshness("/nonexistent/backup.dump", max_age_hours=24)

    def test_check_freshness_default_max_age(self):
        """check_freshness uses 24 hours as default max_age_hours."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"backup")
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            # Default should be 24 hours
            result = verifier.check_freshness(temp_path)  # No max_age_hours specified
            assert result is True
        finally:
            Path(temp_path).unlink()


class TestBackupSizeValidation:
    """Test backup file size validation."""

    def test_validate_size_passes_for_nonzero_file(self):
        """validate_size returns True for file with content."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"x" * 1000)  # 1KB of content
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_size(temp_path, min_bytes=100)
            assert result is True
        finally:
            Path(temp_path).unlink()

    def test_validate_size_fails_for_empty_file(self):
        """validate_size returns False for empty (0 byte) file."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            # Don't write anything - empty file
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_size(temp_path, min_bytes=1)
            assert result is False
        finally:
            Path(temp_path).unlink()

    def test_validate_size_fails_when_below_minimum(self):
        """validate_size returns False when file is smaller than min_bytes."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"small")  # 5 bytes
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_size(temp_path, min_bytes=1000)
            assert result is False
        finally:
            Path(temp_path).unlink()

    def test_validate_size_default_minimum(self):
        """validate_size uses reasonable default min_bytes."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"x" * 1024)  # 1KB
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_size(temp_path)  # No min_bytes specified
            assert result is True
        finally:
            Path(temp_path).unlink()


class TestBackupIntegrityValidation:
    """Test backup format integrity validation."""

    def test_validate_integrity_passes_for_valid_gzip(self):
        """validate_integrity returns True for valid gzip file."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as f:
            temp_path = f.name

        try:
            # Create valid gzip file
            with gzip.open(temp_path, "wb") as gz:
                gz.write(b"backup content here")

            verifier = BackupVerifier()
            result = verifier.validate_integrity(temp_path, backup_type="gzip")
            assert result.valid is True
        finally:
            Path(temp_path).unlink()

    def test_validate_integrity_fails_for_corrupted_gzip(self):
        """validate_integrity returns False for corrupted gzip file."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".gz") as f:
            f.write(b"not a valid gzip file")
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_integrity(temp_path, backup_type="gzip")
            assert result.valid is False
            assert "corrupt" in result.error.lower() or "invalid" in result.error.lower()
        finally:
            Path(temp_path).unlink()

    def test_validate_integrity_passes_for_valid_json(self):
        """validate_integrity returns True for valid JSON snapshot."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
            json.dump({"collections": [], "version": "1.0"}, f)
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_integrity(temp_path, backup_type="json")
            assert result.valid is True
        finally:
            Path(temp_path).unlink()

    def test_validate_integrity_fails_for_invalid_json(self):
        """validate_integrity returns False for invalid JSON file."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
            f.write("{not valid json")
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_integrity(temp_path, backup_type="json")
            assert result.valid is False
        finally:
            Path(temp_path).unlink()

    def test_validate_integrity_checks_postgres_dump_header(self):
        """validate_integrity checks PostgreSQL custom format magic bytes."""
        from app.backup.verifier import BackupVerifier

        # PostgreSQL custom format starts with "PGDMP"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"PGDMP" + b"\x00" * 100)  # Valid header
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_integrity(temp_path, backup_type="postgres")
            assert result.valid is True
        finally:
            Path(temp_path).unlink()

    def test_validate_integrity_fails_for_invalid_postgres_dump(self):
        """validate_integrity fails for file without PGDMP header."""
        from app.backup.verifier import BackupVerifier

        with tempfile.NamedTemporaryFile(delete=False, suffix=".dump") as f:
            f.write(b"not a postgres dump")
            temp_path = f.name

        try:
            verifier = BackupVerifier()
            result = verifier.validate_integrity(temp_path, backup_type="postgres")
            assert result.valid is False
        finally:
            Path(temp_path).unlink()


class TestAlertOnFailure:
    """Test alert triggering on verification failure."""

    def test_send_alert_can_be_called(self):
        """send_alert method exists and can be called."""
        from app.backup.verifier import BackupVerifier

        verifier = BackupVerifier()
        # Should not raise
        verifier.send_alert("Test alert", severity="warning")

    def test_send_alert_logs_error(self):
        """send_alert logs the error message."""
        from app.backup.verifier import BackupVerifier

        with patch("app.backup.verifier.logger") as mock_logger:
            verifier = BackupVerifier()
            verifier.send_alert("Backup verification failed", severity="critical")

            mock_logger.error.assert_called()
            call_args = str(mock_logger.error.call_args)
            assert "Backup verification failed" in call_args

    def test_send_alert_with_webhook(self):
        """send_alert can trigger webhook for external alerting."""
        from app.backup.verifier import BackupVerifier

        with patch("app.backup.verifier.requests") as mock_requests:
            mock_requests.post.return_value.status_code = 200

            verifier = BackupVerifier(webhook_url="https://alerts.example.com/hook")
            verifier.send_alert("Critical failure", severity="critical")

            mock_requests.post.assert_called_once()


class TestVerifyAll:
    """Test full verification pipeline."""

    def test_verify_all_returns_verification_report(self):
        """verify_all returns a structured report with all checks."""
        from app.backup.verifier import BackupVerifier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid backup file
            backup_path = Path(tmpdir) / "test.dump"
            with open(backup_path, "wb") as f:
                f.write(b"PGDMP" + b"\x00" * 100)

            config = {
                "postgres": {
                    "path": str(backup_path),
                    "type": "postgres",
                    "max_age_hours": 24,
                    "min_bytes": 10,
                }
            }

            verifier = BackupVerifier()
            report = verifier.verify_all(config)

            assert isinstance(report, dict)
            assert "postgres" in report
            assert "exists" in report["postgres"]
            assert "fresh" in report["postgres"]
            assert "size_ok" in report["postgres"]
            assert "integrity" in report["postgres"]

    def test_verify_all_all_pass(self):
        """verify_all returns success=True when all checks pass."""
        from app.backup.verifier import BackupVerifier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid gzip backup
            backup_path = Path(tmpdir) / "test.gz"
            with gzip.open(backup_path, "wb") as gz:
                gz.write(b"backup content" * 100)

            config = {
                "qdrant": {
                    "path": str(backup_path),
                    "type": "gzip",
                    "max_age_hours": 24,
                    "min_bytes": 10,
                }
            }

            verifier = BackupVerifier()
            report = verifier.verify_all(config)

            assert report["qdrant"]["exists"] is True
            assert report["qdrant"]["fresh"] is True
            assert report["qdrant"]["size_ok"] is True
            assert report["qdrant"]["integrity"].valid is True
            assert report["success"] is True

    def test_verify_all_triggers_alert_on_failure(self):
        """verify_all triggers alert when any check fails."""
        from app.backup.verifier import BackupVerifier

        config = {
            "postgres": {
                "path": "/nonexistent/backup.dump",
                "type": "postgres",
                "max_age_hours": 24,
                "min_bytes": 1000,
            }
        }

        with patch.object(BackupVerifier, "send_alert") as mock_alert:
            verifier = BackupVerifier()
            report = verifier.verify_all(config)

            assert report["success"] is False
            mock_alert.assert_called()

    def test_verify_all_handles_multiple_services(self):
        """verify_all can verify multiple services in one run."""
        from app.backup.verifier import BackupVerifier

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create backups for multiple services
            pg_path = Path(tmpdir) / "postgres.dump"
            with open(pg_path, "wb") as f:
                f.write(b"PGDMP" + b"\x00" * 100)

            qdrant_path = Path(tmpdir) / "qdrant.gz"
            with gzip.open(qdrant_path, "wb") as gz:
                gz.write(b"snapshot data")

            config = {
                "postgres": {
                    "path": str(pg_path),
                    "type": "postgres",
                    "max_age_hours": 24,
                    "min_bytes": 10,
                },
                "qdrant": {
                    "path": str(qdrant_path),
                    "type": "gzip",
                    "max_age_hours": 24,
                    "min_bytes": 10,
                },
            }

            verifier = BackupVerifier()
            report = verifier.verify_all(config)

            assert "postgres" in report
            assert "qdrant" in report


class TestIntegrityResult:
    """Test IntegrityResult dataclass."""

    def test_integrity_result_can_be_imported(self):
        """IntegrityResult can be imported from verifier module."""
        from app.backup.verifier import IntegrityResult

        assert IntegrityResult is not None

    def test_integrity_result_valid(self):
        """IntegrityResult can represent valid state."""
        from app.backup.verifier import IntegrityResult

        result = IntegrityResult(valid=True, error=None)
        assert result.valid is True
        assert result.error is None

    def test_integrity_result_invalid_with_error(self):
        """IntegrityResult can represent invalid state with error message."""
        from app.backup.verifier import IntegrityResult

        result = IntegrityResult(valid=False, error="File is corrupted")
        assert result.valid is False
        assert result.error == "File is corrupted"


class TestRestoreTest:
    """Test restore verification capability."""

    def test_restore_test_method_exists(self):
        """restore_test method exists on BackupVerifier."""
        from app.backup.verifier import BackupVerifier

        verifier = BackupVerifier()
        assert hasattr(verifier, "restore_test")
        assert callable(verifier.restore_test)

    def test_restore_test_runs_postgres_restore_check(self):
        """restore_test can verify PostgreSQL backup is restorable."""
        from app.backup.verifier import BackupVerifier

        with patch("app.backup.verifier.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = b"table users: 100 rows"

            verifier = BackupVerifier()
            result = verifier.restore_test(
                path="/backups/postgres.dump",
                backup_type="postgres",
                connection_string="postgresql://test:test@localhost:5432/test_db",
            )

            assert result.success is True

    def test_restore_test_detects_restore_failure(self):
        """restore_test detects when restore command fails."""
        from app.backup.verifier import BackupVerifier

        with patch("app.backup.verifier.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = b"pg_restore: error: could not open input file"

            verifier = BackupVerifier()
            result = verifier.restore_test(
                path="/backups/corrupt.dump",
                backup_type="postgres",
                connection_string="postgresql://test:test@localhost:5432/test_db",
            )

            assert result.success is False
            assert "error" in result.message.lower()
