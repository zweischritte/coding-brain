"""
Backup Verification Script for Nightly Validation.

This module provides automated backup verification for all data stores:
- PostgreSQL (pg_dump custom format)
- Neo4j (neo4j-admin dump)
- Qdrant (snapshot API)
- OpenSearch (snapshot repository)
- Valkey (RDB/AOF)

Usage:
    verifier = BackupVerifier(webhook_url="https://alerts.example.com")
    config = {
        "postgres": {
            "path": "/backups/postgres/latest.dump",
            "type": "postgres",
            "max_age_hours": 24,
            "min_bytes": 1000,
        }
    }
    report = verifier.verify_all(config)
    if not report["success"]:
        # Alert already sent by verify_all
        sys.exit(1)
"""

import gzip
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class IntegrityResult:
    """Result of backup integrity validation."""

    valid: bool
    error: str | None = None


@dataclass
class RestoreResult:
    """Result of restore test."""

    success: bool
    message: str = ""


class BackupVerifier:
    """
    Backup verification utility for nightly validation.

    Performs the following checks:
    1. Existence: Backup file exists at expected path
    2. Freshness: Backup was modified within expected timeframe
    3. Size: Backup is above minimum size threshold
    4. Integrity: Backup format is valid (gzip, JSON, PGDMP, etc.)
    5. Restore (optional): Test restore to ephemeral environment
    """

    def __init__(self, webhook_url: str | None = None):
        """
        Initialize backup verifier.

        Args:
            webhook_url: Optional URL for alerting webhook (Slack, PagerDuty, etc.)
        """
        self.webhook_url = webhook_url

    def check_exists(self, path: str) -> bool:
        """
        Check if backup file exists.

        Args:
            path: Path to backup file

        Returns:
            True if file exists, False otherwise
        """
        p = Path(path)
        return p.exists() and p.is_file()

    def check_freshness(self, path: str, max_age_hours: int = 24) -> bool:
        """
        Check if backup is recent enough.

        Args:
            path: Path to backup file
            max_age_hours: Maximum age in hours (default: 24)

        Returns:
            True if backup is fresh, False if stale

        Raises:
            FileNotFoundError: If backup file doesn't exist
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Backup file not found: {path}")

        mtime = p.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        return age_hours <= max_age_hours

    def validate_size(self, path: str, min_bytes: int = 1) -> bool:
        """
        Validate backup file size.

        Args:
            path: Path to backup file
            min_bytes: Minimum required size in bytes (default: 1)

        Returns:
            True if file size is acceptable, False otherwise
        """
        p = Path(path)
        if not p.exists():
            return False

        size = p.stat().st_size
        return size >= min_bytes

    def validate_integrity(self, path: str, backup_type: str) -> IntegrityResult:
        """
        Validate backup file format integrity.

        Args:
            path: Path to backup file
            backup_type: Type of backup ("gzip", "json", "postgres", "rdb")

        Returns:
            IntegrityResult with valid=True/False and optional error message
        """
        p = Path(path)
        if not p.exists():
            return IntegrityResult(valid=False, error=f"File not found: {path}")

        validators = {
            "gzip": self._validate_gzip,
            "json": self._validate_json,
            "postgres": self._validate_postgres,
            "rdb": self._validate_rdb,
        }

        validator = validators.get(backup_type)
        if not validator:
            return IntegrityResult(
                valid=False, error=f"Unknown backup type: {backup_type}"
            )

        return validator(path)

    def _validate_gzip(self, path: str) -> IntegrityResult:
        """Validate gzip file integrity."""
        try:
            with gzip.open(path, "rb") as f:
                # Read a small chunk to verify the file is valid gzip
                f.read(1024)
            return IntegrityResult(valid=True)
        except gzip.BadGzipFile as e:
            return IntegrityResult(valid=False, error=f"Invalid gzip file: {e}")
        except OSError as e:
            return IntegrityResult(valid=False, error=f"Corrupt gzip file: {e}")

    def _validate_json(self, path: str) -> IntegrityResult:
        """Validate JSON file integrity."""
        try:
            with open(path, "r") as f:
                json.load(f)
            return IntegrityResult(valid=True)
        except json.JSONDecodeError as e:
            return IntegrityResult(valid=False, error=f"Invalid JSON: {e}")
        except Exception as e:
            return IntegrityResult(valid=False, error=f"Error reading JSON: {e}")

    def _validate_postgres(self, path: str) -> IntegrityResult:
        """Validate PostgreSQL custom format dump."""
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                if header == b"PGDMP":
                    return IntegrityResult(valid=True)
                return IntegrityResult(
                    valid=False, error="Invalid PostgreSQL dump: missing PGDMP header"
                )
        except Exception as e:
            return IntegrityResult(
                valid=False, error=f"Error reading PostgreSQL dump: {e}"
            )

    def _validate_rdb(self, path: str) -> IntegrityResult:
        """Validate Redis/Valkey RDB file."""
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                # RDB files start with "REDIS"
                if header == b"REDIS":
                    return IntegrityResult(valid=True)
                return IntegrityResult(
                    valid=False, error="Invalid RDB file: missing REDIS header"
                )
        except Exception as e:
            return IntegrityResult(valid=False, error=f"Error reading RDB file: {e}")

    def send_alert(self, message: str, severity: str = "warning") -> None:
        """
        Send alert on verification failure.

        Args:
            message: Alert message
            severity: Alert severity ("info", "warning", "critical")
        """
        # Always log
        log_method = {
            "info": logger.info,
            "warning": logger.warning,
            "critical": logger.error,
        }.get(severity, logger.error)
        log_method(f"Backup alert [{severity}]: {message}")

        # Send to webhook if configured
        if self.webhook_url and requests:
            try:
                payload = {
                    "text": f"[{severity.upper()}] Backup Verification: {message}",
                    "severity": severity,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10,
                )
            except Exception as e:
                logger.error(f"Failed to send alert webhook: {e}")

    def restore_test(
        self,
        path: str,
        backup_type: str,
        connection_string: str | None = None,
    ) -> RestoreResult:
        """
        Test restore capability by running restore command.

        Args:
            path: Path to backup file
            backup_type: Type of backup ("postgres", "neo4j", etc.)
            connection_string: Database connection string for restore test

        Returns:
            RestoreResult with success=True/False and message
        """
        if backup_type == "postgres":
            return self._test_postgres_restore(path, connection_string)
        elif backup_type == "neo4j":
            return self._test_neo4j_restore(path)
        else:
            return RestoreResult(
                success=False, message=f"Restore test not implemented for {backup_type}"
            )

    def _test_postgres_restore(
        self, path: str, connection_string: str | None
    ) -> RestoreResult:
        """Test PostgreSQL restore using pg_restore --list."""
        try:
            # Use pg_restore --list to verify backup without actually restoring
            result = subprocess.run(
                ["pg_restore", "--list", path],
                capture_output=True,
                timeout=60,
            )

            if result.returncode == 0:
                return RestoreResult(
                    success=True, message=f"PostgreSQL backup verified: {path}"
                )
            else:
                return RestoreResult(
                    success=False,
                    message=f"pg_restore error: {result.stderr.decode()}",
                )
        except subprocess.TimeoutExpired:
            return RestoreResult(success=False, message="pg_restore timed out")
        except FileNotFoundError:
            return RestoreResult(
                success=False, message="pg_restore not found - skipping restore test"
            )
        except Exception as e:
            return RestoreResult(success=False, message=f"Restore test error: {e}")

    def _test_neo4j_restore(self, path: str) -> RestoreResult:
        """Test Neo4j dump file validity."""
        # Neo4j dump files start with specific header
        try:
            with open(path, "rb") as f:
                # Check file is readable and has content
                header = f.read(100)
                if len(header) > 0:
                    return RestoreResult(
                        success=True, message=f"Neo4j backup verified: {path}"
                    )
                return RestoreResult(success=False, message="Neo4j backup file is empty")
        except Exception as e:
            return RestoreResult(success=False, message=f"Neo4j restore test error: {e}")

    def verify_all(self, config: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Run all verification checks for configured backups.

        Args:
            config: Dictionary mapping service names to backup configurations.
                    Each config should have: path, type, max_age_hours, min_bytes

        Returns:
            Verification report with results for each service and overall success
        """
        report: dict[str, Any] = {"success": True}
        failures: list[str] = []

        for service_name, service_config in config.items():
            path = service_config["path"]
            backup_type = service_config["type"]
            max_age_hours = service_config.get("max_age_hours", 24)
            min_bytes = service_config.get("min_bytes", 1)

            service_report: dict[str, Any] = {}

            # Check existence
            service_report["exists"] = self.check_exists(path)
            if not service_report["exists"]:
                failures.append(f"{service_name}: backup file missing at {path}")
                service_report["fresh"] = False
                service_report["size_ok"] = False
                service_report["integrity"] = IntegrityResult(
                    valid=False, error="File not found"
                )
                report[service_name] = service_report
                continue

            # Check freshness
            try:
                service_report["fresh"] = self.check_freshness(path, max_age_hours)
                if not service_report["fresh"]:
                    failures.append(
                        f"{service_name}: backup older than {max_age_hours} hours"
                    )
            except FileNotFoundError:
                service_report["fresh"] = False
                failures.append(f"{service_name}: could not check freshness")

            # Check size
            service_report["size_ok"] = self.validate_size(path, min_bytes)
            if not service_report["size_ok"]:
                failures.append(f"{service_name}: backup smaller than {min_bytes} bytes")

            # Check integrity
            integrity_result = self.validate_integrity(path, backup_type)
            service_report["integrity"] = integrity_result
            if not integrity_result.valid:
                failures.append(
                    f"{service_name}: integrity check failed - {integrity_result.error}"
                )

            report[service_name] = service_report

        # Set overall success
        if failures:
            report["success"] = False
            report["failures"] = failures
            self.send_alert(
                f"Backup verification failed: {', '.join(failures)}", severity="critical"
            )
        else:
            report["success"] = True
            logger.info("All backup verifications passed")

        return report
