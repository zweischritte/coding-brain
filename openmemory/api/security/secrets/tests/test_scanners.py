"""Tests for secret scanners (fast and deep).

This module tests the tiered scanning system per section 5.5 (FR-011):
- Fast sync scan: <20ms for quick pattern matching
- Deep async scan: Comprehensive analysis with active verification
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from openmemory.api.security.secrets.patterns import Confidence, SecretType
from openmemory.api.security.secrets.quarantine import (
    QuarantineEntry,
    QuarantineReason,
    QuarantineState,
)
from openmemory.api.security.secrets.fast_scanner import (
    FastScanner,
    FastScanConfig,
    FastScanResult,
    ScanContext,
)
from openmemory.api.security.secrets.deep_scanner import (
    DeepScanner,
    DeepScanConfig,
    DeepScanResult,
    VerificationResult,
    VerificationStatus,
)


# ============================================================================
# FastScanConfig Tests
# ============================================================================


class TestFastScanConfig:
    """Tests for FastScanConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FastScanConfig()
        assert config.timeout_ms <= 20  # Must be fast
        assert config.min_confidence >= Confidence.MEDIUM
        assert config.skip_large_files is True
        assert config.max_file_size_kb >= 1024

    def test_custom_config(self):
        """Test custom configuration."""
        config = FastScanConfig(
            timeout_ms=10,
            min_confidence=Confidence.HIGH,
            skip_large_files=False,
            max_file_size_kb=512,
        )
        assert config.timeout_ms == 10
        assert config.min_confidence == Confidence.HIGH
        assert config.skip_large_files is False
        assert config.max_file_size_kb == 512


# ============================================================================
# ScanContext Tests
# ============================================================================


class TestScanContext:
    """Tests for ScanContext."""

    def test_context_creation(self):
        """Test creating a scan context."""
        ctx = ScanContext(
            file_path="/path/to/file.py",
            repo_id="repo-123",
            commit_sha="abc123",
        )
        assert ctx.file_path == "/path/to/file.py"
        assert ctx.repo_id == "repo-123"
        assert ctx.commit_sha == "abc123"

    def test_context_with_metadata(self):
        """Test context with additional metadata."""
        ctx = ScanContext(
            file_path="/config.yaml",
            metadata={"branch": "main", "author": "user@example.com"},
        )
        assert ctx.metadata["branch"] == "main"


# ============================================================================
# FastScanResult Tests
# ============================================================================


class TestFastScanResult:
    """Tests for FastScanResult."""

    def test_result_creation(self):
        """Test creating a scan result."""
        result = FastScanResult(
            matches=[],
            scan_time_ms=5.2,
            file_path="/test.py",
            file_size_bytes=1024,
            patterns_checked=25,
        )
        assert result.scan_time_ms == 5.2
        assert result.file_path == "/test.py"
        assert result.patterns_checked == 25

    def test_result_with_matches(self):
        """Test result with detected secrets."""
        from openmemory.api.security.secrets.patterns import SecretMatch

        matches = [
            SecretMatch(
                secret_type=SecretType.AWS_ACCESS_KEY,
                matched_value="AKIA...",
                redacted_value="AKIA...",
                confidence=Confidence.HIGH,
                line_number=10,
            )
        ]
        result = FastScanResult(
            matches=matches,
            scan_time_ms=8.1,
            file_path="/config.py",
            file_size_bytes=2048,
            patterns_checked=25,
        )
        assert len(result.matches) == 1
        assert result.has_secrets is True

    def test_result_to_dict(self):
        """Test serializing result to dictionary."""
        result = FastScanResult(
            matches=[],
            scan_time_ms=3.5,
            file_path="/test.py",
            file_size_bytes=512,
            patterns_checked=25,
        )
        d = result.to_dict()
        assert "scan_time_ms" in d
        assert "file_path" in d
        assert "secrets_found" in d


# ============================================================================
# FastScanner Tests
# ============================================================================


class TestFastScanner:
    """Tests for FastScanner."""

    @pytest.fixture
    def scanner(self):
        """Create a scanner instance."""
        return FastScanner()

    def test_scan_clean_file(self, scanner):
        """Test scanning a file with no secrets."""
        content = """
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
"""
        result = scanner.scan(content, file_path="/hello.py")
        assert result.has_secrets is False
        assert len(result.matches) == 0
        assert result.scan_time_ms < 20  # Must be fast

    def test_scan_with_aws_key(self, scanner):
        """Test scanning a file with AWS key."""
        content = """
import boto3

AWS_ACCESS_KEY_ID = "AKIAJ5Q7R2D9K3G4N8M1"
client = boto3.client('s3')
"""
        result = scanner.scan(content, file_path="/config.py")
        assert result.has_secrets is True
        assert any(m.secret_type == SecretType.AWS_ACCESS_KEY for m in result.matches)

    def test_scan_with_github_token(self, scanner):
        """Test scanning a file with GitHub token."""
        content = """
GITHUB_TOKEN = "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
"""
        result = scanner.scan(content, file_path="/.env")
        assert result.has_secrets is True
        assert any(m.secret_type == SecretType.GITHUB_PAT for m in result.matches)

    def test_scan_with_private_key(self, scanner):
        """Test scanning a file with private key."""
        content = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy
-----END RSA PRIVATE KEY-----"""
        result = scanner.scan(content, file_path="/key.pem")
        assert result.has_secrets is True
        assert any(m.secret_type == SecretType.PRIVATE_KEY_RSA for m in result.matches)

    def test_scan_performance(self, scanner):
        """Test that scanning completes within time limit."""
        # Large-ish content
        content = "x" * 10000 + "\nAKIAJ5Q7R2D9K3G4N8M1\n" + "y" * 10000
        result = scanner.scan(content, file_path="/large.txt")
        assert result.scan_time_ms < 20  # Must complete in <20ms

    def test_scan_empty_content(self, scanner):
        """Test scanning empty content."""
        result = scanner.scan("", file_path="/empty.txt")
        assert result.has_secrets is False
        assert len(result.matches) == 0

    def test_scan_with_context(self, scanner):
        """Test scanning with context."""
        ctx = ScanContext(
            file_path="/src/config.py",
            repo_id="my-repo",
            commit_sha="abc123",
        )
        content = 'API_KEY = "AKIAJ5Q7R2D9K3G4N8M1"'
        result = scanner.scan_with_context(content, ctx)
        assert result.file_path == "/src/config.py"

    def test_scan_multiple_secrets(self, scanner):
        """Test scanning file with multiple secrets."""
        content = """
AWS_ACCESS_KEY_ID = "AKIAJ5Q7R2D9K3G4N8M1"
GITHUB_TOKEN = "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
DATABASE_URL = "postgresql://user:p4ssw0rd@localhost:5432/db"
"""
        result = scanner.scan(content, file_path="/secrets.env")
        assert result.has_secrets is True
        assert len(result.matches) >= 3

    def test_scan_skips_placeholders(self, scanner):
        """Test that placeholders are not flagged."""
        content = """
# Example configuration
AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY_HERE"
GITHUB_TOKEN = "ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
"""
        result = scanner.scan(content, file_path="/example.env")
        # Placeholders should be low confidence or not matched
        high_confidence = [m for m in result.matches if m.confidence >= Confidence.HIGH]
        assert len(high_confidence) == 0

    def test_batch_scan(self, scanner):
        """Test scanning multiple files."""
        files = [
            ("/file1.py", "AWS_KEY = 'AKIAJ5Q7R2D9K3G4N8M1'"),
            ("/file2.py", "print('no secrets here')"),
            ("/file3.py", "GITHUB_TOKEN = 'ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8'"),
        ]
        results = scanner.scan_batch(files)
        assert len(results) == 3
        assert results[0].has_secrets is True
        assert results[1].has_secrets is False
        assert results[2].has_secrets is True

    def test_scan_statistics(self, scanner):
        """Test getting scan statistics."""
        # Perform some scans
        scanner.scan("content1", "/file1.py")
        scanner.scan("AKIAJ5Q7R2D9K3G4N8M1", "/file2.py")

        stats = scanner.get_statistics()
        assert "total_scans" in stats
        assert "avg_scan_time_ms" in stats
        assert stats["total_scans"] >= 2


# ============================================================================
# DeepScanConfig Tests
# ============================================================================


class TestDeepScanConfig:
    """Tests for DeepScanConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepScanConfig()
        assert config.enable_verification is True
        assert config.verification_timeout_s >= 5
        assert config.entropy_threshold >= 3.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeepScanConfig(
            enable_verification=False,
            verification_timeout_s=10,
            entropy_threshold=4.0,
        )
        assert config.enable_verification is False
        assert config.verification_timeout_s == 10
        assert config.entropy_threshold == 4.0


# ============================================================================
# VerificationResult Tests
# ============================================================================


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_verification_active(self):
        """Test active verification result."""
        result = VerificationResult(
            status=VerificationStatus.ACTIVE,
            verified_at=datetime.now(timezone.utc),
            details="AWS key is valid and active",
        )
        assert result.status == VerificationStatus.ACTIVE
        assert result.is_verified is True

    def test_verification_inactive(self):
        """Test inactive verification result."""
        result = VerificationResult(
            status=VerificationStatus.INACTIVE,
            verified_at=datetime.now(timezone.utc),
            details="Key has been rotated",
        )
        assert result.status == VerificationStatus.INACTIVE
        assert result.is_verified is True

    def test_verification_unknown(self):
        """Test unknown verification result."""
        result = VerificationResult(
            status=VerificationStatus.UNKNOWN,
            details="Could not verify",
        )
        assert result.status == VerificationStatus.UNKNOWN
        assert result.is_verified is False


# ============================================================================
# DeepScanResult Tests
# ============================================================================


class TestDeepScanResult:
    """Tests for DeepScanResult."""

    def test_result_creation(self):
        """Test creating a deep scan result."""
        result = DeepScanResult(
            entry_id="test-123",
            original_confidence=Confidence.MEDIUM,
            final_confidence=Confidence.HIGH,
            verification=None,
            entropy_score=4.2,
            context_analysis="High entropy string in config file",
            recommendations=["Rotate immediately", "Add to .gitignore"],
        )
        assert result.entry_id == "test-123"
        assert result.final_confidence == Confidence.HIGH
        assert len(result.recommendations) == 2

    def test_result_with_verification(self):
        """Test result with verification."""
        verification = VerificationResult(
            status=VerificationStatus.ACTIVE,
            verified_at=datetime.now(timezone.utc),
        )
        result = DeepScanResult(
            entry_id="test-456",
            original_confidence=Confidence.HIGH,
            final_confidence=Confidence.VERIFIED,
            verification=verification,
            entropy_score=4.8,
        )
        assert result.verification is not None
        assert result.final_confidence == Confidence.VERIFIED


# ============================================================================
# DeepScanner Tests
# ============================================================================


class TestDeepScanner:
    """Tests for DeepScanner."""

    @pytest.fixture
    def scanner(self):
        """Create a deep scanner instance."""
        return DeepScanner()

    def test_analyze_quarantined_entry(self, scanner):
        """Test analyzing a quarantined entry."""
        entry = QuarantineEntry(
            entry_id="test-001",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...N8M1",
            confidence=Confidence.MEDIUM,
            file_path="/config.py",
            line_number=10,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        content = 'AWS_KEY = "AKIAJ5Q7R2D9K3G4N8M1"'

        result = scanner.analyze(entry, content)
        assert result.entry_id == entry.entry_id
        assert result.original_confidence == Confidence.MEDIUM
        # Should upgrade confidence based on analysis
        assert result.final_confidence >= Confidence.MEDIUM

    def test_entropy_analysis(self, scanner):
        """Test entropy analysis."""
        # High entropy string
        high_entropy = "a8f3k2j9d7s5g6h4w2e1r9t8y7u6i5o4p3"
        score = scanner.calculate_entropy(high_entropy)
        assert score > 3.5

        # Low entropy string (repetitive)
        low_entropy = "aaaaaabbbbbbcccccc"
        score = scanner.calculate_entropy(low_entropy)
        assert score < 3.0

    def test_context_analysis(self, scanner):
        """Test context analysis for secrets."""
        entry = QuarantineEntry(
            entry_id="test-002",
            secret_type=SecretType.GENERIC_PASSWORD,
            redacted_value="pass...123",
            confidence=Confidence.MEDIUM,
            file_path="/test/fixtures.py",
            line_number=5,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        content = '''
# Test fixtures
TEST_PASSWORD = "testpassword123"
'''
        result = scanner.analyze(entry, content)
        # Test context should lower confidence
        assert "test" in result.context_analysis.lower()

    def test_recommendations_for_aws_key(self, scanner):
        """Test recommendations for AWS key."""
        entry = QuarantineEntry(
            entry_id="test-003",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...N8M1",
            confidence=Confidence.HIGH,
            file_path="/config.py",
            line_number=10,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        content = 'AWS_KEY = "AKIAJ5Q7R2D9K3G4N8M1"'

        result = scanner.analyze(entry, content)
        # Should include AWS-specific recommendations
        assert any("rotate" in r.lower() or "iam" in r.lower() for r in result.recommendations)

    def test_recommendations_for_private_key(self, scanner):
        """Test recommendations for private key."""
        entry = QuarantineEntry(
            entry_id="test-004",
            secret_type=SecretType.PRIVATE_KEY_RSA,
            redacted_value="-----...-----",
            confidence=Confidence.HIGH,
            file_path="/key.pem",
            line_number=1,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        content = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy
-----END RSA PRIVATE KEY-----"""

        result = scanner.analyze(entry, content)
        # Should include key-specific recommendations
        assert len(result.recommendations) > 0

    def test_batch_analysis(self, scanner):
        """Test analyzing multiple entries."""
        entries_with_content = [
            (
                QuarantineEntry(
                    entry_id=f"test-{i}",
                    secret_type=SecretType.GENERIC_SECRET,
                    redacted_value=f"sec...{i}",
                    confidence=Confidence.MEDIUM,
                    file_path=f"/file{i}.py",
                    line_number=i * 10,
                    state=QuarantineState.PENDING,
                    reason=QuarantineReason.FAST_SCAN_DETECTED,
                    created_at=datetime.now(timezone.utc),
                ),
                f'SECRET_{i} = "secret_value_{i}"',
            )
            for i in range(3)
        ]

        results = scanner.analyze_batch(entries_with_content)
        assert len(results) == 3

    def test_verification_disabled(self):
        """Test scanner with verification disabled."""
        config = DeepScanConfig(enable_verification=False)
        scanner = DeepScanner(config=config)

        entry = QuarantineEntry(
            entry_id="test-005",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...N8M1",
            confidence=Confidence.HIGH,
            file_path="/config.py",
            line_number=10,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        content = 'AWS_KEY = "AKIAJ5Q7R2D9K3G4N8M1"'

        result = scanner.analyze(entry, content)
        # No verification should be performed
        assert result.verification is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestScannerIntegration:
    """Integration tests for scanner workflow."""

    def test_fast_then_deep_scan(self):
        """Test fast scan followed by deep scan."""
        from openmemory.api.security.secrets.quarantine import QuarantineManager

        content = """
# Configuration file
AWS_ACCESS_KEY_ID = "AKIAJ5Q7R2D9K3G4N8M1"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYa1b2c3d4"
"""
        # Step 1: Fast scan
        fast_scanner = FastScanner()
        fast_result = fast_scanner.scan(content, file_path="/config.py")
        assert fast_result.has_secrets is True

        # Step 2: Quarantine detected secrets
        manager = QuarantineManager()
        entries = [manager.process_match(m) for m in fast_result.matches]
        assert len(entries) >= 1

        # Step 3: Deep scan each entry
        deep_scanner = DeepScanner()
        for entry in entries:
            deep_result = deep_scanner.analyze(entry, content)
            assert deep_result is not None
            assert deep_result.final_confidence >= entry.confidence

    def test_scan_performance_benchmark(self):
        """Benchmark scan performance."""
        scanner = FastScanner()

        # Generate test content
        content = "\n".join([
            f'VAR_{i} = "value_{i}"'
            for i in range(100)
        ])
        content += '\nGITHUB_TOKEN = "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"\n'

        # Run multiple scans and verify performance
        times = []
        for _ in range(10):
            result = scanner.scan(content, file_path="/benchmark.py")
            times.append(result.scan_time_ms)

        avg_time = sum(times) / len(times)
        # Average should be under 20ms per plan requirements
        assert avg_time < 20, f"Average scan time {avg_time}ms exceeds 20ms limit"
