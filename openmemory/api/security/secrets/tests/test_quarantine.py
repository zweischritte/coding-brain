"""Tests for secret quarantine system.

This module tests the quarantine workflow for detected secrets:
- Quarantine state management
- Secret status tracking
- Integration with scanning pipeline
- Retention and cleanup policies
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from openmemory.api.security.secrets.patterns import Confidence, SecretMatch, SecretType
from openmemory.api.security.secrets.quarantine import (
    QuarantineConfig,
    QuarantineEntry,
    QuarantineReason,
    QuarantineState,
    QuarantineStore,
    MemoryQuarantineStore,
    QuarantineDecision,
    QuarantineManager,
)


# ============================================================================
# QuarantineState Tests
# ============================================================================


class TestQuarantineState:
    """Tests for QuarantineState enumeration."""

    def test_quarantine_states_exist(self):
        """Test all quarantine states are defined."""
        assert QuarantineState.PENDING is not None
        assert QuarantineState.CONFIRMED is not None
        assert QuarantineState.FALSE_POSITIVE is not None
        assert QuarantineState.RELEASED is not None
        assert QuarantineState.DELETED is not None

    def test_terminal_states(self):
        """Test terminal states are identified correctly."""
        terminal = {QuarantineState.RELEASED, QuarantineState.DELETED}
        non_terminal = {QuarantineState.PENDING, QuarantineState.CONFIRMED, QuarantineState.FALSE_POSITIVE}

        for state in terminal:
            assert state.is_terminal() is True

        for state in non_terminal:
            assert state.is_terminal() is False


class TestQuarantineReason:
    """Tests for QuarantineReason enumeration."""

    def test_quarantine_reasons_exist(self):
        """Test all quarantine reasons are defined."""
        assert QuarantineReason.FAST_SCAN_DETECTED is not None
        assert QuarantineReason.DEEP_SCAN_DETECTED is not None
        assert QuarantineReason.VERIFIED_SECRET is not None
        assert QuarantineReason.HIGH_ENTROPY is not None
        assert QuarantineReason.MANUAL_REPORT is not None


# ============================================================================
# QuarantineEntry Tests
# ============================================================================


class TestQuarantineEntry:
    """Tests for QuarantineEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a quarantine entry."""
        now = datetime.now(timezone.utc)
        entry = QuarantineEntry(
            entry_id="test-123",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...CDEF",
            confidence=Confidence.HIGH,
            file_path="/path/to/file.py",
            line_number=42,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=now,
        )
        assert entry.entry_id == "test-123"
        assert entry.secret_type == SecretType.AWS_ACCESS_KEY
        assert entry.state == QuarantineState.PENDING
        assert entry.reason == QuarantineReason.FAST_SCAN_DETECTED
        assert entry.created_at == now

    def test_entry_with_content_hash(self):
        """Test entry with content hash."""
        entry = QuarantineEntry(
            entry_id="test-456",
            secret_type=SecretType.GITHUB_PAT,
            redacted_value="ghp_...xxxx",
            confidence=Confidence.HIGH,
            file_path="/path/to/config.yaml",
            line_number=10,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.DEEP_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
            content_hash="sha256:abc123",
        )
        assert entry.content_hash == "sha256:abc123"

    def test_entry_to_dict(self):
        """Test serializing entry to dictionary."""
        now = datetime.now(timezone.utc)
        entry = QuarantineEntry(
            entry_id="test-789",
            secret_type=SecretType.JWT_TOKEN,
            redacted_value="eyJh...xxxxx",
            confidence=Confidence.MEDIUM,
            file_path="/config.json",
            line_number=5,
            state=QuarantineState.CONFIRMED,
            reason=QuarantineReason.VERIFIED_SECRET,
            created_at=now,
        )
        d = entry.to_dict()
        assert d["entry_id"] == "test-789"
        assert d["secret_type"] == "jwt_token"
        assert d["state"] == "confirmed"
        assert d["reason"] == "verified_secret"
        # Should not include actual secret values
        assert "content_hash" not in d or d.get("content_hash") is None

    def test_entry_from_secret_match(self):
        """Test creating entry from SecretMatch."""
        match = SecretMatch(
            secret_type=SecretType.AWS_ACCESS_KEY,
            matched_value="AKIA1234567890ABCDEF",
            redacted_value="AKIA...CDEF",
            confidence=Confidence.HIGH,
            line_number=42,
            column_start=10,
            column_end=30,
            file_path="/path/to/file.py",
        )
        entry = QuarantineEntry.from_secret_match(
            match=match,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
        )
        assert entry.secret_type == SecretType.AWS_ACCESS_KEY
        assert entry.redacted_value == "AKIA...CDEF"
        assert entry.confidence == Confidence.HIGH
        assert entry.file_path == "/path/to/file.py"
        assert entry.line_number == 42
        assert entry.state == QuarantineState.PENDING
        assert entry.entry_id is not None


# ============================================================================
# QuarantineConfig Tests
# ============================================================================


class TestQuarantineConfig:
    """Tests for QuarantineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QuarantineConfig()
        assert config.auto_confirm_verified is True
        assert config.retention_days >= 30
        assert config.max_pending_entries >= 1000

    def test_custom_config(self):
        """Test custom configuration."""
        config = QuarantineConfig(
            auto_confirm_verified=False,
            retention_days=90,
            max_pending_entries=5000,
        )
        assert config.auto_confirm_verified is False
        assert config.retention_days == 90
        assert config.max_pending_entries == 5000


# ============================================================================
# MemoryQuarantineStore Tests
# ============================================================================


class TestMemoryQuarantineStore:
    """Tests for in-memory quarantine store."""

    @pytest.fixture
    def store(self):
        """Create a store instance."""
        return MemoryQuarantineStore()

    def test_add_entry(self, store):
        """Test adding an entry to the store."""
        entry = QuarantineEntry(
            entry_id="test-001",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...CDEF",
            confidence=Confidence.HIGH,
            file_path="/test.py",
            line_number=1,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        store.add(entry)
        assert store.get(entry.entry_id) == entry

    def test_get_nonexistent_entry(self, store):
        """Test getting a nonexistent entry."""
        result = store.get("nonexistent-id")
        assert result is None

    def test_update_entry(self, store):
        """Test updating an entry's state."""
        entry = QuarantineEntry(
            entry_id="test-002",
            secret_type=SecretType.GITHUB_PAT,
            redacted_value="ghp_...xxxx",
            confidence=Confidence.HIGH,
            file_path="/test.py",
            line_number=5,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.DEEP_SCAN_DETECTED,
            created_at=datetime.now(timezone.utc),
        )
        store.add(entry)

        # Update to confirmed
        store.update_state(entry.entry_id, QuarantineState.CONFIRMED)
        updated = store.get(entry.entry_id)
        assert updated.state == QuarantineState.CONFIRMED

    def test_delete_entry(self, store):
        """Test deleting an entry."""
        entry = QuarantineEntry(
            entry_id="test-003",
            secret_type=SecretType.JWT_TOKEN,
            redacted_value="eyJh...xxxx",
            confidence=Confidence.MEDIUM,
            file_path="/config.json",
            line_number=10,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.HIGH_ENTROPY,
            created_at=datetime.now(timezone.utc),
        )
        store.add(entry)
        assert store.get(entry.entry_id) is not None

        store.delete(entry.entry_id)
        assert store.get(entry.entry_id) is None

    def test_list_by_state(self, store):
        """Test listing entries by state."""
        now = datetime.now(timezone.utc)

        # Add entries with different states
        pending_entry = QuarantineEntry(
            entry_id="pending-001",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...AAAA",
            confidence=Confidence.HIGH,
            file_path="/test1.py",
            line_number=1,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=now,
        )
        confirmed_entry = QuarantineEntry(
            entry_id="confirmed-001",
            secret_type=SecretType.GITHUB_PAT,
            redacted_value="ghp_...BBBB",
            confidence=Confidence.HIGH,
            file_path="/test2.py",
            line_number=2,
            state=QuarantineState.CONFIRMED,
            reason=QuarantineReason.VERIFIED_SECRET,
            created_at=now,
        )
        store.add(pending_entry)
        store.add(confirmed_entry)

        pending = store.list_by_state(QuarantineState.PENDING)
        assert len(pending) == 1
        assert pending[0].entry_id == "pending-001"

        confirmed = store.list_by_state(QuarantineState.CONFIRMED)
        assert len(confirmed) == 1
        assert confirmed[0].entry_id == "confirmed-001"

    def test_list_by_file(self, store):
        """Test listing entries by file path."""
        now = datetime.now(timezone.utc)

        entry1 = QuarantineEntry(
            entry_id="file-001",
            secret_type=SecretType.AWS_ACCESS_KEY,
            redacted_value="AKIA...AAAA",
            confidence=Confidence.HIGH,
            file_path="/src/config.py",
            line_number=10,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=now,
        )
        entry2 = QuarantineEntry(
            entry_id="file-002",
            secret_type=SecretType.GITHUB_PAT,
            redacted_value="ghp_...BBBB",
            confidence=Confidence.HIGH,
            file_path="/src/config.py",
            line_number=20,
            state=QuarantineState.CONFIRMED,
            reason=QuarantineReason.VERIFIED_SECRET,
            created_at=now,
        )
        entry3 = QuarantineEntry(
            entry_id="file-003",
            secret_type=SecretType.JWT_TOKEN,
            redacted_value="eyJh...CCCC",
            confidence=Confidence.MEDIUM,
            file_path="/other/file.py",
            line_number=5,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.HIGH_ENTROPY,
            created_at=now,
        )
        store.add(entry1)
        store.add(entry2)
        store.add(entry3)

        results = store.list_by_file("/src/config.py")
        assert len(results) == 2

    def test_count_by_state(self, store):
        """Test counting entries by state."""
        now = datetime.now(timezone.utc)

        for i in range(3):
            store.add(QuarantineEntry(
                entry_id=f"pending-{i}",
                secret_type=SecretType.GENERIC_SECRET,
                redacted_value=f"secret...{i}",
                confidence=Confidence.MEDIUM,
                file_path="/test.py",
                line_number=i,
                state=QuarantineState.PENDING,
                reason=QuarantineReason.FAST_SCAN_DETECTED,
                created_at=now,
            ))

        for i in range(2):
            store.add(QuarantineEntry(
                entry_id=f"confirmed-{i}",
                secret_type=SecretType.GENERIC_SECRET,
                redacted_value=f"secret...{i}",
                confidence=Confidence.HIGH,
                file_path="/test.py",
                line_number=i + 10,
                state=QuarantineState.CONFIRMED,
                reason=QuarantineReason.VERIFIED_SECRET,
                created_at=now,
            ))

        counts = store.count_by_state()
        assert counts[QuarantineState.PENDING] == 3
        assert counts[QuarantineState.CONFIRMED] == 2

    def test_cleanup_expired(self, store):
        """Test cleaning up expired entries."""
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=60)

        # Add old entry
        old_entry = QuarantineEntry(
            entry_id="old-001",
            secret_type=SecretType.GENERIC_SECRET,
            redacted_value="old...secret",
            confidence=Confidence.LOW,
            file_path="/old.py",
            line_number=1,
            state=QuarantineState.FALSE_POSITIVE,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=old_date,
        )
        store.add(old_entry)

        # Add recent entry
        recent_entry = QuarantineEntry(
            entry_id="recent-001",
            secret_type=SecretType.GENERIC_SECRET,
            redacted_value="new...secret",
            confidence=Confidence.LOW,
            file_path="/new.py",
            line_number=1,
            state=QuarantineState.PENDING,
            reason=QuarantineReason.FAST_SCAN_DETECTED,
            created_at=now,
        )
        store.add(recent_entry)

        # Cleanup with 30-day retention
        deleted = store.cleanup_expired(retention_days=30)
        assert deleted == 1
        assert store.get("old-001") is None
        assert store.get("recent-001") is not None


# ============================================================================
# QuarantineDecision Tests
# ============================================================================


class TestQuarantineDecision:
    """Tests for QuarantineDecision enumeration."""

    def test_decision_values(self):
        """Test all decision values exist."""
        assert QuarantineDecision.QUARANTINE is not None
        assert QuarantineDecision.ALLOW is not None
        assert QuarantineDecision.BLOCK is not None

    def test_decision_from_confidence(self):
        """Test mapping confidence to decision."""
        # High confidence should quarantine
        assert QuarantineDecision.from_confidence(Confidence.HIGH) == QuarantineDecision.QUARANTINE
        assert QuarantineDecision.from_confidence(Confidence.VERIFIED) == QuarantineDecision.BLOCK

        # Low confidence might be allowed
        assert QuarantineDecision.from_confidence(Confidence.LOW) == QuarantineDecision.ALLOW


# ============================================================================
# QuarantineManager Tests
# ============================================================================


class TestQuarantineManager:
    """Tests for QuarantineManager."""

    @pytest.fixture
    def manager(self):
        """Create a manager instance."""
        config = QuarantineConfig(retention_days=30)
        return QuarantineManager(config=config)

    def test_process_secret_match(self, manager):
        """Test processing a secret match."""
        match = SecretMatch(
            secret_type=SecretType.AWS_ACCESS_KEY,
            matched_value="AKIA1234567890ABCDEF",
            redacted_value="AKIA...CDEF",
            confidence=Confidence.HIGH,
            line_number=42,
            file_path="/test.py",
        )

        entry = manager.process_match(match)
        assert entry is not None
        assert entry.secret_type == SecretType.AWS_ACCESS_KEY
        assert entry.state == QuarantineState.PENDING

    def test_process_verified_secret_auto_confirm(self):
        """Test auto-confirming verified secrets."""
        config = QuarantineConfig(auto_confirm_verified=True)
        manager = QuarantineManager(config=config)

        match = SecretMatch(
            secret_type=SecretType.AWS_ACCESS_KEY,
            matched_value="AKIA1234567890ABCDEF",
            redacted_value="AKIA...CDEF",
            confidence=Confidence.VERIFIED,
            line_number=42,
            file_path="/test.py",
        )

        entry = manager.process_match(match)
        assert entry.state == QuarantineState.CONFIRMED

    def test_mark_false_positive(self, manager):
        """Test marking an entry as false positive."""
        match = SecretMatch(
            secret_type=SecretType.GENERIC_SECRET,
            matched_value="not-a-secret",
            redacted_value="not-...ret",
            confidence=Confidence.MEDIUM,
            line_number=10,
            file_path="/test.py",
        )

        entry = manager.process_match(match)
        manager.mark_false_positive(entry.entry_id, reason="Test data")

        updated = manager.get_entry(entry.entry_id)
        assert updated.state == QuarantineState.FALSE_POSITIVE

    def test_release_entry(self, manager):
        """Test releasing an entry."""
        match = SecretMatch(
            secret_type=SecretType.GENERIC_PASSWORD,
            matched_value="password123",
            redacted_value="pass...123",
            confidence=Confidence.MEDIUM,
            line_number=5,
            file_path="/test.py",
        )

        entry = manager.process_match(match)
        manager.release(entry.entry_id, reason="Rotated secret")

        updated = manager.get_entry(entry.entry_id)
        assert updated.state == QuarantineState.RELEASED

    def test_delete_entry(self, manager):
        """Test deleting quarantined content."""
        match = SecretMatch(
            secret_type=SecretType.PRIVATE_KEY_RSA,
            matched_value="-----BEGIN RSA PRIVATE KEY-----",
            redacted_value="-----...-----",
            confidence=Confidence.HIGH,
            line_number=1,
            file_path="/keys.pem",
        )

        entry = manager.process_match(match)
        manager.delete(entry.entry_id, reason="Secret removed from repo")

        updated = manager.get_entry(entry.entry_id)
        assert updated.state == QuarantineState.DELETED

    def test_get_statistics(self, manager):
        """Test getting quarantine statistics."""
        # Add some entries
        for i in range(3):
            match = SecretMatch(
                secret_type=SecretType.GENERIC_SECRET,
                matched_value=f"secret-{i}",
                redacted_value=f"sec...{i}",
                confidence=Confidence.MEDIUM,
                line_number=i,
                file_path="/test.py",
            )
            manager.process_match(match)

        stats = manager.get_statistics()
        assert "total_entries" in stats
        assert "by_state" in stats
        assert "by_type" in stats
        assert stats["total_entries"] >= 3

    def test_list_pending(self, manager):
        """Test listing pending entries."""
        match = SecretMatch(
            secret_type=SecretType.JWT_TOKEN,
            matched_value="eyJhbGciOiJIUzI1NiJ9.xxxx.xxxx",
            redacted_value="eyJh...xxxx",
            confidence=Confidence.HIGH,
            line_number=1,
            file_path="/test.py",
        )
        manager.process_match(match)

        pending = manager.list_pending()
        assert len(pending) >= 1

    def test_list_by_file(self, manager):
        """Test listing entries for a specific file."""
        for i in range(2):
            match = SecretMatch(
                secret_type=SecretType.GENERIC_SECRET,
                matched_value=f"secret-{i}",
                redacted_value=f"sec...{i}",
                confidence=Confidence.MEDIUM,
                line_number=i * 10,
                file_path="/src/config.py",
            )
            manager.process_match(match)

        # Add one in a different file
        other_match = SecretMatch(
            secret_type=SecretType.GENERIC_SECRET,
            matched_value="other-secret",
            redacted_value="oth...ret",
            confidence=Confidence.MEDIUM,
            line_number=1,
            file_path="/other/file.py",
        )
        manager.process_match(other_match)

        results = manager.list_by_file("/src/config.py")
        assert len(results) == 2

    def test_cleanup(self, manager):
        """Test cleanup of expired entries."""
        # This would normally clean up old entries
        deleted = manager.cleanup()
        assert deleted >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestQuarantineIntegration:
    """Integration tests for quarantine workflow."""

    def test_full_quarantine_workflow(self):
        """Test complete quarantine workflow from detection to resolution."""
        config = QuarantineConfig(auto_confirm_verified=True)
        manager = QuarantineManager(config=config)

        # Step 1: Fast scan detects potential secret
        match = SecretMatch(
            secret_type=SecretType.AWS_ACCESS_KEY,
            matched_value="AKIAJ5Q7R2D9K3G4N8M1",
            redacted_value="AKIA...N8M1",
            confidence=Confidence.HIGH,
            line_number=42,
            file_path="/src/config.py",
        )
        entry = manager.process_match(match)
        assert entry.state == QuarantineState.PENDING

        # Step 2: Deep scan confirms it's a real secret
        manager.confirm(entry.entry_id, reason="Active AWS key verified")
        updated = manager.get_entry(entry.entry_id)
        assert updated.state == QuarantineState.CONFIRMED

        # Step 3: Security team rotates the key and releases
        manager.release(entry.entry_id, reason="Key rotated, old key invalidated")
        final = manager.get_entry(entry.entry_id)
        assert final.state == QuarantineState.RELEASED

    def test_false_positive_workflow(self):
        """Test false positive resolution workflow."""
        manager = QuarantineManager()

        # Detected as potential secret
        match = SecretMatch(
            secret_type=SecretType.GENERIC_SECRET,
            matched_value="placeholder_value",
            redacted_value="plac...lue",
            confidence=Confidence.MEDIUM,
            line_number=10,
            file_path="/test/fixtures.py",
        )
        entry = manager.process_match(match)

        # Marked as false positive after review
        manager.mark_false_positive(
            entry.entry_id,
            reason="Test fixture, not a real secret",
        )
        updated = manager.get_entry(entry.entry_id)
        assert updated.state == QuarantineState.FALSE_POSITIVE

    def test_batch_processing(self):
        """Test processing multiple matches at once."""
        manager = QuarantineManager()

        matches = [
            SecretMatch(
                secret_type=SecretType.AWS_ACCESS_KEY,
                matched_value=f"AKIAJ5Q7R2D9K3G4N{i:03d}",
                redacted_value=f"AKIA...N{i:03d}",
                confidence=Confidence.HIGH,
                line_number=i * 10,
                file_path="/config.py",
            )
            for i in range(5)
        ]

        entries = manager.process_batch(matches)
        assert len(entries) == 5

        pending = manager.list_pending()
        assert len(pending) >= 5
