"""Secret quarantine management.

This module implements the quarantine workflow for detected secrets per section 5.5:
- Quarantine state management (pending, confirmed, false positive, released, deleted)
- Secret status tracking
- Integration with scanning pipeline
- Retention and cleanup policies
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from .patterns import Confidence, SecretMatch, SecretType


# ============================================================================
# Enums
# ============================================================================


class QuarantineState(str, Enum):
    """State of a quarantined secret."""

    PENDING = "pending"  # Awaiting review
    CONFIRMED = "confirmed"  # Verified as real secret
    FALSE_POSITIVE = "false_positive"  # Marked as not a secret
    RELEASED = "released"  # Secret rotated/invalidated
    DELETED = "deleted"  # Content deleted from system

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in {QuarantineState.RELEASED, QuarantineState.DELETED}


class QuarantineReason(str, Enum):
    """Reason for quarantine."""

    FAST_SCAN_DETECTED = "fast_scan_detected"
    DEEP_SCAN_DETECTED = "deep_scan_detected"
    VERIFIED_SECRET = "verified_secret"
    HIGH_ENTROPY = "high_entropy"
    MANUAL_REPORT = "manual_report"


class QuarantineDecision(str, Enum):
    """Decision for handling a detected secret."""

    QUARANTINE = "quarantine"  # Put in quarantine for review
    ALLOW = "allow"  # Allow (low confidence)
    BLOCK = "block"  # Block immediately (verified)

    @classmethod
    def from_confidence(cls, confidence: Confidence) -> "QuarantineDecision":
        """Map confidence level to decision.

        Args:
            confidence: The confidence level

        Returns:
            Appropriate quarantine decision
        """
        if confidence == Confidence.VERIFIED:
            return cls.BLOCK
        elif confidence >= Confidence.MEDIUM:
            return cls.QUARANTINE
        return cls.ALLOW


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class QuarantineEntry:
    """An entry in the quarantine store."""

    entry_id: str
    secret_type: SecretType
    redacted_value: str
    confidence: Confidence
    file_path: str | None
    line_number: int
    state: QuarantineState
    reason: QuarantineReason
    created_at: datetime
    updated_at: datetime | None = None
    content_hash: str | None = None
    notes: str = ""
    reviewed_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "secret_type": self.secret_type.value,
            "redacted_value": self.redacted_value,
            "confidence": self.confidence.name.lower(),
            "file_path": self.file_path,
            "line_number": self.line_number,
            "state": self.state.value,
            "reason": self.reason.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "notes": self.notes,
            "reviewed_by": self.reviewed_by,
        }

    @classmethod
    def from_secret_match(
        cls,
        match: SecretMatch,
        reason: QuarantineReason,
        state: QuarantineState = QuarantineState.PENDING,
    ) -> "QuarantineEntry":
        """Create an entry from a secret match.

        Args:
            match: The detected secret match
            reason: Reason for quarantine
            state: Initial state

        Returns:
            New quarantine entry
        """
        return cls(
            entry_id=str(uuid.uuid4()),
            secret_type=match.secret_type,
            redacted_value=match.redacted_value,
            confidence=match.confidence,
            file_path=match.file_path,
            line_number=match.line_number,
            state=state,
            reason=reason,
            created_at=datetime.now(timezone.utc),
        )


@dataclass
class QuarantineConfig:
    """Configuration for quarantine management."""

    auto_confirm_verified: bool = True  # Auto-confirm verified secrets
    retention_days: int = 90  # Days to retain resolved entries
    max_pending_entries: int = 10000  # Max pending entries before alert


# ============================================================================
# Quarantine Store Interface
# ============================================================================


class QuarantineStore:
    """Interface for quarantine storage."""

    def add(self, entry: QuarantineEntry) -> None:
        """Add an entry to the store."""
        raise NotImplementedError

    def get(self, entry_id: str) -> QuarantineEntry | None:
        """Get an entry by ID."""
        raise NotImplementedError

    def update_state(
        self,
        entry_id: str,
        state: QuarantineState,
        notes: str = "",
        reviewed_by: str | None = None,
    ) -> None:
        """Update an entry's state."""
        raise NotImplementedError

    def delete(self, entry_id: str) -> None:
        """Delete an entry."""
        raise NotImplementedError

    def list_by_state(self, state: QuarantineState) -> list[QuarantineEntry]:
        """List entries by state."""
        raise NotImplementedError

    def list_by_file(self, file_path: str) -> list[QuarantineEntry]:
        """List entries for a file."""
        raise NotImplementedError

    def count_by_state(self) -> dict[QuarantineState, int]:
        """Count entries by state."""
        raise NotImplementedError

    def cleanup_expired(self, retention_days: int) -> int:
        """Clean up expired entries. Returns count deleted."""
        raise NotImplementedError


# ============================================================================
# Memory Quarantine Store
# ============================================================================


class MemoryQuarantineStore(QuarantineStore):
    """In-memory quarantine store for development/testing."""

    def __init__(self):
        """Initialize the store."""
        self._entries: dict[str, QuarantineEntry] = {}

    def add(self, entry: QuarantineEntry) -> None:
        """Add an entry to the store."""
        self._entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> QuarantineEntry | None:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def update_state(
        self,
        entry_id: str,
        state: QuarantineState,
        notes: str = "",
        reviewed_by: str | None = None,
    ) -> None:
        """Update an entry's state."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.state = state
            entry.updated_at = datetime.now(timezone.utc)
            if notes:
                entry.notes = notes
            if reviewed_by:
                entry.reviewed_by = reviewed_by

    def delete(self, entry_id: str) -> None:
        """Delete an entry."""
        self._entries.pop(entry_id, None)

    def list_by_state(self, state: QuarantineState) -> list[QuarantineEntry]:
        """List entries by state."""
        return [e for e in self._entries.values() if e.state == state]

    def list_by_file(self, file_path: str) -> list[QuarantineEntry]:
        """List entries for a file."""
        return [e for e in self._entries.values() if e.file_path == file_path]

    def count_by_state(self) -> dict[QuarantineState, int]:
        """Count entries by state."""
        counts: dict[QuarantineState, int] = {}
        for entry in self._entries.values():
            counts[entry.state] = counts.get(entry.state, 0) + 1
        return counts

    def cleanup_expired(self, retention_days: int) -> int:
        """Clean up expired entries."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        to_delete = [
            entry_id
            for entry_id, entry in self._entries.items()
            if entry.created_at < cutoff
        ]
        for entry_id in to_delete:
            del self._entries[entry_id]
        return len(to_delete)


# ============================================================================
# Quarantine Manager
# ============================================================================


class QuarantineManager:
    """Manages the quarantine workflow.

    This class provides high-level operations for:
    - Processing detected secrets
    - Managing quarantine state
    - Generating statistics
    """

    def __init__(
        self,
        config: QuarantineConfig | None = None,
        store: QuarantineStore | None = None,
    ):
        """Initialize the manager.

        Args:
            config: Quarantine configuration
            store: Storage backend (defaults to in-memory)
        """
        self._config = config or QuarantineConfig()
        self._store = store or MemoryQuarantineStore()

    def process_match(self, match: SecretMatch) -> QuarantineEntry:
        """Process a detected secret match.

        Args:
            match: The detected secret

        Returns:
            Created quarantine entry
        """
        # Determine initial state
        if self._config.auto_confirm_verified and match.confidence == Confidence.VERIFIED:
            state = QuarantineState.CONFIRMED
            reason = QuarantineReason.VERIFIED_SECRET
        else:
            state = QuarantineState.PENDING
            reason = QuarantineReason.FAST_SCAN_DETECTED

        entry = QuarantineEntry.from_secret_match(
            match=match,
            reason=reason,
            state=state,
        )
        self._store.add(entry)
        return entry

    def process_batch(self, matches: list[SecretMatch]) -> list[QuarantineEntry]:
        """Process multiple secret matches.

        Args:
            matches: List of detected secrets

        Returns:
            List of created entries
        """
        return [self.process_match(m) for m in matches]

    def get_entry(self, entry_id: str) -> QuarantineEntry | None:
        """Get a quarantine entry by ID.

        Args:
            entry_id: Entry identifier

        Returns:
            The entry or None if not found
        """
        return self._store.get(entry_id)

    def confirm(self, entry_id: str, reason: str = "") -> None:
        """Confirm an entry as a real secret.

        Args:
            entry_id: Entry to confirm
            reason: Optional reason for confirmation
        """
        self._store.update_state(
            entry_id,
            QuarantineState.CONFIRMED,
            notes=reason,
        )

    def mark_false_positive(self, entry_id: str, reason: str = "") -> None:
        """Mark an entry as a false positive.

        Args:
            entry_id: Entry to mark
            reason: Reason for marking as false positive
        """
        self._store.update_state(
            entry_id,
            QuarantineState.FALSE_POSITIVE,
            notes=reason,
        )

    def release(self, entry_id: str, reason: str = "") -> None:
        """Release an entry (secret rotated/invalidated).

        Args:
            entry_id: Entry to release
            reason: Reason for release
        """
        self._store.update_state(
            entry_id,
            QuarantineState.RELEASED,
            notes=reason,
        )

    def delete(self, entry_id: str, reason: str = "") -> None:
        """Mark an entry as deleted (content removed).

        Args:
            entry_id: Entry to mark deleted
            reason: Reason for deletion
        """
        self._store.update_state(
            entry_id,
            QuarantineState.DELETED,
            notes=reason,
        )

    def list_pending(self) -> list[QuarantineEntry]:
        """List all pending entries.

        Returns:
            List of pending entries
        """
        return self._store.list_by_state(QuarantineState.PENDING)

    def list_by_file(self, file_path: str) -> list[QuarantineEntry]:
        """List entries for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of entries for the file
        """
        return self._store.list_by_file(file_path)

    def get_statistics(self) -> dict[str, Any]:
        """Get quarantine statistics.

        Returns:
            Dictionary with statistics
        """
        counts = self._store.count_by_state()
        total = sum(counts.values())

        # Count by type
        by_type: dict[str, int] = {}
        for state in QuarantineState:
            for entry in self._store.list_by_state(state):
                type_name = entry.secret_type.value
                by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_entries": total,
            "by_state": {k.value: v for k, v in counts.items()},
            "by_type": by_type,
        }

    def cleanup(self) -> int:
        """Clean up expired entries.

        Returns:
            Number of entries deleted
        """
        return self._store.cleanup_expired(self._config.retention_days)
