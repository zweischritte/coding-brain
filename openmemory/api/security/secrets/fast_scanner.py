"""Fast synchronous secret scanner.

This module implements the fast sync scan per section 5.5 (FR-011):
- Target: <20ms scan time
- Quick pattern matching for common secrets
- Quarantine potential secrets for async deep scan
"""

import time
from dataclasses import dataclass, field
from typing import Any

from .patterns import Confidence, SecretMatch, SecretPatternLibrary


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FastScanConfig:
    """Configuration for fast scanning."""

    timeout_ms: float = 20.0  # Max scan time in milliseconds
    min_confidence: Confidence = Confidence.MEDIUM  # Minimum confidence to report
    skip_large_files: bool = True  # Skip files over size limit
    max_file_size_kb: int = 2048  # Max file size to scan (2MB)
    enable_statistics: bool = True  # Track scan statistics


@dataclass
class ScanContext:
    """Context for a scan operation."""

    file_path: str
    repo_id: str | None = None
    commit_sha: str | None = None
    branch: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Scan Result
# ============================================================================


@dataclass
class FastScanResult:
    """Result from a fast scan."""

    matches: list[SecretMatch]
    scan_time_ms: float
    file_path: str
    file_size_bytes: int
    patterns_checked: int
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def has_secrets(self) -> bool:
        """Check if any secrets were found."""
        return len(self.matches) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "secrets_found": len(self.matches),
            "has_secrets": self.has_secrets,
            "scan_time_ms": round(self.scan_time_ms, 2),
            "file_size_bytes": self.file_size_bytes,
            "patterns_checked": self.patterns_checked,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "matches": [
                {
                    "type": m.secret_type.value,
                    "redacted": m.redacted_value,
                    "confidence": m.confidence.name.lower(),
                    "line": m.line_number,
                }
                for m in self.matches
            ],
        }


# ============================================================================
# Fast Scanner
# ============================================================================


class FastScanner:
    """Fast synchronous secret scanner.

    Designed for quick pattern-based detection with <20ms target.
    Detected secrets are quarantined for deep async analysis.
    """

    def __init__(self, config: FastScanConfig | None = None):
        """Initialize the scanner.

        Args:
            config: Scanner configuration
        """
        self._config = config or FastScanConfig()
        self._library = SecretPatternLibrary()
        self._pattern_count = len(self._library.list_patterns())

        # Statistics
        self._total_scans = 0
        self._total_scan_time_ms = 0.0
        self._secrets_found = 0

    def scan(self, content: str, file_path: str = "") -> FastScanResult:
        """Scan content for secrets.

        Args:
            content: File content to scan
            file_path: Path to the file (for context)

        Returns:
            Scan result with detected secrets
        """
        start_time = time.perf_counter()

        # Check file size
        content_size = len(content.encode("utf-8"))
        if self._config.skip_large_files:
            max_bytes = self._config.max_file_size_kb * 1024
            if content_size > max_bytes:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return FastScanResult(
                    matches=[],
                    scan_time_ms=elapsed_ms,
                    file_path=file_path,
                    file_size_bytes=content_size,
                    patterns_checked=0,
                    skipped=True,
                    skip_reason=f"File size {content_size} exceeds limit {max_bytes}",
                )

        # Perform pattern matching
        matches = self._library.scan(content)

        # Filter by minimum confidence
        filtered = [m for m in matches if m.confidence >= self._config.min_confidence]

        # Add file path to matches
        for match in filtered:
            match.file_path = file_path

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        if self._config.enable_statistics:
            self._total_scans += 1
            self._total_scan_time_ms += elapsed_ms
            self._secrets_found += len(filtered)

        return FastScanResult(
            matches=filtered,
            scan_time_ms=elapsed_ms,
            file_path=file_path,
            file_size_bytes=content_size,
            patterns_checked=self._pattern_count,
        )

    def scan_with_context(self, content: str, context: ScanContext) -> FastScanResult:
        """Scan content with additional context.

        Args:
            content: File content to scan
            context: Scan context with metadata

        Returns:
            Scan result
        """
        return self.scan(content, file_path=context.file_path)

    def scan_batch(
        self, files: list[tuple[str, str]]
    ) -> list[FastScanResult]:
        """Scan multiple files.

        Args:
            files: List of (file_path, content) tuples

        Returns:
            List of scan results
        """
        return [self.scan(content, file_path) for file_path, content in files]

    def get_statistics(self) -> dict[str, Any]:
        """Get scanner statistics.

        Returns:
            Dictionary with scan statistics
        """
        avg_time = (
            self._total_scan_time_ms / self._total_scans
            if self._total_scans > 0
            else 0.0
        )
        return {
            "total_scans": self._total_scans,
            "total_scan_time_ms": round(self._total_scan_time_ms, 2),
            "avg_scan_time_ms": round(avg_time, 2),
            "secrets_found": self._secrets_found,
            "patterns_loaded": self._pattern_count,
        }

    def reset_statistics(self) -> None:
        """Reset scanner statistics."""
        self._total_scans = 0
        self._total_scan_time_ms = 0.0
        self._secrets_found = 0
