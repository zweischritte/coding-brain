"""Secret detection subsystem for OpenMemory API.

This module implements tiered secret detection per section 5.5 (FR-011):
- Fast sync scan (<20ms): Quick pattern matching for common secrets
- Deep async scan: Comprehensive analysis with active verification
- Quarantine workflow: Staged handling of potential secrets
- Integration points: pre-commit, pre-index, CI/CD
"""

from .patterns import (
    SecretType,
    Confidence,
    SecretPattern,
    SecretMatch,
    SecretPatternLibrary,
)
from .quarantine import (
    QuarantineState,
    QuarantineReason,
    QuarantineDecision,
    QuarantineEntry,
    QuarantineConfig,
    QuarantineStore,
    MemoryQuarantineStore,
    QuarantineManager,
)
from .fast_scanner import (
    FastScanner,
    FastScanConfig,
    FastScanResult,
    ScanContext,
)
from .deep_scanner import (
    DeepScanner,
    DeepScanConfig,
    DeepScanResult,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    # Patterns
    "SecretType",
    "Confidence",
    "SecretPattern",
    "SecretMatch",
    "SecretPatternLibrary",
    # Quarantine
    "QuarantineState",
    "QuarantineReason",
    "QuarantineDecision",
    "QuarantineEntry",
    "QuarantineConfig",
    "QuarantineStore",
    "MemoryQuarantineStore",
    "QuarantineManager",
    # Fast Scanner
    "FastScanner",
    "FastScanConfig",
    "FastScanResult",
    "ScanContext",
    # Deep Scanner
    "DeepScanner",
    "DeepScanConfig",
    "DeepScanResult",
    "VerificationResult",
    "VerificationStatus",
]
