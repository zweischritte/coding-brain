"""Deep asynchronous secret scanner.

This module implements the deep async scan per section 5.5 (FR-011):
- Comprehensive analysis with active verification
- Context-aware confidence adjustment
- Entropy analysis
- Verification against cloud providers (when enabled)
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .patterns import Confidence, SecretType
from .quarantine import QuarantineEntry, QuarantineState


# ============================================================================
# Enums
# ============================================================================


class VerificationStatus(str, Enum):
    """Status of secret verification."""

    UNKNOWN = "unknown"  # Could not verify
    ACTIVE = "active"  # Secret is active/valid
    INACTIVE = "inactive"  # Secret is invalid/revoked
    EXPIRED = "expired"  # Secret has expired
    ERROR = "error"  # Verification failed with error


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class DeepScanConfig:
    """Configuration for deep scanning."""

    enable_verification: bool = True  # Enable active verification
    verification_timeout_s: int = 10  # Timeout for verification calls
    entropy_threshold: float = 3.5  # Minimum entropy for high confidence
    context_keywords_test: list[str] = field(
        default_factory=lambda: ["test", "mock", "fake", "example", "sample", "fixture"]
    )
    context_keywords_prod: list[str] = field(
        default_factory=lambda: ["prod", "production", "live", "deploy"]
    )


# ============================================================================
# Verification Result
# ============================================================================


@dataclass
class VerificationResult:
    """Result of secret verification."""

    status: VerificationStatus
    verified_at: datetime | None = None
    details: str = ""
    error: str | None = None

    @property
    def is_verified(self) -> bool:
        """Check if verification was successful."""
        return self.status in {VerificationStatus.ACTIVE, VerificationStatus.INACTIVE}


# ============================================================================
# Deep Scan Result
# ============================================================================


@dataclass
class DeepScanResult:
    """Result from deep scan analysis."""

    entry_id: str
    original_confidence: Confidence
    final_confidence: Confidence
    verification: VerificationResult | None = None
    entropy_score: float = 0.0
    context_analysis: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "original_confidence": self.original_confidence.name.lower(),
            "final_confidence": self.final_confidence.name.lower(),
            "verification": (
                {
                    "status": self.verification.status.value,
                    "verified_at": (
                        self.verification.verified_at.isoformat()
                        if self.verification.verified_at
                        else None
                    ),
                    "details": self.verification.details,
                }
                if self.verification
                else None
            ),
            "entropy_score": round(self.entropy_score, 2),
            "context_analysis": self.context_analysis,
            "recommendations": self.recommendations,
        }


# ============================================================================
# Deep Scanner
# ============================================================================


class DeepScanner:
    """Deep asynchronous secret scanner.

    Provides comprehensive analysis including:
    - Entropy analysis
    - Context-aware confidence adjustment
    - Active verification (when enabled)
    - Remediation recommendations
    """

    def __init__(self, config: DeepScanConfig | None = None):
        """Initialize the deep scanner.

        Args:
            config: Scanner configuration
        """
        self._config = config or DeepScanConfig()

    def analyze(
        self,
        entry: QuarantineEntry,
        content: str,
    ) -> DeepScanResult:
        """Analyze a quarantined entry.

        Args:
            entry: The quarantined entry
            content: Original file content

        Returns:
            Deep scan result
        """
        # Calculate entropy of the redacted value (approximate)
        entropy = self.calculate_entropy(entry.redacted_value)

        # Analyze context
        context_info = self._analyze_context(entry, content)

        # Determine final confidence
        final_confidence = self._calculate_final_confidence(
            entry.confidence,
            entropy,
            context_info,
        )

        # Perform verification if enabled
        verification = None
        if self._config.enable_verification and entry.state == QuarantineState.PENDING:
            # Note: Actual verification would call cloud provider APIs
            # For now, we just mark as unknown
            verification = VerificationResult(status=VerificationStatus.UNKNOWN)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            entry.secret_type,
            final_confidence,
            context_info,
        )

        return DeepScanResult(
            entry_id=entry.entry_id,
            original_confidence=entry.confidence,
            final_confidence=final_confidence,
            verification=verification,
            entropy_score=entropy,
            context_analysis=context_info,
            recommendations=recommendations,
        )

    def analyze_batch(
        self,
        entries_with_content: list[tuple[QuarantineEntry, str]],
    ) -> list[DeepScanResult]:
        """Analyze multiple entries.

        Args:
            entries_with_content: List of (entry, content) tuples

        Returns:
            List of deep scan results
        """
        return [
            self.analyze(entry, content)
            for entry, content in entries_with_content
        ]

    def calculate_entropy(self, value: str) -> float:
        """Calculate Shannon entropy of a string.

        Args:
            value: The string to analyze

        Returns:
            Entropy value (higher = more random)
        """
        if not value or len(value) < 2:
            return 0.0

        # Count character frequencies
        freq: dict[str, int] = {}
        for char in value:
            freq[char] = freq.get(char, 0) + 1

        # Calculate entropy
        length = len(value)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _analyze_context(self, entry: QuarantineEntry, content: str) -> str:
        """Analyze the context around a secret.

        Args:
            entry: The quarantined entry
            content: File content

        Returns:
            Context analysis string
        """
        context_parts = []

        # Check file path for context clues
        file_path = entry.file_path or ""
        file_lower = file_path.lower()

        # Test context indicators
        test_indicators = ["test", "spec", "fixture", "mock", "__tests__", "tests/"]
        if any(ind in file_lower for ind in test_indicators):
            context_parts.append("Test/fixture context detected")

        # Production context indicators
        prod_indicators = ["prod", "deploy", "live", ".env.prod"]
        if any(ind in file_lower for ind in prod_indicators):
            context_parts.append("Production context detected - HIGH PRIORITY")

        # Check content for context keywords
        content_lower = content.lower()
        for keyword in self._config.context_keywords_test:
            if keyword in content_lower:
                context_parts.append(f"Test keyword '{keyword}' found in content")
                break

        for keyword in self._config.context_keywords_prod:
            if keyword in content_lower:
                context_parts.append(f"Production keyword '{keyword}' found")
                break

        # Check for comment context
        lines = content.split("\n")
        if entry.line_number > 0 and entry.line_number <= len(lines):
            line = lines[entry.line_number - 1]
            if "#" in line or "//" in line:
                comment_start = line.find("#") if "#" in line else line.find("//")
                if "example" in line[comment_start:].lower():
                    context_parts.append("Appears to be example in comment")

        if not context_parts:
            context_parts.append("No specific context indicators found")

        return "; ".join(context_parts)

    def _calculate_final_confidence(
        self,
        original: Confidence,
        entropy: float,
        context: str,
    ) -> Confidence:
        """Calculate final confidence based on analysis.

        Args:
            original: Original confidence from pattern match
            entropy: Entropy score
            context: Context analysis string

        Returns:
            Adjusted confidence level
        """
        confidence = original

        # Boost confidence for high entropy
        if entropy > self._config.entropy_threshold and confidence < Confidence.HIGH:
            confidence = Confidence.HIGH

        # Reduce confidence for test context
        if "test" in context.lower() or "fixture" in context.lower():
            if confidence > Confidence.LOW:
                confidence = Confidence(max(1, confidence - 1))

        # Boost confidence for production context
        if "production" in context.lower() and confidence < Confidence.HIGH:
            confidence = Confidence.HIGH

        return confidence

    def _generate_recommendations(
        self,
        secret_type: SecretType,
        confidence: Confidence,
        context: str,
    ) -> list[str]:
        """Generate remediation recommendations.

        Args:
            secret_type: Type of secret
            confidence: Final confidence level
            context: Context analysis

        Returns:
            List of recommendations
        """
        recommendations = []

        # Type-specific recommendations
        if secret_type == SecretType.AWS_ACCESS_KEY:
            recommendations.extend([
                "Rotate AWS credentials immediately",
                "Review IAM policies for least privilege",
                "Enable AWS CloudTrail for access logging",
            ])
        elif secret_type == SecretType.AWS_SECRET_KEY:
            recommendations.extend([
                "Rotate AWS secret key immediately",
                "Use AWS Secrets Manager or Parameter Store",
            ])
        elif secret_type in (SecretType.GITHUB_PAT, SecretType.GITHUB_TOKEN):
            recommendations.extend([
                "Revoke GitHub token immediately",
                "Use fine-grained personal access tokens with minimal scopes",
                "Consider using GitHub Apps for automation",
            ])
        elif secret_type in (
            SecretType.PRIVATE_KEY_RSA,
            SecretType.PRIVATE_KEY_ECDSA,
            SecretType.PRIVATE_KEY_SSH,
            SecretType.PRIVATE_KEY_PGP,
        ):
            recommendations.extend([
                "Revoke and regenerate the private key",
                "Store private keys in a secure vault",
                "Never commit private keys to version control",
            ])
        elif secret_type == SecretType.DATABASE_URL:
            recommendations.extend([
                "Rotate database credentials",
                "Use environment variables or secret managers",
                "Enable database access logging",
            ])
        elif secret_type == SecretType.JWT_TOKEN:
            recommendations.extend([
                "Invalidate the JWT token if possible",
                "Review token expiration settings",
                "Implement token rotation",
            ])

        # General recommendations based on confidence
        if confidence >= Confidence.HIGH:
            recommendations.insert(0, "URGENT: This appears to be a real secret")

        # Context-based recommendations
        if "production" in context.lower():
            recommendations.insert(0, "CRITICAL: Secret in production context - immediate action required")
        elif "test" in context.lower():
            recommendations.append("Consider using mock values in tests")

        # Add .gitignore recommendation
        recommendations.append("Ensure sensitive files are in .gitignore")

        return recommendations
