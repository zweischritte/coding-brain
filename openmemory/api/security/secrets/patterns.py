"""Secret detection pattern library.

This module implements the pattern library for detecting various secret types:
- API keys (AWS, GCP, Azure, GitHub, etc.)
- Private keys (RSA, ECDSA, PGP, SSH)
- Database credentials and connection strings
- JWT tokens and bearer tokens
- Generic high-entropy strings

Per section 5.5 (FR-011): Tiered scanning with fast sync scan (<20ms).
"""

import math
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable


# ============================================================================
# Enums
# ============================================================================


class SecretType(str, Enum):
    """Types of secrets that can be detected."""

    # AWS
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"

    # GitHub
    GITHUB_TOKEN = "github_token"
    GITHUB_PAT = "github_pat"
    GITLAB_TOKEN = "gitlab_token"

    # Private Keys
    PRIVATE_KEY_RSA = "private_key_rsa"
    PRIVATE_KEY_ECDSA = "private_key_ecdsa"
    PRIVATE_KEY_PGP = "private_key_pgp"
    PRIVATE_KEY_SSH = "private_key_ssh"

    # GCP
    GCP_API_KEY = "gcp_api_key"
    GCP_SERVICE_ACCOUNT = "gcp_service_account"

    # Azure
    AZURE_CONNECTION_STRING = "azure_connection_string"
    AZURE_CLIENT_SECRET = "azure_client_secret"

    # Database
    DATABASE_URL = "database_url"
    JDBC_CONNECTION = "jdbc_connection"

    # Tokens
    JWT_TOKEN = "jwt_token"
    BEARER_TOKEN = "bearer_token"
    SLACK_TOKEN = "slack_token"
    STRIPE_KEY = "stripe_key"
    TWILIO_KEY = "twilio_key"
    SENDGRID_KEY = "sendgrid_key"

    # Package Registries
    NPM_TOKEN = "npm_token"
    PYPI_TOKEN = "pypi_token"
    DOCKER_AUTH = "docker_auth"

    # Generic
    GENERIC_SECRET = "generic_secret"
    GENERIC_PASSWORD = "generic_password"
    HIGH_ENTROPY = "high_entropy"


class Confidence(IntEnum):
    """Confidence level for a detected secret."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERIFIED = 4


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SecretPattern:
    """A pattern for detecting a specific type of secret."""

    pattern: str
    secret_type: SecretType
    description: str
    confidence: Confidence
    capture_group: int = 0
    validator: Callable[[str], bool] | None = None
    compiled: re.Pattern | None = field(default=None, repr=False)

    def __post_init__(self):
        """Compile the regex pattern."""
        if self.compiled is None:
            self.compiled = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)


@dataclass
class SecretMatch:
    """A detected secret match."""

    secret_type: SecretType
    matched_value: str
    redacted_value: str
    confidence: Confidence
    line_number: int
    column_start: int = 0
    column_end: int = 0
    pattern_description: str = ""
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization (without exposing secret)."""
        return {
            "secret_type": self.secret_type.value,
            "redacted_value": self.redacted_value,
            "confidence": self.confidence.name.lower(),
            "line_number": self.line_number,
            "column_start": self.column_start,
            "column_end": self.column_end,
            "pattern_description": self.pattern_description,
            "file_path": self.file_path,
        }


# ============================================================================
# Helper Functions
# ============================================================================


def _redact_value(value: str, prefix_len: int = 4, suffix_len: int = 4) -> str:
    """Redact a secret value while preserving prefix and suffix.

    Args:
        value: The secret value to redact
        prefix_len: Number of characters to show at start
        suffix_len: Number of characters to show at end

    Returns:
        Redacted string like "AKIA...CDEF"
    """
    if len(value) <= prefix_len + suffix_len + 3:
        return value[:prefix_len] + "..."

    return f"{value[:prefix_len]}...{value[-suffix_len:]}"


def _calculate_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string.

    Args:
        s: Input string

    Returns:
        Entropy value (higher = more random)
    """
    if not s:
        return 0.0

    # Count character frequencies
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1

    # Calculate entropy
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)

    return entropy


def _is_placeholder(value: str) -> bool:
    """Check if a value looks like a placeholder.

    Args:
        value: The value to check

    Returns:
        True if value looks like a placeholder
    """
    placeholder_indicators = [
        "EXAMPLE",
        "YOUR_",
        "_HERE",
        "PLACEHOLDER",
        "XXXXXXXX",
        "xxxxxxxx",
        "12345678",
        "test",
        "mock",
        "fake",
        "dummy",
        "<YOUR",
        "${",
        "{{",
    ]

    value_upper = value.upper()
    value_lower = value.lower()

    for indicator in placeholder_indicators:
        if indicator in value_upper or indicator in value_lower:
            return True

    # All same character
    if len(set(value)) <= 2:
        return True

    return False


def _get_line_number(text: str, position: int) -> int:
    """Get line number for a position in text.

    Args:
        text: The full text
        position: Character position

    Returns:
        Line number (1-indexed)
    """
    return text[:position].count("\n") + 1


def _get_column_position(text: str, position: int) -> int:
    """Get column position for a position in text.

    Args:
        text: The full text
        position: Character position

    Returns:
        Column position (0-indexed)
    """
    last_newline = text.rfind("\n", 0, position)
    if last_newline == -1:
        return position
    return position - last_newline - 1


# ============================================================================
# SecretPatternLibrary
# ============================================================================


class SecretPatternLibrary:
    """Library of patterns for detecting secrets.

    This class provides fast pattern-based secret detection with:
    - Comprehensive patterns for common secret types
    - Confidence scoring based on pattern specificity
    - False positive reduction for placeholders and examples
    - Position tracking for matches
    """

    # Default patterns organized by category
    DEFAULT_PATTERNS = [
        # =====================================================================
        # AWS
        # =====================================================================
        SecretPattern(
            pattern=r"(?:^|[^A-Z0-9])(AKIA[0-9A-Z]{16})(?:[^A-Z0-9]|$)",
            secret_type=SecretType.AWS_ACCESS_KEY,
            description="AWS Access Key ID",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(?:aws_secret_access_key|aws_secret_key)\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
            secret_type=SecretType.AWS_SECRET_KEY,
            description="AWS Secret Access Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # GitHub
        # =====================================================================
        SecretPattern(
            pattern=r"(ghp_[a-zA-Z0-9]{36})",
            secret_type=SecretType.GITHUB_PAT,
            description="GitHub Personal Access Token (Classic)",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59})",
            secret_type=SecretType.GITHUB_PAT,
            description="GitHub Personal Access Token (Fine-grained)",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(gho_[a-zA-Z0-9]{36})",
            secret_type=SecretType.GITHUB_TOKEN,
            description="GitHub OAuth Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(ghu_[a-zA-Z0-9]{36})",
            secret_type=SecretType.GITHUB_TOKEN,
            description="GitHub App User Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(ghs_[a-zA-Z0-9]{36})",
            secret_type=SecretType.GITHUB_TOKEN,
            description="GitHub App Server Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(ghr_[a-zA-Z0-9]{36})",
            secret_type=SecretType.GITHUB_TOKEN,
            description="GitHub Refresh Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # GitLab
        # =====================================================================
        SecretPattern(
            pattern=r"(glpat-[a-zA-Z0-9_-]{20,})",
            secret_type=SecretType.GITLAB_TOKEN,
            description="GitLab Personal Access Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # Private Keys
        # =====================================================================
        SecretPattern(
            pattern=r"(-----BEGIN RSA PRIVATE KEY-----[\s\S]*?-----END RSA PRIVATE KEY-----)",
            secret_type=SecretType.PRIVATE_KEY_RSA,
            description="RSA Private Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(-----BEGIN EC PRIVATE KEY-----[\s\S]*?-----END EC PRIVATE KEY-----)",
            secret_type=SecretType.PRIVATE_KEY_ECDSA,
            description="ECDSA Private Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]*?-----END OPENSSH PRIVATE KEY-----)",
            secret_type=SecretType.PRIVATE_KEY_SSH,
            description="OpenSSH Private Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----)",
            secret_type=SecretType.PRIVATE_KEY_PGP,
            description="PGP Private Key Block",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(-----BEGIN PRIVATE KEY-----[\s\S]*?-----END PRIVATE KEY-----)",
            secret_type=SecretType.PRIVATE_KEY_RSA,
            description="Generic Private Key (PKCS#8)",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # GCP
        # =====================================================================
        SecretPattern(
            pattern=r"(AIza[0-9A-Za-z_-]{35})",
            secret_type=SecretType.GCP_API_KEY,
            description="GCP API Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r'"type"\s*:\s*"service_account"',
            secret_type=SecretType.GCP_SERVICE_ACCOUNT,
            description="GCP Service Account Key",
            confidence=Confidence.HIGH,
            capture_group=0,
        ),

        # =====================================================================
        # Azure
        # =====================================================================
        SecretPattern(
            pattern=r"(DefaultEndpointsProtocol=https?;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]+;)",
            secret_type=SecretType.AZURE_CONNECTION_STRING,
            description="Azure Storage Connection String",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # Database URLs
        # =====================================================================
        SecretPattern(
            pattern=r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:]+:[^@]+@[^\s]+)",
            secret_type=SecretType.DATABASE_URL,
            description="Database Connection URL",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(jdbc:[a-z]+://[^\s]+(?:password|pwd)=[^\s&]+)",
            secret_type=SecretType.JDBC_CONNECTION,
            description="JDBC Connection String with Password",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # JWT / Bearer
        # =====================================================================
        SecretPattern(
            pattern=r"(eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)",
            secret_type=SecretType.JWT_TOKEN,
            description="JWT Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # Slack
        # =====================================================================
        SecretPattern(
            pattern=r"(xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24})",
            secret_type=SecretType.SLACK_TOKEN,
            description="Slack Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # Stripe
        # =====================================================================
        SecretPattern(
            pattern=r"(sk_live_[a-zA-Z0-9]{24,})",
            secret_type=SecretType.STRIPE_KEY,
            description="Stripe Live Secret Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r"(rk_live_[a-zA-Z0-9]{24,})",
            secret_type=SecretType.STRIPE_KEY,
            description="Stripe Restricted Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # SendGrid
        # =====================================================================
        SecretPattern(
            pattern=r"(SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43})",
            secret_type=SecretType.SENDGRID_KEY,
            description="SendGrid API Key",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # NPM
        # =====================================================================
        SecretPattern(
            pattern=r"(npm_[a-zA-Z0-9]{36})",
            secret_type=SecretType.NPM_TOKEN,
            description="NPM Access Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # PyPI
        # =====================================================================
        SecretPattern(
            pattern=r"(pypi-[a-zA-Z0-9_-]{50,})",
            secret_type=SecretType.PYPI_TOKEN,
            description="PyPI API Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        ),

        # =====================================================================
        # Generic Secrets
        # =====================================================================
        SecretPattern(
            pattern=r'(?:password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']',
            secret_type=SecretType.GENERIC_PASSWORD,
            description="Generic Password Assignment",
            confidence=Confidence.MEDIUM,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r'(?:secret|secret_key|api_secret)\s*[=:]\s*["\']([^"\']{8,})["\']',
            secret_type=SecretType.GENERIC_SECRET,
            description="Generic Secret Assignment",
            confidence=Confidence.MEDIUM,
            capture_group=1,
        ),
        SecretPattern(
            pattern=r'(?:api_key|apikey|api-key)\s*[=:]\s*["\']([^"\']{16,})["\']',
            secret_type=SecretType.GENERIC_SECRET,
            description="Generic API Key Assignment",
            confidence=Confidence.MEDIUM,
            capture_group=1,
        ),
    ]

    def __init__(self):
        """Initialize the pattern library with default patterns."""
        self._patterns: list[SecretPattern] = []

        # Load default patterns
        for pattern in self.DEFAULT_PATTERNS:
            self._patterns.append(pattern)

    def add_pattern(self, pattern: SecretPattern) -> None:
        """Add a custom pattern to the library.

        Args:
            pattern: The pattern to add
        """
        self._patterns.append(pattern)

    def scan(self, text: str | None) -> list[SecretMatch]:
        """Scan text for secrets.

        Args:
            text: The text to scan

        Returns:
            List of detected secret matches
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return []

        matches: list[SecretMatch] = []
        seen_values: set[str] = set()

        for pattern in self._patterns:
            if pattern.compiled is None:
                continue

            for match in pattern.compiled.finditer(text):
                # Get the matched value
                if pattern.capture_group > 0 and match.lastindex and match.lastindex >= pattern.capture_group:
                    matched_value = match.group(pattern.capture_group)
                    start_pos = match.start(pattern.capture_group)
                    end_pos = match.end(pattern.capture_group)
                else:
                    matched_value = match.group(0)
                    start_pos = match.start()
                    end_pos = match.end()

                # Skip duplicates
                if matched_value in seen_values:
                    continue
                seen_values.add(matched_value)

                # Run custom validator if present
                if pattern.validator and not pattern.validator(matched_value):
                    continue

                # Adjust confidence for placeholders
                confidence = pattern.confidence
                if _is_placeholder(matched_value):
                    confidence = Confidence.LOW

                # Get position info
                line_number = _get_line_number(text, start_pos)
                column_start = _get_column_position(text, start_pos)
                column_end = column_start + len(matched_value)

                # Create match
                secret_match = SecretMatch(
                    secret_type=pattern.secret_type,
                    matched_value=matched_value,
                    redacted_value=_redact_value(matched_value),
                    confidence=confidence,
                    line_number=line_number,
                    column_start=column_start,
                    column_end=column_end,
                    pattern_description=pattern.description,
                )

                matches.append(secret_match)

        return matches

    def list_patterns(self) -> list[dict[str, Any]]:
        """List all registered patterns.

        Returns:
            List of pattern dictionaries
        """
        return [
            {
                "pattern": p.pattern,
                "secret_type": p.secret_type.value,
                "description": p.description,
                "confidence": p.confidence.name.lower(),
            }
            for p in self._patterns
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the pattern library.

        Returns:
            Dictionary with pattern statistics
        """
        by_type: dict[str, int] = {}
        for p in self._patterns:
            type_name = p.secret_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total_patterns": len(self._patterns),
            "patterns_by_type": by_type,
        }
