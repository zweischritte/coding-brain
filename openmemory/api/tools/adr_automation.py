"""ADR Automation Tool (FR-014).

This module provides the ADR (Architecture Decision Record) automation tool:
- ADRConfig: Configuration for ADR detection and generation
- ADRHeuristic: Base class for detection heuristics
- DetectionResult: Result of heuristic evaluation
- Individual heuristics: Dependency, API, Config, Schema, Security, Pattern, CrossCutting, Performance
- ADRHeuristicEngine: Combines multiple heuristics for detection
- ChangeAnalyzer: Parses git diffs to extract change information
- ADRTemplate: Template structure for ADR generation
- ADRContext: Context extracted from code changes for ADR
- ADRGenerator: Generates ADR content from context
- ADRAutomationTool: Main tool entry point

Heuristics:
- DependencyHeuristic: Detects significant new dependencies
- APIChangeHeuristic: Detects new/breaking API changes
- ConfigurationHeuristic: Detects feature flags, infrastructure changes
- SchemaHeuristic: Detects database schema changes
- SecurityHeuristic: Detects auth, encryption, permission changes
- PatternHeuristic: Detects architectural pattern introductions
- CrossCuttingHeuristic: Detects logging, monitoring, caching additions
- PerformanceHeuristic: Detects performance optimizations
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ADRAutomationError(Exception):
    """Base exception for ADR automation errors."""

    pass


class HeuristicError(ADRAutomationError):
    """Raised when heuristic evaluation fails."""

    pass


class TemplateError(ADRAutomationError):
    """Raised when template rendering fails."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ADRConfig:
    """Configuration for ADR automation tool.

    Args:
        min_confidence: Minimum confidence threshold for ADR creation (0.0-1.0)
        auto_link_to_code: Automatically link ADR to code changes
        template_version: ADR template version to use
        include_related_adrs: Include related existing ADRs in context
        max_related_adrs: Maximum number of related ADRs to include
        include_impact_analysis: Include impact analysis in ADR
    """

    min_confidence: float = 0.6
    auto_link_to_code: bool = True
    template_version: str = "1.0"
    include_related_adrs: bool = True
    max_related_adrs: int = 5
    include_impact_analysis: bool = True


# =============================================================================
# Detection Result
# =============================================================================


@dataclass
class DetectionResult:
    """Result of a heuristic evaluation.

    Args:
        detected: Whether the heuristic detected an ADR-worthy change
        confidence: Confidence score (0.0-1.0)
        reason: Human-readable explanation
    """

    detected: bool
    confidence: float
    reason: str


# =============================================================================
# Aggregate Result
# =============================================================================


@dataclass
class AggregateResult:
    """Aggregated result from multiple heuristics.

    Args:
        should_create_adr: Whether an ADR should be created
        confidence: Overall confidence score
        triggered_heuristics: List of heuristic names that triggered
        reasons: List of reasons from triggered heuristics
    """

    should_create_adr: bool
    confidence: float
    triggered_heuristics: list[str]
    reasons: list[str]


# =============================================================================
# Base Heuristic
# =============================================================================


class ADRHeuristic(ABC):
    """Base class for ADR detection heuristics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return heuristic name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return heuristic description."""
        pass

    @abstractmethod
    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate whether a change warrants an ADR.

        Args:
            change: Dictionary containing change information:
                - file_path: Path to changed file
                - change_type: Type of change (added/modified/deleted)
                - diff: Raw diff content
                - added_lines: List of added lines
                - removed_lines: List of removed lines

        Returns:
            DetectionResult with detection status, confidence, and reason
        """
        pass


# =============================================================================
# Dependency Heuristic
# =============================================================================


class DependencyHeuristic(ADRHeuristic):
    """Detects significant new dependency additions."""

    # Dependency file patterns
    DEPENDENCY_FILES = [
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.py",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "Pipfile",
        "Pipfile.lock",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
    ]

    # Patterns for significant dependencies (frameworks, databases, etc.)
    SIGNIFICANT_DEPS = [
        r"redis",
        r"celery",
        r"kafka",
        r"rabbitmq",
        r"elasticsearch",
        r"mongodb",
        r"postgresql",
        r"mysql",
        r"sqlite",
        r"fastapi",
        r"flask",
        r"django",
        r"express",
        r"react",
        r"vue",
        r"angular",
        r"tensorflow",
        r"pytorch",
        r"kubernetes",
        r"docker",
        r"aws-",
        r"azure-",
        r"google-cloud",
        r"graphql",
        r"grpc",
        r"protobuf",
    ]

    @property
    def name(self) -> str:
        return "dependency"

    @property
    def description(self) -> str:
        return "Detects significant new dependency additions"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate dependency changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])
        removed_lines = change.get("removed_lines", [])

        # Check if this is a dependency file
        if not any(file_path.endswith(dep_file) for dep_file in self.DEPENDENCY_FILES):
            return DetectionResult(
                detected=False, confidence=0.0, reason="Not a dependency file"
            )

        # Check for version bump only (same dependency, different version)
        if self._is_version_bump_only(added_lines, removed_lines):
            return DetectionResult(
                detected=False,
                confidence=0.2,
                reason="Version bump only, not a new dependency",
            )

        # Check for significant new dependencies
        significant_deps_found = []
        for line in added_lines:
            line_lower = line.lower()
            for pattern in self.SIGNIFICANT_DEPS:
                if re.search(pattern, line_lower):
                    significant_deps_found.append(pattern)

        if significant_deps_found:
            deps_str = ", ".join(set(significant_deps_found))
            return DetectionResult(
                detected=True,
                confidence=0.85,
                reason=f"Significant new dependencies added: {deps_str}",
            )

        # Any new dependency in dependency files
        if added_lines and not removed_lines:
            return DetectionResult(
                detected=True,
                confidence=0.7,
                reason="New dependency added to project",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No significant dependency changes"
        )

    def _is_version_bump_only(
        self, added_lines: list[str], removed_lines: list[str]
    ) -> bool:
        """Check if changes are just version bumps."""
        if not added_lines or not removed_lines:
            return False

        # Extract package names from lines
        added_packages = set()
        removed_packages = set()

        for line in added_lines:
            # Handle various formats: "package>=1.0", "package==1.0", "package": "^1.0"
            match = re.search(r'["\']?([a-zA-Z][\w-]*)', line)
            if match:
                added_packages.add(match.group(1).lower())

        for line in removed_lines:
            match = re.search(r'["\']?([a-zA-Z][\w-]*)', line)
            if match:
                removed_packages.add(match.group(1).lower())

        # If same packages in added and removed, it's a version bump
        return added_packages == removed_packages and len(added_packages) > 0


# =============================================================================
# API Change Heuristic
# =============================================================================


class APIChangeHeuristic(ADRHeuristic):
    """Detects API changes (REST, GraphQL, gRPC)."""

    # API-related file patterns
    API_FILE_PATTERNS = [
        r"routes?\.py$",
        r"api\.py$",
        r"endpoints?\.py$",
        r"views?\.py$",
        r"controllers?\.py$",
        r"schema\.graphql$",
        r"\.graphql$",
        r"\.proto$",
        r"openapi\.yaml$",
        r"swagger\.yaml$",
        r"api/.*\.py$",
        r"router/.*\.py$",
    ]

    # API decorator/annotation patterns
    API_PATTERNS = [
        r"@router\.(get|post|put|patch|delete|options|head)",
        r"@app\.(get|post|put|patch|delete|route)",
        r"@api_view",
        r"@require_http_methods",
        r"router\.(get|post|put|patch|delete)",
        r"type\s+Query\s*{",
        r"type\s+Mutation\s*{",
        r"service\s+\w+\s*{",
        r"rpc\s+\w+\s*\(",
        r"/api/v\d+/",
    ]

    @property
    def name(self) -> str:
        return "api_change"

    @property
    def description(self) -> str:
        return "Detects new or breaking API changes"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate API changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])
        removed_lines = change.get("removed_lines", [])
        diff = change.get("diff", "")

        # Check if this is an API-related file
        is_api_file = any(
            re.search(pattern, file_path) for pattern in self.API_FILE_PATTERNS
        )

        if not is_api_file:
            return DetectionResult(
                detected=False, confidence=0.0, reason="Not an API-related file"
            )

        # Check for API patterns in added lines
        api_patterns_found = []
        for line in added_lines:
            for pattern in self.API_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    api_patterns_found.append(pattern)

        # Check for breaking changes (removed API with added replacement)
        breaking_change = self._detect_breaking_change(added_lines, removed_lines)

        if breaking_change:
            return DetectionResult(
                detected=True,
                confidence=0.9,
                reason="Breaking API change detected (endpoint modified or removed)",
            )

        if api_patterns_found:
            return DetectionResult(
                detected=True,
                confidence=0.8,
                reason=f"New API endpoint or schema change detected",
            )

        # GraphQL/Proto files always significant
        if file_path.endswith((".graphql", ".proto")):
            return DetectionResult(
                detected=True,
                confidence=0.75,
                reason="GraphQL schema or protobuf definition changed",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No significant API changes"
        )

    def _detect_breaking_change(
        self, added_lines: list[str], removed_lines: list[str]
    ) -> bool:
        """Detect if changes represent a breaking API change."""
        if not removed_lines:
            return False

        # Look for URL path changes
        old_paths = set()
        new_paths = set()

        path_pattern = r"/api/[^\s\"\']+|@\w+\.(get|post|put|patch|delete)\([\"\'](/[^\s\"\']+)"

        for line in removed_lines:
            matches = re.findall(r"/api/[^\s\"\']+", line)
            old_paths.update(matches)

        for line in added_lines:
            matches = re.findall(r"/api/[^\s\"\']+", line)
            new_paths.update(matches)

        # Breaking if paths were removed or significantly changed
        if old_paths and old_paths != new_paths:
            return True

        # Check for type changes in parameters
        type_patterns = [
            r":\s*(int|str|UUID|bool|float)",
            r":\s*(string|number|boolean|ID)",
        ]

        old_types = []
        new_types = []

        for line in removed_lines:
            for pattern in type_patterns:
                matches = re.findall(pattern, line)
                old_types.extend(matches)

        for line in added_lines:
            for pattern in type_patterns:
                matches = re.findall(pattern, line)
                new_types.extend(matches)

        if old_types and new_types and old_types != new_types:
            return True

        return False


# =============================================================================
# Configuration Heuristic
# =============================================================================


class ConfigurationHeuristic(ADRHeuristic):
    """Detects significant configuration changes."""

    CONFIG_FILE_PATTERNS = [
        r"config/.*\.py$",
        r"settings\.py$",
        r"config\.py$",
        r"\.env",
        r"\.env\.example$",
        r"docker-compose\.ya?ml$",
        r"kubernetes/.*\.ya?ml$",
        r"k8s/.*\.ya?ml$",
        r"terraform/.*\.tf$",
        r"ansible/.*\.ya?ml$",
        r"infrastructure/.*",
    ]

    # Significant configuration patterns
    SIGNIFICANT_PATTERNS = [
        r"ENABLE_\w+",
        r"FEATURE_\w+",
        r"getenv\s*\(",
        r"os\.environ",
        r"image:\s*\w+",
        r"DATABASE_\w+",
        r"REDIS_\w+",
        r"CACHE_\w+",
        r"SECRET_\w+",
        r"API_\w+_URL",
        r"^\s*\w+:\s*$",  # YAML service definition
    ]

    # Minor tweaks that don't warrant ADR
    MINOR_PATTERNS = [
        r"LOG_LEVEL",
        r"DEBUG",
        r"VERBOSE",
        r"LOG_FORMAT",
    ]

    @property
    def name(self) -> str:
        return "configuration"

    @property
    def description(self) -> str:
        return "Detects significant configuration changes"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate configuration changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])
        removed_lines = change.get("removed_lines", [])

        # Check if config file
        is_config_file = any(
            re.search(pattern, file_path) for pattern in self.CONFIG_FILE_PATTERNS
        )

        if not is_config_file:
            return DetectionResult(
                detected=False, confidence=0.0, reason="Not a configuration file"
            )

        # Check for minor tweaks
        if self._is_minor_tweak(added_lines, removed_lines):
            return DetectionResult(
                detected=False,
                confidence=0.3,
                reason="Minor configuration tweak (logging, debug)",
            )

        # Check for significant patterns
        significant_found = []
        for line in added_lines:
            for pattern in self.SIGNIFICANT_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    significant_found.append(line.strip()[:50])
                    break

        if significant_found:
            return DetectionResult(
                detected=True,
                confidence=0.7,
                reason=f"Significant configuration change: {significant_found[0]}",
            )

        # Docker compose changes are always significant
        if "docker-compose" in file_path:
            return DetectionResult(
                detected=True,
                confidence=0.75,
                reason="Docker Compose infrastructure change",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No significant configuration changes"
        )

    def _is_minor_tweak(
        self, added_lines: list[str], removed_lines: list[str]
    ) -> bool:
        """Check if changes are minor tweaks."""
        all_lines = " ".join(added_lines + removed_lines)
        for pattern in self.MINOR_PATTERNS:
            if re.search(pattern, all_lines, re.IGNORECASE):
                # Check if any significant patterns exist
                for sig_pattern in self.SIGNIFICANT_PATTERNS:
                    if re.search(sig_pattern, all_lines, re.IGNORECASE):
                        return False
                return True
        return False


# =============================================================================
# Schema Heuristic
# =============================================================================


class SchemaHeuristic(ADRHeuristic):
    """Detects database schema changes."""

    SCHEMA_FILE_PATTERNS = [
        r"migrations?/.*\.py$",
        r"alembic/.*\.py$",
        r"models?\.py$",
        r"models?/.*\.py$",
        r"schema\.py$",
        r"entities?\.py$",
        r"migrations?/.*\.sql$",
        r"flyway/.*\.sql$",
    ]

    SCHEMA_PATTERNS = [
        r"create_table",
        r"drop_table",
        r"add_column",
        r"drop_column",
        r"alter_column",
        r"create_index",
        r"drop_index",
        r"__tablename__",
        r"class\s+\w+\s*\(.*Base",
        r"Column\s*\(",
        r"ForeignKey\s*\(",
        r"relationship\s*\(",
        r"CREATE TABLE",
        r"ALTER TABLE",
        r"DROP TABLE",
    ]

    @property
    def name(self) -> str:
        return "schema"

    @property
    def description(self) -> str:
        return "Detects database schema changes"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate schema changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])

        # Check if schema file
        is_schema_file = any(
            re.search(pattern, file_path) for pattern in self.SCHEMA_FILE_PATTERNS
        )

        if not is_schema_file:
            return DetectionResult(
                detected=False, confidence=0.0, reason="Not a schema-related file"
            )

        # Check for schema patterns
        schema_ops = []
        is_destructive = False

        for line in added_lines:
            for pattern in self.SCHEMA_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    schema_ops.append(pattern)
                    if "drop" in pattern.lower():
                        is_destructive = True
                    break

        if is_destructive:
            return DetectionResult(
                detected=True,
                confidence=0.9,
                reason="Destructive schema change detected (drop operation)",
            )

        if schema_ops:
            return DetectionResult(
                detected=True,
                confidence=0.8,
                reason=f"Database schema change: {schema_ops[0]}",
            )

        # Migration files are always significant
        if re.search(r"migrations?/", file_path):
            return DetectionResult(
                detected=True,
                confidence=0.75,
                reason="Database migration file detected",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No significant schema changes"
        )


# =============================================================================
# Security Heuristic
# =============================================================================


class SecurityHeuristic(ADRHeuristic):
    """Detects security-related changes."""

    SECURITY_FILE_PATTERNS = [
        r"auth/.*\.py$",
        r"authentication/.*\.py$",
        r"authorization/.*\.py$",
        r"security/.*\.py$",
        r"crypto/.*\.py$",
        r"encryption/.*\.py$",
        r"permissions?\.py$",
        r"rbac\.py$",
        r"oauth.*\.py$",
        r"jwt.*\.py$",
    ]

    SECURITY_PATTERNS = [
        r"class\s+.*Auth",
        r"class\s+.*Encrypt",
        r"class\s+.*Permission",
        r"class\s+.*OAuth",
        r"Fernet",
        r"bcrypt",
        r"hashlib",
        r"cryptography",
        r"authenticate",
        r"authorize",
        r"verify_token",
        r"create_token",
        r"check_permission",
        r"ADMIN_PERMISSIONS",
        r"ROLE_",
        r"jwt\.",
        r"OAuth2",
    ]

    @property
    def name(self) -> str:
        return "security"

    @property
    def description(self) -> str:
        return "Detects security-related changes"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate security changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])

        # Check if security file
        is_security_file = any(
            re.search(pattern, file_path) for pattern in self.SECURITY_FILE_PATTERNS
        )

        # Check for security patterns in content
        security_patterns_found = []
        for line in added_lines:
            for pattern in self.SECURITY_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    security_patterns_found.append(pattern)
                    break

        if security_patterns_found:
            return DetectionResult(
                detected=True,
                confidence=0.85,
                reason=f"Security-related change: {security_patterns_found[0]}",
            )

        if is_security_file and added_lines:
            return DetectionResult(
                detected=True,
                confidence=0.8,
                reason="Changes to security module",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No security-related changes"
        )


# =============================================================================
# Pattern Heuristic
# =============================================================================


class PatternHeuristic(ADRHeuristic):
    """Detects architectural pattern introductions."""

    PATTERN_INDICATORS = [
        (r"class\s+\w*Repository", "Repository pattern"),
        (r"class\s+\w*Factory", "Factory pattern"),
        (r"class\s+\w*Adapter", "Adapter pattern"),
        (r"class\s+\w*Strategy", "Strategy pattern"),
        (r"class\s+\w*Observer", "Observer pattern"),
        (r"class\s+\w*Singleton", "Singleton pattern"),
        (r"class\s+\w*Facade", "Facade pattern"),
        (r"class\s+\w*Decorator", "Decorator pattern"),
        (r"class\s+\w*Command", "Command pattern"),
        (r"class\s+\w*Handler", "Handler pattern"),
        (r"\(ABC\)", "Abstract base class"),
        (r"@abstractmethod", "Abstract method"),
        (r"layer", "Architectural layer"),
        (r"service\s+layer", "Service layer"),
        (r"adapter\s+layer", "Adapter layer"),
        (r"domain\s+layer", "Domain layer"),
    ]

    @property
    def name(self) -> str:
        return "pattern"

    @property
    def description(self) -> str:
        return "Detects architectural pattern introductions"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate pattern changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])
        diff = change.get("diff", "")

        all_content = "\n".join(added_lines) + diff

        patterns_found = []
        for pattern, name in self.PATTERN_INDICATORS:
            if re.search(pattern, all_content, re.IGNORECASE):
                patterns_found.append(name)

        if patterns_found:
            # Remove duplicates
            patterns_found = list(set(patterns_found))
            return DetectionResult(
                detected=True,
                confidence=0.75,
                reason=f"Architectural pattern detected: {', '.join(patterns_found)}",
            )

        # Check for new __init__.py with layer indicators
        if file_path.endswith("__init__.py") and any(
            layer in file_path for layer in ["adapters", "services", "repositories", "domain"]
        ):
            return DetectionResult(
                detected=True,
                confidence=0.7,
                reason="New architectural layer introduced",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No architectural pattern changes"
        )


# =============================================================================
# Cross-Cutting Heuristic
# =============================================================================


class CrossCuttingHeuristic(ADRHeuristic):
    """Detects cross-cutting concern changes."""

    CROSSCUTTING_PATTERNS = [
        (r"class\s+\w*Logger", "Logging infrastructure"),
        (r"class\s+\w*Metrics", "Metrics collection"),
        (r"class\s+\w*Cache", "Caching layer"),
        (r"opentelemetry", "OpenTelemetry observability"),
        (r"prometheus", "Prometheus metrics"),
        (r"grafana", "Grafana monitoring"),
        (r"structlog|StructuredLog", "Structured logging"),
        (r"create_counter|create_histogram", "Metrics instrumentation"),
        (r"redis\.Redis|RedisCache", "Redis caching"),
        (r"memcached|Memcache", "Memcached caching"),
        (r"Middleware", "Middleware addition"),
        (r"interceptor", "Interceptor pattern"),
    ]

    CROSSCUTTING_FILES = [
        r"logging/.*\.py$",
        r"observability/.*\.py$",
        r"metrics/.*\.py$",
        r"cache/.*\.py$",
        r"caching/.*\.py$",
        r"middleware/.*\.py$",
        r"telemetry/.*\.py$",
    ]

    @property
    def name(self) -> str:
        return "cross_cutting"

    @property
    def description(self) -> str:
        return "Detects cross-cutting concern changes"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate cross-cutting changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])
        diff = change.get("diff", "")

        all_content = "\n".join(added_lines) + diff

        # Check for cross-cutting patterns
        concerns_found = []
        for pattern, name in self.CROSSCUTTING_PATTERNS:
            if re.search(pattern, all_content, re.IGNORECASE):
                concerns_found.append(name)

        if concerns_found:
            concerns_found = list(set(concerns_found))
            return DetectionResult(
                detected=True,
                confidence=0.8,
                reason=f"Cross-cutting concern: {', '.join(concerns_found)}",
            )

        # Check file patterns
        for pattern in self.CROSSCUTTING_FILES:
            if re.search(pattern, file_path):
                return DetectionResult(
                    detected=True,
                    confidence=0.75,
                    reason="Cross-cutting infrastructure file",
                )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No cross-cutting concern changes"
        )


# =============================================================================
# Performance Heuristic
# =============================================================================


class PerformanceHeuristic(ADRHeuristic):
    """Detects performance optimization changes."""

    PERFORMANCE_PATTERNS = [
        (r"@lru_cache|@cache|@cached", "Caching decorator"),
        (r"asyncio\.gather", "Async parallelization"),
        (r"async\s+def", "Async conversion"),
        (r"create_index|add_index", "Database indexing"),
        (r"bulk_\w+|batch_", "Bulk operations"),
        (r"connection_pool|pool_size", "Connection pooling"),
        (r"prefetch|preload", "Prefetching"),
        (r"lazy_load|LazyLoad", "Lazy loading"),
        (r"ThreadPool|ProcessPool", "Thread/process pooling"),
        (r"concurrent\.futures", "Concurrent execution"),
        (r"optimize|optimization", "General optimization"),
    ]

    @property
    def name(self) -> str:
        return "performance"

    @property
    def description(self) -> str:
        return "Detects performance optimization changes"

    def evaluate(self, change: dict[str, Any]) -> DetectionResult:
        """Evaluate performance changes."""
        file_path = change.get("file_path", "")
        added_lines = change.get("added_lines", [])
        removed_lines = change.get("removed_lines", [])
        diff = change.get("diff", "")

        all_added = "\n".join(added_lines) + diff

        # Check for performance patterns
        optimizations_found = []
        for pattern, name in self.PERFORMANCE_PATTERNS:
            if re.search(pattern, all_added, re.IGNORECASE):
                optimizations_found.append(name)

        # Check for sync to async conversion
        if removed_lines and added_lines:
            removed_content = "\n".join(removed_lines)
            if re.search(r"def\s+\w+", removed_content) and re.search(
                r"async\s+def", all_added
            ):
                optimizations_found.append("Sync to async conversion")

        if optimizations_found:
            optimizations_found = list(set(optimizations_found))
            return DetectionResult(
                detected=True,
                confidence=0.75,
                reason=f"Performance optimization: {', '.join(optimizations_found)}",
            )

        return DetectionResult(
            detected=False, confidence=0.0, reason="No performance optimization changes"
        )


# =============================================================================
# Heuristic Engine
# =============================================================================


class ADRHeuristicEngine:
    """Engine that combines multiple heuristics for ADR detection."""

    def __init__(self, config: Optional[ADRConfig] = None):
        """Initialize with configuration and default heuristics.

        Args:
            config: Optional ADR configuration
        """
        self.config = config or ADRConfig()
        self.heuristics: list[ADRHeuristic] = []

        # Register default heuristics
        self._register_default_heuristics()

    def _register_default_heuristics(self) -> None:
        """Register all default heuristics."""
        self.heuristics = [
            DependencyHeuristic(),
            APIChangeHeuristic(),
            ConfigurationHeuristic(),
            SchemaHeuristic(),
            SecurityHeuristic(),
            PatternHeuristic(),
            CrossCuttingHeuristic(),
            PerformanceHeuristic(),
        ]

    def register(self, heuristic: ADRHeuristic) -> None:
        """Register a custom heuristic.

        Args:
            heuristic: Heuristic to register
        """
        self.heuristics.append(heuristic)

    def evaluate(self, change: dict[str, Any]) -> list[DetectionResult]:
        """Evaluate a change against all heuristics.

        Args:
            change: Change dictionary

        Returns:
            List of DetectionResults from all heuristics
        """
        results = []
        for heuristic in self.heuristics:
            try:
                result = heuristic.evaluate(change)
                results.append(result)
            except Exception as e:
                logger.warning(f"Heuristic {heuristic.name} failed: {e}")
                results.append(
                    DetectionResult(
                        detected=False,
                        confidence=0.0,
                        reason=f"Heuristic error: {e}",
                    )
                )
        return results

    def evaluate_aggregate(self, change: dict[str, Any]) -> AggregateResult:
        """Evaluate and aggregate results from all heuristics.

        Args:
            change: Change dictionary

        Returns:
            AggregateResult with combined detection
        """
        results = self.evaluate(change)

        triggered = []
        reasons = []
        max_confidence = 0.0

        for i, result in enumerate(results):
            if result.detected:
                heuristic_name = self.heuristics[i].name
                triggered.append(heuristic_name)
                reasons.append(result.reason)
                max_confidence = max(max_confidence, result.confidence)

        should_create = (
            len(triggered) > 0 and max_confidence >= self.config.min_confidence
        )

        return AggregateResult(
            should_create_adr=should_create,
            confidence=max_confidence,
            triggered_heuristics=triggered,
            reasons=reasons,
        )


# =============================================================================
# Change Analyzer
# =============================================================================


class ChangeAnalyzer:
    """Parses git diffs to extract change information."""

    def parse_diff(self, diff_text: str) -> list[dict[str, Any]]:
        """Parse a git diff into structured change information.

        Args:
            diff_text: Raw git diff output

        Returns:
            List of change dictionaries
        """
        changes = []
        current_file = None
        current_change = None

        lines = diff_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # New file diff header
            if line.startswith("diff --git"):
                if current_change:
                    changes.append(current_change)

                # Extract file path
                match = re.search(r"diff --git a/(.*) b/(.*)", line)
                if match:
                    current_file = match.group(2)
                    current_change = {
                        "file_path": current_file,
                        "change_type": "modified",
                        "diff": "",
                        "added_lines": [],
                        "removed_lines": [],
                        "line_start": 0,
                        "line_end": 0,
                    }

            # New file
            elif line.startswith("new file mode"):
                if current_change:
                    current_change["change_type"] = "added"

            # Deleted file
            elif line.startswith("deleted file mode"):
                if current_change:
                    current_change["change_type"] = "deleted"

            # Hunk header
            elif line.startswith("@@") and current_change:
                match = re.search(r"@@ -(\d+)", line)
                if match:
                    current_change["line_start"] = int(match.group(1))

                match = re.search(r"\+(\d+),?(\d+)?", line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    current_change["line_end"] = start + count

            # Added line
            elif line.startswith("+") and not line.startswith("+++") and current_change:
                current_change["added_lines"].append(line[1:])
                current_change["diff"] += line + "\n"

            # Removed line
            elif line.startswith("-") and not line.startswith("---") and current_change:
                current_change["removed_lines"].append(line[1:])
                current_change["diff"] += line + "\n"

            # Context line
            elif line.startswith(" ") and current_change:
                current_change["diff"] += line + "\n"

            i += 1

        if current_change:
            changes.append(current_change)

        return changes


# =============================================================================
# ADR Template
# =============================================================================


@dataclass
class ADRTemplate:
    """Template structure for ADR generation.

    Args:
        title: ADR title
        status: Current status (Proposed, Accepted, Deprecated, Superseded)
        context: Problem context and background
        decision: The decision that was made
        consequences: List of consequences (positive and negative)
        date: Creation date
        deciders: List of decision makers
        technical_story: Related technical story/ticket
    """

    title: str = ""
    status: str = "Proposed"
    context: str = ""
    decision: str = ""
    consequences: list[str] = field(default_factory=list)
    date: Optional[datetime] = None
    deciders: list[str] = field(default_factory=list)
    technical_story: str = ""

    def get_sections(self) -> dict[str, Any]:
        """Get all sections as a dictionary."""
        return {
            "title": self.title,
            "status": self.status,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "date": self.date,
            "deciders": self.deciders,
            "technical_story": self.technical_story,
        }

    def render(self) -> str:
        """Render ADR as Markdown.

        Returns:
            Markdown formatted ADR
        """
        lines = []

        # Title
        lines.append(f"# ADR: {self.title}")
        lines.append("")

        # Date
        date_str = (
            self.date.strftime("%Y-%m-%d") if self.date else datetime.now().strftime("%Y-%m-%d")
        )
        lines.append(f"**Date:** {date_str}")
        lines.append("")

        # Status
        lines.append("## Status")
        lines.append("")
        lines.append(self.status)
        lines.append("")

        # Context
        lines.append("## Context")
        lines.append("")
        lines.append(self.context if self.context else "_Context to be filled in._")
        lines.append("")

        # Decision
        lines.append("## Decision")
        lines.append("")
        lines.append(self.decision if self.decision else "_Decision to be filled in._")
        lines.append("")

        # Consequences
        lines.append("## Consequences")
        lines.append("")
        if self.consequences:
            for consequence in self.consequences:
                lines.append(f"- {consequence}")
        else:
            lines.append("_Consequences to be filled in._")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# ADR Context
# =============================================================================


@dataclass
class ADRContext:
    """Context extracted from code changes for ADR generation.

    Args:
        files_changed: List of changed file paths
        change_summary: Summary of changes
        triggered_heuristics: List of triggered heuristic names
        reasons: List of detection reasons
        related_symbols: Related code symbols
        related_adrs: Related existing ADRs
    """

    files_changed: list[str] = field(default_factory=list)
    change_summary: str = ""
    triggered_heuristics: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    related_symbols: list[dict[str, Any]] = field(default_factory=list)
    related_adrs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_changes(
        cls,
        changes: list[dict[str, Any]],
        graph_driver: Any = None,
        retriever: Any = None,
    ) -> "ADRContext":
        """Create context from a list of changes.

        Args:
            changes: List of change dictionaries
            graph_driver: Optional graph driver for symbol lookup
            retriever: Optional retriever for related ADRs

        Returns:
            ADRContext instance
        """
        files_changed = [c.get("file_path", "") for c in changes]

        # Build change summary
        summary_parts = []
        for change in changes:
            file_path = change.get("file_path", "")
            change_type = change.get("change_type", "modified")
            added_count = len(change.get("added_lines", []))
            removed_count = len(change.get("removed_lines", []))
            summary_parts.append(
                f"{file_path}: {change_type} (+{added_count}/-{removed_count})"
            )

        change_summary = "\n".join(summary_parts)

        # Get related symbols from graph
        related_symbols = []
        if graph_driver:
            try:
                for change in changes:
                    file_path = change.get("file_path", "")
                    # Query graph for symbols in this file
                    # This is a placeholder - actual implementation would query the graph
                    pass
            except Exception as e:
                logger.warning(f"Error getting related symbols: {e}")

        # Get related ADRs from retriever
        related_adrs = []
        if retriever:
            try:
                # Build query from file changes
                query_text = " ".join(files_changed)
                try:
                    from retrieval.trihybrid import TriHybridQuery
                except ImportError:
                    from openmemory.api.retrieval.trihybrid import TriHybridQuery

                query = TriHybridQuery(query_text=query_text, size=5)
                result = retriever.retrieve(query, index_name="adrs")
                for hit in result.hits:
                    related_adrs.append(
                        {
                            "id": hit.id,
                            "title": hit.source.get("title", ""),
                            "content": hit.source.get("content", ""),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error getting related ADRs: {e}")

        return cls(
            files_changed=files_changed,
            change_summary=change_summary,
            related_symbols=related_symbols,
            related_adrs=related_adrs,
        )


# =============================================================================
# ADR Generator
# =============================================================================


class ADRGenerator:
    """Generates ADR content from context."""

    def generate(self, context: ADRContext) -> ADRTemplate:
        """Generate an ADR template from context.

        Args:
            context: ADRContext with change information

        Returns:
            Populated ADRTemplate
        """
        # Generate title from heuristics and files
        title = self._generate_title(context)

        # Generate context section
        context_text = self._generate_context_section(context)

        # Generate decision section
        decision = self._generate_decision_section(context)

        # Generate consequences
        consequences = self._generate_consequences(context)

        return ADRTemplate(
            title=title,
            status="Proposed",
            context=context_text,
            decision=decision,
            consequences=consequences,
            date=datetime.now(),
        )

    def _generate_title(self, context: ADRContext) -> str:
        """Generate ADR title from context."""
        if context.triggered_heuristics:
            # Use first triggered heuristic as basis
            heuristic = context.triggered_heuristics[0]
            heuristic_titles = {
                "dependency": "Add New Dependency",
                "api_change": "API Change",
                "configuration": "Configuration Change",
                "schema": "Database Schema Change",
                "security": "Security Enhancement",
                "pattern": "Architectural Pattern",
                "cross_cutting": "Cross-Cutting Concern",
                "performance": "Performance Optimization",
            }
            base_title = heuristic_titles.get(heuristic, "Architecture Decision")

            # Try to extract specific detail from reasons
            if context.reasons:
                reason = context.reasons[0]
                # Extract key detail from reason
                if ":" in reason:
                    detail = reason.split(":")[-1].strip()[:50]
                    return f"{base_title}: {detail}"

            return base_title

        # Fallback to file-based title
        if context.files_changed:
            file_path = context.files_changed[0]
            return f"Changes to {file_path}"

        return "Architecture Decision"

    def _generate_context_section(self, context: ADRContext) -> str:
        """Generate context section from context."""
        parts = []

        parts.append("This decision addresses changes to the following files:")
        parts.append("")
        for file_path in context.files_changed[:5]:
            parts.append(f"- `{file_path}`")

        if len(context.files_changed) > 5:
            parts.append(f"- ... and {len(context.files_changed) - 5} more files")

        parts.append("")

        if context.reasons:
            parts.append("The changes triggered the following considerations:")
            parts.append("")
            for reason in context.reasons:
                parts.append(f"- {reason}")

        return "\n".join(parts)

    def _generate_decision_section(self, context: ADRContext) -> str:
        """Generate decision section from context."""
        if context.triggered_heuristics:
            heuristic = context.triggered_heuristics[0]
            decisions = {
                "dependency": "We will add a new dependency to the project to address specific requirements.",
                "api_change": "We will introduce/modify API endpoints to support new functionality.",
                "configuration": "We will introduce new configuration options to support operational requirements.",
                "schema": "We will modify the database schema to support new data requirements.",
                "security": "We will implement security enhancements to improve system protection.",
                "pattern": "We will introduce an architectural pattern to improve code organization.",
                "cross_cutting": "We will implement cross-cutting infrastructure for system-wide concerns.",
                "performance": "We will implement performance optimizations to improve system efficiency.",
            }
            return decisions.get(heuristic, "We will implement the proposed changes.")

        return "We will implement the proposed changes as described in the context."

    def _generate_consequences(self, context: ADRContext) -> list[str]:
        """Generate consequences from context."""
        consequences = []

        if context.triggered_heuristics:
            heuristic = context.triggered_heuristics[0]
            consequence_map = {
                "dependency": [
                    "New dependency adds functionality",
                    "Additional maintenance burden for dependency updates",
                    "Potential for security vulnerabilities in third-party code",
                ],
                "api_change": [
                    "New/modified API endpoints provide required functionality",
                    "May require client updates for breaking changes",
                    "API documentation needs updating",
                ],
                "configuration": [
                    "New configuration enables operational flexibility",
                    "Documentation and deployment procedures need updating",
                    "Additional environment variables to manage",
                ],
                "schema": [
                    "Database schema supports new data requirements",
                    "Migration needs testing in staging environment",
                    "Potential for data migration issues",
                ],
                "security": [
                    "Improved security posture",
                    "May impact performance due to additional checks",
                    "Security documentation needs updating",
                ],
                "pattern": [
                    "Improved code organization and maintainability",
                    "Team needs to understand and follow new patterns",
                    "May require refactoring of related code",
                ],
                "cross_cutting": [
                    "System-wide functionality implemented consistently",
                    "Additional infrastructure to maintain",
                    "Potential performance overhead",
                ],
                "performance": [
                    "Improved system performance",
                    "Added complexity from optimization",
                    "Needs performance testing to validate improvements",
                ],
            }
            consequences = consequence_map.get(heuristic, [])

        if not consequences:
            consequences = [
                "Changes implemented as proposed",
                "Documentation may need updating",
                "Testing required to validate changes",
            ]

        return consequences


# =============================================================================
# Analysis Result
# =============================================================================


@dataclass
class AnalysisResult:
    """Result of ADR analysis.

    Args:
        should_create_adr: Whether an ADR should be created
        confidence: Overall confidence score
        triggered_heuristics: List of triggered heuristic names
        reasons: List of detection reasons
        generated_adr: Generated ADR template (if should_create_adr is True)
        code_links: Links to code changes
        related_adrs: Related existing ADRs
        impact_analysis: Impact analysis result
    """

    should_create_adr: bool
    confidence: float
    triggered_heuristics: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    generated_adr: Optional[ADRTemplate] = None
    code_links: list[dict[str, Any]] = field(default_factory=list)
    related_adrs: list[dict[str, Any]] = field(default_factory=list)
    impact_analysis: Optional[dict[str, Any]] = None


# =============================================================================
# Main Tool
# =============================================================================


class ADRAutomationTool:
    """MCP tool for automated ADR detection and generation.

    This tool analyzes code changes and determines if an ADR should be created,
    then generates a draft ADR with appropriate content.
    """

    def __init__(
        self,
        graph_driver: Any,
        retriever: Any,
        config: Optional[ADRConfig] = None,
    ):
        """Initialize ADR automation tool.

        Args:
            graph_driver: Neo4j driver for CODE_* graph
            retriever: TriHybridRetriever for related ADRs
            config: Optional configuration
        """
        self.graph_driver = graph_driver
        self.retriever = retriever
        self.config = config or ADRConfig()

        self._engine = ADRHeuristicEngine(config=self.config)
        self._analyzer = ChangeAnalyzer()
        self._generator = ADRGenerator()

    def analyze(self, changes: list[dict[str, Any]]) -> AnalysisResult:
        """Analyze code changes for ADR worthiness.

        Args:
            changes: List of change dictionaries

        Returns:
            AnalysisResult with detection and generation results
        """
        # Evaluate each change against heuristics
        all_triggered = []
        all_reasons = []
        max_confidence = 0.0

        for change in changes:
            result = self._engine.evaluate_aggregate(change)
            if result.should_create_adr:
                all_triggered.extend(result.triggered_heuristics)
                all_reasons.extend(result.reasons)
                max_confidence = max(max_confidence, result.confidence)

        # Deduplicate
        all_triggered = list(set(all_triggered))
        all_reasons = list(set(all_reasons))

        should_create = (
            len(all_triggered) > 0 and max_confidence >= self.config.min_confidence
        )

        # Generate ADR if triggered
        generated_adr = None
        code_links = []
        related_adrs = []
        impact_analysis = None

        if should_create:
            # Build context
            context = ADRContext.from_changes(
                changes,
                graph_driver=self.graph_driver,
                retriever=self.retriever if self.config.include_related_adrs else None,
            )
            context.triggered_heuristics = all_triggered
            context.reasons = all_reasons

            # Generate ADR
            generated_adr = self._generator.generate(context)

            # Link to code
            if self.config.auto_link_to_code:
                code_links = [
                    {
                        "file_path": c.get("file_path"),
                        "change_type": c.get("change_type"),
                        "line_start": c.get("line_start"),
                        "line_end": c.get("line_end"),
                    }
                    for c in changes
                ]

            # Related ADRs
            if self.config.include_related_adrs:
                related_adrs = context.related_adrs

            # Impact analysis
            if self.config.include_impact_analysis:
                impact_analysis = self._perform_impact_analysis(changes)

        return AnalysisResult(
            should_create_adr=should_create,
            confidence=max_confidence,
            triggered_heuristics=all_triggered,
            reasons=all_reasons,
            generated_adr=generated_adr,
            code_links=code_links,
            related_adrs=related_adrs,
            impact_analysis=impact_analysis,
        )

    def _perform_impact_analysis(
        self, changes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform impact analysis for changes.

        Args:
            changes: List of change dictionaries

        Returns:
            Impact analysis result
        """
        try:
            from openmemory.api.tools.impact_analysis import ImpactAnalysisTool

            # Get symbols from changed files
            affected_files = [c.get("file_path") for c in changes]

            return {
                "affected_files": affected_files,
                "file_count": len(affected_files),
            }
        except Exception as e:
            logger.warning(f"Impact analysis failed: {e}")
            return {"error": str(e)}

    def get_mcp_schema(self) -> dict[str, Any]:
        """Get MCP tool schema.

        Returns:
            MCP schema dictionary
        """
        return {
            "name": "suggest_adr",
            "description": "Analyzes code changes and suggests if an ADR should be created",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "changes": {
                        "type": "array",
                        "description": "List of code changes to analyze",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to changed file",
                                },
                                "change_type": {
                                    "type": "string",
                                    "enum": ["added", "modified", "deleted"],
                                },
                                "diff": {
                                    "type": "string",
                                    "description": "Raw diff content",
                                },
                                "added_lines": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "removed_lines": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["file_path"],
                        },
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Minimum confidence threshold (0.0-1.0)",
                        "default": 0.6,
                    },
                },
                "required": ["changes"],
            },
        }

    def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute tool via MCP interface.

        Args:
            input_data: MCP input data

        Returns:
            Result dictionary
        """
        changes = input_data.get("changes", [])
        min_confidence = input_data.get("min_confidence", self.config.min_confidence)

        # Temporarily update config
        original_confidence = self.config.min_confidence
        self.config.min_confidence = min_confidence
        self._engine.config.min_confidence = min_confidence

        try:
            result = self.analyze(changes)

            output = {
                "should_create_adr": result.should_create_adr,
                "confidence": result.confidence,
                "triggered_heuristics": result.triggered_heuristics,
                "reasons": result.reasons,
            }

            if result.generated_adr:
                output["generated_adr"] = {
                    "title": result.generated_adr.title,
                    "status": result.generated_adr.status,
                    "context": result.generated_adr.context,
                    "decision": result.generated_adr.decision,
                    "consequences": result.generated_adr.consequences,
                    "markdown": result.generated_adr.render(),
                }

            if result.code_links:
                output["code_links"] = result.code_links

            if result.related_adrs:
                output["related_adrs"] = result.related_adrs

            if result.impact_analysis:
                output["impact_analysis"] = result.impact_analysis

            return output

        finally:
            # Restore original config
            self.config.min_confidence = original_confidence
            self._engine.config.min_confidence = original_confidence


# =============================================================================
# Factory Function
# =============================================================================


def create_adr_automation_tool(
    graph_driver: Any,
    retriever: Any,
    config: Optional[ADRConfig] = None,
) -> ADRAutomationTool:
    """Create an ADR automation tool.

    Args:
        graph_driver: Neo4j driver for CODE_* graph
        retriever: TriHybridRetriever for related ADRs
        config: Optional configuration

    Returns:
        Configured ADRAutomationTool
    """
    return ADRAutomationTool(
        graph_driver=graph_driver,
        retriever=retriever,
        config=config,
    )
