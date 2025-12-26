"""Tests for ADR Automation Tool (FR-014).

This module tests the ADR (Architecture Decision Record) automation tool with TDD approach:
- ADRConfig: Configuration for ADR detection and generation
- ADRHeuristic: Individual heuristic for detecting ADR-worthy changes
- ADRHeuristicEngine: Combines multiple heuristics for detection
- ChangeAnalyzer: Analyzes code changes for architectural significance
- ADRTemplate: Template structure for ADR generation
- ADRContext: Context extracted from code changes for ADR
- ADRGenerator: Generates ADR content from context
- ADRAutomationTool: Main tool entry point

Heuristics covered:
- New dependency added (significant library additions)
- API changes (breaking or new endpoints)
- Configuration changes (environment, feature flags)
- Data model changes (schema migrations, new entities)
- Security-related changes (auth, encryption)
- Architecture pattern changes (design patterns, layers)
- Cross-cutting concern changes (logging, monitoring)
- Performance optimization changes
"""

from __future__ import annotations

import time
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_dependency_change() -> dict[str, Any]:
    """Return a sample change adding a new dependency."""
    return {
        "file_path": "pyproject.toml",
        "change_type": "modified",
        "diff": """
+[project.dependencies]
+redis = ">=4.0.0"
+celery = ">=5.0.0"
""",
        "added_lines": ['+redis = ">=4.0.0"', '+celery = ">=5.0.0"'],
        "removed_lines": [],
    }


@pytest.fixture
def sample_api_change() -> dict[str, Any]:
    """Return a sample change adding a new API endpoint."""
    return {
        "file_path": "src/api/routes.py",
        "change_type": "modified",
        "diff": """
+@router.post("/api/v2/users")
+async def create_user_v2(request: CreateUserRequest):
+    '''New user creation endpoint with enhanced validation.'''
+    pass
""",
        "added_lines": [
            '@router.post("/api/v2/users")',
            "async def create_user_v2(request: CreateUserRequest):",
        ],
        "removed_lines": [],
    }


@pytest.fixture
def sample_breaking_api_change() -> dict[str, Any]:
    """Return a sample breaking API change."""
    return {
        "file_path": "src/api/routes.py",
        "change_type": "modified",
        "diff": """
-@router.get("/api/users/{user_id}")
-async def get_user(user_id: int):
+@router.get("/api/v2/users/{user_id}")
+async def get_user(user_id: UUID):
""",
        "added_lines": [
            '@router.get("/api/v2/users/{user_id}")',
            "async def get_user(user_id: UUID):",
        ],
        "removed_lines": [
            '@router.get("/api/users/{user_id}")',
            "async def get_user(user_id: int):",
        ],
    }


@pytest.fixture
def sample_config_change() -> dict[str, Any]:
    """Return a sample configuration change."""
    return {
        "file_path": "config/settings.py",
        "change_type": "modified",
        "diff": """
+ENABLE_NEW_AUTH_FLOW = os.getenv("ENABLE_NEW_AUTH_FLOW", "false") == "true"
+MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
""",
        "added_lines": [
            'ENABLE_NEW_AUTH_FLOW = os.getenv("ENABLE_NEW_AUTH_FLOW", "false") == "true"',
            'MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))',
        ],
        "removed_lines": [],
    }


@pytest.fixture
def sample_schema_change() -> dict[str, Any]:
    """Return a sample database schema change."""
    return {
        "file_path": "migrations/001_add_user_preferences.py",
        "change_type": "added",
        "diff": """
+class Migration:
+    def upgrade(self):
+        op.create_table(
+            'user_preferences',
+            sa.Column('id', sa.UUID(), primary_key=True),
+            sa.Column('user_id', sa.UUID(), nullable=False),
+            sa.Column('theme', sa.String(50)),
+            sa.Column('notifications_enabled', sa.Boolean()),
+        )
""",
        "added_lines": ["op.create_table(", "'user_preferences',"],
        "removed_lines": [],
    }


@pytest.fixture
def sample_security_change() -> dict[str, Any]:
    """Return a sample security-related change."""
    return {
        "file_path": "src/auth/encryption.py",
        "change_type": "added",
        "diff": """
+class EncryptionService:
+    def __init__(self, key_provider: KeyProvider):
+        self._cipher = Fernet(key_provider.get_key())
+
+    def encrypt(self, data: bytes) -> bytes:
+        return self._cipher.encrypt(data)
""",
        "added_lines": [
            "class EncryptionService:",
            "self._cipher = Fernet(key_provider.get_key())",
        ],
        "removed_lines": [],
    }


@pytest.fixture
def sample_pattern_change() -> dict[str, Any]:
    """Return a sample architectural pattern change."""
    return {
        "file_path": "src/services/base.py",
        "change_type": "added",
        "diff": """
+class Repository(ABC):
+    '''Base repository pattern for data access.'''
+
+    @abstractmethod
+    def get(self, id: UUID) -> Optional[Entity]:
+        pass
+
+    @abstractmethod
+    def save(self, entity: Entity) -> None:
+        pass
""",
        "added_lines": [
            "class Repository(ABC):",
            "Base repository pattern for data access",
        ],
        "removed_lines": [],
    }


@pytest.fixture
def sample_trivial_change() -> dict[str, Any]:
    """Return a trivial change that should not trigger ADR."""
    return {
        "file_path": "src/utils/helpers.py",
        "change_type": "modified",
        "diff": """
-    return value.strip()
+    return value.strip().lower()
""",
        "added_lines": ["    return value.strip().lower()"],
        "removed_lines": ["    return value.strip()"],
    }


@pytest.fixture
def mock_graph_driver():
    """Create a mock Neo4j graph driver."""
    driver = MagicMock()
    driver.get_node.return_value = None
    driver.get_outgoing_edges.return_value = []
    return driver


@pytest.fixture
def mock_retriever():
    """Create a mock tri-hybrid retriever."""
    retriever = MagicMock()
    mock_result = MagicMock()
    mock_result.hits = []
    mock_result.total = 0
    retriever.retrieve.return_value = mock_result
    return retriever


# =============================================================================
# ADRConfig Tests
# =============================================================================


class TestADRConfig:
    """Tests for ADRConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from openmemory.api.tools.adr_automation import ADRConfig

        config = ADRConfig()

        assert config.min_confidence == 0.6
        assert config.auto_link_to_code is True
        assert config.template_version == "1.0"
        assert config.include_related_adrs is True
        assert config.max_related_adrs == 5
        assert config.include_impact_analysis is True

    def test_custom_values(self):
        """Test custom configuration values."""
        from openmemory.api.tools.adr_automation import ADRConfig

        config = ADRConfig(
            min_confidence=0.8,
            auto_link_to_code=False,
            template_version="2.0",
            include_related_adrs=False,
            max_related_adrs=10,
            include_impact_analysis=False,
        )

        assert config.min_confidence == 0.8
        assert config.auto_link_to_code is False
        assert config.template_version == "2.0"
        assert config.include_related_adrs is False
        assert config.max_related_adrs == 10
        assert config.include_impact_analysis is False

    def test_min_confidence_range(self):
        """Test min_confidence should be between 0 and 1."""
        from openmemory.api.tools.adr_automation import ADRConfig

        config = ADRConfig(min_confidence=0.0)
        assert config.min_confidence == 0.0

        config = ADRConfig(min_confidence=1.0)
        assert config.min_confidence == 1.0


# =============================================================================
# ADRHeuristic Tests
# =============================================================================


class TestADRHeuristic:
    """Tests for ADRHeuristic base class."""

    def test_heuristic_has_name_and_description(self):
        """Test heuristic has name and description."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        heuristic = DependencyHeuristic()

        assert heuristic.name is not None
        assert len(heuristic.name) > 0
        assert heuristic.description is not None
        assert len(heuristic.description) > 0

    def test_heuristic_returns_detection_result(self, sample_dependency_change):
        """Test heuristic returns a DetectionResult."""
        from openmemory.api.tools.adr_automation import (
            DependencyHeuristic,
            DetectionResult,
        )

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(sample_dependency_change)

        assert isinstance(result, DetectionResult)
        assert hasattr(result, "detected")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reason")

    def test_heuristic_confidence_in_range(self, sample_dependency_change):
        """Test heuristic confidence is between 0 and 1."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(sample_dependency_change)

        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Dependency Heuristic Tests
# =============================================================================


class TestDependencyHeuristic:
    """Tests for dependency change detection."""

    def test_detects_new_dependency_in_pyproject(self, sample_dependency_change):
        """Test detection of new dependency in pyproject.toml."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(sample_dependency_change)

        assert result.detected is True
        assert result.confidence >= 0.7
        assert "dependency" in result.reason.lower() or "redis" in result.reason.lower()

    def test_detects_new_dependency_in_requirements(self):
        """Test detection of new dependency in requirements.txt."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        change = {
            "file_path": "requirements.txt",
            "change_type": "modified",
            "diff": "+elasticsearch>=8.0.0",
            "added_lines": ["+elasticsearch>=8.0.0"],
            "removed_lines": [],
        }

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True
        assert result.confidence >= 0.7

    def test_detects_dependency_in_package_json(self):
        """Test detection of new dependency in package.json."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        change = {
            "file_path": "package.json",
            "change_type": "modified",
            "diff": '+"react-query": "^4.0.0"',
            "added_lines": ['"react-query": "^4.0.0"'],
            "removed_lines": [],
        }

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_ignores_version_bump_only(self):
        """Test ignores simple version bump (not significant)."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        change = {
            "file_path": "pyproject.toml",
            "change_type": "modified",
            "diff": """
-requests = ">=2.28.0"
+requests = ">=2.29.0"
""",
            "added_lines": ['requests = ">=2.29.0"'],
            "removed_lines": ['requests = ">=2.28.0"'],
        }

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(change)

        # Version bumps should have lower confidence or not be detected
        assert result.detected is False or result.confidence < 0.5

    def test_ignores_non_dependency_files(self, sample_trivial_change):
        """Test ignores non-dependency files."""
        from openmemory.api.tools.adr_automation import DependencyHeuristic

        heuristic = DependencyHeuristic()
        result = heuristic.evaluate(sample_trivial_change)

        assert result.detected is False


# =============================================================================
# API Change Heuristic Tests
# =============================================================================


class TestAPIChangeHeuristic:
    """Tests for API change detection."""

    def test_detects_new_api_endpoint(self, sample_api_change):
        """Test detection of new API endpoint."""
        from openmemory.api.tools.adr_automation import APIChangeHeuristic

        heuristic = APIChangeHeuristic()
        result = heuristic.evaluate(sample_api_change)

        assert result.detected is True
        assert result.confidence >= 0.7
        assert "api" in result.reason.lower() or "endpoint" in result.reason.lower()

    def test_detects_breaking_api_change(self, sample_breaking_api_change):
        """Test detection of breaking API change."""
        from openmemory.api.tools.adr_automation import APIChangeHeuristic

        heuristic = APIChangeHeuristic()
        result = heuristic.evaluate(sample_breaking_api_change)

        assert result.detected is True
        assert result.confidence >= 0.8  # Breaking changes should have high confidence

    def test_detects_graphql_schema_change(self):
        """Test detection of GraphQL schema change."""
        from openmemory.api.tools.adr_automation import APIChangeHeuristic

        change = {
            "file_path": "schema.graphql",
            "change_type": "modified",
            "diff": """
+type Query {
+    userById(id: ID!): User
+}
""",
            "added_lines": ["type Query {", "userById(id: ID!): User"],
            "removed_lines": [],
        }

        heuristic = APIChangeHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_protobuf_change(self):
        """Test detection of protobuf API change."""
        from openmemory.api.tools.adr_automation import APIChangeHeuristic

        change = {
            "file_path": "proto/user.proto",
            "change_type": "added",
            "diff": """
+service UserService {
+    rpc GetUser(GetUserRequest) returns (User);
+}
""",
            "added_lines": ["service UserService {", "rpc GetUser(GetUserRequest)"],
            "removed_lines": [],
        }

        heuristic = APIChangeHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_ignores_internal_function(self, sample_trivial_change):
        """Test ignores internal function changes."""
        from openmemory.api.tools.adr_automation import APIChangeHeuristic

        heuristic = APIChangeHeuristic()
        result = heuristic.evaluate(sample_trivial_change)

        assert result.detected is False


# =============================================================================
# Configuration Change Heuristic Tests
# =============================================================================


class TestConfigurationHeuristic:
    """Tests for configuration change detection."""

    def test_detects_new_feature_flag(self, sample_config_change):
        """Test detection of new feature flag."""
        from openmemory.api.tools.adr_automation import ConfigurationHeuristic

        heuristic = ConfigurationHeuristic()
        result = heuristic.evaluate(sample_config_change)

        assert result.detected is True
        assert result.confidence >= 0.6

    def test_detects_env_file_change(self):
        """Test detection of environment file change."""
        from openmemory.api.tools.adr_automation import ConfigurationHeuristic

        change = {
            "file_path": ".env.example",
            "change_type": "modified",
            "diff": """
+DATABASE_REPLICA_URL=
+CACHE_TTL_SECONDS=300
""",
            "added_lines": ["DATABASE_REPLICA_URL=", "CACHE_TTL_SECONDS=300"],
            "removed_lines": [],
        }

        heuristic = ConfigurationHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_docker_compose_change(self):
        """Test detection of docker-compose infrastructure change."""
        from openmemory.api.tools.adr_automation import ConfigurationHeuristic

        change = {
            "file_path": "docker-compose.yml",
            "change_type": "modified",
            "diff": """
+  redis:
+    image: redis:7
+    ports:
+      - "6379:6379"
""",
            "added_lines": ["redis:", "image: redis:7"],
            "removed_lines": [],
        }

        heuristic = ConfigurationHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_ignores_minor_config_tweaks(self):
        """Test ignores minor configuration tweaks."""
        from openmemory.api.tools.adr_automation import ConfigurationHeuristic

        change = {
            "file_path": "config/logging.py",
            "change_type": "modified",
            "diff": """
-LOG_LEVEL = "INFO"
+LOG_LEVEL = "DEBUG"
""",
            "added_lines": ['LOG_LEVEL = "DEBUG"'],
            "removed_lines": ['LOG_LEVEL = "INFO"'],
        }

        heuristic = ConfigurationHeuristic()
        result = heuristic.evaluate(change)

        # Minor tweaks should not be detected or have low confidence
        assert result.detected is False or result.confidence < 0.5


# =============================================================================
# Schema Change Heuristic Tests
# =============================================================================


class TestSchemaHeuristic:
    """Tests for database schema change detection."""

    def test_detects_migration_file(self, sample_schema_change):
        """Test detection of database migration."""
        from openmemory.api.tools.adr_automation import SchemaHeuristic

        heuristic = SchemaHeuristic()
        result = heuristic.evaluate(sample_schema_change)

        assert result.detected is True
        assert result.confidence >= 0.7
        assert "schema" in result.reason.lower() or "table" in result.reason.lower()

    def test_detects_model_change(self):
        """Test detection of ORM model change."""
        from openmemory.api.tools.adr_automation import SchemaHeuristic

        change = {
            "file_path": "src/models/user.py",
            "change_type": "modified",
            "diff": """
+class UserPreferences(Base):
+    __tablename__ = 'user_preferences'
+    id = Column(UUID, primary_key=True)
+    user_id = Column(UUID, ForeignKey('users.id'))
""",
            "added_lines": ["class UserPreferences(Base):", "__tablename__"],
            "removed_lines": [],
        }

        heuristic = SchemaHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_column_removal(self):
        """Test detection of column removal (breaking change)."""
        from openmemory.api.tools.adr_automation import SchemaHeuristic

        change = {
            "file_path": "migrations/002_remove_legacy_column.py",
            "change_type": "added",
            "diff": """
+def upgrade():
+    op.drop_column('users', 'legacy_field')
""",
            "added_lines": ["op.drop_column('users', 'legacy_field')"],
            "removed_lines": [],
        }

        heuristic = SchemaHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True
        assert result.confidence >= 0.8  # Destructive changes should have high confidence


# =============================================================================
# Security Change Heuristic Tests
# =============================================================================


class TestSecurityHeuristic:
    """Tests for security-related change detection."""

    def test_detects_encryption_addition(self, sample_security_change):
        """Test detection of encryption addition."""
        from openmemory.api.tools.adr_automation import SecurityHeuristic

        heuristic = SecurityHeuristic()
        result = heuristic.evaluate(sample_security_change)

        assert result.detected is True
        assert result.confidence >= 0.8

    def test_detects_auth_change(self):
        """Test detection of authentication change."""
        from openmemory.api.tools.adr_automation import SecurityHeuristic

        change = {
            "file_path": "src/auth/oauth.py",
            "change_type": "added",
            "diff": """
+class OAuth2Provider:
+    def authenticate(self, token: str) -> Optional[User]:
+        '''Authenticate using OAuth 2.0 bearer token.'''
+        pass
""",
            "added_lines": ["class OAuth2Provider:", "authenticate"],
            "removed_lines": [],
        }

        heuristic = SecurityHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_permission_change(self):
        """Test detection of permission/authorization change."""
        from openmemory.api.tools.adr_automation import SecurityHeuristic

        change = {
            "file_path": "src/auth/permissions.py",
            "change_type": "modified",
            "diff": """
+ADMIN_PERMISSIONS = ["read", "write", "delete", "admin"]
+
+def check_permission(user: User, permission: str) -> bool:
+    return permission in user.permissions
""",
            "added_lines": ["ADMIN_PERMISSIONS", "check_permission"],
            "removed_lines": [],
        }

        heuristic = SecurityHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True


# =============================================================================
# Architecture Pattern Heuristic Tests
# =============================================================================


class TestPatternHeuristic:
    """Tests for architectural pattern change detection."""

    def test_detects_repository_pattern(self, sample_pattern_change):
        """Test detection of repository pattern introduction."""
        from openmemory.api.tools.adr_automation import PatternHeuristic

        heuristic = PatternHeuristic()
        result = heuristic.evaluate(sample_pattern_change)

        assert result.detected is True
        assert result.confidence >= 0.7

    def test_detects_factory_pattern(self):
        """Test detection of factory pattern introduction."""
        from openmemory.api.tools.adr_automation import PatternHeuristic

        change = {
            "file_path": "src/factories/service_factory.py",
            "change_type": "added",
            "diff": """
+class ServiceFactory:
+    '''Factory for creating service instances.'''
+
+    @classmethod
+    def create(cls, service_type: str) -> Service:
+        if service_type == "email":
+            return EmailService()
+        elif service_type == "sms":
+            return SMSService()
""",
            "added_lines": ["class ServiceFactory:", "Factory for creating"],
            "removed_lines": [],
        }

        heuristic = PatternHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_layer_introduction(self):
        """Test detection of new architectural layer."""
        from openmemory.api.tools.adr_automation import PatternHeuristic

        change = {
            "file_path": "src/adapters/__init__.py",
            "change_type": "added",
            "diff": """
+'''Adapter layer for external service integrations.'''
+
+from .email_adapter import EmailAdapter
+from .payment_adapter import PaymentAdapter
""",
            "added_lines": ["Adapter layer", "EmailAdapter", "PaymentAdapter"],
            "removed_lines": [],
        }

        heuristic = PatternHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True


# =============================================================================
# Cross-Cutting Concern Heuristic Tests
# =============================================================================


class TestCrossCuttingHeuristic:
    """Tests for cross-cutting concern change detection."""

    def test_detects_logging_addition(self):
        """Test detection of logging infrastructure addition."""
        from openmemory.api.tools.adr_automation import CrossCuttingHeuristic

        change = {
            "file_path": "src/logging/structured_logger.py",
            "change_type": "added",
            "diff": """
+class StructuredLogger:
+    '''Structured JSON logging for observability.'''
+
+    def __init__(self, context: Dict[str, Any]):
+        self._context = context
+        self._handler = logging.StreamHandler()
""",
            "added_lines": ["class StructuredLogger:", "Structured JSON logging"],
            "removed_lines": [],
        }

        heuristic = CrossCuttingHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_monitoring_addition(self):
        """Test detection of monitoring/observability addition."""
        from openmemory.api.tools.adr_automation import CrossCuttingHeuristic

        change = {
            "file_path": "src/observability/metrics.py",
            "change_type": "added",
            "diff": """
+from opentelemetry import metrics
+
+class MetricsCollector:
+    def __init__(self):
+        self.meter = metrics.get_meter("app_metrics")
+        self.request_counter = self.meter.create_counter("http_requests")
""",
            "added_lines": ["opentelemetry", "MetricsCollector", "create_counter"],
            "removed_lines": [],
        }

        heuristic = CrossCuttingHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_caching_addition(self):
        """Test detection of caching layer addition."""
        from openmemory.api.tools.adr_automation import CrossCuttingHeuristic

        change = {
            "file_path": "src/cache/redis_cache.py",
            "change_type": "added",
            "diff": """
+class RedisCache:
+    '''Redis-based caching layer for frequently accessed data.'''
+
+    def __init__(self, redis_url: str):
+        self._client = redis.Redis.from_url(redis_url)
""",
            "added_lines": ["class RedisCache:", "caching layer"],
            "removed_lines": [],
        }

        heuristic = CrossCuttingHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True


# =============================================================================
# Performance Change Heuristic Tests
# =============================================================================


class TestPerformanceHeuristic:
    """Tests for performance optimization change detection."""

    def test_detects_caching_optimization(self):
        """Test detection of caching optimization."""
        from openmemory.api.tools.adr_automation import PerformanceHeuristic

        change = {
            "file_path": "src/services/user_service.py",
            "change_type": "modified",
            "diff": """
+from functools import lru_cache
+
+@lru_cache(maxsize=1000)
+def get_user_permissions(user_id: UUID) -> List[str]:
+    '''Cached user permissions lookup.'''
""",
            "added_lines": ["lru_cache", "@lru_cache(maxsize=1000)"],
            "removed_lines": [],
        }

        heuristic = PerformanceHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_async_optimization(self):
        """Test detection of async/await optimization."""
        from openmemory.api.tools.adr_automation import PerformanceHeuristic

        change = {
            "file_path": "src/services/data_service.py",
            "change_type": "modified",
            "diff": """
-def fetch_all_data():
-    result1 = fetch_data_1()
-    result2 = fetch_data_2()
-    return result1, result2
+async def fetch_all_data():
+    result1, result2 = await asyncio.gather(
+        fetch_data_1(),
+        fetch_data_2()
+    )
+    return result1, result2
""",
            "added_lines": ["async def fetch_all_data", "asyncio.gather"],
            "removed_lines": ["def fetch_all_data"],
        }

        heuristic = PerformanceHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True

    def test_detects_indexing_optimization(self):
        """Test detection of database indexing optimization."""
        from openmemory.api.tools.adr_automation import PerformanceHeuristic

        change = {
            "file_path": "migrations/003_add_indexes.py",
            "change_type": "added",
            "diff": """
+def upgrade():
+    op.create_index('idx_users_email', 'users', ['email'])
+    op.create_index('idx_orders_user_date', 'orders', ['user_id', 'created_at'])
""",
            "added_lines": ["create_index", "idx_users_email", "idx_orders_user_date"],
            "removed_lines": [],
        }

        heuristic = PerformanceHeuristic()
        result = heuristic.evaluate(change)

        assert result.detected is True


# =============================================================================
# ADRHeuristicEngine Tests
# =============================================================================


class TestADRHeuristicEngine:
    """Tests for the heuristic engine that combines multiple heuristics."""

    def test_engine_registers_all_heuristics(self):
        """Test engine registers all built-in heuristics."""
        from openmemory.api.tools.adr_automation import ADRHeuristicEngine

        engine = ADRHeuristicEngine()

        assert len(engine.heuristics) >= 8  # At least 8 built-in heuristics

    def test_engine_evaluates_all_heuristics(self, sample_dependency_change):
        """Test engine evaluates all heuristics."""
        from openmemory.api.tools.adr_automation import ADRHeuristicEngine

        engine = ADRHeuristicEngine()
        results = engine.evaluate(sample_dependency_change)

        # Should return results for all heuristics
        assert len(results) == len(engine.heuristics)

    def test_engine_aggregates_results(self, sample_dependency_change):
        """Test engine aggregates detection results."""
        from openmemory.api.tools.adr_automation import ADRHeuristicEngine

        engine = ADRHeuristicEngine()
        result = engine.evaluate_aggregate(sample_dependency_change)

        assert hasattr(result, "should_create_adr")
        assert hasattr(result, "confidence")
        assert hasattr(result, "triggered_heuristics")
        assert hasattr(result, "reasons")

    def test_engine_respects_min_confidence(self, sample_trivial_change):
        """Test engine respects minimum confidence threshold."""
        from openmemory.api.tools.adr_automation import ADRConfig, ADRHeuristicEngine

        config = ADRConfig(min_confidence=0.9)
        engine = ADRHeuristicEngine(config=config)
        result = engine.evaluate_aggregate(sample_trivial_change)

        # Trivial changes should not meet high confidence threshold
        assert result.should_create_adr is False

    def test_engine_custom_heuristic_registration(self):
        """Test engine allows custom heuristic registration."""
        from openmemory.api.tools.adr_automation import (
            ADRHeuristic,
            ADRHeuristicEngine,
            DetectionResult,
        )

        class CustomHeuristic(ADRHeuristic):
            @property
            def name(self) -> str:
                return "custom"

            @property
            def description(self) -> str:
                return "Custom heuristic"

            def evaluate(self, change: dict) -> DetectionResult:
                return DetectionResult(
                    detected=True,
                    confidence=1.0,
                    reason="Custom detection",
                )

        engine = ADRHeuristicEngine()
        initial_count = len(engine.heuristics)

        engine.register(CustomHeuristic())

        assert len(engine.heuristics) == initial_count + 1


# =============================================================================
# ChangeAnalyzer Tests
# =============================================================================


class TestChangeAnalyzer:
    """Tests for the change analyzer that processes diffs."""

    def test_analyzer_parses_git_diff(self):
        """Test analyzer parses git diff output."""
        from openmemory.api.tools.adr_automation import ChangeAnalyzer

        analyzer = ChangeAnalyzer()

        diff_text = """
diff --git a/src/api/routes.py b/src/api/routes.py
index abc123..def456 100644
--- a/src/api/routes.py
+++ b/src/api/routes.py
@@ -10,6 +10,10 @@ from fastapi import APIRouter
+@router.post("/api/v2/users")
+async def create_user_v2():
+    pass
"""

        changes = analyzer.parse_diff(diff_text)

        assert len(changes) >= 1
        assert changes[0]["file_path"] == "src/api/routes.py"
        assert len(changes[0]["added_lines"]) > 0

    def test_analyzer_handles_new_file(self):
        """Test analyzer handles new file creation."""
        from openmemory.api.tools.adr_automation import ChangeAnalyzer

        analyzer = ChangeAnalyzer()

        diff_text = """
diff --git a/src/new_feature.py b/src/new_feature.py
new file mode 100644
index 0000000..abc123
--- /dev/null
+++ b/src/new_feature.py
@@ -0,0 +1,10 @@
+class NewFeature:
+    pass
"""

        changes = analyzer.parse_diff(diff_text)

        assert len(changes) == 1
        assert changes[0]["file_path"] == "src/new_feature.py"
        assert changes[0]["change_type"] == "added"

    def test_analyzer_handles_deleted_file(self):
        """Test analyzer handles file deletion."""
        from openmemory.api.tools.adr_automation import ChangeAnalyzer

        analyzer = ChangeAnalyzer()

        diff_text = """
diff --git a/src/old_feature.py b/src/old_feature.py
deleted file mode 100644
index abc123..0000000
--- a/src/old_feature.py
+++ /dev/null
@@ -1,10 +0,0 @@
-class OldFeature:
-    pass
"""

        changes = analyzer.parse_diff(diff_text)

        assert len(changes) == 1
        assert changes[0]["file_path"] == "src/old_feature.py"
        assert changes[0]["change_type"] == "deleted"

    def test_analyzer_extracts_line_numbers(self):
        """Test analyzer extracts line numbers for changes."""
        from openmemory.api.tools.adr_automation import ChangeAnalyzer

        analyzer = ChangeAnalyzer()

        diff_text = """
diff --git a/src/api/routes.py b/src/api/routes.py
@@ -10,6 +10,10 @@
 existing_line
+new_line_1
+new_line_2
"""

        changes = analyzer.parse_diff(diff_text)

        assert "line_start" in changes[0]
        assert "line_end" in changes[0]


# =============================================================================
# ADRTemplate Tests
# =============================================================================


class TestADRTemplate:
    """Tests for ADR template structure."""

    def test_template_has_required_sections(self):
        """Test ADR template has all required sections."""
        from openmemory.api.tools.adr_automation import ADRTemplate

        template = ADRTemplate()

        sections = template.get_sections()

        assert "title" in sections
        assert "status" in sections
        assert "context" in sections
        assert "decision" in sections
        assert "consequences" in sections

    def test_template_default_status_is_proposed(self):
        """Test ADR template default status is 'Proposed'."""
        from openmemory.api.tools.adr_automation import ADRTemplate

        template = ADRTemplate()

        assert template.status == "Proposed"

    def test_template_renders_markdown(self):
        """Test ADR template renders to markdown."""
        from openmemory.api.tools.adr_automation import ADRTemplate

        template = ADRTemplate(
            title="Use Redis for Caching",
            context="We need to improve performance for frequently accessed data.",
            decision="We will use Redis as our caching layer.",
            consequences=[
                "Improved response times for cached data",
                "Additional infrastructure to maintain",
            ],
        )

        markdown = template.render()

        assert "# ADR:" in markdown or "# Use Redis for Caching" in markdown
        assert "## Status" in markdown
        assert "Proposed" in markdown
        assert "## Context" in markdown
        assert "## Decision" in markdown
        assert "## Consequences" in markdown


# =============================================================================
# ADRContext Tests
# =============================================================================


class TestADRContext:
    """Tests for ADR context extraction."""

    def test_context_extracts_from_changes(self, sample_dependency_change):
        """Test context extraction from code changes."""
        from openmemory.api.tools.adr_automation import ADRContext

        context = ADRContext.from_changes([sample_dependency_change])

        assert context.files_changed is not None
        assert len(context.files_changed) >= 1
        assert context.change_summary is not None

    def test_context_includes_related_symbols(
        self, sample_api_change, mock_graph_driver
    ):
        """Test context includes related symbols from graph."""
        from openmemory.api.tools.adr_automation import ADRContext

        context = ADRContext.from_changes(
            changes=[sample_api_change],
            graph_driver=mock_graph_driver,
        )

        assert hasattr(context, "related_symbols")

    def test_context_includes_related_adrs(self, sample_api_change, mock_retriever):
        """Test context includes related existing ADRs."""
        from openmemory.api.tools.adr_automation import ADRContext

        mock_hit = MagicMock()
        mock_hit.id = "adr_001"
        mock_hit.source = {"title": "API Versioning Strategy", "content": "..."}
        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_retriever.retrieve.return_value = mock_result

        context = ADRContext.from_changes(
            changes=[sample_api_change],
            retriever=mock_retriever,
        )

        assert hasattr(context, "related_adrs")


# =============================================================================
# ADRGenerator Tests
# =============================================================================


class TestADRGenerator:
    """Tests for ADR content generation."""

    def test_generator_creates_title(self, sample_dependency_change):
        """Test generator creates meaningful title."""
        from openmemory.api.tools.adr_automation import ADRContext, ADRGenerator

        context = ADRContext.from_changes([sample_dependency_change])
        generator = ADRGenerator()

        adr = generator.generate(context)

        assert adr.title is not None
        assert len(adr.title) > 0
        # Title should be descriptive
        assert len(adr.title) >= 10

    def test_generator_creates_context_section(self, sample_dependency_change):
        """Test generator creates context section."""
        from openmemory.api.tools.adr_automation import ADRContext, ADRGenerator

        context = ADRContext.from_changes([sample_dependency_change])
        generator = ADRGenerator()

        adr = generator.generate(context)

        assert adr.context is not None
        assert len(adr.context) > 0

    def test_generator_creates_decision_section(self, sample_dependency_change):
        """Test generator creates decision section."""
        from openmemory.api.tools.adr_automation import ADRContext, ADRGenerator

        context = ADRContext.from_changes([sample_dependency_change])
        generator = ADRGenerator()

        adr = generator.generate(context)

        assert adr.decision is not None
        assert len(adr.decision) > 0

    def test_generator_creates_consequences_section(self, sample_dependency_change):
        """Test generator creates consequences section."""
        from openmemory.api.tools.adr_automation import ADRContext, ADRGenerator

        context = ADRContext.from_changes([sample_dependency_change])
        generator = ADRGenerator()

        adr = generator.generate(context)

        assert adr.consequences is not None
        assert len(adr.consequences) >= 1


# =============================================================================
# ADRAutomationTool Tests
# =============================================================================


class TestADRAutomationTool:
    """Tests for the main ADR automation tool."""

    def test_tool_initialization(self, mock_graph_driver, mock_retriever):
        """Test tool initializes with required dependencies."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        assert tool is not None
        assert tool.config is not None

    def test_tool_with_custom_config(self, mock_graph_driver, mock_retriever):
        """Test tool accepts custom configuration."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool, ADRConfig

        config = ADRConfig(min_confidence=0.8)
        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        assert tool.config.min_confidence == 0.8

    def test_tool_analyzes_changes(
        self, mock_graph_driver, mock_retriever, sample_dependency_change
    ):
        """Test tool analyzes code changes."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze([sample_dependency_change])

        assert result is not None
        assert hasattr(result, "should_create_adr")
        assert hasattr(result, "confidence")

    def test_tool_generates_adr_when_triggered(
        self, mock_graph_driver, mock_retriever, sample_dependency_change
    ):
        """Test tool generates ADR when heuristics trigger."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze([sample_dependency_change])

        if result.should_create_adr:
            assert result.generated_adr is not None
            assert result.generated_adr.title is not None

    def test_tool_links_adr_to_code(
        self, mock_graph_driver, mock_retriever, sample_dependency_change
    ):
        """Test tool links generated ADR to code changes."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool, ADRConfig

        config = ADRConfig(auto_link_to_code=True)
        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.analyze([sample_dependency_change])

        if result.should_create_adr and result.generated_adr:
            assert result.code_links is not None
            assert len(result.code_links) >= 1

    def test_tool_returns_related_adrs(
        self, mock_graph_driver, mock_retriever, sample_api_change
    ):
        """Test tool returns related existing ADRs."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool, ADRConfig

        # Setup mock to return related ADR
        mock_hit = MagicMock()
        mock_hit.id = "adr_001"
        mock_hit.source = {"title": "API Versioning Strategy"}
        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_retriever.retrieve.return_value = mock_result

        config = ADRConfig(include_related_adrs=True)
        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.analyze([sample_api_change])

        assert hasattr(result, "related_adrs")

    def test_tool_includes_impact_analysis(
        self, mock_graph_driver, mock_retriever, sample_breaking_api_change
    ):
        """Test tool includes impact analysis for breaking changes."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool, ADRConfig

        config = ADRConfig(include_impact_analysis=True)
        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        result = tool.analyze([sample_breaking_api_change])

        if result.should_create_adr:
            assert hasattr(result, "impact_analysis")

    def test_tool_no_adr_for_trivial_changes(
        self, mock_graph_driver, mock_retriever, sample_trivial_change
    ):
        """Test tool does not suggest ADR for trivial changes."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        result = tool.analyze([sample_trivial_change])

        assert result.should_create_adr is False


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_adr_automation_tool(self, mock_graph_driver, mock_retriever):
        """Test create_adr_automation_tool factory function."""
        from openmemory.api.tools.adr_automation import create_adr_automation_tool

        tool = create_adr_automation_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        assert tool is not None

    def test_create_adr_automation_tool_with_config(
        self, mock_graph_driver, mock_retriever
    ):
        """Test create_adr_automation_tool with custom config."""
        from openmemory.api.tools.adr_automation import (
            ADRConfig,
            create_adr_automation_tool,
        )

        config = ADRConfig(min_confidence=0.9)
        tool = create_adr_automation_tool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
            config=config,
        )

        assert tool.config.min_confidence == 0.9


# =============================================================================
# MCP Tool Interface Tests
# =============================================================================


class TestMCPToolInterface:
    """Tests for MCP tool interface compliance."""

    def test_tool_has_mcp_schema(self, mock_graph_driver, mock_retriever):
        """Test tool provides MCP schema."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        schema = tool.get_mcp_schema()

        assert "name" in schema
        assert "description" in schema
        assert "inputSchema" in schema

    def test_tool_input_schema_valid(self, mock_graph_driver, mock_retriever):
        """Test tool input schema is valid JSON Schema."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        schema = tool.get_mcp_schema()
        input_schema = schema["inputSchema"]

        assert "type" in input_schema
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

    def test_tool_execute_via_mcp(
        self, mock_graph_driver, mock_retriever, sample_dependency_change
    ):
        """Test tool execution via MCP interface."""
        from openmemory.api.tools.adr_automation import ADRAutomationTool

        tool = ADRAutomationTool(
            graph_driver=mock_graph_driver,
            retriever=mock_retriever,
        )

        # Execute via MCP-style interface
        input_data = {"changes": [sample_dependency_change]}
        result = tool.execute(input_data)

        assert result is not None
        assert "should_create_adr" in result or hasattr(result, "should_create_adr")
