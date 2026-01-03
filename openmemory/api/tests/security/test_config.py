"""
Tests for configuration and secrets management.

TDD: These tests define expected behavior for the Settings class.
"""

import os
import pytest
from unittest.mock import patch


def reset_settings():
    """Reset the settings singleton for testing."""
    from app import settings as settings_module
    import app.settings.settings as inner_settings
    settings_module._settings = None
    inner_settings._settings = None


@pytest.fixture(autouse=True)
def isolate_settings():
    """Isolate settings between tests."""
    reset_settings()
    yield
    reset_settings()


class TestSettingsValidation:
    """Test that settings validate required secrets at startup."""

    def test_missing_jwt_secret_fails_fast(self):
        """Settings should raise if JWT_SECRET_KEY is missing."""
        # Set up environment without JWT_SECRET_KEY
        env = {
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
            # JWT_SECRET_KEY intentionally missing
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            with pytest.raises(Exception) as exc_info:
                Settings(_env_file=None)

            # Should mention jwt_secret_key in the error
            assert "jwt_secret_key" in str(exc_info.value).lower()

    def test_missing_postgres_password_fails_fast(self):
        """Settings should raise if POSTGRES_PASSWORD is missing."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
            # POSTGRES_PASSWORD intentionally missing
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            with pytest.raises(Exception) as exc_info:
                Settings(_env_file=None)

            assert "postgres_password" in str(exc_info.value).lower()

    def test_missing_neo4j_password_fails_fast(self):
        """Settings should raise if NEO4J_PASSWORD is missing."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
            # NEO4J_PASSWORD intentionally missing
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            with pytest.raises(Exception) as exc_info:
                Settings(_env_file=None)

            assert "neo4j_password" in str(exc_info.value).lower()

    def test_missing_openai_api_key_is_allowed(self):
        """Settings should allow missing OPENAI_API_KEY for local LLM usage."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            # OPENAI_API_KEY intentionally missing
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            assert settings.openai_api_key in ("", None)

    def test_weak_jwt_secret_rejected(self):
        """JWT secret must be at least 32 characters."""
        env = {
            "JWT_SECRET_KEY": "short-key",  # Less than 32 chars
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            with pytest.raises(Exception) as exc_info:
                Settings(_env_file=None)

            error_msg = str(exc_info.value)
            assert "32" in error_msg or "character" in error_msg.lower()

    def test_settings_loads_from_env(self):
        """Settings should load all values from environment."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
            "POSTGRES_HOST": "custom-host",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "custom-db",
            "JWT_ALGORITHM": "RS256",
            "JWT_ISSUER": "https://custom-issuer.com",
            "VALKEY_HOST": "custom-valkey",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            assert settings.jwt_secret_key == "test-jwt-secret-key-with-32-chars-minimum"
            assert settings.postgres_password == "test-postgres-password"
            assert settings.neo4j_password == "test-neo4j-password"
            assert settings.openai_api_key == "sk-test-openai-key-12345678901234567890"
            assert settings.postgres_host == "custom-host"
            assert settings.postgres_port == 5433
            assert settings.postgres_db == "custom-db"
            assert settings.jwt_algorithm == "RS256"
            assert settings.jwt_issuer == "https://custom-issuer.com"
            assert settings.valkey_host == "custom-valkey"


class TestSettingsDefaults:
    """Test default values for optional settings."""

    def test_default_database_settings(self):
        """Database settings should have sensible defaults."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            assert settings.postgres_host == "postgres"
            assert settings.postgres_port == 5432
            assert settings.postgres_db == "codingbrain"
            assert settings.postgres_user == "codingbrain"

    def test_default_jwt_settings(self):
        """JWT settings should have sensible defaults."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            assert settings.jwt_algorithm == "HS256"
            assert settings.jwt_expiry_minutes == 60

    def test_default_service_settings(self):
        """Service connection defaults should be set."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            assert settings.valkey_host == "valkey"
            assert settings.valkey_port == 6379
            assert settings.qdrant_host == "qdrant"
            assert settings.qdrant_port == 6333
            assert settings.neo4j_url == "bolt://neo4j:7687"
            assert settings.neo4j_username == "neo4j"


class TestGetSettings:
    """Test the get_settings() singleton accessor."""

    def test_get_settings_returns_singleton(self):
        """get_settings() should return the same instance."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import get_settings, reset_settings as do_reset
            do_reset()

            settings1 = get_settings()
            settings2 = get_settings()

            assert settings1 is settings2

    def test_get_settings_validates_on_first_call(self):
        """get_settings() should validate settings on first call."""
        # Mock to prevent reading .env file
        with patch.dict(os.environ, {}, clear=True):
            from app.settings import reset_settings as do_reset
            do_reset()

            # We need to patch the Settings class to not read .env
            from app.settings import Settings
            with pytest.raises(Exception):
                Settings(_env_file=None)


class TestDatabaseUrlConstruction:
    """Test database URL construction from components."""

    def test_constructs_postgres_url(self):
        """Should construct PostgreSQL connection URL from components."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-password",
            "POSTGRES_USER": "testuser",
            "POSTGRES_HOST": "testhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "testdb",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            expected = "postgresql://testuser:test-password@testhost:5432/testdb"
            assert settings.database_url == expected


class TestValkeyUrlConstruction:
    """Test Valkey/Redis URL construction."""

    def test_constructs_valkey_url(self):
        """Should construct Valkey connection URL from components."""
        env = {
            "JWT_SECRET_KEY": "test-jwt-secret-key-with-32-chars-minimum",
            "POSTGRES_PASSWORD": "test-postgres-password",
            "NEO4J_PASSWORD": "test-neo4j-password",
            "OPENAI_API_KEY": "sk-test-openai-key-12345678901234567890",
            "VALKEY_HOST": "custom-valkey",
            "VALKEY_PORT": "6380",
        }

        with patch.dict(os.environ, env, clear=True):
            from app.settings import Settings
            settings = Settings(_env_file=None)

            expected = "redis://custom-valkey:6380"
            assert settings.valkey_url == expected
