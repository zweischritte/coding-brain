"""
Pydantic settings for all services and secrets.

This module provides centralized configuration management with:
- Required secrets validation at startup (fail-fast)
- Type-safe access to all configuration values
- Sensible defaults for optional settings
- Singleton pattern for consistent access
"""

from pydantic import field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required secrets will raise ValidationError if not provided.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars not in schema
    )

    # ==========================================
    # Required Secrets - fail fast if missing
    # ==========================================
    jwt_secret_key: str
    postgres_password: str
    neo4j_password: str
    # Optional: only needed for OpenAI-backed features (e.g., embeddings)
    openai_api_key: str = ""

    # ==========================================
    # PostgreSQL Configuration
    # ==========================================
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "codingbrain"
    postgres_user: str = "codingbrain"

    # ==========================================
    # Neo4j Configuration
    # ==========================================
    neo4j_url: str = "bolt://neo4j:7687"
    neo4j_username: str = "neo4j"
    neo4j_database: str = "neo4j"

    # ==========================================
    # Valkey/Redis Configuration
    # ==========================================
    valkey_host: str = "valkey"
    valkey_port: int = 6379

    # ==========================================
    # Qdrant Configuration
    # ==========================================
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333

    # ==========================================
    # OpenSearch Configuration
    # ==========================================
    opensearch_hosts: str = "opensearch:9200"
    opensearch_username: str = ""
    opensearch_password: str = ""
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = True
    opensearch_timeout: int = 30
    opensearch_max_retries: int = 3
    opensearch_pool_connections: int = 10
    opensearch_pool_maxsize: int = 10

    # ==========================================
    # JWT Configuration
    # ==========================================
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "https://codingbrain.local"
    jwt_audience: str = "https://api.codingbrain.local"
    jwt_expiry_minutes: int = 60

    # ==========================================
    # CORS Configuration
    # ==========================================
    cors_allowed_origins: str = "http://localhost:3000,http://localhost:3433"

    # ==========================================
    # Application Configuration
    # ==========================================
    user: str = "default_user"
    enable_new_auth_flow: bool = False

    # ==========================================
    # Validators
    # ==========================================
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """JWT secret must be at least 32 characters for security."""
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        return v

    # ==========================================
    # Computed Properties
    # ==========================================
    @computed_field
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL from components."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field
    @property
    def valkey_url(self) -> str:
        """Construct Valkey/Redis connection URL."""
        return f"redis://{self.valkey_host}:{self.valkey_port}"


# ==========================================
# Singleton Access
# ==========================================
_settings: Settings | None = None


def get_settings() -> Settings:
    """
    Get the singleton Settings instance.

    Returns:
        Settings: The application settings

    Raises:
        ValidationError: If required settings are missing or invalid
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    Reset the settings singleton (for testing purposes).
    """
    global _settings
    _settings = None
