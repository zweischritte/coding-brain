import os

USER_ID = os.getenv("USER", "default_user")
DEFAULT_APP_ID = "openmemory"


# =============================================================================
# Feature Flags for Business Concept System
# =============================================================================

class BusinessConceptsConfig:
    """
    Configuration for Business Concepts integration.

    Environment variables:
    - BUSINESS_CONCEPTS_ENABLED: Master switch for concept system (default: false)
    - BUSINESS_CONCEPTS_AUTO_EXTRACT: Enable auto-extraction on memory add (default: false)
    - BUSINESS_CONCEPTS_OPENAI_API_KEY: OpenAI API key for extraction (required for auto-extract)
    - BUSINESS_CONCEPTS_EXTRACTION_MODEL: Model for extraction (default: gpt-4o-mini)
    - BUSINESS_CONCEPTS_MIN_CONFIDENCE: Minimum confidence for concept storage (default: 0.5)
    - BUSINESS_CONCEPTS_CONTRADICTION_DETECTION: Enable contradiction detection (default: false)

    Vector Embedding Configuration:
    - BUSINESS_CONCEPTS_EMBEDDING_ENABLED: Enable Qdrant vector embeddings (default: true when concepts enabled)
    - BUSINESS_CONCEPTS_EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)
    - BUSINESS_CONCEPTS_EMBEDDING_DIMS: Embedding dimensions (default: 1536)
    - BUSINESS_CONCEPTS_COLLECTION: Qdrant collection name (default: business_concepts)
    - BUSINESS_CONCEPTS_SIMILARITY_THRESHOLD: Threshold for deduplication (default: 0.75)
    - BUSINESS_CONCEPTS_QDRANT_HOST: Qdrant host (default: mem0_store, uses QDRANT_HOST if not set)
    - BUSINESS_CONCEPTS_QDRANT_PORT: Qdrant port (default: 6333, uses QDRANT_PORT if not set)
    """

    @staticmethod
    def is_enabled() -> bool:
        """Check if business concepts system is enabled."""
        return os.getenv("BUSINESS_CONCEPTS_ENABLED", "false").lower() == "true"

    @staticmethod
    def is_auto_extract_enabled() -> bool:
        """Check if auto-extraction on memory add is enabled."""
        return (
            BusinessConceptsConfig.is_enabled()
            and os.getenv("BUSINESS_CONCEPTS_AUTO_EXTRACT", "false").lower() == "true"
        )

    @staticmethod
    def is_contradiction_detection_enabled() -> bool:
        """Check if contradiction detection is enabled."""
        return (
            BusinessConceptsConfig.is_enabled()
            and os.getenv("BUSINESS_CONCEPTS_CONTRADICTION_DETECTION", "false").lower() == "true"
        )

    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key for extraction."""
        return os.getenv("BUSINESS_CONCEPTS_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    @staticmethod
    def get_extraction_model() -> str:
        """Get model for concept extraction."""
        return os.getenv("BUSINESS_CONCEPTS_EXTRACTION_MODEL", "gpt-4o-mini")

    @staticmethod
    def get_min_confidence() -> float:
        """Get minimum confidence threshold for storing concepts."""
        try:
            return float(os.getenv("BUSINESS_CONCEPTS_MIN_CONFIDENCE", "0.5"))
        except ValueError:
            return 0.5

    @staticmethod
    def get_max_tokens_per_chunk() -> int:
        """Get max tokens per extraction chunk."""
        try:
            return int(os.getenv("BUSINESS_CONCEPTS_MAX_TOKENS_PER_CHUNK", "8000"))
        except ValueError:
            return 8000

    # =========================================================================
    # Vector Embedding Configuration
    # =========================================================================

    @staticmethod
    def is_embedding_enabled() -> bool:
        """Check if concept vector embeddings are enabled."""
        return (
            BusinessConceptsConfig.is_enabled()
            and os.getenv("BUSINESS_CONCEPTS_EMBEDDING_ENABLED", "true").lower() == "true"
        )

    @staticmethod
    def get_embedding_model() -> str:
        """Get the embedding model name."""
        return os.getenv("BUSINESS_CONCEPTS_EMBEDDING_MODEL", "text-embedding-3-small")

    @staticmethod
    def get_embedding_dims() -> int:
        """Get embedding dimensions."""
        try:
            return int(os.getenv("BUSINESS_CONCEPTS_EMBEDDING_DIMS", "1536"))
        except ValueError:
            return 1536

    @staticmethod
    def get_collection_name() -> str:
        """Get the Qdrant collection name for concepts."""
        return os.getenv("BUSINESS_CONCEPTS_COLLECTION", "business_concepts")

    @staticmethod
    def get_similarity_threshold() -> float:
        """Get similarity threshold for concept deduplication."""
        try:
            return float(os.getenv("BUSINESS_CONCEPTS_SIMILARITY_THRESHOLD", "0.75"))
        except ValueError:
            return 0.75

    @staticmethod
    def get_qdrant_host() -> str:
        """Get Qdrant host for concept vector store."""
        return os.getenv(
            "BUSINESS_CONCEPTS_QDRANT_HOST",
            os.getenv("QDRANT_HOST", "codingbrain_store")
        )

    @staticmethod
    def get_qdrant_port() -> int:
        """Get Qdrant port for concept vector store."""
        try:
            return int(os.getenv(
                "BUSINESS_CONCEPTS_QDRANT_PORT",
                os.getenv("QDRANT_PORT", "6433")
            ))
        except ValueError:
            return 6433


# Convenience alias
business_concepts_config = BusinessConceptsConfig()