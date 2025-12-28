"""GDPR Schemas for API responses.

This module defines the Pydantic models and dataclasses used for GDPR
operations including Subject Access Request (SAR) responses and
deletion results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SARResponse:
    """Subject Access Request response containing all user data.

    This response aggregates data from all stores (PostgreSQL, Neo4j,
    Qdrant, OpenSearch, Valkey) for a specific user.

    Attributes:
        user_id: The user whose data was exported
        export_date: Timestamp of the export
        format_version: Schema version for forwards compatibility
        postgres: Data from PostgreSQL (users, memories, apps, feedback, experiments)
        neo4j: Data from Neo4j (nodes, relationships)
        qdrant: Data from Qdrant (embeddings)
        opensearch: Data from OpenSearch (indexed documents)
        valkey: Data from Valkey (session data)
        export_duration_ms: Time taken to complete the export
        partial: True if any store failed during export
        errors: List of error messages from failed stores
    """
    user_id: str
    export_date: datetime
    format_version: str = "1.0"

    # Data from each store
    postgres: Dict[str, Any] = field(default_factory=dict)
    neo4j: Dict[str, Any] = field(default_factory=dict)
    qdrant: Dict[str, Any] = field(default_factory=dict)
    opensearch: Dict[str, Any] = field(default_factory=dict)
    valkey: Dict[str, Any] = field(default_factory=dict)

    # Property alias for consistent access
    @property
    def stores(self) -> Dict[str, Dict[str, Any]]:
        """Get all store data as a dictionary."""
        return {
            "postgres": self.postgres,
            "neo4j": self.neo4j,
            "qdrant": self.qdrant,
            "opensearch": self.opensearch,
            "valkey": self.valkey,
        }

    # Metadata
    export_duration_ms: Optional[int] = None
    partial: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "export_date": self.export_date.isoformat(),
            "format_version": self.format_version,
            "postgres": self.postgres,
            "neo4j": self.neo4j,
            "qdrant": self.qdrant,
            "opensearch": self.opensearch,
            "valkey": self.valkey,
            "export_duration_ms": self.export_duration_ms,
            "partial": self.partial,
            "errors": self.errors,
        }


@dataclass
class DeletionResult:
    """Result of a user deletion operation.

    This result contains the status of deletion from each store,
    with counts of deleted records and any errors encountered.

    Attributes:
        audit_id: Unique identifier for this deletion operation
        user_id: The user whose data was deleted
        timestamp: When the deletion was performed
        results: Per-store deletion results
        success: True if all deletions succeeded
        errors: List of error messages from failed deletions
    """
    audit_id: str
    user_id: str
    timestamp: datetime
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    success: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "audit_id": self.audit_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "results": self.results,
            "success": self.success,
            "errors": self.errors,
        }
