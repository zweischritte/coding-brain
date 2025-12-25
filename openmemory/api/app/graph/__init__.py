"""
Graph module for OpenMemory Neo4j integration.

This module provides:
- neo4j_client: Neo4j driver lifecycle management
- metadata_projector: Deterministic metadata-to-graph projection
"""

from app.graph.neo4j_client import (
    get_neo4j_driver,
    close_neo4j_driver,
    is_neo4j_configured,
    get_neo4j_session,
)
from app.graph.metadata_projector import MetadataProjector

__all__ = [
    "get_neo4j_driver",
    "close_neo4j_driver",
    "is_neo4j_configured",
    "get_neo4j_session",
    "MetadataProjector",
]
