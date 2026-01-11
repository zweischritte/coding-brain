"""
Vector store metadata synchronization utilities.

This module provides functions to sync metadata to Qdrant without re-computing
embeddings. This is critical for metadata-only updates where the text content
hasn't changed.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def build_qdrant_payload(memory, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Qdrant payload from memory and metadata.

    Matches the structure used in memory creation (see mem0 add flow) and
    the REST endpoint update logic.

    Args:
        memory: Memory ORM object with content, user, created_at, updated_at
        metadata: Current metadata dict from memory.metadata_

    Returns:
        Dict with all payload fields for Qdrant
    """
    content = memory.content or ""

    payload = {
        "data": content,
        "user_id": memory.user.user_id if memory.user else None,
        "hash": hashlib.md5(content.encode()).hexdigest(),
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
        # Structured metadata fields
        "source_app": metadata.get("source_app"),
        "mcp_client": metadata.get("mcp_client"),
        "category": metadata.get("category"),
        "scope": metadata.get("scope"),
        "artifact_type": metadata.get("artifact_type"),
        "artifact_ref": metadata.get("artifact_ref"),
        "entity": metadata.get("entity"),
        "source": metadata.get("source"),
        "access_entity": metadata.get("access_entity"),
        "evidence": metadata.get("evidence"),
        "tags": metadata.get("tags", {}),
    }

    # Remove None values to keep payload clean
    return {k: v for k, v in payload.items() if v is not None}


def sync_metadata_to_qdrant(
    memory_id: str,
    memory,
    metadata: Dict[str, Any],
    memory_client: Optional[Any] = None,
) -> bool:
    """
    Sync metadata to Qdrant without re-computing embeddings.

    Uses set_payload() to preserve existing vectors. This is the correct
    approach for metadata-only updates.

    Args:
        memory_id: UUID string of the memory
        memory: Memory ORM object
        metadata: Updated metadata dict
        memory_client: Optional memory client (will get if not provided)

    Returns:
        True if sync succeeded, False otherwise
    """
    if memory_client is None:
        try:
            from app.mcp_server import get_memory_client_safe
            memory_client = get_memory_client_safe()
        except (ImportError, Exception) as e:
            logger.warning(f"Cannot get memory client for Qdrant sync: {e}")
            return False

    if not memory_client:
        logger.warning(f"Cannot sync metadata to Qdrant: memory client unavailable")
        return False

    try:
        payload = build_qdrant_payload(memory, metadata)

        # Use set_payload() to preserve the vector!
        # This is critical - using update() with vector=None would DELETE the vector.
        vs = getattr(memory_client, "vector_store", None)
        if vs is None:
            logger.warning(f"Cannot sync metadata to Qdrant: vector_store not available")
            return False

        # Try the new set_payload method first, fall back to client.set_payload
        if hasattr(vs, "set_payload"):
            vs.set_payload(
                vector_id=memory_id,
                payload=payload,
            )
        elif hasattr(vs, "client") and hasattr(vs, "collection_name"):
            # Fallback to direct client access (for compatibility)
            vs.client.set_payload(
                collection_name=vs.collection_name,
                payload=payload,
                points=[memory_id],
            )
        else:
            logger.warning(f"Cannot sync metadata to Qdrant: set_payload not supported")
            return False

        logger.debug(f"Synced metadata to Qdrant for memory {memory_id}")
        return True

    except Exception as e:
        logger.warning(f"Failed to sync metadata to Qdrant for {memory_id}: {e}")
        return False


def sync_metadata_to_qdrant_with_mcp_client(
    memory_id: str,
    memory,
    metadata: Dict[str, Any],
    mcp_client_override: Optional[str] = None,
    memory_client: Optional[Any] = None,
) -> bool:
    """
    Sync metadata to Qdrant with optional mcp_client override.

    This extends sync_metadata_to_qdrant to allow changing the mcp_client field
    (which app wrote the memory). Uses set_payload() to preserve existing vectors.

    Args:
        memory_id: UUID string of the memory
        memory: Memory ORM object
        metadata: Updated metadata dict
        mcp_client_override: If provided, overrides the mcp_client field in Qdrant
        memory_client: Optional memory client (will get if not provided)

    Returns:
        True if sync succeeded, False otherwise
    """
    if memory_client is None:
        try:
            from app.mcp_server import get_memory_client_safe
            memory_client = get_memory_client_safe()
        except (ImportError, Exception) as e:
            logger.warning(f"Cannot get memory client for Qdrant sync: {e}")
            return False

    if not memory_client:
        logger.warning(f"Cannot sync metadata to Qdrant: memory client unavailable")
        return False

    try:
        # Build base payload
        payload = build_qdrant_payload(memory, metadata)

        # Apply mcp_client override if provided
        if mcp_client_override is not None:
            payload["mcp_client"] = mcp_client_override

        # Use set_payload() to preserve the vector
        vs = getattr(memory_client, "vector_store", None)
        if vs is None:
            logger.warning(f"Cannot sync metadata to Qdrant: vector_store not available")
            return False

        # Try the new set_payload method first, fall back to client.set_payload
        if hasattr(vs, "set_payload"):
            vs.set_payload(
                vector_id=memory_id,
                payload=payload,
            )
        elif hasattr(vs, "client") and hasattr(vs, "collection_name"):
            # Fallback to direct client access (for compatibility)
            vs.client.set_payload(
                collection_name=vs.collection_name,
                payload=payload,
                points=[memory_id],
            )
        else:
            logger.warning(f"Cannot sync metadata to Qdrant: set_payload not supported")
            return False

        logger.debug(f"Synced metadata to Qdrant for memory {memory_id} (mcp_client={mcp_client_override})")
        return True

    except Exception as e:
        logger.warning(f"Failed to sync metadata to Qdrant for {memory_id}: {e}")
        return False
