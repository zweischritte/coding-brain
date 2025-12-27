"""
Valkey-backed episodic memory store with TTL and recency decay.

This module implements session-scoped ephemeral memory storage using Valkey (Redis).
Key features:
- Session-scoped storage with configurable TTL (default 24 hours)
- Tenant isolation via user_id key prefix
- Recency decay support for memory prioritization
- Reference resolution within session context
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from memory.episodic import (
    EpisodicMemory,
    EpisodicMemoryConfig,
    EpisodicMemoryStore,
)


logger = logging.getLogger(__name__)


class ValkeyClientProtocol(Protocol):
    """Protocol for Valkey/Redis client operations."""

    def get(self, key: str) -> Optional[bytes]: ...
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool: ...
    def setex(self, key: str, seconds: int, value: str) -> bool: ...
    def delete(self, *keys: str) -> int: ...
    def keys(self, pattern: str) -> list[bytes]: ...
    def mget(self, keys: list[str]) -> list[Optional[bytes]]: ...
    def ttl(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> bool: ...
    def zadd(self, key: str, mapping: dict[str, float], nx: bool = False, xx: bool = False) -> int: ...
    def zrange(self, key: str, start: int, end: int, withscores: bool = False, desc: bool = False) -> list: ...
    def zincrby(self, key: str, amount: float, member: str) -> float: ...
    def zrem(self, key: str, *members: str) -> int: ...
    def ping(self) -> bool: ...


class ValkeyEpisodicStore(EpisodicMemoryStore):
    """
    Valkey-backed implementation of EpisodicMemoryStore.

    Provides session-scoped ephemeral memory with:
    - TTL-based expiration (default 24 hours)
    - Tenant isolation via user_id key prefix
    - Recency scoring via sorted sets
    - Reference resolution within session context

    Key format: episodic:{user_id}:{session_id}:{memory_id}
    Session set: episodic:session:{user_id}:{session_id} (sorted by recency)
    """

    def __init__(
        self,
        client: ValkeyClientProtocol,
        user_id: str,
        config: Optional[EpisodicMemoryConfig] = None,
    ):
        """
        Initialize the Valkey episodic store.

        Args:
            client: Valkey/Redis client instance
            user_id: The current user's ID for tenant isolation
            config: Configuration for episodic memory behavior
        """
        self.client = client
        self.user_id = user_id
        self.config = config or EpisodicMemoryConfig()
        self._ttl_seconds = int(self.config.session_ttl.total_seconds())

    def _memory_key(self, session_id: str, memory_id: str) -> str:
        """Generate the Redis key for a memory."""
        return f"episodic:{self.user_id}:{session_id}:{memory_id}"

    def _session_key(self, session_id: str) -> str:
        """Generate the Redis key for a session's sorted set."""
        return f"episodic:session:{self.user_id}:{session_id}"

    def _serialize_memory(self, memory: EpisodicMemory) -> str:
        """Serialize an EpisodicMemory to JSON."""
        data = asdict(memory)
        # Convert datetime to ISO format
        data["created_at"] = memory.created_at.isoformat()
        return json.dumps(data)

    def _deserialize_memory(self, data: bytes | str) -> EpisodicMemory:
        """Deserialize JSON to an EpisodicMemory."""
        # Handle both bytes (real Valkey) and str (mock)
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        obj = json.loads(data)
        # Convert ISO string back to datetime
        obj["created_at"] = datetime.fromisoformat(obj["created_at"])
        return EpisodicMemory(**obj)

    def add(self, memory: EpisodicMemory) -> None:
        """
        Add an episodic memory to the store.

        The memory is stored with a TTL and added to the session's
        sorted set for recency-based queries.

        Args:
            memory: The episodic memory to add
        """
        key = self._memory_key(memory.session_id, memory.memory_id)
        session_key = self._session_key(memory.session_id)

        # Store the memory with TTL
        serialized = self._serialize_memory(memory)
        self.client.setex(key, self._ttl_seconds, serialized)

        # Add to session sorted set with recency score
        # Using recency_score as the score for sorting
        self.client.zadd(session_key, {memory.memory_id: memory.recency_score})

        # Set TTL on the session sorted set as well
        self.client.expire(session_key, self._ttl_seconds)

        logger.debug(
            f"Added episodic memory {memory.memory_id} to session {memory.session_id}"
        )

    def get(self, memory_id: str, session_id: Optional[str] = None) -> Optional[EpisodicMemory]:
        """
        Get an episodic memory by ID.

        Args:
            memory_id: The memory ID to retrieve
            session_id: The session ID (required for key construction)

        Returns:
            The memory if found and owned by current user, None otherwise
        """
        if session_id is None:
            logger.warning("session_id is required for get()")
            return None

        key = self._memory_key(session_id, memory_id)
        data = self.client.get(key)

        if data is None:
            return None

        try:
            memory = self._deserialize_memory(data)
            # Verify tenant isolation
            if memory.user_id != self.user_id:
                logger.warning(
                    f"Tenant isolation violation attempt: {self.user_id} tried to access {memory.user_id}'s memory"
                )
                return None
            return memory
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to deserialize memory {memory_id}: {e}")
            return None

    def get_session_memories(
        self,
        session_id: str,
        include_summarized: bool = False,
    ) -> list[EpisodicMemory]:
        """
        Get all memories for a session, sorted by recency.

        Args:
            session_id: The session ID
            include_summarized: Whether to include summarized memories

        Returns:
            List of memories sorted by recency score (highest first)
        """
        session_key = self._session_key(session_id)

        # Get memory IDs from sorted set (sorted by score descending)
        memory_ids_with_scores = self.client.zrange(
            session_key, 0, -1, withscores=True, desc=True
        )

        if not memory_ids_with_scores:
            return []

        # Fetch all memories
        memories: list[EpisodicMemory] = []
        for item in memory_ids_with_scores:
            if isinstance(item, tuple):
                memory_id, score = item
            else:
                memory_id = item
                score = 0.0

            # Decode memory_id if it's bytes
            if isinstance(memory_id, bytes):
                memory_id = memory_id.decode("utf-8")

            memory = self.get(memory_id, session_id=session_id)
            if memory is None:
                continue

            # Update recency score from sorted set
            memory.recency_score = float(score)

            # Filter summarized if requested
            if not include_summarized and memory.summarized:
                continue

            memories.append(memory)

        return memories

    def update_recency(
        self,
        session_id: str,
        decay_factor: float = 0.9,
    ) -> None:
        """
        Decay recency scores for all session memories.

        Args:
            session_id: The session ID
            decay_factor: Factor to multiply scores by (0-1)
        """
        session_key = self._session_key(session_id)

        # Get all memory IDs with current scores
        memory_ids_with_scores = self.client.zrange(
            session_key, 0, -1, withscores=True
        )

        if not memory_ids_with_scores:
            return

        for item in memory_ids_with_scores:
            if isinstance(item, tuple):
                memory_id, current_score = item
            else:
                continue

            # Decode memory_id if it's bytes
            if isinstance(memory_id, bytes):
                memory_id = memory_id.decode("utf-8")

            # Calculate new score
            new_score = float(current_score) * decay_factor

            # Update the score in the sorted set
            self.client.zadd(session_key, {memory_id: new_score}, xx=True)

            # Also update the stored memory object
            memory = self.get(memory_id, session_id=session_id)
            if memory:
                memory.recency_score = new_score
                key = self._memory_key(session_id, memory_id)
                serialized = self._serialize_memory(memory)
                # Preserve existing TTL
                ttl = self.client.ttl(key)
                if ttl > 0:
                    self.client.setex(key, ttl, serialized)
                else:
                    self.client.setex(key, self._ttl_seconds, serialized)

    def mark_summarized(self, memory_ids: list[str], session_id: Optional[str] = None) -> None:
        """
        Mark memories as summarized.

        Args:
            memory_ids: List of memory IDs to mark
            session_id: The session ID (required)
        """
        if session_id is None:
            logger.warning("session_id is required for mark_summarized()")
            return

        for memory_id in memory_ids:
            memory = self.get(memory_id, session_id=session_id)
            if memory is None:
                continue

            memory.summarized = True
            key = self._memory_key(session_id, memory_id)
            serialized = self._serialize_memory(memory)

            # Preserve existing TTL
            ttl = self.client.ttl(key)
            if ttl > 0:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.setex(key, self._ttl_seconds, serialized)

    def delete(self, memory_id: str, session_id: Optional[str] = None) -> bool:
        """
        Delete an episodic memory.

        Args:
            memory_id: The memory ID to delete
            session_id: The session ID (required)

        Returns:
            True if deleted, False if not found or not owned
        """
        if session_id is None:
            logger.warning("session_id is required for delete()")
            return False

        # Check ownership first
        memory = self.get(memory_id, session_id=session_id)
        if memory is None:
            return False

        key = self._memory_key(session_id, memory_id)
        session_key = self._session_key(session_id)

        # Delete from storage
        deleted = self.client.delete(key)

        # Remove from session sorted set
        self.client.zrem(session_key, memory_id)

        return deleted > 0

    def expire_session(self, session_id: str) -> None:
        """
        Expire all memories in a session.

        Args:
            session_id: The session ID to expire
        """
        session_key = self._session_key(session_id)

        # Get all memory IDs
        memory_ids = self.client.zrange(session_key, 0, -1)

        if memory_ids:
            # Delete all memory keys
            keys_to_delete = []
            for memory_id in memory_ids:
                if isinstance(memory_id, bytes):
                    memory_id = memory_id.decode("utf-8")
                keys_to_delete.append(self._memory_key(session_id, memory_id))

            if keys_to_delete:
                self.client.delete(*keys_to_delete)

        # Delete the session sorted set
        self.client.delete(session_key)

    def health_check(self) -> bool:
        """
        Check if the Valkey connection is healthy.

        Returns:
            True if connected, False otherwise
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Valkey health check failed: {e}")
            return False


def get_valkey_episodic_store(
    user_id: str,
    config: Optional[EpisodicMemoryConfig] = None,
) -> Optional[ValkeyEpisodicStore]:
    """
    Factory function to create a ValkeyEpisodicStore.

    Attempts to connect to Valkey using environment configuration.
    Returns None if Valkey is not available.

    Args:
        user_id: The user ID for tenant isolation
        config: Optional configuration

    Returns:
        ValkeyEpisodicStore instance or None if unavailable
    """
    import os

    try:
        import redis

        host = os.getenv("VALKEY_HOST", "valkey")
        port = int(os.getenv("VALKEY_PORT", "6379"))

        client = redis.Redis(host=host, port=port, socket_timeout=5)
        client.ping()

        return ValkeyEpisodicStore(client=client, user_id=user_id, config=config)
    except Exception as e:
        logger.warning(f"Failed to connect to Valkey: {e}")
        return None
