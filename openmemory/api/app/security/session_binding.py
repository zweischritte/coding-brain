"""
Session binding store for MCP SSE authentication.

Binds SSE session_id to authenticated principal to prevent session hijacking.
"""
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
from uuid import UUID


@dataclass
class SessionBinding:
    """Binding between SSE session and authenticated principal."""

    session_id: UUID
    user_id: str
    org_id: str
    issued_at: datetime
    expires_at: datetime
    dpop_thumbprint: Optional[str] = None


class MemorySessionBindingStore:
    """Thread-safe in-memory session binding store with TTL.

    Used for local development and single-process deployments.
    For multi-worker production deployments, use ValkeySessionBindingStore (Phase 1).
    """

    def __init__(self, default_ttl_seconds: int = 1800):
        self._bindings: Dict[UUID, SessionBinding] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl_seconds

    def create(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> SessionBinding:
        """Create a new session binding."""
        ttl = ttl_seconds or self._default_ttl
        now = datetime.now(timezone.utc)
        binding = SessionBinding(
            session_id=session_id,
            user_id=user_id,
            org_id=org_id,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl),
            dpop_thumbprint=dpop_thumbprint,
        )
        with self._lock:
            self._bindings[session_id] = binding
        return binding

    def get(self, session_id: UUID) -> Optional[SessionBinding]:
        """Get binding if exists and not expired."""
        with self._lock:
            binding = self._bindings.get(session_id)
            if binding and binding.expires_at > datetime.now(timezone.utc):
                return binding
            elif binding:
                del self._bindings[session_id]  # Cleanup expired
            return None

    def validate(
        self,
        session_id: UUID,
        user_id: str,
        org_id: str,
        dpop_thumbprint: Optional[str] = None,
    ) -> bool:
        """Validate that session binding matches principal."""
        binding = self.get(session_id)
        if not binding:
            return False
        if binding.user_id != user_id or binding.org_id != org_id:
            return False
        if binding.dpop_thumbprint and dpop_thumbprint != binding.dpop_thumbprint:
            return False
        return True

    def delete(self, session_id: UUID) -> bool:
        """Delete a session binding."""
        with self._lock:
            if session_id in self._bindings:
                del self._bindings[session_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired bindings. Returns count of removed bindings."""
        now = datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            expired = [
                sid
                for sid, b in self._bindings.items()
                if b.expires_at <= now
            ]
            for sid in expired:
                del self._bindings[sid]
                removed += 1
        return removed


# Singleton instance
_session_binding_store: Optional[MemorySessionBindingStore] = None


def get_session_binding_store() -> MemorySessionBindingStore:
    """Get the session binding store singleton."""
    global _session_binding_store
    if _session_binding_store is None:
        ttl = int(os.environ.get("MCP_SESSION_TTL_SECONDS", "1800"))
        _session_binding_store = MemorySessionBindingStore(default_ttl_seconds=ttl)
    return _session_binding_store


def reset_session_binding_store() -> None:
    """Reset the singleton (for testing only)."""
    global _session_binding_store
    _session_binding_store = None
