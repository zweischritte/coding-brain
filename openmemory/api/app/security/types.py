"""
Core security types for authentication and authorization.

These types define the principal model, token claims, scopes, and error types
used throughout the security module.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Set
from uuid import UUID

ALLOWED_ACCESS_ENTITY_PREFIXES = {"user", "team", "project", "org"}

class AuthenticationError(Exception):
    """Raised when authentication fails (invalid or missing credentials)."""

    def __init__(self, message: str = "Authentication required", code: str = "UNAUTHENTICATED"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AuthorizationError(Exception):
    """Raised when authorization fails (insufficient permissions)."""

    def __init__(self, message: str = "Insufficient permissions", code: str = "FORBIDDEN"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class Scope(str, Enum):
    """
    OAuth 2.0 scopes for RBAC enforcement.

    Scopes follow the pattern: resource:action
    """
    # Memory operations
    MEMORIES_READ = "memories:read"
    MEMORIES_WRITE = "memories:write"
    MEMORIES_DELETE = "memories:delete"

    # App management
    APPS_READ = "apps:read"
    APPS_WRITE = "apps:write"
    APPS_DELETE = "apps:delete"

    # Graph operations
    GRAPH_READ = "graph:read"
    GRAPH_WRITE = "graph:write"

    # Entity operations
    ENTITIES_READ = "entities:read"
    ENTITIES_WRITE = "entities:write"

    # Admin operations
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"

    # Stats and analytics
    STATS_READ = "stats:read"

    # Backup/export operations
    BACKUP_READ = "backup:read"
    BACKUP_WRITE = "backup:write"

    # Feedback operations (Phase 5)
    FEEDBACK_READ = "feedback:read"
    FEEDBACK_WRITE = "feedback:write"

    # Experiment operations (Phase 5)
    EXPERIMENTS_READ = "experiments:read"
    EXPERIMENTS_WRITE = "experiments:write"

    # Search operations (Phase 5)
    SEARCH_READ = "search:read"

    # GDPR operations (Phase 4.5)
    GDPR_READ = "gdpr:read"      # SAR export
    GDPR_DELETE = "gdpr:delete"  # User deletion (right to erasure)


@dataclass
class TokenClaims:
    """
    Validated JWT token claims.

    These are the claims extracted from a validated JWT token.
    """
    # Standard JWT claims
    sub: str  # Subject (user identifier)
    iss: str  # Issuer
    aud: str  # Audience
    exp: datetime  # Expiration time
    iat: datetime  # Issued at time
    jti: str  # JWT ID (for replay prevention)

    # Custom claims
    org_id: str  # Organization/tenant ID
    scopes: Set[str] = field(default_factory=set)  # Granted scopes
    grants: Set[str] = field(default_factory=set)  # Access entity grants

    # Optional claims
    email: Optional[str] = None
    name: Optional[str] = None

    def has_scope(self, scope: Scope | str) -> bool:
        """Check if the token has a specific scope."""
        scope_str = scope.value if isinstance(scope, Scope) else scope
        return scope_str in self.scopes

    def has_any_scope(self, scopes: Set[Scope | str]) -> bool:
        """Check if the token has any of the specified scopes."""
        return any(self.has_scope(s) for s in scopes)

    def has_all_scopes(self, scopes: Set[Scope | str]) -> bool:
        """Check if the token has all of the specified scopes."""
        return all(self.has_scope(s) for s in scopes)

    def has_grant(self, grant: str) -> bool:
        """Check if the token has a specific grant."""
        return grant in self.grants

    def has_any_grant(self, grants: Set[str]) -> bool:
        """Check if the token has any of the specified grants."""
        return any(self.has_grant(g) for g in grants)


@dataclass
class Principal:
    """
    The authenticated principal (user) for a request.

    This is the primary identity object passed to route handlers
    after successful authentication.
    """
    # Core identity
    user_id: str  # External user identifier (from JWT sub claim)
    org_id: str  # Organization/tenant ID

    # Token claims (for scope checking)
    claims: TokenClaims

    # Optional database user ID (resolved after auth)
    db_user_id: Optional[UUID] = None

    # DPoP binding (if present)
    dpop_thumbprint: Optional[str] = None

    def has_scope(self, scope: Scope | str) -> bool:
        """Check if the principal has a specific scope."""
        return self.claims.has_scope(scope)

    def has_any_scope(self, scopes: Set[Scope | str]) -> bool:
        """Check if the principal has any of the specified scopes."""
        return self.claims.has_any_scope(scopes)

    def has_all_scopes(self, scopes: Set[Scope | str]) -> bool:
        """Check if the principal has all of the specified scopes."""
        return self.claims.has_all_scopes(scopes)

    def require_scope(self, scope: Scope | str) -> None:
        """Raise AuthorizationError if scope is missing."""
        if not self.has_scope(scope):
            scope_str = scope.value if isinstance(scope, Scope) else scope
            raise AuthorizationError(
                message=f"Required scope '{scope_str}' not granted",
                code="INSUFFICIENT_SCOPE"
            )

    def require_all_scopes(self, scopes: Set[Scope | str]) -> None:
        """Raise AuthorizationError if any scope is missing."""
        for scope in scopes:
            self.require_scope(scope)

    def has_grant(self, grant: str) -> bool:
        """Check if the principal has a specific grant."""
        return self.claims.has_grant(grant)

    def has_any_grant(self, grants: Set[str]) -> bool:
        """Check if the principal has any of the specified grants."""
        return self.claims.has_any_grant(grants)

    def get_allowed_access_entities(self) -> Set[str]:
        """Get all access_entity values this principal can access.

        Always includes user:<user_id> plus any explicit grants.
        """
        allowed = set(self.claims.grants)
        # Always include user grant based on user_id
        allowed.add(f"user:{self.user_id}")
        return allowed

    def can_access(self, access_entity: str) -> bool:
        """Check if the principal can access a memory with the given access_entity.

        This implements hierarchical grant expansion:
        - org:X grant allows access to org:X, project:X/*, team:X/*
        - project:X grant allows access to project:X, team:X/*
        - team:X grant allows access to team:X only
        - user:X grant allows access to user:X only
        """
        if not access_entity or ":" not in access_entity:
            return False

        prefix, path = access_entity.split(":", 1)
        if prefix not in ALLOWED_ACCESS_ENTITY_PREFIXES:
            return False

        # Direct match
        if access_entity in self.claims.grants:
            return True

        # Always have access to own user scope
        if access_entity == f"user:{self.user_id}":
            return True

        # Hierarchical expansion
        for grant in self.claims.grants:
            if ":" not in grant:
                continue

            grant_prefix, grant_path = grant.split(":", 1)

            # org grant expands to project/team/client under that org
            if grant_prefix == "org":
                # org:cloudfactory allows project:cloudfactory/*
                if prefix in ("project", "team") and path.startswith(f"{grant_path}/"):
                    return True
                # org:cloudfactory also allows org:cloudfactory itself
                if prefix == "org" and path == grant_path:
                    return True

            # project grant expands to teams under that project
            elif grant_prefix == "project":
                # project:cloudfactory/acme/billing allows team:cloudfactory/acme/billing/*
                if prefix == "team" and path.startswith(f"{grant_path}/"):
                    return True
                # project:X also allows project:X itself
                if prefix == "project" and path == grant_path:
                    return True

            # team grant - exact match only (already checked above)
            # user grant - exact match only (already checked above)

        return False
