"""RBAC (Role-Based Access Control) permission enforcement.

This module implements the role permission matrix per implementation plan section 4.6:
| Role | repo_read | repo_write | memory_write | admin_policy | audit_read | model_select |
|------|-----------|------------|--------------|--------------|-----------|--------------|
| enterprise_owner | yes | yes | yes | yes | yes | yes |
| org_owner | yes | yes | yes | yes (org) | yes | yes |
| admin | yes | yes | yes | limited | no | yes |
| maintainer | yes | yes | yes | no | no | no |
| reviewer | yes | no | no | no | no | no |
| user | yes | no | personal only | no | no | no |
| security_admin | no | no | no | no | yes | no |

Scopes are required in addition to role permissions; deny if either role or scope is missing.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ============================================================================
# Exceptions
# ============================================================================


class PermissionDeniedError(Exception):
    """Permission denied for the requested action."""

    pass


class ScopeMismatchError(PermissionDeniedError):
    """Resource scope doesn't match principal scope."""

    pass


# ============================================================================
# Enums
# ============================================================================


class Role(str, Enum):
    """User roles per implementation plan section 4.1."""

    ENTERPRISE_OWNER = "enterprise_owner"
    ORG_OWNER = "org_owner"
    ADMIN = "admin"
    MAINTAINER = "maintainer"
    REVIEWER = "reviewer"
    USER = "user"
    SECURITY_ADMIN = "security_admin"


class Permission(str, Enum):
    """Permissions that can be checked."""

    REPO_READ = "repo_read"
    REPO_WRITE = "repo_write"
    MEMORY_WRITE = "memory_write"
    ADMIN_POLICY = "admin_policy"
    ADMIN_POLICY_LIMITED = "admin_policy_limited"  # For admin role
    AUDIT_READ = "audit_read"
    MODEL_SELECT = "model_select"
    EMBEDDING_READ = "embedding_read"


class Scope(str, Enum):
    """OAuth scopes per implementation plan section 4.1."""

    REPOSITORY_READ = "repository:read"
    REPOSITORY_WRITE = "repository:write"
    EMBEDDING_READ = "embedding:read"
    MODEL_SELECT = "model:select"
    AUDIT_READ = "audit:read"
    ADMIN_POLICY = "admin:policy"
    MEMORY_WRITE = "memory:write"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Principal:
    """Represents an authenticated principal (user/service).

    Per implementation plan section 4.1:
    principal_id, user_id, org_id, enterprise_id, project_ids[],
    team_ids[], session_id, roles[], scopes[], geo_scope
    """

    user_id: str
    org_id: str
    enterprise_id: str
    roles: list[Role | str]
    scopes: list[Scope | str]
    principal_id: str | None = None
    session_id: str | None = None
    team_ids: list[str] = field(default_factory=list)
    project_ids: list[str] = field(default_factory=list)
    geo_scope: str | None = None

    def __post_init__(self):
        """Normalize roles and scopes to enums."""
        self.roles = [self._normalize_role(r) for r in self.roles]
        self.scopes = [self._normalize_scope(s) for s in self.scopes]

    def _normalize_role(self, role: Role | str) -> Role:
        """Convert string role to Role enum."""
        if isinstance(role, Role):
            return role
        try:
            return Role(role.lower())
        except ValueError:
            # Return as-is for unknown roles (will fail permission check)
            return role

    def _normalize_scope(self, scope: Scope | str) -> Scope:
        """Convert string scope to Scope enum."""
        if isinstance(scope, Scope):
            return scope
        try:
            return Scope(scope.lower())
        except ValueError:
            return scope

    def has_role(self, role: Role) -> bool:
        """Check if principal has a specific role."""
        return role in self.roles

    def has_scope(self, scope: Scope) -> bool:
        """Check if principal has a specific scope."""
        return scope in self.scopes

    def is_in_team(self, team_id: str) -> bool:
        """Check if principal is a member of a team."""
        return team_id in self.team_ids

    def is_in_org(self, org_id: str) -> bool:
        """Check if principal belongs to an org."""
        return self.org_id == org_id

    def is_in_enterprise(self, enterprise_id: str) -> bool:
        """Check if principal belongs to an enterprise."""
        return self.enterprise_id == enterprise_id


# ============================================================================
# Role-Permission Matrix
# ============================================================================


# Maps roles to their granted permissions
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ENTERPRISE_OWNER: {
        Permission.REPO_READ,
        Permission.REPO_WRITE,
        Permission.MEMORY_WRITE,
        Permission.ADMIN_POLICY,
        Permission.ADMIN_POLICY_LIMITED,
        Permission.AUDIT_READ,
        Permission.MODEL_SELECT,
        Permission.EMBEDDING_READ,
    },
    Role.ORG_OWNER: {
        Permission.REPO_READ,
        Permission.REPO_WRITE,
        Permission.MEMORY_WRITE,
        Permission.ADMIN_POLICY,  # Org-scoped only
        Permission.ADMIN_POLICY_LIMITED,
        Permission.AUDIT_READ,
        Permission.MODEL_SELECT,
        Permission.EMBEDDING_READ,
    },
    Role.ADMIN: {
        Permission.REPO_READ,
        Permission.REPO_WRITE,
        Permission.MEMORY_WRITE,
        Permission.ADMIN_POLICY_LIMITED,  # Limited only
        Permission.MODEL_SELECT,
        Permission.EMBEDDING_READ,
    },
    Role.MAINTAINER: {
        Permission.REPO_READ,
        Permission.REPO_WRITE,
        Permission.MEMORY_WRITE,
        Permission.EMBEDDING_READ,
    },
    Role.REVIEWER: {
        Permission.REPO_READ,
        Permission.EMBEDDING_READ,
    },
    Role.USER: {
        Permission.REPO_READ,
        Permission.MEMORY_WRITE,  # Personal only
        Permission.EMBEDDING_READ,
    },
    Role.SECURITY_ADMIN: {
        Permission.AUDIT_READ,
    },
}

# Maps permissions to required scopes
PERMISSION_SCOPES: dict[Permission, Scope] = {
    Permission.REPO_READ: Scope.REPOSITORY_READ,
    Permission.REPO_WRITE: Scope.REPOSITORY_WRITE,
    Permission.MEMORY_WRITE: Scope.MEMORY_WRITE,
    Permission.ADMIN_POLICY: Scope.ADMIN_POLICY,
    Permission.ADMIN_POLICY_LIMITED: Scope.ADMIN_POLICY,  # Same scope, different permission level
    Permission.AUDIT_READ: Scope.AUDIT_READ,
    Permission.MODEL_SELECT: Scope.MODEL_SELECT,
    Permission.EMBEDDING_READ: Scope.EMBEDDING_READ,
}


# ============================================================================
# RBAC Enforcer
# ============================================================================


class RBACEnforcer:
    """Enforces role-based access control.

    Features:
    - Role-permission matrix evaluation
    - Scope requirement enforcement
    - Resource-level scope checking (org, team, user)
    - Multiple role support with permission union
    """

    def check_permission(
        self,
        principal: Principal,
        permission: Permission,
        resource_org_id: str | None = None,
        resource_enterprise_id: str | None = None,
        resource_team_id: str | None = None,
        resource_user_id: str | None = None,
        resource_project_id: str | None = None,
    ) -> bool:
        """Check if a principal has a permission.

        Args:
            principal: The principal to check
            permission: The permission to check
            resource_org_id: Optional org ID of the resource
            resource_enterprise_id: Optional enterprise ID of the resource
            resource_team_id: Optional team ID of the resource
            resource_user_id: Optional user ID of the resource owner
            resource_project_id: Optional project ID of the resource

        Returns:
            True if the principal has the permission, False otherwise
        """
        if principal is None:
            raise ValueError("Principal cannot be None")

        # Check if any role grants this permission
        if not self._has_role_permission(principal, permission):
            return False

        # Check if the required scope is present
        if not self._has_required_scope(principal, permission):
            return False

        # Check resource-level scope
        if not self._check_resource_scope(
            principal,
            permission,
            resource_org_id,
            resource_enterprise_id,
            resource_team_id,
            resource_user_id,
            resource_project_id,
        ):
            return False

        return True

    def enforce(
        self,
        principal: Principal,
        permission: Permission,
        resource_org_id: str | None = None,
        resource_enterprise_id: str | None = None,
        resource_team_id: str | None = None,
        resource_user_id: str | None = None,
        resource_project_id: str | None = None,
    ) -> None:
        """Enforce that a principal has a permission.

        Args:
            Same as check_permission

        Raises:
            PermissionDeniedError: If permission is denied
            ScopeMismatchError: If resource scope doesn't match
        """
        if principal is None:
            raise ValueError("Principal cannot be None")

        # Check if any role grants this permission
        if not self._has_role_permission(principal, permission):
            raise PermissionDeniedError(
                f"Permission {permission.name} denied: no role grants this permission"
            )

        # Check if the required scope is present
        if not self._has_required_scope(principal, permission):
            raise PermissionDeniedError(
                f"Permission {permission.name} denied: missing required scope"
            )

        # Check resource-level scope
        if not self._check_resource_scope(
            principal,
            permission,
            resource_org_id,
            resource_enterprise_id,
            resource_team_id,
            resource_user_id,
            resource_project_id,
        ):
            raise ScopeMismatchError(
                f"Permission {permission.name} denied: resource scope mismatch"
            )

    def get_effective_permissions(self, principal: Principal) -> set[Permission]:
        """Get all effective permissions for a principal.

        Args:
            principal: The principal to check

        Returns:
            Set of permissions the principal has
        """
        permissions = set()

        for permission in Permission:
            if self.check_permission(principal, permission):
                permissions.add(permission)

        return permissions

    def _has_role_permission(
        self, principal: Principal, permission: Permission
    ) -> bool:
        """Check if any of the principal's roles grant the permission."""
        for role in principal.roles:
            if not isinstance(role, Role):
                continue
            role_perms = ROLE_PERMISSIONS.get(role, set())
            if permission in role_perms:
                return True
        return False

    def _has_required_scope(
        self, principal: Principal, permission: Permission
    ) -> bool:
        """Check if the principal has the required scope for the permission."""
        required_scope = PERMISSION_SCOPES.get(permission)
        if required_scope is None:
            return True  # No scope requirement

        return principal.has_scope(required_scope)

    def _check_resource_scope(
        self,
        principal: Principal,
        permission: Permission,
        resource_org_id: str | None,
        resource_enterprise_id: str | None,
        resource_team_id: str | None,
        resource_user_id: str | None,
        resource_project_id: str | None,
    ) -> bool:
        """Check resource-level scope restrictions."""
        # Enterprise owners can access anything in their enterprise
        if principal.has_role(Role.ENTERPRISE_OWNER):
            if resource_enterprise_id and not principal.is_in_enterprise(
                resource_enterprise_id
            ):
                return False
            return True

        # Org owners are scoped to their org for admin_policy
        if permission == Permission.ADMIN_POLICY:
            if principal.has_role(Role.ORG_OWNER):
                if resource_org_id and not principal.is_in_org(resource_org_id):
                    return False
                return True

        # Regular users can only write their own memories
        if permission == Permission.MEMORY_WRITE:
            if principal.has_role(Role.USER) and not any(
                role in [Role.ENTERPRISE_OWNER, Role.ORG_OWNER, Role.ADMIN, Role.MAINTAINER]
                for role in principal.roles
                if isinstance(role, Role)
            ):
                if resource_user_id and resource_user_id != principal.user_id:
                    return False

        # Check org scope for non-enterprise-owner users
        if resource_org_id and not principal.has_role(Role.ENTERPRISE_OWNER):
            if not principal.is_in_org(resource_org_id):
                return False

        # Check team scope
        if resource_team_id:
            if not principal.is_in_team(resource_team_id):
                # Enterprise and org owners can access all teams in their scope
                if not (
                    principal.has_role(Role.ENTERPRISE_OWNER)
                    or principal.has_role(Role.ORG_OWNER)
                ):
                    return False

        return True
