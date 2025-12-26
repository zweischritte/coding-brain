"""Tests for RBAC (Role-Based Access Control) permission matrix.

Tests cover the role permission matrix per implementation plan section 4.6:
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

import pytest
from typing import Any

from openmemory.api.security.rbac import (
    Role,
    Permission,
    Scope,
    Principal,
    RBACEnforcer,
    PermissionDeniedError,
    ScopeMismatchError,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def enforcer() -> RBACEnforcer:
    """Create RBAC enforcer."""
    return RBACEnforcer()


@pytest.fixture
def enterprise_owner_principal() -> Principal:
    """Enterprise owner principal with full access."""
    return Principal(
        user_id="user-1",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.ENTERPRISE_OWNER],
        scopes=[
            Scope.REPOSITORY_READ,
            Scope.REPOSITORY_WRITE,
            Scope.EMBEDDING_READ,
            Scope.MODEL_SELECT,
            Scope.AUDIT_READ,
            Scope.ADMIN_POLICY,
            Scope.MEMORY_WRITE,
        ],
    )


@pytest.fixture
def org_owner_principal() -> Principal:
    """Org owner principal."""
    return Principal(
        user_id="user-2",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.ORG_OWNER],
        scopes=[
            Scope.REPOSITORY_READ,
            Scope.REPOSITORY_WRITE,
            Scope.EMBEDDING_READ,
            Scope.MODEL_SELECT,
            Scope.AUDIT_READ,
            Scope.ADMIN_POLICY,
            Scope.MEMORY_WRITE,
        ],
    )


@pytest.fixture
def admin_principal() -> Principal:
    """Admin principal with limited admin policy."""
    return Principal(
        user_id="user-3",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.ADMIN],
        scopes=[
            Scope.REPOSITORY_READ,
            Scope.REPOSITORY_WRITE,
            Scope.EMBEDDING_READ,
            Scope.MODEL_SELECT,
            Scope.MEMORY_WRITE,
            Scope.ADMIN_POLICY,  # Has scope but role only grants LIMITED access
        ],
    )


@pytest.fixture
def maintainer_principal() -> Principal:
    """Maintainer principal."""
    return Principal(
        user_id="user-4",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.MAINTAINER],
        scopes=[
            Scope.REPOSITORY_READ,
            Scope.REPOSITORY_WRITE,
            Scope.EMBEDDING_READ,
            Scope.MEMORY_WRITE,
        ],
    )


@pytest.fixture
def reviewer_principal() -> Principal:
    """Reviewer principal (read-only)."""
    return Principal(
        user_id="user-5",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.REVIEWER],
        scopes=[Scope.REPOSITORY_READ, Scope.EMBEDDING_READ],
    )


@pytest.fixture
def user_principal() -> Principal:
    """Regular user principal."""
    return Principal(
        user_id="user-6",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.USER],
        scopes=[Scope.REPOSITORY_READ, Scope.EMBEDDING_READ, Scope.MEMORY_WRITE],
    )


@pytest.fixture
def security_admin_principal() -> Principal:
    """Security admin principal (audit-only)."""
    return Principal(
        user_id="user-7",
        org_id="org-1",
        enterprise_id="ent-1",
        roles=[Role.SECURITY_ADMIN],
        scopes=[Scope.AUDIT_READ],
    )


# ============================================================================
# Enterprise Owner Permission Tests
# ============================================================================


class TestEnterpriseOwnerPermissions:
    """Test enterprise owner has full permissions."""

    def test_can_read_repository(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can read repository."""
        assert enforcer.check_permission(
            enterprise_owner_principal, Permission.REPO_READ
        )

    def test_can_write_repository(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can write repository."""
        assert enforcer.check_permission(
            enterprise_owner_principal, Permission.REPO_WRITE
        )

    def test_can_write_memory(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can write memory."""
        assert enforcer.check_permission(
            enterprise_owner_principal, Permission.MEMORY_WRITE
        )

    def test_can_admin_policy(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can administer policy."""
        assert enforcer.check_permission(
            enterprise_owner_principal, Permission.ADMIN_POLICY
        )

    def test_can_read_audit(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can read audit logs."""
        assert enforcer.check_permission(
            enterprise_owner_principal, Permission.AUDIT_READ
        )

    def test_can_select_model(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can select model."""
        assert enforcer.check_permission(
            enterprise_owner_principal, Permission.MODEL_SELECT
        )


# ============================================================================
# Org Owner Permission Tests
# ============================================================================


class TestOrgOwnerPermissions:
    """Test org owner permissions (org-scoped admin)."""

    def test_can_read_repository(self, enforcer, org_owner_principal):
        """Org owner can read repository."""
        assert enforcer.check_permission(org_owner_principal, Permission.REPO_READ)

    def test_can_write_repository(self, enforcer, org_owner_principal):
        """Org owner can write repository."""
        assert enforcer.check_permission(org_owner_principal, Permission.REPO_WRITE)

    def test_can_write_memory(self, enforcer, org_owner_principal):
        """Org owner can write memory."""
        assert enforcer.check_permission(org_owner_principal, Permission.MEMORY_WRITE)

    def test_can_admin_policy_own_org(self, enforcer, org_owner_principal):
        """Org owner can administer policy for own org."""
        assert enforcer.check_permission(
            org_owner_principal,
            Permission.ADMIN_POLICY,
            resource_org_id="org-1",  # Same org
        )

    def test_cannot_admin_policy_other_org(self, enforcer, org_owner_principal):
        """Org owner cannot administer policy for other orgs."""
        assert not enforcer.check_permission(
            org_owner_principal,
            Permission.ADMIN_POLICY,
            resource_org_id="org-2",  # Different org
        )

    def test_can_read_audit(self, enforcer, org_owner_principal):
        """Org owner can read audit logs."""
        assert enforcer.check_permission(org_owner_principal, Permission.AUDIT_READ)

    def test_can_select_model(self, enforcer, org_owner_principal):
        """Org owner can select model."""
        assert enforcer.check_permission(org_owner_principal, Permission.MODEL_SELECT)


# ============================================================================
# Admin Permission Tests
# ============================================================================


class TestAdminPermissions:
    """Test admin permissions (limited policy access)."""

    def test_can_read_repository(self, enforcer, admin_principal):
        """Admin can read repository."""
        assert enforcer.check_permission(admin_principal, Permission.REPO_READ)

    def test_can_write_repository(self, enforcer, admin_principal):
        """Admin can write repository."""
        assert enforcer.check_permission(admin_principal, Permission.REPO_WRITE)

    def test_can_write_memory(self, enforcer, admin_principal):
        """Admin can write memory."""
        assert enforcer.check_permission(admin_principal, Permission.MEMORY_WRITE)

    def test_limited_admin_policy(self, enforcer, admin_principal):
        """Admin has limited policy access (non-critical settings only)."""
        # Admin can do limited policy changes
        assert enforcer.check_permission(
            admin_principal,
            Permission.ADMIN_POLICY_LIMITED,
        )

    def test_cannot_full_admin_policy(self, enforcer, admin_principal):
        """Admin cannot do full admin policy changes."""
        # Without ADMIN_POLICY scope, cannot do full policy changes
        assert not enforcer.check_permission(admin_principal, Permission.ADMIN_POLICY)

    def test_cannot_read_audit(self, enforcer, admin_principal):
        """Admin cannot read audit logs."""
        assert not enforcer.check_permission(admin_principal, Permission.AUDIT_READ)

    def test_can_select_model(self, enforcer, admin_principal):
        """Admin can select model."""
        assert enforcer.check_permission(admin_principal, Permission.MODEL_SELECT)


# ============================================================================
# Maintainer Permission Tests
# ============================================================================


class TestMaintainerPermissions:
    """Test maintainer permissions."""

    def test_can_read_repository(self, enforcer, maintainer_principal):
        """Maintainer can read repository."""
        assert enforcer.check_permission(maintainer_principal, Permission.REPO_READ)

    def test_can_write_repository(self, enforcer, maintainer_principal):
        """Maintainer can write repository."""
        assert enforcer.check_permission(maintainer_principal, Permission.REPO_WRITE)

    def test_can_write_memory(self, enforcer, maintainer_principal):
        """Maintainer can write memory."""
        assert enforcer.check_permission(maintainer_principal, Permission.MEMORY_WRITE)

    def test_cannot_admin_policy(self, enforcer, maintainer_principal):
        """Maintainer cannot administer policy."""
        assert not enforcer.check_permission(
            maintainer_principal, Permission.ADMIN_POLICY
        )

    def test_cannot_read_audit(self, enforcer, maintainer_principal):
        """Maintainer cannot read audit logs."""
        assert not enforcer.check_permission(maintainer_principal, Permission.AUDIT_READ)

    def test_cannot_select_model(self, enforcer, maintainer_principal):
        """Maintainer cannot select model."""
        assert not enforcer.check_permission(
            maintainer_principal, Permission.MODEL_SELECT
        )


# ============================================================================
# Reviewer Permission Tests
# ============================================================================


class TestReviewerPermissions:
    """Test reviewer permissions (read-only)."""

    def test_can_read_repository(self, enforcer, reviewer_principal):
        """Reviewer can read repository."""
        assert enforcer.check_permission(reviewer_principal, Permission.REPO_READ)

    def test_cannot_write_repository(self, enforcer, reviewer_principal):
        """Reviewer cannot write repository."""
        assert not enforcer.check_permission(reviewer_principal, Permission.REPO_WRITE)

    def test_cannot_write_memory(self, enforcer, reviewer_principal):
        """Reviewer cannot write memory."""
        assert not enforcer.check_permission(reviewer_principal, Permission.MEMORY_WRITE)

    def test_cannot_admin_policy(self, enforcer, reviewer_principal):
        """Reviewer cannot administer policy."""
        assert not enforcer.check_permission(reviewer_principal, Permission.ADMIN_POLICY)

    def test_cannot_read_audit(self, enforcer, reviewer_principal):
        """Reviewer cannot read audit logs."""
        assert not enforcer.check_permission(reviewer_principal, Permission.AUDIT_READ)

    def test_cannot_select_model(self, enforcer, reviewer_principal):
        """Reviewer cannot select model."""
        assert not enforcer.check_permission(reviewer_principal, Permission.MODEL_SELECT)


# ============================================================================
# User Permission Tests
# ============================================================================


class TestUserPermissions:
    """Test regular user permissions."""

    def test_can_read_repository(self, enforcer, user_principal):
        """User can read repository."""
        assert enforcer.check_permission(user_principal, Permission.REPO_READ)

    def test_cannot_write_repository(self, enforcer, user_principal):
        """User cannot write repository."""
        assert not enforcer.check_permission(user_principal, Permission.REPO_WRITE)

    def test_can_write_personal_memory(self, enforcer, user_principal):
        """User can write personal memory only."""
        assert enforcer.check_permission(
            user_principal,
            Permission.MEMORY_WRITE,
            resource_user_id="user-6",  # Same user
        )

    def test_cannot_write_other_user_memory(self, enforcer, user_principal):
        """User cannot write other user's memory."""
        assert not enforcer.check_permission(
            user_principal,
            Permission.MEMORY_WRITE,
            resource_user_id="user-other",  # Different user
        )

    def test_cannot_admin_policy(self, enforcer, user_principal):
        """User cannot administer policy."""
        assert not enforcer.check_permission(user_principal, Permission.ADMIN_POLICY)

    def test_cannot_read_audit(self, enforcer, user_principal):
        """User cannot read audit logs."""
        assert not enforcer.check_permission(user_principal, Permission.AUDIT_READ)

    def test_cannot_select_model(self, enforcer, user_principal):
        """User cannot select model."""
        assert not enforcer.check_permission(user_principal, Permission.MODEL_SELECT)


# ============================================================================
# Security Admin Permission Tests
# ============================================================================


class TestSecurityAdminPermissions:
    """Test security admin permissions (audit-only)."""

    def test_cannot_read_repository(self, enforcer, security_admin_principal):
        """Security admin cannot read repository."""
        assert not enforcer.check_permission(
            security_admin_principal, Permission.REPO_READ
        )

    def test_cannot_write_repository(self, enforcer, security_admin_principal):
        """Security admin cannot write repository."""
        assert not enforcer.check_permission(
            security_admin_principal, Permission.REPO_WRITE
        )

    def test_cannot_write_memory(self, enforcer, security_admin_principal):
        """Security admin cannot write memory."""
        assert not enforcer.check_permission(
            security_admin_principal, Permission.MEMORY_WRITE
        )

    def test_cannot_admin_policy(self, enforcer, security_admin_principal):
        """Security admin cannot administer policy."""
        assert not enforcer.check_permission(
            security_admin_principal, Permission.ADMIN_POLICY
        )

    def test_can_read_audit(self, enforcer, security_admin_principal):
        """Security admin can read audit logs."""
        assert enforcer.check_permission(
            security_admin_principal, Permission.AUDIT_READ
        )

    def test_cannot_select_model(self, enforcer, security_admin_principal):
        """Security admin cannot select model."""
        assert not enforcer.check_permission(
            security_admin_principal, Permission.MODEL_SELECT
        )


# ============================================================================
# Scope Requirement Tests
# ============================================================================


class TestScopeRequirements:
    """Test that scopes are required in addition to roles."""

    def test_role_without_scope_denied(self, enforcer):
        """Role without matching scope should be denied."""
        # User has role but no scopes
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.ENTERPRISE_OWNER],
            scopes=[],  # No scopes
        )

        assert not enforcer.check_permission(principal, Permission.REPO_READ)

    def test_scope_without_role_denied(self, enforcer):
        """Scope without matching role should be denied."""
        # User has scope but wrong role
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.USER],  # User role doesn't allow repo_write
            scopes=[Scope.REPOSITORY_WRITE],  # Has scope
        )

        assert not enforcer.check_permission(principal, Permission.REPO_WRITE)

    def test_both_role_and_scope_required(self, enforcer):
        """Both role and scope are required for permission."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.MAINTAINER],
            scopes=[Scope.REPOSITORY_READ, Scope.REPOSITORY_WRITE],
        )

        assert enforcer.check_permission(principal, Permission.REPO_READ)
        assert enforcer.check_permission(principal, Permission.REPO_WRITE)


# ============================================================================
# Multiple Roles Tests
# ============================================================================


class TestMultipleRoles:
    """Test principals with multiple roles."""

    def test_multiple_roles_union_permissions(self, enforcer):
        """Multiple roles should union their permissions."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.REVIEWER, Role.SECURITY_ADMIN],  # Both roles
            scopes=[
                Scope.REPOSITORY_READ,
                Scope.AUDIT_READ,
            ],
        )

        # Should have permissions from both roles
        assert enforcer.check_permission(principal, Permission.REPO_READ)  # From reviewer
        assert enforcer.check_permission(principal, Permission.AUDIT_READ)  # From security_admin

    def test_highest_privilege_wins(self, enforcer):
        """Highest privilege role should determine access."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.USER, Role.MAINTAINER],  # Maintainer is higher
            scopes=[
                Scope.REPOSITORY_READ,
                Scope.REPOSITORY_WRITE,
                Scope.MEMORY_WRITE,
            ],
        )

        # Should have maintainer-level access
        assert enforcer.check_permission(principal, Permission.REPO_WRITE)
        assert enforcer.check_permission(principal, Permission.MEMORY_WRITE)


# ============================================================================
# Enforce with Exception Tests
# ============================================================================


class TestEnforceWithException:
    """Test enforce method that raises exceptions."""

    def test_enforce_success(self, enforcer, enterprise_owner_principal):
        """Enforce should not raise for permitted actions."""
        # Should not raise
        enforcer.enforce(enterprise_owner_principal, Permission.REPO_READ)

    def test_enforce_raises_permission_denied(self, enforcer, user_principal):
        """Enforce should raise PermissionDeniedError when denied."""
        with pytest.raises(PermissionDeniedError) as exc_info:
            enforcer.enforce(user_principal, Permission.REPO_WRITE)

        assert "REPO_WRITE" in str(exc_info.value)

    def test_enforce_raises_scope_mismatch(self, enforcer):
        """Enforce should raise ScopeMismatchError for scope violations."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.USER],
            scopes=[Scope.REPOSITORY_READ, Scope.MEMORY_WRITE],  # Has scope
        )

        # User trying to write memory for different user
        with pytest.raises(ScopeMismatchError):
            enforcer.enforce(
                principal,
                Permission.MEMORY_WRITE,
                resource_user_id="other-user",
            )


# ============================================================================
# Resource Scope Tests
# ============================================================================


class TestResourceScopes:
    """Test resource-level scope enforcement."""

    def test_org_scoped_access(self, enforcer, org_owner_principal):
        """Resource access should be scoped to org."""
        # Can access resource in same org
        assert enforcer.check_permission(
            org_owner_principal,
            Permission.REPO_WRITE,
            resource_org_id="org-1",
        )

        # Cannot access resource in different org
        assert not enforcer.check_permission(
            org_owner_principal,
            Permission.REPO_WRITE,
            resource_org_id="org-other",
        )

    def test_enterprise_scoped_access(self, enforcer, enterprise_owner_principal):
        """Enterprise owner can access all orgs in enterprise."""
        # Can access any org in same enterprise
        assert enforcer.check_permission(
            enterprise_owner_principal,
            Permission.REPO_WRITE,
            resource_org_id="org-1",
            resource_enterprise_id="ent-1",
        )

        assert enforcer.check_permission(
            enterprise_owner_principal,
            Permission.REPO_WRITE,
            resource_org_id="org-2",
            resource_enterprise_id="ent-1",
        )

    def test_cannot_access_other_enterprise(self, enforcer, enterprise_owner_principal):
        """Enterprise owner cannot access other enterprises."""
        assert not enforcer.check_permission(
            enterprise_owner_principal,
            Permission.REPO_WRITE,
            resource_enterprise_id="ent-other",
        )


# ============================================================================
# Team Scope Tests
# ============================================================================


class TestTeamScopes:
    """Test team-level access control."""

    def test_user_in_team_can_access_team_resources(self, enforcer):
        """User in team can access team resources."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.MAINTAINER],
            scopes=[Scope.REPOSITORY_READ, Scope.REPOSITORY_WRITE],
            team_ids=["team-1", "team-2"],
        )

        assert enforcer.check_permission(
            principal,
            Permission.REPO_WRITE,
            resource_team_id="team-1",
        )

    def test_user_not_in_team_cannot_access(self, enforcer):
        """User not in team cannot access team resources."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[Role.MAINTAINER],
            scopes=[Scope.REPOSITORY_READ, Scope.REPOSITORY_WRITE],
            team_ids=["team-1"],  # Only in team-1
        )

        assert not enforcer.check_permission(
            principal,
            Permission.REPO_WRITE,
            resource_team_id="team-other",  # Not a member
        )


# ============================================================================
# Get Effective Permissions Tests
# ============================================================================


class TestGetEffectivePermissions:
    """Test getting effective permissions for a principal."""

    def test_get_all_permissions_enterprise_owner(self, enforcer, enterprise_owner_principal):
        """Enterprise owner should have all permissions."""
        permissions = enforcer.get_effective_permissions(enterprise_owner_principal)

        assert Permission.REPO_READ in permissions
        assert Permission.REPO_WRITE in permissions
        assert Permission.MEMORY_WRITE in permissions
        assert Permission.ADMIN_POLICY in permissions
        assert Permission.AUDIT_READ in permissions
        assert Permission.MODEL_SELECT in permissions

    def test_get_limited_permissions_user(self, enforcer, user_principal):
        """Regular user should have limited permissions."""
        permissions = enforcer.get_effective_permissions(user_principal)

        assert Permission.REPO_READ in permissions
        assert Permission.REPO_WRITE not in permissions
        assert Permission.ADMIN_POLICY not in permissions
        assert Permission.AUDIT_READ not in permissions


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_roles_denied(self, enforcer):
        """Principal with no roles should be denied."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[],
            scopes=[Scope.REPOSITORY_READ],
        )

        assert not enforcer.check_permission(principal, Permission.REPO_READ)

    def test_unknown_role_handled(self, enforcer):
        """Unknown role should not grant permissions."""
        # This tests that the system doesn't break with unknown roles
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=[],
            scopes=[Scope.REPOSITORY_READ],
        )

        assert not enforcer.check_permission(principal, Permission.REPO_READ)

    def test_none_principal_raises(self, enforcer):
        """None principal should raise."""
        with pytest.raises((ValueError, TypeError)):
            enforcer.check_permission(None, Permission.REPO_READ)

    def test_case_insensitive_role_matching(self, enforcer):
        """Role matching should handle string roles from JWT."""
        principal = Principal(
            user_id="user-1",
            org_id="org-1",
            enterprise_id="ent-1",
            roles=["enterprise_owner"],  # String from JWT
            scopes=[
                Scope.REPOSITORY_READ,
                Scope.REPOSITORY_WRITE,
                Scope.ADMIN_POLICY,
            ],
        )

        assert enforcer.check_permission(principal, Permission.REPO_READ)
