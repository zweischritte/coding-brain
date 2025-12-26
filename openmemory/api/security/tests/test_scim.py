"""Tests for SCIM 2.0 integration stubs.

Tests cover SCIM 2.0 per implementation plan section 4.8:
- User provisioning/deprovisioning
- Group management
- SCIM groups mapping to IdP roles and OAuth scopes
- Orphan detection (suspend unlinked users within 72 hours)
"""

import copy
import pytest
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

from openmemory.api.security.scim import (
    SCIMUser,
    SCIMGroup,
    SCIMService,
    SCIMProvisioningError,
    SCIMDeactivationError,
    SCIMGroupMappingError,
    UserStatus,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def scim_service() -> SCIMService:
    """Create SCIM service stub."""
    return SCIMService()


@pytest.fixture
def sample_user_data() -> dict[str, Any]:
    """Sample SCIM user data from IdP."""
    return {
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
        "id": "ext-user-123",
        "externalId": "ext-user-123",
        "userName": "john.doe@example.com",
        "name": {
            "givenName": "John",
            "familyName": "Doe",
            "formatted": "John Doe",
        },
        "emails": [
            {"value": "john.doe@example.com", "primary": True, "type": "work"}
        ],
        "active": True,
        "groups": [
            {"value": "group-dev-team", "display": "Development Team"},
            {"value": "group-maintainers", "display": "Maintainers"},
        ],
        "meta": {
            "resourceType": "User",
            "created": "2024-01-15T10:00:00Z",
            "lastModified": "2024-06-20T14:30:00Z",
        },
    }


@pytest.fixture
def sample_group_data() -> dict[str, Any]:
    """Sample SCIM group data from IdP."""
    return {
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
        "id": "group-dev-team",
        "displayName": "Development Team",
        "members": [
            {"value": "ext-user-123", "display": "john.doe@example.com"},
            {"value": "ext-user-456", "display": "jane.smith@example.com"},
        ],
        "meta": {
            "resourceType": "Group",
            "created": "2024-01-01T00:00:00Z",
            "lastModified": "2024-06-15T09:00:00Z",
        },
    }


# ============================================================================
# User Provisioning Tests
# ============================================================================


class TestUserProvisioning:
    """Test SCIM user provisioning."""

    def test_provision_new_user(self, scim_service, sample_user_data):
        """New user should be provisioned from SCIM data."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        assert user.external_id == "ext-user-123"
        assert user.email == "john.doe@example.com"
        assert user.display_name == "John Doe"
        assert user.org_id == "org-1"
        assert user.status == UserStatus.ACTIVE

    def test_provision_user_maps_groups(self, scim_service, sample_user_data):
        """User provisioning should map SCIM groups."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        assert "group-dev-team" in user.group_ids
        assert "group-maintainers" in user.group_ids

    def test_provision_inactive_user_creates_suspended(
        self, scim_service, sample_user_data
    ):
        """Inactive user in SCIM should be created as suspended."""
        sample_user_data["active"] = False

        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        assert user.status == UserStatus.SUSPENDED

    def test_provision_duplicate_external_id_updates(
        self, scim_service, sample_user_data
    ):
        """Provisioning with existing external_id should update user."""
        # First provision
        user1 = scim_service.provision_user(sample_user_data, org_id="org-1")

        # Update name and provision again (use deepcopy to avoid mutating fixture)
        updated_data = copy.deepcopy(sample_user_data)
        updated_data["name"]["givenName"] = "Johnny"
        updated_data["name"]["formatted"] = "Johnny Doe"  # SCIM IdP updates formatted too
        user2 = scim_service.provision_user(updated_data, org_id="org-1")

        assert user2.internal_id == user1.internal_id
        assert user2.display_name == "Johnny Doe"

    def test_provision_missing_required_field_raises(self, scim_service):
        """Missing required field should raise error."""
        invalid_data = {
            "id": "user-123",
            # Missing userName
        }

        with pytest.raises(SCIMProvisioningError) as exc_info:
            scim_service.provision_user(invalid_data, org_id="org-1")

        assert "userName" in str(exc_info.value)

    def test_provision_generates_internal_id(self, scim_service, sample_user_data):
        """Provisioning should generate internal ID."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        assert user.internal_id is not None
        assert len(user.internal_id) > 0


# ============================================================================
# User Deprovisioning Tests
# ============================================================================


class TestUserDeprovisioning:
    """Test SCIM user deprovisioning."""

    def test_deprovision_user_by_external_id(self, scim_service, sample_user_data):
        """User should be deprovisioned by external ID."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        result = scim_service.deprovision_user(
            external_id="ext-user-123", org_id="org-1"
        )

        assert result.status == UserStatus.DEPROVISIONED

    def test_deprovision_unknown_user_raises(self, scim_service):
        """Deprovisioning unknown user should raise error."""
        with pytest.raises(SCIMDeactivationError) as exc_info:
            scim_service.deprovision_user(external_id="unknown", org_id="org-1")

        assert "not found" in str(exc_info.value).lower()

    def test_deprovision_removes_group_memberships(
        self, scim_service, sample_user_data
    ):
        """Deprovisioning should remove group memberships."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")
        assert len(user.group_ids) > 0

        result = scim_service.deprovision_user(
            external_id="ext-user-123", org_id="org-1"
        )

        assert len(result.group_ids) == 0

    def test_deprovision_preserves_audit_trail(self, scim_service, sample_user_data):
        """Deprovisioning should preserve audit information."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        result = scim_service.deprovision_user(
            external_id="ext-user-123", org_id="org-1"
        )

        assert result.deprovisioned_at is not None
        assert result.internal_id == user.internal_id


# ============================================================================
# User Update Tests
# ============================================================================


class TestUserUpdate:
    """Test SCIM user updates."""

    def test_update_user_email(self, scim_service, sample_user_data):
        """User email should be updateable."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        updated_data = sample_user_data.copy()
        updated_data["emails"] = [
            {"value": "john.updated@example.com", "primary": True}
        ]

        updated = scim_service.update_user(
            external_id="ext-user-123",
            org_id="org-1",
            updates=updated_data,
        )

        assert updated.email == "john.updated@example.com"

    def test_update_user_groups(self, scim_service, sample_user_data):
        """User group memberships should be updateable."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        updated_data = sample_user_data.copy()
        updated_data["groups"] = [
            {"value": "group-new-team", "display": "New Team"},
        ]

        updated = scim_service.update_user(
            external_id="ext-user-123",
            org_id="org-1",
            updates=updated_data,
        )

        assert "group-new-team" in updated.group_ids
        assert "group-dev-team" not in updated.group_ids

    def test_update_user_status(self, scim_service, sample_user_data):
        """User active status should be updateable."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")
        assert user.status == UserStatus.ACTIVE

        updated_data = sample_user_data.copy()
        updated_data["active"] = False

        updated = scim_service.update_user(
            external_id="ext-user-123",
            org_id="org-1",
            updates=updated_data,
        )

        assert updated.status == UserStatus.SUSPENDED


# ============================================================================
# Group Management Tests
# ============================================================================


class TestGroupManagement:
    """Test SCIM group management."""

    def test_create_group(self, scim_service, sample_group_data):
        """Group should be created from SCIM data."""
        group = scim_service.create_group(sample_group_data, org_id="org-1")

        assert group.external_id == "group-dev-team"
        assert group.display_name == "Development Team"
        assert group.org_id == "org-1"

    def test_group_includes_members(self, scim_service, sample_group_data):
        """Group should include member references."""
        group = scim_service.create_group(sample_group_data, org_id="org-1")

        assert "ext-user-123" in group.member_external_ids
        assert "ext-user-456" in group.member_external_ids

    def test_update_group_members(self, scim_service, sample_group_data):
        """Group members should be updateable."""
        group = scim_service.create_group(sample_group_data, org_id="org-1")

        updated_data = sample_group_data.copy()
        updated_data["members"] = [
            {"value": "ext-user-789", "display": "new.user@example.com"},
        ]

        updated = scim_service.update_group(
            external_id="group-dev-team",
            org_id="org-1",
            updates=updated_data,
        )

        assert "ext-user-789" in updated.member_external_ids
        assert "ext-user-123" not in updated.member_external_ids

    def test_delete_group(self, scim_service, sample_group_data):
        """Group should be deletable."""
        group = scim_service.create_group(sample_group_data, org_id="org-1")

        scim_service.delete_group(external_id="group-dev-team", org_id="org-1")

        # Should not find the group
        result = scim_service.get_group(external_id="group-dev-team", org_id="org-1")
        assert result is None


# ============================================================================
# Role and Scope Mapping Tests
# ============================================================================


class TestRoleAndScopeMapping:
    """Test SCIM group to role/scope mapping."""

    def test_group_maps_to_role(self, scim_service):
        """SCIM group should map to internal role."""
        scim_service.configure_group_role_mapping(
            scim_group_id="group-maintainers",
            internal_role="maintainer",
        )

        roles = scim_service.get_roles_for_groups(["group-maintainers"])
        assert "maintainer" in roles

    def test_group_maps_to_scopes(self, scim_service):
        """SCIM group should map to OAuth scopes."""
        scim_service.configure_group_scope_mapping(
            scim_group_id="group-dev-team",
            scopes=["repository:read", "repository:write"],
        )

        scopes = scim_service.get_scopes_for_groups(["group-dev-team"])
        assert "repository:read" in scopes
        assert "repository:write" in scopes

    def test_multiple_groups_combine_roles(self, scim_service):
        """Multiple groups should combine roles."""
        scim_service.configure_group_role_mapping("group-1", "user")
        scim_service.configure_group_role_mapping("group-2", "maintainer")

        roles = scim_service.get_roles_for_groups(["group-1", "group-2"])

        assert "user" in roles
        assert "maintainer" in roles

    def test_unmapped_group_returns_default(self, scim_service):
        """Unmapped group should return default role."""
        roles = scim_service.get_roles_for_groups(["unmapped-group"])

        # Default role should be "user"
        assert "user" in roles

    def test_invalid_role_mapping_raises(self, scim_service):
        """Invalid role mapping should raise error."""
        with pytest.raises(SCIMGroupMappingError):
            scim_service.configure_group_role_mapping(
                scim_group_id="group-1",
                internal_role="",  # Empty role is invalid
            )


# ============================================================================
# Orphan Detection Tests
# ============================================================================


class TestOrphanDetection:
    """Test orphan user detection and suspension."""

    def test_detect_orphaned_users(self, scim_service, sample_user_data):
        """Users not in IdP should be detected as orphans."""
        # Provision user
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        # Simulate IdP sync with user missing
        idp_users = []  # No users from IdP

        orphans = scim_service.detect_orphaned_users(
            org_id="org-1",
            idp_user_ids=idp_users,
        )

        assert user.external_id in [o.external_id for o in orphans]

    def test_orphan_marked_for_suspension(self, scim_service, sample_user_data):
        """Orphaned users should be marked for suspension."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        orphans = scim_service.detect_orphaned_users(
            org_id="org-1",
            idp_user_ids=[],
        )

        # Mark for suspension
        scim_service.mark_orphans_for_suspension(orphans)

        updated_user = scim_service.get_user(
            external_id="ext-user-123", org_id="org-1"
        )
        assert updated_user.orphaned_at is not None
        assert updated_user.status == UserStatus.ORPHAN_PENDING

    def test_suspend_orphans_after_72_hours(self, scim_service, sample_user_data):
        """Orphans should be suspended after 72 hours."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        # Mark as orphan with old timestamp
        orphaned_time = datetime.now(timezone.utc) - timedelta(hours=73)
        scim_service._users[user.external_id].orphaned_at = orphaned_time
        scim_service._users[user.external_id].status = UserStatus.ORPHAN_PENDING

        # Run suspension job
        suspended = scim_service.suspend_stale_orphans(org_id="org-1")

        assert len(suspended) == 1
        assert suspended[0].status == UserStatus.SUSPENDED

    def test_orphan_within_72_hours_not_suspended(
        self, scim_service, sample_user_data
    ):
        """Orphans within 72 hours should not be suspended."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        # Mark as orphan with recent timestamp
        orphaned_time = datetime.now(timezone.utc) - timedelta(hours=24)
        scim_service._users[user.external_id].orphaned_at = orphaned_time
        scim_service._users[user.external_id].status = UserStatus.ORPHAN_PENDING

        # Run suspension job
        suspended = scim_service.suspend_stale_orphans(org_id="org-1")

        assert len(suspended) == 0

    def test_reappeared_orphan_restored(self, scim_service, sample_user_data):
        """Orphan that reappears in IdP should be restored."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        # Mark as orphan
        scim_service._users[user.external_id].status = UserStatus.ORPHAN_PENDING
        scim_service._users[user.external_id].orphaned_at = datetime.now(timezone.utc)

        # User reappears in IdP sync
        restored = scim_service.reconcile_user(
            external_id="ext-user-123",
            org_id="org-1",
            user_data=sample_user_data,
        )

        assert restored.status == UserStatus.ACTIVE
        assert restored.orphaned_at is None


# ============================================================================
# Query Tests
# ============================================================================


class TestSCIMQueries:
    """Test SCIM query operations."""

    def test_get_user_by_external_id(self, scim_service, sample_user_data):
        """User should be retrievable by external ID."""
        scim_service.provision_user(sample_user_data, org_id="org-1")

        user = scim_service.get_user(external_id="ext-user-123", org_id="org-1")

        assert user is not None
        assert user.email == "john.doe@example.com"

    def test_get_nonexistent_user_returns_none(self, scim_service):
        """Getting nonexistent user should return None."""
        user = scim_service.get_user(external_id="unknown", org_id="org-1")
        assert user is None

    def test_list_users_by_org(self, scim_service, sample_user_data):
        """Users should be listable by org."""
        scim_service.provision_user(sample_user_data, org_id="org-1")

        sample_user_data["id"] = "ext-user-456"
        sample_user_data["userName"] = "jane@example.com"
        scim_service.provision_user(sample_user_data, org_id="org-1")

        users = scim_service.list_users(org_id="org-1")

        assert len(users) == 2

    def test_list_users_filtered_by_status(self, scim_service, sample_user_data):
        """Users should be filterable by status."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")
        scim_service._users[user.external_id].status = UserStatus.SUSPENDED

        active_users = scim_service.list_users(
            org_id="org-1", status=UserStatus.ACTIVE
        )

        assert len(active_users) == 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_provision_empty_groups_list(self, scim_service, sample_user_data):
        """User with no groups should be handled."""
        sample_user_data["groups"] = []

        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        assert user.group_ids == []

    def test_provision_missing_optional_fields(self, scim_service):
        """Optional fields should have defaults."""
        minimal_data = {
            "id": "ext-user-123",
            "userName": "minimal@example.com",
            "active": True,
        }

        user = scim_service.provision_user(minimal_data, org_id="org-1")

        assert user.email == "minimal@example.com"
        assert user.display_name == "minimal@example.com"  # Fallback to userName

    def test_concurrent_updates_handled(self, scim_service, sample_user_data):
        """Concurrent updates should be handled gracefully."""
        user = scim_service.provision_user(sample_user_data, org_id="org-1")

        # Simulate concurrent updates (use deepcopy to avoid shared mutation)
        update1 = copy.deepcopy(sample_user_data)
        update1["name"]["givenName"] = "John1"
        update1["name"]["formatted"] = "John1 Doe"  # SCIM IdP updates formatted too

        update2 = copy.deepcopy(sample_user_data)
        update2["name"]["givenName"] = "John2"
        update2["name"]["formatted"] = "John2 Doe"  # SCIM IdP updates formatted too

        scim_service.update_user("ext-user-123", "org-1", update1)
        scim_service.update_user("ext-user-123", "org-1", update2)

        # Last update should win
        final = scim_service.get_user("ext-user-123", "org-1")
        assert final.display_name == "John2 Doe"
