"""SCIM 2.0 integration stubs for user provisioning.

This module implements SCIM 2.0 stubs per implementation plan section 4.8:
- User provisioning/deprovisioning
- Group management
- SCIM groups mapping to IdP roles and OAuth scopes
- Orphan detection (suspend unlinked users within 72 hours)

Note: This is a stub implementation for Phase 0b. Full SCIM endpoint
integration will be implemented in Phase 3.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# ============================================================================
# Exceptions
# ============================================================================


class SCIMProvisioningError(Exception):
    """Error during user provisioning."""

    pass


class SCIMDeactivationError(Exception):
    """Error during user deactivation."""

    pass


class SCIMGroupMappingError(Exception):
    """Error in group to role/scope mapping."""

    pass


# ============================================================================
# Enums
# ============================================================================


class UserStatus(str, Enum):
    """User account status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPROVISIONED = "deprovisioned"
    ORPHAN_PENDING = "orphan_pending"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SCIMUser:
    """Represents a user provisioned via SCIM."""

    internal_id: str
    external_id: str
    email: str
    display_name: str
    org_id: str
    status: UserStatus = UserStatus.ACTIVE
    group_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deprovisioned_at: datetime | None = None
    orphaned_at: datetime | None = None
    raw_scim_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SCIMGroup:
    """Represents a group provisioned via SCIM."""

    internal_id: str
    external_id: str
    display_name: str
    org_id: str
    member_external_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_scim_data: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SCIM Service
# ============================================================================


class SCIMService:
    """SCIM 2.0 service stub for user and group provisioning.

    This is a stub implementation that stores data in-memory.
    Production implementation will integrate with the database.
    """

    # Default orphan suspension window (72 hours per plan)
    ORPHAN_SUSPENSION_HOURS = 72

    def __init__(self):
        """Initialize SCIM service with in-memory storage."""
        # In-memory storage (stub)
        self._users: dict[str, SCIMUser] = {}  # external_id -> user
        self._groups: dict[str, SCIMGroup] = {}  # external_id -> group

        # Group to role/scope mappings
        self._group_role_mappings: dict[str, str] = {}  # group_id -> role
        self._group_scope_mappings: dict[str, list[str]] = {}  # group_id -> scopes

    # ========================================================================
    # User Provisioning
    # ========================================================================

    def provision_user(
        self, user_data: dict[str, Any], org_id: str
    ) -> SCIMUser:
        """Provision a user from SCIM data.

        Args:
            user_data: SCIM user resource data
            org_id: The organization ID

        Returns:
            The provisioned or updated user

        Raises:
            SCIMProvisioningError: If provisioning fails
        """
        # Validate required fields
        if "userName" not in user_data:
            raise SCIMProvisioningError("Missing required field: userName")

        external_id = user_data.get("id") or user_data.get("externalId")
        if not external_id:
            raise SCIMProvisioningError("Missing required field: id or externalId")

        # Check if user already exists (update case)
        existing = self._users.get(external_id)

        # Extract user data
        email = self._extract_email(user_data)
        display_name = self._extract_display_name(user_data)
        group_ids = self._extract_group_ids(user_data)
        is_active = user_data.get("active", True)

        if existing:
            # Update existing user
            existing.email = email
            existing.display_name = display_name
            existing.group_ids = group_ids
            existing.status = UserStatus.ACTIVE if is_active else UserStatus.SUSPENDED
            existing.updated_at = datetime.now(timezone.utc)
            existing.raw_scim_data = user_data
            return existing
        else:
            # Create new user
            user = SCIMUser(
                internal_id=str(uuid.uuid4()),
                external_id=external_id,
                email=email,
                display_name=display_name,
                org_id=org_id,
                status=UserStatus.ACTIVE if is_active else UserStatus.SUSPENDED,
                group_ids=group_ids,
                raw_scim_data=user_data,
            )
            self._users[external_id] = user
            return user

    def deprovision_user(self, external_id: str, org_id: str) -> SCIMUser:
        """Deprovision a user.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID

        Returns:
            The deprovisioned user

        Raises:
            SCIMDeactivationError: If user not found
        """
        user = self._users.get(external_id)
        if not user or user.org_id != org_id:
            raise SCIMDeactivationError(f"User not found: {external_id}")

        user.status = UserStatus.DEPROVISIONED
        user.group_ids = []
        user.deprovisioned_at = datetime.now(timezone.utc)
        user.updated_at = datetime.now(timezone.utc)

        return user

    def update_user(
        self, external_id: str, org_id: str, updates: dict[str, Any]
    ) -> SCIMUser:
        """Update a user from SCIM data.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID
            updates: SCIM user resource updates

        Returns:
            The updated user

        Raises:
            SCIMDeactivationError: If user not found
        """
        user = self._users.get(external_id)
        if not user or user.org_id != org_id:
            raise SCIMDeactivationError(f"User not found: {external_id}")

        # Update fields
        if "emails" in updates:
            user.email = self._extract_email(updates)

        if "name" in updates or "userName" in updates:
            user.display_name = self._extract_display_name(updates)

        if "groups" in updates:
            user.group_ids = self._extract_group_ids(updates)

        if "active" in updates:
            user.status = (
                UserStatus.ACTIVE if updates["active"] else UserStatus.SUSPENDED
            )

        user.updated_at = datetime.now(timezone.utc)
        user.raw_scim_data = updates

        return user

    def get_user(self, external_id: str, org_id: str) -> SCIMUser | None:
        """Get a user by external ID.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID

        Returns:
            The user or None if not found
        """
        user = self._users.get(external_id)
        if user and user.org_id == org_id:
            return user
        return None

    def list_users(
        self, org_id: str, status: UserStatus | None = None
    ) -> list[SCIMUser]:
        """List users in an organization.

        Args:
            org_id: The organization ID
            status: Optional status filter

        Returns:
            List of users
        """
        users = [u for u in self._users.values() if u.org_id == org_id]
        if status:
            users = [u for u in users if u.status == status]
        return users

    # ========================================================================
    # Group Management
    # ========================================================================

    def create_group(
        self, group_data: dict[str, Any], org_id: str
    ) -> SCIMGroup:
        """Create a group from SCIM data.

        Args:
            group_data: SCIM group resource data
            org_id: The organization ID

        Returns:
            The created group
        """
        external_id = group_data.get("id")
        display_name = group_data.get("displayName", "")
        members = group_data.get("members", [])

        member_ids = [m.get("value") for m in members if m.get("value")]

        group = SCIMGroup(
            internal_id=str(uuid.uuid4()),
            external_id=external_id,
            display_name=display_name,
            org_id=org_id,
            member_external_ids=member_ids,
            raw_scim_data=group_data,
        )

        self._groups[external_id] = group
        return group

    def update_group(
        self, external_id: str, org_id: str, updates: dict[str, Any]
    ) -> SCIMGroup:
        """Update a group from SCIM data.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID
            updates: SCIM group resource updates

        Returns:
            The updated group
        """
        group = self._groups.get(external_id)
        if not group or group.org_id != org_id:
            raise SCIMGroupMappingError(f"Group not found: {external_id}")

        if "displayName" in updates:
            group.display_name = updates["displayName"]

        if "members" in updates:
            members = updates["members"]
            group.member_external_ids = [
                m.get("value") for m in members if m.get("value")
            ]

        group.updated_at = datetime.now(timezone.utc)
        group.raw_scim_data = updates

        return group

    def get_group(self, external_id: str, org_id: str) -> SCIMGroup | None:
        """Get a group by external ID.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID

        Returns:
            The group or None if not found
        """
        group = self._groups.get(external_id)
        if group and group.org_id == org_id:
            return group
        return None

    def delete_group(self, external_id: str, org_id: str) -> None:
        """Delete a group.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID
        """
        group = self._groups.get(external_id)
        if group and group.org_id == org_id:
            del self._groups[external_id]

    # ========================================================================
    # Role and Scope Mapping
    # ========================================================================

    def configure_group_role_mapping(
        self, scim_group_id: str, internal_role: str
    ) -> None:
        """Configure a group to role mapping.

        Args:
            scim_group_id: The SCIM group ID
            internal_role: The internal role name

        Raises:
            SCIMGroupMappingError: If role is invalid
        """
        if not internal_role:
            raise SCIMGroupMappingError("Role cannot be empty")

        self._group_role_mappings[scim_group_id] = internal_role

    def configure_group_scope_mapping(
        self, scim_group_id: str, scopes: list[str]
    ) -> None:
        """Configure a group to scopes mapping.

        Args:
            scim_group_id: The SCIM group ID
            scopes: List of OAuth scopes
        """
        self._group_scope_mappings[scim_group_id] = scopes

    def get_roles_for_groups(self, group_ids: list[str]) -> list[str]:
        """Get roles for a list of groups.

        Args:
            group_ids: List of SCIM group IDs

        Returns:
            List of unique roles
        """
        roles = set()
        for group_id in group_ids:
            role = self._group_role_mappings.get(group_id)
            if role:
                roles.add(role)

        # Default role if no mappings found
        if not roles:
            roles.add("user")

        return list(roles)

    def get_scopes_for_groups(self, group_ids: list[str]) -> list[str]:
        """Get OAuth scopes for a list of groups.

        Args:
            group_ids: List of SCIM group IDs

        Returns:
            List of unique scopes
        """
        scopes = set()
        for group_id in group_ids:
            group_scopes = self._group_scope_mappings.get(group_id, [])
            scopes.update(group_scopes)
        return list(scopes)

    # ========================================================================
    # Orphan Detection
    # ========================================================================

    def detect_orphaned_users(
        self, org_id: str, idp_user_ids: list[str]
    ) -> list[SCIMUser]:
        """Detect users not present in IdP.

        Args:
            org_id: The organization ID
            idp_user_ids: List of external IDs from IdP

        Returns:
            List of orphaned users
        """
        idp_set = set(idp_user_ids)
        orphans = []

        for user in self._users.values():
            if user.org_id != org_id:
                continue
            if user.status in [UserStatus.DEPROVISIONED, UserStatus.ORPHAN_PENDING]:
                continue
            if user.external_id not in idp_set:
                orphans.append(user)

        return orphans

    def mark_orphans_for_suspension(self, orphans: list[SCIMUser]) -> None:
        """Mark orphaned users for suspension.

        Args:
            orphans: List of orphaned users
        """
        now = datetime.now(timezone.utc)
        for user in orphans:
            user.status = UserStatus.ORPHAN_PENDING
            user.orphaned_at = now
            user.updated_at = now

    def suspend_stale_orphans(self, org_id: str) -> list[SCIMUser]:
        """Suspend orphans older than 72 hours.

        Args:
            org_id: The organization ID

        Returns:
            List of suspended users
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.ORPHAN_SUSPENSION_HOURS)
        suspended = []

        for user in self._users.values():
            if user.org_id != org_id:
                continue
            if user.status != UserStatus.ORPHAN_PENDING:
                continue
            if user.orphaned_at and user.orphaned_at < cutoff:
                user.status = UserStatus.SUSPENDED
                user.updated_at = now
                suspended.append(user)

        return suspended

    def reconcile_user(
        self, external_id: str, org_id: str, user_data: dict[str, Any]
    ) -> SCIMUser:
        """Reconcile a user that reappeared in IdP.

        Args:
            external_id: The SCIM external ID
            org_id: The organization ID
            user_data: Current SCIM data

        Returns:
            The reconciled user
        """
        user = self._users.get(external_id)
        if not user or user.org_id != org_id:
            # Provision as new user
            return self.provision_user(user_data, org_id)

        # Restore orphaned user
        if user.status == UserStatus.ORPHAN_PENDING:
            user.status = UserStatus.ACTIVE
            user.orphaned_at = None
            user.updated_at = datetime.now(timezone.utc)

        return user

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _extract_email(self, user_data: dict[str, Any]) -> str:
        """Extract primary email from SCIM user data."""
        emails = user_data.get("emails", [])
        if emails:
            # Find primary email
            for email in emails:
                if email.get("primary"):
                    return email.get("value", "")
            # Fallback to first email
            return emails[0].get("value", "")
        # Fallback to userName
        return user_data.get("userName", "")

    def _extract_display_name(self, user_data: dict[str, Any]) -> str:
        """Extract display name from SCIM user data."""
        name = user_data.get("name", {})
        if name.get("formatted"):
            return name["formatted"]
        if name.get("givenName") or name.get("familyName"):
            parts = [name.get("givenName", ""), name.get("familyName", "")]
            return " ".join(p for p in parts if p)
        # Fallback to userName
        return user_data.get("userName", "")

    def _extract_group_ids(self, user_data: dict[str, Any]) -> list[str]:
        """Extract group IDs from SCIM user data."""
        groups = user_data.get("groups", [])
        return [g.get("value") for g in groups if g.get("value")]
