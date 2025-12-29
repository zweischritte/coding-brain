"""
Access control utilities for multi-user memory routing.

Provides helpers for resolving and checking access_entity values
based on Principal grants.
"""

from typing import Set

from .types import Principal


def resolve_access_entities(principal: Principal) -> Set[str]:
    """Resolve all access_entity values a principal can access.

    This returns the explicit grants plus the implied user grant.
    Note: This does NOT expand hierarchies - use principal.can_access()
    for hierarchical access checking.

    Args:
        principal: The authenticated principal

    Returns:
        Set of access_entity values the principal has explicit grants for
    """
    return principal.get_allowed_access_entities()


def can_write_to_access_entity(principal: Principal, access_entity: str) -> bool:
    """Check if principal can write (create/update/delete) to an access_entity.

    For group-editable policy: any member of the access_entity can write.
    This means having a grant that allows access to the access_entity.

    Args:
        principal: The authenticated principal
        access_entity: The access_entity to check write access for

    Returns:
        True if principal can write to the access_entity
    """
    return principal.can_access(access_entity)


def can_read_access_entity(principal: Principal, access_entity: str) -> bool:
    """Check if principal can read from an access_entity.

    Args:
        principal: The authenticated principal
        access_entity: The access_entity to check read access for

    Returns:
        True if principal can read from the access_entity
    """
    return principal.can_access(access_entity)
