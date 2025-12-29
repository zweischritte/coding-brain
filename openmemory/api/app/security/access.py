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


def get_default_access_entity(principal: Principal) -> str:
    """Get the default access_entity for a principal (their user access_entity).

    Args:
        principal: The authenticated principal

    Returns:
        The user:<user_id> access_entity
    """
    return f"user:{principal.user_id}"


def build_access_entity_patterns(principal: Principal) -> list[str]:
    """Build SQL LIKE patterns for access_entity filtering with hierarchy expansion.

    For org grants, we need to match:
    - org:X (exact match)
    - project:X/* (all projects under org)
    - team:X/* (all teams under org)
    - client:X/* (all clients under org)

    For project grants:
    - project:X (exact match)
    - team:X/* (all teams under project)

    Args:
        principal: The authenticated principal

    Returns:
        List of (exact_matches, like_patterns) for SQL WHERE clause
    """
    exact_matches = set()
    like_patterns = set()

    # Always include user grant
    exact_matches.add(f"user:{principal.user_id}")

    for grant in principal.claims.grants:
        if ":" not in grant:
            continue

        prefix, path = grant.split(":", 1)
        exact_matches.add(grant)

        if prefix == "org":
            # org:cloudfactory allows project:cloudfactory/*, team:cloudfactory/*, client:cloudfactory/*
            like_patterns.add(f"project:{path}/%")
            like_patterns.add(f"team:{path}/%")
            like_patterns.add(f"client:{path}/%")
        elif prefix == "project":
            # project:cloudfactory/acme/billing allows team:cloudfactory/acme/billing/*
            like_patterns.add(f"team:{path}/%")

    return list(exact_matches), list(like_patterns)


def filter_memories_by_access(
    principal: Principal,
    memories: list,
    get_access_entity: callable = lambda m: m.metadata_.get("access_entity") if m.metadata_ else None,
) -> list:
    """Filter a list of memories to only include those the principal can access.

    Args:
        principal: The authenticated principal
        memories: List of memory objects
        get_access_entity: Function to extract access_entity from a memory object

    Returns:
        Filtered list of memories
    """
    result = []
    for memory in memories:
        access_entity = get_access_entity(memory)
        # If no access_entity, check if it's the user's own memory
        if access_entity is None:
            # Legacy memory without access_entity - allow owner only
            if hasattr(memory, 'user') and memory.user and memory.user.user_id == principal.user_id:
                result.append(memory)
            continue

        if principal.can_access(access_entity):
            result.append(memory)

    return result


def check_create_access(principal: Principal, access_entity: str) -> bool:
    """Check if principal can create a memory with the given access_entity.

    Args:
        principal: The authenticated principal
        access_entity: The access_entity for the new memory

    Returns:
        True if principal can create a memory with this access_entity
    """
    return principal.can_access(access_entity)
