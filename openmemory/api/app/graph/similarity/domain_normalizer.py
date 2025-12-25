"""
Domain normalization module for entity normalization.

Handles domain/URL-based entity variants by extracting the core
name from entities with common domain suffixes.

Examples:
    - "eljuego.community" -> core: "eljuego"
    - "el_juego-community" -> core: "el_juego"
    - "project.de" -> core: "project"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


# Known TLDs and community/project suffixes
DOMAIN_SUFFIXES = [
    # Domain TLDs (ordered by length for greedy matching)
    '.community',
    '.org',
    '.net',
    '.com',
    '.app',
    '.io',
    '.de',
    # Common project/community suffixes with separators
    '-community',
    '_community',
    '-project',
    '_project',
    '-team',
    '_team',
    '-app',
    '_app',
]


@dataclass
class DomainMatch:
    """A domain-based match between two entities."""
    original: str
    domain_core: str
    suffix_removed: str


def extract_domain_core(entity: str) -> Optional[str]:
    """
    Extract the core name from a domain-based entity.

    Args:
        entity: Entity name possibly containing domain suffix

    Returns:
        Core name without suffix, or None if no suffix found

    Examples:
        - "eljuego.community" -> "eljuego"
        - "el_juego-community" -> "el_juego"
        - "plain_entity" -> None
    """
    normalized = entity.lower()

    for suffix in DOMAIN_SUFFIXES:
        if normalized.endswith(suffix):
            core = entity[:-len(suffix)]
            if core:  # Ensure we have something left
                return core

    return None


def find_domain_matches(
    entities: List[str],
) -> List[Tuple[str, str, float]]:
    """
    Find entity pairs based on domain normalization.

    For each entity with a domain suffix, finds other entities
    that match its core name (after stripping separators).

    Args:
        entities: List of all entity names

    Returns:
        List of (entity_a, entity_b, confidence) tuples
        Confidence is typically 0.95 for domain matches

    Examples:
        - ("eljuego.community", "el_juego", 0.95) - core matches after normalization
    """
    matches = []

    # Extract cores for entities with domain suffixes
    cores = {}
    for entity in entities:
        core = extract_domain_core(entity)
        if core:
            # Normalize core for comparison (remove separators)
            core_normalized = core.lower().replace('_', '').replace('-', '').replace('.', '').replace(' ', '')
            cores[entity] = (core, core_normalized)

    # Compare cores with all other entities
    seen_pairs = set()

    for entity_with_suffix, (core, core_normalized) in cores.items():
        for other_entity in entities:
            if other_entity == entity_with_suffix:
                continue

            # Create sorted pair to avoid duplicates
            pair = tuple(sorted([entity_with_suffix, other_entity]))
            if pair in seen_pairs:
                continue

            # Normalize other entity for comparison
            other_normalized = (
                other_entity.lower()
                .replace('_', '')
                .replace('-', '')
                .replace('.', '')
                .replace(' ', '')
            )

            # Check if core matches other entity
            if other_normalized == core_normalized:
                matches.append((
                    entity_with_suffix,
                    other_entity,
                    0.95  # High confidence for domain matches
                ))
                seen_pairs.add(pair)

    return matches
