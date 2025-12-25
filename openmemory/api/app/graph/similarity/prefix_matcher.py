"""
Prefix/suffix matching module for entity normalization.

Identifies entity variants where one entity is a prefix or suffix
of another, indicating they may refer to the same thing.

Examples:
    - "marie" is a prefix of "marie_schubenz"
    - "el_juego" is a prefix of "el_juego-community"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class PrefixMatch:
    """A prefix/suffix match between two entities."""
    shorter: str
    longer: str
    match_type: str  # "prefix" or "suffix"
    overlap_ratio: float  # Ratio of shorter length to longer length


def find_prefix_matches(
    entities: List[str],
    min_prefix_len: int = 4,
    min_overlap_ratio: float = 0.5,
) -> List[PrefixMatch]:
    """
    Find entity pairs where one entity is a prefix/suffix of another.

    Args:
        entities: List of all entity names
        min_prefix_len: Minimum length of the shorter entity
        min_overlap_ratio: Minimum ratio (shorter_len / longer_len)

    Returns:
        List of PrefixMatch objects

    Example matches:
        - ("marie", "marie_schubenz", "prefix", 0.38)
        - ("el_juego", "el_juego_community", "prefix", 0.47)
    """
    matches = []

    # Pre-compute normalized forms (lowercase, no separators for matching)
    normalized = [
        (e, e.lower().replace('_', '').replace('-', '').replace('.', '').replace(' ', ''))
        for e in entities
    ]

    for i, (orig1, norm1) in enumerate(normalized):
        for j in range(i + 1, len(normalized)):
            orig2, norm2 = normalized[j]

            # Determine shorter and longer
            if len(norm1) <= len(norm2):
                shorter_orig, shorter_norm = orig1, norm1
                longer_orig, longer_norm = orig2, norm2
            else:
                shorter_orig, shorter_norm = orig2, norm2
                longer_orig, longer_norm = orig1, norm1

            # Skip if shorter is too short
            if len(shorter_norm) < min_prefix_len:
                continue

            # Skip if identical (handled by case normalizer)
            if shorter_norm == longer_norm:
                continue

            # Check for prefix match
            if longer_norm.startswith(shorter_norm):
                overlap = len(shorter_norm) / len(longer_norm)
                if overlap >= min_overlap_ratio:
                    matches.append(PrefixMatch(
                        shorter=shorter_orig,
                        longer=longer_orig,
                        match_type="prefix",
                        overlap_ratio=overlap
                    ))
                continue  # Don't check suffix if prefix matched

            # Check for suffix match
            if longer_norm.endswith(shorter_norm):
                overlap = len(shorter_norm) / len(longer_norm)
                if overlap >= min_overlap_ratio:
                    matches.append(PrefixMatch(
                        shorter=shorter_orig,
                        longer=longer_orig,
                        match_type="suffix",
                        overlap_ratio=overlap
                    ))

    return matches
