"""
String similarity module for entity normalization.

Uses Levenshtein distance and fuzzy matching to identify
entity variants that are similar after normalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Import rapidfuzz with fallback to simple implementation
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


@dataclass
class StringSimilarityMatch:
    """A string similarity match between two entities."""
    entity_a: str
    entity_b: str
    score: float  # 0.0 - 1.0
    method: str   # "levenshtein", "jaro_winkler", "normalized_exact"


def strip_separators(name: str) -> str:
    """
    Remove all separator characters for comparison.

    Normalizes: underscores, dots, hyphens, and spaces to nothing.

    Examples:
        - "el_juego" -> "eljuego"
        - "el-juego" -> "eljuego"
        - "el.juego" -> "eljuego"
        - "El Juego" -> "eljuego"
    """
    return re.sub(r'[._\-\s]', '', name.lower())


def levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Calculate Levenshtein distance as a ratio (1.0 = identical).

    Uses rapidfuzz if available for performance, otherwise falls
    back to a simple implementation.
    """
    if RAPIDFUZZ_AVAILABLE:
        return fuzz.ratio(s1, s2) / 100.0
    else:
        return _simple_levenshtein_ratio(s1, s2)


def _simple_levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Simple Levenshtein ratio implementation (fallback).

    Returns: 1.0 - (distance / max_len)
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Create distance matrix
    distances = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        distances[i][0] = i
    for j in range(len2 + 1):
        distances[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,      # deletion
                distances[i][j - 1] + 1,      # insertion
                distances[i - 1][j - 1] + cost  # substitution
            )

    distance = distances[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (distance / max_len)


def find_string_similar_entities(
    entities: List[str],
    threshold: float = 0.85,
) -> List[StringSimilarityMatch]:
    """
    Find entity pairs with high string similarity.

    Compares all entity pairs after stripping separators.
    Pairs that are identical after normalization are skipped
    (handled by case-based normalization).

    Args:
        entities: List of all entity names
        threshold: Minimum similarity (0.85 = 85%)

    Returns:
        List of matches with scores

    Performance:
        O(n^2) where n = number of entities
        Uses early termination for efficiency
    """
    matches = []

    # Pre-compute normalized forms for efficiency
    normalized = [(e, strip_separators(e)) for e in entities]

    for i, (orig1, norm1) in enumerate(normalized):
        for j in range(i + 1, len(normalized)):
            orig2, norm2 = normalized[j]

            # Skip if identical after normalization (handled by case normalizer)
            if norm1 == norm2:
                continue

            # Skip if length difference is too large (early termination)
            len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
            if len_ratio < threshold:
                continue

            # Calculate similarity
            score = levenshtein_ratio(norm1, norm2)

            if score >= threshold:
                matches.append(StringSimilarityMatch(
                    entity_a=orig1,
                    entity_b=orig2,
                    score=score,
                    method="levenshtein"
                ))

    return matches
