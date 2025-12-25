"""
Similarity module for semantic entity normalization.

Contains various similarity algorithms for detecting entity variants:
- String similarity (Levenshtein, fuzzy matching)
- Prefix/suffix matching
- Domain normalization (.community, etc.)
- Embedding similarity (optional, API-based)
"""

from .string_similarity import (
    StringSimilarityMatch,
    strip_separators,
    levenshtein_ratio,
    find_string_similar_entities,
)

from .prefix_matcher import (
    PrefixMatch,
    find_prefix_matches,
)

from .domain_normalizer import (
    DomainMatch,
    extract_domain_core,
    find_domain_matches,
)

__all__ = [
    # String similarity
    "StringSimilarityMatch",
    "strip_separators",
    "levenshtein_ratio",
    "find_string_similar_entities",
    # Prefix matcher
    "PrefixMatch",
    "find_prefix_matches",
    # Domain normalizer
    "DomainMatch",
    "extract_domain_core",
    "find_domain_matches",
]
