"""
Shared pytest fixtures for benchmark tests.
"""

import pytest
from typing import List, Set, Dict, Tuple


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )


@pytest.fixture
def mock_embedding_1024() -> List[float]:
    """Mock 1024-dimensional embedding for unit tests."""
    return [0.1] * 1024


@pytest.fixture
def mock_embedding_3072() -> List[float]:
    """Mock 3072-dimensional embedding (Gemini) for unit tests."""
    return [0.1] * 3072


@pytest.fixture
def sample_ranked_results() -> List[List[str]]:
    """Sample ranked results for MRR/NDCG tests."""
    return [
        ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Query 1
        ["doc2", "doc3", "doc1", "doc5", "doc4"],  # Query 2
        ["doc5", "doc4", "doc3", "doc2", "doc1"],  # Query 3
    ]


@pytest.fixture
def sample_relevant_docs() -> List[Set[str]]:
    """Sample relevant docs for MRR tests."""
    return [
        {"doc1"},      # Query 1: relevant is rank 1
        {"doc1"},      # Query 2: relevant is rank 3
        {"doc1"},      # Query 3: relevant is rank 5
    ]


@pytest.fixture
def sample_graded_relevance() -> List[Dict[str, int]]:
    """Sample graded relevance for NDCG tests (0-2 scale)."""
    return [
        {"doc1": 2, "doc2": 1, "doc3": 0, "doc4": 0, "doc5": 0},  # Query 1
        {"doc1": 2, "doc2": 1, "doc3": 1, "doc4": 0, "doc5": 0},  # Query 2
        {"doc1": 2, "doc2": 0, "doc3": 0, "doc4": 0, "doc5": 1},  # Query 3
    ]


@pytest.fixture
def sample_codesearchnet_data() -> List[Tuple[str, Set[str]]]:
    """Small sample CodeSearchNet-style data for fast tests."""
    return [
        ("find files in directory", {"doc1", "doc2"}),
        ("sort list of numbers", {"doc3"}),
        ("read json file", {"doc4", "doc5"}),
        ("create http server", {"doc6"}),
    ]


@pytest.fixture
def latency_samples() -> List[float]:
    """Sample latency values in milliseconds for percentile tests."""
    # 100 samples with some outliers
    return (
        [10.0] * 50 +   # 50 samples at 10ms
        [20.0] * 30 +   # 30 samples at 20ms
        [50.0] * 15 +   # 15 samples at 50ms
        [100.0] * 4 +   # 4 samples at 100ms (P95 area)
        [500.0] * 1     # 1 outlier at 500ms (P99 area)
    )
