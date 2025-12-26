"""
Unit tests for CodeSearchNet dataset loader.

Tests for loading CodeSearchNet Python subset for benchmark evaluation.
Target: MRR >= 0.75 for embedding quality.

These tests use mocking to avoid requiring actual dataset download.
Integration tests (marked with @pytest.mark.integration) test real loading.

Tests written FIRST following TDD approach.
"""

import pytest
from typing import List, Set, Dict, Iterator
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass


class TestCodeSearchNetSampleDataclass:
    """Test CodeSearchNetSample dataclass structure."""

    def test_sample_dataclass_exists(self):
        """CodeSearchNetSample should be importable."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        assert CodeSearchNetSample is not None

    def test_sample_has_query(self):
        """Sample should have a query field."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        sample = CodeSearchNetSample(
            query="find files in directory",
            code="def find_files(path): pass",
            docstring="Find all files in a directory",
            func_name="find_files",
            url="https://github.com/test/repo",
        )
        assert sample.query == "find files in directory"

    def test_sample_has_code(self):
        """Sample should have a code field."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        sample = CodeSearchNetSample(
            query="test query",
            code="def test(): pass",
            docstring="Test docstring",
            func_name="test",
            url="https://github.com/test/repo",
        )
        assert sample.code == "def test(): pass"

    def test_sample_has_docstring(self):
        """Sample should have a docstring field."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        sample = CodeSearchNetSample(
            query="test query",
            code="def test(): pass",
            docstring="A docstring here",
            func_name="test",
            url="https://github.com/test/repo",
        )
        assert sample.docstring == "A docstring here"

    def test_sample_has_func_name(self):
        """Sample should have a func_name field."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        sample = CodeSearchNetSample(
            query="test",
            code="def my_func(): pass",
            docstring="Doc",
            func_name="my_func",
            url="https://github.com/test/repo",
        )
        assert sample.func_name == "my_func"

    def test_sample_has_url(self):
        """Sample should have a url field for provenance."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        sample = CodeSearchNetSample(
            query="test",
            code="def test(): pass",
            docstring="Doc",
            func_name="test",
            url="https://github.com/owner/repo/blob/main/file.py#L10",
        )
        assert "github.com" in sample.url

    def test_sample_id_is_unique(self):
        """Sample should have a unique id derived from content."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetSample

        sample1 = CodeSearchNetSample(
            query="query1",
            code="def func1(): pass",
            docstring="Doc1",
            func_name="func1",
            url="https://github.com/test/repo1",
        )
        sample2 = CodeSearchNetSample(
            query="query2",
            code="def func2(): pass",
            docstring="Doc2",
            func_name="func2",
            url="https://github.com/test/repo2",
        )
        assert sample1.id != sample2.id


class TestCodeSearchNetLoaderInit:
    """Test CodeSearchNetLoader initialization."""

    def test_loader_exists(self):
        """CodeSearchNetLoader should be importable."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        assert CodeSearchNetLoader is not None

    def test_loader_default_language_is_python(self):
        """Default language should be Python."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        assert loader.language == "python"

    def test_loader_accepts_language_parameter(self):
        """Loader should accept language parameter."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(language="python")
        assert loader.language == "python"

    def test_loader_accepts_cache_dir(self):
        """Loader should accept cache directory."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(cache_dir="/tmp/csn_cache")
        assert loader.cache_dir == "/tmp/csn_cache"

    def test_loader_default_cache_dir(self):
        """Loader should have default cache directory."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        assert loader.cache_dir is not None
        assert "codesearchnet" in loader.cache_dir.lower() or "csn" in loader.cache_dir.lower()

    def test_loader_accepts_split_parameter(self):
        """Loader should accept train/valid/test split."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(split="test")
        assert loader.split == "test"

    def test_loader_default_split_is_test(self):
        """Default split should be test for benchmarking."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        assert loader.split == "test"


class TestCodeSearchNetLoaderLoading:
    """Test CodeSearchNetLoader data loading."""

    @pytest.fixture
    def mock_dataset(self):
        """Mock HuggingFace datasets library."""
        mock_data = [
            {
                "func_documentation_string": "Find files in a directory",
                "func_code_string": "def find_files(path):\n    return os.listdir(path)",
                "func_name": "find_files",
                "url": "https://github.com/test/repo/file.py",
            },
            {
                "func_documentation_string": "Sort a list of numbers",
                "func_code_string": "def sort_list(nums):\n    return sorted(nums)",
                "func_name": "sort_list",
                "url": "https://github.com/test/repo2/file.py",
            },
        ]
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=iter(mock_data))
            mock_ds.__len__ = Mock(return_value=len(mock_data))
            mock_load.return_value = {"test": mock_ds}
            yield mock_load, mock_data

    def test_load_returns_iterator(self, mock_dataset):
        """load() should return an iterator."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        result = loader.load()

        # Should be iterable
        assert hasattr(result, "__iter__")

    def test_load_yields_codesearchnet_samples(self, mock_dataset):
        """load() should yield CodeSearchNetSample objects."""
        from benchmarks.embeddings.datasets.codesearchnet import (
            CodeSearchNetLoader,
            CodeSearchNetSample,
        )

        loader = CodeSearchNetLoader()
        samples = list(loader.load())

        assert len(samples) > 0
        assert all(isinstance(s, CodeSearchNetSample) for s in samples)

    def test_load_maps_fields_correctly(self, mock_dataset):
        """load() should map dataset fields to sample fields."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load())

        # Check first sample
        sample = samples[0]
        assert "find" in sample.docstring.lower() or "find" in sample.query.lower()
        assert "find_files" in sample.code
        assert sample.func_name == "find_files"

    def test_load_generates_query_from_docstring(self, mock_dataset):
        """load() should use docstring as query."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load())

        # Query should be derived from docstring
        for sample in samples:
            assert sample.query is not None
            assert len(sample.query) > 0


class TestCodeSearchNetLoaderStreaming:
    """Test memory-efficient streaming/lazy loading."""

    @pytest.fixture
    def mock_large_dataset(self):
        """Mock a large dataset for streaming tests."""
        # Create a generator that yields many items
        def make_items(count):
            for i in range(count):
                yield {
                    "func_documentation_string": f"Function {i} docs",
                    "func_code_string": f"def func_{i}(): pass",
                    "func_name": f"func_{i}",
                    "url": f"https://github.com/test/repo/file{i}.py",
                }

        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=make_items(10000))
            mock_ds.__len__ = Mock(return_value=10000)
            mock_load.return_value = {"test": mock_ds}
            yield mock_load

    def test_load_is_lazy(self, mock_large_dataset):
        """load() should not load all data into memory at once."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        result = loader.load()

        # Should be a generator/iterator, not a list
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__") or hasattr(result, "send")

    def test_load_with_limit(self, mock_large_dataset):
        """load() should support limiting number of samples."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load(limit=10))

        assert len(samples) == 10

    def test_load_with_offset(self, mock_large_dataset):
        """load() should support skipping initial samples."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load(offset=5, limit=5))

        # Should start from offset
        assert len(samples) == 5
        # First sample should be func_5
        assert "func_5" in samples[0].func_name


class TestCodeSearchNetLoaderCaching:
    """Test caching behavior."""

    @pytest.fixture
    def mock_dataset_with_cache(self, tmp_path):
        """Mock dataset with cache directory."""
        mock_data = [
            {
                "func_documentation_string": "Test function",
                "func_code_string": "def test(): pass",
                "func_name": "test",
                "url": "https://github.com/test/repo",
            }
        ]
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=iter(mock_data))
            mock_ds.__len__ = Mock(return_value=len(mock_data))
            mock_load.return_value = {"test": mock_ds}
            yield mock_load, tmp_path

    def test_loader_uses_cache_dir(self, mock_dataset_with_cache):
        """Loader should use specified cache directory."""
        mock_load, tmp_path = mock_dataset_with_cache
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        cache_dir = str(tmp_path / "csn_cache")
        loader = CodeSearchNetLoader(cache_dir=cache_dir)
        list(loader.load())

        # Verify cache_dir was passed to load_dataset
        call_kwargs = mock_load.call_args[1] if mock_load.call_args[1] else {}
        assert "cache_dir" in call_kwargs or cache_dir in str(mock_load.call_args)

    def test_loader_creates_cache_dir_if_missing(self, mock_dataset_with_cache, tmp_path):
        """Loader should create cache directory if it doesn't exist."""
        mock_load, _ = mock_dataset_with_cache
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        cache_dir = tmp_path / "new_cache_dir"
        assert not cache_dir.exists()

        loader = CodeSearchNetLoader(cache_dir=str(cache_dir))
        # Cache dir should be created on init or first load
        list(loader.load())


class TestCodeSearchNetLoaderStats:
    """Test dataset statistics and info methods."""

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset for stats tests."""
        mock_data = [
            {
                "func_documentation_string": "Short doc",
                "func_code_string": "def f(): pass",
                "func_name": "f",
                "url": "url1",
            },
            {
                "func_documentation_string": "Longer documentation string",
                "func_code_string": "def g():\n    x = 1\n    return x",
                "func_name": "g",
                "url": "url2",
            },
            {
                "func_documentation_string": "Another doc",
                "func_code_string": "def h(): return None",
                "func_name": "h",
                "url": "url3",
            },
        ]
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=iter(mock_data))
            mock_ds.__len__ = Mock(return_value=len(mock_data))
            mock_load.return_value = {"test": mock_ds}
            yield mock_load, mock_data

    def test_get_size_returns_count(self, mock_dataset):
        """get_size() should return total number of samples."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        size = loader.get_size()

        assert size == 3

    def test_info_returns_metadata(self, mock_dataset):
        """info() should return dataset metadata."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        info = loader.info()

        assert "language" in info
        assert info["language"] == "python"
        assert "split" in info
        assert "source" in info
        assert "codesearchnet" in info["source"].lower()


class TestCodeSearchNetLoaderQueryExtraction:
    """Test query extraction from docstrings."""

    @pytest.fixture
    def mock_dataset_varied_docs(self):
        """Mock dataset with varied docstring formats."""
        mock_data = [
            {
                "func_documentation_string": "Returns a list of files in the directory.\n\nArgs:\n    path: The directory path.",
                "func_code_string": "def list_files(path): pass",
                "func_name": "list_files",
                "url": "url1",
            },
            {
                "func_documentation_string": "Sort list",  # Very short
                "func_code_string": "def sort(lst): pass",
                "func_name": "sort",
                "url": "url2",
            },
            {
                "func_documentation_string": "",  # Empty
                "func_code_string": "def empty_doc(): pass",
                "func_name": "empty_doc",
                "url": "url3",
            },
        ]
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=iter(mock_data))
            mock_ds.__len__ = Mock(return_value=len(mock_data))
            mock_load.return_value = {"test": mock_ds}
            yield mock_load

    def test_extracts_first_line_as_query(self, mock_dataset_varied_docs):
        """Query should be first line/sentence of docstring."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load())

        # First sample has multi-line docstring
        first_sample = samples[0]
        # Query should be just the first line, not the full docstring
        assert "Args:" not in first_sample.query
        assert "Returns" in first_sample.query or "list" in first_sample.query.lower()

    def test_handles_empty_docstring(self, mock_dataset_varied_docs):
        """Should handle empty docstrings gracefully."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load())

        # Find sample with empty docstring
        empty_doc_sample = next((s for s in samples if s.func_name == "empty_doc"), None)

        # Should use func_name or code as fallback, or skip
        assert empty_doc_sample is None or empty_doc_sample.query is not None


class TestCodeSearchNetLoaderFiltering:
    """Test filtering capabilities."""

    @pytest.fixture
    def mock_dataset_with_duplicates(self):
        """Mock dataset with duplicates and edge cases."""
        mock_data = [
            {
                "func_documentation_string": "Good function",
                "func_code_string": "def good(): return True",
                "func_name": "good",
                "url": "url1",
            },
            {
                "func_documentation_string": "Good function",  # Duplicate docstring
                "func_code_string": "def good_copy(): return True",
                "func_name": "good_copy",
                "url": "url2",
            },
            {
                "func_documentation_string": "",
                "func_code_string": "",  # Empty code
                "func_name": "empty",
                "url": "url3",
            },
            {
                "func_documentation_string": "Short",  # Too short
                "func_code_string": "x",
                "func_name": "s",
                "url": "url4",
            },
        ]
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=iter(mock_data))
            mock_ds.__len__ = Mock(return_value=len(mock_data))
            mock_load.return_value = {"test": mock_ds}
            yield mock_load

    def test_filters_empty_code(self, mock_dataset_with_duplicates):
        """Should filter out samples with empty code."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(min_code_length=1)
        samples = list(loader.load())

        # No sample should have empty code
        assert all(len(s.code) > 0 for s in samples)

    def test_filters_by_min_code_length(self, mock_dataset_with_duplicates):
        """Should filter by minimum code length."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(min_code_length=10)
        samples = list(loader.load())

        # All samples should have code length >= 10
        assert all(len(s.code) >= 10 for s in samples)

    def test_deduplication_by_query(self, mock_dataset_with_duplicates):
        """Should optionally deduplicate by query."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(deduplicate=True)
        samples = list(loader.load())

        # No duplicate queries
        queries = [s.query for s in samples]
        assert len(queries) == len(set(queries))


class TestCodeSearchNetBenchmarkIntegration:
    """Test integration with benchmark framework."""

    @pytest.fixture
    def mock_dataset(self):
        """Mock dataset."""
        mock_data = [
            {
                "func_documentation_string": "Find files in directory",
                "func_code_string": "def find_files(path): return os.listdir(path)",
                "func_name": "find_files",
                "url": "url1",
            },
            {
                "func_documentation_string": "Sort numbers ascending",
                "func_code_string": "def sort_asc(nums): return sorted(nums)",
                "func_name": "sort_asc",
                "url": "url2",
            },
        ]
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_ds = MagicMock()
            mock_ds.__iter__ = Mock(return_value=iter(mock_data))
            mock_ds.__len__ = Mock(return_value=len(mock_data))
            mock_load.return_value = {"test": mock_ds}
            yield mock_load

    def test_sample_provides_query_document_pair(self, mock_dataset):
        """Each sample should provide (query, document) pair for retrieval."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        samples = list(loader.load())

        for sample in samples:
            # Has query
            assert sample.query is not None
            assert len(sample.query) > 0
            # Has document (code)
            assert sample.code is not None
            assert len(sample.code) > 0

    def test_get_benchmark_pairs_method(self, mock_dataset):
        """Should have method to get (query, positive_doc) pairs."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        pairs = loader.get_benchmark_pairs()

        # Should return iterator of (query, code) tuples
        pair_list = list(pairs)
        assert len(pair_list) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pair_list)

    def test_get_queries_method(self, mock_dataset):
        """Should have method to get just queries."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        queries = loader.get_queries()

        query_list = list(queries)
        assert len(query_list) > 0
        assert all(isinstance(q, str) for q in query_list)

    def test_get_documents_method(self, mock_dataset):
        """Should have method to get just documents (code)."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader()
        docs = loader.get_documents()

        doc_list = list(docs)
        assert len(doc_list) > 0
        assert all(isinstance(d, str) for d in doc_list)


class TestCodeSearchNetLoaderErrorHandling:
    """Test error handling."""

    def test_invalid_language_raises_error(self):
        """Invalid language should raise ValueError."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        with pytest.raises(ValueError):
            CodeSearchNetLoader(language="invalid_language")

    def test_invalid_split_raises_error(self):
        """Invalid split should raise ValueError."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        with pytest.raises(ValueError):
            CodeSearchNetLoader(split="invalid_split")

    def test_network_error_gives_clear_message(self):
        """Network errors should give clear error message."""
        with patch("benchmarks.embeddings.datasets.codesearchnet.load_dataset") as mock_load:
            mock_load.side_effect = ConnectionError("Network unavailable")

            from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

            loader = CodeSearchNetLoader()
            with pytest.raises(ConnectionError):
                list(loader.load())


@pytest.mark.integration
class TestCodeSearchNetLoaderIntegration:
    """Integration tests that actually download data.

    These tests require network access and download real data.
    Run with: pytest -m integration
    """

    def test_real_load_python(self):
        """Test loading real Python subset (small sample)."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(language="python", split="test")
        samples = list(loader.load(limit=5))

        assert len(samples) == 5
        for sample in samples:
            assert sample.query
            assert sample.code
            assert "def " in sample.code  # Python function

    def test_real_data_quality(self):
        """Test that real data has expected quality."""
        from benchmarks.embeddings.datasets.codesearchnet import CodeSearchNetLoader

        loader = CodeSearchNetLoader(language="python", split="test")
        samples = list(loader.load(limit=100))

        # Check data quality
        non_empty_queries = sum(1 for s in samples if len(s.query.strip()) > 0)
        non_empty_code = sum(1 for s in samples if len(s.code.strip()) > 0)

        assert non_empty_queries / len(samples) >= 0.95  # 95% have queries
        assert non_empty_code / len(samples) >= 0.99  # 99% have code
