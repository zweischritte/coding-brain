"""
CodeSearchNet dataset loader for benchmark evaluation.

Loads the CodeSearchNet Python subset for evaluating embedding quality.
Target: MRR >= 0.75 for production readiness.

The CodeSearchNet dataset contains function-level code with natural language
docstrings, making it ideal for code search benchmarking.

Dataset source: https://huggingface.co/datasets/code_search_net
"""

from dataclasses import dataclass, field
from typing import Iterator, Optional, Dict, Any, Tuple, List
import hashlib
import os
from pathlib import Path

# Import datasets library - will be mocked in unit tests
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # Will fail gracefully in tests


# Supported languages in CodeSearchNet
SUPPORTED_LANGUAGES = frozenset({"python", "javascript", "java", "go", "ruby", "php"})

# Supported splits
SUPPORTED_SPLITS = frozenset({"train", "valid", "test"})


def _generate_sample_id(code: str, func_name: str, url: str) -> str:
    """Generate a unique ID for a sample based on its content."""
    content = f"{code}:{func_name}:{url}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _extract_query_from_docstring(docstring: str) -> str:
    """
    Extract a search query from a docstring.

    Uses the first sentence/line as the query, which is typically
    the summary description of the function.

    Args:
        docstring: Full docstring text.

    Returns:
        First line/sentence suitable for use as a search query.
    """
    if not docstring or not docstring.strip():
        return ""

    # Get first line
    first_line = docstring.strip().split("\n")[0].strip()

    # If first line ends with a period, take up to that
    if ". " in first_line:
        first_line = first_line.split(". ")[0] + "."

    return first_line


@dataclass
class CodeSearchNetSample:
    """
    A single sample from the CodeSearchNet dataset.

    Attributes:
        query: Natural language query (derived from docstring).
        code: The function code.
        docstring: Full docstring of the function.
        func_name: Name of the function.
        url: GitHub URL for provenance.
        id: Unique identifier for this sample.
    """

    query: str
    code: str
    docstring: str
    func_name: str
    url: str
    id: str = field(default="")

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = _generate_sample_id(self.code, self.func_name, self.url)


class CodeSearchNetLoader:
    """
    Loader for the CodeSearchNet dataset.

    Provides streaming/lazy loading of the dataset for memory efficiency.
    Supports filtering, caching, and various access patterns for benchmarking.

    Example:
        loader = CodeSearchNetLoader(language="python", split="test")
        for sample in loader.load(limit=1000):
            print(f"Query: {sample.query}")
            print(f"Code: {sample.code[:100]}...")

    Attributes:
        language: Programming language subset (default: "python").
        split: Dataset split - train/valid/test (default: "test").
        cache_dir: Directory for caching downloaded data.
        min_code_length: Minimum code length filter (default: 0).
        deduplicate: Whether to deduplicate by query (default: False).
    """

    def __init__(
        self,
        language: str = "python",
        split: str = "test",
        cache_dir: Optional[str] = None,
        min_code_length: int = 0,
        deduplicate: bool = False,
    ):
        """
        Initialize the CodeSearchNet loader.

        Args:
            language: Programming language to load (python, javascript, etc.).
            split: Dataset split (train, valid, test).
            cache_dir: Directory to cache downloaded data.
            min_code_length: Filter samples with code shorter than this.
            deduplicate: Remove samples with duplicate queries.

        Raises:
            ValueError: If language or split is invalid.
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
            )

        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split: {split}. "
                f"Supported: {', '.join(sorted(SUPPORTED_SPLITS))}"
            )

        self.language = language
        self.split = split
        self.min_code_length = min_code_length
        self.deduplicate = deduplicate

        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "codesearchnet"
            )
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Lazy-loaded dataset
        self._dataset = None
        self._size = None

    def _get_dataset(self):
        """Load the dataset (lazy loading)."""
        if self._dataset is None:
            if load_dataset is None:
                raise ImportError(
                    "The 'datasets' library is required. "
                    "Install it with: pip install datasets"
                )
            # Use claudios/code_search_net parquet version since original
            # code_search_net dataset script format is no longer supported
            self._dataset = load_dataset(
                "claudios/code_search_net",
                self.language,
                cache_dir=self.cache_dir,
            )
        return self._dataset

    def load(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Iterator[CodeSearchNetSample]:
        """
        Load samples from the dataset.

        This is a generator that yields samples lazily for memory efficiency.

        Args:
            limit: Maximum number of samples to yield.
            offset: Number of samples to skip at the beginning.

        Yields:
            CodeSearchNetSample objects.
        """
        dataset = self._get_dataset()
        split_data = dataset[self.split]

        seen_queries = set() if self.deduplicate else None
        count = 0
        skipped = 0

        for item in split_data:
            # Handle offset
            if skipped < offset:
                skipped += 1
                continue

            # Extract fields with fallbacks
            docstring = item.get("func_documentation_string", "") or ""
            code = item.get("func_code_string", "") or ""
            func_name = item.get("func_name", "") or ""
            url = item.get("url", "") or ""

            # Apply min_code_length filter
            if len(code) < self.min_code_length:
                continue

            # Extract query from docstring
            query = _extract_query_from_docstring(docstring)

            # Skip samples with empty query and empty docstring
            if not query and not docstring:
                # Use func_name as fallback if no docstring
                if func_name:
                    query = func_name.replace("_", " ")
                else:
                    continue  # Skip completely empty samples

            # Handle deduplication
            if self.deduplicate and seen_queries is not None:
                if query in seen_queries:
                    continue
                seen_queries.add(query)

            sample = CodeSearchNetSample(
                query=query,
                code=code,
                docstring=docstring,
                func_name=func_name,
                url=url,
            )

            yield sample
            count += 1

            if limit is not None and count >= limit:
                break

    def get_size(self) -> int:
        """
        Get the total number of samples in the dataset split.

        Returns:
            Number of samples.
        """
        if self._size is None:
            dataset = self._get_dataset()
            self._size = len(dataset[self.split])
        return self._size

    def info(self) -> Dict[str, Any]:
        """
        Get metadata about the dataset.

        Returns:
            Dictionary with dataset information.
        """
        return {
            "source": "codesearchnet",
            "language": self.language,
            "split": self.split,
            "cache_dir": self.cache_dir,
            "min_code_length": self.min_code_length,
            "deduplicate": self.deduplicate,
        }

    def get_benchmark_pairs(
        self,
        limit: Optional[int] = None,
    ) -> Iterator[Tuple[str, str]]:
        """
        Get (query, code) pairs for benchmarking.

        This is the primary interface for retrieval benchmarks.
        Each pair represents a query and its ground-truth relevant document.

        Args:
            limit: Maximum number of pairs to yield.

        Yields:
            Tuples of (query, code).
        """
        for sample in self.load(limit=limit):
            yield (sample.query, sample.code)

    def get_queries(self, limit: Optional[int] = None) -> Iterator[str]:
        """
        Get just the queries for benchmarking.

        Args:
            limit: Maximum number of queries to yield.

        Yields:
            Query strings.
        """
        for sample in self.load(limit=limit):
            yield sample.query

    def get_documents(self, limit: Optional[int] = None) -> Iterator[str]:
        """
        Get just the code documents for indexing.

        Args:
            limit: Maximum number of documents to yield.

        Yields:
            Code strings.
        """
        for sample in self.load(limit=limit):
            yield sample.code
