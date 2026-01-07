"""
Simplified Evaluation Framework for Coding Brain RAG Quality (PRD-08-SIMPLE).

This module provides:
- Golden dataset loading (30 samples across local, global, code query types)
- Context Precision metric (retrieval quality)
- Answer Relevancy metric (response quality via keyword matching)
- Manual evaluation runner with CLI output

Usage:
    python -m openmemory.api.app.evaluation.simple_eval
"""

from dataclasses import dataclass
from typing import Literal, Callable, Awaitable, Any
import json
import time


@dataclass
class GoldenSample:
    """
    A single golden sample for evaluation.

    Attributes:
        id: Unique identifier (e.g., "local_001")
        query: The search query to evaluate
        query_type: Category of query ("local", "global", or "code")
        expected_memory_ids: Memory IDs that should be retrieved (empty if unknown)
        expected_answer_keywords: Keywords that should appear in the answer
    """

    id: str
    query: str
    query_type: Literal["local", "global", "code"]
    expected_memory_ids: list[str]
    expected_answer_keywords: list[str]


@dataclass
class EvalResult:
    """
    Evaluation result for a single sample.

    Attributes:
        sample_id: ID of the evaluated sample
        query_type: The query type (local/global/code)
        context_precision: Precision of retrieved memories (0.0-1.0)
        answer_relevancy: Relevancy of answer based on keywords (0.0-1.0)
        latency_ms: Time taken for the search in milliseconds
    """

    sample_id: str
    query_type: str
    context_precision: float
    answer_relevancy: float
    latency_ms: float


def load_golden_dataset(path: str) -> list[GoldenSample]:
    """
    Load golden samples from a JSON file.

    Args:
        path: Path to the golden dataset JSON file

    Returns:
        List of GoldenSample objects

    Raises:
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(path, "r") as f:
        data = json.load(f)

    return [
        GoldenSample(
            id=sample["id"],
            query=sample["query"],
            query_type=sample["query_type"],
            expected_memory_ids=sample.get("expected_memory_ids", []),
            expected_answer_keywords=sample.get("expected_answer_keywords", []),
        )
        for sample in data["samples"]
    ]


def evaluate_context_precision(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    """
    Calculate context precision: what percentage of retrieved memories are relevant.

    Precision = |intersection(retrieved, expected)| / |retrieved|

    Args:
        retrieved_ids: IDs of memories that were retrieved
        expected_ids: IDs of memories that should have been retrieved

    Returns:
        Precision score (0.0-1.0), or 0.5 if no ground truth available
    """
    if not retrieved_ids:
        return 0.0

    if not expected_ids:
        # No ground truth available, return neutral score
        return 0.5

    hits = len(set(retrieved_ids) & set(expected_ids))
    return hits / len(retrieved_ids)


def evaluate_answer_relevancy(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """
    Calculate answer relevancy based on keyword coverage.

    Relevancy = |keywords found| / |expected keywords|

    Args:
        answer: The answer text to evaluate
        expected_keywords: Keywords that should appear in the answer

    Returns:
        Relevancy score (0.0-1.0), or 0.5 if no keywords to check
    """
    if not expected_keywords or not answer:
        return 0.5

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


# Type alias for search function
SearchFn = Callable[[str, int], Awaitable[dict[str, Any]]]


async def evaluate_sample(
    sample: GoldenSample,
    search_fn: SearchFn,
) -> EvalResult:
    """
    Evaluate a single golden sample.

    Args:
        sample: The golden sample to evaluate
        search_fn: Async function that performs search (query, limit) -> results dict

    Returns:
        EvalResult with precision, relevancy, and latency metrics
    """
    start_time = time.time()
    result = await search_fn(sample.query, 10)
    latency_ms = (time.time() - start_time) * 1000

    results = result.get("results", [])

    # Extract retrieved IDs
    retrieved_ids = [r.get("id") for r in results if r.get("id")]

    # Concatenate memory content for answer relevancy
    answer_parts = [r.get("memory", "") for r in results]
    answer = " ".join(answer_parts)

    # Calculate metrics
    precision = evaluate_context_precision(retrieved_ids, sample.expected_memory_ids)
    relevancy = evaluate_answer_relevancy(answer, sample.expected_answer_keywords)

    return EvalResult(
        sample_id=sample.id,
        query_type=sample.query_type,
        context_precision=precision,
        answer_relevancy=relevancy,
        latency_ms=latency_ms,
    )


async def run_evaluation(
    mock_search: SearchFn,
    dataset_path: str,
) -> dict:
    """
    Run full evaluation on all samples in the golden dataset.

    Args:
        mock_search: Async search function to evaluate
        dataset_path: Path to the golden dataset JSON file

    Returns:
        Summary dict with overall and per-type metrics
    """
    samples = load_golden_dataset(dataset_path)
    results: list[EvalResult] = []

    for sample in samples:
        try:
            result = await evaluate_sample(sample, mock_search)
            results.append(result)
        except Exception as e:
            print(f"Failed {sample.id}: {e}")
            continue

    if not results:
        return {
            "total_samples": 0,
            "overall": {
                "avg_precision": 0.0,
                "avg_relevancy": 0.0,
                "avg_latency_ms": 0.0,
            },
            "by_type": {},
        }

    # Group results by query type
    by_type: dict[str, list[EvalResult]] = {"local": [], "global": [], "code": []}
    for r in results:
        if r.query_type in by_type:
            by_type[r.query_type].append(r)

    # Calculate overall metrics
    summary = {
        "total_samples": len(results),
        "overall": {
            "avg_precision": sum(r.context_precision for r in results) / len(results),
            "avg_relevancy": sum(r.answer_relevancy for r in results) / len(results),
            "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        },
        "by_type": {},
    }

    # Calculate per-type metrics
    for qtype, type_results in by_type.items():
        if type_results:
            summary["by_type"][qtype] = {
                "count": len(type_results),
                "avg_precision": sum(r.context_precision for r in type_results)
                / len(type_results),
                "avg_relevancy": sum(r.answer_relevancy for r in type_results)
                / len(type_results),
                "avg_latency_ms": sum(r.latency_ms for r in type_results)
                / len(type_results),
            }

    return summary


def print_summary(summary: dict) -> None:
    """
    Print evaluation summary to console in a formatted way.

    Args:
        summary: Summary dict from run_evaluation
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {summary['total_samples']}")

    overall = summary["overall"]
    print("\nOverall:")
    print(f"  Context Precision: {overall['avg_precision']:.2%}")
    print(f"  Answer Relevancy:  {overall['avg_relevancy']:.2%}")
    print(f"  Avg Latency:       {overall['avg_latency_ms']:.0f}ms")

    print("\nBy Query Type:")
    for qtype, metrics in summary.get("by_type", {}).items():
        print(f"  {qtype.upper()} (n={metrics['count']}):")
        print(f"    Precision:  {metrics['avg_precision']:.2%}")
        print(f"    Relevancy:  {metrics['avg_relevancy']:.2%}")
        print(f"    Latency:    {metrics['avg_latency_ms']:.0f}ms")

    print("\n" + "=" * 60)


# CLI entry point
if __name__ == "__main__":
    import asyncio
    import os

    # Determine the data path relative to this file
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    dataset_path = os.path.join(data_dir, "golden_30.json")

    # Import the actual search_memory function
    try:
        from app.mcp_server import search_memory as mcp_search

        async def search_wrapper(query: str, limit: int) -> dict:
            """Wrapper to call MCP search_memory and parse JSON result."""
            result_json = await mcp_search(query=query, limit=limit)
            return json.loads(result_json)

        async def main():
            summary = await run_evaluation(search_wrapper, dataset_path)
            print_summary(summary)

        asyncio.run(main())

    except ImportError as e:
        print(f"Could not import search_memory: {e}")
        print("Run this from the openmemory/api directory with the proper environment.")
