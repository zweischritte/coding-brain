#!/usr/bin/env python
"""Benchmark execution script.

Run embedding and lexical backend benchmarks, generating reports.

Usage:
    # Run with small sample for testing
    uv run python openmemory/api/benchmarks/run_benchmarks.py --sample-limit 50

    # Run full benchmark
    uv run python openmemory/api/benchmarks/run_benchmarks.py --sample-limit 500

    # Run lexical-only if Ollama not available
    uv run python openmemory/api/benchmarks/run_benchmarks.py --lexical-only

    # Run specific models
    uv run python openmemory/api/benchmarks/run_benchmarks.py --models qwen3-8b nomic

    # Save results to file
    uv run python openmemory/api/benchmarks/run_benchmarks.py --output docs/BENCHMARK-RESULTS.md
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from openmemory.api.benchmarks.runner.benchmark_runner import BenchmarkRunner
from openmemory.api.benchmarks.runner.results import BenchmarkConfig
from openmemory.api.benchmarks.reporter.benchmark_reporter import (
    BenchmarkReporter,
    ReportFormat,
)


# Available models and backends
AVAILABLE_MODELS = ["qwen3-8b", "nomic", "gemini"]
AVAILABLE_BACKENDS = ["tantivy", "opensearch"]

# Models that require Ollama
OLLAMA_MODELS = {"qwen3-8b", "nomic"}


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_available_ollama_models() -> set:
    """Get list of models pulled in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return set()

        models = set()
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                # First column is model name
                model_name = line.split()[0].split(":")[0]
                models.add(model_name)
        return models
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()


def detect_available_models(requested: List[str]) -> tuple[List[str], List[str]]:
    """Detect which requested models are actually available.

    Args:
        requested: List of requested model names

    Returns:
        Tuple of (available_models, skipped_models)
    """
    available = []
    skipped = []

    ollama_running = check_ollama_running()
    ollama_models = get_available_ollama_models() if ollama_running else set()

    for model in requested:
        if model in OLLAMA_MODELS:
            if not ollama_running:
                skipped.append(f"{model} (Ollama not running)")
                continue

            # Map our model names to Ollama model names
            ollama_name_map = {
                "qwen3-8b": "qwen3-embedding",
                "nomic": "nomic-embed-text",
            }
            ollama_name = ollama_name_map.get(model, model)

            if ollama_name not in ollama_models:
                skipped.append(f"{model} (model not pulled: {ollama_name})")
                continue

            available.append(model)

        elif model == "gemini":
            # Gemini requires API key
            import os

            if not os.environ.get("GOOGLE_API_KEY"):
                skipped.append(f"{model} (GOOGLE_API_KEY not set)")
                continue
            available.append(model)

        else:
            skipped.append(f"{model} (unknown model)")

    return available, skipped


def run_benchmarks(
    sample_limit: int,
    models: List[str],
    backends: List[str],
    num_runs: int = 1,
    output_path: Optional[Path] = None,
    console_output: bool = True,
) -> None:
    """Run benchmarks and generate report.

    Args:
        sample_limit: Maximum number of samples to use
        models: List of embedding model names
        backends: List of lexical backend names
        num_runs: Number of benchmark runs for averaging
        output_path: Optional path to save markdown report
        console_output: Whether to print to console
    """
    print("=" * 60)
    print("Phase 0a Benchmark Runner")
    print("=" * 60)
    print()

    # Detect available models
    available_models, skipped_models = detect_available_models(models)

    if skipped_models:
        print("Skipping unavailable models:")
        for skipped in skipped_models:
            print(f"  - {skipped}")
        print()

    if not available_models and not backends:
        print("ERROR: No models or backends available to benchmark!")
        print()
        print("To run embedding benchmarks:")
        print("  1. Start Ollama: ollama serve")
        print("  2. Pull models: ollama pull qwen3-embedding:8b")
        print("                  ollama pull nomic-embed-text")
        print()
        print("To run lexical-only:")
        print("  uv run python openmemory/api/benchmarks/run_benchmarks.py --lexical-only")
        sys.exit(1)

    if available_models:
        print(f"Running embedding benchmarks with: {', '.join(available_models)}")
    if backends:
        print(f"Running lexical benchmarks with: {', '.join(backends)}")
    print(f"Sample limit: {sample_limit}")
    print(f"Number of runs: {num_runs}")
    print()

    # Create benchmark config
    config = BenchmarkConfig(
        dataset_name="codesearchnet",
        dataset_language="python",
        dataset_split="test",
        sample_limit=sample_limit,
        embedding_models=available_models,
        lexical_backends=backends,
        mrr_k=10,
        ndcg_k=10,
    )

    # Run benchmarks
    runner = BenchmarkRunner(config)

    print("Loading dataset samples...")
    try:
        samples = runner.load_samples()
        print(f"Loaded {len(samples)} samples")
        print()
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # Run benchmarks
    print("Running benchmarks...")
    print("-" * 40)

    try:
        if num_runs > 1:
            result = runner.run_multiple(num_runs)
        else:
            result = runner.run()
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()
    print(f"Benchmark completed in {result.total_duration_seconds:.2f} seconds")
    print()

    # Generate report
    reporter = BenchmarkReporter(result)

    # Console output
    if console_output:
        print(reporter.to_console())
        print()

    # Markdown output
    markdown_report = reporter.to_markdown()

    # Add generation metadata
    metadata_header = f"""<!-- Generated by run_benchmarks.py on {datetime.now().isoformat()} -->
<!-- Sample limit: {sample_limit}, Runs: {num_runs} -->
<!-- Models tested: {', '.join(available_models) if available_models else 'None'} -->
<!-- Backends tested: {', '.join(backends) if backends else 'None'} -->
<!-- Skipped: {', '.join(skipped_models) if skipped_models else 'None'} -->

"""
    markdown_report = metadata_header + markdown_report

    # Save to file if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_report)
        print(f"Report saved to: {output_path}")
        print()

    # Print threshold check
    validation = reporter.validate_thresholds()
    print("-" * 40)
    print("Threshold Validation (MRR >= 0.75, NDCG >= 0.80):")
    print(f"  Models passing MRR: {validation['models_passing_mrr']}/{validation['total_models']}")
    print(f"  Models passing NDCG: {validation['models_passing_ndcg']}/{validation['total_models']}")
    print(f"  Production-ready: {validation['models_production_ready']}/{validation['total_models']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 0a benchmarks for embedding models and lexical backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  %(prog)s --sample-limit 50

  # Full benchmark
  %(prog)s --sample-limit 500

  # Lexical backends only (no Ollama needed)
  %(prog)s --lexical-only

  # Specific models
  %(prog)s --models qwen3-8b nomic --sample-limit 100

  # Save results
  %(prog)s --sample-limit 500 --output docs/BENCHMARK-RESULTS.md
        """,
    )

    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Maximum number of samples to use (default: 100)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=AVAILABLE_MODELS,
        help=f"Embedding models to benchmark (default: {' '.join(AVAILABLE_MODELS)})",
    )

    parser.add_argument(
        "--backends",
        nargs="+",
        default=AVAILABLE_BACKENDS,
        help=f"Lexical backends to benchmark (default: {' '.join(AVAILABLE_BACKENDS)})",
    )

    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Only run lexical backend benchmarks (skip embedding models)",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs to average (default: 1)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Path to save markdown report (e.g., docs/BENCHMARK-RESULTS.md)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress console output (only show errors and summary)",
    )

    args = parser.parse_args()

    # Handle lexical-only mode
    models = [] if args.lexical_only else args.models
    backends = args.backends

    run_benchmarks(
        sample_limit=args.sample_limit,
        models=models,
        backends=backends,
        num_runs=args.num_runs,
        output_path=args.output,
        console_output=not args.quiet,
    )


if __name__ == "__main__":
    main()
