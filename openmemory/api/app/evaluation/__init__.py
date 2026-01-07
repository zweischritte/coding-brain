"""
Simplified Evaluation Framework for Coding Brain RAG Quality.

This module provides:
- Golden dataset loading (30 samples across local, global, code query types)
- Context Precision metric (retrieval quality)
- Answer Relevancy metric (response quality)
- Manual evaluation runner

Usage:
    python -m openmemory.api.app.evaluation.simple_eval
"""

from app.evaluation.simple_eval import (
    GoldenSample,
    EvalResult,
    load_golden_dataset,
    evaluate_context_precision,
    evaluate_answer_relevancy,
    evaluate_sample,
    run_evaluation,
    print_summary,
)

__all__ = [
    "GoldenSample",
    "EvalResult",
    "load_golden_dataset",
    "evaluate_context_precision",
    "evaluate_answer_relevancy",
    "evaluate_sample",
    "run_evaluation",
    "print_summary",
]
