"""Feedback module for retrieval quality improvement.

This module provides:
- FeedbackEvent: Dataclass for feedback events (implicit/explicit)
- FeedbackStore: Append-only storage for feedback events
- ProvideFeedbackTool: MCP tool for explicit user feedback
- ABTestingFramework: A/B testing for retrieval experiments
- RRFWeightOptimizer: Optimize RRF weights from feedback data
- RetrievalInstrumentation: Per-stage latency and evaluation harness
"""

__all__ = [
    # Core types
    "FeedbackEvent",
    "FeedbackOutcome",
    "FeedbackType",
    "FeedbackStore",
    "InMemoryFeedbackStore",
    # MCP tool
    "ProvideFeedbackInput",
    "ProvideFeedbackOutput",
    "ProvideFeedbackConfig",
    "ProvideFeedbackTool",
    "ProvideFeedbackError",
    "create_provide_feedback_tool",
    # A/B testing
    "Experiment",
    "ExperimentVariant",
    "ExperimentStatus",
    "ExperimentConfig",
    "VariantAssignment",
    "GuardrailConfig",
    "GuardrailResult",
    "ABTestingFramework",
    # RRF optimizer
    "RRFWeights",
    "WeightProposal",
    "OptimizationConfig",
    "OptimizationResult",
    "RRFWeightOptimizer",
    # Instrumentation
    "StageType",
    "StageLatency",
    "RetrievalMetrics",
    "QueryExecution",
    "MetricConfig",
    "EvaluationResult",
    "EvaluationHarness",
    "RetrievalInstrumentation",
]

# Module mappings for lazy imports
_MODULE_MAP = {
    # events
    "FeedbackEvent": "events",
    "FeedbackOutcome": "events",
    "FeedbackType": "events",
    # store
    "FeedbackStore": "store",
    "InMemoryFeedbackStore": "store",
    # tool
    "ProvideFeedbackInput": "provide_feedback",
    "ProvideFeedbackOutput": "provide_feedback",
    "ProvideFeedbackConfig": "provide_feedback",
    "ProvideFeedbackTool": "provide_feedback",
    "ProvideFeedbackError": "provide_feedback",
    "create_provide_feedback_tool": "provide_feedback",
    # ab_testing
    "Experiment": "ab_testing",
    "ExperimentVariant": "ab_testing",
    "ExperimentStatus": "ab_testing",
    "ExperimentConfig": "ab_testing",
    "VariantAssignment": "ab_testing",
    "GuardrailConfig": "ab_testing",
    "GuardrailResult": "ab_testing",
    "ABTestingFramework": "ab_testing",
    # optimizer
    "RRFWeights": "optimizer",
    "WeightProposal": "optimizer",
    "OptimizationConfig": "optimizer",
    "OptimizationResult": "optimizer",
    "RRFWeightOptimizer": "optimizer",
    # instrumentation
    "StageType": "instrumentation",
    "StageLatency": "instrumentation",
    "RetrievalMetrics": "instrumentation",
    "QueryExecution": "instrumentation",
    "MetricConfig": "instrumentation",
    "EvaluationResult": "instrumentation",
    "EvaluationHarness": "instrumentation",
    "RetrievalInstrumentation": "instrumentation",
}


def __getattr__(name):
    """Lazy import attributes."""
    if name in _MODULE_MAP:
        import importlib

        module_name = _MODULE_MAP[name]
        module = importlib.import_module(f"openmemory.api.feedback.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
