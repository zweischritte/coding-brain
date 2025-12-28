"""
Prometheus Metrics for Phase 6.

Features:
- HTTP request metrics (count, duration)
- Dependency health metrics
- Circuit breaker state metrics
- Custom business metrics support
- /metrics endpoint for Prometheus scraping
"""
import time
from typing import Callable, List, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST, REGISTRY
)

# HTTP Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Dependency health metrics
dependency_health = Gauge(
    "dependency_health",
    "Dependency health status (1=healthy, 0=unhealthy)",
    ["dependency"]
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 0.5=half-open)",
    ["service"]
)

# Business metrics
memories_created_total = Counter(
    "memories_created_total",
    "Total memories created",
    ["org_id", "vault"]
)

search_queries_total = Counter(
    "search_queries_total",
    "Total search queries",
    ["org_id", "search_type"]
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP request metrics.

    Records:
    - Request count by method, endpoint, status code
    - Request duration by method, endpoint
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and record metrics."""
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time

        # Normalize endpoint path (remove IDs, limit cardinality)
        endpoint = self._normalize_path(request.url.path)

        # Record metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code)
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(duration)

        return response

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path to reduce cardinality.

        Replaces UUIDs and numeric IDs with placeholders.
        """
        import re

        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
            flags=re.IGNORECASE
        )

        # Replace numeric IDs
        path = re.sub(r"/\d+(/|$)", "/{id}\\1", path)

        return path


def create_metrics_app() -> FastAPI:
    """
    Create a FastAPI app with /metrics endpoint.

    Returns:
        FastAPI app with metrics endpoint
    """
    app = FastAPI(title="Metrics", docs_url=None, redoc_url=None)

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )

    return app


def register_counter(
    name: str,
    description: str,
    labels: List[str],
) -> Counter:
    """
    Register a custom counter metric.

    Args:
        name: Metric name
        description: Metric description
        labels: Label names

    Returns:
        Registered Counter
    """
    return Counter(name, description, labels)


def register_histogram(
    name: str,
    description: str,
    labels: List[str],
    buckets: Optional[List[float]] = None,
) -> Histogram:
    """
    Register a custom histogram metric.

    Args:
        name: Metric name
        description: Metric description
        labels: Label names
        buckets: Histogram buckets (optional)

    Returns:
        Registered Histogram
    """
    if buckets:
        return Histogram(name, description, labels, buckets=buckets)
    return Histogram(name, description, labels)


def register_gauge(
    name: str,
    description: str,
    labels: List[str],
) -> Gauge:
    """
    Register a custom gauge metric.

    Args:
        name: Metric name
        description: Metric description
        labels: Label names

    Returns:
        Registered Gauge
    """
    return Gauge(name, description, labels)


def update_dependency_health(dependency: str, healthy: bool) -> None:
    """
    Update dependency health metric.

    Args:
        dependency: Dependency name
        healthy: Whether the dependency is healthy
    """
    dependency_health.labels(dependency=dependency).set(1.0 if healthy else 0.0)


def update_circuit_breaker_state(service: str, state: str) -> None:
    """
    Update circuit breaker state metric.

    Args:
        service: Service name
        state: Circuit state ("closed", "open", "half_open")
    """
    state_values = {
        "closed": 0.0,
        "open": 1.0,
        "half_open": 0.5,
    }
    value = state_values.get(state, 0.0)
    circuit_breaker_state.labels(service=service).set(value)
