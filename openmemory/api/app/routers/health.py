"""
Health check endpoints for Kubernetes/container orchestration probes.

Provides:
- /health/live: Liveness probe - is the process running?
- /health/ready: Readiness probe - is the app ready to serve traffic?
- /health/deps: Dependency health - status of all external dependencies

Phase 6 Enhancements:
- Async health checks with timeout per dependency
- Latency measurement for each dependency check
- Timestamp in all health responses
- Normalized status values: 'healthy' or 'unavailable'
"""
import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Response, status

router = APIRouter(prefix="/health", tags=["health"])

# Timeout for individual dependency checks (in seconds)
HEALTH_CHECK_TIMEOUT = 2.0


def _measure_health_check(check_fn) -> dict[str, Any]:
    """Wrapper to measure latency of a health check function."""
    start = time.perf_counter()
    try:
        result = check_fn()
        latency_ms = (time.perf_counter() - start) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        return result
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {"status": "unavailable", "error": str(e), "latency_ms": round(latency_ms, 2)}


def _check_postgres() -> dict[str, Any]:
    """Check PostgreSQL connection health (internal)."""
    from app.database import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "healthy"}


def _check_neo4j() -> dict[str, Any]:
    """Check Neo4j connection health (internal)."""
    from app.graph.neo4j_client import get_neo4j_driver
    driver = get_neo4j_driver()
    if driver is None:
        return {"status": "unavailable", "error": "Driver not configured"}
    with driver.session() as session:
        session.run("RETURN 1")
    return {"status": "healthy"}


def _check_opensearch() -> dict[str, Any]:
    """Check OpenSearch connection health (internal)."""
    from opensearchpy import OpenSearch
    hosts = os.getenv("OPENSEARCH_HOSTS", "localhost:9200")
    client = OpenSearch(
        hosts=[{"host": hosts.split(":")[0], "port": int(hosts.split(":")[1]) if ":" in hosts else 9200}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        timeout=2,  # Reduced timeout for health checks
    )
    if client.ping():
        return {"status": "healthy"}
    return {"status": "unavailable", "error": "Ping failed"}


def _check_qdrant() -> dict[str, Any]:
    """Check Qdrant connection health (internal)."""
    from qdrant_client import QdrantClient
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=host, port=port, timeout=2, check_compatibility=False)
    # Try to list collections as a health check
    client.get_collections()
    return {"status": "healthy"}


def _check_valkey() -> dict[str, Any]:
    """Check Valkey (Redis) connection health (internal)."""
    import redis
    host = os.getenv("VALKEY_HOST", "localhost")
    port = int(os.getenv("VALKEY_PORT", "6379"))
    client = redis.Redis(host=host, port=port, socket_timeout=2)
    if client.ping():
        return {"status": "healthy"}
    return {"status": "unavailable", "error": "Ping failed"}


def _get_ollama_base_url() -> str:
    env_base_url = os.getenv("OLLAMA_BASE_URL")
    if env_base_url:
        return env_base_url
    env_host = os.getenv("OLLAMA_HOST")
    if env_host:
        if env_host.startswith(("http://", "https://")):
            return env_host
        if ":" in env_host:
            return f"http://{env_host}"
        return f"http://{env_host}:11434"
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://localhost:11434"


def _check_ollama() -> dict[str, Any]:
    """Check Ollama connection health (internal)."""
    base_url = _get_ollama_base_url().rstrip("/")
    with httpx.Client(timeout=2.0) as client:
        response = client.get(f"{base_url}/api/tags")
        response.raise_for_status()
    return {"status": "healthy"}


def check_postgres_health() -> dict[str, Any]:
    """Check PostgreSQL connection health with latency measurement."""
    return _measure_health_check(_check_postgres)


def check_neo4j_health() -> dict[str, Any]:
    """Check Neo4j connection health with latency measurement."""
    return _measure_health_check(_check_neo4j)


def check_opensearch_health() -> dict[str, Any]:
    """Check OpenSearch connection health with latency measurement."""
    return _measure_health_check(_check_opensearch)


def check_qdrant_health() -> dict[str, Any]:
    """Check Qdrant connection health with latency measurement."""
    return _measure_health_check(_check_qdrant)


def check_valkey_health() -> dict[str, Any]:
    """Check Valkey (Redis) connection health with latency measurement."""
    return _measure_health_check(_check_valkey)


def check_ollama_health() -> dict[str, Any]:
    """Check Ollama connection health with latency measurement."""
    return _measure_health_check(_check_ollama)

@router.get("/live")
async def liveness():
    """
    Liveness probe endpoint.

    Returns 200 if the application process is running and responsive.
    Used by container orchestrators to determine if the container should be restarted.
    """
    return {"status": "ok"}


@router.get("/ready")
async def readiness(response: Response):
    """
    Readiness probe endpoint.

    Returns 200 if the application is ready to serve traffic.
    Returns 503 if any critical dependency is unavailable.
    """
    # Check critical dependencies
    deps = {
        "postgres": check_postgres_health(),
        "neo4j": check_neo4j_health(),
        "ollama": check_ollama_health(),
    }

    # Determine overall status
    all_healthy = all(dep["status"] == "healthy" for dep in deps.values())

    timestamp = datetime.now(timezone.utc).isoformat()

    if all_healthy:
        return {"status": "ok", "timestamp": timestamp, "dependencies": deps}
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "timestamp": timestamp, "dependencies": deps}


@router.get("/deps")
async def dependency_health(response: Response):
    """
    Detailed dependency health endpoint.

    Returns status of all external dependencies:
    - postgres: PostgreSQL database
    - neo4j: Neo4j graph database
    - opensearch: OpenSearch for full-text search
    - qdrant: Qdrant vector database
    - valkey: Valkey (Redis-compatible) cache
    - ollama: Ollama embedder service
    """
    deps = {
        "postgres": check_postgres_health(),
        "neo4j": check_neo4j_health(),
        "opensearch": check_opensearch_health(),
        "qdrant": check_qdrant_health(),
        "valkey": check_valkey_health(),
        "ollama": check_ollama_health(),
    }

    # Determine overall status
    healthy_count = sum(1 for dep in deps.values() if dep["status"] == "healthy")
    total = len(deps)

    timestamp = datetime.now(timezone.utc).isoformat()

    if healthy_count == total:
        overall_status = "ok"
    elif healthy_count > 0:
        overall_status = "degraded"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        overall_status = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": overall_status,
        "timestamp": timestamp,
        "healthy_count": healthy_count,
        "total_count": total,
        "dependencies": deps,
    }
