"""
Health check endpoints for Kubernetes/container orchestration probes.

Provides:
- /health/live: Liveness probe - is the process running?
- /health/ready: Readiness probe - is the app ready to serve traffic?
- /health/deps: Dependency health - status of all external dependencies
"""
import os
from typing import Any

from fastapi import APIRouter, Response, status

router = APIRouter(prefix="/health", tags=["health"])


def check_postgres_health() -> dict[str, Any]:
    """Check PostgreSQL connection health."""
    try:
        from app.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_neo4j_health() -> dict[str, Any]:
    """Check Neo4j connection health."""
    try:
        from app.graph.neo4j_client import get_neo4j_driver
        driver = get_neo4j_driver()
        if driver is None:
            return {"status": "unhealthy", "error": "Driver not configured"}
        with driver.session() as session:
            session.run("RETURN 1")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_opensearch_health() -> dict[str, Any]:
    """Check OpenSearch connection health."""
    try:
        from opensearchpy import OpenSearch
        hosts = os.getenv("OPENSEARCH_HOSTS", "localhost:9200")
        client = OpenSearch(
            hosts=[{"host": hosts.split(":")[0], "port": int(hosts.split(":")[1]) if ":" in hosts else 9200}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            timeout=5,
        )
        if client.ping():
            return {"status": "healthy"}
        return {"status": "unhealthy", "error": "Ping failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_qdrant_health() -> dict[str, Any]:
    """Check Qdrant connection health."""
    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        client = QdrantClient(host=host, port=port, timeout=5)
        # Try to list collections as a health check
        client.get_collections()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def check_valkey_health() -> dict[str, Any]:
    """Check Valkey (Redis) connection health."""
    try:
        import redis
        host = os.getenv("VALKEY_HOST", "localhost")
        port = int(os.getenv("VALKEY_PORT", "6379"))
        client = redis.Redis(host=host, port=port, socket_timeout=5)
        if client.ping():
            return {"status": "healthy"}
        return {"status": "unhealthy", "error": "Ping failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


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
    }

    # Determine overall status
    all_healthy = all(dep["status"] == "healthy" for dep in deps.values())

    if all_healthy:
        return {"status": "ok", "dependencies": deps}
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "dependencies": deps}


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
    """
    deps = {
        "postgres": check_postgres_health(),
        "neo4j": check_neo4j_health(),
        "opensearch": check_opensearch_health(),
        "qdrant": check_qdrant_health(),
        "valkey": check_valkey_health(),
    }

    # Determine overall status
    healthy_count = sum(1 for dep in deps.values() if dep["status"] == "healthy")
    total = len(deps)

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
        "healthy_count": healthy_count,
        "total_count": total,
        "dependencies": deps,
    }
