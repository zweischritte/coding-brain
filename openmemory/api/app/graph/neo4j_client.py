"""
Neo4j client wrapper for OpenMemory.

Provides:
- Driver lifecycle management (singleton pattern)
- Session creation with optional database parameter
- Environment variable parsing for connection details
- Graceful handling of missing Neo4j configuration

Environment Variables:
- NEO4J_URL: Connection URL (e.g., bolt://localhost:7687)
- NEO4J_USERNAME: Username for authentication
- NEO4J_PASSWORD: Password for authentication
- NEO4J_DATABASE: Database name (optional, defaults to 'neo4j')
"""

import atexit
import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

# Neo4j driver import with graceful fallback
try:
    from neo4j import GraphDatabase, Driver
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    Driver = None
    ServiceUnavailable = Exception
    AuthError = Exception

logger = logging.getLogger(__name__)

# Global driver instance (singleton)
_neo4j_driver: Optional["Driver"] = None
_driver_initialized: bool = False

# Health check state (circuit breaker pattern)
_neo4j_healthy: bool = True
_neo4j_last_health_check: float = 0.0
_HEALTH_CHECK_INTERVAL: float = 60.0  # seconds between health checks
_HEALTH_CHECK_TIMEOUT: float = 0.5    # timeout for health check query


def _parse_env_value(value: Optional[str]) -> Optional[str]:
    """
    Parse a configuration value that may be an env: reference.

    Args:
        value: The value to parse, which may be "env:VAR_NAME" or a literal value

    Returns:
        The resolved value, or None if not found
    """
    if value is None:
        return None
    if isinstance(value, str) and value.startswith("env:"):
        env_var = value[4:]  # Strip "env:" prefix
        return os.environ.get(env_var)
    return value


def get_neo4j_config() -> dict:
    """
    Get Neo4j configuration from environment variables.

    Returns:
        Dict with url, username, password, database keys
    """
    return {
        "url": os.environ.get("NEO4J_URL"),
        "username": os.environ.get("NEO4J_USERNAME"),
        "password": os.environ.get("NEO4J_PASSWORD"),
        "database": os.environ.get("NEO4J_DATABASE", "neo4j"),
    }


def is_neo4j_configured() -> bool:
    """
    Check if Neo4j is configured via environment variables.

    Returns:
        True if all required Neo4j environment variables are set
    """
    config = get_neo4j_config()
    return all([
        config["url"],
        config["username"],
        config["password"],
    ])


def get_neo4j_driver() -> Optional["Driver"]:
    """
    Get or create the Neo4j driver singleton.

    Returns:
        Neo4j Driver instance, or None if not configured or unavailable

    Note:
        The driver is created once and reused. It is automatically closed
        when the process exits via atexit registration.
    """
    global _neo4j_driver, _driver_initialized

    if not NEO4J_AVAILABLE:
        logger.debug("Neo4j package not installed")
        return None

    if _driver_initialized:
        return _neo4j_driver

    _driver_initialized = True

    if not is_neo4j_configured():
        logger.debug("Neo4j not configured (missing environment variables)")
        return None

    config = get_neo4j_config()

    try:
        _neo4j_driver = GraphDatabase.driver(
            config["url"],
            auth=(config["username"], config["password"]),
        )
        # Verify connectivity
        _neo4j_driver.verify_connectivity()
        logger.info(f"Neo4j driver connected to {config['url']}")

        # Register cleanup on exit
        atexit.register(close_neo4j_driver)

        return _neo4j_driver

    except AuthError as e:
        logger.error(f"Neo4j authentication failed: {e}")
        _neo4j_driver = None
        return None
    except ServiceUnavailable as e:
        logger.warning(f"Neo4j service unavailable: {e}")
        _neo4j_driver = None
        return None
    except Exception as e:
        logger.error(f"Failed to create Neo4j driver: {e}")
        _neo4j_driver = None
        return None


def close_neo4j_driver():
    """
    Close the Neo4j driver if it exists.

    Called automatically at process exit via atexit registration.
    """
    global _neo4j_driver, _driver_initialized

    if _neo4j_driver is not None:
        try:
            _neo4j_driver.close()
            logger.info("Neo4j driver closed")
        except Exception as e:
            logger.warning(f"Error closing Neo4j driver: {e}")
        finally:
            _neo4j_driver = None
            _driver_initialized = False


def reset_neo4j_driver():
    """
    Reset the driver state (for testing or config changes).

    Forces the next get_neo4j_driver() call to create a new connection.
    """
    global _neo4j_driver, _driver_initialized
    close_neo4j_driver()
    _driver_initialized = False


@contextmanager
def get_neo4j_session(database: Optional[str] = None):
    """
    Context manager for Neo4j sessions.

    Args:
        database: Optional database name. If not provided, uses NEO4J_DATABASE
                 environment variable or defaults to 'neo4j'.

    Yields:
        Neo4j Session object

    Raises:
        RuntimeError: If Neo4j driver is not available

    Example:
        with get_neo4j_session() as session:
            result = session.run("MATCH (n) RETURN n LIMIT 10")
    """
    driver = get_neo4j_driver()

    if driver is None:
        raise RuntimeError("Neo4j driver not available")

    db = database or os.environ.get("NEO4J_DATABASE", "neo4j")

    session = driver.session(database=db)
    try:
        yield session
    finally:
        session.close()


def execute_with_retry(
    cypher: str,
    parameters: dict = None,
    database: Optional[str] = None,
    max_retries: int = 3,
):
    """
    Execute a Cypher query with retry logic.

    Args:
        cypher: Cypher query string
        parameters: Query parameters
        database: Optional database name
        max_retries: Maximum retry attempts

    Returns:
        Query result records as list of dicts

    Raises:
        RuntimeError: If Neo4j is not available
        Exception: If all retries fail
    """
    driver = get_neo4j_driver()
    if driver is None:
        raise RuntimeError("Neo4j driver not available")

    last_error = None

    for attempt in range(max_retries):
        try:
            with get_neo4j_session(database) as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except ServiceUnavailable as e:
            last_error = e
            logger.warning(f"Neo4j unavailable (attempt {attempt + 1}/{max_retries}): {e}")
            # Reset driver for next attempt
            reset_neo4j_driver()
        except Exception as e:
            last_error = e
            logger.error(f"Neo4j query failed: {e}")
            break

    raise last_error


def is_neo4j_healthy() -> bool:
    """
    Check Neo4j health with circuit breaker pattern.

    Uses a cached health status to avoid excessive health checks.
    Only performs actual health check if the interval has passed.

    The circuit breaker prevents cascading failures:
    - If Neo4j is unavailable, operations skip graph queries
    - Periodic health checks allow recovery when Neo4j comes back

    Returns:
        True if Neo4j is healthy and available, False otherwise

    Performance:
        - Returns cached result in ~0.001ms
        - Health check takes up to 500ms (with timeout)
    """
    global _neo4j_healthy, _neo4j_last_health_check

    if not NEO4J_AVAILABLE:
        return False

    if not is_neo4j_configured():
        return False

    now = time.time()

    # Return cached result if within check interval
    if now - _neo4j_last_health_check < _HEALTH_CHECK_INTERVAL:
        return _neo4j_healthy

    # Perform health check
    _neo4j_last_health_check = now

    driver = get_neo4j_driver()
    if driver is None:
        _neo4j_healthy = False
        return False

    try:
        # Quick connectivity check with timeout
        with driver.session() as session:
            # Simple query that exercises the connection
            result = session.run(
                "RETURN 1 AS health",
                timeout=_HEALTH_CHECK_TIMEOUT
            )
            record = result.single()
            _neo4j_healthy = record is not None and record["health"] == 1

    except ServiceUnavailable:
        logger.warning("Neo4j health check failed: service unavailable")
        _neo4j_healthy = False
    except Exception as e:
        logger.warning(f"Neo4j health check failed: {e}")
        _neo4j_healthy = False

    if not _neo4j_healthy:
        logger.info("Neo4j marked as unhealthy, will retry in 60s")

    return _neo4j_healthy


def mark_neo4j_unhealthy():
    """
    Manually mark Neo4j as unhealthy.

    Called when a graph operation fails, to prevent further
    attempts until the next health check.

    This is useful for fail-fast behavior during search operations.
    """
    global _neo4j_healthy
    _neo4j_healthy = False
    logger.debug("Neo4j marked unhealthy due to operation failure")


def get_health_status() -> dict:
    """
    Get detailed health status for diagnostics.

    Returns:
        Dict with health status details:
        - healthy: Current health status
        - last_check: Seconds since last health check
        - next_check: Seconds until next health check
        - driver_initialized: Whether driver is initialized
        - configured: Whether Neo4j env vars are set
    """
    now = time.time()
    time_since_check = now - _neo4j_last_health_check if _neo4j_last_health_check > 0 else None

    return {
        "healthy": _neo4j_healthy,
        "last_check_seconds_ago": round(time_since_check, 1) if time_since_check else None,
        "next_check_in_seconds": round(
            max(0, _HEALTH_CHECK_INTERVAL - time_since_check), 1
        ) if time_since_check else 0,
        "driver_initialized": _driver_initialized,
        "configured": is_neo4j_configured(),
        "package_available": NEO4J_AVAILABLE,
    }
