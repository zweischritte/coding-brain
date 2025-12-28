"""
Circuit Breaker Implementation for Phase 6.

Features:
- State machine: closed -> open -> half-open -> closed
- Configurable failure threshold and recovery timeout
- Per-service circuit breaker instances
- Degraded mode response support
"""
import time
import threading
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
from pydantic import BaseModel


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and calls are blocked."""

    def __init__(self, service: str, message: str = "Circuit is open"):
        self.service = service
        self.message = message
        super().__init__(f"{service}: {message}")


class DegradedResponse(BaseModel):
    """Response schema for degraded mode operations."""

    status: Literal["degraded"] = "degraded"
    message: str
    available_sources: List[str]
    unavailable_sources: List[str]
    partial_results: Optional[Any] = None


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_exceptions: tuple = (Exception,)


# Circuit breaker configurations per service
CIRCUIT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "neo4j": {
        "failure_threshold": 5,
        "recovery_timeout": 30,
    },
    "qdrant": {
        "failure_threshold": 5,
        "recovery_timeout": 30,
    },
    "opensearch": {
        "failure_threshold": 5,
        "recovery_timeout": 30,
    },
    "valkey": {
        "failure_threshold": 5,
        "recovery_timeout": 30,
    },
    "reranker": {
        "failure_threshold": 3,
        "recovery_timeout": 60,
    },
}


class ServiceCircuitBreaker:
    """
    Circuit breaker implementation with state machine.

    States:
    - closed: Normal operation, failures are counted
    - open: Circuit is tripped, calls are blocked
    - half_open: Recovery attempt, allows one probe call

    Usage:
        cb = ServiceCircuitBreaker("my_service", failure_threshold=5)
        with cb:
            # Call external service
            result = external_service.call()
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Service name for identification
            failure_threshold: Number of failures before opening
            recovery_timeout: Seconds before attempting recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state: Literal["closed", "open", "half_open"] = "closed"
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state, checking for timeout transitions."""
        with self._lock:
            if self._state == "open" and self._should_attempt_recovery():
                self._state = "half_open"
            return self._state

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery timeout has elapsed."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _force_half_open(self) -> None:
        """Force transition to half-open state (for testing)."""
        with self._lock:
            self._state = "half_open"

    def __enter__(self):
        """Enter circuit breaker context."""
        current_state = self.state  # This checks for timeout transitions

        with self._lock:
            if current_state == "open":
                raise CircuitOpenError(self.name)

            # Allow call in closed or half_open state
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context, recording success or failure."""
        with self._lock:
            if exc_type is None:
                # Success
                self._on_success()
            else:
                # Failure
                self._on_failure()

        # Don't suppress the exception
        return False

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == "half_open":
            # Recovery successful, close the circuit
            self._state = "closed"
            self._failure_count = 0
            self._last_failure_time = None
        elif self._state == "closed":
            # Reset failure count on success in closed state
            self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1

        if self._state == "half_open":
            # Recovery failed, reopen - set failure time to now for fresh timeout
            self._state = "open"
            self._last_failure_time = time.time()
        elif self._state == "closed" and self._failure_count >= self.failure_threshold:
            # Threshold reached, open the circuit
            self._state = "open"
            self._last_failure_time = time.time()


# Registry of circuit breakers
_circuit_registry: Dict[str, ServiceCircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(service: str) -> ServiceCircuitBreaker:
    """
    Get or create a circuit breaker for a service.

    Args:
        service: Service name

    Returns:
        ServiceCircuitBreaker instance
    """
    with _registry_lock:
        if service not in _circuit_registry:
            config = CIRCUIT_CONFIGS.get(service, {})
            _circuit_registry[service] = ServiceCircuitBreaker(
                name=service,
                failure_threshold=config.get("failure_threshold", 5),
                recovery_timeout=config.get("recovery_timeout", 30),
            )
        return _circuit_registry[service]


def get_all_circuit_states() -> Dict[str, str]:
    """
    Get states of all registered circuit breakers.

    Returns:
        Dictionary of service name -> circuit state
    """
    with _registry_lock:
        return {name: cb.state for name, cb in _circuit_registry.items()}
