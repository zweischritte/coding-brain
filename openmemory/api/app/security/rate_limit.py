"""
Rate Limiting with Token Bucket Algorithm for Phase 6.

Features:
- Token bucket rate limiting algorithm
- Configurable requests per minute and burst size
- Per-endpoint rate limit configuration
- FastAPI middleware for automatic enforcement
- X-RateLimit headers on responses
"""
import time
import threading
from typing import Dict, Tuple, Any, Optional, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# Endpoint rate limit configurations
ENDPOINT_LIMITS: Dict[str, Dict[str, int]] = {
    "/v1/memories": {
        "requests_per_minute": 60,
        "burst_size": 20,
    },
    "/v1/search": {
        "requests_per_minute": 30,
        "burst_size": 10,
    },
    "/v1/graph": {
        "requests_per_minute": 20,
        "burst_size": 5,
    },
    "default": {
        "requests_per_minute": 100,
        "burst_size": 30,
    },
}


class RateLimiter:
    """
    Token bucket rate limiter implementation.

    The token bucket algorithm allows for:
    - Steady rate limiting (requests_per_minute)
    - Burst handling (burst_size)
    - Smooth refill over time

    Usage:
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        allowed, info = limiter.check("user_123")
        if not allowed:
            # Rate limited
            print(f"Retry after {info['reset']} seconds")
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum sustained request rate
            burst_size: Maximum burst capacity
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second

        # Token buckets per key
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a request is allowed.

        Args:
            key: Identifier for rate limiting (e.g., user_id, ip)

        Returns:
            Tuple of (allowed, info) where info contains:
            - limit: Maximum tokens
            - remaining: Tokens remaining
            - reset: Time until next refill (Unix timestamp)
        """
        with self._lock:
            now = time.time()
            bucket = self._get_or_create_bucket(key, now)

            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_refill"]
            bucket["tokens"] = min(
                self.burst_size,
                bucket["tokens"] + elapsed * self.refill_rate
            )
            bucket["last_refill"] = now

            # Calculate reset time (when next token will be available)
            if bucket["tokens"] < 1:
                time_until_token = (1 - bucket["tokens"]) / self.refill_rate
                reset_time = now + time_until_token
            else:
                reset_time = now

            info = {
                "limit": self.burst_size,
                "remaining": max(0, int(bucket["tokens"]) - 1),
                "reset": int(reset_time),
            }

            # Check if request allowed
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                info["remaining"] = max(0, int(bucket["tokens"]))
                return True, info
            else:
                return False, info

    def _get_or_create_bucket(self, key: str, now: float) -> Dict[str, float]:
        """Get or create a token bucket for a key."""
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": float(self.burst_size),
                "last_refill": now,
            }
        return self._buckets[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Adds rate limit headers and returns 429 when limit exceeded.
    """

    def __init__(
        self,
        app: FastAPI,
        limiter: Optional[RateLimiter] = None,
        key_func: Optional[Callable[[Request], str]] = None,
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            limiter: RateLimiter instance (uses default if not provided)
            key_func: Function to extract rate limit key from request
        """
        super().__init__(app)
        self.limiter = limiter or RateLimiter()
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, request: Request) -> str:
        """Default key function using client IP."""
        # Get client IP from X-Forwarded-For or direct connection
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health endpoints
        if request.url.path.startswith("/health"):
            return await call_next(request)

        # Get rate limit key
        key = self.key_func(request)

        # Check rate limit
        allowed, info = self.limiter.check(key)

        if not allowed:
            # Return 429 Too Many Requests
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "Rate limit exceeded",
                    "retry_after": info["reset"] - int(time.time()),
                },
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": str(max(1, info["reset"] - int(time.time()))),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response


def get_limiter_for_endpoint(path: str) -> RateLimiter:
    """
    Get a rate limiter configured for a specific endpoint.

    Args:
        path: Request path

    Returns:
        Configured RateLimiter
    """
    # Find matching endpoint config
    for endpoint, config in ENDPOINT_LIMITS.items():
        if endpoint != "default" and path.startswith(endpoint):
            return RateLimiter(
                requests_per_minute=config["requests_per_minute"],
                burst_size=config["burst_size"],
            )

    # Use default
    default_config = ENDPOINT_LIMITS["default"]
    return RateLimiter(
        requests_per_minute=default_config["requests_per_minute"],
        burst_size=default_config["burst_size"],
    )
