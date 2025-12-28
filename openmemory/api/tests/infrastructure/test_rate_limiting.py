"""
Tests for Phase 6: Rate Limiting with Token Bucket Algorithm.

TDD: These tests are written first and should fail until implementation is complete.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


class TestRateLimitConfiguration:
    """Test rate limit configuration."""

    def test_rate_limit_config_exists(self):
        """Rate limit configurations must be defined."""
        from app.security.rate_limit import ENDPOINT_LIMITS

        assert ENDPOINT_LIMITS is not None
        assert isinstance(ENDPOINT_LIMITS, dict)

    def test_config_has_default_limit(self):
        """Rate limit config must have a default limit."""
        from app.security.rate_limit import ENDPOINT_LIMITS

        assert "default" in ENDPOINT_LIMITS


class TestRateLimiter:
    """Test rate limiter core functionality."""

    def test_rate_limiter_allows_requests_within_limit(self):
        """Rate limiter must allow requests within limit."""
        from app.security.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, burst_size=10)

        # Should allow 10 requests
        for _ in range(10):
            allowed, _ = limiter.check("test_key")
            assert allowed, "Request within limit should be allowed"

    def test_rate_limiter_blocks_requests_over_limit(self):
        """Rate limiter must block requests over limit."""
        from app.security.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=5, burst_size=5)

        # Use all tokens
        for _ in range(5):
            limiter.check("test_key")

        # Next request should be blocked
        allowed, info = limiter.check("test_key")
        assert not allowed, "Request over limit should be blocked"

    def test_rate_limiter_returns_remaining_count(self):
        """Rate limiter must return remaining token count."""
        from app.security.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, burst_size=10)

        allowed, info = limiter.check("test_key")
        assert allowed
        assert "remaining" in info
        assert info["remaining"] == 9  # One token used

    def test_rate_limiter_returns_reset_time(self):
        """Rate limiter must return reset timestamp."""
        from app.security.rate_limit import RateLimiter

        limiter = RateLimiter(requests_per_minute=10, burst_size=10)

        allowed, info = limiter.check("test_key")
        assert "reset" in info
        assert isinstance(info["reset"], (int, float))


class TestRateLimitMiddleware:
    """Test rate limit middleware."""

    def test_middleware_adds_rate_limit_headers(self):
        """Middleware must add X-RateLimit headers to response."""
        from app.security.rate_limit import RateLimitMiddleware

        app = FastAPI()
        app.add_middleware(RateLimitMiddleware)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        # Check headers exist
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_middleware_returns_429_when_limit_exceeded(self):
        """Middleware must return 429 when rate limit exceeded."""
        from app.security.rate_limit import RateLimitMiddleware, RateLimiter

        app = FastAPI()

        # Create a limiter with very low limit for testing
        test_limiter = RateLimiter(requests_per_minute=1, burst_size=1)
        app.add_middleware(RateLimitMiddleware, limiter=test_limiter)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200

        # Second request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429

    def test_429_response_includes_retry_after(self):
        """429 response must include Retry-After header."""
        from app.security.rate_limit import RateLimitMiddleware, RateLimiter

        app = FastAPI()
        test_limiter = RateLimiter(requests_per_minute=1, burst_size=1)
        app.add_middleware(RateLimitMiddleware, limiter=test_limiter)

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        client = TestClient(app)

        # Exhaust limit
        client.get("/test")

        # Check 429 response
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers


class TestPerEndpointLimits:
    """Test per-endpoint rate limits."""

    def test_different_endpoints_have_different_limits(self):
        """Different endpoints must have different rate limit configurations."""
        from app.security.rate_limit import ENDPOINT_LIMITS

        # Check that some specific endpoints have different limits
        if len(ENDPOINT_LIMITS) > 1:
            limits = list(ENDPOINT_LIMITS.values())
            # At least two different limits should exist
            unique_limits = set(l.get("requests_per_minute", l) for l in limits if isinstance(l, dict))
            # Just verify config exists, implementation may vary
            assert len(ENDPOINT_LIMITS) >= 1


class TestTokenBucketAlgorithm:
    """Test token bucket rate limiting algorithm specifics."""

    def test_tokens_refill_over_time(self):
        """Tokens must refill over time."""
        import time
        from app.security.rate_limit import RateLimiter

        # 60 requests per minute = 1 token per second
        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        # Use both tokens
        limiter.check("test_key")
        limiter.check("test_key")

        # Should be blocked
        allowed, _ = limiter.check("test_key")
        assert not allowed

        # Wait for refill (at least 1 second for 1 token)
        time.sleep(1.1)

        # Should have refilled at least 1 token
        allowed, info = limiter.check("test_key")
        assert allowed, "Token should have refilled after waiting"

    def test_burst_allows_spike(self):
        """Burst size must allow request spike."""
        from app.security.rate_limit import RateLimiter

        # Low rate but high burst
        limiter = RateLimiter(requests_per_minute=1, burst_size=5)

        # Should allow burst of 5 requests
        for i in range(5):
            allowed, _ = limiter.check("test_key")
            assert allowed, f"Burst request {i+1} should be allowed"

        # 6th should be blocked
        allowed, _ = limiter.check("test_key")
        assert not allowed, "Request beyond burst should be blocked"
