"""
Security middleware for FastAPI.

Provides:
- SecurityHeadersMiddleware: Adds security headers to all responses
- AuthExceptionMiddleware: Converts auth exceptions to proper HTTP responses
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from .types import AuthenticationError, AuthorizationError


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.

    Headers added:
    - Content-Security-Policy (CSP)
    - Strict-Transport-Security (HSTS)
    - X-Frame-Options
    - X-Content-Type-Options
    - X-XSS-Protection
    - Referrer-Policy
    - Permissions-Policy
    """

    # Default CSP policy - restrictive by default
    DEFAULT_CSP = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'; "
        "base-uri 'self'"
    )

    # HSTS max-age (1 year recommended for production)
    HSTS_MAX_AGE = 31536000

    # Permissions policy
    DEFAULT_PERMISSIONS_POLICY = (
        "geolocation=(), "
        "camera=(), "
        "microphone=(), "
        "payment=(), "
        "usb=(), "
        "magnetometer=(), "
        "gyroscope=(), "
        "accelerometer=()"
    )

    def __init__(self, app, csp: str = None, hsts_max_age: int = None):
        """
        Initialize the security headers middleware.

        Args:
            app: The ASGI application
            csp: Custom Content-Security-Policy header value
            hsts_max_age: Custom HSTS max-age value in seconds
        """
        super().__init__(app)
        self.csp = csp or self.DEFAULT_CSP
        self.hsts_max_age = hsts_max_age or self.HSTS_MAX_AGE

    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        # Content-Security-Policy
        response.headers["Content-Security-Policy"] = self.csp

        # Strict-Transport-Security (HSTS)
        response.headers["Strict-Transport-Security"] = (
            f"max-age={self.hsts_max_age}; includeSubDomains"
        )

        # X-Frame-Options (clickjacking protection)
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options (MIME sniffing protection)
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection (legacy XSS protection)
        # Note: Modern browsers are deprecating this, but it's defense-in-depth
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions-Policy (formerly Feature-Policy)
        response.headers["Permissions-Policy"] = self.DEFAULT_PERMISSIONS_POLICY

        return response


class AuthExceptionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that converts authentication/authorization exceptions to HTTP responses.

    This allows route handlers to raise AuthenticationError and AuthorizationError
    and have them automatically converted to proper 401/403 responses with
    appropriate headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Handle auth exceptions and convert to proper HTTP responses."""
        try:
            return await call_next(request)
        except AuthenticationError as e:
            return JSONResponse(
                status_code=401,
                content={
                    "error": e.code,
                    "message": e.message,
                },
                headers={
                    "WWW-Authenticate": 'Bearer realm="api", error="invalid_token"',
                },
            )
        except AuthorizationError as e:
            return JSONResponse(
                status_code=403,
                content={
                    "error": e.code,
                    "message": e.message,
                },
            )
