"""
Security middleware for FastAPI.

Provides:
- SecurityHeadersMiddleware: Adds security headers to all responses
- AuthExceptionMiddleware: Converts auth exceptions to proper HTTP responses
"""

from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from .types import AuthenticationError, AuthorizationError


class SecurityHeadersMiddleware:
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
        self.app = app
        self.csp = csp or self.DEFAULT_CSP
        self.hsts_max_age = hsts_max_age or self.HSTS_MAX_AGE

    async def __call__(self, scope, receive, send):
        """Add security headers without buffering streaming responses."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Content-Security-Policy"] = self.csp
                headers["Strict-Transport-Security"] = (
                    f"max-age={self.hsts_max_age}; includeSubDomains"
                )
                headers["X-Frame-Options"] = "DENY"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-XSS-Protection"] = "1; mode=block"
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                headers["Permissions-Policy"] = self.DEFAULT_PERMISSIONS_POLICY
            await send(message)

        await self.app(scope, receive, send_wrapper)


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
