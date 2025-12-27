"""
FastAPI exception handlers for security exceptions.

Converts AuthenticationError and AuthorizationError to proper HTTP responses.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .types import AuthenticationError, AuthorizationError


def register_security_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers for security exceptions.

    Args:
        app: The FastAPI application
    """

    @app.exception_handler(AuthenticationError)
    async def authentication_error_handler(
        request: Request,
        exc: AuthenticationError,
    ) -> JSONResponse:
        """Handle AuthenticationError and return 401."""
        return JSONResponse(
            status_code=401,
            content={
                "error": exc.code,
                "message": exc.message,
            },
            headers={
                "WWW-Authenticate": 'Bearer realm="api", error="invalid_token"',
            },
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(
        request: Request,
        exc: AuthorizationError,
    ) -> JSONResponse:
        """Handle AuthorizationError and return 403."""
        return JSONResponse(
            status_code=403,
            content={
                "error": exc.code,
                "message": exc.message,
            },
        )
