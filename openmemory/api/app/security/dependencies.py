"""
FastAPI dependencies for authentication and authorization.

Provides:
- get_current_principal: Extract and validate the authenticated principal
- require_scopes: Dependency factory for scope-based authorization
- get_optional_principal: Optional authentication (for public endpoints)
"""

from typing import Callable, Optional

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt import validate_jwt, validate_iat_not_future
from .types import AuthenticationError, AuthorizationError, Principal, Scope

# HTTPBearer with auto_error=False so we can handle missing auth ourselves
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_principal(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    dpop: Optional[str] = Header(None, alias="DPoP"),
) -> Principal:
    """
    Extract and validate the authenticated principal from the request.

    This dependency:
    1. Validates the JWT access token from the Authorization header
    2. Validates DPoP proof if present
    3. Extracts claims and builds the Principal object
    4. Injects org_id for tenant scoping

    Raises:
        AuthenticationError: If authentication fails
    """
    # Check for authorization header
    if not authorization:
        raise AuthenticationError(
            message="Missing authorization header",
            code="MISSING_AUTH"
        )

    if not authorization.credentials:
        raise AuthenticationError(
            message="Empty authorization token",
            code="EMPTY_TOKEN"
        )

    # Validate the JWT token
    token = authorization.credentials
    claims = validate_jwt(token)

    # Additional validation: check iat is not in the future
    validate_iat_not_future(claims.iat)

    # Build the principal
    principal = Principal(
        user_id=claims.sub,
        org_id=claims.org_id,
        claims=claims,
    )

    # If DPoP header is present, validate it
    if dpop:
        # DPoP validation will be implemented in a separate module
        # For now, just extract the request info for later validation
        from .dpop import DPoPValidator, get_dpop_cache

        cache = await get_dpop_cache()
        if cache:
            validator = DPoPValidator(cache)
            # Get the full URL for htu validation
            http_uri = str(request.url)
            http_method = request.method

            dpop_proof = await validator.validate(
                dpop_proof=dpop,
                http_method=http_method,
                http_uri=http_uri,
                access_token=token,
            )

            # Store the DPoP thumbprint on the principal
            thumbprint = await validator.get_thumbprint(dpop)
            principal = Principal(
                user_id=claims.sub,
                org_id=claims.org_id,
                claims=claims,
                dpop_thumbprint=thumbprint,
            )

    return principal


async def get_optional_principal(
    request: Request,
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    dpop: Optional[str] = Header(None, alias="DPoP"),
) -> Optional[Principal]:
    """
    Optionally extract the authenticated principal.

    Returns None if no valid credentials are provided.
    Used for endpoints that support both authenticated and anonymous access.

    If credentials are provided but invalid, raises AuthenticationError.
    """
    # No auth header = anonymous access
    if not authorization or not authorization.credentials:
        return None

    # Auth header present but might be invalid - validate it
    return await get_current_principal(request, authorization, dpop)


def require_scopes(*scopes: Scope | str) -> Callable:
    """
    Factory for creating scope-checking dependencies.

    Usage:
        @router.get("/protected")
        async def protected_endpoint(
            principal: Principal = Depends(require_scopes(Scope.MEMORIES_READ))
        ):
            ...

    Args:
        scopes: One or more required scopes

    Returns:
        A dependency that validates the principal has all required scopes
    """

    async def check_scopes(
        principal: Principal = Depends(get_current_principal),
    ) -> Principal:
        for scope in scopes:
            if not principal.has_scope(scope):
                scope_str = scope.value if isinstance(scope, Scope) else scope
                raise AuthorizationError(
                    message=f"Required scope '{scope_str}' not granted",
                    code="INSUFFICIENT_SCOPE",
                )
        return principal

    return check_scopes
