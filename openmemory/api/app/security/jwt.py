"""
JWT validation and configuration.

Provides JWT token validation with support for:
- Multiple signing algorithms (HS256, RS256, ES256, EdDSA)
- Issuer and audience validation
- Expiration and iat validation
- Custom claims extraction (org_id, scopes)
"""

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional, Set

from jose import jwt as jose_jwt
from jose.exceptions import ExpiredSignatureError, JWTClaimsError, JWTError

from .types import AuthenticationError, TokenClaims


@lru_cache()
def get_jwt_config() -> Dict[str, Any]:
    """
    Get JWT configuration from environment.

    Returns:
        Dict with:
        - secret_key: The signing key (for HS256) or public key (for RS/ES/EdDSA)
        - algorithm: The signing algorithm
        - issuer: Expected token issuer
        - audience: Expected token audience
    """
    return {
        "secret_key": os.environ.get("JWT_SECRET_KEY", ""),
        "algorithm": os.environ.get("JWT_ALGORITHM", "HS256"),
        "issuer": os.environ.get("JWT_ISSUER", ""),
        "audience": os.environ.get("JWT_AUDIENCE", ""),
    }


def validate_jwt(token: str) -> TokenClaims:
    """
    Validate a JWT token and extract claims.

    Args:
        token: The JWT token string

    Returns:
        TokenClaims with validated and parsed claims

    Raises:
        AuthenticationError: If token is invalid, expired, or malformed
    """
    config = get_jwt_config()

    if not config["secret_key"]:
        raise AuthenticationError(
            message="JWT validation not configured",
            code="SERVER_CONFIG_ERROR"
        )

    try:
        # Decode and validate the token
        options = {
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "require_exp": True,
            "require_iat": True,
        }

        # Only verify issuer/audience if configured
        if config["issuer"]:
            options["verify_iss"] = True
        if config["audience"]:
            options["verify_aud"] = True

        payload = jose_jwt.decode(
            token,
            config["secret_key"],
            algorithms=[config["algorithm"]],
            issuer=config["issuer"] if config["issuer"] else None,
            audience=config["audience"] if config["audience"] else None,
            options=options,
        )

        # Validate required claims
        if "sub" not in payload:
            raise AuthenticationError(
                message="Token missing required 'sub' claim",
                code="INVALID_TOKEN"
            )

        if "org_id" not in payload:
            raise AuthenticationError(
                message="Token missing required 'org_id' claim",
                code="INVALID_TOKEN"
            )

        if "jti" not in payload:
            raise AuthenticationError(
                message="Token missing required 'jti' claim",
                code="INVALID_TOKEN"
            )

        # Parse scopes from space-delimited string
        scope_str = payload.get("scope", "")
        scopes: Set[str] = set(scope_str.split()) if scope_str else set()

        # Build TokenClaims
        return TokenClaims(
            sub=payload["sub"],
            iss=payload.get("iss", ""),
            aud=payload.get("aud", ""),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=payload["jti"],
            org_id=payload["org_id"],
            scopes=scopes,
            email=payload.get("email"),
            name=payload.get("name"),
        )

    except ExpiredSignatureError:
        raise AuthenticationError(
            message="Token has expired",
            code="TOKEN_EXPIRED"
        )
    except JWTClaimsError as e:
        raise AuthenticationError(
            message=f"Token claims validation failed: {e}",
            code="INVALID_CLAIMS"
        )
    except JWTError as e:
        raise AuthenticationError(
            message=f"Invalid token: {e}",
            code="INVALID_TOKEN"
        )
    except Exception as e:
        raise AuthenticationError(
            message=f"Token validation error: {e}",
            code="VALIDATION_ERROR"
        )


def validate_iat_not_future(iat: datetime, max_clock_skew_seconds: int = 30) -> None:
    """
    Validate that the token was not issued in the future.

    Args:
        iat: The token's issued-at timestamp
        max_clock_skew_seconds: Allowed clock skew in seconds

    Raises:
        AuthenticationError: If iat is too far in the future
    """
    now = datetime.now(timezone.utc)
    if iat > now:
        # Allow some clock skew
        skew = (iat - now).total_seconds()
        if skew > max_clock_skew_seconds:
            raise AuthenticationError(
                message="Token issued in the future",
                code="INVALID_IAT"
            )
