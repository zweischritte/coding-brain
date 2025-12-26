"""JWT Validator for OAuth 2.1 compliance.

This module implements JWT validation per the implementation plan section 4.3:
- JWT signature validation via JWKs (cache + rotation)
- Mandatory claim checks: iss, aud, exp, iat, nbf, sub
- PKCE S256 code verifier validation
- Token revocation checking
- Issuer allowlist enforcement
"""

import base64
import hashlib
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jwt
from jwt import PyJWKClient, PyJWKClientError
from jwt.exceptions import (
    DecodeError,
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    InvalidTokenError,
)


# ============================================================================
# Exceptions
# ============================================================================


class JWTValidationError(Exception):
    """Base exception for JWT validation errors."""

    pass


class JWTExpiredError(JWTValidationError):
    """Token has expired."""

    pass


class JWTInvalidClaimsError(JWTValidationError):
    """Token has missing or invalid claims."""

    pass


class JWTInvalidSignatureError(JWTValidationError):
    """Token signature is invalid."""

    pass


class JWTRevokedError(JWTValidationError):
    """Token has been revoked."""

    pass


class PKCEValidationError(Exception):
    """PKCE validation failed."""

    pass


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class RevocationChecker(Protocol):
    """Protocol for token revocation checking."""

    def is_revoked(self, token_id: str, claims: dict[str, Any]) -> bool:
        """Check if a token is revoked.

        Args:
            token_id: The token identifier (jti claim or token hash)
            claims: The decoded JWT claims

        Returns:
            True if the token is revoked, False otherwise
        """
        ...


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class JWTValidationResult:
    """Result of successful JWT validation."""

    # Required claims
    subject: str
    issuer: str
    audience: str | list[str]
    expires_at: int
    issued_at: int
    not_before: int

    # Organization claims
    org_id: str
    enterprise_id: str
    roles: list[str]
    scopes: list[str]
    team_ids: list[str] = field(default_factory=list)
    project_ids: list[str] = field(default_factory=list)

    # Optional claims
    session_id: str | None = None
    user_id: str | None = None
    token_id: str | None = None
    geo_scope: str | None = None

    # Raw claims for extension
    raw_claims: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# JWKS Cache Entry
# ============================================================================


@dataclass
class _JWKSCacheEntry:
    """Cache entry for JWKS with expiry."""

    jwks: dict[str, Any]
    fetched_at: float


# ============================================================================
# JWT Validator
# ============================================================================


class JWTValidator:
    """Validates JWTs according to OAuth 2.1 requirements.

    Features:
    - JWKS-based signature validation with caching
    - Mandatory claim validation (iss, aud, exp, iat, nbf, sub)
    - Issuer allowlist enforcement
    - Audience validation
    - Optional revocation checking
    - Configurable clock skew tolerance
    - Token lifetime enforcement
    """

    # Required JWT claims per implementation plan
    REQUIRED_CLAIMS = frozenset(
        ["sub", "iss", "aud", "exp", "iat", "nbf", "org_id", "roles", "scopes"]
    )

    # Allowed algorithms (no 'none' allowed)
    ALLOWED_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

    def __init__(
        self,
        issuer_allowlist: list[str],
        audience: str,
        revocation_checker: RevocationChecker | None = None,
        jwks_cache_ttl_seconds: int = 3600,
        clock_skew_seconds: int = 0,
        max_token_lifetime_seconds: int | None = None,
    ):
        """Initialize the JWT validator.

        Args:
            issuer_allowlist: List of allowed token issuers
            audience: Expected audience value
            revocation_checker: Optional checker for token revocation
            jwks_cache_ttl_seconds: How long to cache JWKS (default 1 hour)
            clock_skew_seconds: Tolerance for clock differences
            max_token_lifetime_seconds: Maximum allowed token lifetime
        """
        self._issuer_allowlist = set(issuer_allowlist)
        self._audience = audience
        self._revocation_checker = revocation_checker
        self._jwks_cache_ttl = jwks_cache_ttl_seconds
        self._clock_skew = clock_skew_seconds
        self._max_lifetime = max_token_lifetime_seconds

        # JWKS cache: issuer -> cache entry
        self._jwks_cache: dict[str, dict[str, Any] | _JWKSCacheEntry] = {}

    def validate(self, token: str) -> JWTValidationResult:
        """Validate a JWT and return the decoded claims.

        Args:
            token: The JWT to validate

        Returns:
            JWTValidationResult with validated claims

        Raises:
            JWTValidationError: Base exception for validation failures
            JWTExpiredError: Token has expired
            JWTInvalidClaimsError: Missing or invalid claims
            JWTInvalidSignatureError: Invalid signature
            JWTRevokedError: Token has been revoked
        """
        # Validate token format
        if not token or not isinstance(token, str) or not token.strip():
            raise JWTValidationError("Token is empty or invalid")

        # Check token structure
        parts = token.split(".")
        if len(parts) != 3:
            raise JWTValidationError("Invalid JWT structure")

        try:
            # Decode header to get issuer for JWKS lookup
            unverified_header = jwt.get_unverified_header(token)
            unverified_claims = jwt.decode(token, options={"verify_signature": False})
        except DecodeError as e:
            raise JWTValidationError(f"Failed to decode token: {e}")

        # Check algorithm
        algorithm = unverified_header.get("alg")
        if algorithm not in self.ALLOWED_ALGORITHMS:
            raise JWTInvalidSignatureError(
                f"Algorithm '{algorithm}' is not allowed. Use one of: {self.ALLOWED_ALGORITHMS}"
            )

        # Pre-validate required claims before PyJWT validation
        # This gives clearer error messages about which claim is missing
        pre_required = ["sub", "iss", "aud", "exp", "iat", "nbf"]
        for claim in pre_required:
            if claim not in unverified_claims:
                raise JWTInvalidClaimsError(f"Missing required claim: {claim}")

        # Validate issuer is in allowlist
        issuer = unverified_claims.get("iss")
        if issuer not in self._issuer_allowlist:
            raise JWTInvalidClaimsError(
                f"Issuer '{issuer}' is not in the allowed issuers list"
            )

        # Get JWKS for issuer
        jwks = self._get_jwks(issuer)

        # Find the signing key
        kid = unverified_header.get("kid")
        signing_key = self._find_key(jwks, kid)

        if signing_key is None:
            raise JWTInvalidSignatureError(
                f"No matching key found for kid '{kid}'"
            )

        # Verify signature and decode claims
        try:
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=self.ALLOWED_ALGORITHMS,
                audience=self._audience,
                issuer=list(self._issuer_allowlist),
                leeway=self._clock_skew,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "require": ["exp", "iat", "nbf", "iss", "aud", "sub"],
                },
            )
        except ExpiredSignatureError:
            raise JWTExpiredError("Token has expired")
        except InvalidAudienceError:
            raise JWTInvalidClaimsError("Invalid audience claim")
        except InvalidIssuerError:
            raise JWTInvalidClaimsError("Invalid issuer claim")
        except InvalidSignatureError:
            raise JWTInvalidSignatureError("Token signature verification failed")
        except InvalidTokenError as e:
            error_msg = str(e).lower()
            # Convert missing claim errors to JWTInvalidClaimsError
            if "missing" in error_msg or "required" in error_msg:
                # Extract claim name from error message
                for claim in ["sub", "iss", "aud", "exp", "iat", "nbf"]:
                    if claim in error_msg:
                        raise JWTInvalidClaimsError(f"Missing required claim: {claim}")
                raise JWTInvalidClaimsError(f"Missing required claim: {e}")
            # Convert "not yet valid" errors to JWTInvalidClaimsError
            if "not yet valid" in error_msg:
                if "iat" in error_msg:
                    raise JWTInvalidClaimsError(f"Token iat claim is in the future")
                if "nbf" in error_msg:
                    raise JWTInvalidClaimsError(f"Token nbf - not yet valid")
                raise JWTInvalidClaimsError(f"Token is not yet valid: {e}")
            raise JWTValidationError(f"Token validation failed: {e}")

        # Check for all required claims
        self._validate_required_claims(claims)

        # Check token lifetime if configured
        if self._max_lifetime is not None:
            self._validate_token_lifetime(claims)

        # Check iat is not in the future (with clock skew tolerance)
        self._validate_iat(claims)

        # Check for revocation
        if self._revocation_checker is not None:
            token_id = claims.get("jti") or claims.get("token_id") or self._hash_token(token)
            if self._revocation_checker.is_revoked(token_id, claims):
                raise JWTRevokedError("Token has been revoked")

        # Build result
        return self._build_result(claims)

    def _get_jwks(self, issuer: str) -> dict[str, Any]:
        """Get JWKS for an issuer, using cache if available."""
        cache_entry = self._jwks_cache.get(issuer)

        # Check if we have a cached entry
        if cache_entry is not None:
            # Handle both dict (from tests) and _JWKSCacheEntry
            if isinstance(cache_entry, dict):
                return cache_entry
            elif isinstance(cache_entry, _JWKSCacheEntry):
                # Check if cache is still valid
                if time.time() - cache_entry.fetched_at < self._jwks_cache_ttl:
                    return cache_entry.jwks

        # Fetch fresh JWKS
        jwks = self._fetch_jwks(issuer)

        # Cache it
        self._jwks_cache[issuer] = _JWKSCacheEntry(
            jwks=jwks,
            fetched_at=time.time(),
        )

        return jwks

    def _fetch_jwks(self, issuer: str) -> dict[str, Any]:
        """Fetch JWKS from the issuer's well-known endpoint.

        This method can be overridden or mocked for testing.
        """
        jwks_uri = f"{issuer.rstrip('/')}/.well-known/jwks.json"

        try:
            client = PyJWKClient(jwks_uri)
            # Get the keys as a dict
            keys = []
            for key in client.get_jwk_set().keys:
                key_dict = {
                    "kty": key.key_type,
                    "kid": key.key_id,
                }
                # Add algorithm if present
                if hasattr(key, "algorithm") and key.algorithm:
                    key_dict["alg"] = key.algorithm
                # Add RSA-specific fields
                if hasattr(key, "n"):
                    key_dict["n"] = key.n
                if hasattr(key, "e"):
                    key_dict["e"] = key.e
                keys.append(key_dict)

            return {"keys": keys}
        except PyJWKClientError as e:
            raise JWTValidationError(f"Failed to fetch JWKS from {jwks_uri}: {e}")

    def _find_key(self, jwks: dict[str, Any], kid: str | None) -> Any:
        """Find a signing key in the JWKS by key ID."""
        keys = jwks.get("keys", [])

        for key_data in keys:
            if key_data.get("kid") == kid:
                return self._jwk_to_key(key_data)

        return None

    def _jwk_to_key(self, jwk_data: dict[str, Any]) -> Any:
        """Convert JWK data to a key object for verification."""
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
        from cryptography.hazmat.backends import default_backend

        kty = jwk_data.get("kty")

        if kty == "RSA":
            # Decode n and e from base64url
            n = self._base64url_decode_int(jwk_data["n"])
            e = self._base64url_decode_int(jwk_data["e"])

            # Build RSA public key
            public_numbers = RSAPublicNumbers(e, n)
            public_key = public_numbers.public_key(default_backend())

            return public_key

        raise JWTValidationError(f"Unsupported key type: {kty}")

    def _base64url_decode_int(self, value: str) -> int:
        """Decode a base64url-encoded integer."""
        # Add padding if needed
        padding = 4 - len(value) % 4
        if padding != 4:
            value += "=" * padding

        data = base64.urlsafe_b64decode(value)
        return int.from_bytes(data, byteorder="big")

    def _validate_required_claims(self, claims: dict[str, Any]) -> None:
        """Validate that all required claims are present."""
        missing = []

        for claim in self.REQUIRED_CLAIMS:
            if claim not in claims:
                missing.append(claim)

        if missing:
            raise JWTInvalidClaimsError(
                f"Missing required claims: {', '.join(missing)}"
            )

    def _validate_token_lifetime(self, claims: dict[str, Any]) -> None:
        """Validate token lifetime doesn't exceed maximum."""
        iat = claims.get("iat", 0)
        exp = claims.get("exp", 0)
        lifetime = exp - iat

        if lifetime > self._max_lifetime:
            raise JWTInvalidClaimsError(
                f"Token lifetime ({lifetime}s) exceeds maximum ({self._max_lifetime}s)"
            )

    def _validate_iat(self, claims: dict[str, Any]) -> None:
        """Validate that iat is not in the future."""
        iat = claims.get("iat", 0)
        now = time.time()

        # Allow for clock skew
        if iat > now + self._clock_skew:
            raise JWTInvalidClaimsError(
                f"Token iat claim ({iat}) is in the future"
            )

    def _hash_token(self, token: str) -> str:
        """Generate a hash of the token for revocation checking."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _build_result(self, claims: dict[str, Any]) -> JWTValidationResult:
        """Build a validation result from claims."""
        audience = claims.get("aud")

        return JWTValidationResult(
            # Required claims
            subject=claims["sub"],
            issuer=claims["iss"],
            audience=audience,
            expires_at=claims["exp"],
            issued_at=claims["iat"],
            not_before=claims["nbf"],
            # Organization claims
            org_id=claims["org_id"],
            enterprise_id=claims.get("enterprise_id", ""),
            roles=claims.get("roles", []),
            scopes=claims.get("scopes", []),
            team_ids=claims.get("team_ids", []),
            project_ids=claims.get("project_ids", []),
            # Optional claims
            session_id=claims.get("session_id"),
            user_id=claims.get("user_id"),
            token_id=claims.get("token_id") or claims.get("jti"),
            geo_scope=claims.get("geo_scope"),
            # Raw claims
            raw_claims=claims,
        )


# ============================================================================
# PKCE Validator
# ============================================================================


class PKCEValidator:
    """Validates PKCE S256 code verifiers.

    Per OAuth 2.1/RFC 7636, PKCE (Proof Key for Code Exchange) adds security
    to the authorization code flow by requiring a code verifier.

    Only S256 method is supported (plain is rejected per implementation plan).
    """

    # PKCE verifier character set: [A-Za-z0-9-._~]
    VERIFIER_PATTERN = re.compile(r"^[A-Za-z0-9\-._~]+$")

    # Length constraints per RFC 7636
    MIN_VERIFIER_LENGTH = 43
    MAX_VERIFIER_LENGTH = 128

    def validate_s256(self, code_verifier: str, code_challenge: str) -> bool:
        """Validate a PKCE S256 code verifier against a code challenge.

        Args:
            code_verifier: The original code verifier
            code_challenge: The code challenge sent in the authorization request

        Returns:
            True if the verifier matches the challenge

        Raises:
            PKCEValidationError: If the verifier format is invalid
        """
        self._validate_verifier_format(code_verifier)

        # Calculate expected challenge
        verifier_bytes = code_verifier.encode("ascii")
        digest = hashlib.sha256(verifier_bytes).digest()
        expected_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

        return expected_challenge == code_challenge

    def validate_plain(self, code_verifier: str, code_challenge: str) -> bool:
        """Validate a PKCE plain code verifier.

        Per implementation plan, only S256 is allowed. Plain method is rejected.

        Raises:
            PKCEValidationError: Always, as plain method is not allowed
        """
        raise PKCEValidationError(
            "PKCE 'plain' method is not allowed. Only S256 is supported."
        )

    def generate_s256_pair(self) -> tuple[str, str]:
        """Generate a new PKCE code verifier and challenge pair.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate random verifier (64 bytes -> 86 base64url chars)
        random_bytes = secrets.token_bytes(64)
        code_verifier = base64.urlsafe_b64encode(random_bytes).rstrip(b"=").decode("ascii")

        # Truncate to max length if needed
        if len(code_verifier) > self.MAX_VERIFIER_LENGTH:
            code_verifier = code_verifier[: self.MAX_VERIFIER_LENGTH]

        # Calculate challenge
        verifier_bytes = code_verifier.encode("ascii")
        digest = hashlib.sha256(verifier_bytes).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

        return code_verifier, code_challenge

    def _validate_verifier_format(self, code_verifier: str) -> None:
        """Validate the format of a code verifier."""
        length = len(code_verifier)

        if length < self.MIN_VERIFIER_LENGTH:
            raise PKCEValidationError(
                f"Code verifier length ({length}) is less than minimum ({self.MIN_VERIFIER_LENGTH})"
            )

        if length > self.MAX_VERIFIER_LENGTH:
            raise PKCEValidationError(
                f"Code verifier length ({length}) exceeds maximum ({self.MAX_VERIFIER_LENGTH})"
            )

        if not self.VERIFIER_PATTERN.match(code_verifier):
            raise PKCEValidationError(
                "Code verifier contains invalid characters. Only [A-Za-z0-9-._~] are allowed."
            )
