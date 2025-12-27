"""
DPoP (Demonstrating Proof of Possession) validation.

Implements RFC 9449 for proof-of-possession tokens to prevent token theft.
Uses Valkey for replay prevention via jti caching.
"""

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

from jose import jwt as jose_jwt
from jose.exceptions import JWTError

from .types import AuthenticationError


class CacheProtocol(Protocol):
    """Protocol for cache operations (Valkey/Redis)."""

    async def get(self, key: str) -> Optional[str]:
        """Get a value from cache."""
        ...

    async def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set a value with expiration."""
        ...


@dataclass
class DPoPProof:
    """Validated DPoP proof structure."""

    jti: str  # Unique token ID
    htm: str  # HTTP method
    htu: str  # HTTP target URI
    iat: datetime  # Issued at time
    ath: Optional[str] = None  # Access token hash (for bound tokens)
    nonce: Optional[str] = None  # Server-provided nonce


# Global cache instance (lazy initialized)
_dpop_cache: Optional[CacheProtocol] = None


async def get_dpop_cache() -> Optional[CacheProtocol]:
    """
    Get the DPoP replay cache (Valkey connection).

    Returns None if Valkey is not configured.
    """
    global _dpop_cache

    if _dpop_cache is not None:
        return _dpop_cache

    # Try to connect to Valkey if configured
    valkey_url = os.environ.get("VALKEY_URL")
    if not valkey_url:
        return None

    try:
        import redis.asyncio as redis
        _dpop_cache = redis.from_url(valkey_url)
        return _dpop_cache
    except ImportError:
        return None
    except Exception:
        return None


class DPoPValidator:
    """
    Validates DPoP proofs per RFC 9449.

    Features:
    - JWK thumbprint validation
    - Replay prevention via Valkey jti cache
    - HTTP method and URI binding validation
    - Access token binding (ath claim)
    """

    # DPoP proof lifetime (RFC 9449 recommends checking iat)
    MAX_AGE_SECONDS = 60

    # Cache key prefix for replay prevention
    CACHE_PREFIX = "dpop:jti:"

    # Cache TTL for jti values (slightly longer than MAX_AGE to handle clock skew)
    JTI_CACHE_TTL = 120

    def __init__(self, cache: CacheProtocol):
        """Initialize with a cache for replay prevention."""
        self.cache = cache

    async def validate(
        self,
        dpop_proof: str,
        http_method: str,
        http_uri: str,
        access_token: Optional[str] = None,
    ) -> DPoPProof:
        """
        Validate a DPoP proof.

        Args:
            dpop_proof: The DPoP header value (JWT)
            http_method: The HTTP method of the request
            http_uri: The HTTP target URI of the request
            access_token: The access token (for ath binding validation)

        Returns:
            Validated DPoPProof

        Raises:
            AuthenticationError: If validation fails
        """
        try:
            # Decode without verification first to get the header
            unverified = jose_jwt.get_unverified_header(dpop_proof)
        except JWTError as e:
            raise AuthenticationError(
                message=f"Invalid DPoP proof format: {e}",
                code="INVALID_DPOP"
            )

        # Validate header
        self._validate_header(unverified)

        # Extract JWK from header
        jwk = unverified.get("jwk")
        if not jwk:
            raise AuthenticationError(
                message="DPoP proof missing JWK in header",
                code="INVALID_DPOP"
            )

        # Check JWK doesn't contain private key material
        if "d" in jwk:
            raise AuthenticationError(
                message="DPoP JWK must not contain private key material",
                code="INVALID_DPOP"
            )

        # Decode and verify the proof using the embedded JWK
        try:
            # Convert JWK to PEM for verification
            payload = jose_jwt.decode(
                dpop_proof,
                jwk,
                algorithms=[unverified.get("alg", "ES256")],
                options={"verify_aud": False, "verify_iss": False},
            )
        except JWTError as e:
            raise AuthenticationError(
                message=f"DPoP signature verification failed: {e}",
                code="INVALID_DPOP"
            )

        # Validate required claims
        self._validate_claims(payload, http_method, http_uri)

        # Check for replay
        jti = payload["jti"]
        await self._check_replay(jti)

        # Validate ath if access_token provided
        ath = payload.get("ath")
        if access_token and ath:
            expected_ath = self._compute_ath(access_token)
            if ath != expected_ath:
                raise AuthenticationError(
                    message="DPoP access token hash (ath) mismatch",
                    code="INVALID_DPOP"
                )

        # Cache the jti to prevent replay
        await self._cache_jti(jti)

        return DPoPProof(
            jti=jti,
            htm=payload["htm"],
            htu=payload["htu"],
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            ath=ath,
            nonce=payload.get("nonce"),
        )

    def _validate_header(self, header: Dict[str, Any]) -> None:
        """Validate DPoP JWT header."""
        # typ must be "dpop+jwt"
        typ = header.get("typ")
        if typ != "dpop+jwt":
            raise AuthenticationError(
                message=f"Invalid DPoP typ header: expected 'dpop+jwt', got '{typ}'",
                code="INVALID_DPOP"
            )

        # alg must be an asymmetric algorithm
        alg = header.get("alg")
        if alg in ["HS256", "HS384", "HS512", "none"]:
            raise AuthenticationError(
                message="DPoP must use asymmetric algorithm",
                code="INVALID_DPOP"
            )

    def _validate_claims(
        self,
        payload: Dict[str, Any],
        http_method: str,
        http_uri: str,
    ) -> None:
        """Validate DPoP JWT claims."""
        # jti is required
        if "jti" not in payload:
            raise AuthenticationError(
                message="DPoP proof missing 'jti' claim",
                code="INVALID_DPOP"
            )

        # htm is required and must match
        htm = payload.get("htm")
        if not htm:
            raise AuthenticationError(
                message="DPoP proof missing 'htm' claim",
                code="INVALID_DPOP"
            )
        if htm.upper() != http_method.upper():
            raise AuthenticationError(
                message=f"DPoP htm mismatch: expected '{http_method}', got '{htm}'",
                code="INVALID_DPOP"
            )

        # htu is required and must match
        htu = payload.get("htu")
        if not htu:
            raise AuthenticationError(
                message="DPoP proof missing 'htu' claim",
                code="INVALID_DPOP"
            )
        # Compare URLs (normalize trailing slashes)
        if htu.rstrip("/") != http_uri.rstrip("/"):
            raise AuthenticationError(
                message=f"DPoP htu mismatch: expected '{http_uri}', got '{htu}'",
                code="INVALID_DPOP"
            )

        # iat is required and must be recent
        iat = payload.get("iat")
        if not iat:
            raise AuthenticationError(
                message="DPoP proof missing 'iat' claim",
                code="INVALID_DPOP"
            )

        now = datetime.now(timezone.utc).timestamp()
        age = now - iat

        # Check not too old
        if age > self.MAX_AGE_SECONDS:
            raise AuthenticationError(
                message=f"DPoP proof expired (age: {age:.0f}s, max: {self.MAX_AGE_SECONDS}s)",
                code="DPOP_EXPIRED"
            )

        # Check not in the future (with small clock skew allowance)
        if age < -10:  # Allow 10 seconds of clock skew
            raise AuthenticationError(
                message="DPoP proof issued in the future",
                code="INVALID_DPOP"
            )

    async def _check_replay(self, jti: str) -> None:
        """Check if this jti has been used before."""
        cache_key = f"{self.CACHE_PREFIX}{jti}"

        try:
            existing = await self.cache.get(cache_key)
            if existing:
                raise AuthenticationError(
                    message="DPoP proof reuse detected",
                    code="DPOP_REPLAY"
                )
        except AuthenticationError:
            raise
        except Exception:
            # If cache check fails, allow the request (fail open for availability)
            pass

    async def _cache_jti(self, jti: str) -> None:
        """Cache the jti to prevent replay."""
        cache_key = f"{self.CACHE_PREFIX}{jti}"

        try:
            await self.cache.setex(cache_key, self.JTI_CACHE_TTL, "1")
        except Exception:
            # If caching fails, log but don't fail the request
            pass

    def _compute_ath(self, access_token: str) -> str:
        """Compute the access token hash (ath) for DPoP binding."""
        token_hash = hashlib.sha256(access_token.encode("ascii")).digest()
        return base64.urlsafe_b64encode(token_hash).decode().rstrip("=")

    async def get_thumbprint(self, dpop_proof: str) -> str:
        """
        Extract the JWK thumbprint from a DPoP proof.

        This is used to bind the DPoP key to the access token.
        Per RFC 7638: JWK Thumbprint.
        """
        try:
            header = jose_jwt.get_unverified_header(dpop_proof)
            jwk = header.get("jwk", {})

            # Compute thumbprint per RFC 7638
            # For EC keys: kty, crv, x, y (in lexicographic order)
            # For RSA keys: e, kty, n (in lexicographic order)
            kty = jwk.get("kty")

            if kty == "EC":
                thumbprint_input = {
                    "crv": jwk.get("crv"),
                    "kty": "EC",
                    "x": jwk.get("x"),
                    "y": jwk.get("y"),
                }
            elif kty == "RSA":
                thumbprint_input = {
                    "e": jwk.get("e"),
                    "kty": "RSA",
                    "n": jwk.get("n"),
                }
            elif kty == "OKP":  # EdDSA
                thumbprint_input = {
                    "crv": jwk.get("crv"),
                    "kty": "OKP",
                    "x": jwk.get("x"),
                }
            else:
                raise AuthenticationError(
                    message=f"Unsupported key type: {kty}",
                    code="INVALID_DPOP"
                )

            # Serialize with sorted keys and no whitespace
            canonical = json.dumps(thumbprint_input, sort_keys=True, separators=(",", ":"))
            thumbprint_hash = hashlib.sha256(canonical.encode("utf-8")).digest()
            return base64.urlsafe_b64encode(thumbprint_hash).decode().rstrip("=")

        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(
                message=f"Failed to compute JWK thumbprint: {e}",
                code="INVALID_DPOP"
            )
