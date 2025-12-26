"""DPoP (Demonstrating Proof of Possession) Validator.

This module implements DPoP validation per RFC 9449 and the implementation plan:
- DPoP proof JWT validation
- Method and URI binding
- Nonce handling for replay prevention
- Access token binding via ath claim
- Key thumbprint calculation per RFC 7638
"""

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import jwt
from jwt.exceptions import DecodeError, InvalidTokenError


# ============================================================================
# Exceptions
# ============================================================================


class DPoPValidationError(Exception):
    """Base exception for DPoP validation errors."""

    pass


class DPoPProofExpiredError(DPoPValidationError):
    """DPoP proof has expired."""

    pass


class DPoPReplayError(DPoPValidationError):
    """DPoP proof has been replayed."""

    pass


class DPoPMismatchError(DPoPValidationError):
    """DPoP proof doesn't match expected values."""

    pass


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class NonceProvider(Protocol):
    """Protocol for DPoP nonce management."""

    def get_expected_nonce(self) -> str | None:
        """Get the expected nonce value, if any."""
        ...

    def validate_nonce(self, nonce: str) -> bool:
        """Validate a nonce value."""
        ...


@runtime_checkable
class ReplayCache(Protocol):
    """Protocol for DPoP replay prevention."""

    def check_and_store(self, jti: str, exp_time: int) -> bool:
        """Check if jti has been seen and store it.

        Args:
            jti: The unique token identifier
            exp_time: When the proof expires

        Returns:
            True if this is a new jti, False if it's a replay
        """
        ...


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DPoPProof:
    """Validated DPoP proof data."""

    method: str
    uri: str
    issued_at: int
    jti: str
    key_thumbprint: str
    nonce: str | None = None
    access_token_hash: str | None = None
    jwk: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DPoP Validator
# ============================================================================


class DPoPValidator:
    """Validates DPoP proofs per RFC 9449.

    Features:
    - JWT structure validation (typ, alg, jwk headers)
    - Method and URI binding verification
    - Proof expiration and timing checks
    - Nonce handling for replay prevention
    - Access token binding via ath claim
    - Key thumbprint calculation
    """

    # Allowed asymmetric algorithms for DPoP
    ALLOWED_ALGORITHMS = ["ES256", "ES384", "ES512", "RS256", "RS384", "RS512"]

    # Required claims in DPoP proof
    REQUIRED_CLAIMS = frozenset(["htm", "htu", "iat", "jti"])

    def __init__(
        self,
        max_age_seconds: int = 120,
        clock_skew_seconds: int = 0,
        nonce_provider: NonceProvider | None = None,
        replay_cache: ReplayCache | None = None,
    ):
        """Initialize the DPoP validator.

        Args:
            max_age_seconds: Maximum age of DPoP proof in seconds (default 2 minutes)
            clock_skew_seconds: Tolerance for clock differences
            nonce_provider: Optional nonce provider for server-side nonces
            replay_cache: Optional cache for replay prevention
        """
        self._max_age = max_age_seconds
        self._clock_skew = clock_skew_seconds
        self._nonce_provider = nonce_provider
        self._replay_cache = replay_cache

    def validate(
        self,
        proof: str,
        method: str,
        uri: str,
        access_token: str | None = None,
        expected_thumbprint: str | None = None,
    ) -> DPoPProof:
        """Validate a DPoP proof.

        Args:
            proof: The DPoP proof JWT
            method: The HTTP method of the request
            uri: The URI of the request
            access_token: Optional access token to bind to (validates ath claim)
            expected_thumbprint: Optional expected key thumbprint (for cnf binding)

        Returns:
            DPoPProof with validated data

        Raises:
            DPoPValidationError: Base exception for validation failures
            DPoPProofExpiredError: Proof has expired
            DPoPReplayError: Proof has been replayed
            DPoPMismatchError: Proof doesn't match expected values
        """
        # Validate proof is present
        if not proof or not isinstance(proof, str) or not proof.strip():
            raise DPoPValidationError("DPoP proof is empty or invalid")

        # Decode without verification first to get header
        try:
            unverified_header = jwt.get_unverified_header(proof)
            unverified_claims = jwt.decode(proof, options={"verify_signature": False})
        except DecodeError as e:
            raise DPoPValidationError(f"Failed to decode DPoP proof: {e}")

        # Validate header
        self._validate_header(unverified_header)

        # Get JWK from header
        jwk = unverified_header.get("jwk", {})

        # Validate required claims are present
        self._validate_required_claims(unverified_claims)

        # Verify signature using JWK from header
        algorithm = unverified_header.get("alg")
        signing_key = self._jwk_to_key(jwk)

        try:
            claims = jwt.decode(
                proof,
                signing_key,
                algorithms=[algorithm],
                options={
                    "verify_signature": True,
                    "verify_exp": False,  # DPoP uses iat-based expiry
                    "verify_iat": False,  # We handle iat validation ourselves
                    "verify_aud": False,
                    "verify_iss": False,
                },
            )
        except InvalidTokenError as e:
            raise DPoPValidationError(f"DPoP signature verification failed: {e}")

        # Validate timing
        self._validate_timing(claims)

        # Validate method and URI binding
        self._validate_binding(claims, method, uri)

        # Validate nonce if provider is configured
        self._validate_nonce(claims)

        # Check for replay
        self._check_replay(claims)

        # Validate access token binding if provided
        self._validate_access_token_binding(claims, access_token)

        # Calculate key thumbprint
        thumbprint = self._calculate_thumbprint(jwk)

        # Validate expected thumbprint if provided
        if expected_thumbprint is not None and thumbprint != expected_thumbprint:
            raise DPoPMismatchError(
                f"Key thumbprint mismatch: expected {expected_thumbprint}, got {thumbprint}"
            )

        return DPoPProof(
            method=claims["htm"],
            uri=claims["htu"],
            issued_at=claims["iat"],
            jti=claims["jti"],
            key_thumbprint=thumbprint,
            nonce=claims.get("nonce"),
            access_token_hash=claims.get("ath"),
            jwk=jwk,
        )

    def _validate_header(self, header: dict[str, Any]) -> None:
        """Validate DPoP proof header."""
        # Check typ header
        typ = header.get("typ")
        if typ != "dpop+jwt":
            raise DPoPValidationError(
                f"Invalid DPoP typ header: expected 'dpop+jwt', got '{typ}'"
            )

        # Check algorithm
        alg = header.get("alg")
        if alg not in self.ALLOWED_ALGORITHMS:
            raise DPoPValidationError(
                f"Invalid DPoP algorithm '{alg}'. Allowed: {self.ALLOWED_ALGORITHMS}"
            )

        # Check JWK is present
        if "jwk" not in header:
            raise DPoPValidationError("DPoP proof is missing jwk header")

    def _validate_required_claims(self, claims: dict[str, Any]) -> None:
        """Validate required claims are present."""
        for claim in self.REQUIRED_CLAIMS:
            if claim not in claims:
                raise DPoPValidationError(f"DPoP proof is missing required claim: {claim}")

    def _validate_timing(self, claims: dict[str, Any]) -> None:
        """Validate proof timing."""
        iat = claims.get("iat", 0)
        now = time.time()

        # Check if iat is in the future (with clock skew)
        if iat > now + self._clock_skew:
            raise DPoPValidationError(
                f"DPoP proof iat is in the future: {iat} > {now}"
            )

        # Check if proof has expired
        age = now - iat
        if age > self._max_age + self._clock_skew:
            raise DPoPProofExpiredError(
                f"DPoP proof has expired: age {age}s > max {self._max_age}s"
            )

    def _validate_binding(
        self, claims: dict[str, Any], expected_method: str, expected_uri: str
    ) -> None:
        """Validate method and URI binding."""
        htm = claims.get("htm", "")
        htu = claims.get("htu", "")

        # Case-insensitive method comparison
        if htm.upper() != expected_method.upper():
            raise DPoPMismatchError(
                f"DPoP method mismatch: expected {expected_method}, got {htm}"
            )

        # Parse URIs for comparison (ignore query string per RFC 9449)
        proof_parsed = urlparse(htu)
        expected_parsed = urlparse(expected_uri)

        proof_base = f"{proof_parsed.scheme}://{proof_parsed.netloc}{proof_parsed.path}"
        expected_base = f"{expected_parsed.scheme}://{expected_parsed.netloc}{expected_parsed.path}"

        if proof_base != expected_base:
            raise DPoPMismatchError(
                f"DPoP URI mismatch: expected {expected_base}, got {proof_base}"
            )

    def _validate_nonce(self, claims: dict[str, Any]) -> None:
        """Validate nonce if nonce provider is configured."""
        if self._nonce_provider is None:
            return

        expected_nonce = self._nonce_provider.get_expected_nonce()
        if expected_nonce is not None:
            proof_nonce = claims.get("nonce")
            if proof_nonce is None:
                raise DPoPValidationError("DPoP proof is missing required nonce")

            if not self._nonce_provider.validate_nonce(proof_nonce):
                raise DPoPValidationError(f"DPoP nonce is invalid: {proof_nonce}")

    def _check_replay(self, claims: dict[str, Any]) -> None:
        """Check for replay attacks."""
        if self._replay_cache is None:
            return

        jti = claims.get("jti", "")
        iat = claims.get("iat", 0)

        # Calculate expiry time for cache entry
        exp_time = iat + self._max_age

        if not self._replay_cache.check_and_store(jti, exp_time):
            raise DPoPReplayError(f"DPoP proof has been replayed: jti={jti}")

    def _validate_access_token_binding(
        self, claims: dict[str, Any], access_token: str | None
    ) -> None:
        """Validate access token binding via ath claim."""
        if access_token is None:
            return

        ath = claims.get("ath")
        if ath is None:
            raise DPoPValidationError(
                "DPoP proof is missing ath claim required for access token binding"
            )

        # Calculate expected ath
        token_hash = hashlib.sha256(access_token.encode()).digest()
        expected_ath = base64.urlsafe_b64encode(token_hash).rstrip(b"=").decode("ascii")

        if ath != expected_ath:
            raise DPoPMismatchError(
                "DPoP ath claim does not match access token hash"
            )

    def _jwk_to_key(self, jwk: dict[str, Any]) -> Any:
        """Convert JWK to a key object for verification."""
        kty = jwk.get("kty")

        if kty == "EC":
            return self._ec_jwk_to_key(jwk)
        elif kty == "RSA":
            return self._rsa_jwk_to_key(jwk)
        else:
            raise DPoPValidationError(f"Unsupported key type: {kty}")

    def _ec_jwk_to_key(self, jwk: dict[str, Any]) -> Any:
        """Convert EC JWK to public key."""
        from cryptography.hazmat.primitives.asymmetric.ec import (
            EllipticCurvePublicNumbers,
            SECP256R1,
            SECP384R1,
            SECP521R1,
        )
        from cryptography.hazmat.backends import default_backend

        crv = jwk.get("crv")
        curve_map = {
            "P-256": SECP256R1(),
            "P-384": SECP384R1(),
            "P-521": SECP521R1(),
        }

        curve = curve_map.get(crv)
        if curve is None:
            raise DPoPValidationError(f"Unsupported EC curve: {crv}")

        x = self._base64url_decode_int(jwk["x"])
        y = self._base64url_decode_int(jwk["y"])

        public_numbers = EllipticCurvePublicNumbers(x, y, curve)
        return public_numbers.public_key(default_backend())

    def _rsa_jwk_to_key(self, jwk: dict[str, Any]) -> Any:
        """Convert RSA JWK to public key."""
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
        from cryptography.hazmat.backends import default_backend

        n = self._base64url_decode_int(jwk["n"])
        e = self._base64url_decode_int(jwk["e"])

        public_numbers = RSAPublicNumbers(e, n)
        return public_numbers.public_key(default_backend())

    def _base64url_decode_int(self, value: str) -> int:
        """Decode a base64url-encoded integer."""
        # Add padding if needed
        padding = 4 - len(value) % 4
        if padding != 4:
            value += "=" * padding

        data = base64.urlsafe_b64decode(value)
        return int.from_bytes(data, byteorder="big")

    def _calculate_thumbprint(self, jwk: dict[str, Any]) -> str:
        """Calculate JWK thumbprint per RFC 7638."""
        kty = jwk.get("kty")

        if kty == "EC":
            # For EC keys: {"crv", "kty", "x", "y"} in lexicographic order
            canonical = {
                "crv": jwk["crv"],
                "kty": jwk["kty"],
                "x": jwk["x"],
                "y": jwk["y"],
            }
        elif kty == "RSA":
            # For RSA keys: {"e", "kty", "n"} in lexicographic order
            canonical = {
                "e": jwk["e"],
                "kty": jwk["kty"],
                "n": jwk["n"],
            }
        else:
            raise DPoPValidationError(f"Cannot calculate thumbprint for key type: {kty}")

        # Serialize with no whitespace
        canonical_json = json.dumps(canonical, separators=(",", ":"), sort_keys=True)

        # SHA-256 hash and base64url encode
        digest = hashlib.sha256(canonical_json.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


# ============================================================================
# DPoP Proof Creation Helper
# ============================================================================


def create_dpop_proof(
    private_key: bytes,
    method: str,
    uri: str,
    nonce: str | None = None,
    access_token: str | None = None,
    algorithm: str = "ES256",
) -> str:
    """Create a DPoP proof JWT.

    This is a helper function for creating DPoP proofs, primarily for testing.

    Args:
        private_key: The private key in PEM format
        method: The HTTP method
        uri: The request URI
        nonce: Optional server-provided nonce
        access_token: Optional access token (generates ath claim)
        algorithm: The signing algorithm (default ES256)

    Returns:
        The DPoP proof JWT string
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    # Load private key to extract public key
    key = load_pem_private_key(private_key, password=None)

    # Build JWK from public key
    public_key = key.public_key()
    jwk = _public_key_to_jwk(public_key)

    # Build payload
    now = int(time.time())
    jti = base64.urlsafe_b64encode(secrets.token_bytes(16)).decode("ascii")

    payload = {
        "htm": method,
        "htu": uri,
        "iat": now,
        "jti": jti,
    }

    if nonce is not None:
        payload["nonce"] = nonce

    if access_token is not None:
        token_hash = hashlib.sha256(access_token.encode()).digest()
        payload["ath"] = base64.urlsafe_b64encode(token_hash).rstrip(b"=").decode("ascii")

    # Build header
    header = {
        "typ": "dpop+jwt",
        "alg": algorithm,
        "jwk": jwk,
    }

    return jwt.encode(payload, private_key, algorithm=algorithm, headers=header)


def _public_key_to_jwk(public_key: Any) -> dict[str, Any]:
    """Convert a public key to JWK format."""
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

    if isinstance(public_key, EllipticCurvePublicKey):
        numbers = public_key.public_numbers()
        curve_name = public_key.curve.name

        curve_map = {
            "secp256r1": "P-256",
            "secp384r1": "P-384",
            "secp521r1": "P-521",
        }

        crv = curve_map.get(curve_name, curve_name)

        # Determine byte length based on curve
        byte_length = {
            "P-256": 32,
            "P-384": 48,
            "P-521": 66,
        }.get(crv, 32)

        def _int_to_base64url(n: int, length: int) -> str:
            data = n.to_bytes(length, byteorder="big")
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

        return {
            "kty": "EC",
            "crv": crv,
            "x": _int_to_base64url(numbers.x, byte_length),
            "y": _int_to_base64url(numbers.y, byte_length),
        }

    elif isinstance(public_key, RSAPublicKey):
        numbers = public_key.public_numbers()

        def _int_to_base64url(n: int) -> str:
            byte_length = (n.bit_length() + 7) // 8
            data = n.to_bytes(byte_length, byteorder="big")
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

        return {
            "kty": "RSA",
            "n": _int_to_base64url(numbers.n),
            "e": _int_to_base64url(numbers.e),
        }

    else:
        raise ValueError(f"Unsupported key type: {type(public_key)}")
