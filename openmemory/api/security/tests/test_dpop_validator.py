"""Tests for DPoP (Demonstrating Proof of Possession) validator.

DPoP is mandatory per implementation plan section 4.3 to prevent token replay.
Tests cover RFC 9449 requirements:
- DPoP proof validation
- Token binding via cnf claim
- Method and URI binding
- Nonce handling for replay prevention
- Key thumbprint validation
"""

import base64
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openmemory.api.security.dpop_validator import (
    DPoPValidator,
    DPoPValidationError,
    DPoPProofExpiredError,
    DPoPReplayError,
    DPoPMismatchError,
    DPoPProof,
    create_dpop_proof,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def ec_key_pair():
    """Generate EC P-256 key pair for testing."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization

    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return {
        "private_key": private_key,
        "public_key": public_key,
        "private_pem": private_pem,
        "public_pem": public_pem,
    }


@pytest.fixture
def jwk_from_ec(ec_key_pair) -> dict[str, Any]:
    """Convert EC public key to JWK format."""
    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey

    public_key: EllipticCurvePublicKey = ec_key_pair["public_key"]
    numbers = public_key.public_numbers()

    def _int_to_base64url(n: int, length: int) -> str:
        data = n.to_bytes(length, byteorder="big")
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _int_to_base64url(numbers.x, 32),
        "y": _int_to_base64url(numbers.y, 32),
    }


@pytest.fixture
def dpop_thumbprint(jwk_from_ec) -> str:
    """Calculate JWK thumbprint for DPoP key."""
    # Per RFC 7638, thumbprint is SHA-256 of canonical JWK
    # For EC keys: {"crv", "kty", "x", "y"} in that order
    canonical = json.dumps(
        {
            "crv": jwk_from_ec["crv"],
            "kty": jwk_from_ec["kty"],
            "x": jwk_from_ec["x"],
            "y": jwk_from_ec["y"],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@pytest.fixture
def create_dpop_proof_fixture(ec_key_pair, jwk_from_ec):
    """Factory to create DPoP proofs."""
    import jwt

    def _create(
        htm: str = "POST",
        htu: str = "https://api.example.com/token",
        iat: int | None = None,
        jti: str | None = None,
        nonce: str | None = None,
        ath: str | None = None,
    ) -> str:
        if iat is None:
            iat = int(time.time())
        if jti is None:
            jti = base64.urlsafe_b64encode(hashlib.sha256(str(time.time()).encode()).digest()[:16]).decode()

        payload = {
            "htm": htm,
            "htu": htu,
            "iat": iat,
            "jti": jti,
        }

        if nonce is not None:
            payload["nonce"] = nonce

        if ath is not None:
            payload["ath"] = ath

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk_from_ec,
        }

        return jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

    return _create


@pytest.fixture
def validator() -> DPoPValidator:
    """Create DPoP validator."""
    return DPoPValidator()


# ============================================================================
# DPoP Proof Structure Tests
# ============================================================================


class TestDPoPProofStructure:
    """Test DPoP proof JWT structure validation."""

    def test_valid_dpop_proof_accepted(
        self, validator, create_dpop_proof_fixture
    ):
        """Valid DPoP proof should be accepted."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )

        assert result.method == "POST"
        assert result.uri == "https://api.example.com/token"

    def test_missing_typ_header_rejected(self, validator, ec_key_pair, jwk_from_ec):
        """DPoP proof without typ header should be rejected."""
        import jwt

        payload = {
            "htm": "POST",
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        # Missing typ header
        header = {
            "alg": "ES256",
            "jwk": jwk_from_ec,
        }

        proof = jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "typ" in str(exc_info.value).lower()

    def test_wrong_typ_header_rejected(self, validator, ec_key_pair, jwk_from_ec):
        """DPoP proof with wrong typ header should be rejected."""
        import jwt

        payload = {
            "htm": "POST",
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        header = {
            "typ": "JWT",  # Wrong - should be dpop+jwt
            "alg": "ES256",
            "jwk": jwk_from_ec,
        }

        proof = jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "typ" in str(exc_info.value).lower() or "dpop+jwt" in str(exc_info.value).lower()

    def test_missing_jwk_header_rejected(self, validator, ec_key_pair):
        """DPoP proof without jwk header should be rejected."""
        import jwt

        payload = {
            "htm": "POST",
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            # Missing jwk
        }

        proof = jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "jwk" in str(exc_info.value).lower()


# ============================================================================
# Method and URI Binding Tests
# ============================================================================


class TestMethodAndURIBinding:
    """Test DPoP proof method and URI binding."""

    def test_method_mismatch_rejected(
        self, validator, create_dpop_proof_fixture
    ):
        """DPoP proof with mismatched HTTP method should be rejected."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        with pytest.raises(DPoPMismatchError) as exc_info:
            validator.validate(
                proof,
                method="GET",  # Mismatch
                uri="https://api.example.com/token",
            )

        assert "method" in str(exc_info.value).lower()

    def test_uri_mismatch_rejected(
        self, validator, create_dpop_proof_fixture
    ):
        """DPoP proof with mismatched URI should be rejected."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        with pytest.raises(DPoPMismatchError) as exc_info:
            validator.validate(
                proof,
                method="POST",
                uri="https://api.example.com/other",  # Mismatch
            )

        assert "uri" in str(exc_info.value).lower()

    def test_case_insensitive_method_matching(
        self, validator, create_dpop_proof_fixture
    ):
        """HTTP method matching should be case-insensitive."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        # Should accept lowercase method
        result = validator.validate(
            proof,
            method="post",
            uri="https://api.example.com/token",
        )

        assert result.method == "POST"

    def test_uri_scheme_and_host_only(
        self, validator, create_dpop_proof_fixture
    ):
        """URI matching should compare scheme, host, and path."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        # Query string should be ignored per RFC 9449
        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token?param=value",
        )

        assert result.uri == "https://api.example.com/token"


# ============================================================================
# Expiration and Timing Tests
# ============================================================================


class TestExpirationAndTiming:
    """Test DPoP proof expiration and timing validation."""

    def test_expired_proof_rejected(
        self, validator, create_dpop_proof_fixture
    ):
        """Expired DPoP proof should be rejected."""
        old_iat = int(time.time()) - 300  # 5 minutes ago

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            iat=old_iat,
        )

        with pytest.raises(DPoPProofExpiredError):
            validator.validate(
                proof,
                method="POST",
                uri="https://api.example.com/token",
            )

    def test_future_iat_rejected(
        self, validator, create_dpop_proof_fixture
    ):
        """DPoP proof with future iat should be rejected."""
        future_iat = int(time.time()) + 300  # 5 minutes in future

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            iat=future_iat,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(
                proof,
                method="POST",
                uri="https://api.example.com/token",
            )

        assert "iat" in str(exc_info.value).lower() or "future" in str(exc_info.value).lower()

    def test_configurable_max_age(self, create_dpop_proof_fixture):
        """DPoP max age should be configurable."""
        # Create validator with 10 minute max age
        validator = DPoPValidator(max_age_seconds=600)

        old_iat = int(time.time()) - 300  # 5 minutes ago

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            iat=old_iat,
        )

        # Should be accepted with 10 minute max age
        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )

        assert result is not None

    def test_clock_skew_tolerance(self, create_dpop_proof_fixture):
        """DPoP should allow configurable clock skew tolerance."""
        validator = DPoPValidator(clock_skew_seconds=60)

        # Create proof 30 seconds in future
        future_iat = int(time.time()) + 30

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            iat=future_iat,
        )

        # Should be accepted with 60 second clock skew
        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )

        assert result is not None


# ============================================================================
# Nonce Handling Tests
# ============================================================================


class TestNonceHandling:
    """Test DPoP nonce handling for replay prevention."""

    def test_nonce_required_when_configured(self, create_dpop_proof_fixture):
        """DPoP nonce should be required when nonce provider is configured."""
        nonce_provider = MagicMock()
        nonce_provider.get_expected_nonce.return_value = "expected-nonce"

        validator = DPoPValidator(nonce_provider=nonce_provider)

        # Proof without nonce
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(
                proof,
                method="POST",
                uri="https://api.example.com/token",
            )

        assert "nonce" in str(exc_info.value).lower()

    def test_valid_nonce_accepted(self, create_dpop_proof_fixture):
        """DPoP proof with valid nonce should be accepted."""
        nonce_provider = MagicMock()
        nonce_provider.get_expected_nonce.return_value = "valid-nonce"
        nonce_provider.validate_nonce.return_value = True

        validator = DPoPValidator(nonce_provider=nonce_provider)

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            nonce="valid-nonce",
        )

        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )

        assert result.nonce == "valid-nonce"

    def test_invalid_nonce_rejected(self, create_dpop_proof_fixture):
        """DPoP proof with invalid nonce should be rejected."""
        nonce_provider = MagicMock()
        nonce_provider.get_expected_nonce.return_value = "expected-nonce"
        nonce_provider.validate_nonce.return_value = False

        validator = DPoPValidator(nonce_provider=nonce_provider)

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            nonce="wrong-nonce",
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(
                proof,
                method="POST",
                uri="https://api.example.com/token",
            )

        assert "nonce" in str(exc_info.value).lower()


# ============================================================================
# Replay Prevention Tests
# ============================================================================


class TestReplayPrevention:
    """Test DPoP replay prevention."""

    def test_jti_required(self, validator, ec_key_pair, jwk_from_ec):
        """DPoP proof must contain jti claim."""
        import jwt

        payload = {
            "htm": "POST",
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            # Missing jti
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk_from_ec,
        }

        proof = jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "jti" in str(exc_info.value).lower()

    def test_replay_detected(self, create_dpop_proof_fixture):
        """Replayed DPoP proof should be rejected."""
        replay_cache = MagicMock()
        replay_cache.check_and_store.side_effect = [True, False]  # First OK, second replay

        validator = DPoPValidator(replay_cache=replay_cache)

        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
            jti="unique-jti-123",
        )

        # First use should succeed
        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )
        assert result is not None

        # Second use should be rejected as replay
        with pytest.raises(DPoPReplayError):
            validator.validate(
                proof,
                method="POST",
                uri="https://api.example.com/token",
            )


# ============================================================================
# Access Token Binding Tests
# ============================================================================


class TestAccessTokenBinding:
    """Test DPoP binding to access tokens via ath claim."""

    def test_ath_claim_validated(self, create_dpop_proof_fixture):
        """DPoP ath claim should match access token hash."""
        access_token = "example-access-token-value"
        token_hash = hashlib.sha256(access_token.encode()).digest()
        ath = base64.urlsafe_b64encode(token_hash).rstrip(b"=").decode("ascii")

        validator = DPoPValidator()

        proof = create_dpop_proof_fixture(
            htm="GET",
            htu="https://api.example.com/resource",
            ath=ath,
        )

        result = validator.validate(
            proof,
            method="GET",
            uri="https://api.example.com/resource",
            access_token=access_token,
        )

        assert result.access_token_hash == ath

    def test_ath_mismatch_rejected(self, create_dpop_proof_fixture):
        """DPoP proof with wrong ath should be rejected."""
        access_token = "actual-access-token"
        wrong_ath = base64.urlsafe_b64encode(
            hashlib.sha256(b"different-token").digest()
        ).rstrip(b"=").decode("ascii")

        validator = DPoPValidator()

        proof = create_dpop_proof_fixture(
            htm="GET",
            htu="https://api.example.com/resource",
            ath=wrong_ath,
        )

        with pytest.raises(DPoPMismatchError) as exc_info:
            validator.validate(
                proof,
                method="GET",
                uri="https://api.example.com/resource",
                access_token=access_token,
            )

        assert "ath" in str(exc_info.value).lower() or "access token" in str(exc_info.value).lower()

    def test_ath_required_when_access_token_provided(self, create_dpop_proof_fixture):
        """DPoP proof must have ath when access token is provided."""
        validator = DPoPValidator()

        # Proof without ath claim
        proof = create_dpop_proof_fixture(
            htm="GET",
            htu="https://api.example.com/resource",
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(
                proof,
                method="GET",
                uri="https://api.example.com/resource",
                access_token="some-access-token",
            )

        assert "ath" in str(exc_info.value).lower()


# ============================================================================
# Key Thumbprint Tests
# ============================================================================


class TestKeyThumbprint:
    """Test DPoP key thumbprint calculation and validation."""

    def test_thumbprint_calculated_correctly(
        self, validator, create_dpop_proof_fixture, dpop_thumbprint
    ):
        """DPoP key thumbprint should be calculated per RFC 7638."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )

        assert result.key_thumbprint == dpop_thumbprint

    def test_cnf_claim_binding(
        self, validator, create_dpop_proof_fixture, dpop_thumbprint
    ):
        """DPoP key should match cnf claim in access token."""
        proof = create_dpop_proof_fixture(
            htm="GET",
            htu="https://api.example.com/resource",
        )

        # Validate that thumbprint can be used for cnf binding
        result = validator.validate(
            proof,
            method="GET",
            uri="https://api.example.com/resource",
            expected_thumbprint=dpop_thumbprint,
        )

        assert result.key_thumbprint == dpop_thumbprint

    def test_thumbprint_mismatch_rejected(
        self, validator, create_dpop_proof_fixture
    ):
        """DPoP proof with mismatched thumbprint should be rejected."""
        proof = create_dpop_proof_fixture(
            htm="GET",
            htu="https://api.example.com/resource",
        )

        with pytest.raises(DPoPMismatchError) as exc_info:
            validator.validate(
                proof,
                method="GET",
                uri="https://api.example.com/resource",
                expected_thumbprint="wrong-thumbprint",
            )

        assert "thumbprint" in str(exc_info.value).lower()


# ============================================================================
# Algorithm Validation Tests
# ============================================================================


class TestAlgorithmValidation:
    """Test DPoP algorithm validation."""

    def test_es256_accepted(self, validator, create_dpop_proof_fixture):
        """ES256 algorithm should be accepted."""
        proof = create_dpop_proof_fixture(
            htm="POST",
            htu="https://api.example.com/token",
        )

        result = validator.validate(
            proof,
            method="POST",
            uri="https://api.example.com/token",
        )

        assert result is not None

    def test_none_algorithm_rejected(self, validator, ec_key_pair, jwk_from_ec):
        """'none' algorithm should be rejected."""
        import jwt

        payload = {
            "htm": "POST",
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "none",
            "jwk": jwk_from_ec,
        }

        # Create unsigned token
        proof = jwt.encode(payload, None, algorithm="none", headers=header)

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "algorithm" in str(exc_info.value).lower()

    def test_symmetric_algorithm_rejected(self, validator, jwk_from_ec):
        """Symmetric algorithms should be rejected for DPoP."""
        import jwt

        payload = {
            "htm": "POST",
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "HS256",
            "jwk": jwk_from_ec,
        }

        proof = jwt.encode(payload, "secret", algorithm="HS256", headers=header)

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "algorithm" in str(exc_info.value).lower()


# ============================================================================
# DPoP Proof Creation Tests
# ============================================================================


class TestDPoPProofCreation:
    """Test DPoP proof creation helper."""

    def test_create_proof_basic(self, ec_key_pair):
        """create_dpop_proof should create valid proof."""
        proof = create_dpop_proof(
            private_key=ec_key_pair["private_pem"],
            method="POST",
            uri="https://api.example.com/token",
        )

        assert proof is not None
        assert proof.count(".") == 2  # Valid JWT structure

    def test_create_proof_with_nonce(self, ec_key_pair):
        """create_dpop_proof should include nonce when provided."""
        import jwt

        proof = create_dpop_proof(
            private_key=ec_key_pair["private_pem"],
            method="POST",
            uri="https://api.example.com/token",
            nonce="server-nonce",
        )

        # Decode and verify nonce
        claims = jwt.decode(proof, options={"verify_signature": False})
        assert claims["nonce"] == "server-nonce"

    def test_create_proof_with_access_token(self, ec_key_pair):
        """create_dpop_proof should include ath when access token provided."""
        import jwt

        access_token = "my-access-token"
        proof = create_dpop_proof(
            private_key=ec_key_pair["private_pem"],
            method="GET",
            uri="https://api.example.com/resource",
            access_token=access_token,
        )

        # Decode and verify ath
        claims = jwt.decode(proof, options={"verify_signature": False})
        expected_ath = base64.urlsafe_b64encode(
            hashlib.sha256(access_token.encode()).digest()
        ).rstrip(b"=").decode("ascii")

        assert claims["ath"] == expected_ath


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_proof_rejected(self, validator):
        """Malformed DPoP proof should be rejected."""
        with pytest.raises(DPoPValidationError):
            validator.validate("not-a-jwt", method="POST", uri="https://example.com")

    def test_empty_proof_rejected(self, validator):
        """Empty DPoP proof should be rejected."""
        with pytest.raises(DPoPValidationError):
            validator.validate("", method="POST", uri="https://example.com")

    def test_none_proof_rejected(self, validator):
        """None DPoP proof should be rejected."""
        with pytest.raises(DPoPValidationError):
            validator.validate(None, method="POST", uri="https://example.com")

    def test_missing_htm_rejected(self, validator, ec_key_pair, jwk_from_ec):
        """DPoP proof without htm claim should be rejected."""
        import jwt

        payload = {
            # Missing htm
            "htu": "https://api.example.com/token",
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk_from_ec,
        }

        proof = jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "htm" in str(exc_info.value).lower()

    def test_missing_htu_rejected(self, validator, ec_key_pair, jwk_from_ec):
        """DPoP proof without htu claim should be rejected."""
        import jwt

        payload = {
            "htm": "POST",
            # Missing htu
            "iat": int(time.time()),
            "jti": "unique-id",
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk_from_ec,
        }

        proof = jwt.encode(
            payload,
            ec_key_pair["private_pem"],
            algorithm="ES256",
            headers=header,
        )

        with pytest.raises(DPoPValidationError) as exc_info:
            validator.validate(proof, method="POST", uri="https://api.example.com/token")

        assert "htu" in str(exc_info.value).lower()
