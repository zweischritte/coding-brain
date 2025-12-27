"""
Tests for DPoP (Demonstrating Proof of Possession) validation.

TDD: These tests define the expected behavior for DPoP per RFC 9449.
Tests should fail until implementation is complete.
"""

import base64
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from jose import jwt as jose_jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

from app.security.types import AuthenticationError
from app.security.dpop import DPoPValidator, DPoPProof


def generate_ec_key_pair():
    """Generate an EC key pair for testing DPoP proofs."""
    if not HAS_CRYPTO:
        pytest.skip("cryptography not installed")

    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key


def create_dpop_proof(
    private_key,
    public_key,
    htm: str = "GET",
    htu: str = "https://api.example.com/resource",
    jti: str = None,
    iat: datetime = None,
    ath: str = None,
    nonce: str = None,
) -> str:
    """Create a DPoP proof JWT for testing."""
    if not HAS_CRYPTO:
        pytest.skip("cryptography not installed")

    # Export public key as JWK
    public_numbers = public_key.public_numbers()
    x = base64.urlsafe_b64encode(
        public_numbers.x.to_bytes(32, byteorder="big")
    ).decode().rstrip("=")
    y = base64.urlsafe_b64encode(
        public_numbers.y.to_bytes(32, byteorder="big")
    ).decode().rstrip("=")

    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "x": x,
        "y": y,
    }

    # Build DPoP proof header and payload
    header = {
        "typ": "dpop+jwt",
        "alg": "ES256",
        "jwk": jwk,
    }

    now = iat or datetime.now(timezone.utc)
    payload = {
        "jti": jti or f"dpop-jti-{int(time.time() * 1000)}",
        "htm": htm,
        "htu": htu,
        "iat": int(now.timestamp()),
    }

    if ath:
        payload["ath"] = ath

    if nonce:
        payload["nonce"] = nonce

    # Sign with private key
    # Using python-jose for signing
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)


def compute_ath(access_token: str) -> str:
    """Compute the access token hash (ath) for DPoP binding."""
    token_hash = hashlib.sha256(access_token.encode("ascii")).digest()
    return base64.urlsafe_b64encode(token_hash).decode().rstrip("=")


@pytest.fixture
def mock_cache():
    """Create a mock cache for DPoP replay prevention."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)  # No cached jti by default
    cache.setex = AsyncMock(return_value=True)
    return cache


@pytest.fixture
def dpop_validator(mock_cache):
    """Create a DPoP validator with mock cache."""
    return DPoPValidator(cache=mock_cache)


@pytest.fixture
def ec_key_pair():
    """Generate an EC key pair for testing."""
    if not HAS_CRYPTO:
        pytest.skip("cryptography not installed")
    return generate_ec_key_pair()


class TestDPoPProofValidation:
    """Tests for DPoP proof structure validation."""

    @pytest.mark.asyncio
    async def test_rejects_missing_typ_header(self, dpop_validator, ec_key_pair):
        """DPoP proof must have typ='dpop+jwt' header."""
        private_key, public_key = ec_key_pair

        # Create proof without typ header (manually)
        public_numbers = public_key.public_numbers()
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": base64.urlsafe_b64encode(
                public_numbers.x.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
            "y": base64.urlsafe_b64encode(
                public_numbers.y.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
        }

        header = {
            # Missing typ: "dpop+jwt"
            "alg": "ES256",
            "jwk": jwk,
        }

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        payload = {
            "jti": "test-jti",
            "htm": "GET",
            "htu": "https://api.example.com/resource",
            "iat": int(datetime.now(timezone.utc).timestamp()),
        }

        proof = jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)

        with pytest.raises(AuthenticationError) as exc_info:
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

        assert "typ" in str(exc_info.value.message).lower() or "invalid" in str(exc_info.value.message).lower()

    @pytest.mark.asyncio
    async def test_rejects_missing_jwk_header(self, dpop_validator, ec_key_pair):
        """DPoP proof must include JWK in header."""
        private_key, _ = ec_key_pair

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            # Missing jwk
        }

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        payload = {
            "jti": "test-jti",
            "htm": "GET",
            "htu": "https://api.example.com/resource",
            "iat": int(datetime.now(timezone.utc).timestamp()),
        }

        proof = jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

    @pytest.mark.asyncio
    async def test_rejects_jwk_with_private_key(self, dpop_validator, ec_key_pair):
        """DPoP JWK must not contain private key material."""
        private_key, public_key = ec_key_pair

        public_numbers = public_key.public_numbers()
        # Include 'd' parameter (private key) - this should be rejected
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": base64.urlsafe_b64encode(
                public_numbers.x.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
            "y": base64.urlsafe_b64encode(
                public_numbers.y.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
            "d": "private-key-material",  # This should cause rejection
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk,
        }

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        payload = {
            "jti": "test-jti",
            "htm": "GET",
            "htu": "https://api.example.com/resource",
            "iat": int(datetime.now(timezone.utc).timestamp()),
        }

        proof = jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")


class TestDPoPPayloadValidation:
    """Tests for DPoP proof payload validation."""

    @pytest.mark.asyncio
    async def test_rejects_missing_jti(self, dpop_validator, ec_key_pair):
        """DPoP proof must have jti claim."""
        private_key, public_key = ec_key_pair

        public_numbers = public_key.public_numbers()
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": base64.urlsafe_b64encode(
                public_numbers.x.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
            "y": base64.urlsafe_b64encode(
                public_numbers.y.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk,
        }

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        payload = {
            # Missing jti
            "htm": "GET",
            "htu": "https://api.example.com/resource",
            "iat": int(datetime.now(timezone.utc).timestamp()),
        }

        proof = jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

    @pytest.mark.asyncio
    async def test_rejects_missing_htm(self, dpop_validator, ec_key_pair):
        """DPoP proof must have htm (HTTP method) claim."""
        private_key, public_key = ec_key_pair

        public_numbers = public_key.public_numbers()
        jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": base64.urlsafe_b64encode(
                public_numbers.x.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
            "y": base64.urlsafe_b64encode(
                public_numbers.y.to_bytes(32, byteorder="big")
            ).decode().rstrip("="),
        }

        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": jwk,
        }

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        payload = {
            "jti": "test-jti",
            # Missing htm
            "htu": "https://api.example.com/resource",
            "iat": int(datetime.now(timezone.utc).timestamp()),
        }

        proof = jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

    @pytest.mark.asyncio
    async def test_rejects_mismatched_htm(self, dpop_validator, ec_key_pair):
        """DPoP htm must match the actual HTTP method."""
        private_key, public_key = ec_key_pair

        proof = create_dpop_proof(
            private_key, public_key,
            htm="POST",  # Proof says POST
            htu="https://api.example.com/resource",
        )

        # But request is GET
        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

    @pytest.mark.asyncio
    async def test_rejects_mismatched_htu(self, dpop_validator, ec_key_pair):
        """DPoP htu must match the actual request URI."""
        private_key, public_key = ec_key_pair

        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/other",  # Proof says /other
        )

        # But request is to /resource
        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

    @pytest.mark.asyncio
    async def test_rejects_expired_proof(self, dpop_validator, ec_key_pair):
        """DPoP proof iat must be recent (within MAX_AGE_SECONDS)."""
        private_key, public_key = ec_key_pair

        old_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            iat=old_time,
        )

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

    @pytest.mark.asyncio
    async def test_rejects_future_proof(self, dpop_validator, ec_key_pair):
        """DPoP proof iat must not be in the future."""
        private_key, public_key = ec_key_pair

        future_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            iat=future_time,
        )

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")


class TestDPoPReplayPrevention:
    """Tests for DPoP replay prevention via Valkey."""

    @pytest.mark.asyncio
    async def test_caches_jti_after_validation(self, dpop_validator, mock_cache, ec_key_pair):
        """Validated DPoP jti should be cached to prevent replay."""
        private_key, public_key = ec_key_pair

        jti = "unique-dpop-jti-12345"
        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            jti=jti,
        )

        await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

        # Verify jti was cached
        mock_cache.setex.assert_called_once()
        call_args = mock_cache.setex.call_args
        assert jti in call_args[0][0]  # Key contains jti
        assert call_args[0][1] >= 60  # TTL at least 60 seconds

    @pytest.mark.asyncio
    async def test_rejects_replayed_jti(self, dpop_validator, mock_cache, ec_key_pair):
        """Should reject DPoP proof with previously used jti."""
        private_key, public_key = ec_key_pair

        jti = "replayed-jti-12345"

        # Simulate jti already in cache
        mock_cache.get = AsyncMock(return_value="1")

        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            jti=jti,
        )

        with pytest.raises(AuthenticationError) as exc_info:
            await dpop_validator.validate(proof, "GET", "https://api.example.com/resource")

        assert "replay" in str(exc_info.value.message).lower() or "reuse" in str(exc_info.value.message).lower()


class TestDPoPAccessTokenBinding:
    """Tests for DPoP access token binding (ath claim)."""

    @pytest.mark.asyncio
    async def test_validates_ath_claim(self, dpop_validator, ec_key_pair):
        """ath claim must match the hash of the access token."""
        private_key, public_key = ec_key_pair

        access_token = "test-access-token-12345"
        correct_ath = compute_ath(access_token)

        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            ath=correct_ath,
        )

        # Should not raise
        result = await dpop_validator.validate(
            proof, "GET", "https://api.example.com/resource",
            access_token=access_token
        )
        assert result.ath == correct_ath

    @pytest.mark.asyncio
    async def test_rejects_incorrect_ath(self, dpop_validator, ec_key_pair):
        """Should reject proof with incorrect ath (access token hash)."""
        private_key, public_key = ec_key_pair

        access_token = "test-access-token-12345"
        wrong_ath = compute_ath("different-token")

        proof = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            ath=wrong_ath,
        )

        with pytest.raises(AuthenticationError):
            await dpop_validator.validate(
                proof, "GET", "https://api.example.com/resource",
                access_token=access_token
            )


class TestDPoPThumbprint:
    """Tests for JWK thumbprint extraction."""

    @pytest.mark.asyncio
    async def test_extracts_consistent_thumbprint(self, dpop_validator, ec_key_pair):
        """Thumbprint extraction should be consistent for same key."""
        private_key, public_key = ec_key_pair

        proof1 = create_dpop_proof(
            private_key, public_key,
            htm="GET",
            htu="https://api.example.com/resource",
            jti="jti-1",
        )

        proof2 = create_dpop_proof(
            private_key, public_key,
            htm="POST",
            htu="https://api.example.com/other",
            jti="jti-2",
        )

        thumbprint1 = await dpop_validator.get_thumbprint(proof1)
        thumbprint2 = await dpop_validator.get_thumbprint(proof2)

        assert thumbprint1 == thumbprint2

    @pytest.mark.asyncio
    async def test_different_keys_different_thumbprints(self, dpop_validator):
        """Different keys should produce different thumbprints."""
        if not HAS_CRYPTO:
            pytest.skip("cryptography not installed")

        private_key1, public_key1 = generate_ec_key_pair()
        private_key2, public_key2 = generate_ec_key_pair()

        proof1 = create_dpop_proof(
            private_key1, public_key1,
            htm="GET",
            htu="https://api.example.com/resource",
        )

        proof2 = create_dpop_proof(
            private_key2, public_key2,
            htm="GET",
            htu="https://api.example.com/resource",
        )

        thumbprint1 = await dpop_validator.get_thumbprint(proof1)
        thumbprint2 = await dpop_validator.get_thumbprint(proof2)

        assert thumbprint1 != thumbprint2


class TestDPoPValidatorIntegration:
    """Integration tests for DPoP validation flow."""

    @pytest.mark.asyncio
    async def test_full_validation_flow(self, dpop_validator, mock_cache, ec_key_pair):
        """Test complete DPoP validation with all checks."""
        private_key, public_key = ec_key_pair

        access_token = "access-token-12345"
        ath = compute_ath(access_token)

        proof = create_dpop_proof(
            private_key, public_key,
            htm="POST",
            htu="https://api.example.com/memories",
            ath=ath,
        )

        result = await dpop_validator.validate(
            proof,
            http_method="POST",
            http_uri="https://api.example.com/memories",
            access_token=access_token,
        )

        assert isinstance(result, DPoPProof)
        assert result.htm == "POST"
        assert result.htu == "https://api.example.com/memories"
        assert result.ath == ath
        assert result.jti is not None
