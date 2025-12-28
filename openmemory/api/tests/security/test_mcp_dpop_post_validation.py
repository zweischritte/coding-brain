"""
Tests for DPoP validation on MCP POST requests.

When a session was bound with a DPoP thumbprint at GET time,
subsequent POST requests MUST include a valid DPoP proof with matching thumbprint.
"""
import base64
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from jose import jwt as jose_jwt
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

from app.security.session_binding import (
    MemorySessionBindingStore,
    get_session_binding_store,
    reset_session_binding_store,
)


def run_async(coro):
    """Run async helper in sync tests (Python 3.14 has no default loop)."""
    import asyncio

    return asyncio.run(coro)


# Test constants
TEST_SESSION_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TEST_USER_ID = "user-123"
TEST_ORG_ID = "org-abc"


def generate_ec_key_pair():
    """Generate an EC key pair for testing DPoP proofs."""
    if not HAS_CRYPTO:
        pytest.skip("cryptography not installed")

    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    return private_key, public_key


def compute_jwk_thumbprint(public_key) -> str:
    """Compute JWK thumbprint per RFC 7638."""
    if not HAS_CRYPTO:
        pytest.skip("cryptography not installed")

    public_numbers = public_key.public_numbers()
    x = base64.urlsafe_b64encode(
        public_numbers.x.to_bytes(32, byteorder="big")
    ).decode().rstrip("=")
    y = base64.urlsafe_b64encode(
        public_numbers.y.to_bytes(32, byteorder="big")
    ).decode().rstrip("=")

    # JWK thumbprint uses lexicographically sorted keys
    thumbprint_input = {
        "crv": "P-256",
        "kty": "EC",
        "x": x,
        "y": y,
    }

    canonical = json.dumps(thumbprint_input, sort_keys=True, separators=(",", ":"))
    thumbprint_hash = hashlib.sha256(canonical.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(thumbprint_hash).decode().rstrip("=")


def create_dpop_proof(
    private_key,
    public_key,
    htm: str = "POST",
    htu: str = "https://api.example.com/mcp/messages",
    jti: str = None,
    iat: datetime = None,
) -> str:
    """Create a DPoP proof JWT for testing."""
    if not HAS_CRYPTO:
        pytest.skip("cryptography not installed")

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

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return jose_jwt.encode(payload, private_pem, algorithm="ES256", headers=header)


class TestDPoPValidationOnPost:
    """Tests for DPoP validation during POST message handling."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_post_succeeds_without_dpop_when_binding_has_none(self):
        """POST without DPoP should succeed when session wasn't bound with DPoP."""
        store = get_session_binding_store()

        # Create session without DPoP binding
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=None,
        )

        # Validate without DPoP should succeed
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is True

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_post_fails_without_dpop_when_binding_has_dpop(self):
        """POST without DPoP should fail when session was bound with DPoP."""
        store = get_session_binding_store()
        private_key, public_key = generate_ec_key_pair()
        thumbprint = compute_jwk_thumbprint(public_key)

        # Create session with DPoP binding
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint,
        )

        # Validate without DPoP should fail
        is_valid = store.validate(TEST_SESSION_ID, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is False

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_post_succeeds_with_matching_dpop_thumbprint(self):
        """POST with matching DPoP should succeed."""
        store = get_session_binding_store()
        private_key, public_key = generate_ec_key_pair()
        thumbprint = compute_jwk_thumbprint(public_key)

        # Create session with DPoP binding
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint,
        )

        # Validate with matching thumbprint should succeed
        is_valid = store.validate(
            TEST_SESSION_ID,
            TEST_USER_ID,
            TEST_ORG_ID,
            dpop_thumbprint=thumbprint,
        )
        assert is_valid is True

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_post_fails_with_wrong_dpop_thumbprint(self):
        """POST with wrong DPoP key should fail."""
        store = get_session_binding_store()

        # Create session with one key
        private_key1, public_key1 = generate_ec_key_pair()
        thumbprint1 = compute_jwk_thumbprint(public_key1)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint1,
        )

        # Validate with different key should fail
        private_key2, public_key2 = generate_ec_key_pair()
        thumbprint2 = compute_jwk_thumbprint(public_key2)

        is_valid = store.validate(
            TEST_SESSION_ID,
            TEST_USER_ID,
            TEST_ORG_ID,
            dpop_thumbprint=thumbprint2,
        )
        assert is_valid is False


class TestDPoPProofExtraction:
    """Tests for extracting and validating DPoP proof from headers."""

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_proof_thumbprint_matches_bound_key(self):
        """DPoP proof thumbprint should be extractable and match binding."""
        from app.security.dpop import DPoPValidator

        private_key, public_key = generate_ec_key_pair()
        expected_thumbprint = compute_jwk_thumbprint(public_key)

        # Create a DPoP proof
        dpop_proof = create_dpop_proof(
            private_key=private_key,
            public_key=public_key,
            htm="POST",
            htu="https://api.example.com/mcp/messages",
        )

        # Create a mock cache for the validator
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.setex = AsyncMock(return_value=True)

        validator = DPoPValidator(mock_cache)

        # Extract thumbprint
        thumbprint = run_async(validator.get_thumbprint(dpop_proof))

        assert thumbprint == expected_thumbprint

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_proof_validation_succeeds_with_correct_method_and_uri(self):
        """DPoP proof validation should succeed with correct htm and htu."""
        from app.security.dpop import DPoPValidator

        private_key, public_key = generate_ec_key_pair()

        # Create a DPoP proof for POST to specific URI
        dpop_proof = create_dpop_proof(
            private_key=private_key,
            public_key=public_key,
            htm="POST",
            htu="https://api.example.com/mcp/messages",
        )

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.setex = AsyncMock(return_value=True)

        validator = DPoPValidator(mock_cache)

        proof = run_async(
            validator.validate(
                dpop_proof=dpop_proof,
                http_method="POST",
                http_uri="https://api.example.com/mcp/messages",
            )
        )

        assert proof.htm == "POST"
        assert proof.htu == "https://api.example.com/mcp/messages"

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_proof_validation_fails_with_wrong_method(self):
        """DPoP proof validation should fail with wrong HTTP method."""
        from app.security.dpop import DPoPValidator
        from app.security.types import AuthenticationError

        private_key, public_key = generate_ec_key_pair()

        # Create a DPoP proof for GET
        dpop_proof = create_dpop_proof(
            private_key=private_key,
            public_key=public_key,
            htm="GET",
            htu="https://api.example.com/mcp/messages",
        )

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.setex = AsyncMock(return_value=True)

        validator = DPoPValidator(mock_cache)

        with pytest.raises(AuthenticationError) as exc_info:
            run_async(
                validator.validate(
                    dpop_proof=dpop_proof,
                    http_method="POST",  # Different method
                    http_uri="https://api.example.com/mcp/messages",
                )
            )

        assert "htm mismatch" in str(exc_info.value.message)

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_proof_validation_fails_with_wrong_uri(self):
        """DPoP proof validation should fail with wrong HTTP URI."""
        from app.security.dpop import DPoPValidator
        from app.security.types import AuthenticationError

        private_key, public_key = generate_ec_key_pair()

        # Create a DPoP proof for one URI
        dpop_proof = create_dpop_proof(
            private_key=private_key,
            public_key=public_key,
            htm="POST",
            htu="https://api.example.com/mcp/messages",
        )

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.setex = AsyncMock(return_value=True)

        validator = DPoPValidator(mock_cache)

        with pytest.raises(AuthenticationError) as exc_info:
            run_async(
                validator.validate(
                    dpop_proof=dpop_proof,
                    http_method="POST",
                    http_uri="https://api.example.com/other/endpoint",  # Different URI
                )
            )

        assert "htu mismatch" in str(exc_info.value.message)


class TestDPoPPostHandlerIntegration:
    """Integration tests for DPoP validation in POST handler."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_full_dpop_binding_flow(self):
        """Test complete DPoP binding from GET through POST."""
        from app.security.dpop import DPoPValidator

        store = get_session_binding_store()
        private_key, public_key = generate_ec_key_pair()

        # At GET time: Extract thumbprint and bind to session
        get_dpop_proof = create_dpop_proof(
            private_key=private_key,
            public_key=public_key,
            htm="GET",
            htu="https://api.example.com/mcp/sse",
        )

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.setex = AsyncMock(return_value=True)

        validator = DPoPValidator(mock_cache)

        thumbprint = run_async(validator.get_thumbprint(get_dpop_proof))

        # Create session with DPoP binding
        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint,
        )

        # At POST time: Validate DPoP proof has matching thumbprint
        post_dpop_proof = create_dpop_proof(
            private_key=private_key,  # Same key
            public_key=public_key,
            htm="POST",
            htu="https://api.example.com/mcp/messages",
        )

        post_thumbprint = run_async(validator.get_thumbprint(post_dpop_proof))

        # Validate session binding with DPoP
        is_valid = store.validate(
            TEST_SESSION_ID,
            TEST_USER_ID,
            TEST_ORG_ID,
            dpop_thumbprint=post_thumbprint,
        )
        assert is_valid is True

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_binding_fails_with_different_key_on_post(self):
        """POST with different DPoP key than GET should fail."""
        from app.security.dpop import DPoPValidator

        store = get_session_binding_store()

        # GET with key1
        private_key1, public_key1 = generate_ec_key_pair()
        thumbprint1 = compute_jwk_thumbprint(public_key1)

        store.create(
            session_id=TEST_SESSION_ID,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint1,
        )

        # POST with key2 (different key)
        private_key2, public_key2 = generate_ec_key_pair()
        thumbprint2 = compute_jwk_thumbprint(public_key2)

        is_valid = store.validate(
            TEST_SESSION_ID,
            TEST_USER_ID,
            TEST_ORG_ID,
            dpop_thumbprint=thumbprint2,
        )
        assert is_valid is False
