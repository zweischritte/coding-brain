"""
End-to-end tests for MCP SSE session binding.

Tests the full session lifecycle with real SSE-like connections:
- GET → capture session_id → POST with session_id
- Session hijacking prevention
- Session expiry handling
- DPoP binding verification
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


# Test constants
TEST_USER_ID = "user-e2e-test"
TEST_USER_ID_2 = "user-e2e-attacker"
TEST_ORG_ID = "org-e2e-test"
TEST_ORG_ID_2 = "org-e2e-attacker"


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

    thumbprint_input = {
        "crv": "P-256",
        "kty": "EC",
        "x": x,
        "y": y,
    }

    canonical = json.dumps(thumbprint_input, sort_keys=True, separators=(",", ":"))
    thumbprint_hash = hashlib.sha256(canonical.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(thumbprint_hash).decode().rstrip("=")


class TestMcpSseSessionLifecycle:
    """E2E tests for full MCP SSE session lifecycle."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    def test_full_session_flow_success(self):
        """Test complete GET → POST flow with same user succeeds."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Step 1: GET handler creates session binding
        binding = store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )
        assert binding is not None
        assert binding.session_id == session_id

        # Step 2: POST handler validates session binding
        is_valid = store.validate(session_id, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is True

    def test_session_hijacking_attempt_fails(self):
        """Test that different user cannot use another's session."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Step 1: User A creates session via GET
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Step 2: User B tries to use User A's session via POST
        is_valid = store.validate(session_id, TEST_USER_ID_2, TEST_ORG_ID_2)
        assert is_valid is False

    def test_cross_org_session_hijacking_fails(self):
        """Test that same user in different org cannot hijack session."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Create session for user in org A
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Same user in different org cannot use session
        is_valid = store.validate(session_id, TEST_USER_ID, TEST_ORG_ID_2)
        assert is_valid is False

    def test_session_expiry_prevents_access(self):
        """Test that expired sessions cannot be accessed."""
        # Use 1 second TTL
        store = MemorySessionBindingStore(default_ttl_seconds=1)
        session_id = uuid.uuid4()

        # Create session
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Wait for expiry
        time.sleep(1.1)

        # Validation should fail for expired session
        is_valid = store.validate(session_id, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is False

    def test_disconnect_cleanup_prevents_reuse(self):
        """Test that cleaned up session cannot be reused."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Create session
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Simulate disconnect cleanup
        store.delete(session_id)

        # Session should no longer be valid
        is_valid = store.validate(session_id, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is False

    def test_multiple_concurrent_sessions(self):
        """Test that multiple sessions for same user work independently."""
        store = get_session_binding_store()
        session_1 = uuid.uuid4()
        session_2 = uuid.uuid4()

        # User creates two concurrent sessions
        store.create(session_id=session_1, user_id=TEST_USER_ID, org_id=TEST_ORG_ID)
        store.create(session_id=session_2, user_id=TEST_USER_ID, org_id=TEST_ORG_ID)

        # Both sessions should be valid
        assert store.validate(session_1, TEST_USER_ID, TEST_ORG_ID) is True
        assert store.validate(session_2, TEST_USER_ID, TEST_ORG_ID) is True

        # Delete one session
        store.delete(session_1)

        # Only deleted session should be invalid
        assert store.validate(session_1, TEST_USER_ID, TEST_ORG_ID) is False
        assert store.validate(session_2, TEST_USER_ID, TEST_ORG_ID) is True


class TestMcpSseDpopBinding:
    """E2E tests for DPoP binding through session lifecycle."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_bound_session_requires_dpop_on_post(self):
        """Test that DPoP-bound session rejects POST without DPoP."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Generate key pair
        private_key, public_key = generate_ec_key_pair()
        thumbprint = compute_jwk_thumbprint(public_key)

        # Create session with DPoP binding
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint,
        )

        # POST without DPoP should fail
        is_valid = store.validate(session_id, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is False

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_bound_session_accepts_matching_key(self):
        """Test that DPoP-bound session accepts POST with matching key."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Generate key pair
        private_key, public_key = generate_ec_key_pair()
        thumbprint = compute_jwk_thumbprint(public_key)

        # Create session with DPoP binding
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint,
        )

        # POST with matching DPoP should succeed
        is_valid = store.validate(
            session_id, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint=thumbprint
        )
        assert is_valid is True

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_dpop_bound_session_rejects_different_key(self):
        """Test that DPoP-bound session rejects POST with different key."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Generate two key pairs
        private_key1, public_key1 = generate_ec_key_pair()
        thumbprint1 = compute_jwk_thumbprint(public_key1)

        private_key2, public_key2 = generate_ec_key_pair()
        thumbprint2 = compute_jwk_thumbprint(public_key2)

        # Create session with first key
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
            dpop_thumbprint=thumbprint1,
        )

        # POST with second key should fail
        is_valid = store.validate(
            session_id, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint=thumbprint2
        )
        assert is_valid is False

    @pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")
    def test_non_dpop_session_accepts_request_with_dpop(self):
        """Test that non-DPoP session accepts POST even with DPoP header."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # Create session without DPoP binding
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Generate key pair
        private_key, public_key = generate_ec_key_pair()
        thumbprint = compute_jwk_thumbprint(public_key)

        # POST with DPoP should still succeed (DPoP not required)
        is_valid = store.validate(
            session_id, TEST_USER_ID, TEST_ORG_ID,
            dpop_thumbprint=thumbprint
        )
        assert is_valid is True


class TestMcpSseSessionIsolation:
    """E2E tests for session isolation and security boundaries."""

    def setup_method(self):
        """Reset session binding store before each test."""
        reset_session_binding_store()

    def teardown_method(self):
        """Reset session binding store after each test."""
        reset_session_binding_store()

    def test_session_id_guessing_attack_fails(self):
        """Test that guessing session IDs doesn't work."""
        store = get_session_binding_store()

        # Create a real session
        real_session = uuid.uuid4()
        store.create(
            session_id=real_session,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # Try to guess/brute force session IDs
        for _ in range(100):
            guessed_session = uuid.uuid4()
            is_valid = store.validate(
                guessed_session, TEST_USER_ID, TEST_ORG_ID
            )
            # All guesses should fail (statistically impossible to guess UUID4)
            assert is_valid is False

    def test_replay_same_session_id_fails_after_overwrite(self):
        """Test that replaying old session after reconnect fails."""
        store = get_session_binding_store()
        session_id = uuid.uuid4()

        # User A creates original session
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID,
            org_id=TEST_ORG_ID,
        )

        # User B manages to reuse the same session_id (unlikely but test edge case)
        store.create(
            session_id=session_id,
            user_id=TEST_USER_ID_2,
            org_id=TEST_ORG_ID_2,
        )

        # User A's credentials should no longer work
        is_valid = store.validate(session_id, TEST_USER_ID, TEST_ORG_ID)
        assert is_valid is False

        # User B's credentials should work
        is_valid = store.validate(session_id, TEST_USER_ID_2, TEST_ORG_ID_2)
        assert is_valid is True

    def test_concurrent_users_different_sessions(self):
        """Test that multiple users can have concurrent sessions."""
        store = get_session_binding_store()

        # Create sessions for multiple users
        sessions = {}
        for i in range(10):
            session_id = uuid.uuid4()
            user_id = f"user-{i}"
            org_id = f"org-{i}"
            store.create(session_id=session_id, user_id=user_id, org_id=org_id)
            sessions[session_id] = (user_id, org_id)

        # Each user should only be able to validate their own session
        for session_id, (user_id, org_id) in sessions.items():
            # Own session validates
            assert store.validate(session_id, user_id, org_id) is True

            # Other users' sessions don't validate
            for other_sid, (other_uid, other_oid) in sessions.items():
                if other_sid != session_id:
                    assert store.validate(session_id, other_uid, other_oid) is False
