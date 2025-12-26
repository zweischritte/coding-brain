"""Tests for JWT validator.

Tests cover OAuth 2.1 requirements per implementation plan:
- JWT claim validation (iss, aud, exp, iat, nbf, sub)
- PKCE S256 code verifier validation
- Token expiration checks
- Audience validation
- Issuer validation
- JWKs-based signature validation
- Revocation checking
"""

import base64
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# These imports will fail until implementation exists
from openmemory.api.security.jwt_validator import (
    JWTValidator,
    JWTValidationError,
    JWTExpiredError,
    JWTInvalidClaimsError,
    JWTInvalidSignatureError,
    JWTRevokedError,
    PKCEValidator,
    PKCEValidationError,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def valid_issuer() -> str:
    """Valid issuer URL."""
    return "https://auth.example.com"


@pytest.fixture
def valid_audience() -> str:
    """Valid audience."""
    return "openmemory-api"


@pytest.fixture
def rsa_key_pair():
    """Generate RSA key pair for testing."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
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
def jwk_from_rsa(rsa_key_pair) -> dict[str, Any]:
    """Convert RSA public key to JWK format."""
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

    public_key: RSAPublicKey = rsa_key_pair["public_key"]
    numbers = public_key.public_numbers()

    def _int_to_base64url(n: int, length: int) -> str:
        data = n.to_bytes(length, byteorder="big")
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    return {
        "kty": "RSA",
        "kid": "test-key-id",
        "use": "sig",
        "alg": "RS256",
        "n": _int_to_base64url(numbers.n, 256),
        "e": _int_to_base64url(numbers.e, 3),
    }


@pytest.fixture
def jwks_response(jwk_from_rsa) -> dict[str, Any]:
    """Mock JWKS response."""
    return {"keys": [jwk_from_rsa]}


@pytest.fixture
def valid_claims(valid_issuer, valid_audience) -> dict[str, Any]:
    """Valid JWT claims."""
    now = int(time.time())
    return {
        "sub": "user-123",
        "iss": valid_issuer,
        "aud": valid_audience,
        "exp": now + 3600,  # 1 hour from now
        "iat": now,
        "nbf": now,
        "org_id": "org-456",
        "enterprise_id": "ent-789",
        "roles": ["user"],
        "scopes": ["repository:read"],
        "team_ids": ["team-1"],
        "project_ids": ["proj-1"],
    }


@pytest.fixture
def create_signed_jwt(rsa_key_pair):
    """Factory to create signed JWTs."""
    import jwt

    def _create(claims: dict[str, Any], kid: str = "test-key-id") -> str:
        return jwt.encode(
            claims,
            rsa_key_pair["private_pem"],
            algorithm="RS256",
            headers={"kid": kid},
        )

    return _create


@pytest.fixture
def validator(valid_issuer, valid_audience, jwks_response) -> JWTValidator:
    """Create JWT validator with mocked JWKS endpoint."""
    validator = JWTValidator(
        issuer_allowlist=[valid_issuer],
        audience=valid_audience,
    )
    # Mock the JWKS fetching
    validator._jwks_cache = {valid_issuer: jwks_response}
    return validator


# ============================================================================
# JWT Claim Validation Tests
# ============================================================================


class TestJWTClaimValidation:
    """Test JWT mandatory claim validation."""

    def test_valid_jwt_with_all_required_claims(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with all required claims should validate successfully."""
        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)

        assert result.subject == "user-123"
        assert result.issuer == valid_claims["iss"]
        assert result.audience == valid_claims["aud"]
        assert result.org_id == "org-456"
        assert result.roles == ["user"]
        assert result.scopes == ["repository:read"]

    def test_missing_sub_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'sub' claim should raise JWTInvalidClaimsError."""
        del valid_claims["sub"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "sub" in str(exc_info.value)

    def test_missing_iss_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'iss' claim should raise JWTInvalidClaimsError."""
        del valid_claims["iss"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "iss" in str(exc_info.value)

    def test_missing_aud_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'aud' claim should raise JWTInvalidClaimsError."""
        del valid_claims["aud"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "aud" in str(exc_info.value)

    def test_missing_exp_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'exp' claim should raise JWTInvalidClaimsError."""
        del valid_claims["exp"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "exp" in str(exc_info.value)

    def test_missing_iat_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'iat' claim should raise JWTInvalidClaimsError."""
        del valid_claims["iat"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "iat" in str(exc_info.value)

    def test_missing_nbf_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'nbf' claim should raise JWTInvalidClaimsError."""
        del valid_claims["nbf"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "nbf" in str(exc_info.value)

    def test_missing_org_id_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'org_id' claim should raise JWTInvalidClaimsError."""
        del valid_claims["org_id"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "org_id" in str(exc_info.value)

    def test_missing_roles_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'roles' claim should raise JWTInvalidClaimsError."""
        del valid_claims["roles"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "roles" in str(exc_info.value)

    def test_missing_scopes_claim_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT without 'scopes' claim should raise JWTInvalidClaimsError."""
        del valid_claims["scopes"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "scopes" in str(exc_info.value)


# ============================================================================
# Token Expiration Tests
# ============================================================================


class TestTokenExpiration:
    """Test JWT expiration handling."""

    def test_expired_token_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """Expired JWT should raise JWTExpiredError."""
        valid_claims["exp"] = int(time.time()) - 3600  # 1 hour ago
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTExpiredError):
            validator.validate(token)

    def test_token_not_yet_valid_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with future 'nbf' should raise JWTInvalidClaimsError."""
        valid_claims["nbf"] = int(time.time()) + 3600  # 1 hour from now
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "nbf" in str(exc_info.value).lower() or "not yet valid" in str(
            exc_info.value
        ).lower()

    def test_token_with_future_iat_raises_error(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with future 'iat' should raise JWTInvalidClaimsError."""
        valid_claims["iat"] = int(time.time()) + 3600  # 1 hour from now
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "iat" in str(exc_info.value).lower()

    def test_token_expiring_soon_is_valid(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT expiring in 1 minute should still be valid."""
        valid_claims["exp"] = int(time.time()) + 60  # 1 minute from now
        token = create_signed_jwt(valid_claims)

        result = validator.validate(token)
        assert result.subject == "user-123"

    def test_token_max_lifetime_enforcement(
        self, valid_issuer, valid_audience, jwks_response, create_signed_jwt, valid_claims
    ):
        """JWT with exp > 1 hour should be rejected when enforced."""
        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
            max_token_lifetime_seconds=3600,  # 1 hour max
        )
        validator._jwks_cache = {valid_issuer: jwks_response}

        # Token valid for 2 hours
        valid_claims["exp"] = int(time.time()) + 7200
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "lifetime" in str(exc_info.value).lower()


# ============================================================================
# Issuer Validation Tests
# ============================================================================


class TestIssuerValidation:
    """Test JWT issuer validation."""

    def test_valid_issuer_accepted(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with allowed issuer should be accepted."""
        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)
        assert result.issuer == valid_claims["iss"]

    def test_invalid_issuer_rejected(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with disallowed issuer should be rejected."""
        valid_claims["iss"] = "https://evil.example.com"
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "issuer" in str(exc_info.value).lower()

    def test_multiple_issuers_allowlist(
        self, valid_audience, jwks_response, create_signed_jwt, valid_claims
    ):
        """Validator should accept multiple allowed issuers."""
        issuer1 = "https://auth1.example.com"
        issuer2 = "https://auth2.example.com"

        validator = JWTValidator(
            issuer_allowlist=[issuer1, issuer2],
            audience=valid_audience,
        )
        validator._jwks_cache = {
            issuer1: jwks_response,
            issuer2: jwks_response,
        }

        # Test both issuers
        for issuer in [issuer1, issuer2]:
            valid_claims["iss"] = issuer
            token = create_signed_jwt(valid_claims)
            result = validator.validate(token)
            assert result.issuer == issuer


# ============================================================================
# Audience Validation Tests
# ============================================================================


class TestAudienceValidation:
    """Test JWT audience validation."""

    def test_valid_audience_accepted(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with correct audience should be accepted."""
        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)
        assert result.audience == valid_claims["aud"]

    def test_invalid_audience_rejected(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with wrong audience should be rejected."""
        valid_claims["aud"] = "wrong-audience"
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "audience" in str(exc_info.value).lower()

    def test_audience_as_list_with_valid_value(
        self, validator, create_signed_jwt, valid_claims, valid_audience
    ):
        """JWT with audience as list containing valid value should be accepted."""
        valid_claims["aud"] = ["other-service", valid_audience, "another-service"]
        token = create_signed_jwt(valid_claims)

        result = validator.validate(token)
        assert valid_audience in result.audience

    def test_audience_as_list_without_valid_value(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with audience list not containing valid value should be rejected."""
        valid_claims["aud"] = ["service1", "service2"]
        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTInvalidClaimsError) as exc_info:
            validator.validate(token)

        assert "audience" in str(exc_info.value).lower()


# ============================================================================
# Signature Validation Tests
# ============================================================================


class TestSignatureValidation:
    """Test JWT signature validation."""

    def test_valid_signature_accepted(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with valid signature should be accepted."""
        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)
        assert result.subject == "user-123"

    def test_invalid_signature_rejected(
        self, validator, rsa_key_pair, valid_claims
    ):
        """JWT with tampered signature should be rejected."""
        from cryptography.hazmat.primitives.asymmetric import rsa

        # Create a different key pair for signing
        different_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        from cryptography.hazmat.primitives import serialization

        different_private_pem = different_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        import jwt

        token = jwt.encode(
            valid_claims,
            different_private_pem,
            algorithm="RS256",
            headers={"kid": "test-key-id"},
        )

        with pytest.raises(JWTInvalidSignatureError):
            validator.validate(token)

    def test_unknown_key_id_rejected(
        self, validator, create_signed_jwt, valid_claims
    ):
        """JWT with unknown key ID should be rejected."""
        import jwt

        # Create token with different kid
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        token = jwt.encode(
            valid_claims,
            private_pem,
            algorithm="RS256",
            headers={"kid": "unknown-key-id"},
        )

        with pytest.raises(JWTInvalidSignatureError) as exc_info:
            validator.validate(token)

        assert "key" in str(exc_info.value).lower()

    def test_malformed_token_rejected(self, validator):
        """Malformed JWT should be rejected."""
        with pytest.raises(JWTValidationError):
            validator.validate("not-a-valid-jwt")

    def test_none_algorithm_rejected(self, validator, valid_claims):
        """JWT with 'none' algorithm should be rejected."""
        import jwt

        # Create unsigned token with 'none' algorithm
        token = jwt.encode(valid_claims, None, algorithm="none")

        with pytest.raises(JWTInvalidSignatureError):
            validator.validate(token)


# ============================================================================
# Token Revocation Tests
# ============================================================================


class TestTokenRevocation:
    """Test JWT revocation checking."""

    def test_revoked_token_rejected(
        self, valid_issuer, valid_audience, jwks_response, create_signed_jwt, valid_claims
    ):
        """Revoked JWT should be rejected."""
        revocation_checker = MagicMock()
        revocation_checker.is_revoked.return_value = True

        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
            revocation_checker=revocation_checker,
        )
        validator._jwks_cache = {valid_issuer: jwks_response}

        token = create_signed_jwt(valid_claims)

        with pytest.raises(JWTRevokedError):
            validator.validate(token)

        revocation_checker.is_revoked.assert_called_once()

    def test_non_revoked_token_accepted(
        self, valid_issuer, valid_audience, jwks_response, create_signed_jwt, valid_claims
    ):
        """Non-revoked JWT should be accepted."""
        revocation_checker = MagicMock()
        revocation_checker.is_revoked.return_value = False

        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
            revocation_checker=revocation_checker,
        )
        validator._jwks_cache = {valid_issuer: jwks_response}

        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)

        assert result.subject == "user-123"
        revocation_checker.is_revoked.assert_called_once()


# ============================================================================
# JWKS Fetching Tests
# ============================================================================


class TestJWKSFetching:
    """Test JWKS endpoint fetching and caching."""

    def test_jwks_fetched_on_first_validation(
        self, valid_issuer, valid_audience, create_signed_jwt, valid_claims, jwks_response
    ):
        """JWKS should be fetched from issuer on first validation."""
        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
        )

        token = create_signed_jwt(valid_claims)

        with patch.object(validator, "_fetch_jwks", return_value=jwks_response) as mock_fetch:
            result = validator.validate(token)

            mock_fetch.assert_called_once_with(valid_issuer)
            assert result.subject == "user-123"

    def test_jwks_cached_after_fetch(
        self, valid_issuer, valid_audience, create_signed_jwt, valid_claims, jwks_response
    ):
        """JWKS should be cached after initial fetch."""
        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
        )

        token = create_signed_jwt(valid_claims)

        with patch.object(validator, "_fetch_jwks", return_value=jwks_response) as mock_fetch:
            # First validation
            validator.validate(token)
            # Second validation
            validator.validate(token)

            # Should only fetch once
            mock_fetch.assert_called_once()

    def test_jwks_cache_expiry(
        self, valid_issuer, valid_audience, create_signed_jwt, valid_claims, jwks_response
    ):
        """JWKS cache should expire after configured time."""
        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
            jwks_cache_ttl_seconds=1,  # 1 second cache
        )

        token = create_signed_jwt(valid_claims)

        with patch.object(validator, "_fetch_jwks", return_value=jwks_response) as mock_fetch:
            # First validation
            validator.validate(token)

            # Wait for cache to expire
            time.sleep(1.1)

            # Second validation should refetch
            validator.validate(token)

            assert mock_fetch.call_count == 2


# ============================================================================
# PKCE Validation Tests
# ============================================================================


class TestPKCEValidation:
    """Test PKCE S256 code verifier validation."""

    def test_valid_pkce_s256_verifier(self):
        """Valid PKCE S256 verifier should be accepted."""
        # Generate a valid code verifier
        code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

        # Calculate expected challenge
        verifier_bytes = code_verifier.encode("ascii")
        digest = hashlib.sha256(verifier_bytes).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

        pkce = PKCEValidator()
        assert pkce.validate_s256(code_verifier, code_challenge) is True

    def test_invalid_pkce_verifier_rejected(self):
        """Invalid PKCE verifier (wrong value) should return False."""
        # Use a valid-length verifier that doesn't match the challenge
        code_verifier = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"  # 43 'a' chars
        code_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

        pkce = PKCEValidator()
        # The verifier is valid format but doesn't match the challenge
        assert pkce.validate_s256(code_verifier, code_challenge) is False

    def test_pkce_verifier_too_short_rejected(self):
        """PKCE verifier shorter than 43 characters should be rejected."""
        code_verifier = "short"  # Too short

        pkce = PKCEValidator()
        with pytest.raises(PKCEValidationError) as exc_info:
            pkce.validate_s256(code_verifier, "any-challenge")

        assert "length" in str(exc_info.value).lower()

    def test_pkce_verifier_too_long_rejected(self):
        """PKCE verifier longer than 128 characters should be rejected."""
        code_verifier = "a" * 129  # Too long

        pkce = PKCEValidator()
        with pytest.raises(PKCEValidationError) as exc_info:
            pkce.validate_s256(code_verifier, "any-challenge")

        assert "length" in str(exc_info.value).lower()

    def test_pkce_verifier_invalid_characters_rejected(self):
        """PKCE verifier with invalid characters should be rejected."""
        # Contains invalid characters (only [A-Za-z0-9-._~] allowed)
        code_verifier = "valid-verifier-with-invalid-char$!" + "a" * 20

        pkce = PKCEValidator()
        with pytest.raises(PKCEValidationError) as exc_info:
            pkce.validate_s256(code_verifier, "any-challenge")

        assert "character" in str(exc_info.value).lower()

    def test_pkce_generate_verifier_and_challenge(self):
        """PKCEValidator should generate valid verifier/challenge pairs."""
        pkce = PKCEValidator()
        verifier, challenge = pkce.generate_s256_pair()

        # Verifier should be valid length
        assert 43 <= len(verifier) <= 128

        # Challenge should validate against verifier
        assert pkce.validate_s256(verifier, challenge) is True

    def test_pkce_plain_method_rejected(self):
        """PKCE 'plain' method should be rejected (only S256 allowed)."""
        pkce = PKCEValidator()

        with pytest.raises(PKCEValidationError) as exc_info:
            pkce.validate_plain("verifier", "verifier")

        assert "s256" in str(exc_info.value).lower() or "plain" in str(exc_info.value).lower()


# ============================================================================
# OAuth Claim Schema Tests
# ============================================================================


class TestOAuthClaimSchema:
    """Test OAuth claim schema per implementation plan section 4.4."""

    def test_optional_claims_accepted(
        self, validator, create_signed_jwt, valid_claims
    ):
        """Optional claims (session_id, user_id, token_id) should be accepted."""
        valid_claims["session_id"] = "sess-123"
        valid_claims["user_id"] = "user-456"
        valid_claims["token_id"] = "tok-789"
        valid_claims["geo_scope"] = "eu-west-1"

        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)

        assert result.session_id == "sess-123"
        assert result.user_id == "user-456"
        assert result.token_id == "tok-789"
        assert result.geo_scope == "eu-west-1"

    def test_team_ids_as_list(
        self, validator, create_signed_jwt, valid_claims
    ):
        """team_ids should be parsed as list."""
        valid_claims["team_ids"] = ["team-1", "team-2", "team-3"]
        token = create_signed_jwt(valid_claims)

        result = validator.validate(token)
        assert result.team_ids == ["team-1", "team-2", "team-3"]

    def test_project_ids_as_list(
        self, validator, create_signed_jwt, valid_claims
    ):
        """project_ids should be parsed as list."""
        valid_claims["project_ids"] = ["proj-1", "proj-2"]
        token = create_signed_jwt(valid_claims)

        result = validator.validate(token)
        assert result.project_ids == ["proj-1", "proj-2"]

    def test_enterprise_id_claim(
        self, validator, create_signed_jwt, valid_claims
    ):
        """enterprise_id should be extracted from claims."""
        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)

        assert result.enterprise_id == "ent-789"


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_token_rejected(self, validator):
        """Empty token should be rejected."""
        with pytest.raises(JWTValidationError):
            validator.validate("")

    def test_none_token_rejected(self, validator):
        """None token should be rejected."""
        with pytest.raises(JWTValidationError):
            validator.validate(None)

    def test_whitespace_only_token_rejected(self, validator):
        """Whitespace-only token should be rejected."""
        with pytest.raises(JWTValidationError):
            validator.validate("   ")

    def test_token_with_extra_segments_rejected(self, validator):
        """Token with extra segments should be rejected."""
        with pytest.raises(JWTValidationError):
            validator.validate("a.b.c.d.e")

    def test_clock_skew_tolerance(
        self, valid_issuer, valid_audience, jwks_response, create_signed_jwt, valid_claims
    ):
        """Validator should allow configurable clock skew tolerance."""
        validator = JWTValidator(
            issuer_allowlist=[valid_issuer],
            audience=valid_audience,
            clock_skew_seconds=60,  # 60 second tolerance
        )
        validator._jwks_cache = {valid_issuer: jwks_response}

        # Token that expired 30 seconds ago should still be valid with 60s tolerance
        valid_claims["exp"] = int(time.time()) - 30
        token = create_signed_jwt(valid_claims)

        result = validator.validate(token)
        assert result.subject == "user-123"

    def test_validation_result_contains_raw_claims(
        self, validator, create_signed_jwt, valid_claims
    ):
        """Validation result should contain raw claims for extension."""
        token = create_signed_jwt(valid_claims)
        result = validator.validate(token)

        assert hasattr(result, "raw_claims")
        assert result.raw_claims["sub"] == "user-123"
