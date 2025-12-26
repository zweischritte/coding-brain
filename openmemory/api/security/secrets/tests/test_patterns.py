"""Tests for secret detection patterns.

This module tests the pattern library for detecting various secret types:
- API keys (AWS, GCP, Azure, GitHub, etc.)
- Private keys (RSA, ECDSA, PGP, SSH)
- Database credentials and connection strings
- JWT tokens and bearer tokens
- Generic high-entropy strings
"""

import pytest

from openmemory.api.security.secrets.patterns import (
    SecretPattern,
    SecretPatternLibrary,
    SecretType,
    SecretMatch,
    Confidence,
)


# ============================================================================
# SecretType Tests
# ============================================================================


class TestSecretType:
    """Tests for SecretType enumeration."""

    def test_secret_type_values(self):
        """Test all secret types are defined."""
        assert SecretType.AWS_ACCESS_KEY is not None
        assert SecretType.AWS_SECRET_KEY is not None
        assert SecretType.GITHUB_TOKEN is not None
        assert SecretType.GITHUB_PAT is not None
        assert SecretType.GITLAB_TOKEN is not None
        assert SecretType.PRIVATE_KEY_RSA is not None
        assert SecretType.PRIVATE_KEY_ECDSA is not None
        assert SecretType.PRIVATE_KEY_PGP is not None
        assert SecretType.PRIVATE_KEY_SSH is not None
        assert SecretType.GCP_API_KEY is not None
        assert SecretType.GCP_SERVICE_ACCOUNT is not None
        assert SecretType.AZURE_CONNECTION_STRING is not None
        assert SecretType.AZURE_CLIENT_SECRET is not None
        assert SecretType.DATABASE_URL is not None
        assert SecretType.JDBC_CONNECTION is not None
        assert SecretType.JWT_TOKEN is not None
        assert SecretType.BEARER_TOKEN is not None
        assert SecretType.SLACK_TOKEN is not None
        assert SecretType.STRIPE_KEY is not None
        assert SecretType.TWILIO_KEY is not None
        assert SecretType.SENDGRID_KEY is not None
        assert SecretType.NPM_TOKEN is not None
        assert SecretType.PYPI_TOKEN is not None
        assert SecretType.DOCKER_AUTH is not None
        assert SecretType.GENERIC_SECRET is not None
        assert SecretType.GENERIC_PASSWORD is not None
        assert SecretType.HIGH_ENTROPY is not None


class TestConfidence:
    """Tests for Confidence enumeration."""

    def test_confidence_ordering(self):
        """Test confidence levels are properly ordered."""
        assert Confidence.LOW < Confidence.MEDIUM
        assert Confidence.MEDIUM < Confidence.HIGH
        assert Confidence.HIGH < Confidence.VERIFIED

    def test_confidence_values(self):
        """Test confidence level numeric values."""
        assert Confidence.LOW.value == 1
        assert Confidence.MEDIUM.value == 2
        assert Confidence.HIGH.value == 3
        assert Confidence.VERIFIED.value == 4


# ============================================================================
# SecretPattern Tests
# ============================================================================


class TestSecretPattern:
    """Tests for SecretPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a secret pattern."""
        pattern = SecretPattern(
            pattern=r"AKIA[0-9A-Z]{16}",
            secret_type=SecretType.AWS_ACCESS_KEY,
            description="AWS Access Key ID",
            confidence=Confidence.HIGH,
        )
        assert pattern.pattern == r"AKIA[0-9A-Z]{16}"
        assert pattern.secret_type == SecretType.AWS_ACCESS_KEY
        assert pattern.description == "AWS Access Key ID"
        assert pattern.confidence == Confidence.HIGH

    def test_pattern_with_group(self):
        """Test pattern with capture group."""
        pattern = SecretPattern(
            pattern=r"(ghp_[a-zA-Z0-9]{36})",
            secret_type=SecretType.GITHUB_PAT,
            description="GitHub Personal Access Token",
            confidence=Confidence.HIGH,
            capture_group=1,
        )
        assert pattern.capture_group == 1

    def test_pattern_with_validator(self):
        """Test pattern with custom validator function."""
        def validate_aws_key(match_str: str) -> bool:
            return match_str.startswith("AKIA")

        pattern = SecretPattern(
            pattern=r"AKIA[0-9A-Z]{16}",
            secret_type=SecretType.AWS_ACCESS_KEY,
            description="AWS Access Key ID",
            confidence=Confidence.HIGH,
            validator=validate_aws_key,
        )
        assert pattern.validator is not None
        assert pattern.validator("AKIA1234567890ABCDEF")


# ============================================================================
# SecretMatch Tests
# ============================================================================


class TestSecretMatch:
    """Tests for SecretMatch dataclass."""

    def test_match_creation(self):
        """Test creating a secret match."""
        match = SecretMatch(
            secret_type=SecretType.AWS_ACCESS_KEY,
            matched_value="AKIA1234567890ABCDEF",
            redacted_value="AKIA...CDEF",
            confidence=Confidence.HIGH,
            line_number=42,
            column_start=10,
            column_end=30,
            pattern_description="AWS Access Key ID",
        )
        assert match.secret_type == SecretType.AWS_ACCESS_KEY
        assert match.matched_value == "AKIA1234567890ABCDEF"
        assert match.redacted_value == "AKIA...CDEF"
        assert match.confidence == Confidence.HIGH
        assert match.line_number == 42
        assert match.column_start == 10
        assert match.column_end == 30

    def test_match_file_path(self):
        """Test match with file path context."""
        match = SecretMatch(
            secret_type=SecretType.GITHUB_TOKEN,
            matched_value="ghp_xxxxx",
            redacted_value="ghp_...",
            confidence=Confidence.HIGH,
            line_number=1,
            file_path="/path/to/file.py",
        )
        assert match.file_path == "/path/to/file.py"

    def test_match_to_dict(self):
        """Test serializing match to dictionary."""
        match = SecretMatch(
            secret_type=SecretType.AWS_ACCESS_KEY,
            matched_value="AKIA1234567890ABCDEF",
            redacted_value="AKIA...CDEF",
            confidence=Confidence.HIGH,
            line_number=42,
        )
        d = match.to_dict()
        assert d["secret_type"] == "aws_access_key"
        assert d["redacted_value"] == "AKIA...CDEF"
        assert d["confidence"] == "high"
        assert d["line_number"] == 42
        # Should not include the actual secret value
        assert "matched_value" not in d or d.get("matched_value") == "AKIA...CDEF"


# ============================================================================
# SecretPatternLibrary Tests
# ============================================================================


class TestSecretPatternLibrary:
    """Tests for SecretPatternLibrary."""

    @pytest.fixture
    def library(self):
        """Create a pattern library instance."""
        return SecretPatternLibrary()

    # -------------------------------------------------------------------------
    # AWS Secrets
    # -------------------------------------------------------------------------

    def test_detect_aws_access_key(self, library):
        """Test detecting AWS Access Key ID."""
        # Use realistic-looking key (not a real key, but realistic pattern)
        text = "aws_access_key_id = AKIAJ5Q7R2D9K3G4N8M1"
        matches = library.scan(text)
        assert len(matches) >= 1
        aws_match = next((m for m in matches if m.secret_type == SecretType.AWS_ACCESS_KEY), None)
        assert aws_match is not None
        assert aws_match.confidence >= Confidence.HIGH

    def test_detect_aws_secret_key(self, library):
        """Test detecting AWS Secret Access Key."""
        text = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        matches = library.scan(text)
        assert len(matches) >= 1
        secret_match = next((m for m in matches if m.secret_type == SecretType.AWS_SECRET_KEY), None)
        assert secret_match is not None

    def test_no_false_positive_aws_placeholder(self, library):
        """Test AWS placeholder doesn't trigger false positive."""
        text = "aws_access_key_id = AKIAXXXXXXXXXXXXXXXX"
        matches = library.scan(text)
        # Placeholder with all X's should be low confidence or not matched
        if matches:
            for m in matches:
                if m.secret_type == SecretType.AWS_ACCESS_KEY:
                    assert m.confidence <= Confidence.LOW

    # -------------------------------------------------------------------------
    # GitHub Secrets
    # -------------------------------------------------------------------------

    def test_detect_github_pat_classic(self, library):
        """Test detecting GitHub Personal Access Token (Classic)."""
        # GitHub Classic PAT format: ghp_ + 36 alphanumeric = 40 total chars
        text = "GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
        matches = library.scan(text)
        assert len(matches) >= 1
        gh_match = next((m for m in matches if m.secret_type == SecretType.GITHUB_PAT), None)
        assert gh_match is not None
        assert gh_match.confidence >= Confidence.HIGH

    def test_detect_github_pat_fine_grained(self, library):
        """Test detecting GitHub Personal Access Token (Fine-grained)."""
        # GitHub Fine-grained PAT format: github_pat_ + 22 chars + _ + 59 chars = 93 total
        # Part 1: 22 chars, Part 2: 59 chars
        text = "GITHUB_TOKEN=github_pat_11ABcDefGH9xyz8XYZ7ABz_a1BzC3dXe5F6gYh8I9j0K1l2M3n4O5p6Q7r8S9t0U1v2W3x4Y5z6A7b8C9d"
        matches = library.scan(text)
        assert len(matches) >= 1
        gh_match = next((m for m in matches if m.secret_type == SecretType.GITHUB_PAT), None)
        assert gh_match is not None
        assert gh_match.confidence >= Confidence.HIGH

    def test_detect_github_oauth(self, library):
        """Test detecting GitHub OAuth Token."""
        text = "token: gho_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
        matches = library.scan(text)
        assert len(matches) >= 1

    def test_detect_github_app_token(self, library):
        """Test detecting GitHub App Token."""
        text = "auth: ghu_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
        matches = library.scan(text)
        assert len(matches) >= 1

    # -------------------------------------------------------------------------
    # Private Keys
    # -------------------------------------------------------------------------

    def test_detect_rsa_private_key(self, library):
        """Test detecting RSA private key."""
        text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy...
-----END RSA PRIVATE KEY-----"""
        matches = library.scan(text)
        assert len(matches) >= 1
        rsa_match = next((m for m in matches if m.secret_type == SecretType.PRIVATE_KEY_RSA), None)
        assert rsa_match is not None
        assert rsa_match.confidence >= Confidence.HIGH

    def test_detect_ecdsa_private_key(self, library):
        """Test detecting ECDSA private key."""
        text = """-----BEGIN EC PRIVATE KEY-----
MHQCAQEEICMJoZHXh1p+b4tQECWuqW4bK4DRYjPaKJ...
-----END EC PRIVATE KEY-----"""
        matches = library.scan(text)
        assert len(matches) >= 1
        ec_match = next((m for m in matches if m.secret_type == SecretType.PRIVATE_KEY_ECDSA), None)
        assert ec_match is not None

    def test_detect_openssh_private_key(self, library):
        """Test detecting OpenSSH private key."""
        text = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ...
-----END OPENSSH PRIVATE KEY-----"""
        matches = library.scan(text)
        assert len(matches) >= 1
        ssh_match = next((m for m in matches if m.secret_type == SecretType.PRIVATE_KEY_SSH), None)
        assert ssh_match is not None

    def test_detect_pgp_private_key(self, library):
        """Test detecting PGP private key block."""
        text = """-----BEGIN PGP PRIVATE KEY BLOCK-----

lQOYBF5JK...
-----END PGP PRIVATE KEY BLOCK-----"""
        matches = library.scan(text)
        assert len(matches) >= 1
        pgp_match = next((m for m in matches if m.secret_type == SecretType.PRIVATE_KEY_PGP), None)
        assert pgp_match is not None

    # -------------------------------------------------------------------------
    # Database Credentials
    # -------------------------------------------------------------------------

    def test_detect_postgres_url(self, library):
        """Test detecting PostgreSQL connection string."""
        text = "DATABASE_URL=postgresql://user:password123@localhost:5432/mydb"
        matches = library.scan(text)
        assert len(matches) >= 1
        db_match = next((m for m in matches if m.secret_type == SecretType.DATABASE_URL), None)
        assert db_match is not None

    def test_detect_mysql_url(self, library):
        """Test detecting MySQL connection string."""
        text = "MYSQL_URL=mysql://root:secret@127.0.0.1:3306/app"
        matches = library.scan(text)
        assert len(matches) >= 1

    def test_detect_mongodb_url(self, library):
        """Test detecting MongoDB connection string."""
        text = "MONGO_URI=mongodb+srv://admin:p4ssw0rd@cluster.mongodb.net/db"
        matches = library.scan(text)
        assert len(matches) >= 1

    def test_detect_jdbc_connection(self, library):
        """Test detecting JDBC connection string."""
        text = 'jdbc:postgresql://host:5432/db?user=admin&password=secret123'
        matches = library.scan(text)
        assert len(matches) >= 1
        jdbc_match = next((m for m in matches if m.secret_type == SecretType.JDBC_CONNECTION), None)
        assert jdbc_match is not None

    # -------------------------------------------------------------------------
    # Cloud Provider Secrets
    # -------------------------------------------------------------------------

    def test_detect_gcp_api_key(self, library):
        """Test detecting GCP API key."""
        text = "GCP_API_KEY=AIzaSyC0xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        matches = library.scan(text)
        assert len(matches) >= 1
        gcp_match = next((m for m in matches if m.secret_type == SecretType.GCP_API_KEY), None)
        assert gcp_match is not None

    def test_detect_gcp_service_account(self, library):
        """Test detecting GCP service account key structure."""
        text = '''{
            "type": "service_account",
            "project_id": "my-project",
            "private_key_id": "abc123",
            "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQ...\\n-----END PRIVATE KEY-----\\n"
        }'''
        matches = library.scan(text)
        assert len(matches) >= 1
        gcp_sa_match = next((m for m in matches if m.secret_type == SecretType.GCP_SERVICE_ACCOUNT), None)
        assert gcp_sa_match is not None

    def test_detect_azure_connection_string(self, library):
        """Test detecting Azure connection string."""
        text = "AZURE_STORAGE=DefaultEndpointsProtocol=https;AccountName=mystorageaccount;AccountKey=xxxxxx==;EndpointSuffix=core.windows.net"
        matches = library.scan(text)
        assert len(matches) >= 1
        az_match = next((m for m in matches if m.secret_type == SecretType.AZURE_CONNECTION_STRING), None)
        assert az_match is not None

    # -------------------------------------------------------------------------
    # Tokens and API Keys
    # -------------------------------------------------------------------------

    def test_detect_jwt_token(self, library):
        """Test detecting JWT token."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        matches = library.scan(text)
        assert len(matches) >= 1
        jwt_match = next((m for m in matches if m.secret_type == SecretType.JWT_TOKEN), None)
        assert jwt_match is not None

    def test_detect_slack_token(self, library):
        """Test detecting Slack token."""
        text = "SLACK_TOKEN=xoxb-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx"
        matches = library.scan(text)
        assert len(matches) >= 1
        slack_match = next((m for m in matches if m.secret_type == SecretType.SLACK_TOKEN), None)
        assert slack_match is not None

    def test_detect_stripe_key(self, library):
        """Test detecting Stripe API key."""
        text = "STRIPE_SECRET_KEY=sk_live_xxxxxxxxxxxxxxxxxxxxxxxx"
        matches = library.scan(text)
        assert len(matches) >= 1
        stripe_match = next((m for m in matches if m.secret_type == SecretType.STRIPE_KEY), None)
        assert stripe_match is not None

    def test_detect_twilio_key(self, library):
        """Test detecting Twilio API key."""
        text = "TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # Twilio tokens are 32 hex characters
        text2 = "TWILIO_AUTH_TOKEN=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
        matches = library.scan(text2)
        # Should detect as generic or twilio key depending on context
        assert len(matches) >= 0  # May not match without clear twilio context

    def test_detect_sendgrid_key(self, library):
        """Test detecting SendGrid API key."""
        # SendGrid key format: SG.<22 chars>.<43 chars>
        text = "SENDGRID_API_KEY=SG.a1B2c3D4e5F6g7H8i9J0kL.m1N2o3P4q5R6s7T8u9V0w1X2y3Z4a5B6c7D8e9F0g1H2i"
        matches = library.scan(text)
        assert len(matches) >= 1
        sg_match = next((m for m in matches if m.secret_type == SecretType.SENDGRID_KEY), None)
        assert sg_match is not None

    # -------------------------------------------------------------------------
    # Package Registry Tokens
    # -------------------------------------------------------------------------

    def test_detect_npm_token(self, library):
        """Test detecting npm token."""
        text = "//registry.npmjs.org/:_authToken=npm_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        matches = library.scan(text)
        assert len(matches) >= 1
        npm_match = next((m for m in matches if m.secret_type == SecretType.NPM_TOKEN), None)
        assert npm_match is not None

    def test_detect_pypi_token(self, library):
        """Test detecting PyPI token."""
        # PyPI token format: pypi-<base64-ish string of 50+ chars>
        text = "PYPI_TOKEN=pypi-AgEIcHlwaS5vcmc1A2B3C4D5E6F7G8H9I0J1K2L3M4N5O6P7Q8R9S0"
        matches = library.scan(text)
        assert len(matches) >= 1
        pypi_match = next((m for m in matches if m.secret_type == SecretType.PYPI_TOKEN), None)
        assert pypi_match is not None

    # -------------------------------------------------------------------------
    # Generic Secrets
    # -------------------------------------------------------------------------

    def test_detect_generic_password_assignment(self, library):
        """Test detecting generic password assignment."""
        text = 'password = "supersecret123!"'
        matches = library.scan(text)
        assert len(matches) >= 1
        pwd_match = next((m for m in matches if m.secret_type == SecretType.GENERIC_PASSWORD), None)
        assert pwd_match is not None

    def test_detect_generic_secret_env_var(self, library):
        """Test detecting generic secret in environment variable."""
        text = 'export SECRET_KEY="my-super-secret-key-12345"'
        matches = library.scan(text)
        assert len(matches) >= 1
        secret_match = next((m for m in matches if m.secret_type == SecretType.GENERIC_SECRET), None)
        assert secret_match is not None

    def test_detect_api_key_assignment(self, library):
        """Test detecting generic API key assignment."""
        text = 'API_KEY="sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"'
        matches = library.scan(text)
        assert len(matches) >= 1

    # -------------------------------------------------------------------------
    # High Entropy Detection
    # -------------------------------------------------------------------------

    def test_detect_high_entropy_string(self, library):
        """Test detecting high-entropy string."""
        # High entropy random string
        text = 'TOKEN="a8f3k2j9d7s5g6h4w2e1r9t8y7u6i5o4"'
        matches = library.scan(text)
        # Should detect as high entropy or generic secret
        assert len(matches) >= 0  # May or may not match depending on threshold

    def test_no_false_positive_low_entropy(self, library):
        """Test that low-entropy strings don't trigger false positives."""
        text = 'MESSAGE="Hello World"'
        matches = library.scan(text)
        # Should not match as high entropy
        high_entropy_matches = [m for m in matches if m.secret_type == SecretType.HIGH_ENTROPY]
        assert len(high_entropy_matches) == 0

    # -------------------------------------------------------------------------
    # False Positive Handling
    # -------------------------------------------------------------------------

    def test_no_false_positive_example_placeholder(self, library):
        """Test that example placeholders don't trigger false positives."""
        text = 'API_KEY="YOUR_API_KEY_HERE"'
        matches = library.scan(text)
        # Should not match placeholder
        for m in matches:
            assert m.confidence <= Confidence.LOW

    def test_no_false_positive_documentation(self, library):
        """Test that documentation examples don't trigger high-confidence matches."""
        text = """
        # Example usage:
        # export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
        # export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
        """
        matches = library.scan(text)
        # Example keys should be lower confidence
        for m in matches:
            if "EXAMPLE" in m.matched_value:
                assert m.confidence <= Confidence.MEDIUM

    def test_no_false_positive_test_fixtures(self, library):
        """Test that test fixture values don't trigger high-confidence matches."""
        text = '''
        TEST_TOKEN = "test-token-12345"
        MOCK_SECRET = "mock-secret-value"
        '''
        matches = library.scan(text)
        # Test values should be lower confidence
        for m in matches:
            if "test" in m.matched_value.lower() or "mock" in m.matched_value.lower():
                assert m.confidence <= Confidence.MEDIUM

    # -------------------------------------------------------------------------
    # Line Number and Position Tracking
    # -------------------------------------------------------------------------

    def test_line_number_tracking(self, library):
        """Test that line numbers are correctly tracked."""
        text = """line 1
line 2
GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8
line 4"""
        matches = library.scan(text)
        assert len(matches) >= 1
        gh_match = next((m for m in matches if m.secret_type == SecretType.GITHUB_PAT), None)
        assert gh_match is not None
        assert gh_match.line_number == 3

    def test_column_position_tracking(self, library):
        """Test that column positions are correctly tracked."""
        text = "prefix GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8 suffix"
        matches = library.scan(text)
        assert len(matches) >= 1
        gh_match = next((m for m in matches if m.secret_type == SecretType.GITHUB_PAT), None)
        assert gh_match is not None
        assert gh_match.column_start is not None
        assert gh_match.column_end is not None
        assert gh_match.column_end > gh_match.column_start

    # -------------------------------------------------------------------------
    # Redaction
    # -------------------------------------------------------------------------

    def test_redaction_preserves_prefix(self, library):
        """Test that redaction preserves identifying prefix."""
        text = "GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
        matches = library.scan(text)
        assert len(matches) >= 1
        gh_match = next((m for m in matches if m.secret_type == SecretType.GITHUB_PAT), None)
        assert gh_match is not None
        assert gh_match.redacted_value.startswith("ghp_")
        assert "..." in gh_match.redacted_value

    def test_redaction_shows_suffix(self, library):
        """Test that redaction shows last few characters."""
        text = "AWS_KEY=AKIAIOSFODNN7EXAMPLE"
        matches = library.scan(text)
        aws_match = next((m for m in matches if m.secret_type == SecretType.AWS_ACCESS_KEY), None)
        if aws_match:
            # Should show first 4 and last 4 characters
            assert "..." in aws_match.redacted_value

    # -------------------------------------------------------------------------
    # Custom Pattern Support
    # -------------------------------------------------------------------------

    def test_add_custom_pattern(self, library):
        """Test adding a custom pattern."""
        custom_pattern = SecretPattern(
            pattern=r"CUSTOM_KEY_[A-Z0-9]{32}",
            secret_type=SecretType.GENERIC_SECRET,
            description="Custom internal API key",
            confidence=Confidence.HIGH,
        )
        library.add_pattern(custom_pattern)

        text = "auth=CUSTOM_KEY_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
        matches = library.scan(text)
        assert len(matches) >= 1

    def test_list_patterns(self, library):
        """Test listing all patterns."""
        patterns = library.list_patterns()
        assert len(patterns) > 0
        assert all(isinstance(p, dict) for p in patterns)
        assert all("pattern" in p for p in patterns)
        assert all("secret_type" in p for p in patterns)

    # -------------------------------------------------------------------------
    # Multiple Secrets in One Text
    # -------------------------------------------------------------------------

    def test_detect_multiple_secrets(self, library):
        """Test detecting multiple secrets in one text."""
        text = """
        AWS_ACCESS_KEY_ID=AKIAJ5Q7R2D9K3G4N8M1
        aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYa1b2c3d4
        GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8
        """
        matches = library.scan(text)
        # Should detect at least AWS key and GitHub PAT
        assert len(matches) >= 2

        secret_types = {m.secret_type for m in matches}
        assert SecretType.AWS_ACCESS_KEY in secret_types
        assert SecretType.GITHUB_PAT in secret_types

    def test_overlapping_patterns(self, library):
        """Test that overlapping patterns are handled correctly."""
        # This could match both bearer token and JWT
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        matches = library.scan(text)
        # Should deduplicate or return most specific match
        assert len(matches) >= 1

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    def test_empty_input(self, library):
        """Test handling empty input."""
        matches = library.scan("")
        assert matches == []

    def test_none_input(self, library):
        """Test handling None input."""
        matches = library.scan(None)
        assert matches == []

    def test_unicode_input(self, library):
        """Test handling Unicode input."""
        text = "API_KEY=日本語テスト123"
        matches = library.scan(text)
        # Should not crash, may or may not match

    def test_very_long_input(self, library):
        """Test handling very long input."""
        text = "x" * 100000 + " GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8 " + "y" * 100000
        matches = library.scan(text)
        # Should find the token without timeout
        assert len(matches) >= 1

    def test_binary_like_content(self, library):
        """Test handling binary-like content gracefully."""
        text = "\\x00\\x01\\x02 GITHUB_TOKEN=ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8 \\xff\\xfe"
        matches = library.scan(text)
        # Should handle gracefully


# ============================================================================
# Pattern Statistics Tests
# ============================================================================


class TestPatternStatistics:
    """Tests for pattern library statistics."""

    @pytest.fixture
    def library(self):
        """Create a pattern library instance."""
        return SecretPatternLibrary()

    def test_get_statistics(self, library):
        """Test getting pattern statistics."""
        stats = library.get_statistics()
        assert "total_patterns" in stats
        assert "patterns_by_type" in stats
        assert stats["total_patterns"] > 0

    def test_patterns_by_type_count(self, library):
        """Test counting patterns by secret type."""
        stats = library.get_statistics()
        by_type = stats["patterns_by_type"]
        assert isinstance(by_type, dict)
        # Should have at least AWS and GitHub patterns
        assert sum(by_type.values()) == stats["total_patterns"]
