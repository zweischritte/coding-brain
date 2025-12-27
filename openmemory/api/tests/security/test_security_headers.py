"""
Tests for security headers middleware.

TDD: These tests define the expected security headers for all responses.
Tests should fail until implementation is complete.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.security.middleware import SecurityHeadersMiddleware


@pytest.fixture
def app():
    """Create a test FastAPI application with security headers middleware."""
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "ok"}

    @app.get("/html")
    async def html_endpoint():
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<html><body>Hello</body></html>")

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestContentSecurityPolicy:
    """Tests for Content-Security-Policy header."""

    def test_csp_header_present(self, client):
        """Response should include Content-Security-Policy header."""
        response = client.get("/test")
        assert "Content-Security-Policy" in response.headers

    def test_csp_default_src_self(self, client):
        """CSP should restrict default-src to 'self'."""
        response = client.get("/test")
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp

    def test_csp_script_src_self(self, client):
        """CSP should restrict script-src to 'self' (no inline scripts)."""
        response = client.get("/test")
        csp = response.headers["Content-Security-Policy"]
        assert "script-src 'self'" in csp

    def test_csp_frame_ancestors_none(self, client):
        """CSP should prevent framing with frame-ancestors 'none'."""
        response = client.get("/test")
        csp = response.headers["Content-Security-Policy"]
        assert "frame-ancestors 'none'" in csp

    def test_csp_form_action_self(self, client):
        """CSP should restrict form-action to 'self'."""
        response = client.get("/test")
        csp = response.headers["Content-Security-Policy"]
        assert "form-action 'self'" in csp

    def test_csp_base_uri_self(self, client):
        """CSP should restrict base-uri to 'self' to prevent base tag injection."""
        response = client.get("/test")
        csp = response.headers["Content-Security-Policy"]
        assert "base-uri 'self'" in csp


class TestStrictTransportSecurity:
    """Tests for Strict-Transport-Security (HSTS) header."""

    def test_hsts_header_present(self, client):
        """Response should include Strict-Transport-Security header."""
        response = client.get("/test")
        assert "Strict-Transport-Security" in response.headers

    def test_hsts_max_age_at_least_one_year(self, client):
        """HSTS max-age should be at least 1 year (31536000 seconds)."""
        response = client.get("/test")
        hsts = response.headers["Strict-Transport-Security"]
        # Extract max-age value
        import re
        match = re.search(r"max-age=(\d+)", hsts)
        assert match is not None
        max_age = int(match.group(1))
        assert max_age >= 31536000  # At least 1 year

    def test_hsts_include_subdomains(self, client):
        """HSTS should include includeSubDomains directive."""
        response = client.get("/test")
        hsts = response.headers["Strict-Transport-Security"]
        assert "includeSubDomains" in hsts


class TestXFrameOptions:
    """Tests for X-Frame-Options header."""

    def test_x_frame_options_present(self, client):
        """Response should include X-Frame-Options header."""
        response = client.get("/test")
        assert "X-Frame-Options" in response.headers

    def test_x_frame_options_deny(self, client):
        """X-Frame-Options should be DENY to prevent clickjacking."""
        response = client.get("/test")
        assert response.headers["X-Frame-Options"] == "DENY"


class TestXContentTypeOptions:
    """Tests for X-Content-Type-Options header."""

    def test_x_content_type_options_present(self, client):
        """Response should include X-Content-Type-Options header."""
        response = client.get("/test")
        assert "X-Content-Type-Options" in response.headers

    def test_x_content_type_options_nosniff(self, client):
        """X-Content-Type-Options should be 'nosniff' to prevent MIME sniffing."""
        response = client.get("/test")
        assert response.headers["X-Content-Type-Options"] == "nosniff"


class TestXXSSProtection:
    """Tests for X-XSS-Protection header."""

    def test_x_xss_protection_present(self, client):
        """Response should include X-XSS-Protection header."""
        response = client.get("/test")
        assert "X-XSS-Protection" in response.headers

    def test_x_xss_protection_enabled_block(self, client):
        """X-XSS-Protection should be '1; mode=block'."""
        response = client.get("/test")
        # Note: Modern browsers are deprecating this, but it's still a defense-in-depth measure
        assert response.headers["X-XSS-Protection"] in ["1; mode=block", "0"]


class TestReferrerPolicy:
    """Tests for Referrer-Policy header."""

    def test_referrer_policy_present(self, client):
        """Response should include Referrer-Policy header."""
        response = client.get("/test")
        assert "Referrer-Policy" in response.headers

    def test_referrer_policy_strict(self, client):
        """Referrer-Policy should be strict to prevent leaking referrer."""
        response = client.get("/test")
        policy = response.headers["Referrer-Policy"]
        # Accept various strict policies
        assert policy in [
            "strict-origin-when-cross-origin",
            "strict-origin",
            "same-origin",
            "no-referrer",
        ]


class TestPermissionsPolicy:
    """Tests for Permissions-Policy header."""

    def test_permissions_policy_present(self, client):
        """Response should include Permissions-Policy header."""
        response = client.get("/test")
        assert "Permissions-Policy" in response.headers

    def test_permissions_policy_restricts_geolocation(self, client):
        """Permissions-Policy should restrict geolocation."""
        response = client.get("/test")
        policy = response.headers["Permissions-Policy"]
        assert "geolocation" in policy

    def test_permissions_policy_restricts_camera(self, client):
        """Permissions-Policy should restrict camera access."""
        response = client.get("/test")
        policy = response.headers["Permissions-Policy"]
        assert "camera" in policy

    def test_permissions_policy_restricts_microphone(self, client):
        """Permissions-Policy should restrict microphone access."""
        response = client.get("/test")
        policy = response.headers["Permissions-Policy"]
        assert "microphone" in policy


class TestSecurityHeadersForAllResponses:
    """Tests that security headers are applied to all responses."""

    def test_headers_on_json_response(self, client):
        """Security headers should be present on JSON responses."""
        response = client.get("/test")
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_headers_on_html_response(self, client):
        """Security headers should be present on HTML responses."""
        response = client.get("/html")
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_headers_on_handled_error_response(self, app):
        """Security headers should be present on handled error responses."""
        from fastapi import HTTPException

        @app.get("/handled-error")
        async def handled_error_endpoint():
            raise HTTPException(status_code=400, detail="Bad request")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/handled-error")
        assert response.status_code == 400
        assert "X-Frame-Options" in response.headers
        assert "X-Content-Type-Options" in response.headers

    def test_headers_on_404_response(self, client):
        """Security headers should be present on 404 responses."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        assert "X-Frame-Options" in response.headers
        assert "X-Content-Type-Options" in response.headers


class TestCustomCSPConfiguration:
    """Tests for custom CSP configuration."""

    def test_custom_csp_value(self):
        """Should accept custom CSP value."""
        custom_csp = "default-src 'self'; script-src 'self' 'unsafe-inline'"

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware, csp=custom_csp)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "ok"}

        client = TestClient(app)
        response = client.get("/test")
        assert response.headers["Content-Security-Policy"] == custom_csp

    def test_custom_hsts_max_age(self):
        """Should accept custom HSTS max-age value."""
        custom_max_age = 86400  # 1 day

        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware, hsts_max_age=custom_max_age)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "ok"}

        client = TestClient(app)
        response = client.get("/test")
        hsts = response.headers["Strict-Transport-Security"]
        assert f"max-age={custom_max_age}" in hsts
