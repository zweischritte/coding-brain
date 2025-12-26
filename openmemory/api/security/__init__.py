"""Security module for OpenMemory API.

This module implements OAuth 2.1 security features per the implementation plan:
- JWT validation with claim verification (iss, aud, exp, iat, nbf, sub)
- PKCE S256 code verifier validation
- DPoP (Demonstrating Proof of Possession) token binding
- RBAC permission matrix
- SCIM 2.0 integration stubs
- Prompt injection defense patterns
"""
