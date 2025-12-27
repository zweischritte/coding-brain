"""
Security module for Phase 1: Security Enforcement Baseline.

This module provides:
- JWT + DPoP token validation
- Principal extraction and authentication dependencies
- RBAC scope enforcement
- Security headers middleware
- MCP tool permission checks
"""

from .types import Principal, TokenClaims, Scope, AuthenticationError, AuthorizationError
from .dependencies import get_current_principal, require_scopes, get_optional_principal
from .dpop import DPoPValidator, DPoPProof, get_dpop_cache
from .middleware import SecurityHeadersMiddleware, AuthExceptionMiddleware
from .jwt import validate_jwt, get_jwt_config
from .exception_handlers import register_security_exception_handlers

__all__ = [
    # Types
    "Principal",
    "TokenClaims",
    "Scope",
    "AuthenticationError",
    "AuthorizationError",
    # Dependencies
    "get_current_principal",
    "require_scopes",
    "get_optional_principal",
    # DPoP
    "DPoPValidator",
    "DPoPProof",
    "get_dpop_cache",
    # JWT
    "validate_jwt",
    "get_jwt_config",
    # Middleware
    "SecurityHeadersMiddleware",
    "AuthExceptionMiddleware",
    # Exception handlers
    "register_security_exception_handlers",
]
