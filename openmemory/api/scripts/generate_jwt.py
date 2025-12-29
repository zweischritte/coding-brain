#!/usr/bin/env python3
"""
Generate a JWT token for Coding Brain API access.

Usage:
    python scripts/generate_jwt.py
    python scripts/generate_jwt.py --user my_user --scopes "memories:read code:write"
    python scripts/generate_jwt.py --user my_user --grants "org:cloudfactory team:cloudfactory/backend"
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from jose import jwt
except ImportError:
    print("Error: python-jose not installed. Run: pip install python-jose[cryptography]")
    sys.exit(1)


# All available scopes
ALL_SCOPES = [
    "memories:read",
    "memories:write",
    "memories:delete",
    "apps:read",
    "apps:write",
    "entities:read",
    "entities:write",
    "stats:read",
    "graph:read",
    "graph:write",
    "search:read",
    "code:read",
    "code:write",
    "mcp:access",
]


def generate_token(
    user_id: str = "default_user",
    org_id: str = "default_org",
    scopes: list[str] = None,
    grants: list[str] = None,
    secret_key: str = None,
    issuer: str = "https://codingbrain.local",
    audience: str = "https://api.codingbrain.local",
    expires_days: int = 365,
) -> str:
    """Generate a JWT token with the specified claims.

    Args:
        user_id: The user identifier (sub claim)
        org_id: The organization/tenant ID
        scopes: List of OAuth2 scopes (default: all)
        grants: List of access_entity grants for multi-user memory routing
                Format: prefix:value (e.g., org:cloudfactory, team:cloudfactory/backend)
        secret_key: JWT signing key (default: from JWT_SECRET_KEY env)
        issuer: Token issuer (iss claim)
        audience: Token audience (aud claim)
        expires_days: Token validity in days
    """

    if secret_key is None:
        secret_key = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_jwt_secret_at_least_32_chars_long")

    if scopes is None:
        scopes = ALL_SCOPES

    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "org_id": org_id,
        "jti": str(uuid.uuid4()),
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + timedelta(days=expires_days),
        "scope": " ".join(scopes),
    }

    # Add grants if specified (for multi-user memory routing)
    if grants:
        payload["grants"] = grants

    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Generate JWT token for Coding Brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default token (all scopes, no explicit grants)
  python scripts/generate_jwt.py

  # Token with specific scopes
  python scripts/generate_jwt.py --scopes "memories:read memories:write"

  # Token with access grants for multi-user memory routing
  python scripts/generate_jwt.py --grants "org:cloudfactory team:cloudfactory/backend"

  # Full example with user, org, and grants
  python scripts/generate_jwt.py --user alice --org acme --grants "org:acme project:acme/billing"

Grant formats:
  user:<user_id>      - Access to personal memories
  team:<org>/<team>   - Access to team memories
  project:<org>/<path> - Access to project memories (expands to teams below)
  org:<org>           - Access to org memories (expands to projects/teams below)
""",
    )
    parser.add_argument("--user", default="default_user", help="User ID (sub claim)")
    parser.add_argument("--org", default="default_org", help="Organization ID")
    parser.add_argument("--scopes", default=None, help="Space-separated scopes (default: all)")
    parser.add_argument(
        "--grants",
        default=None,
        help="Space-separated access grants for multi-user memory routing (e.g., 'org:cloudfactory team:cloudfactory/backend')",
    )
    parser.add_argument("--secret", default=None, help="JWT secret key (default: from JWT_SECRET_KEY env)")
    parser.add_argument("--expires", type=int, default=365, help="Expiration in days")
    parser.add_argument("--list-scopes", action="store_true", help="List all available scopes")

    args = parser.parse_args()

    if args.list_scopes:
        print("Available scopes:")
        for scope in ALL_SCOPES:
            print(f"  {scope}")
        return

    scopes = args.scopes.split() if args.scopes else None
    grants = args.grants.split() if args.grants else None

    token = generate_token(
        user_id=args.user,
        org_id=args.org,
        scopes=scopes,
        grants=grants,
        secret_key=args.secret,
        expires_days=args.expires,
    )

    print(f"\nGenerated JWT token:\n")
    print(token)
    print(f"\nUser: {args.user}")
    print(f"Org: {args.org}")
    print(f"Scopes: {' '.join(scopes) if scopes else 'all'}")
    print(f"Grants: {' '.join(grants) if grants else 'none (user:<user_id> implied)'}")
    print(f"Expires: {args.expires} days")
    print(f"\nTo use in .env:")
    print(f"NEXT_PUBLIC_API_TOKEN={token}")


if __name__ == "__main__":
    main()
