"""Tests for MCP structured input error hints."""

import json
from datetime import datetime, timezone

import pytest

from app.security.types import Principal, TokenClaims


def _make_principal(user_id: str) -> Principal:
    claims = TokenClaims(
        sub=user_id,
        iss="https://auth.test.example.com",
        aud="https://api.test.example.com",
        exp=datetime.now(timezone.utc),
        iat=datetime.now(timezone.utc),
        jti=f"jti-{user_id}",
        org_id="test-org",
        scopes={"memories:write"},
        grants={f"user:{user_id}"},
    )
    return Principal(user_id=user_id, org_id="test-org", claims=claims)


@pytest.mark.asyncio
async def test_add_memories_missing_access_entity_returns_hint():
    from app.mcp_server import (
        add_memories,
        user_id_var,
        client_name_var,
        principal_var,
    )

    user_token = user_id_var.set("alice")
    client_token = client_name_var.set("test-app")
    principal_token = principal_var.set(_make_principal("alice"))

    try:
        result = await add_memories(
            text="Test memory",
            category="decision",
            scope="team",
        )
    finally:
        principal_var.reset(principal_token)
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)

    response = json.loads(result)

    assert "error" in response
    assert "access_entity is required for scope='team'" in response["error"]
    assert "hint" in response
