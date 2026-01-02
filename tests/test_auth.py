"""
Authentication Module Tests
"""

import pytest
import time
from common.auth import (
    extract_bearer_token,
    decode_jwt_hs256,
    validate_claims,
    extract_scopes,
    require_scope,
    generate_jwt_hs256,
    JWTBearer,
    JWTConfig,
    AuthError,
)


class TestExtractBearerToken:
    """Tests for extract_bearer_token function."""

    def test_valid_bearer_token(self):
        token = extract_bearer_token("Bearer abc123")
        assert token == "abc123"

    def test_valid_bearer_case_insensitive(self):
        token = extract_bearer_token("bearer abc123")
        assert token == "abc123"

    def test_missing_authorization(self):
        with pytest.raises(AuthError, match="Missing Authorization header"):
            extract_bearer_token(None)

    def test_empty_authorization(self):
        with pytest.raises(AuthError, match="Missing Authorization header"):
            extract_bearer_token("")

    def test_invalid_format_no_bearer(self):
        with pytest.raises(AuthError, match="Invalid Authorization header"):
            extract_bearer_token("Basic abc123")

    def test_invalid_format_no_token(self):
        with pytest.raises(AuthError, match="Invalid Authorization header"):
            extract_bearer_token("Bearer")


class TestJWTEncoding:
    """Tests for JWT encoding and decoding."""

    def test_generate_and_decode(self):
        payload = {"sub": "user123", "exp": int(time.time()) + 3600}
        secret = "test-secret"

        token = generate_jwt_hs256(payload, secret)
        decoded = decode_jwt_hs256(token, secret)

        assert decoded["sub"] == "user123"

    def test_invalid_signature(self):
        payload = {"sub": "user123", "exp": int(time.time()) + 3600}
        token = generate_jwt_hs256(payload, "secret1")

        with pytest.raises(AuthError, match="Invalid JWT signature"):
            decode_jwt_hs256(token, "wrong-secret")

    def test_malformed_token(self):
        with pytest.raises(AuthError, match="Malformed JWT"):
            decode_jwt_hs256("not.a.valid.token.parts", "secret")


class TestValidateClaims:
    """Tests for JWT claim validation."""

    def test_valid_claims(self):
        payload = {
            "sub": "user123",
            "exp": int(time.time()) + 3600,
            "aud": "my-app",
            "iss": "auth-server",
        }
        # Should not raise
        validate_claims(payload, audience="my-app", issuer="auth-server")

    def test_expired_token(self):
        payload = {"sub": "user123", "exp": int(time.time()) - 100}
        with pytest.raises(AuthError, match="Token expired"):
            validate_claims(payload, audience=None, issuer=None)

    def test_invalid_audience(self):
        payload = {"sub": "user123", "aud": "other-app"}
        with pytest.raises(AuthError, match="Invalid audience"):
            validate_claims(payload, audience="my-app", issuer=None)

    def test_invalid_issuer(self):
        payload = {"sub": "user123", "iss": "other-server"}
        with pytest.raises(AuthError, match="Invalid issuer"):
            validate_claims(payload, audience=None, issuer="auth-server")


class TestExtractScopes:
    """Tests for scope extraction."""

    def test_scope_as_string(self):
        payload = {"scope": "read write admin"}
        scopes = list(extract_scopes(payload))
        assert scopes == ["read", "write", "admin"]

    def test_scope_as_list(self):
        payload = {"scopes": ["read", "write"]}
        scopes = list(extract_scopes(payload))
        assert scopes == ["read", "write"]

    def test_no_scopes(self):
        payload = {"sub": "user123"}
        scopes = list(extract_scopes(payload))
        assert scopes == []


class TestRequireScope:
    """Tests for scope requirement."""

    def test_has_required_scope(self):
        payload = {"scope": "read write admin"}
        # Should not raise
        require_scope(payload, "write")

    def test_missing_required_scope(self):
        payload = {"scope": "read"}
        with pytest.raises(AuthError, match="Missing required scope"):
            require_scope(payload, "admin")


class TestJWTBearer:
    """Tests for JWTBearer class."""

    def test_valid_authentication(self):
        config = JWTConfig(
            secret="test-secret",
            required_scope="api:read",
        )
        bearer = JWTBearer(config)

        payload = {
            "sub": "user123",
            "exp": int(time.time()) + 3600,
            "scope": "api:read api:write",
        }
        token = generate_jwt_hs256(payload, "test-secret")

        result = bearer(f"Bearer {token}")
        assert result["sub"] == "user123"

    def test_missing_scope(self):
        config = JWTConfig(
            secret="test-secret",
            required_scope="admin:write",
        )
        bearer = JWTBearer(config)

        payload = {
            "sub": "user123",
            "exp": int(time.time()) + 3600,
            "scope": "api:read",
        }
        token = generate_jwt_hs256(payload, "test-secret")

        with pytest.raises(AuthError, match="Missing required scope"):
            bearer(f"Bearer {token}")
