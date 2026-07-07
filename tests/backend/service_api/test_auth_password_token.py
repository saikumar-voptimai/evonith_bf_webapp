"""Tests for Phase 5 password and token services."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.services.password_service import PasswordService, PasswordValidationError
from apps.backend_api.app.services.token_service import TokenConfigurationError, TokenError, TokenService


def _settings(**overrides) -> BackendSettings:
    values = {
        "backend_env": "test",
        "auth_secret_key": "test-secret",
        "auth_password_hash_scheme": "bcrypt",
        "auth_min_password_length": 8,
    }
    values.update(overrides)
    return BackendSettings(**values)


def test_bcrypt_password_hash_verifies_and_is_not_plaintext():
    service = PasswordService(_settings())

    hashed = service.hash_password("safe-pass-123")
    result = service.verify_password(password="safe-pass-123", stored_hash=hashed)

    assert hashed != "safe-pass-123"
    assert hashed.startswith("$2")
    assert result.valid is True
    assert result.legacy_hash is False


def test_legacy_sha256_password_hash_can_be_verified_and_upgraded():
    service = PasswordService(_settings())
    legacy_hash = hashlib.sha256("legacy-pass".encode()).hexdigest()

    result = service.verify_password(password="legacy-pass", stored_hash=legacy_hash)

    assert result.valid is True
    assert result.legacy_hash is True
    assert result.needs_rehash is True


def test_new_password_minimum_length_is_enforced():
    service = PasswordService(_settings(auth_min_password_length=12))

    with pytest.raises(PasswordValidationError):
        service.hash_password("too-short")


def test_token_roundtrip_contains_access_claims():
    service = TokenService(_settings())

    token = service.create_access_token(
        user_id="user-1",
        username="operator",
        role="user",
        permissions=[],
    )
    claims = service.verify_access_token(token.token)

    assert claims["sub"] == "user-1"
    assert claims["username"] == "operator"
    assert claims["type"] == "access"


def test_expired_token_is_rejected():
    service = TokenService(_settings(auth_access_token_expire_minutes=1))
    token = service.create_access_token(
        user_id="user-1",
        username="operator",
        role="user",
        permissions=[],
        now=datetime.now(timezone.utc) - timedelta(minutes=2),
    )

    with pytest.raises(TokenError) as exc_info:
        service.verify_access_token(token.token)

    assert exc_info.value.code == "TOKEN_EXPIRED"


def test_production_requires_auth_secret_key():
    service = TokenService(
        _settings(
            backend_env="production",
            auth_secret_key="",
            auth_require_secret_in_production=True,
        )
    )

    with pytest.raises(TokenConfigurationError):
        service.create_access_token(
            user_id="user-1",
            username="operator",
            role="user",
            permissions=[],
        )
