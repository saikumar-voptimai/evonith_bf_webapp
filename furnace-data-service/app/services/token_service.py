"""Access token creation and verification for backend auth."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import BackendSettings, load_backend_settings

_DEV_SECRET = secrets.token_urlsafe(32)
_PRODUCTION_ENVS = {"prod", "production", "edge"}


class TokenError(RuntimeError):
    """Raised when a token cannot be accepted."""

    def __init__(self, code: str, message: str, status_code: int = 401) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class TokenConfigurationError(TokenError):
    """Raised when token configuration is unsafe."""

    def __init__(self, message: str) -> None:
        super().__init__("AUTH_CONFIG_ERROR", message, status_code=500)


@dataclass(frozen=True)
class AccessToken:
    token: str
    expires_at: datetime
    expires_in: int


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


class TokenService:
    """Create and verify signed bearer tokens."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def _secret(self) -> bytes:
        secret = str(self.settings.auth_secret_key or "").strip()
        if secret:
            return secret.encode("utf-8")

        if (
            self.settings.backend_env.strip().lower() in _PRODUCTION_ENVS
            and self.settings.auth_require_secret_in_production
        ):
            raise TokenConfigurationError(
                "EVONITH_AUTH_SECRET_KEY must be set in production."
            )

        return _DEV_SECRET.encode("utf-8")

    def _sign(self, signing_input: str) -> str:
        if self.settings.auth_algorithm != "HS256":
            raise TokenConfigurationError(
                f"Unsupported auth token algorithm: {self.settings.auth_algorithm}"
            )
        digest = hmac.new(
            self._secret(),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return _b64encode(digest)

    def create_access_token(
        self,
        *,
        user_id: str,
        username: str,
        role: str,
        permissions: list[str],
        now: datetime | None = None,
    ) -> AccessToken:
        """Create a signed access token."""
        issued_at = now or datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(
            minutes=self.settings.auth_access_token_expire_minutes
        )
        header = {"alg": self.settings.auth_algorithm, "typ": "JWT"}
        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "permissions": sorted(permissions),
            "type": "access",
            "iat": int(issued_at.timestamp()),
            "exp": int(expires_at.timestamp()),
        }
        signing_input = ".".join(
            [_b64encode(_json_bytes(header)), _b64encode(_json_bytes(payload))]
        )
        token = f"{signing_input}.{self._sign(signing_input)}"
        return AccessToken(
            token=token,
            expires_at=expires_at,
            expires_in=max(0, int((expires_at - issued_at).total_seconds())),
        )

    def verify_access_token(self, token: str) -> dict[str, Any]:
        """Verify a bearer token and return claims."""
        parts = str(token or "").split(".")
        if len(parts) != 3:
            raise TokenError("INVALID_TOKEN", "Invalid access token.")

        signing_input = ".".join(parts[:2])
        expected_signature = self._sign(signing_input)
        if not hmac.compare_digest(expected_signature, parts[2]):
            raise TokenError("INVALID_TOKEN", "Invalid access token.")

        try:
            header = json.loads(_b64decode(parts[0]))
            payload = json.loads(_b64decode(parts[1]))
        except (ValueError, json.JSONDecodeError) as exc:
            raise TokenError("INVALID_TOKEN", "Invalid access token.") from exc

        if header.get("alg") != self.settings.auth_algorithm:
            raise TokenError("INVALID_TOKEN", "Invalid access token.")
        if payload.get("type") != "access":
            raise TokenError("INVALID_TOKEN", "Invalid token type.")

        expires_at = int(payload.get("exp") or 0)
        if expires_at <= int(datetime.now(timezone.utc).timestamp()):
            raise TokenError("TOKEN_EXPIRED", "Access token has expired.")

        if not payload.get("sub"):
            raise TokenError("INVALID_TOKEN", "Invalid access token.")

        return payload
