"""Session tokens of the twin-api, shared with the Streamlit pages under ``/legacy`` (SV-0.4 #273, SV-6.6 #320).

Tokens are JSON Web Tokens (RFC 7519) signed with HMAC-SHA256 (JWS HS256, RFC 7515), via PyJWT. Only HS256 is accepted
(``algorithms=["HS256"]``), so ``"alg": "none"`` and other algorithms are refused. PyJWT checks the signature and the
presence of ``exp``; the validity period is checked here against the caller's clock, so the API, Streamlit and their
tests use one time source. Times are whole seconds since the Unix epoch (UTC).

A session is valid when the token verifies, its issuer is the twin-api, its role is one of ``services.auth.ROLES``, it
carries a token id (``jti``) and that id is not revoked. Revoked ids live in the API process (``api.security.limits.
DenyList``) and, with ``BF_REDIS_URL``, in Redis (``RedisRevocations``), where the Streamlit process sees them too.

No FastAPI and no Streamlit import: both sides import this module.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jwt
from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from services.auth import Principal, normalise_role

log = logging.getLogger("services.session_tokens")

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM = "HS256"
ISSUER = "bf02-twin-api"
LEEWAY_S = 30
REVOKED_KEY_PREFIX = "twin:auth:revoked:"


class TokenError(Exception):
    """The token is malformed, carries a bad signature, or is outside its validity period."""


def encode(claims: dict[str, Any], key: bytes) -> str:
    return jwt.encode(claims, key, algorithm=ALGORITHM)


def _int_claim(claims: dict[str, Any], name: str) -> int | None:
    value = claims.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TokenError(f"Claim {name} must be an integer.")
    return value


def decode(token: str, key: bytes, *, now: int, leeway: int = LEEWAY_S) -> dict[str, Any]:
    """Verify the signature and the validity period; return the claims.

    ``exp`` is required; ``iat`` and ``nbf``, when present, may not lie in the future by more than ``leeway`` seconds.
    """
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[ALGORITHM],
            options={"require": ["exp"], "verify_exp": False, "verify_iat": False, "verify_nbf": False},
        )
    except jwt.InvalidTokenError as exc:
        raise TokenError(str(exc)) from exc
    exp = _int_claim(claims, "exp")
    if exp is None or now > exp + leeway:
        raise TokenError("The token has expired.")
    for name in ("iat", "nbf"):
        value = _int_claim(claims, name)
        if value is not None and value > now + leeway:
            raise TokenError(f"The token is not valid yet ({name}).")
    return claims


@dataclass(frozen=True)
class Session:
    """A valid session: the signed-in user and the token claims."""

    principal: Principal
    claims: Mapping[str, Any]

    @property
    def jti(self) -> str:
        return str(self.claims["jti"])

    @property
    def expires_at(self) -> int:
        """End of the session, s since the Unix epoch (UTC)."""
        return int(self.claims["exp"])


def verify_session(
    token: str | None, key: bytes, *, now: int, is_revoked: Callable[[str], bool] | None = None
) -> Session | None:
    """The session of ``token`` when it is valid and not revoked; None otherwise (never raises on a bad token)."""
    if not token:
        return None
    try:
        claims = decode(token, key, now=now)
    except TokenError:
        return None
    role = normalise_role(claims.get("role"))
    jti = claims.get("jti")
    if role is None or not isinstance(jti, str) or not jti or claims.get("iss") != ISSUER:
        return None
    if is_revoked is not None and is_revoked(jti):
        return None
    uid = claims.get("uid")
    principal = Principal(username=str(claims.get("sub")), role=role, user_id=uid if isinstance(uid, str) else None)
    return Session(principal=principal, claims=claims)


class RevocationStore(Protocol):
    def contains(self, jti: str) -> bool:
        """The token id was revoked and the token has not expired yet."""
        ...

    def add(self, jti: str, expires_at: float) -> None:
        """Revoke ``jti`` until ``expires_at`` (s since the Unix epoch)."""
        ...


class RedisRevocations:
    """Revoked token ids in Redis, shared by the API process and Streamlit.

    One key per id (``twin:auth:revoked:<jti>``) that expires ``LEEWAY_S`` after the token. While Redis is not
    reachable, ``contains`` answers False and ``add`` drops the entry (logged at most once per ``retry_s``); the
    connection is tried again after ``retry_s``, so an outage costs one connect timeout (``timeout_s``) per ``retry_s``.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 0.25,
        retry_s: float = 15.0,
        client: Any = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._retry_s = retry_s
        self._client = client
        self._clock = clock
        self._wall_clock = wall_clock
        self._down_until = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def key(jti: str) -> str:
        return REVOKED_KEY_PREFIX + jti

    def _redis(self) -> Any:
        with self._lock:
            if self._client is None:
                import redis

                self._client = redis.Redis.from_url(
                    self._url, socket_timeout=self._timeout_s, socket_connect_timeout=self._timeout_s
                )
            return self._client

    @property
    def reachable(self) -> bool:
        """False while the last failure is less than ``retry_s`` ago."""
        return self._clock() >= self._down_until

    def _failed(self, action: str, exc: Exception) -> None:
        if self.reachable:
            log.warning("revoked sessions: Redis not reachable (%s: %s); retry in %.0f s", action, exc, self._retry_s)
        self._down_until = self._clock() + self._retry_s

    def contains(self, jti: str) -> bool:
        if not self.reachable:
            return False
        try:
            return bool(self._redis().exists(self.key(jti)))
        except Exception as exc:  # redis.RedisError, OSError: an unreachable store revokes nothing
            self._failed("check", exc)
            return False

    def add(self, jti: str, expires_at: float) -> None:
        ttl = math.ceil(expires_at - self._wall_clock()) + LEEWAY_S
        if ttl <= 0 or not self.reachable:
            return
        try:
            self._redis().set(self.key(jti), b"1", ex=ttl)
        except Exception as exc:
            self._failed("revoke", exc)


class SessionSettings(BaseSettings):
    """Session settings read by the twin-api (``api.settings.ApiSettings`` extends them) and by Streamlit."""

    model_config = SettingsConfigDict(env_prefix="BF_API_", env_file=REPO_ROOT / ".env", extra="ignore")

    jwt_secret: SecretStr | None = Field(
        None, description="HS256 signing key, at least 32 characters; required in prod (generate: openssl rand -hex 32)"
    )
    cookie_name: str = "bf_session"
    redis_url: str | None = Field(
        None,
        validation_alias=AliasChoices("BF_REDIS_URL", "BF_API_REDIS_URL", "redis_url"),
        description="Redis shared with the worker, e.g. redis://127.0.0.1:6379/0; without it events stay in-process",
    )

    @field_validator("jwt_secret")
    @classmethod
    def _secret_length(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and len(value.get_secret_value()) < 32:
            raise ValueError("BF_API_JWT_SECRET must have at least 32 characters")
        return value

    def signing_key(self) -> bytes | None:
        return None if self.jwt_secret is None else self.jwt_secret.get_secret_value().encode()

    def revocations(self) -> RedisRevocations | None:
        """The shared revocation list when ``BF_REDIS_URL`` is set."""
        return RedisRevocations(self.redis_url) if self.redis_url else None
