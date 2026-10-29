"""Session tokens shared by the twin-api and the Streamlit pages under /legacy (SV-6.6, #320)."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from services.session_tokens import (
    ISSUER,
    LEEWAY_S,
    RedisRevocations,
    SessionSettings,
    TokenError,
    decode,
    encode,
    verify_session,
)
from tests.redis_server import needs_redis, redis_server

KEY = b"k" * 48
NOW = 1_793_000_000


def claims(**overrides):
    base = {
        "iss": ISSUER,
        "sub": "sup",
        "uid": "id-sup",
        "role": "supervisor",
        "jti": "jti-1",
        "iat": NOW,
        "exp": NOW + 3600,
    }
    base.update(overrides)
    return base


def test_valid_token_maps_to_principal_and_claims() -> None:
    session = verify_session(encode(claims(), KEY), KEY, now=NOW)

    assert session is not None
    assert (session.principal.username, session.principal.role, session.principal.user_id) == (
        "sup",
        "supervisor",
        "id-sup",
    )
    assert session.jti == "jti-1"
    assert session.expires_at == NOW + 3600


@pytest.mark.parametrize("role", ["user", "supervisor", "admin"])
def test_roles_are_normalised(role: str) -> None:
    session = verify_session(encode(claims(role=role.upper()), KEY), KEY, now=NOW)
    assert session is not None and session.principal.role == role


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        ({"role": "root"}, "unknown role"),
        ({"iss": "other"}, "other issuer"),
        ({"jti": None}, "no token id"),
        ({"jti": ""}, "empty token id"),
        ({"exp": NOW - LEEWAY_S - 1}, "expired"),
        ({"nbf": NOW + LEEWAY_S + 1}, "not valid yet"),
    ],
)
def test_invalid_tokens_give_no_session(bad: dict, reason: str) -> None:
    token = encode(claims(**bad), KEY)
    assert verify_session(token, KEY, now=NOW) is None, reason


def test_expiry_leeway_matches_decode() -> None:
    token = encode(claims(exp=NOW - LEEWAY_S), KEY)
    assert decode(token, KEY, now=NOW)["jti"] == "jti-1"
    assert verify_session(token, KEY, now=NOW) is not None
    assert verify_session(token, KEY, now=NOW + 1) is None


def test_wrong_key_malformed_and_empty_tokens() -> None:
    token = encode(claims(), KEY)
    assert verify_session(token, b"x" * 48, now=NOW) is None
    assert verify_session("not.a.token", KEY, now=NOW) is None
    assert verify_session(None, KEY, now=NOW) is None
    assert verify_session("", KEY, now=NOW) is None
    with pytest.raises(TokenError):
        decode(token, b"x" * 48, now=NOW)


def test_revoked_id_gives_no_session_and_is_checked_last() -> None:
    asked: list[str] = []

    def revoked(jti: str) -> bool:
        asked.append(jti)
        return jti == "jti-1"

    assert verify_session(encode(claims(), KEY), KEY, now=NOW, is_revoked=revoked) is None
    assert verify_session(encode(claims(jti="jti-2"), KEY), KEY, now=NOW, is_revoked=revoked) is not None
    assert verify_session(encode(claims(iss="other"), KEY), KEY, now=NOW, is_revoked=revoked) is None
    assert asked == ["jti-1", "jti-2"]  # a token that fails earlier never reaches the store


# ---------------------------------------------------------------------------------------------- RedisRevocations


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int]] = {}
        self.down = False
        self.calls = 0

    def _check(self) -> None:
        self.calls += 1
        if self.down:
            raise ConnectionError("refused")

    def set(self, key: str, value: bytes, ex: int) -> None:
        self._check()
        self.store[key] = (value, ex)

    def exists(self, key: str) -> int:
        self._check()
        return int(key in self.store)


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_redis_revocations_add_and_contains_with_ttl() -> None:
    fake = FakeRedis()
    wall = Clock(float(NOW))
    store = RedisRevocations("redis://unused", client=fake, wall_clock=wall)

    store.add("jti-1", NOW + 100)
    assert fake.store[RedisRevocations.key("jti-1")] == (b"1", 100 + LEEWAY_S)
    assert store.contains("jti-1")
    assert not store.contains("jti-2")

    store.add("jti-old", NOW - LEEWAY_S - 1)  # already expired: nothing to revoke
    assert RedisRevocations.key("jti-old") not in fake.store


def test_redis_revocations_outage_revokes_nothing_and_retries_later() -> None:
    fake = FakeRedis()
    mono = Clock(1000.0)
    store = RedisRevocations("redis://unused", client=fake, clock=mono, wall_clock=Clock(float(NOW)), retry_s=15.0)
    store.add("jti-1", NOW + 100)

    fake.down = True
    assert not store.contains("jti-1")  # the failure is seen once ...
    calls = fake.calls
    assert not store.contains("jti-1")  # ... then no round trip until retry_s has passed
    store.add("jti-2", NOW + 100)
    assert fake.calls == calls
    assert not store.reachable

    fake.down = False
    mono.t += 15.0
    assert store.reachable
    assert store.contains("jti-1")
    assert RedisRevocations.key("jti-2") not in fake.store  # dropped during the outage


@needs_redis
def test_redis_revocations_against_a_server(tmp_path) -> None:
    with redis_server(tmp_path) as url:
        api_side = RedisRevocations(url)
        streamlit_side = RedisRevocations(url)
        jti = uuid.uuid4().hex
        api_side.add(jti, api_side._wall_clock() + 60)
        assert streamlit_side.contains(jti)  # a sign-out in one process is seen by the other
        assert not streamlit_side.contains(uuid.uuid4().hex)


# ------------------------------------------------------------------------------------------------ SessionSettings


def test_session_settings_key_and_revocations(monkeypatch) -> None:
    monkeypatch.delenv("BF_API_JWT_SECRET", raising=False)
    monkeypatch.delenv("BF_REDIS_URL", raising=False)
    monkeypatch.delenv("BF_API_REDIS_URL", raising=False)
    settings = SessionSettings(_env_file=None)
    assert settings.signing_key() is None
    assert settings.revocations() is None
    assert settings.cookie_name == "bf_session"

    monkeypatch.setenv("BF_API_JWT_SECRET", "s" * 32)
    monkeypatch.setenv("BF_REDIS_URL", "redis://127.0.0.1:1/0")
    settings = SessionSettings(_env_file=None)
    assert settings.signing_key() == b"s" * 32
    assert settings.revocations() is not None

    with pytest.raises(ValidationError):
        SessionSettings(_env_file=None, jwt_secret="too-short")
