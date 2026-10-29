"""Single sign-on of the Streamlit pages under /legacy from the twin-api session cookie (SV-6.6, #320).

``utils.session`` is imported with a Streamlit stub (session state, ``st.context.cookies``, ``st.rerun``); the token
rules are the real ``services.session_tokens``.
"""

from __future__ import annotations

import types

import pytest

from services.session_tokens import ISSUER, encode
from tests.module_stubs import import_with_stubs

KEY = b"k" * 48
NOW = 1_793_000_000


class _RerunRaised(RuntimeError):
    pass


class FakeRevocations:
    def __init__(self) -> None:
        self.revoked: dict[str, float] = {}

    def contains(self, jti: str) -> bool:
        return jti in self.revoked

    def add(self, jti: str, expires_at: float) -> None:
        self.revoked[jti] = expires_at


def _rerun() -> None:
    raise _RerunRaised()


@pytest.fixture()
def env(monkeypatch):
    """The session module, the Streamlit stub and the SSO settings it sees."""
    st = types.SimpleNamespace(session_state={}, rerun=_rerun, context=types.SimpleNamespace(cookies={}))
    session = import_with_stubs("utils.session", {"streamlit": vars(st)})
    revocations = FakeRevocations()
    clock = types.SimpleNamespace(t=float(NOW))
    sso = session.ApiSso(key=KEY, cookie_name="bf_session", revocations=revocations, clock=lambda: clock.t)
    monkeypatch.setattr(session, "api_sso", lambda: sso)
    return types.SimpleNamespace(session=session, st=st, revocations=revocations, clock=clock, sso=sso)


def token(role: str = "supervisor", *, jti: str = "jti-1", username: str = "sup", **overrides) -> str:
    claims = {
        "iss": ISSUER,
        "sub": username,
        "uid": f"id-{username}",
        "role": role,
        "jti": jti,
        "iat": NOW,
        "exp": NOW + 12 * 3600,
    }
    claims.update(overrides)
    return encode(claims, KEY)


def set_cookie(env, value: str | None) -> None:
    env.st.context.cookies = {} if value is None else {"bf_session": value}


def test_api_cookie_signs_the_streamlit_session_in(env) -> None:
    set_cookie(env, token())

    assert env.session.sync_api_session()

    state = env.st.session_state
    assert env.session.is_logged_in()
    assert state["auth_user"] == "sup"
    assert state["auth_user_id"] == "id-sup"
    assert state["role"] == "supervisor"
    assert state[env.session.AUTH_SOURCE_KEY] == env.session.AUTH_SOURCE_API
    assert state[env.session.AUTH_JTI_KEY] == "jti-1"
    assert state[env.session.AUTH_EXPIRES_KEY] == NOW + 12 * 3600
    assert env.session.current_user_id() == "id-sup"


@pytest.mark.parametrize("role", ["user", "supervisor", "admin"])
def test_role_restrictions_match_the_streamlit_login(env, role: str) -> None:
    session, st = env.session, env.st
    session.login_user("someone", role, user_id="id-x")
    expected = (
        session.is_admin(),
        session.is_supervisor(),
        set(session.current_permissions()),
        session.has_permission("users:write"),
        session.has_permission("hopper:write"),
    )
    st.session_state.clear()

    set_cookie(env, token(role, username="someone"))
    assert session.sync_api_session()

    assert st.session_state["role"] == role
    actual = (
        session.is_admin(),
        session.is_supervisor(),
        set(session.current_permissions()),
        session.has_permission("users:write"),
        session.has_permission("hopper:write"),
    )
    assert actual == expected
    assert session.is_admin() == (role == "admin")
    assert session.is_supervisor() == (role == "supervisor")


@pytest.mark.parametrize(
    "bad",
    [
        {"exp": NOW - 3600},
        {"iss": "someone-else"},
        {"role": "root"},
        {"jti": None},
    ],
)
def test_invalid_cookie_leaves_the_login_form(env, bad: dict) -> None:
    set_cookie(env, token(**bad))
    assert not env.session.sync_api_session()
    assert not env.session.is_logged_in()


def test_wrong_key_and_missing_cookie(env) -> None:
    set_cookie(env, encode({"iss": ISSUER, "sub": "x", "role": "admin", "jti": "j", "exp": NOW + 10}, b"z" * 48))
    assert not env.session.sync_api_session()
    set_cookie(env, None)
    assert not env.session.sync_api_session()
    assert not env.session.is_logged_in()


def test_no_signing_key_means_streamlit_login_only(env, monkeypatch) -> None:
    monkeypatch.setattr(env.session, "api_sso", lambda: None)
    set_cookie(env, token())
    assert not env.session.sync_api_session()
    assert not env.session.is_logged_in()


def test_revoked_or_expired_api_session_ends_the_streamlit_session(env) -> None:
    set_cookie(env, token())
    assert env.session.sync_api_session()

    env.revocations.add("jti-1", NOW + 12 * 3600)  # sign-out in the Svelte frontend, seen through Redis
    assert not env.session.sync_api_session()
    assert not env.session.is_logged_in()
    assert env.session.AUTH_SOURCE_KEY not in env.st.session_state

    env.revocations.revoked.clear()
    assert env.session.sync_api_session()
    env.clock.t = NOW + 12 * 3600 + 60  # the API session has expired
    assert not env.session.sync_api_session()
    assert not env.session.is_logged_in()


def test_new_api_session_replaces_the_old_one(env) -> None:
    set_cookie(env, token("user", jti="jti-1", username="op"))
    assert env.session.sync_api_session()
    set_cookie(env, token("admin", jti="jti-2", username="adm"))
    assert env.session.sync_api_session()
    assert env.st.session_state["auth_user"] == "adm"
    assert env.session.is_admin()
    assert env.st.session_state[env.session.AUTH_JTI_KEY] == "jti-2"


def test_streamlit_login_is_not_overridden_by_the_cookie(env) -> None:
    env.session.login_user("form_user", "admin", user_id="id-form")
    set_cookie(env, token("user", username="op"))
    assert not env.session.sync_api_session()
    assert env.st.session_state["auth_user"] == "form_user"
    assert env.session.is_admin()
    assert env.session.AUTH_SOURCE_KEY not in env.st.session_state


def test_logout_revokes_the_api_session_and_declines_the_cookie(env) -> None:
    set_cookie(env, token())
    assert env.session.sync_api_session()

    with pytest.raises(_RerunRaised):
        env.session.logout_user()

    assert not env.session.is_logged_in()
    assert env.revocations.revoked == {"jti-1": float(NOW + 12 * 3600)}
    assert env.st.session_state[env.session.SSO_DECLINED_KEY] == "jti-1"
    env.revocations.revoked.clear()  # without Redis the cookie stays valid for the API ...
    assert not env.session.sync_api_session()  # ... and this Streamlit session still refuses it
    set_cookie(env, token(jti="jti-2"))  # a new sign-in in the frontend signs in again
    assert env.session.sync_api_session()
    assert env.st.session_state[env.session.AUTH_JTI_KEY] == "jti-2"


def test_form_logout_leaves_api_sessions_alone(env) -> None:
    env.session.login_user("form_user", "user")
    with pytest.raises(_RerunRaised):
        env.session.logout_user()
    assert env.revocations.revoked == {}
    assert env.session.SSO_DECLINED_KEY not in env.st.session_state


def test_shared_token_module_pulls_no_fastapi_into_streamlit() -> None:
    """Streamlit verifies the cookie through ``services.session_tokens``: neither side imports FastAPI or ``api.*``."""
    import re
    from pathlib import Path

    for path in ("src/services/session_tokens.py", "src/utils/session.py"):
        source = Path(path).read_text(encoding="utf-8")
        imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, re.M)
        assert not [m for m in imports if m == "fastapi" or m == "api" or m.startswith("api.")], path
