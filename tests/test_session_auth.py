from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


class _RerunRaised(RuntimeError):
    pass


def _load_session_module(monkeypatch):
    streamlit_stub = types.SimpleNamespace(
        session_state={},
        rerun=lambda: (_ for _ in ()).throw(_RerunRaised()),
    )
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    sys.modules.pop("utils.session", None)
    return importlib.import_module("utils.session"), streamlit_stub


def test_session_auth_does_not_import_cookie_component() -> None:
    source = Path("src/utils/session.py").read_text(encoding="utf-8")

    assert "streamlit_cookies_manager" not in source
    assert "EncryptedCookieManager" not in source
    assert "cookies.ready" not in source


def test_login_stores_role_and_derived_permissions(monkeypatch) -> None:
    session, streamlit_stub = _load_session_module(monkeypatch)

    session.login_user("shift_supervisor", "supervisor")

    assert streamlit_stub.session_state["auth_user"] == "shift_supervisor"
    assert streamlit_stub.session_state["role"] == "supervisor"
    assert "hopper:write" in streamlit_stub.session_state["permissions"]
    assert "users:write" not in streamlit_stub.session_state["permissions"]
    assert session.is_logged_in()
    assert session.has_permission("feedback:moderate")


def test_logout_clears_auth_state(monkeypatch) -> None:
    session, streamlit_stub = _load_session_module(monkeypatch)
    session.login_user("admin", "admin")

    with pytest.raises(_RerunRaised):
        session.logout_user()

    assert "auth_user" not in streamlit_stub.session_state
    assert "role" not in streamlit_stub.session_state
    assert "permissions" not in streamlit_stub.session_state
