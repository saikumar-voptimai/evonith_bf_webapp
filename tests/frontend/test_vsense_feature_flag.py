from __future__ import annotations

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled


def test_vsense_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_VSENSE", raising=False)
    monkeypatch.delenv("USE_BACKEND_API_RECOMMENDATIONS", raising=False)

    assert is_backend_api_enabled("vsense") is False


def test_vsense_flag_selects_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_VSENSE", "true")
    monkeypatch.setenv("USE_BACKEND_API_RECOMMENDATIONS", "false")

    assert is_backend_api_enabled("vsense") is True


def test_recommendations_flag_is_temporary_vsense_alias(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_VSENSE", raising=False)
    monkeypatch.setenv("USE_BACKEND_API_RECOMMENDATIONS", "true")

    assert is_backend_api_enabled("vsense") is True
