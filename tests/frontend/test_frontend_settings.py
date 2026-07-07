"""Tests for frontend API settings and feature flags."""

from apps.frontend_streamlit.config.frontend_settings import DEFAULT_BACKEND_API_BASE_URL, is_backend_api_enabled, load_frontend_settings


def test_default_backend_api_base_url(monkeypatch):
    monkeypatch.delenv("BACKEND_API_BASE_URL", raising=False)

    settings = load_frontend_settings()

    assert settings.backend_api_base_url == DEFAULT_BACKEND_API_BASE_URL


def test_trailing_slash_is_normalized(monkeypatch):
    monkeypatch.setenv("BACKEND_API_BASE_URL", "http://localhost:8080/api/v1/")

    settings = load_frontend_settings()

    assert settings.backend_api_base_url == "http://localhost:8080/api/v1"


def test_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("BACKEND_API_BASE_URL", "https://backend.example/api/v1")
    monkeypatch.setenv("USE_BACKEND_API", "true")
    monkeypatch.setenv("BACKEND_API_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("BACKEND_API_CONNECT_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("BACKEND_API_MAX_RETRIES", "2")
    monkeypatch.setenv("BACKEND_API_VERIFY_SSL", "false")
    monkeypatch.setenv("SHOW_BACKEND_STATUS_BADGE", "false")
    monkeypatch.setenv("SHOW_ADVANCED_BACKEND_STATUS", "true")
    monkeypatch.setenv("USE_BACKEND_API_AUTH", "true")
    monkeypatch.setenv("USE_BACKEND_API_ADMIN", "true")
    monkeypatch.setenv("USE_BACKEND_API_DATA_EXPLORER", "true")
    monkeypatch.setenv("USE_BACKEND_API_OPS", "true")

    settings = load_frontend_settings()

    assert settings.backend_api_base_url == "https://backend.example/api/v1"
    assert settings.use_backend_api is True
    assert settings.backend_api_timeout_seconds == 12
    assert settings.backend_api_connect_timeout_seconds == 3
    assert settings.backend_api_max_retries == 2
    assert settings.backend_api_verify_ssl is False
    assert settings.show_backend_status_badge is False
    assert settings.show_advanced_backend_status is True
    assert settings.page_api_flags["auth"] is True
    assert settings.page_api_flags["admin"] is True
    assert settings.page_api_flags["data_explorer"] is True
    assert settings.page_api_flags["ops"] is True
    assert is_backend_api_enabled("auth") is True
    assert is_backend_api_enabled("admin") is True
    assert is_backend_api_enabled("data_explorer") is True
    assert is_backend_api_enabled("ops") is True
