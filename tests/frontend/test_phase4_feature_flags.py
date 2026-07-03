"""Tests for Phase 4 API migration feature flags."""

from config.frontend_settings import is_backend_api_enabled, load_frontend_settings


def test_data_explorer_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_DATA_EXPLORER", raising=False)

    assert is_backend_api_enabled("data_explorer") is False


def test_data_explorer_flag_selects_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_DATA_EXPLORER", "true")

    assert is_backend_api_enabled("data_explorer") is True


def test_datasets_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_DATASETS", raising=False)

    assert is_backend_api_enabled("datasets") is False


def test_datasets_flag_selects_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_DATASETS", "true")

    assert is_backend_api_enabled("datasets") is True


def test_data_api_limits_load_from_env(monkeypatch):
    monkeypatch.setenv("DATA_API_MAX_PREVIEW_ROWS", "25")
    monkeypatch.setenv("DATA_API_MAX_JSON_ROWS", "100")

    settings = load_frontend_settings()

    assert settings.data_api_max_preview_rows == 25
    assert settings.data_api_max_json_rows == 100
