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


def test_feedback_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_FEEDBACK", raising=False)

    assert is_backend_api_enabled("feedback") is False


def test_feedback_flag_selects_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_FEEDBACK", "true")

    assert is_backend_api_enabled("feedback") is True


def test_phase7_compute_flags_default_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_MATERIAL_BALANCE", raising=False)
    monkeypatch.delenv("USE_BACKEND_API_RECOMMENDATIONS", raising=False)
    monkeypatch.delenv("USE_BACKEND_API_BLEND_OPTIMIZER", raising=False)

    assert is_backend_api_enabled("material_balance") is False
    assert is_backend_api_enabled("recommendations") is False
    assert is_backend_api_enabled("blend_optimizer") is False


def test_phase7_compute_flags_select_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_MATERIAL_BALANCE", "true")
    monkeypatch.setenv("USE_BACKEND_API_RECOMMENDATIONS", "true")
    monkeypatch.setenv("USE_BACKEND_API_BLEND_OPTIMIZER", "true")

    assert is_backend_api_enabled("material_balance") is True
    assert is_backend_api_enabled("recommendations") is True
    assert is_backend_api_enabled("blend_optimizer") is True


def test_phase8_copilot_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_COPILOT", raising=False)

    assert is_backend_api_enabled("copilot") is False


def test_phase8_copilot_flag_selects_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_COPILOT", "true")

    assert is_backend_api_enabled("copilot") is True


def test_phase9_furnacemind_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_FURNACEMIND", raising=False)

    assert is_backend_api_enabled("furnacemind") is False


def test_phase9_furnacemind_flag_selects_api_mode(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_FURNACEMIND", "true")

    assert is_backend_api_enabled("furnacemind") is True


def test_data_api_limits_load_from_env(monkeypatch):
    monkeypatch.setenv("DATA_API_MAX_PREVIEW_ROWS", "25")
    monkeypatch.setenv("DATA_API_MAX_JSON_ROWS", "100")

    settings = load_frontend_settings()

    assert settings.data_api_max_preview_rows == 25
    assert settings.data_api_max_json_rows == 100
