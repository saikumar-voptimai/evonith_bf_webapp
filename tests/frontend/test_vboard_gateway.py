from __future__ import annotations

import pytest

from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.vboard_gateway import (
    ApiVBoardGateway,
    DirectVBoardGateway,
    get_vboard_gateway,
)


def test_vboard_feature_flag_defaults_to_direct_mode(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_VBOARD", raising=False)

    assert isinstance(get_vboard_gateway(), DirectVBoardGateway)


def test_vboard_api_mode_requires_auth_flag(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_VBOARD", "true")
    monkeypatch.delenv("USE_BACKEND_API_AUTH", raising=False)

    with pytest.raises(BackendApiHTTPError) as exc_info:
        get_vboard_gateway(access_token="token")

    assert exc_info.value.error_code == "AUTH_REQUIRED"


def test_vboard_api_mode_requires_token_and_never_falls_back(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_VBOARD", "true")
    monkeypatch.setenv("USE_BACKEND_API_AUTH", "true")

    with pytest.raises(BackendApiHTTPError):
        get_vboard_gateway(access_token="")

    assert isinstance(get_vboard_gateway(access_token="token"), ApiVBoardGateway)
