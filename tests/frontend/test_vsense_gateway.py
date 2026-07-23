from __future__ import annotations

import pytest

from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError
from apps.frontend_streamlit.services.vsense_gateway import ApiVSenseGateway, DirectVSenseGateway, get_vsense_gateway


def test_vsense_gateway_defaults_to_direct(monkeypatch):
    monkeypatch.delenv("USE_BACKEND_API_VSENSE", raising=False)
    monkeypatch.delenv("USE_BACKEND_API_RECOMMENDATIONS", raising=False)

    assert isinstance(get_vsense_gateway(), DirectVSenseGateway)


def test_vsense_gateway_uses_api_only_with_auth_and_token(monkeypatch):
    monkeypatch.setenv("USE_BACKEND_API_VSENSE", "true")
    monkeypatch.delenv("USE_BACKEND_API_AUTH", raising=False)

    with pytest.raises(BackendApiHTTPError) as exc_info:
        get_vsense_gateway(access_token="token")
    assert exc_info.value.error_code == "AUTH_REQUIRED"

    monkeypatch.setenv("USE_BACKEND_API_AUTH", "true")
    with pytest.raises(BackendApiHTTPError):
        get_vsense_gateway(access_token=None)

    assert isinstance(get_vsense_gateway(access_token="token"), ApiVSenseGateway)
