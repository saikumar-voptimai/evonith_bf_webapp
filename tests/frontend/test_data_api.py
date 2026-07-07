"""Tests for frontend data API adapter."""

from __future__ import annotations

import pytest

from apps.frontend_streamlit.services.api_errors import BackendUnavailableError
from apps.frontend_streamlit.services.data_api import export_data, get_artifact_download_url, list_data_sources, preview_data


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls: list[tuple] = []

    def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        return {"request_id": "id", "data": [{"id": "online"}], "meta": {"api_version": "v1"}}

    def post(self, path, json=None, params=None):
        self.calls.append(("POST", path, json, params))
        return {"request_id": "id", "data": {"ok": True}, "meta": {"api_version": "v1"}}


def test_list_data_sources_calls_sources_endpoint():
    client = FakeClient()

    result = list_data_sources(client)

    assert result == [{"id": "online"}]
    assert client.calls == [("GET", "/data/sources", None)]


def test_preview_data_calls_preview_endpoint():
    client = FakeClient()

    preview_data({"source": "offline"}, client)

    assert client.calls == [("POST", "/data/preview", {"source": "offline"}, None)]


def test_export_data_calls_export_endpoint():
    client = FakeClient()

    export_data({"source": "offline"}, client=client)

    assert client.calls == [
        ("POST", "/data/export", {"query": {"source": "offline"}, "format": "csv"}, None)
    ]


def test_artifact_download_url_uses_base_url():
    assert (
        get_artifact_download_url("abc", FakeClient())
        == "http://localhost:8080/api/v1/data/artifacts/abc/download"
    )


def test_backend_unavailable_propagates_cleanly():
    class UnavailableClient(FakeClient):
        def get(self, path, params=None):
            raise BackendUnavailableError("offline", request_id="req")

    with pytest.raises(BackendUnavailableError) as exc_info:
        list_data_sources(UnavailableClient())

    assert exc_info.value.request_id == "req"
