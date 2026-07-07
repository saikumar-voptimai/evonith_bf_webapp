"""Tests for frontend dataset API adapter."""

from __future__ import annotations

import pytest

from apps.frontend_streamlit.services.api_errors import BackendUnavailableError
from apps.frontend_streamlit.services.dataset_api import (
    get_dataset_job,
    get_dataset_job_download_url,
    list_datasets,
    preview_dataset,
    refresh_dataset,
)


class FakeClient:
    base_url = "http://localhost:8080/api/v1"

    def __init__(self):
        self.calls: list[tuple] = []

    def get(self, path, params=None):
        self.calls.append(("GET", path, params))
        return {"request_id": "id", "data": {"ok": True}, "meta": {"api_version": "v1"}}

    def post(self, path, json=None, params=None):
        self.calls.append(("POST", path, json, params))
        return {"request_id": "id", "data": {"job_id": "job"}, "meta": {"api_version": "v1"}}


def test_list_datasets_calls_datasets_endpoint():
    client = FakeClient()

    list_datasets(client)

    assert client.calls == [("GET", "/datasets", None)]


def test_preview_dataset_calls_preview_endpoint():
    client = FakeClient()

    preview_dataset("static_ml_dataset", limit=25, client=client)

    assert client.calls == [("GET", "/datasets/static_ml_dataset/preview", {"limit": 25})]


def test_refresh_dataset_calls_refresh_endpoint():
    client = FakeClient()

    refresh_dataset({"dataset_id": "static_ml_dataset"}, client)

    assert client.calls == [
        ("POST", "/datasets/refresh", {"dataset_id": "static_ml_dataset"}, None)
    ]


def test_get_dataset_job_calls_job_endpoint():
    client = FakeClient()

    get_dataset_job("job-1", client)

    assert client.calls == [("GET", "/datasets/jobs/job-1", None)]


def test_job_download_url_uses_base_url():
    assert (
        get_dataset_job_download_url("job-1", FakeClient())
        == "http://localhost:8080/api/v1/datasets/jobs/job-1/download"
    )


def test_backend_unavailable_propagates_cleanly():
    class UnavailableClient(FakeClient):
        def get(self, path, params=None):
            raise BackendUnavailableError("offline", request_id="req")

    with pytest.raises(BackendUnavailableError) as exc_info:
        list_datasets(UnavailableClient())

    assert exc_info.value.request_id == "req"
