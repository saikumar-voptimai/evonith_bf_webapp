"""Small client for backend-managed static dataset refresh endpoints."""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = "http://localhost:8080"


def _base_url() -> str:
    return (
        os.getenv("FURNACE_DATA_API_URL")
        or os.getenv("FURNACE_DATA_SERVICE_URL")
        or DEFAULT_API_BASE_URL
    ).rstrip("/")


def get_static_status(
    *,
    auto_enqueue: bool = True,
    rm_choice: str = "Full",
    triggered_by: str | None = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Fetch static dataset refresh status from the backend."""

    params: dict[str, Any] = {"auto_enqueue": auto_enqueue, "rm_choice": rm_choice}
    if triggered_by:
        params["triggered_by"] = triggered_by
    return _request("GET", "/api/v1/datasets/static/status", params=params, timeout=timeout)


def get_refresh_run(run_id: str, *, timeout: float = 5.0) -> dict[str, Any]:
    """Fetch backend refresh-run metadata."""

    return _request("GET", f"/api/v1/datasets/refresh-runs/{run_id}", timeout=timeout)


def _request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(method, f"{_base_url()}{path}", **kwargs)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {
            "state": "failed",
            "latest_version_id": None,
            "active_table": "ml_dataset.active_hourly",
            "confirmed_start": None,
            "confirmed_end": None,
            "last_refresh_at": None,
            "run_id": None,
            "message": "Dataset refresh status is unavailable.",
            "error_message": str(exc),
        }
