"""
Tests for /dataset/* routes.

Focuses on input validation and state-dependent responses
(cache-info empty, static 404, bad date order).
No real InfluxDB calls.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# GET /dataset/cache-info
# ---------------------------------------------------------------------------

class TestCacheInfo:
    def test_no_cache_returns_no_cache_status(self, client, tmp_path):
        # Point static_dir at an empty tmp dir with no cache_meta.json
        with patch("app.routes.dataset.settings") as mock_settings:
            mock_settings.static_dir = tmp_path
            mock_settings.offline_lag_days = 3
            mock_settings.static_max_versions = 3
            mock_settings.legacy_csv_path = ""

            resp = client.get("/dataset/cache-info")

        assert resp.status_code == 200
        assert resp.json()["status"] == "no_cache"

    def test_cache_present_returns_metadata(self, client, tmp_path):
        from furnace_data.dataset.static import CacheMeta, _save_meta
        meta = CacheMeta(
            rm_choice="charge",
            data_start="2023-01-01",
            confirmed_end="2026-03-28",
            raw_end="2026-03-31",
            last_updated="2026-04-01T09:00:00",
            offline_lag_days=3,
            rows=19000,
            columns=120,
            csv_file="ml_dataset_20260401_090000.csv",
        )
        _save_meta(tmp_path, meta)

        with patch("app.routes.dataset.settings") as mock_settings:
            mock_settings.static_dir = tmp_path
            mock_settings.offline_lag_days = 3
            mock_settings.static_max_versions = 3
            mock_settings.legacy_csv_path = ""

            resp = client.get("/dataset/cache-info")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["confirmed_end"] == "2026-03-28"
        assert body["raw_end"] == "2026-03-31"
        assert body["rows"] == 19000
        assert body["offline_lag_days"] == 3


# ---------------------------------------------------------------------------
# GET /dataset/static
# ---------------------------------------------------------------------------

class TestStaticDownload:
    def test_no_file_returns_404(self, client, tmp_path):
        with patch("app.routes.dataset.settings") as mock_settings:
            mock_settings.static_dir = tmp_path
            mock_settings.offline_lag_days = 3
            mock_settings.static_max_versions = 3
            mock_settings.legacy_csv_path = ""

            resp = client.get("/dataset/static")
        assert resp.status_code == 404

    def test_file_present_returns_csv(self, client, tmp_path, sample_ml_df):
        from furnace_data.dataset.static import CacheMeta, _save_meta
        csv_name = "ml_dataset_20260401_090000.csv"
        sample_ml_df.to_csv(tmp_path / csv_name)
        meta = CacheMeta(csv_file=csv_name, rows=len(sample_ml_df))
        _save_meta(tmp_path, meta)

        with patch("app.routes.dataset.settings") as mock_settings:
            mock_settings.static_dir = tmp_path
            mock_settings.offline_lag_days = 3
            mock_settings.static_max_versions = 3
            mock_settings.legacy_csv_path = ""

            resp = client.get("/dataset/static")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /dataset/fetch — input validation
# ---------------------------------------------------------------------------

class TestFetchDataset:
    def test_start_after_end_returns_400(self, client):
        resp = client.post("/dataset/fetch", json={
            "start_date": "2026-03-31",
            "end_date": "2026-03-01",
            "rm_choice": "charge",
        })
        assert resp.status_code == 400
        assert "start_date" in resp.json()["detail"].lower()

    def test_valid_request_returns_task_id(self, client):
        with patch("app.routes.dataset.task_manager.run_in_background"):
            resp = client.post("/dataset/fetch", json={
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
                "rm_choice": "charge",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "pending"

    def test_invalid_rm_choice_returns_422(self, client):
        resp = client.post("/dataset/fetch", json={
            "start_date": "2026-01-01",
            "end_date": "2026-01-07",
            "rm_choice": "invalid_choice",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /dataset/status/{task_id}
# ---------------------------------------------------------------------------

class TestTaskStatus:
    def test_unknown_task_id_returns_404(self, client):
        resp = client.get("/dataset/status/nonexistent_task_id")
        assert resp.status_code == 404

    def test_known_task_returns_status(self, client):
        with patch("app.routes.dataset.task_manager.run_in_background"):
            create_resp = client.post("/dataset/fetch", json={
                "start_date": "2026-01-01",
                "end_date": "2026-01-03",
                "rm_choice": "dpr",
            })
        task_id = create_resp.json()["task_id"]

        status_resp = client.get(f"/dataset/status/{task_id}")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["task_id"] == task_id
        assert body["status"] in ("pending", "running", "completed", "failed")


# ---------------------------------------------------------------------------
# GET /dataset/download/{task_id}
# ---------------------------------------------------------------------------

class TestDownload:
    def test_unknown_task_returns_404(self, client):
        resp = client.get("/dataset/download/no_such_id")
        assert resp.status_code == 404

    def test_pending_task_returns_409(self, client):
        with patch("app.routes.dataset.task_manager.run_in_background"):
            create_resp = client.post("/dataset/fetch", json={
                "start_date": "2026-01-01",
                "end_date": "2026-01-03",
                "rm_choice": "charge",
            })
        task_id = create_resp.json()["task_id"]

        # Task is still pending (background was patched to do nothing)
        resp = client.get(f"/dataset/download/{task_id}")
        assert resp.status_code == 409
