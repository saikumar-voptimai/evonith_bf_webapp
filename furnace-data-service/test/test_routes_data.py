"""
Tests for /data/online/* and /data/offline/* routes.

InfluxDB calls are mocked at the core-function level so no credentials needed.
"""
from unittest.mock import patch

import pandas as pd
import pytest


ONLINE_MEASUREMENTS = [
    "process_params",
    "temperature_profile",
    "heatload_delta_t",
    "cooling_water",
    "delta_t",
    "miscellaneous",
]


# ---------------------------------------------------------------------------
# GET /data/online/measurements
# ---------------------------------------------------------------------------

class TestMeasurementsList:
    def test_returns_all_six_measurements(self, client):
        resp = client.get("/data/online/measurements")
        assert resp.status_code == 200
        body = resp.json()
        assert "measurements" in body
        for m in ONLINE_MEASUREMENTS:
            assert m in body["measurements"]

    def test_each_measurement_has_field_list(self, client):
        resp = client.get("/data/online/measurements")
        for m, fields in resp.json()["measurements"].items():
            assert isinstance(fields, list)


# ---------------------------------------------------------------------------
# GET /data/offline/report-types
# ---------------------------------------------------------------------------

class TestOfflineReportTypes:
    def test_returns_known_types(self, client):
        resp = client.get("/data/offline/report-types")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("HM_SLAG", "CHARGE", "RM_COMPOSITION", "DPR"):
            assert key in body

    def test_values_are_measurement_strings(self, client):
        resp = client.get("/data/offline/report-types")
        for v in resp.json().values():
            assert isinstance(v, str)
            assert len(v) > 0

    def test_neon_tables_endpoint_lists_charge_data(self, client):
        resp = client.get("/data/offline/neon-tables")
        assert resp.status_code == 200
        body = resp.json()
        assert body["report_map"]["CHARGE"] == "charge_data"
        assert "charge_data" in body["tables"]
        assert "date_time" in body["tables"]["charge_data"]


# ---------------------------------------------------------------------------
# POST /data/online/fetch — input validation
# ---------------------------------------------------------------------------

class TestOnlineFetchValidation:
    def test_missing_time_params_returns_400(self, client):
        resp = client.post("/data/online/fetch", json={
            "measurements": ["process_params"],
            # neither preset nor start_time/end_time
        })
        assert resp.status_code == 400
        assert "preset" in resp.json()["detail"].lower() or "start_time" in resp.json()["detail"].lower()

    def test_unknown_measurement_returns_400(self, client):
        resp = client.post("/data/online/fetch", json={
            "measurements": ["nonexistent_measurement"],
            "preset": "last 1 hour",
        })
        assert resp.status_code == 400
        assert "unknown" in resp.json()["detail"].lower()

    def test_invalid_query_type_returns_422(self, client):
        resp = client.post("/data/online/fetch", json={
            "measurements": ["process_params"],
            "preset": "last 1 hour",
            "query_type": "invalid_type",
        })
        assert resp.status_code == 422

    def test_empty_measurements_list_returns_422(self, client):
        resp = client.post("/data/online/fetch", json={
            "measurements": [],
            "preset": "last 1 hour",
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /data/online/fetch — successful fetch (mocked)
# ---------------------------------------------------------------------------

class TestOnlineFetchSuccess:
    def test_json_response_shape(self, client, sample_process_df):
        with patch("app.routes.data.fetch_online", return_value=sample_process_df):
            resp = client.post("/data/online/fetch", json={
                "measurements": ["process_params"],
                "preset": "last 1 hour",
                "query_type": "windowed-average",
                "window": "15m",
                "format": "json",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert "meta" in body
        assert "columns" in body
        assert "data" in body
        assert body["meta"]["rows"] == len(sample_process_df)
        assert body["meta"]["columns"] == len(sample_process_df.columns)
        assert len(body["data"]) == len(sample_process_df)

    def test_csv_response_content_type(self, client, sample_process_df):
        with patch("app.routes.data.fetch_online", return_value=sample_process_df):
            resp = client.post("/data/online/fetch", json={
                "measurements": ["process_params"],
                "preset": "last 1 hour",
                "format": "csv",
            })
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "hot_blast_vol_nm3h" in resp.text

    def test_multiple_measurements_in_meta(self, client, sample_process_df):
        with patch("app.routes.data.fetch_online", return_value=sample_process_df):
            resp = client.post("/data/online/fetch", json={
                "measurements": ["process_params", "heatload_delta_t"],
                "preset": "last 1 hour",
                "format": "json",
            })
        assert resp.status_code == 200
        meta = resp.json()["meta"]
        assert "process_params" in meta["measurements"]
        assert "heatload_delta_t" in meta["measurements"]

    def test_empty_dataframe_returns_204(self, client):
        with patch("app.routes.data.fetch_online", return_value=pd.DataFrame()):
            resp = client.post("/data/online/fetch", json={
                "measurements": ["process_params"],
                "preset": "last 1 hour",
            })
        assert resp.status_code == 204

    def test_start_end_without_preset(self, client, sample_process_df):
        with patch("app.routes.data.fetch_online", return_value=sample_process_df):
            resp = client.post("/data/online/fetch", json={
                "measurements": ["process_params"],
                "start_time": "2026-01-01T00:00:00",
                "end_time": "2026-01-02T00:00:00",
                "query_type": "ts",
            })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /data/offline/fetch — validation + success
# ---------------------------------------------------------------------------

class TestOfflineFetch:
    def test_missing_time_params_returns_400(self, client):
        resp = client.post("/data/offline/fetch", json={
            "report_type": "HM_SLAG",
            # no preset, no start/end
        })
        assert resp.status_code == 400

    def test_invalid_report_type_returns_422(self, client):
        resp = client.post("/data/offline/fetch", json={
            "report_type": "INVALID",
            "preset": "last 1 month",
        })
        assert resp.status_code == 422

    def test_successful_fetch_json(self, client, sample_hm_df):
        with patch("app.routes.data.fetch_offline", return_value=sample_hm_df):
            resp = client.post("/data/offline/fetch", json={
                "report_type": "HM_SLAG",
                "preset": "last 3 days",
                "format": "json",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["meta"]["report_type"] == "HM_SLAG"
        assert body["meta"]["rows"] == len(sample_hm_df)

    def test_successful_fetch_csv(self, client, sample_hm_df):
        with patch("app.routes.data.fetch_offline", return_value=sample_hm_df):
            resp = client.post("/data/offline/fetch", json={
                "report_type": "HM_SLAG",
                "preset": "last 3 days",
                "format": "csv",
            })
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]

    def test_empty_result_returns_204(self, client):
        with patch("app.routes.data.fetch_offline", return_value=pd.DataFrame()):
            resp = client.post("/data/offline/fetch", json={
                "report_type": "DPR",
                "preset": "last 1 day",
            })
        assert resp.status_code == 204

    def test_neon_source_fetches_charge_windowed_average(self, client, sample_hm_df):
        with patch("app.routes.data.fetch_neon_offline", return_value=sample_hm_df) as mocked:
            resp = client.post("/data/offline/fetch", json={
                "source": "neon_db",
                "report_type": "CHARGE",
                "preset": "last 3 days",
                "query_type": "windowed-average",
                "window": "1 hour",
                "format": "json",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["meta"]["source"] == "neon_db"
        assert body["meta"]["table_name"] == "charge_data"
        assert body["meta"]["query_type"] == "windowed-average"
        assert body["meta"]["window"] == "1 hour"
        mocked.assert_called_once()

    def test_neon_table_override_fetches_raw_table(self, client, sample_hm_df):
        with patch("app.routes.data.fetch_neon_offline", return_value=sample_hm_df) as mocked:
            resp = client.post("/data/offline/fetch", json={
                "source": "neon_db",
                "report_type": "RM_COMPOSITION",
                "table_name": "sinter_chemistry",
                "preset": "last 3 days",
                "query_type": "ts",
            })
        assert resp.status_code == 200
        assert resp.json()["meta"]["table_name"] == "sinter_chemistry"
        assert mocked.call_args.kwargs["query_type"] == "ts"
