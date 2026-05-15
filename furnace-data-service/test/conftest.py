"""
Shared pytest fixtures.

All external I/O (InfluxDB, PostgreSQL) is mocked.
Tests run fully offline — no credentials required.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Set dummy env vars BEFORE any app module is imported
os.environ.setdefault("INFLUX_ONLINE_TOKEN", "test_online_token")
os.environ.setdefault("INFLUX_OFFLINE_TOKEN", "test_offline_token")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_process_df() -> pd.DataFrame:
    """Small process_params-like DataFrame with a proper time index."""
    idx = pd.date_range("2026-01-01 08:00", periods=4, freq="15min")
    return pd.DataFrame(
        {
            "hot_blast_vol_nm3h": [110000.0, 111000.0, 109500.0, 112000.0],
            "body_etaco": [42.1, 42.3, 41.9, 42.5],
            "body_raft": [2280.0, 2290.0, 2275.0, 2300.0],
            "production_per_hour": [75.0, 76.0, 74.5, 77.0],
        },
        index=idx,
    )


@pytest.fixture()
def sample_hm_df() -> pd.DataFrame:
    """Small hot-metal-like DataFrame."""
    idx = pd.date_range("2026-01-01 08:00", periods=3, freq="1h")
    return pd.DataFrame(
        {"CHEM_PCT_C": [4.5, 4.6, 4.4], "CHEM_PCT_SI": [0.3, 0.31, 0.29]},
        index=idx,
    )


@pytest.fixture()
def sample_ml_df() -> pd.DataFrame:
    """Minimal ML dataset DataFrame covering a few days."""
    idx = pd.date_range("2025-10-01", periods=72, freq="1h")
    return pd.DataFrame(
        {
            "PCI_KG/THM": [85.0] * 72,
            "COKE RATE KG/THM": [390.0] * 72,
            "FURNACETOPGASANALYSISCO2ETACO": [42.0] * 72,
            "PRODUCTIONTONNESPERHR": [75.0] * 72,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """
    Session-scoped test client.
    Static dir / results dir are pointed at temp paths via env vars.
    """
    # These are read at Settings() instantiation; must be set before first import
    os.environ.setdefault("RESULTS_DIR", "/tmp/furnace_api_test_results")
    os.environ.setdefault("STATIC_DIR", "/tmp/furnace_api_test_static")

    service_root_text = str(SERVICE_ROOT)
    if service_root_text in sys.path:
        sys.path.remove(service_root_text)
    sys.path.insert(0, service_root_text)

    loaded_app = sys.modules.get("app")
    loaded_path = str(getattr(loaded_app, "__file__", "")) if loaded_app else ""
    if loaded_path.endswith("src\\app.py") or loaded_path.endswith("src/app.py"):
        del sys.modules["app"]

    from app.main import app
    with TestClient(app) as c:
        yield c
