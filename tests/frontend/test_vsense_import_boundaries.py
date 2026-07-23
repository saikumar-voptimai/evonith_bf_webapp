from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_vsense_api_adapter_does_not_import_backend_or_heavy_runtime_clients():
    text = (
        REPO_ROOT / "apps" / "frontend_streamlit" / "services" / "vsense_api.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "apps.backend_api",
        "joblib",
        "scipy",
        "influx",
        "dataframesprocessor",
        "dataset_refresher",
        "openai",
        "psycopg",
        "qdrant",
    )
    assert [item for item in forbidden if item in text] == []
