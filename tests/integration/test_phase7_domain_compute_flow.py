"""Integration smoke for Phase 7 domain compute APIs."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ROOT = REPO_ROOT / "furnace-data-service"
for path in (str(SERVICE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

loaded_app = sys.modules.get("app")
loaded_path = str(getattr(loaded_app, "__file__", "")) if loaded_app else ""
if loaded_path.endswith("src\\app.py") or loaded_path.endswith("src/app.py"):
    del sys.modules["app"]

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.main import create_app


def test_phase7_domain_compute_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = BackendSettings(
        api_prefix="/api/v1",
        backend_env="test",
        compute_require_auth=False,
        feedback_require_auth=False,
    )
    app = create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        mb = client.post(
            "/api/v1/material-balance/run",
            json={
                "source": "input_data",
                "input_data": {"inputs": {"Fe": 10}, "outputs": {"Fe": 9}},
            },
        )
        rec = client.post(
            "/api/v1/recommendations/run",
            json={"input_data": {"signals": {"PCI_KG/THM": 5}}, "max_items": 1},
        )
        context = client.get("/api/v1/blend-optimizer/context")
        optimize = client.post(
            "/api/v1/blend-optimizer/optimize",
            json={
                "materials": [
                    {
                        "material_id": "ore_a",
                        "min_percent": 0,
                        "max_percent": 80,
                        "properties": {"fe_t_pct": 58},
                        "cost": 1000,
                    },
                    {
                        "material_id": "ore_b",
                        "min_percent": 0,
                        "max_percent": 80,
                        "properties": {"fe_t_pct": 62},
                        "cost": 1200,
                    },
                ],
                "constraints": {"target_total_qty_mt": 100},
            },
        )
        feedback = client.get("/api/v1/feedback/config")

    assert mb.status_code == 200
    assert mb.json()["data"]["tables"]["closure"]["returned_rows"] == 1
    assert rec.status_code == 200
    assert len(rec.json()["data"]["items"]) == 1
    assert context.status_code == 200
    assert optimize.status_code == 200
    assert optimize.json()["data"]["best_candidate"]
    assert feedback.status_code == 200
    assert (tmp_path / "runtime").exists()
