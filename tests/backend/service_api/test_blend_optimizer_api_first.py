"""API-first Blend Optimizer contract tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.repositories.blend_optimizer_repository import BlendOptimizerRepository
from apps.backend_api.app.services.blend_optimizer_api_service import BlendOptimizerApiService
from furnace_data.bmo.utils.types import BlendEvaluation, ModelPrediction


def _blend() -> BlendEvaluation:
    return BlendEvaluation(
        quantities_mt={"ore_1": 100.0, "ore_2": 80.0},
        shares_pct={"ore_1": 55.556, "ore_2": 44.444},
        total_qty_mt=180.0,
        ore_cost_total_rs=180000.0,
        ore_cost_per_thm_rs=1800.0,
        fuel_cost_per_thm_rs=100.0,
        objective_rs_per_thm=1900.0,
        fe_t_pct=60.0,
        effective_fe_pct=60.0,
        fe_production_mt=100.0,
        slag_pct=5.0,
        slag_mt=10.0,
        feasible=True,
        violations=[],
        slag_rate_kg_per_thm=100.0,
        slag_basicity=1.1,
        slag_t_basicity=1.3,
        diagnostics={"model_prediction": ModelPrediction(value=100.0, model_loaded=False, scaler_loaded=False, used_fallback=True)},
    )


def _context_snapshot() -> dict:
    return {
        "eligible_materials": [
            {
                "ore_id": "ore_1",
                "display_name": "Ore 1",
                "stock_mt": 10000.0,
                "price_rs_per_mt": 1000.0,
                "min_share_pct": 0.0,
                "max_share_pct": 100.0,
                "chemistry": {"fe_t_pct": 60.0, "sio2_pct": 5.0, "cao_pct": 5.5},
            },
            {
                "ore_id": "ore_2",
                "display_name": "Ore 2",
                "stock_mt": 10000.0,
                "price_rs_per_mt": 1200.0,
                "min_share_pct": 0.0,
                "max_share_pct": 100.0,
                "chemistry": {"fe_t_pct": 62.0, "sio2_pct": 4.0, "cao_pct": 4.4},
            },
        ],
        "hot_metal_chemistry": {"hm_fe_pct_for_target": 100.0},
        "basicity_defaults": {},
        "fuel_ash_inputs": [],
        "flux_inputs": [],
        "dust_inputs": [],
        "slag_balance": {},
    }


def test_total_cost_run_uses_one_lp_call_and_precomputed_de_seed(tmp_path, monkeypatch):
    settings = BackendSettings(
        backend_env="test",
        blend_optimizer_database_url=f"sqlite:///{(tmp_path / 'bmo.db').as_posix()}",
        blend_optimizer_max_iterations=10,
    )
    repository = BlendOptimizerRepository(settings.blend_optimizer_database_url)
    context = repository.create_context(
        {
            "owner_id": "u1",
            "version": "ctx-v1",
            "fingerprint": "ctx-v1",
            "status": "available",
            "request": {},
            "snapshot": _context_snapshot(),
            "diagnostics": {},
            "warnings": [],
        }
    )
    service = BlendOptimizerApiService(settings=settings, repository=repository)
    calls = {"lp": 0, "de": 0}
    blend = _blend()

    def fake_lp(*args, **kwargs):
        calls["lp"] += 1
        return blend, []

    def fake_enrich(**kwargs):
        return blend

    def fake_de(*args, **kwargs):
        calls["de"] += 1
        assert kwargs["precomputed_lp_blend"] is blend
        return blend, []

    monkeypatch.setattr("apps.backend_api.app.services.blend_optimizer_api_service.run_lp_baseline", fake_lp)
    monkeypatch.setattr("apps.backend_api.app.services.blend_optimizer_api_service.evaluate_blend_with_fuel_prediction", fake_enrich)
    monkeypatch.setattr("apps.backend_api.app.services.blend_optimizer_api_service.run_nonlinear_optimizer", fake_de)
    monkeypatch.setattr(service, "fuel_context", lambda provider, cfg: (object(), {}, None, {"model_loaded": False}, []))
    monkeypatch.setattr(service, "predict_si", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "safe_si_status", lambda cfg: {"model_loaded": False})

    created = service.create_run(
        {
            "mode": "total_cost",
            "context_id": context.id,
            "expected_context_version": context.version,
            "scenario": {"targets": {"target_hot_metal_mt": 100.0, "max_slag_mt": 1000.0}},
            "options": {"algorithm_version": "bmo_total_cost_de_legacy_v1", "iteration_budget_id": "quick"},
        },
        {"id": "u1"},
        idempotency_key="total-cost-1",
    )
    processed = service.process_run(created["run_id"], {"id": "u1"})

    assert processed["status"] == "completed"
    assert calls == {"lp": 1, "de": 1}
    assert processed["result"]["advisory_only"] is True
    assert processed["result"]["operator_review_required"] is True
    assert [event["event_type"] for event in service.run_events(created["run_id"], {"id": "u1"})] == [
        "run_queued",
        "run_started",
        "lp_started",
        "lp_completed",
        "de_started",
        "de_completed",
        "run_completed",
    ]


def test_catalog_route_and_run_openapi(app_factory, monkeypatch, tmp_path):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("EVONITH_COMPUTE_REQUIRE_AUTH", "false")
    with TestClient(app_factory(), raise_server_exceptions=False) as client:
        catalog = client.get("/api/v1/blend-optimizer/catalog")
        schema = client.get("/openapi.json").json()

    assert catalog.status_code == 200
    assert catalog.json()["data"]["advisory_only"] is True
    assert "/api/v1/blend-optimizer/runs" in schema["paths"]
    assert schema["paths"]["/api/v1/blend-optimizer/runs"]["post"]["operationId"] == "create_blend_optimizer_run"
