
"""API-first Blend Optimizer orchestration service."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.blend_optimizer_repository import (
    BlendOptimizerContextRecord,
    BlendOptimizerRepository,
    BlendOptimizerRunEventRecord,
    BlendOptimizerRunRecord,
    idempotency_hash,
)
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.material_balance_service import _warning
from furnace_data.bmo.data import EvonithBmoContextProvider
from furnace_data.bmo.utils import (
    DustInput,
    FluxInput,
    FuelAshInput,
    FuelUnitCostModelService,
    OreChemistry,
    OreInput,
    SlagBalanceSettings,
    evaluate_blend_with_fuel_prediction,
    run_lp_baseline,
    run_nonlinear_optimizer,
)
from furnace_data.bmo.utils.calculations import evaluate_blend
from furnace_data.bmo.utils.si_prediction import SiPredictionService
from furnace_data.config import get_config_path

TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "timed_out"}
DISALLOWED_OPTION_KEYS = {
    "model_path", "scaler_path", "feature_vector", "features", "raw_sql",
    "sql", "flux", "filesystem_path", "random_seed", "seed", "maxiter",
    "popsize", "penalty_weights", "callback",
}
DEFAULT_PREFERENCES: dict[str, Any] = {
    "target_hot_metal_mt": 2350.0,
    "max_slag_mt": 750.0,
    "basicity_min": 1.05,
    "basicity_max": 1.12,
    "t_basicity_min": 1.25,
    "t_basicity_max": 1.40,
    "ore_selection": [],
    "ore_share_bounds": {},
    "ore_price_overrides": {},
    "chemistry_overrides": {},
    "display_options": {},
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def user_id(current_user: dict[str, Any] | None) -> str:
    if not current_user:
        return "anonymous"
    return str(current_user.get("id") or current_user.get("username") or "anonymous")


def is_admin(current_user: dict[str, Any] | None) -> bool:
    return str((current_user or {}).get("role") or "").lower() == "admin"


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            clean_key = str(key)
            if any(secret in clean_key.lower() for secret in ("password", "secret", "token", "connection")):
                safe[clean_key] = "[REDACTED]"
            elif clean_key.lower().endswith("path"):
                safe[clean_key] = Path(str(item)).name if item else None
            else:
                safe[clean_key] = json_safe(item)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            return str(value)
    return value


def fingerprint(payload: Any) -> str:
    encoded = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_bmo_config() -> dict[str, Any]:
    path = get_config_path("setting_bmo.yml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return dict(data.get("bmo") or {})


def chemistry_from_payload(payload: dict[str, Any]) -> OreChemistry:
    return OreChemistry(
        fe_t_pct=float(payload.get("fe_t_pct") or 0.0),
        moisture_pct=float(payload.get("moisture_pct") or 0.0),
        feo_pct=float(payload.get("feo_pct") or 0.0),
        sio2_pct=float(payload.get("sio2_pct") or 0.0),
        al2o3_pct=float(payload.get("al2o3_pct") or 0.0),
        cao_pct=float(payload.get("cao_pct") or 0.0),
        mgo_pct=float(payload.get("mgo_pct") or 0.0),
        mno_pct=float(payload.get("mno_pct") or 0.0),
        tio2_pct=float(payload.get("tio2_pct") or 0.0),
        p_pct=float(payload.get("p_pct") or 0.0),
        s_pct=float(payload.get("s_pct") or 0.0),
        zn_pct=float(payload.get("zn_pct") or 0.0),
        na2o_pct=float(payload.get("na2o_pct") or 0.0),
        k2o_pct=float(payload.get("k2o_pct") or 0.0),
    )


def ore_from_payload(payload: dict[str, Any], fallback: dict[str, Any] | None = None) -> OreInput:
    source = {**(fallback or {}), **payload}
    chemistry = source.get("chemistry") or {}
    return OreInput(
        ore_id=str(source.get("ore_id") or source.get("material_id") or ""),
        display_name=str(source.get("display_name") or source.get("name") or source.get("ore_id") or ""),
        stock_mt=float(source.get("stock_mt") or 0.0),
        price_rs_per_mt=float(source.get("price_rs_per_mt") or source.get("price") or 0.0),
        min_share_pct=float(source.get("min_share_pct") if source.get("min_share_pct") is not None else source.get("min_percent") or 0.0),
        max_share_pct=float(source.get("max_share_pct") if source.get("max_share_pct") is not None else source.get("max_percent") or 100.0),
        chemistry=chemistry_from_payload(chemistry),
        metadata=dict(source.get("metadata") or {}),
    )


def dataclass_from_payload(cls: Any, payload: dict[str, Any]) -> Any:
    return cls(**{key: value for key, value in (payload or {}).items() if key in cls.__dataclass_fields__})


class BlendOptimizerApiService:
    """Durable API-first BMO contexts, preferences, runs and events."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        repository: BlendOptimizerRepository | None = None,
        artifact_service: ComputeArtifactService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = getattr(self.settings, "blend_optimizer_database_url", "") or None
        self.repository = repository or BlendOptimizerRepository(database_url)
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)

    def ensure_storage(self) -> None:
        self.repository.ensure_schema()

    def catalog(self) -> dict[str, Any]:
        fuel_status, si_status = self.model_statuses()
        return {
            "catalog_version": "bmo_catalog_v1",
            "display_timezone": getattr(self.settings, "bmo_default_timezone", "Asia/Kolkata"),
            "advisory_only": True,
            "operator_review_required": True,
            "optimization_modes": [
                {"id": "lp_baseline", "label": "LP Baseline", "algorithm_versions": ["bmo_lp_legacy_v1"]},
                {"id": "total_cost", "label": "Total Cost Optimizer", "algorithm_versions": ["bmo_total_cost_de_legacy_v1"]},
            ],
            "targets": ["target_hot_metal_mt", "target_fe_mt", "max_slag_mt", "basicity_min", "basicity_max", "t_basicity_min", "t_basicity_max"],
            "chemistry_fields": ["moisture_pct", "fe_t_pct", "feo_pct", "sio2_pct", "al2o3_pct", "cao_pct", "mgo_pct", "mno_pct", "tio2_pct", "p_pct", "s_pct", "zn_pct", "na2o_pct", "k2o_pct"],
            "material_types": ["ore", "fuel_ash", "flux", "dust"],
            "units": {"stock": "MT", "price": "Rs/MT", "share": "%", "fuel_cost": "Rs/THM", "slag_rate": "kg/THM"},
            "precision": {"quantity_mt": 3, "share_pct": 3, "cost_rs": 2},
            "iteration_budgets": [
                {"id": "quick", "label": "Quick", "maxiter": 10, "popsize": 5},
                {"id": "standard", "label": "Standard", "maxiter": min(80, self.settings.blend_optimizer_max_iterations), "popsize": 10},
            ],
            "capabilities": {"contexts": True, "preferences": True, "runs": True, "events": True, "manual_evaluations": True, "artifacts": True},
            "limits": {"max_selected_ores": getattr(self.settings, "bmo_max_selected_ores", 20), "max_concurrent_runs": getattr(self.settings, "bmo_max_concurrent_runs", 1), "run_timeout_seconds": self.settings.blend_optimizer_timeout_seconds},
            "algorithm_versions": {"lp_baseline": "bmo_lp_legacy_v1", "total_cost": "bmo_total_cost_de_legacy_v1"},
            "model_readiness": {"fuel_unit_cost": fuel_status, "hot_metal_si": si_status},
        }

    def create_context(self, payload: dict[str, Any], current_user: dict[str, Any] | None, *, idempotency_key: str | None) -> dict[str, Any]:
        self.require_idempotency(idempotency_key)
        owner_id = user_id(current_user)
        request_payload = json_safe(payload or {})
        request_fp = fingerprint(request_payload)
        key_hash = idempotency_hash(idempotency_key)
        existing = self.repository.find_context_by_idempotency(owner_id=owner_id, key_hash=key_hash or "") if key_hash else None
        if existing:
            if existing.request_fingerprint != request_fp:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different Blend Optimizer context request.", 409)
            return self.context_response(existing)
        source_refresh = str(request_payload.get("source_refresh") or "use_cached")
        if source_refresh not in {"use_cached", "refresh"}:
            raise ApiError("BMO_INVALID_CONTEXT_REQUEST", "source_refresh must be use_cached or refresh.", 422)
        provider = EvonithBmoContextProvider()
        mode = str(request_payload.get("chemistry_mode") or "latest")
        window_days = int(request_payload.get("chemistry_window_days") or provider.settings.get("chemistry_window_days", 30) or 30)
        ores, diagnostics = provider.build_ore_inputs(mode=mode, window_days=window_days)
        if not ores:
            raise ApiError("BMO_NO_ELIGIBLE_MATERIALS", "No eligible Blend Optimizer materials are available.", 503)
        if any(ore.ore_id.lower() in {"ore_a", "ore_b"} for ore in ores):
            raise ApiError("BMO_SOURCE_DATA_UNAVAILABLE", "Operational BMO context cannot use demonstration materials.", 503)
        bmo_cfg = load_bmo_config()
        hm_snapshot = provider.get_hm_slag_snapshot(mode=mode, window_days=window_days)
        snapshot = {
            "catalog_version": "bmo_catalog_v1",
            "source_refresh": source_refresh,
            "as_of_utc": utc_now().isoformat(),
            "source_provenance": diagnostics,
            "eligible_materials": [json_safe(ore) for ore in ores],
            "active_pellet_ids": list(provider.get_recent_active_pellet_ids() or []),
            "hot_metal_chemistry": json_safe(hm_snapshot),
            "fuel_ash_inputs": list(bmo_cfg.get("fuel_ash_inputs") or []),
            "flux_inputs": list(bmo_cfg.get("flux_inputs") or []),
            "dust_inputs": list(bmo_cfg.get("dust_inputs") or []),
            "slag_balance": dict(bmo_cfg.get("slag_balance") or {}),
            "recent_fuel_rates": {},
            "basicity_defaults": dict(bmo_cfg.get("target") or {}),
            "dataset": {"status": "not_refreshed_by_context", "version": None},
            "model_readiness": self.catalog()["model_readiness"],
            "recent_manual_blend": {},
        }
        version = f"bmoctx_{fingerprint(snapshot)[:24]}"
        snapshot["context_version"] = version
        expires_at = (utc_now() + timedelta(seconds=max(60, int(getattr(self.settings, "bmo_context_ttl_seconds", 1800))))).isoformat()
        record = self.repository.create_context({
            "owner_id": owner_id,
            "version": version,
            "fingerprint": version,
            "status": "available",
            "request": request_payload,
            "snapshot": snapshot,
            "diagnostics": provider.get_data_diagnostics(),
            "warnings": [_warning("BMO_CONTEXT_SOURCE_WARNING", str(item)) for item in diagnostics.get("warnings", [])],
            "idempotency_key_hash": key_hash,
            "request_fingerprint": request_fp,
            "expires_at": expires_at,
        })
        return self.context_response(record)

    def get_context(self, context_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        return self.context_response(self.context_or_404(context_id, current_user))

    def context_diagnostics(self, context_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self.context_or_404(context_id, current_user)
        return {"context_id": record.id, "context_version": record.version, "diagnostics": record.diagnostics, "warnings": record.warnings}

    def get_preferences(self, current_user: dict[str, Any] | None) -> dict[str, Any]:
        owner_id = user_id(current_user)
        record = self.repository.get_preferences(owner_id)
        if record is None:
            return {"owner_id": owner_id, "version": 0, "preferences": dict(DEFAULT_PREFERENCES), "updated_at": None}
        return {"owner_id": record.owner_id, "version": record.version, "preferences": record.preferences, "updated_at": record.updated_at}

    def update_preferences(self, payload: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        owner_id = user_id(current_user)
        expected_version = payload.get("expected_version")
        preferences = {**DEFAULT_PREFERENCES, **dict(payload.get("preferences") or {})}
        try:
            record = self.repository.upsert_preferences(
                owner_id=owner_id,
                preferences=json_safe(preferences),
                expected_version=int(expected_version) if expected_version is not None else None,
            )
        except ValueError as exc:
            raise ApiError("BMO_PREFERENCE_VERSION_CONFLICT", "Preference version is stale.", 409) from exc
        return {"owner_id": record.owner_id, "version": record.version, "preferences": record.preferences, "updated_at": record.updated_at}

    def create_run(self, payload: dict[str, Any], current_user: dict[str, Any] | None, *, idempotency_key: str | None) -> dict[str, Any]:
        self.require_idempotency(idempotency_key)
        request_payload = json_safe(payload or {})
        self.reject_solver_internals(request_payload.get("options") or {})
        owner_id = user_id(current_user)
        mode = str(request_payload.get("mode") or "")
        if mode not in {"lp_baseline", "total_cost"}:
            raise ApiError("BMO_INVALID_RUN_MODE", "Run mode must be lp_baseline or total_cost.", 422)
        context = self.context_or_404(str(request_payload.get("context_id") or ""), current_user)
        self.ensure_context_current(context, str(request_payload.get("expected_context_version") or ""))
        request_fp = fingerprint(request_payload)
        key_hash = idempotency_hash(idempotency_key)
        existing = self.repository.find_run_by_idempotency(owner_id=owner_id, key_hash=key_hash or "") if key_hash else None
        if existing:
            if existing.request_fingerprint != request_fp:
                raise ApiError("IDEMPOTENCY_KEY_REUSED", "Idempotency-Key was already used for a different Blend Optimizer run request.", 409)
            return self.run_response(existing)
        self.validate_scenario(context, request_payload)
        algorithm = str((request_payload.get("options") or {}).get("algorithm_version") or self.catalog()["algorithm_versions"][mode])
        expected_algorithm = "bmo_lp_legacy_v1" if mode == "lp_baseline" else "bmo_total_cost_de_legacy_v1"
        if algorithm != expected_algorithm:
            raise ApiError("BMO_MODEL_INCOMPATIBLE", "Unsupported Blend Optimizer algorithm version for selected mode.", 422)
        run = self.repository.create_run({
            "owner_id": owner_id,
            "mode": mode,
            "context_id": context.id,
            "context_version": context.version,
            "status": "queued",
            "progress": 0.0,
            "current_step": "queued",
            "request": request_payload,
            "idempotency_key_hash": key_hash,
            "request_fingerprint": request_fp,
        })
        self.repository.append_event(run_id=run.id, owner_id=owner_id, event_type="run_queued", payload={"mode": mode, "context_id": context.id})
        return self.run_response(run)

    def process_run(self, run_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        run = self.run_or_404(run_id, current_user)
        if run.status in TERMINAL_RUN_STATUSES:
            return self.run_response(run)
        self.repository.update_run(run.id, status="running", progress=0.05, current_step="loading_context")
        self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="run_started", payload={"mode": run.mode})
        try:
            context = self.context_or_404(run.context_id, current_user)
            result = self.execute_run(run, context)
            artifacts = self.create_artifacts(run, result)
            completed = self.repository.update_run(
                run.id,
                status="completed",
                progress=1.0,
                current_step="completed",
                result=result,
                warnings=result.get("warnings") or [],
                artifacts=artifacts,
            )
            self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="run_completed", payload={"mode": run.mode, "artifact_count": len(artifacts)})
            return self.run_response(completed or self.run_or_404(run.id, current_user))
        except ApiError as exc:
            failed = self.repository.update_run(run.id, status="failed", progress=1.0, current_step="failed", error_code=exc.code, error_message=exc.message)
            self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="run_failed", payload={"code": exc.code, "message": exc.message})
            return self.run_response(failed or run)
        except Exception as exc:
            code = "BMO_TOTAL_COST_FAILED" if run.mode == "total_cost" else "BMO_LP_FAILED"
            failed = self.repository.update_run(run.id, status="failed", progress=1.0, current_step="failed", error_code=code, error_message=str(exc)[:240])
            self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="run_failed", payload={"message": str(exc)[:240]})
            return self.run_response(failed or run)

    def get_run(self, run_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        return self.run_response(self.run_or_404(run_id, current_user))

    def run_events(self, run_id: str, current_user: dict[str, Any] | None, *, after: int | None = None) -> list[dict[str, Any]]:
        self.run_or_404(run_id, current_user)
        return [self.event_response(event) for event in self.repository.list_events(run_id, after=after)]

    def cancel_run(self, run_id: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        run = self.run_or_404(run_id, current_user)
        if run.status in TERMINAL_RUN_STATUSES:
            return self.run_response(run)
        updated = self.repository.update_run(run.id, status="cancelled", progress=run.progress, current_step="cancelled")
        self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="run_cancelled", payload={})
        return self.run_response(updated or run)

    def evaluate_manual_blend(self, run_id: str, payload: dict[str, Any], current_user: dict[str, Any] | None, *, idempotency_key: str | None) -> dict[str, Any]:
        self.require_idempotency(idempotency_key)
        run = self.run_or_404(run_id, current_user)
        context = self.context_or_404(run.context_id, current_user)
        ores = self.ores_from_scenario(context, {"scenario": {"ores": payload.get("ores") or []}})
        quantities = {str(item.get("ore_id")): float(item.get("quantity_mt") or 0.0) for item in payload.get("ores") or []}
        targets = self.targets(run.request, context)
        result = evaluate_blend(
            ores=ores,
            quantities_mt=quantities,
            feo_in_slag_pct=targets["feo_in_slag_pct"],
            hot_metal_target_mt=targets["target_hot_metal_mt"],
        )
        return {"run_id": run.id, "manual_evaluation": self.blend_result(result), "advisory_only": True, "operator_review_required": True}

    def execute_run(self, run: BlendOptimizerRunRecord, context: BlendOptimizerContextRecord) -> dict[str, Any]:
        request = run.request
        targets = self.targets(request, context)
        ores = self.ores_from_scenario(context, request)
        scenario = request.get("scenario") or {}
        fuel_ash_inputs = [dataclass_from_payload(FuelAshInput, item) for item in scenario.get("fuel_ash_inputs") or context.snapshot.get("fuel_ash_inputs") or []]
        flux_inputs = [dataclass_from_payload(FluxInput, item) for item in scenario.get("flux_inputs") or context.snapshot.get("flux_inputs") or []]
        dust_inputs = [dataclass_from_payload(DustInput, item) for item in scenario.get("dust_inputs") or context.snapshot.get("dust_inputs") or []]
        slag_settings = dataclass_from_payload(SlagBalanceSettings, scenario.get("slag_balance") or context.snapshot.get("slag_balance") or {})
        bmo_cfg = load_bmo_config()
        provider = EvonithBmoContextProvider()
        self.repository.update_run(run.id, progress=0.15, current_step="lp_baseline")
        self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="lp_started", payload={"target_fe_mt": targets["target_fe_mt"]})
        lp_physical, lp_errors = run_lp_baseline(
            ores,
            target_production_mt=targets["target_fe_mt"],
            target_slag_qty_mt=targets["max_slag_mt"],
            feo_in_slag_pct=targets["feo_in_slag_pct"],
            target_slag_basicity_min=targets["basicity_min"],
            target_slag_basicity_max=targets["basicity_max"],
            target_slag_t_basicity_min=targets["t_basicity_min"],
            target_slag_t_basicity_max=targets["t_basicity_max"],
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_settings,
            hot_metal_target_mt=targets["target_hot_metal_mt"],
        )
        if lp_physical is None:
            raise ApiError("BMO_LP_INFEASIBLE", "LP baseline is infeasible.", 422, {"errors": lp_errors})
        model_service, process_context, history_df, fuel_status, fuel_warnings = self.fuel_context(provider, bmo_cfg)
        lp_result = evaluate_blend_with_fuel_prediction(
            ores=ores,
            quantities_mt=lp_physical.quantities_mt,
            feo_in_slag_pct=targets["feo_in_slag_pct"],
            model_service=model_service,
            process_context=process_context,
            history_df=history_df,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_settings,
            hot_metal_target_mt=targets["target_hot_metal_mt"],
        )
        lp_result.feasible = lp_physical.feasible
        lp_result.violations = lp_physical.violations
        lp_si = self.predict_si(bmo_cfg, ores, lp_result.quantities_mt, process_context, history_df, targets["target_hot_metal_mt"])
        self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="lp_completed", payload={"feasible": lp_result.feasible})
        selected_result = lp_result
        de_result = None
        de_errors: list[str] = []
        fallback_decision = {"used_lp_fallback": False, "reason": None}
        if run.mode == "total_cost":
            self.repository.update_run(run.id, progress=0.45, current_step="total_cost_de")
            self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="de_started", payload={"lp_seed": "precomputed"})
            de_result, de_errors = run_nonlinear_optimizer(
                ores,
                target_production_mt=targets["target_fe_mt"],
                target_slag_qty_mt=targets["max_slag_mt"],
                feo_in_slag_pct=targets["feo_in_slag_pct"],
                target_slag_basicity_min=targets["basicity_min"],
                target_slag_basicity_max=targets["basicity_max"],
                target_slag_t_basicity_min=targets["t_basicity_min"],
                target_slag_t_basicity_max=targets["t_basicity_max"],
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                de_cfg=self.iteration_budget((run.request.get("options") or {}).get("iteration_budget_id")),
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_settings,
                hot_metal_target_mt=targets["target_hot_metal_mt"],
                precomputed_lp_blend=lp_result,
                precomputed_lp_errors=lp_errors,
            )
            if de_result is None or not de_result.feasible or de_result.objective_rs_per_thm > lp_result.objective_rs_per_thm + 1e-6:
                selected_result = copy.deepcopy(lp_result)
                selected_result.diagnostics = dict(selected_result.diagnostics)
                selected_result.diagnostics["de_fell_back_to_lp"] = True
                fallback_decision = {"used_lp_fallback": True, "reason": "de_missing_infeasible_or_more_expensive"}
            else:
                selected_result = de_result
            self.repository.append_event(run_id=run.id, owner_id=run.owner_id, event_type="de_completed", payload={"fallback": fallback_decision})
        selected_si = lp_si if selected_result is lp_result else self.predict_si(bmo_cfg, ores, selected_result.quantities_mt, process_context, history_df, targets["target_hot_metal_mt"])
        return {
            "run_id": run.id,
            "mode": run.mode,
            "context_id": context.id,
            "context_version": context.version,
            "algorithm_version": "bmo_lp_legacy_v1" if run.mode == "lp_baseline" else "bmo_total_cost_de_legacy_v1",
            "advisory_only": True,
            "operator_review_required": True,
            "targets": targets,
            "lp_result": self.blend_result(lp_result, si_prediction=lp_si),
            "total_cost_result": self.blend_result(de_result, si_prediction=selected_si) if de_result is not None else None,
            "selected_result": self.blend_result(selected_result, si_prediction=selected_si),
            "errors": {"lp": lp_errors, "total_cost": de_errors},
            "fallback_decision": fallback_decision,
            "model_versions": {"fuel_unit_cost": fuel_status, "hot_metal_si": self.safe_si_status(bmo_cfg)},
            "warnings": [_warning("BMO_FUEL_CONTEXT_WARNING", str(item)) for item in fuel_warnings],
            "completed_at": utc_now().isoformat(),
        }

    def fuel_context(self, provider: EvonithBmoContextProvider, bmo_cfg: dict[str, Any]) -> tuple[FuelUnitCostModelService, Any, Any, dict[str, Any], list[str]]:
        warnings: list[str] = []
        try:
            history_df, history_warnings = provider.get_history_frame(max_lag_steps=1)
            warnings.extend(history_warnings)
        except Exception as exc:
            history_df = None
            warnings.append(f"History context unavailable: {exc}")
        try:
            process_context, process_warnings = provider.get_process_context(max_lag_steps=1)
            warnings.extend(process_warnings)
        except Exception as exc:
            process_context = {}
            warnings.append(f"Process context unavailable: {exc}")
        service = FuelUnitCostModelService(bundle_cfg=bmo_cfg.get("model_bundle", {}), fallback_cfg=bmo_cfg.get("fallback_fuel_model", {}))
        try:
            status = json_safe(service.get_bundle_status())
        except Exception as exc:
            status = {"model_loaded": False, "scaler_loaded": False, "bundle_error": str(exc)[:160]}
        return service, process_context, history_df, status, warnings

    def predict_si(self, bmo_cfg: dict[str, Any], ores: list[OreInput], quantities: dict[str, float], process_context: Any, history_df: Any, hot_metal_target_mt: float) -> float | None:
        try:
            prev_si = None
            if history_df is not None and hasattr(history_df, "empty") and not history_df.empty and "CHEM_PCT_SI" in history_df.columns:
                series = history_df["CHEM_PCT_SI"].dropna()
                if not series.empty:
                    prev_si = float(series.iloc[-1])
            return SiPredictionService(bundle_cfg=bmo_cfg.get("si_model_bundle", {})).predict_blend_si(
                ores=ores,
                quantities_mt=quantities,
                process_context=process_context,
                prev_si=prev_si,
                hot_metal_target_mt=hot_metal_target_mt,
            )
        except Exception:
            return None

    def safe_si_status(self, bmo_cfg: dict[str, Any]) -> dict[str, Any]:
        try:
            return json_safe(SiPredictionService(bundle_cfg=bmo_cfg.get("si_model_bundle", {})).get_status())
        except Exception as exc:
            return {"model_loaded": False, "scaler_loaded": False, "bundle_error": str(exc)[:160]}

    def model_statuses(self) -> tuple[dict[str, Any], dict[str, Any]]:
        bmo_cfg = load_bmo_config()
        try:
            fuel = json_safe(FuelUnitCostModelService(bundle_cfg=bmo_cfg.get("model_bundle", {}), fallback_cfg=bmo_cfg.get("fallback_fuel_model", {})).get_bundle_status())
        except Exception as exc:
            fuel = {"model_loaded": False, "scaler_loaded": False, "bundle_error": str(exc)[:160]}
        return fuel, self.safe_si_status(bmo_cfg)

    def targets(self, request: dict[str, Any], context: BlendOptimizerContextRecord) -> dict[str, float]:
        scenario_targets = dict(((request.get("scenario") or {}).get("targets") or {}))
        cfg_targets = dict(context.snapshot.get("basicity_defaults") or {})
        hm = dict(context.snapshot.get("hot_metal_chemistry") or {})
        target_hot_metal_mt = finite(scenario_targets.get("target_hot_metal_mt"), finite(cfg_targets.get("target_production_mt"), 2350.0)) or 2350.0
        hm_fe_pct = finite(hm.get("hm_fe_pct_for_target"), finite(cfg_targets.get("hm_fe_pct_fallback"), 94.5)) or 94.5
        return {
            "target_hot_metal_mt": target_hot_metal_mt,
            "target_fe_mt": target_hot_metal_mt * hm_fe_pct / 100.0,
            "hot_metal_fe_pct": hm_fe_pct,
            "max_slag_mt": finite(scenario_targets.get("max_slag_mt"), finite(cfg_targets.get("target_slag_qty_mt"), 750.0)) or 750.0,
            "basicity_min": finite(scenario_targets.get("basicity_min"), finite(cfg_targets.get("target_slag_basicity_min"), 0.0)) or 0.0,
            "basicity_max": finite(scenario_targets.get("basicity_max"), finite(cfg_targets.get("target_slag_basicity_max"), 10.0)) or 10.0,
            "t_basicity_min": finite(scenario_targets.get("t_basicity_min"), finite(cfg_targets.get("target_slag_t_basicity_min"), 0.0)) or 0.0,
            "t_basicity_max": finite(scenario_targets.get("t_basicity_max"), finite(cfg_targets.get("target_slag_t_basicity_max"), 10.0)) or 10.0,
            "feo_in_slag_pct": finite(scenario_targets.get("feo_in_slag_pct"), 0.4) or 0.4,
        }

    def ores_from_scenario(self, context: BlendOptimizerContextRecord, request: dict[str, Any]) -> list[OreInput]:
        catalog = {str(item.get("ore_id")): item for item in context.snapshot.get("eligible_materials") or []}
        requested = ((request.get("scenario") or {}).get("ores") or [])
        selected_payloads = [item for item in requested if item.get("selected", True)] or [item for item in context.snapshot.get("eligible_materials") or []]
        ores: list[OreInput] = []
        for item in selected_payloads:
            ore_id = str(item.get("ore_id") or item.get("material_id") or "")
            if ore_id not in catalog:
                raise ApiError("BMO_INVALID_MATERIAL", "Selected material is not present in the immutable BMO context.", 422, {"ore_id": ore_id})
            ores.append(ore_from_payload(item, catalog[ore_id]))
        return ores

    def validate_scenario(self, context: BlendOptimizerContextRecord, request: dict[str, Any]) -> None:
        targets = self.targets(request, context)
        if targets["target_hot_metal_mt"] <= 0 or targets["target_fe_mt"] <= 0 or targets["max_slag_mt"] <= 0:
            raise ApiError("BMO_INVALID_TARGET", "BMO targets must be positive finite values.", 422)
        if targets["basicity_min"] > targets["basicity_max"] or targets["t_basicity_min"] > targets["t_basicity_max"]:
            raise ApiError("BMO_INVALID_TARGET", "Basicity minimum cannot exceed maximum.", 422)
        ores = self.ores_from_scenario(context, request)
        max_ores = int(getattr(self.settings, "bmo_max_selected_ores", 20))
        if len(ores) < 2:
            raise ApiError("BMO_INVALID_ORE_SELECTION", "Select at least two ores before running Blend Optimizer.", 422)
        if len(ores) > max_ores:
            raise ApiError("BMO_INVALID_ORE_SELECTION", "Too many ores selected for one Blend Optimizer run.", 422, {"max_selected_ores": max_ores})
        min_sum = sum(ore.min_share_pct for ore in ores)
        max_sum = sum(ore.max_share_pct for ore in ores)
        for ore in ores:
            values = [ore.stock_mt, ore.price_rs_per_mt, ore.min_share_pct, ore.max_share_pct, *json_safe(asdict(ore.chemistry)).values()]
            if any(finite(value) is None for value in values):
                raise ApiError("BMO_INVALID_CHEMISTRY", "Ore stock, price, share and chemistry values must be finite.", 422, {"ore_id": ore.ore_id})
            if ore.stock_mt < 0 or ore.price_rs_per_mt < 0:
                raise ApiError("BMO_INVALID_MATERIAL", "Ore stock and price must be nonnegative.", 422, {"ore_id": ore.ore_id})
            if ore.min_share_pct > ore.max_share_pct:
                raise ApiError("BMO_INVALID_SHARE_BOUNDS", "Ore minimum share cannot exceed maximum share.", 422, {"ore_id": ore.ore_id})
        if min_sum > 100.0 or max_sum < 100.0:
            raise ApiError("BMO_INVALID_SHARE_BOUNDS", "Ore share bounds cannot produce a 100% blend.", 422)

    def iteration_budget(self, budget_id: Any) -> dict[str, Any]:
        selected = str(budget_id or getattr(self.settings, "bmo_default_iteration_budget", "standard"))
        budgets = {
            "quick": {"maxiter": 10, "popsize": 5, "seed": 42, "polish": False},
            "standard": {"maxiter": min(80, self.settings.blend_optimizer_max_iterations), "popsize": 10, "seed": 42, "polish": False},
        }
        if selected not in budgets:
            raise ApiError("BMO_INVALID_RUN_OPTIONS", "Unknown iteration budget.", 422, {"iteration_budget_id": selected})
        return budgets[selected]

    def reject_solver_internals(self, options: dict[str, Any]) -> None:
        blocked = sorted(set(str(key) for key in options) & DISALLOWED_OPTION_KEYS)
        if blocked:
            raise ApiError("BMO_INVALID_RUN_OPTIONS", "Solver internals and model paths cannot be supplied by clients.", 422, {"blocked_keys": blocked})

    def create_artifacts(self, run: BlendOptimizerRunRecord, result: dict[str, Any]) -> list[dict[str, Any]]:
        requested = set((run.request.get("options") or {}).get("create_artifacts") or [])
        if not requested:
            return []
        responses = []
        if "complete_result_json" in requested or "summary_csv" in requested:
            metadata = self.artifacts.create_json_artifact(
                workflow="blend_optimizer",
                filename_prefix=f"{run.mode}_{run.id}_result",
                payload=result,
                owner_user_id=run.owner_id,
                calculation_id=run.id,
            )
            responses.append(self.artifacts.response(metadata, "/blend-optimizer"))
        if "blend_summary_csv" in requested or "summary_csv" in requested:
            selected = result.get("selected_result") or {}
            rows = [
                {"ore_id": ore_id, "quantity_mt": qty, "share_pct": (selected.get("shares_pct") or {}).get(ore_id)}
                for ore_id, qty in (selected.get("quantities_mt") or {}).items()
            ]
            metadata = self.artifacts.create_csv_artifact(
                workflow="blend_optimizer",
                filename_prefix=f"{run.mode}_{run.id}_blend_summary",
                rows=rows,
                owner_user_id=run.owner_id,
                calculation_id=run.id,
            )
            responses.append(self.artifacts.response(metadata, "/blend-optimizer"))
        return responses

    def blend_result(self, blend: Any, *, si_prediction: float | None = None) -> dict[str, Any] | None:
        if blend is None:
            return None
        data = json_safe(blend)
        data["si_prediction_pct"] = si_prediction
        data["advisory_only"] = True
        data["operator_review_required"] = True
        return data

    def context_or_404(self, context_id: str, current_user: dict[str, Any] | None) -> BlendOptimizerContextRecord:
        record = self.repository.get_context(context_id)
        if record is None:
            raise ApiError("BMO_CONTEXT_NOT_FOUND", "Blend Optimizer context not found.", 404)
        if record.owner_id not in {None, user_id(current_user)} and not is_admin(current_user):
            raise ApiError("FORBIDDEN", "Blend Optimizer context access denied.", 403)
        return record

    def run_or_404(self, run_id: str, current_user: dict[str, Any] | None) -> BlendOptimizerRunRecord:
        record = self.repository.get_run(run_id)
        if record is None:
            raise ApiError("BMO_RUN_NOT_FOUND", "Blend Optimizer run not found.", 404)
        if record.owner_id not in {None, user_id(current_user)} and not is_admin(current_user):
            raise ApiError("FORBIDDEN", "Blend Optimizer run access denied.", 403)
        return record

    def ensure_context_current(self, context: BlendOptimizerContextRecord, expected_version: str) -> None:
        if expected_version and expected_version != context.version:
            raise ApiError("BMO_CONTEXT_VERSION_CONFLICT", "Blend Optimizer context version does not match the submitted scenario.", 409)
        if context.expires_at and utc_now() >= context.expires_at.astimezone(timezone.utc):
            raise ApiError("BMO_CONTEXT_EXPIRED", "Blend Optimizer context has expired. Create a new context before running.", 409)

    def require_idempotency(self, key: str | None) -> None:
        if not str(key or "").strip():
            raise ApiError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required for Blend Optimizer mutations.", 400)

    @staticmethod
    def context_response(record: BlendOptimizerContextRecord) -> dict[str, Any]:
        snapshot = dict(record.snapshot)
        return {
            "context_id": record.id,
            "id": record.id,
            "owner_id": record.owner_id,
            "context_version": record.version,
            "fingerprint": record.fingerprint,
            "status": record.status,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "as_of_utc": snapshot.get("as_of_utc"),
            "advisory_only": True,
            "operator_review_required": True,
            **snapshot,
            "warnings": record.warnings,
        }

    @staticmethod
    def run_response(record: BlendOptimizerRunRecord) -> dict[str, Any]:
        return {
            "run_id": record.id,
            "id": record.id,
            "owner_id": record.owner_id,
            "mode": record.mode,
            "context_id": record.context_id,
            "context_version": record.context_version,
            "status": record.status,
            "progress": record.progress,
            "current_step": record.current_step,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "completed_at": record.completed_at,
            "status_path": f"/api/v1/blend-optimizer/runs/{record.id}",
            "events_path": f"/api/v1/blend-optimizer/runs/{record.id}/events",
            "result": record.result or None,
            "warnings": record.warnings,
            "artifacts": record.artifacts,
            "error_code": record.error_code,
            "error_message": record.error_message,
            "advisory_only": True,
            "operator_review_required": True,
        }

    @staticmethod
    def event_response(record: BlendOptimizerRunEventRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "run_id": record.run_id,
            "event_type": record.event_type,
            "sequence": record.sequence,
            "payload": record.payload,
            "created_at": record.created_at,
        }
