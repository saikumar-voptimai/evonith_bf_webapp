"""Backend service for API-first Material Balance workflows."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.material_balance_repository import MaterialBalanceConfigRepository, checksum
from apps.backend_api.app.services.audit_service import AuditService
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from furnace_data.material_balance.constants import ELEMENTS, MATERIAL_REGISTRY
from furnace_data.material_balance.context import ALGORITHM_VERSION, CATALOG_VERSION, MaterialBalanceContextBuilder
from furnace_data.material_balance.data_sources import STATIC_DATASET_ID, clear_day_caches, get_static_dataset_metadata
from furnace_data.material_balance.dpr_mapping import ASH_ANALYSIS_FIELDS, ASH_MATERIAL_CONFIG_KEYS, CANONICAL_MASS_FIELDS, load_full_config
from furnace_data.material_balance.engine import MaterialBalanceEngine
from furnace_data.material_balance.result_builder import build_material_balance_result
from furnace_data.material_balance.types import BalanceResult

_INPUT_STREAMS = [("burden", "Burden", "t"), ("hot_blast", "Hot Blast", "t"), ("oxygen_enrichment", "O2 Enrichment", "t"), ("steam", "Steam", "t")]
_OUTPUT_STREAMS = [("hot_metal", "Hot Metal", "t"), ("slag", "Slag", "t"), ("top_gas", "Top Gas", "t"), ("dust_catcher", "Dust Catcher", "t"), ("unaccounted", "Unaccounted", "t")]
_DPR_SOURCE_FIELDS = [
    {"source_field_id": "total_hot_metal_mt", "label": "Total Hot Metal", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "slag_generation_mt", "label": "Slag Generation", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "coke_total_mt", "label": "Coke Total", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "nut_coke_total_mt", "label": "Nut Coke Total", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "pci_total_mt", "label": "PCI Total", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "ore_total_mt", "label": "Ore Total", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "sinter_total_mt", "label": "Sinter Total", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "pellet_total_mt", "label": "Pellet Total", "unit": "t", "aggregation_policy": "latest_non_null"},
    {"source_field_id": "flux_total_mt", "label": "Flux Total", "unit": "t", "aggregation_policy": "latest_non_null"},
]
_MATERIAL_LABELS = {"coke": "EML Coke", "nutcoke": "Nut Coke", "pci": "PCI Coal"}


def _warning(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details or {}}


def table_data(rows: list[dict[str, Any]], max_rows: int) -> dict[str, Any]:
    returned = rows[:max_rows]
    columns = [
        {"name": key, "type": "number" if isinstance(value, (int, float)) else "string"}
        for key, value in (returned[0].items() if returned else [])
    ]
    return {
        "columns": columns,
        "rows": returned,
        "row_count": len(rows),
        "returned_rows": len(returned),
        "truncated": len(rows) > len(returned),
    }

class MaterialBalanceService:
    """API-facing Material Balance application service."""

    def __init__(self, *, settings: BackendSettings | None = None, artifact_service: ComputeArtifactService | None = None, repository: MaterialBalanceConfigRepository | None = None, audit_service: AuditService | None = None, clock: Any | None = None) -> None:
        self.settings = settings or load_backend_settings()
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)
        self.repository = repository or MaterialBalanceConfigRepository()
        self.audit_service = audit_service
        self.clock = clock
        self._run_cache: dict[str, dict[str, Any]] = {}

    def config(self) -> dict[str, Any]:
        cfg, version = self._effective_config()
        dataset = get_static_dataset_metadata()
        warnings = []
        if dataset.get("status") != "ready":
            warnings.append(_warning("MATERIAL_BALANCE_DATASET_NOT_AVAILABLE", "Static ML dataset is not available for Material Balance runs."))
        return {
            "catalog_version": CATALOG_VERSION,
            "effective_config_version": version,
            "display_timezone": self._display_timezone(),
            "dataset": {"dataset_id": STATIC_DATASET_ID, "version": dataset.get("version"), "status": dataset.get("status") or "missing", "available_date_range": dataset.get("available_date_range") or {"minimum": None, "maximum": None}},
            "defaults": {"rm_lag_hours": 0, "blast_lag_hours": 0, "dust_catcher_t": 0.0, "algorithm_version": getattr(self.settings, "material_balance_default_algorithm_version", ALGORITHM_VERSION)},
            "limits": self._limits(),
            "closure_thresholds": self._thresholds(cfg),
            "elements": [{"id": e.lower(), "label": self._element_label(e), "unit": "t"} for e in ELEMENTS],
            "materials": [{"id": self._material_id(s.name), "label": s.name, "unit": "t"} for s in MATERIAL_REGISTRY],
            "input_streams": [{"id": i[0], "label": i[1], "unit": i[2]} for i in _INPUT_STREAMS],
            "output_streams": [{"id": i[0], "label": i[1], "unit": i[2]} for i in _OUTPUT_STREAMS],
            "algorithm_versions": [{"id": "legacy_v1", "label": "Validated legacy-compatible balance", "tracked_element_ids": [e.lower() for e in ELEMENTS]}],
            "available_sources": ["static_dataset", "input_data"],
            "capabilities": {"runtime_configuration_writable": self._runtime_writable(), "ash_analysis_editable": self._runtime_writable(), "dpr_mapping_editable": self._runtime_writable(), "export_available": True, "async_jobs_required": False},
            "warnings": warnings,
        }

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        source = str(payload.get("source") or "static_dataset")
        if source == "static_dataset":
            try:
                self._validate_static_request(payload)
            except ApiError as exc:
                errors.append(_warning(exc.code, exc.message, exc.details))
        elif source == "input_data":
            if not isinstance(payload.get("input_data"), dict):
                errors.append(_warning("MATERIAL_BALANCE_DATE_REQUIRED", "input_data is required."))
        else:
            errors.append(_warning("MATERIAL_BALANCE_INVALID_SOURCE", "Unsupported Material Balance source."))
        return {"valid": not errors, "errors": errors, "warnings": []}

    def run(self, payload: dict[str, Any], *, route_prefix: str = "/material-balance", current_user: dict[str, Any] | None = None, request_id: str | None = None) -> dict[str, Any]:
        _ = request_id
        source = str(payload.get("source") or "static_dataset")
        if source == "input_data":
            if self._auth_required() and "material_balance:diagnostics" not in self._permissions(current_user):
                raise ApiError("FORBIDDEN", "Explicit input_data mode requires diagnostics permission.", status_code=403)
            return self._run_input_data(payload, route_prefix=route_prefix, current_user=current_user)
        self._validate_static_request(payload)
        cache_key = self._run_cache_key(payload)
        if cache_key in self._run_cache and not payload.get("export_format") and not payload.get("export"):
            return self._run_cache[cache_key]
        cfg, _version = self._effective_config()
        options = payload.get("options") or {}
        context = MaterialBalanceContextBuilder(config=cfg).build(day=self._request_day(payload), rm_lag_hours=int(options.get("rm_lag_hours") or 0), blast_lag_hours=int(options.get("blast_lag_hours") or 0), dust_catcher_t=float(options.get("dust_catcher_t") or 0.0), algorithm_version=str(options.get("algorithm_version") or ALGORITHM_VERSION))
        try:
            result = MaterialBalanceEngine().compute(context)
            data = build_material_balance_result(result, config=cfg)
        except Exception as exc:
            raise ApiError("MATERIAL_BALANCE_CALCULATION_FAILED", "Material Balance calculation failed.", status_code=500) from exc
        artifacts = self._create_artifacts(data, export_format=payload.get("export_format") or ("closure_csv" if payload.get("export") else None), route_prefix=route_prefix, current_user=current_user)
        if artifacts:
            data["artifacts"] = artifacts
        else:
            self._run_cache[cache_key] = data
        return data
    def get_ash_analyses(self) -> dict[str, Any]:
        cfg, version = self._effective_config()
        materials = []
        for material_id, yml_key in ASH_MATERIAL_CONFIG_KEYS.items():
            current = cfg.get(yml_key, {}) or {}
            net_basis = set(cfg.get(f"{material_id}_net_fuel_basis_species", []) or [])
            materials.append({
                "material_id": material_id,
                "label": _MATERIAL_LABELS.get(material_id, material_id.title()),
                "species": [
                    {"species_id": key, "label": label, "basis": "net_fuel" if key in net_basis else "ash", "value": float(current.get(key, 0.0) or 0.0)}
                    for key, label, _kind in ASH_ANALYSIS_FIELDS
                ],
            })
        return {"config_version": version, "materials": materials, "writable": self._runtime_writable()}

    def update_ash_analyses(self, payload: dict[str, Any], *, current_user: dict[str, Any] | None = None, request_id: str | None = None, client_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_runtime_writable()
        cfg, version = self._effective_config()
        self._check_expected_config(payload, version)
        known_materials = set(ASH_MATERIAL_CONFIG_KEYS)
        known_species = {item[0] for item in ASH_ANALYSIS_FIELDS}
        next_cfg = dict(cfg)
        for material in payload.get("materials") or []:
            material_id = str(material.get("material_id") or "")
            if material_id not in known_materials:
                raise ApiError("MATERIAL_BALANCE_INVALID_ASH_ANALYSIS", "Unknown ash material id.", status_code=422)
            net_basis = set(next_cfg.get(f"{material_id}_net_fuel_basis_species", []) or [])
            analysis: dict[str, float] = {}
            ash_total = 0.0
            for species in material.get("species") or []:
                species_id = str(species.get("species_id") or "")
                if species_id not in known_species:
                    raise ApiError("MATERIAL_BALANCE_INVALID_ASH_ANALYSIS", "Unknown ash species id.", status_code=422)
                value = self._finite_float(species.get("value"), "MATERIAL_BALANCE_INVALID_ASH_ANALYSIS")
                if value < 0.0 or value > 100.0:
                    raise ApiError("MATERIAL_BALANCE_INVALID_ASH_ANALYSIS", "Ash species values must be between 0 and 100.", status_code=422)
                analysis[species_id] = value
                if species_id not in net_basis:
                    ash_total += value
            if ash_total > 100.0:
                raise ApiError("MATERIAL_BALANCE_ASH_TOTAL_INVALID", "Ash-basis species total cannot exceed 100%.", status_code=422, details={"material_id": material_id, "ash_basis_total": ash_total})
            next_cfg[ASH_MATERIAL_CONFIG_KEYS[material_id]] = analysis
        revision = self.repository.create_revision(profile_key="plant-default", expected_config_version=version, config=next_cfg, packaged_default_checksum=checksum(load_full_config()), actor_user_id=self._actor_id(current_user), request_id=request_id, client_metadata=client_metadata)
        self._audit("material_balance.config.ash_updated", current_user, request_id, {"version": revision.version})
        self.refresh_cache({"scopes": ["calculation_snapshot"]})
        return self.get_ash_analyses()

    def get_dpr_mapping(self, *, sample_day: date | None = None) -> dict[str, Any]:
        cfg, version = self._effective_config()
        mapping = cfg.get("dpr_field_mapping") or {}
        mapped = [bool(mapping.get(field)) for field in CANONICAL_MASS_FIELDS]
        status = "complete" if all(mapped) else "partial" if any(mapped) else "none"
        return {
            "config_version": version,
            "status": status,
            "canonical_fields": [{"canonical_field_id": field, "label": field.replace("_", " ").title(), "unit": "t", "aggregation_policy": "latest_non_null"} for field in CANONICAL_MASS_FIELDS],
            "mapping": [{"canonical_field_id": field, "source_field_id": mapping.get(field)} for field in CANONICAL_MASS_FIELDS],
            "approved_source_fields": [dict(item, data_type="number") for item in _DPR_SOURCE_FIELDS],
            "selected_day_availability": {"day": sample_day.isoformat()} if sample_day else None,
            "writable": self._runtime_writable(),
        }

    def update_dpr_mapping(self, payload: dict[str, Any], *, current_user: dict[str, Any] | None = None, request_id: str | None = None, client_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_runtime_writable()
        cfg, version = self._effective_config()
        self._check_expected_config(payload, version)
        approved = {item["source_field_id"] for item in _DPR_SOURCE_FIELDS}
        next_mapping = {field: None for field in CANONICAL_MASS_FIELDS}
        seen: set[str] = set()
        for item in payload.get("mapping") or []:
            canonical = str(item.get("canonical_field_id") or "")
            source = item.get("source_field_id")
            if canonical not in CANONICAL_MASS_FIELDS or canonical in seen:
                raise ApiError("MATERIAL_BALANCE_INVALID_DPR_MAPPING", "Invalid or duplicate DPR mapping field.", status_code=422)
            seen.add(canonical)
            if source is not None and str(source) not in approved:
                raise ApiError("MATERIAL_BALANCE_INVALID_DPR_FIELD", "DPR source field is not approved.", status_code=422)
            next_mapping[canonical] = str(source) if source else None
        next_cfg = dict(cfg)
        next_cfg["dpr_field_mapping"] = next_mapping
        revision = self.repository.create_revision(profile_key="plant-default", expected_config_version=version, config=next_cfg, packaged_default_checksum=checksum(load_full_config()), actor_user_id=self._actor_id(current_user), request_id=request_id, client_metadata=client_metadata)
        self._audit("material_balance.config.dpr_mapping_updated", current_user, request_id, {"version": revision.version})
        self.refresh_cache({"scopes": ["calculation_snapshot", "dpr"]})
        return self.get_dpr_mapping()

    def refresh_cache(self, payload: dict[str, Any]) -> dict[str, Any]:
        scopes = [str(item) for item in (payload.get("scopes") or [])]
        invalid = sorted(set(scopes) - {"calculation_snapshot", "dpr"})
        if invalid or not scopes:
            raise ApiError("MATERIAL_BALANCE_CACHE_SCOPE_INVALID", "Material Balance cache scope is invalid.", status_code=422, details={"invalid_scopes": invalid})
        self._run_cache.clear()
        clear_day_caches(payload.get("day") or date.today())
        return {"invalidated_scopes": scopes, "day": payload.get("day")}

    def _run_input_data(self, payload: dict[str, Any], *, route_prefix: str, current_user: dict[str, Any] | None) -> dict[str, Any]:
        data = payload.get("input_data") or {}
        inputs = data.get("inputs") or {}
        outputs = data.get("outputs") or {}
        elements = sorted(set(inputs) | set(outputs))
        element_inputs = {element: {"Input": float(inputs.get(element) or 0.0)} for element in elements}
        element_outputs = {element: {"Output": float(outputs.get(element) or 0.0)} for element in elements}
        rows = []
        for element in elements:
            input_t = float(inputs.get(element) or 0.0)
            output_t = float(outputs.get(element) or 0.0)
            rows.append({"Element": element, "In_t": round(input_t, 2), "Out_t": round(output_t, 2), "Closure_pct": round(output_t / input_t * 100.0, 1) if input_t > 0 else None, "Delta_t": round(output_t - input_t, 2)})
        result = BalanceResult(day=self._request_day(payload) or self._today_ist(), inputs=element_inputs, outputs=element_outputs, closure_table=pd.DataFrame(rows), material_masses={}, gas_phase={"top_gas_method": "explicit_input", "top_gas_fallback_applied": False}, versions={"dataset_version": "explicit-input", "config_version": "explicit-input", "catalog_version": CATALOG_VERSION})
        cfg, _version = self._effective_config()
        response = build_material_balance_result(result, config=cfg)
        artifacts = self._create_artifacts(response, export_format=payload.get("export_format") or ("closure_csv" if payload.get("export") else None), route_prefix=route_prefix, current_user=current_user)
        if artifacts:
            response["artifacts"] = artifacts
        return response
    def _validate_static_request(self, payload: dict[str, Any]) -> None:
        day = self._request_day(payload)
        if day is None:
            raise ApiError("MATERIAL_BALANCE_DATE_REQUIRED", "A completed IST day is required.", status_code=422)
        if getattr(self.settings, "material_balance_require_completed_day", True) and day >= self._today_ist():
            raise ApiError("MATERIAL_BALANCE_PARTIAL_DAY_NOT_ALLOWED", "Today and future dates are not available for completed-day Material Balance runs.", status_code=422)
        dataset = get_static_dataset_metadata()
        if dataset.get("status") == "missing":
            raise ApiError("MATERIAL_BALANCE_DATASET_NOT_AVAILABLE", "Static ML dataset is not available.", status_code=404)
        if dataset.get("status") == "not_ready":
            raise ApiError("MATERIAL_BALANCE_DATASET_NOT_READY", "Static ML dataset is being replaced; retry shortly.", status_code=409)
        range_info = dataset.get("available_date_range") or {}
        minimum = range_info.get("minimum")
        maximum = range_info.get("maximum")
        if (minimum and day < minimum) or (maximum and day > maximum):
            raise ApiError("MATERIAL_BALANCE_DATE_OUT_OF_RANGE", "Date is outside the static dataset availability range.", status_code=422, details={"minimum": str(minimum), "maximum": str(maximum)})
        expected_dataset = payload.get("expected_dataset_version")
        if expected_dataset and expected_dataset != dataset.get("version"):
            raise ApiError("MATERIAL_BALANCE_DATASET_VERSION_CONFLICT", "Static dataset version conflict.", status_code=409, details={"current_version": dataset.get("version")})
        _cfg, config_version = self._effective_config()
        expected_config = payload.get("expected_config_version")
        if expected_config and expected_config != config_version:
            raise ApiError("MATERIAL_BALANCE_CONFIG_VERSION_CONFLICT", "Material Balance configuration version conflict.", status_code=409, details={"current_version": config_version})
        options = payload.get("options") or {}
        limits = self._limits()
        rm_lag = int(options.get("rm_lag_hours") or 0)
        blast_lag = int(options.get("blast_lag_hours") or 0)
        if rm_lag < limits["rm_lag_hours_min"] or rm_lag > limits["rm_lag_hours_max"]:
            raise ApiError("MATERIAL_BALANCE_INVALID_LAG", "RM lag is outside configured limits.", status_code=422)
        if blast_lag < limits["blast_lag_hours_min"] or blast_lag > limits["blast_lag_hours_max"]:
            raise ApiError("MATERIAL_BALANCE_INVALID_LAG", "Blast lag is outside configured limits.", status_code=422)
        dust = self._finite_float(options.get("dust_catcher_t") or 0.0, "MATERIAL_BALANCE_INVALID_DUST_CATCHER")
        if dust < limits["dust_catcher_t_min"] or dust > limits["dust_catcher_t_max"]:
            raise ApiError("MATERIAL_BALANCE_INVALID_DUST_CATCHER", "Dust catcher tonnes are outside configured limits.", status_code=422)
        algorithm = str(options.get("algorithm_version") or ALGORITHM_VERSION)
        if algorithm != "legacy_v1":
            raise ApiError("MATERIAL_BALANCE_INVALID_ALGORITHM", "Unsupported Material Balance algorithm version.", status_code=422)

    def _effective_config(self) -> tuple[dict[str, Any], str]:
        default_config = load_full_config()
        default_version = str(default_config.get("version") or "").strip() or f"mbcfg-{checksum(default_config)[:12]}"
        revision = self.repository.latest_revision("plant-default")
        if revision is not None:
            cfg = dict(revision.config)
            cfg["version"] = revision.version
            return cfg, revision.version
        cfg = dict(default_config)
        cfg["version"] = default_version
        return cfg, default_version

    def _create_artifacts(self, data: dict[str, Any], *, export_format: str | None, route_prefix: str, current_user: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not export_format:
            return []
        if export_format not in {"closure_csv", "full_json"}:
            raise ApiError("MATERIAL_BALANCE_RESULT_INVALID", "Unsupported export format.", status_code=422)
        owner = self._actor_id(current_user)
        calculation_id = str(data.get("calculation_id") or "")
        if export_format == "closure_csv":
            metadata = self.artifacts.create_csv_artifact(workflow="material_balance", filename_prefix="material_balance_closure", rows=data.get("tables", {}).get("closure", {}).get("rows", []), ttl_hours=getattr(self.settings, "material_balance_export_ttl_hours", None), owner_user_id=owner, calculation_id=calculation_id)
        else:
            metadata = self.artifacts.create_json_artifact(workflow="material_balance", filename_prefix="material_balance_full_result", payload=data, ttl_hours=getattr(self.settings, "material_balance_export_ttl_hours", None), owner_user_id=owner, calculation_id=calculation_id)
        return [self.artifacts.response(metadata, route_prefix)]

    def _run_cache_key(self, payload: dict[str, Any]) -> str:
        cfg, version = self._effective_config()
        dataset = get_static_dataset_metadata()
        data = {"dataset_version": dataset.get("version"), "config_version": version, "day": str(self._request_day(payload)), "options": payload.get("options") or {}, "cfg_checksum": checksum(cfg)[:12]}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _thresholds(self, cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
        raw = cfg.get("closure_thresholds") or {}
        good = raw.get("good") or [95, 105]
        warning = raw.get("warning") or [85, 115]
        return {"good": {"minimum": float(good[0]), "maximum": float(good[1])}, "warning": {"minimum": float(warning[0]), "maximum": float(warning[1])}}

    def _limits(self) -> dict[str, Any]:
        return {"rm_lag_hours_min": 0, "rm_lag_hours_max": int(getattr(self.settings, "material_balance_max_rm_lag_hours", 240)), "blast_lag_hours_min": 0, "blast_lag_hours_max": int(getattr(self.settings, "material_balance_max_blast_lag_hours", 48)), "dust_catcher_t_min": 0.0, "dust_catcher_t_max": float(getattr(self.settings, "material_balance_max_dust_catcher_t", 500.0))}

    def _request_day(self, payload: dict[str, Any]) -> date | None:
        value = payload.get("day", payload.get("date"))
        if value is None:
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    def _today_ist(self) -> date:
        tz = ZoneInfo(self._display_timezone())
        now = self.clock() if self.clock is not None else datetime.now(tz)
        if isinstance(now, datetime):
            if now.tzinfo is None:
                now = now.replace(tzinfo=tz)
            return now.astimezone(tz).date()
        return datetime.now(tz).date()

    def _display_timezone(self) -> str:
        return str(getattr(self.settings, "material_balance_default_timezone", "Asia/Kolkata") or "Asia/Kolkata")

    def _runtime_writable(self) -> bool:
        return bool(getattr(self.settings, "material_balance_allow_runtime_config", False))

    def _ensure_runtime_writable(self) -> None:
        if not self._runtime_writable():
            raise ApiError("MATERIAL_BALANCE_CONFIG_READ_ONLY", "Runtime Material Balance configuration writes are disabled.", status_code=403)

    def _check_expected_config(self, payload: dict[str, Any], version: str) -> None:
        expected = str(payload.get("expected_config_version") or "")
        if expected != version:
            raise ApiError("MATERIAL_BALANCE_CONFIG_VERSION_CONFLICT", "Material Balance configuration version conflict.", status_code=409, details={"current_version": version})

    def _auth_required(self) -> bool:
        return bool(getattr(self.settings, "compute_require_auth", True))

    @staticmethod
    def _permissions(user: dict[str, Any] | None) -> set[str]:
        return {str(item) for item in ((user or {}).get("permissions") or [])}

    @staticmethod
    def _actor_id(user: dict[str, Any] | None) -> str | None:
        return str((user or {}).get("id") or "") or None

    @staticmethod
    def _finite_float(value: Any, code: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ApiError(code, "Value must be numeric.", status_code=422) from exc
        if not math.isfinite(number):
            raise ApiError(code, "Value must be finite.", status_code=422)
        return number

    @staticmethod
    def _material_id(label: str) -> str:
        return label.lower().replace(" ", "_")

    @staticmethod
    def _element_label(symbol: str) -> str:
        labels = {"Fe": "Iron", "C": "Carbon", "Si": "Silicon", "Ca": "Calcium", "Mg": "Magnesium", "Al": "Aluminium", "Mn": "Manganese", "S": "Sulfur", "P": "Phosphorus", "O": "Oxygen", "N": "Nitrogen", "H": "Hydrogen"}
        return labels.get(symbol, symbol)

    def _audit(self, event_type: str, current_user: dict[str, Any] | None, request_id: str | None, metadata: dict[str, Any]) -> None:
        if self.audit_service is None:
            return
        self.audit_service.record_event({"request_id": request_id, "actor_user_id": self._actor_id(current_user), "actor_username": (current_user or {}).get("username"), "event_type": event_type, "resource_type": "material-balance", "resource_id": "plant-default", "action": "update", "result": "success", "status_code": 200, "metadata": metadata})