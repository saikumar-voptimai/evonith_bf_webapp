"""Material Balance gateways for API-first mode and direct rollback mode."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services import material_balance_api
from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services.api_errors import BackendApiHTTPError

JsonDict = dict[str, Any]


@runtime_checkable
class MaterialBalanceGateway(Protocol):
    def get_config(self) -> JsonDict: ...
    def run(self, request: JsonDict) -> JsonDict: ...
    def refresh_cache(self, request: JsonDict) -> JsonDict: ...
    def get_ash_analyses(self) -> JsonDict: ...
    def update_ash_analyses(self, request: JsonDict) -> JsonDict: ...
    def get_dpr_mapping(self, *, sample_day: str | None = None) -> JsonDict: ...
    def update_dpr_mapping(self, request: JsonDict) -> JsonDict: ...
    def download_artifact(self, artifact_id: str) -> bytes: ...


class ApiMaterialBalanceGateway:
    def __init__(self, access_token: str, client: ApiClient | None = None) -> None:
        self.access_token = str(access_token or "").strip()
        self.client = client

    def get_config(self) -> JsonDict:
        return material_balance_api.get_material_balance_config(self.access_token, self.client)

    def run(self, request: JsonDict) -> JsonDict:
        return material_balance_api.run_material_balance(request, self.access_token, self.client)

    def refresh_cache(self, request: JsonDict) -> JsonDict:
        return material_balance_api.refresh_material_balance_cache(request, self.access_token, self.client)

    def get_ash_analyses(self) -> JsonDict:
        return material_balance_api.get_material_balance_ash_analyses(self.access_token, self.client)

    def update_ash_analyses(self, request: JsonDict) -> JsonDict:
        return material_balance_api.update_material_balance_ash_analyses(request, self.access_token, self.client)

    def get_dpr_mapping(self, *, sample_day: str | None = None) -> JsonDict:
        return material_balance_api.get_material_balance_dpr_mapping(sample_day, self.access_token, self.client)

    def update_dpr_mapping(self, request: JsonDict) -> JsonDict:
        return material_balance_api.update_material_balance_dpr_mapping(request, self.access_token, self.client)

    def download_artifact(self, artifact_id: str) -> bytes:
        return material_balance_api.download_material_balance_artifact(artifact_id, self.access_token, self.client)


class DirectMaterialBalanceGateway:
    """Deprecated direct gateway kept as a lazy-loaded rollback path."""

    def get_config(self) -> JsonDict:
        from furnace_data.material_balance.constants import ELEMENTS, MATERIAL_REGISTRY
        from furnace_data.material_balance.context import CATALOG_VERSION
        from furnace_data.material_balance.data_sources import STATIC_DATASET_ID, get_static_dataset_metadata
        from furnace_data.material_balance.dpr_mapping import load_full_config

        cfg = load_full_config()
        version = str(cfg.get("version") or "").strip() or f"mbcfg-{_checksum(cfg)[:12]}"
        dataset = get_static_dataset_metadata()
        thresholds = _thresholds(cfg)
        return {
            "catalog_version": CATALOG_VERSION,
            "effective_config_version": version,
            "display_timezone": "Asia/Kolkata",
            "dataset": {"dataset_id": STATIC_DATASET_ID, "version": dataset.get("version"), "status": dataset.get("status") or "missing", "available_date_range": dataset.get("available_date_range") or {"minimum": None, "maximum": None}},
            "defaults": {"rm_lag_hours": 0, "blast_lag_hours": 0, "dust_catcher_t": 0.0, "algorithm_version": "legacy_v1"},
            "limits": {"rm_lag_hours_min": 0, "rm_lag_hours_max": 240, "blast_lag_hours_min": 0, "blast_lag_hours_max": 48, "dust_catcher_t_min": 0.0, "dust_catcher_t_max": 500.0},
            "closure_thresholds": thresholds,
            "elements": [{"id": e.lower(), "label": e, "unit": "t"} for e in ELEMENTS],
            "materials": [{"id": s.name.lower().replace(" ", "_"), "label": s.name, "unit": "t"} for s in MATERIAL_REGISTRY],
            "input_streams": [{"id": i, "label": label, "unit": "t"} for i, label in [("burden", "Burden"), ("hot_blast", "Hot Blast"), ("oxygen_enrichment", "O2 Enrichment"), ("steam", "Steam")]],
            "output_streams": [{"id": i, "label": label, "unit": "t"} for i, label in [("hot_metal", "Hot Metal"), ("slag", "Slag"), ("top_gas", "Top Gas"), ("dust_catcher", "Dust Catcher")]],
            "algorithm_versions": [{"id": "legacy_v1", "label": "Validated legacy-compatible balance", "tracked_element_ids": [e.lower() for e in ELEMENTS]}],
            "capabilities": {"runtime_configuration_writable": True, "ash_analysis_editable": True, "dpr_mapping_editable": True, "export_available": False, "async_jobs_required": False},
            "warnings": [{"code": "MATERIAL_BALANCE_DIRECT_MODE", "message": "Direct Material Balance mode is deprecated; enable USE_BACKEND_API_MATERIAL_BALANCE.", "details": {}}],
        }

    def run(self, request: JsonDict) -> JsonDict:
        from furnace_data.material_balance.compute import run_full_balance
        from furnace_data.material_balance.dpr_mapping import load_full_config
        from furnace_data.material_balance.result_builder import build_material_balance_result

        options = request.get("options") or {}
        run_day = date.fromisoformat(str(request.get("day") or request.get("date")))
        result = run_full_balance(run_day, rm_lag_hours=int(options.get("rm_lag_hours") or 0), blast_lag_hours=int(options.get("blast_lag_hours") or 0), dust_catcher_t=float(options.get("dust_catcher_t") or 0.0))
        payload = build_material_balance_result(result, config=load_full_config())
        payload.setdefault("warnings", []).append({"code": "MATERIAL_BALANCE_DIRECT_MODE", "message": "Direct Material Balance mode is deprecated; enable USE_BACKEND_API_MATERIAL_BALANCE.", "details": {}})
        return payload

    def refresh_cache(self, request: JsonDict) -> JsonDict:
        from furnace_data.material_balance.data_sources import clear_day_caches

        raw_day = request.get("day")
        clear_day_caches(date.fromisoformat(str(raw_day)) if raw_day else date.today())
        return {"invalidated_scopes": list(request.get("scopes") or []), "day": raw_day}

    def get_ash_analyses(self) -> JsonDict:
        from furnace_data.material_balance.dpr_mapping import ASH_ANALYSIS_FIELDS, ASH_MATERIAL_CONFIG_KEYS, load_full_config

        cfg = load_full_config()
        materials = []
        labels = {"coke": "EML Coke", "nutcoke": "Nut Coke", "pci": "PCI Coal"}
        for material_id, yml_key in ASH_MATERIAL_CONFIG_KEYS.items():
            current = cfg.get(yml_key, {}) or {}
            net_basis = set(cfg.get(f"{material_id}_net_fuel_basis_species", []) or [])
            materials.append({"material_id": material_id, "label": labels.get(material_id, material_id.title()), "species": [{"species_id": key, "label": label, "basis": "net_fuel" if key in net_basis else "ash", "value": float(current.get(key, 0.0) or 0.0)} for key, label, _kind in ASH_ANALYSIS_FIELDS]})
        return {"config_version": str(cfg.get("version") or f"mbcfg-{_checksum(cfg)[:12]}"), "materials": materials, "writable": True}

    def update_ash_analyses(self, request: JsonDict) -> JsonDict:
        from furnace_data.material_balance.dpr_mapping import save_ash_analyses

        analyses = {m["material_id"]: {s["species_id"]: s["value"] for s in m.get("species") or []} for m in request.get("materials") or []}
        save_ash_analyses(analyses)
        return self.get_ash_analyses()

    def get_dpr_mapping(self, *, sample_day: str | None = None) -> JsonDict:
        from furnace_data.material_balance.dpr_mapping import CANONICAL_MASS_FIELDS, load_dpr_mapping

        mapping = load_dpr_mapping()
        mapped = [bool(mapping.get(field)) for field in CANONICAL_MASS_FIELDS]
        status = "complete" if all(mapped) else "partial" if any(mapped) else "none"
        source_fields = _approved_dpr_source_fields()
        return {"config_version": "direct-yaml", "status": status, "canonical_fields": [{"canonical_field_id": f, "label": f.replace("_", " ").title(), "unit": "t", "aggregation_policy": "latest_non_null"} for f in CANONICAL_MASS_FIELDS], "mapping": [{"canonical_field_id": f, "source_field_id": mapping.get(f)} for f in CANONICAL_MASS_FIELDS], "approved_source_fields": source_fields, "selected_day_availability": {"day": sample_day} if sample_day else None, "writable": True}

    def update_dpr_mapping(self, request: JsonDict) -> JsonDict:
        from furnace_data.material_balance.dpr_mapping import save_dpr_mapping

        save_dpr_mapping({item["canonical_field_id"]: item.get("source_field_id") for item in request.get("mapping") or []})
        return self.get_dpr_mapping()

    def download_artifact(self, artifact_id: str) -> bytes:
        raise BackendApiHTTPError("Direct Material Balance mode does not create authenticated API artifacts.", status_code=404, error_code="MATERIAL_BALANCE_ARTIFACT_NOT_FOUND")


def get_material_balance_gateway(*, access_token: str | None = None, client: ApiClient | None = None) -> MaterialBalanceGateway:
    if is_backend_api_enabled("material_balance"):
        if not is_backend_api_enabled("auth"):
            raise BackendApiHTTPError("Material Balance API mode requires USE_BACKEND_API_AUTH=true.", status_code=401, error_code="AUTH_REQUIRED")
        token = str(access_token or "").strip()
        if not token:
            raise BackendApiHTTPError("Material Balance API mode requires a backend access token.", status_code=401, error_code="AUTH_REQUIRED")
        return ApiMaterialBalanceGateway(token, client)
    return DirectMaterialBalanceGateway()


def adapt_result_for_plotters(result: JsonDict) -> dict[str, Any]:
    closure_rows = result.get("tables", {}).get("closure", {}).get("rows") or [{"Element": row["symbol"], "In_t": row["input_t"], "Out_t": row["output_t"], "Closure_pct": row["closure_pct"], "Delta_t": row["delta_t"], "Status": row["status"]} for row in result.get("closure") or []]
    return {"closure_table": pd.DataFrame(closure_rows), "inputs": _streams_to_nested(result.get("input_streams") or []), "outputs": _streams_to_nested(result.get("output_streams") or [])}


def _streams_to_nested(streams: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    nested: dict[str, dict[str, float]] = {}
    for stream in streams:
        label = str(stream.get("label") or "")
        for element in stream.get("elements") or []:
            symbol = str(element.get("symbol") or "")
            nested.setdefault(symbol, {})[label] = float(element.get("mass_t") or 0.0)
    return nested


def _checksum(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def _thresholds(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    raw = cfg.get("closure_thresholds") or {}
    good = raw.get("good") or [95, 105]
    warning = raw.get("warning") or [85, 115]
    return {"good": {"minimum": float(good[0]), "maximum": float(good[1])}, "warning": {"minimum": float(warning[0]), "maximum": float(warning[1])}}


def _approved_dpr_source_fields() -> list[dict[str, Any]]:
    fields = ["total_hot_metal_mt", "slag_generation_mt", "coke_total_mt", "nut_coke_total_mt", "pci_total_mt", "ore_total_mt", "sinter_total_mt", "pellet_total_mt", "flux_total_mt"]
    return [{"source_field_id": field, "label": field.replace("_", " ").title(), "data_type": "number", "unit": "t", "aggregation_policy": "latest_non_null"} for field in fields]


__all__ = ["ApiMaterialBalanceGateway", "DirectMaterialBalanceGateway", "MaterialBalanceGateway", "adapt_result_for_plotters", "get_material_balance_gateway"]