"""Backend service for Material Balance compute workflows."""

from __future__ import annotations

import importlib
import importlib.util
from datetime import date, datetime, timezone
import logging
import sys
from typing import Any

import pandas as pd

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.services.compute_artifact_service import ComputeArtifactService
from furnace_data.runtime_paths import get_repo_root

log = logging.getLogger(__name__)


def _ensure_src_package(package_name: str) -> None:
    loaded = sys.modules.get(package_name)
    if loaded is not None and getattr(loaded, "__path__", None) is not None:
        return
    try:
        if importlib.util.find_spec(package_name) is not None:
            return
    except (ImportError, ValueError):
        pass

    src_path = get_repo_root() / "src"
    package_dir = src_path / package_name
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        raise ImportError(f"Source package not found: {package_name}")

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Source package could not be loaded: {package_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


def _ensure_src_path() -> None:
    _ensure_src_package("utils")


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
    """Wrap safe Material Balance calculations for API use."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        artifact_service: ComputeArtifactService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)

    def config(self) -> dict[str, Any]:
        warnings: list[dict[str, Any]] = []
        mappings: dict[str, Any] = {}
        version = None
        try:
            _ensure_src_package("utils")
            from utils.material_balance.dpr_mapping import load_full_config

            cfg = load_full_config()
            mappings = {
                "dpr_mapping": cfg.get("dpr_mapping", {}),
                "closure_thresholds": cfg.get("closure_thresholds", {}),
            }
            version = str(cfg.get("version") or "") or None
        except Exception as exc:
            warnings.append(
                _warning(
                    "MATERIAL_BALANCE_CONFIG_INVALID",
                    "Material Balance config could not be loaded.",
                    {"reason": str(exc)},
                )
            )
        return {
            "mappings": mappings,
            "available_sources": ["static_dataset", "input_data"],
            "defaults": {
                "timezone": "Asia/Kolkata",
                "rm_lag_hours": 0,
                "blast_lag_hours": 0,
                "dust_catcher_t": 0.0,
                "max_json_rows": self.settings.compute_max_json_rows,
            },
            "version": version,
            "warnings": warnings,
        }

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        source = str(payload.get("source") or "static_dataset").strip()
        if source not in {"static_dataset", "input_data"}:
            errors.append(
                _warning(
                    "MATERIAL_BALANCE_INPUT_INVALID",
                    "Unsupported Material Balance source.",
                    {"source": source},
                )
            )
        if source == "input_data" and not isinstance(payload.get("input_data"), dict):
            errors.append(
                _warning(
                    "MATERIAL_BALANCE_INPUT_INVALID",
                    "input_data is required when source is input_data.",
                )
            )
        if payload.get("date") is None and source == "static_dataset":
            warnings.append(
                _warning(
                    "MATERIAL_BALANCE_DATE_DEFAULTED",
                    "No date supplied; service will use yesterday for static dataset runs.",
                )
            )
        return {"valid": not errors, "errors": errors, "warnings": warnings}

    def run(self, payload: dict[str, Any], *, route_prefix: str = "/material-balance") -> dict[str, Any]:
        validation = self.validate(payload)
        if validation["errors"]:
            raise ApiError(
                "MATERIAL_BALANCE_INPUT_INVALID",
                "Material Balance request is invalid.",
                status_code=422,
                details={"errors": validation["errors"]},
            )

        source = str(payload.get("source") or "static_dataset").strip()
        if source == "input_data":
            return self._run_input_data(payload, route_prefix=route_prefix)
        return self._run_static_dataset(payload, route_prefix=route_prefix)

    def _run_input_data(self, payload: dict[str, Any], *, route_prefix: str) -> dict[str, Any]:
        data = payload.get("input_data") or {}
        inputs = data.get("inputs") or {}
        outputs = data.get("outputs") or {}
        rows: list[dict[str, Any]] = []
        elements = sorted(set(inputs) | set(outputs))
        for element in elements:
            in_t = float(inputs.get(element) or 0.0)
            out_t = float(outputs.get(element) or 0.0)
            rows.append(
                {
                    "Element": element,
                    "In_t": round(in_t, 2),
                    "Out_t": round(out_t, 2),
                    "Closure_pct": round((out_t / in_t * 100.0), 1) if in_t > 0 else None,
                    "Delta_t": round(out_t - in_t, 2),
                }
            )
        total_in = sum(float(row["In_t"] or 0.0) for row in rows)
        total_out = sum(float(row["Out_t"] or 0.0) for row in rows)
        return self._build_response(
            rows=rows,
            summary={
                "source": "input_data",
                "total_in_t": round(total_in, 2),
                "total_out_t": round(total_out, 2),
            },
            kpis={
                "overall_closure_pct": round(total_out / total_in * 100.0, 2)
                if total_in > 0
                else None,
                "element_count": len(rows),
            },
            warnings=payload.get("_warnings") or [],
            export=bool(payload.get("export")),
            route_prefix=route_prefix,
        )

    def _run_static_dataset(self, payload: dict[str, Any], *, route_prefix: str) -> dict[str, Any]:
        try:
            _ensure_src_package("utils")
            from utils.material_balance.compute import run_full_balance
        except Exception as exc:
            raise ApiError(
                "MATERIAL_BALANCE_INTERNAL_ERROR",
                "Material Balance engine could not be imported.",
                status_code=500,
            ) from exc

        options = payload.get("options") or {}
        run_day = payload.get("date")
        if isinstance(run_day, str):
            run_day = date.fromisoformat(run_day)
        if run_day is None:
            run_day = date.today()

        try:
            result = run_full_balance(
                run_day,
                rm_lag_hours=int(options.get("rm_lag_hours") or 0),
                blast_lag_hours=int(options.get("blast_lag_hours") or 0),
                dust_catcher_t=float(options.get("dust_catcher_t") or 0.0),
            )
        except Exception as exc:
            raise ApiError(
                "MATERIAL_BALANCE_CALCULATION_FAILED",
                "Material Balance calculation failed.",
                status_code=500,
            ) from exc

        rows = result.closure_table.where(pd.notna(result.closure_table), None).to_dict(
            orient="records"
        )
        total_in = float(result.closure_table["In_t"].sum()) if "In_t" in result.closure_table else 0.0
        total_out = float(result.closure_table["Out_t"].sum()) if "Out_t" in result.closure_table else 0.0
        warnings = [
            _warning("MATERIAL_BALANCE_WARNING", str(message))
            for message in result.warnings
        ]
        return self._build_response(
            rows=rows,
            summary={
                "source": "static_dataset",
                "date": result.day.isoformat(),
                "used_dpr": result.used_dpr,
                "n_rm_rows": result.n_rm_rows,
                "total_in_t": round(total_in, 2),
                "total_out_t": round(total_out, 2),
            },
            kpis={
                "overall_closure_pct": round(total_out / total_in * 100.0, 2)
                if total_in > 0
                else None,
                "hm_mass_t": result.gas_phase.get("hm_mass_t"),
                "slag_mass_t": result.gas_phase.get("slag_mass_t"),
            },
            warnings=warnings,
            export=bool(payload.get("export")),
            route_prefix=route_prefix,
        )

    def _build_response(
        self,
        *,
        rows: list[dict[str, Any]],
        summary: dict[str, Any],
        kpis: dict[str, Any],
        warnings: list[dict[str, Any]],
        export: bool,
        route_prefix: str,
    ) -> dict[str, Any]:
        artifacts = []
        table = table_data(rows, self.settings.compute_max_json_rows)
        if export or table["truncated"]:
            metadata = self.artifacts.create_csv_artifact(
                workflow="material_balance",
                filename_prefix="material_balance_result",
                rows=rows,
            )
            artifacts.append(self.artifacts.response(metadata, route_prefix))
        return {
            "summary": summary,
            "kpis": kpis,
            "tables": {"closure": table},
            "charts": {
                "closure": {
                    "series": [
                        {
                            "name": "Input tonnes",
                            "x": [row["Element"] for row in rows],
                            "y": [row["In_t"] for row in rows],
                            "unit": "t",
                            "metadata": {},
                        },
                        {
                            "name": "Output tonnes",
                            "x": [row["Element"] for row in rows],
                            "y": [row["Out_t"] for row in rows],
                            "unit": "t",
                            "metadata": {},
                        },
                    ]
                }
            },
            "warnings": warnings,
            "artifacts": artifacts,
            "computed_at": datetime.now(timezone.utc),
        }
