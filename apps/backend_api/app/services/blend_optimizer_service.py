"""Backend service for Blend Optimizer API workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.services.compute_artifact_service import ComputeArtifactService
from app.services.material_balance_service import _ensure_src_package, _warning, table_data
from app.services.model_registry_service import ModelRegistryService


class BlendOptimizerService:
    """Prepare context and run bounded blend optimization."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        artifact_service: ComputeArtifactService | None = None,
        model_registry: ModelRegistryService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)
        self.model_registry = model_registry or ModelRegistryService(self.settings)

    def context(self) -> dict[str, Any]:
        warnings: list[dict[str, Any]] = []
        materials = [
            {
                "material_id": "ore_a",
                "name": "Ore A",
                "available": True,
                "min_percent": 0.0,
                "max_percent": 100.0,
                "properties": {"fe_t_pct": 58.0, "sio2_pct": 5.0},
                "cost": 1000.0,
            },
            {
                "material_id": "ore_b",
                "name": "Ore B",
                "available": True,
                "min_percent": 0.0,
                "max_percent": 100.0,
                "properties": {"fe_t_pct": 62.0, "sio2_pct": 4.0},
                "cost": 1200.0,
            },
        ]
        try:
            _ensure_src_package("data")
            from data.bmo import EvonithBmoContextProvider

            provider = EvonithBmoContextProvider()
            ore_inputs, provider_warnings = provider.build_ore_inputs()
            if ore_inputs:
                materials = [
                    {
                        "material_id": ore.ore_id,
                        "name": ore.display_name,
                        "available": True,
                        "min_percent": ore.min_share_pct,
                        "max_percent": ore.max_share_pct,
                        "properties": {
                            "fe_t_pct": ore.chemistry.fe_t_pct,
                            "feo_pct": ore.chemistry.feo_pct,
                            "sio2_pct": ore.chemistry.sio2_pct,
                            "al2o3_pct": ore.chemistry.al2o3_pct,
                        },
                        "cost": ore.price_rs_per_mt,
                    }
                    for ore in ore_inputs
                ]
            warnings.extend(
                _warning("BLEND_OPTIMIZER_CONTEXT_WARNING", str(message))
                for message in provider_warnings
            )
        except Exception as exc:
            warnings.append(
                _warning(
                    "BLEND_OPTIMIZER_CONTEXT_UNAVAILABLE",
                    "Live Blend Optimizer context unavailable; using safe defaults.",
                    {"reason": str(exc)[:160]},
                )
            )
        return {
            "materials": materials[: self.settings.blend_optimizer_max_candidates],
            "constraints": {
                "target_total_qty_mt": 100.0,
                "max_candidates": self.settings.blend_optimizer_max_candidates,
                "max_iterations": self.settings.blend_optimizer_max_iterations,
            },
            "defaults": {
                "objective": "min_cost",
                "include_predictions": self.settings.blend_optimizer_enable_model_predictions,
            },
            "models": self.model_registry.list_models(),
            "warnings": warnings,
        }

    def list_models(self) -> list[dict[str, Any]]:
        return self.model_registry.list_models()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.model_registry.predict(
            str(payload.get("model_name") or ""),
            payload.get("features") or {},
        )

    def optimize(self, payload: dict[str, Any], *, route_prefix: str = "/blend-optimizer") -> dict[str, Any]:
        materials = payload.get("materials") or []
        if not isinstance(materials, list) or not materials:
            raise ApiError(
                "BLEND_OPTIMIZER_INPUT_INVALID",
                "At least one material is required.",
                status_code=422,
            )
        max_candidates = min(
            self.settings.blend_optimizer_max_candidates,
            max(1, int((payload.get("options") or {}).get("max_candidates") or 5)),
        )
        target_total = float((payload.get("constraints") or {}).get("target_total_qty_mt") or 100.0)
        available = [item for item in materials if item.get("available", True)]
        if not available:
            raise ApiError(
                "BLEND_OPTIMIZER_CONSTRAINT_INVALID",
                "No available materials were supplied.",
                status_code=422,
            )

        candidates = self._build_candidates(available, target_total, max_candidates)
        rows = [
            {
                "rank": candidate["rank"],
                "feasible": candidate["feasible"],
                "objective_cost": candidate["metrics"]["objective_cost"],
                **{
                    f"material_{key}": value
                    for key, value in candidate["materials"].items()
                },
            }
            for candidate in candidates
        ]
        artifacts = []
        table = table_data(rows, self.settings.compute_max_json_rows)
        if payload.get("export") or table["truncated"]:
            metadata = self.artifacts.create_csv_artifact(
                workflow="blend_optimizer",
                filename_prefix="blend_optimizer_candidates",
                rows=rows,
            )
            artifacts.append(self.artifacts.response(metadata, route_prefix))
        warnings = []
        if payload.get("include_predictions", True) and not self.settings.blend_optimizer_enable_model_predictions:
            warnings.append(
                _warning(
                    "BLEND_OPTIMIZER_MODEL_PREDICTIONS_DISABLED",
                    "Model predictions are disabled by configuration.",
                )
            )
        return {
            "candidates": candidates,
            "best_candidate": candidates[0] if candidates else None,
            "summary": {
                "candidate_count": len(candidates),
                "target_total_qty_mt": target_total,
                "objective": payload.get("objective") or "min_cost",
            },
            "tables": {"candidates": table},
            "charts": {},
            "warnings": warnings,
            "artifacts": artifacts,
            "computed_at": datetime.now(timezone.utc),
        }

    def _build_candidates(
        self,
        materials: list[dict[str, Any]],
        target_total: float,
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        cleaned = []
        for item in materials:
            material_id = str(item.get("material_id") or "").strip()
            if not material_id:
                continue
            min_pct = max(0.0, float(item.get("min_percent") or 0.0))
            max_pct = min(100.0, float(item.get("max_percent") if item.get("max_percent") is not None else 100.0))
            if min_pct > max_pct:
                raise ApiError(
                    "BLEND_OPTIMIZER_CONSTRAINT_INVALID",
                    "Material min_percent cannot exceed max_percent.",
                    status_code=422,
                    details={"material_id": material_id},
                )
            cleaned.append(
                {
                    "material_id": material_id,
                    "min_pct": min_pct,
                    "max_pct": max_pct,
                    "cost": float(item.get("cost") or 0.0),
                    "fe_t_pct": float((item.get("properties") or {}).get("fe_t_pct") or 0.0),
                }
            )
        if not cleaned:
            raise ApiError("BLEND_OPTIMIZER_INPUT_INVALID", "No valid materials supplied.", 422)

        base_share = 100.0 / len(cleaned)
        shares = {
            item["material_id"]: min(item["max_pct"], max(item["min_pct"], base_share))
            for item in cleaned
        }
        total_share = sum(shares.values())
        if total_share <= 0:
            raise ApiError("BLEND_OPTIMIZER_CONSTRAINT_INVALID", "Total share is zero.", 422)
        shares = {key: value * 100.0 / total_share for key, value in shares.items()}

        candidates: list[dict[str, Any]] = []
        ranked = sorted(cleaned, key=lambda item: item["cost"])
        for rank in range(1, max_candidates + 1):
            candidate_shares = dict(shares)
            if rank > 1 and len(ranked) > 1:
                cheap = ranked[0]
                expensive = ranked[-1]
                shift = min(5.0 * (rank - 1), candidate_shares.get(expensive["material_id"], 0.0))
                candidate_shares[expensive["material_id"]] -= shift
                candidate_shares[cheap["material_id"]] += shift
            quantities = {
                key: round(target_total * pct / 100.0, 3)
                for key, pct in candidate_shares.items()
            }
            cost = sum(
                quantities[item["material_id"]] * item["cost"]
                for item in cleaned
            )
            fe_pct = sum(
                candidate_shares[item["material_id"]] * item["fe_t_pct"] / 100.0
                for item in cleaned
            )
            candidates.append(
                {
                    "rank": rank,
                    "materials": quantities,
                    "metrics": {
                        "objective_cost": round(cost, 2),
                        "weighted_fe_pct": round(fe_pct, 3),
                        "total_qty_mt": target_total,
                    },
                    "feasible": True,
                    "warnings": [],
                }
            )
        return candidates
