"""Canonical V-Sense catalog with public IDs and trusted feature mappings."""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any

from furnace_data.config import load_config


CATALOG_VERSION = "vsense-catalog-v1"
DISPLAY_TIMEZONE = "Asia/Kolkata"
ALGORITHM_VERSION = "legacy_v1"
DEFAULT_ITERATION_BUDGET_ID = "standard"

_OPTIMIZATION_ID_BY_LABEL = {
    "Eta CO": "eta_co",
    "Production Rate": "production_rate",
    "UnitCost": "unit_cost",
    "Unit Cost": "unit_cost",
}

_OPTIMIZATION_LABELS = {
    "eta_co": "Eta CO",
    "production_rate": "Production Rate",
    "unit_cost": "Unit Cost",
}

_TARGET_METADATA = {
    "eta_co": {
        "id": "eta_co",
        "label": "Eta CO",
        "unit": "%",
        "direction": "maximize",
        "precision": 2,
        "feature_name": "FURNACETOPGASANALYSISCO2ETACO",
    },
    "production_rate": {
        "id": "production_rate",
        "label": "Production Rate",
        "unit": "t/h",
        "direction": "maximize",
        "precision": 2,
        "feature_name": "PRODUCTIONTONNESPERHR",
    },
    "unit_cost": {
        "id": "unit_cost",
        "label": "Unit Cost",
        "unit": "lakhs/THM",
        "direction": "minimize",
        "precision": 3,
        "feature_name": "UNITCOST LAKHS/THM",
    },
}

_CONTROL_PARAMETERS = (
    {
        "id": "hot_blast_pressure_bar",
        "label": "Hot Blast Pressure",
        "feature_name": "HOT BLAST PRESSUREBAR",
        "unit": "bar",
        "precision": 3,
        "approved_min": 2.2,
        "approved_max": 2.9,
        "default_value": 2.65,
    },
    {
        "id": "top_pressure_bar",
        "label": "Top Pressure",
        "feature_name": "TOPPRESSUREBAR",
        "unit": "bar",
        "precision": 3,
        "approved_min": 0.8,
        "approved_max": 1.6,
        "default_value": 1.25,
    },
    {
        "id": "hot_blast_temperature_c",
        "label": "Hot Blast Temperature",
        "feature_name": "HOT BLAST TEMP.OC",
        "unit": "degC",
        "precision": 1,
        "approved_min": 900.0,
        "approved_max": 1150.0,
        "default_value": 1040.0,
    },
    {
        "id": "steam_injection_kg_h",
        "label": "Steam Injection",
        "feature_name": "STEAMKGS/HR.",
        "unit": "kg/h",
        "precision": 1,
        "approved_min": 0.0,
        "approved_max": 5000.0,
        "default_value": 2200.0,
    },
    {
        "id": "hot_blast_volume_nm3_h",
        "label": "Hot Blast Volume",
        "feature_name": "HOT BLAST VOLUMENM3/HR.",
        "unit": "Nm3/h",
        "precision": 1,
        "approved_min": 50000.0,
        "approved_max": 160000.0,
        "default_value": 118000.0,
    },
    {
        "id": "oxygen_enrichment_pct",
        "label": "Oxygen Enrichment",
        "feature_name": "O2 ENRICHMENT %",
        "unit": "%",
        "precision": 2,
        "approved_min": 0.0,
        "approved_max": 10.0,
        "default_value": 3.5,
    },
    {
        "id": "pci_kg_thm",
        "label": "PCI",
        "feature_name": "PCI_KG/THM",
        "unit": "kg/THM",
        "precision": 1,
        "approved_min": 0.0,
        "approved_max": 250.0,
        "default_value": 145.0,
    },
)

_INPUT_UNIT_HINTS = (
    ("TEMP", "degC"),
    ("ASH", "%"),
    ("MOIST", "%"),
    ("PCT", "%"),
    ("%", "%"),
    ("BASICITY", "ratio"),
    ("CALC_MT", "MT"),
    ("MT", "MT"),
    ("HRS", "h"),
    ("ANGLE", "deg"),
    ("STOCKROD", "m"),
)


def load_vsense_catalog(
    *,
    context_ttl_seconds: int = 1800,
    max_concurrent_runs: int = 1,
    lambda_min: float = 0.0,
    lambda_max: float = 0.5,
    llm_review_available: bool = False,
    advanced_diagnostics_available: bool = False,
    display_timezone: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-native V-Sense catalog with deployment limits."""

    catalog = copy.deepcopy(_base_catalog())
    catalog["display_timezone"] = str(display_timezone or DISPLAY_TIMEZONE)
    catalog["capabilities"]["llm_review_available"] = bool(llm_review_available)
    catalog["capabilities"]["advanced_diagnostics_available"] = bool(
        advanced_diagnostics_available
    )
    catalog["limits"]["context_ttl_seconds"] = int(context_ttl_seconds)
    catalog["limits"]["max_concurrent_runs"] = int(max_concurrent_runs)
    catalog["limits"]["lambda_min"] = float(lambda_min)
    catalog["limits"]["lambda_max"] = float(lambda_max)
    _validate_catalog(catalog)
    return catalog


def optimization_by_id(catalog: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return optimization definitions keyed by public optimization ID."""

    data = catalog or load_vsense_catalog()
    return {str(item["id"]): dict(item) for item in data["optimization_types"]}


def parameter_by_id(catalog: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return parameter definitions keyed by public parameter ID."""

    data = catalog or load_vsense_catalog()
    return {str(item["id"]): dict(item) for item in data["parameters"]}


def control_parameter_by_feature() -> dict[str, dict[str, Any]]:
    """Return control metadata keyed by trusted internal feature name."""

    return {str(item["feature_name"]): dict(item) for item in _CONTROL_PARAMETERS}


def public_id_for_feature(feature_name: str) -> str:
    """Return a stable public ID for a trusted internal feature name."""

    for item in _CONTROL_PARAMETERS:
        if item["feature_name"] == feature_name:
            return str(item["id"])
    for target in _TARGET_METADATA.values():
        if target["feature_name"] == feature_name:
            return str(target["id"])
    return _slug(feature_name)


def feature_for_parameter_id(parameter_id: str) -> str | None:
    """Return the trusted internal feature name for a public parameter ID."""

    for item in _CONTROL_PARAMETERS:
        if item["id"] == parameter_id:
            return str(item["feature_name"])
    for target in _TARGET_METADATA.values():
        if target["id"] == parameter_id:
            return str(target["feature_name"])
    return _input_feature_by_id().get(str(parameter_id))


def default_control_profile_parameters(optimization_type_id: str) -> list[dict[str, Any]]:
    """Return a complete plant-default profile snapshot for an optimization type."""

    opt = optimization_by_id().get(str(optimization_type_id))
    if opt is None:
        raise KeyError(optimization_type_id)
    params = parameter_by_id()
    profile: list[dict[str, Any]] = []
    for parameter_id in opt["control_parameter_ids"]:
        item = params[str(parameter_id)]
        profile.append(
            {
                "parameter_id": item["id"],
                "mode": "optimize",
                "lower_bound": item["approved_min"],
                "upper_bound": item["approved_max"],
                "fixed_value": None,
            }
        )
    return profile


def target_for_optimization(optimization_type_id: str) -> dict[str, Any]:
    """Return target metadata for an optimization type."""

    target = _TARGET_METADATA.get(str(optimization_type_id))
    if target is None:
        raise KeyError(optimization_type_id)
    return dict(target)


@lru_cache(maxsize=1)
def _base_catalog() -> dict[str, Any]:
    cfg = load_config("setting_vsense.yml")
    optimization_cfg = dict(cfg.get("Optimisation") or {})
    parameters = _parameter_definitions(optimization_cfg)
    optimization_types = _optimization_definitions(optimization_cfg)
    catalog = {
        "catalog_version": CATALOG_VERSION,
        "display_timezone": DISPLAY_TIMEZONE,
        "advisory_only": True,
        "optimization_types": optimization_types,
        "parameters": parameters,
        "algorithm_versions": [
            {
                "id": ALGORITHM_VERSION,
                "label": "Validated legacy-compatible optimiser",
                "status": "active",
            }
        ],
        "iteration_budgets": [
            {
                "id": DEFAULT_ITERATION_BUDGET_ID,
                "label": "Standard",
                "max_iterations": 20,
            }
        ],
        "capabilities": {
            "llm_review_available": False,
            "advanced_diagnostics_available": False,
            "historical_context_available": True,
            "run_cancellation_available": True,
        },
        "limits": {
            "context_ttl_seconds": 1800,
            "max_input_overrides": 100,
            "max_concurrent_runs": 1,
            "lambda_min": 0.0,
            "lambda_max": 0.5,
        },
    }
    _validate_catalog(catalog)
    return catalog


@lru_cache(maxsize=1)
def _input_feature_by_id() -> dict[str, str]:
    cfg = load_config("setting_vsense.yml")
    features: dict[str, str] = {}
    for optimization in (cfg.get("Optimisation") or {}).values():
        for params in (optimization.get("input_params") or {}).values():
            for feature in params or []:
                features.setdefault(_slug(str(feature)), str(feature))
    return features


def _optimization_definitions(optimization_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    control_by_feature = control_parameter_by_feature()
    for label, raw in optimization_cfg.items():
        optimization_id = _OPTIMIZATION_ID_BY_LABEL.get(str(label))
        if not optimization_id:
            continue
        target = _TARGET_METADATA[optimization_id]
        control_parameter_ids = [
            control_by_feature[str(feature)]["id"]
            for feature in raw.get("control_params") or []
            if str(feature) in control_by_feature
        ]
        input_groups = _input_groups(raw.get("input_params") or {})
        definitions.append(
            {
                "id": optimization_id,
                "label": _OPTIMIZATION_LABELS[optimization_id],
                "target": {
                    key: value
                    for key, value in target.items()
                    if key != "feature_name"
                },
                "control_parameter_ids": list(dict.fromkeys(control_parameter_ids)),
                "input_groups": input_groups,
                "impact_target_ids": [
                    item["id"]
                    for item in _TARGET_METADATA.values()
                    if item["id"] != target["id"]
                ],
                "default_algorithm_version": ALGORITHM_VERSION,
            }
        )
    return definitions


def _parameter_definitions(optimization_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for item in _CONTROL_PARAMETERS:
        parameters.append(
            {
                "id": item["id"],
                "label": item["label"],
                "group_id": "control",
                "role": "control",
                "value_type": "number",
                "unit": item["unit"],
                "precision": item["precision"],
                "nullable": False,
                "override_allowed": True,
                "approved_min": item["approved_min"],
                "approved_max": item["approved_max"],
            }
        )
    for target in _TARGET_METADATA.values():
        parameters.append(
            {
                "id": target["id"],
                "label": target["label"],
                "group_id": "target",
                "role": "target",
                "value_type": "number",
                "unit": target["unit"],
                "precision": target["precision"],
                "nullable": False,
                "override_allowed": False,
                "approved_min": None,
                "approved_max": None,
            }
        )
    seen: set[str] = {str(item["id"]) for item in parameters}
    for raw in optimization_cfg.values():
        for group_name, feature_names in (raw.get("input_params") or {}).items():
            for feature_name in feature_names or []:
                parameter_id = _slug(str(feature_name))
                if parameter_id in seen:
                    continue
                seen.add(parameter_id)
                parameters.append(
                    {
                        "id": parameter_id,
                        "label": _label_from_feature(str(feature_name)),
                        "group_id": _slug(str(group_name)),
                        "role": "input",
                        "value_type": "number",
                        "unit": _unit_for_feature(str(feature_name)),
                        "precision": 2,
                        "nullable": True,
                        "override_allowed": True,
                        "approved_min": None,
                        "approved_max": None,
                    }
                )
    return parameters


def _input_groups(input_params: dict[str, list[str]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group_name, features in input_params.items():
        groups.append(
            {
                "id": _slug(str(group_name)),
                "label": str(group_name),
                "parameter_ids": [_slug(str(feature)) for feature in features or []],
            }
        )
    return groups


def _slug(value: str) -> str:
    normalized = value.strip().lower().replace("%", "pct")
    normalized = normalized.replace("/", "_").replace(".", "_")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "parameter"


def _label_from_feature(value: str) -> str:
    return (
        value.replace("_", " ")
        .replace("PCT", "%")
        .replace("CALC MT", "Calc MT")
        .title()
    )


def _unit_for_feature(value: str) -> str | None:
    upper = value.upper()
    for token, unit in _INPUT_UNIT_HINTS:
        if token in upper:
            return unit
    return None


def _validate_catalog(catalog: dict[str, Any]) -> None:
    parameter_ids = [str(item["id"]) for item in catalog["parameters"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("V-Sense catalog parameter IDs must be unique.")
    optimization_ids = [str(item["id"]) for item in catalog["optimization_types"]]
    if len(optimization_ids) != len(set(optimization_ids)):
        raise ValueError("V-Sense catalog optimization IDs must be unique.")
    params = set(parameter_ids)
    algorithm_ids = {str(item["id"]) for item in catalog["algorithm_versions"]}
    for opt in catalog["optimization_types"]:
        if opt["target"]["id"] not in params:
            raise ValueError(f"Unknown target parameter {opt['target']['id']}.")
        if opt["target"]["direction"] not in {"maximize", "minimize"}:
            raise ValueError("Optimization target direction must be explicit.")
        if opt["default_algorithm_version"] not in algorithm_ids:
            raise ValueError("Unknown default algorithm version.")
        for parameter_id in opt["control_parameter_ids"]:
            if parameter_id not in params:
                raise ValueError(f"Unknown control parameter {parameter_id}.")
        for target_id in opt["impact_target_ids"]:
            if target_id not in params:
                raise ValueError(f"Unknown impact target {target_id}.")
        for group in opt["input_groups"]:
            for parameter_id in group["parameter_ids"]:
                if parameter_id not in params:
                    raise ValueError(f"Unknown input parameter {parameter_id}.")

