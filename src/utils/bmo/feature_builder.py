"""Feature construction for BMO fuel-cost inference.

This module translates candidate ore quantities, chemistry, process context,
and historical lag data into the exact feature layout expected by the BMO V4
fuel-cost model artifacts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.optimization_runtime import (
    FeatureBuildResult,
)
from domain.optimization_runtime import (
    normalize_feature_name as _normalize_feature_name,
)
from utils.bmo.calculations import compute_dry_weight_mt, compute_fe_contribution_mt
from utils.bmo.types import OreInput


@dataclass
class PreBuiltFeatureContext:
    """
    Cache the static portion of the BMO model feature vector for DE inner loops.

    The total-cost optimizer evaluates hundreds of candidate blends. The vast
    majority of model feature columns (process context, history lags, fuel/flux
    chemistry, defaults) do not vary across candidates and only the ~30 burden-
    derived columns change. This dataclass holds the precomputed scaler-order
    template vector plus a mapping from dynamic payload keys to the template
    column indexes they overwrite so the inner loop can mutate-in-place rather
    than re-resolve every column on every iteration.

    Args:
         - expected_features: list[str] - Full scaler-ordered feature column names.
         - template_x: "np.ndarray" - 1xN baseline numpy array of feature values.
         - dynamic_col_indexes: dict[str, tuple[int, ...]] - Payload key -> column indexes to update.
         - dynamic_field_names: frozenset[str] - Set of payload keys that drive dynamic columns.
         - selected_features: list[str] | None - Optional post-scaler selected feature names.
         - selected_indexes: "np.ndarray | None" - Pre-resolved selected column indexes (if any).
         - ores_snapshot: list[OreInput] - Ores used to compute candidate dynamic payloads.
         - ore_display_name_by_id: dict[str, str] - Display names keyed by ore id.
         - static_payload: dict[str, float] - Static (process-context) payload reused for fallbacks.
         - default_values: dict[str, float] - Defaults used when building the template.
         - history_present: bool - Whether history data was available at build time.

    Returns:
         - return PreBuiltFeatureContext - Cached state used by predict_with_prebuilt.
    """

    expected_features: list[str]
    template_x: "Any"  # numpy ndarray (1xN). Typed as Any to avoid numpy import at module load
    dynamic_col_indexes: dict[str, tuple[int, ...]]
    dynamic_field_names: frozenset[str]
    selected_features: list[str] | None
    selected_indexes: "Any"  # numpy ndarray of selected col positions or None
    ores_snapshot: list[OreInput]
    ore_display_name_by_id: dict[str, str]
    static_payload: dict[str, float]
    default_values: dict[str, float]
    history_present: bool
    hot_metal_target_mt: float | None = None


def normalize_feature_name(name: str) -> str:
    """
    Normalize a raw feature name using the shared runtime convention.

    The BMO payload mixes training names, process names, and normalized aliases.
    Reusing the shared normalizer keeps lookups consistent with the runtime
    feature builder and avoids duplicate alias rules.

    Args:
         - name: str - Feature name from model metadata, payload, or history.

    Returns:
         - return str - Lowercase underscore-normalized feature key.
    """

    return _normalize_feature_name(name)


_BMO_LAG_PATTERN = re.compile(r"(.+)_lag(\d+)(?:_\(([^)]+)\))?$")


# _TRAINING_ALIASES is the SINGLE source of truth for BMO fuel-cost inference.
# A conceptually similar (but unrelated subsystem) mapping lives in
# src/config/setting_ds_dv.yml::field_mapping and is consumed by the Data
# Explorer page only. The two do NOT cross-reference each other: changes to
# this dict must be made here and exercised via the BMO predict path;
# changes to the yml mapping only affect Data Explorer column rendering.
_TRAINING_ALIASES: dict[str, str] = {
    "ACTUALKG/THM.": "PCI_KG/THM",
    "PCI_2_ASH%": "PCI_ASH%",
    "PCI_2_FC%": "PCI_FC%",
    "PCI_2_IM%": "PCI_IM%",
    "PCI_2_VM%": "PCI_VM%",
    "PCI_2_CALC_MT": "PCI_CALC_MT",
    "SINTER_SP_02_AL2O3%": "SINTER_AL2O3%",
    "SINTER_SP_02_CAO%": "SINTER_CAO%",
    "SINTER_SP_02_COLD_STRENGTH_AI": "SINTER_COLD_STRENGTH_AI",
    "SINTER_SP_02_COLD_STRENGTH_TI": "SINTER_COLD_STRENGTH_TI",
    "SINTER_SP_02_FE(T)%": "SINTER_FE(T)%",
    "SINTER_SP_02_FEO%": "SINTER_FEO%",
    "SINTER_SP_02_HOT_STRENGTH_RDI": "SINTER_HOT_STRENGTH_RDI",
    "SINTER_SP_02_HOT_STRENGTH_RI": "SINTER_HOT_STRENGTH_RI",
    "SINTER_SP_02_K2O%": "SINTER_K2O%",
    "SINTER_SP_02_MGO%": "SINTER_MGO%",
    "SINTER_SP_02_MNO%": "SINTER_MNO%",
    "SINTER_SP_02_NA2O%": "SINTER_NA2O%",
    "SINTER_SP_02_P%": "SINTER_P%",
    "SINTER_SP_02_SIO2%": "SINTER_SIO2%",
    "SINTER_SP_02_BASICITY": "SINTER_BASICITY",
    "SINTER_SP_02_TIO2%": "SINTER_TIO2%",
    "LLOYDS_PELLET_PCT_CAO": "PELLET_PCT_CAO",
    "LLOYDS_PELLET_PCT_AL2O3": "PELLET_PCT_AL2O3",
    "LLOYDS_PELLET_PCT_FE2O3": "PELLET_PCT_FE2O3",
    "LLOYDS_PELLET_PCT_K2O": "PELLET_PCT_K2O",
    "LLOYDS_PELLET_PCT_LOI": "PELLET_PCT_LOI",
    "LLOYDS_PELLET_PCT_MGO": "PELLET_PCT_MGO",
    "LLOYDS_PELLET_PCT_MNO": "PELLET_PCT_MNO",
    "LLOYDS_PELLET_PCT_NA2O": "PELLET_PCT_NA2O",
    "LLOYDS_PELLET_PCT_P": "PELLET_PCT_P",
    "LLOYDS_PELLET_PCT_SIO2": "PELLET_PCT_SIO2",
    "LLOYDS_PELLET_PCT_TIO2": "PELLET_PCT_TIO2",
    "LLOYDS_PELLET_PCT_TM": "PELLET_PCT_TM",
    "BELLY - 15162_TEMP. OCAT-07DEG. OC": "BELLY_TEMP_A",
    "BELLY - 15162_TEMP. OC.1BT-12DEG. OC": "BELLY_TEMP_B",
    "BELLY - 15162_TEMP. OC.2CT-17DEG. OC": "BELLY_TEMP_C",
    "BELLY - 15162_TEMP. OC.3DT-03DEG. OC": "BELLY_TEMP_D",
    "BOSH - 12975_TEMP. OCAT-07DEG. OC": "BOSH_TEMP_A",
    "BOSH - 12975_TEMP. OC.1BT-12DEG. OC": "BOSH_TEMP_B",
    "BOSH - 12975_TEMP. OC.2CT-17DEG. OC": "BOSH_TEMP_C",
    "BOSH - 12975_TEMP. OC.3DT-03DEG. OC": "BOSH_TEMP_D",
    "HEARTH PAD CENTER_TEMP. OCA4.3 MTR.DEG. OC": "HEARTH_TEMP_A",
    "HEARTH PAD CENTER_TEMP. OC.1B5.4 MTR.DEG. OC": "HEARTH_TEMP_B",
    "HEARTH PAD CENTER_TEMP. OC.2C5.7 MTR.DEG. OC": "HEARTH_TEMP_C",
    "HEARTH PAD CENTER_TEMP. OC.3D6.1 MTR.DEG. OC": "HEARTH_TEMP_D",
    "LOWER STACK - 18660_TEMP. OCAT-07DEG. OC": "LOWER_STACK_TEMP_A",
    "LOWER STACK - 18660_TEMP. OC.1BT-12DEG. OC": "LOWER_STACK_TEMP_B",
    "LOWER STACK - 18660_TEMP. OC.2CT-17DEG. OC": "LOWER_STACK_TEMP_C",
    "LOWER STACK - 18660_TEMP. OC.3DT-03DEG. OC": "LOWER_STACK_TEMP_D",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.4AVG.DEG. OC": "FTG_UPTAKE_TEMP_AVG",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.1BT-12DEG. OC": "FTG_UPTAKE_TEMP_B",
    "FURNACE TOP GAS_UPTAKE TEMP. OCAT-16DEG. OC": "FTG_UPTAKE_TEMP_A",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.2CT-08DEG. OC": "FTG_UPTAKE_TEMP_C",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.3DT-04DEG. OC": "FTG_UPTAKE_TEMP_D",
}

_THM_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "COKE_CALC_THM": ("COKE_CALC_MT",),
    "COKE_OFF_THM": ("COKE_OFF_MT",),
    "COKE_ON_THM": ("COKE_ON_MT",),
    "FLUX_CALC_THM": ("FLUX_CALC_MT",),
    "NUTCOKE_CALC_THM": ("NUTCOKE_CALC_MT",),
    "ORE_CALC_THM": ("ORE_CALC_MT",),
    "PCI_CALC_THM": ("PCI_CALC_MT", "PCI_2_CALC_MT"),
    "PCI_2_CALC_THM": ("PCI_2_CALC_MT", "PCI_CALC_MT"),
    "SINTER_CALC_THM": ("SINTER_CALC_MT",),
    "TOTAL_CLO_THM": ("ORE_CALC_MT",),
    "TOTAL_PELLET_CALC_THM": ("TOTAL_PELLET_CALC_MT",),
}

_PRODUCTION_ALIASES: tuple[str, ...] = (
    "PRODUCTIONTONNESPERHR",
    "PRODUCTIONTONNESPERHR.",
    "production_tonnes_per_hr",
    "productiontonnesperhr",
)


_CHEMISTRY_TO_TRAINING_SUFFIX: dict[str, str] = {
    "fe_t_pct": "FE(T)%",
    "moisture_pct": "TM%",
    "feo_pct": "FEO%",
    "sio2_pct": "SIO2%",
    "al2o3_pct": "AL2O3%",
    "cao_pct": "CAO%",
    "mgo_pct": "MGO%",
    "mno_pct": "MNO%",
    "tio2_pct": "TIO2%",
    "p_pct": "P%",
    "na2o_pct": "NA2O%",
    "k2o_pct": "K2O%",
}


@dataclass(frozen=True)
class ParsedLagFeature:
    """
    Represent a BMO lagged training feature.

    BMO V4 feature names encode both row lag and, for some columns, the physical
    impact channel. Keeping the parsed pieces in a small value object makes the
    feature resolver easier to reason about and test.

    Args:
         - base_name: str - Feature name before the lag suffix.
         - lag_steps: int - Number of lag steps encoded in the feature name.
         - impact_label: str | None - Optional physical impact label.

    Returns:
         - return ParsedLagFeature - Parsed lag metadata.
    """

    base_name: str
    lag_steps: int
    impact_label: str | None = None


def parse_bmo_lag_feature_name(feature_name: str) -> ParsedLagFeature | None:
    """
    Parse BMO lag names including dual-impact suffixes.

    Supports simple names such as ``STOCKRODLEVEL_lag1`` and BMO physical
    impact names such as ``ORE_3_PCT_lag4_(MeltImpact)``.

    Args:
         - feature_name: str - Feature name that may include lag metadata.

    Returns:
         - return ParsedLagFeature | None - Parsed lag feature or None.
    """

    match = _BMO_LAG_PATTERN.match(str(feature_name))
    if not match:
        return None
    return ParsedLagFeature(
        base_name=match.group(1),
        lag_steps=int(match.group(2)),
        impact_label=match.group(3),
    )


def max_bmo_lag_steps(feature_names: Iterable[str]) -> int:
    """Return the largest lag step encoded in BMO feature names."""

    max_lag = 0
    for feature in feature_names:
        parsed = parse_bmo_lag_feature_name(str(feature))
        if parsed is not None:
            max_lag = max(max_lag, parsed.lag_steps)
    return max_lag


def _is_sinter(ore: OreInput) -> bool:
    """
    Identify whether an ore input represents sinter material.

    The model feature payload treats sinter separately from CLO and pellet.
    This helper centralizes the loose naming check used by both display names
    and material keys from the ore mapping file.

    Args:
         - ore: OreInput - Ore input to classify.

    Returns:
         - return bool - True when the ore material or display name is sinter.
    """

    material_key = str(ore.metadata.get("material_key", "")).lower()
    return "sinter" in material_key or "sinter" in ore.display_name.lower()


def _is_pellet(ore: OreInput) -> bool:
    """
    Identify whether an ore input represents pellet material.

    Pellet burden has its own model features and ratios. This helper keeps the
    classification logic consistent whether the source name comes from display
    text or mapping metadata.

    Args:
         - ore: OreInput - Ore input to classify.

    Returns:
         - return bool - True when the ore material or display name is pellet.
    """

    material_key = str(ore.metadata.get("material_key", "")).lower()
    return "pellet" in material_key or "pellet" in ore.display_name.lower()


def _ore_slot_number(ore: OreInput) -> int | None:
    """
    Extract the training ore slot number from an ore material key.

    The trained model expects CLO chemistry and percentages in numbered
    ``ORE_1`` through ``ORE_12`` slots. This parser maps configured material keys
    back to those slot numbers.

    Args:
         - ore: OreInput - Ore input containing material metadata.

    Returns:
         - return int | None - Ore slot number from 1 to 12, or None.
    """

    material_key = str(ore.metadata.get("material_key", "")).lower()
    match = re.fullmatch(r"ore_(\d+)", material_key)
    if not match:
        return None
    slot = int(match.group(1))
    return slot if 1 <= slot <= 12 else None


def _weighted_chemistry(
    ores: list[OreInput], quantities_mt: Mapping[str, float], attr: str
) -> float | None:
    """
    Compute dry-quantity-weighted chemistry for selected ore inputs.

    Candidate blend chemistry is sent to the fuel-cost model in the training
    schema. This helper aggregates the selected ore chemistry using the same
    candidate quantities being evaluated by LP or DE.

    Args:
         - ores: list[OreInput] - Ores contributing to the weighted average.
         - quantities_mt: Mapping[str, float] - Ore quantities keyed by ore id.
         - attr: str - Chemistry attribute name to average.

    Returns:
         - return float | None - Weighted chemistry value, or None with no quantity.
    """

    total_qty = 0.0
    weighted = 0.0
    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        dry_qty = compute_dry_weight_mt(qty, ore.chemistry.moisture_pct)
        if dry_qty <= 0:
            continue
        value = getattr(ore.chemistry, attr, None)
        if value is None:
            continue
        total_qty += dry_qty
        weighted += dry_qty * float(value or 0.0)
    if total_qty <= 0:
        return None
    return weighted / total_qty


def _candidate_blend_features(
    quantities_mt: Mapping[str, float],
    ores: list[OreInput],
    hot_metal_target_mt: float | None = None,
) -> dict[str, float]:
    """
    Build training-schema burden features from a candidate BMO blend.

    These features let the BMO V4 fuel model see the candidate burden mix in
    the same naming style used during training. Moisture is included through
    ``TM%`` fields when it is present in the ore chemistry record.

    Args:
         - quantities_mt: Mapping[str, float] - Candidate ore quantities keyed by ore id.
         - ores: list[OreInput] - Ores selected for optimization.

    Returns:
         - return dict[str, float] - Raw burden features matching BMO V4 training names.
    """

    payload: dict[str, float] = {}
    ore_quantities_by_slot = {slot: 0.0 for slot in range(1, 13)}
    sinter_qty = 0.0
    pellet_qty = 0.0
    slot_ores: list[OreInput] = []
    hot_metal_mt = 0.0

    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        dry_qty = compute_dry_weight_mt(qty, ore.chemistry.moisture_pct)
        hot_metal_mt += compute_fe_contribution_mt(dry_qty, ore.chemistry.fe_t_pct)
        if _is_sinter(ore):
            sinter_qty += qty
            continue
        if _is_pellet(ore):
            pellet_qty += qty
            continue

        slot = _ore_slot_number(ore)
        if slot is not None:
            ore_quantities_by_slot[slot] += qty
            slot_ores.append(ore)

    ore_qty = float(sum(ore_quantities_by_slot.values()))
    payload["ORE_CALC_MT"] = ore_qty
    payload["SINTER_CALC_MT"] = sinter_qty
    payload["TOTAL_PELLET_CALC_MT"] = pellet_qty
    hm_basis_mt = float(hot_metal_target_mt or 0.0)
    if hm_basis_mt <= 0.0:
        hm_basis_mt = hot_metal_mt

    payload["ORE_CALC_THM"] = (ore_qty / hm_basis_mt) if hm_basis_mt > 0 else 0.0
    payload["TOTAL_CLO_THM"] = payload["ORE_CALC_THM"]
    payload["SINTER_CALC_THM"] = (
        (sinter_qty / hm_basis_mt) if hm_basis_mt > 0 else 0.0
    )
    payload["TOTAL_PELLET_CALC_THM"] = (
        (pellet_qty / hm_basis_mt) if hm_basis_mt > 0 else 0.0
    )

    for slot, qty in ore_quantities_by_slot.items():
        payload[f"ORE_{slot}_PCT"] = (qty / ore_qty * 100.0) if ore_qty > 0 else 0.0

    payload["SINTER_CLO_RATIO"] = (sinter_qty / ore_qty) if ore_qty > 0 else 0.0
    payload["PELLET_CLO_RATIO"] = (pellet_qty / ore_qty) if ore_qty > 0 else 0.0

    for attr, suffix in _CHEMISTRY_TO_TRAINING_SUFFIX.items():
        ore_value = _weighted_chemistry(slot_ores, quantities_mt, attr)
        if ore_value is not None:
            payload[f"ORE_{suffix}"] = ore_value

    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        if qty <= 0:
            continue
        if _is_sinter(ore):
            for attr, suffix in _CHEMISTRY_TO_TRAINING_SUFFIX.items():
                payload[f"SINTER_SP_02_{suffix}"] = float(
                    getattr(ore.chemistry, attr, 0.0) or 0.0
                )
        elif _is_pellet(ore):
            for attr, suffix in _CHEMISTRY_TO_TRAINING_SUFFIX.items():
                pellet_suffix = suffix.replace("%", "")
                value = float(getattr(ore.chemistry, attr, 0.0) or 0.0)
                payload[f"PELLET_PCT_{pellet_suffix}"] = value
                payload[f"LLOYDS_PELLET_PCT_{pellet_suffix}"] = value

    return payload


def build_feature_payload(
    quantities_mt: Mapping[str, float],
    ore_display_name_by_id: Mapping[str, str],
    process_context: Mapping[str, Any] | None = None,
    ores: list[OreInput] | None = None,
    hot_metal_target_mt: float | None = None,
) -> dict[str, float]:
    """
    Build a raw feature payload for BMO fuel-cost inference.

    The payload includes generic quantity/share features for the fallback
    formula and, when *ores* are supplied, the BMO V4 training-schema burden
    features used by the XGBoost model.

    Args:
         - quantities_mt: Mapping[str, float] - Candidate ore quantities keyed by ore id.
         - ore_display_name_by_id: Mapping[str, str] - Ore display names keyed by ore id.
         - process_context: Mapping[str, Any] | None - Latest process variables.
         - ores: list[OreInput] | None - Ore inputs used to build training features.

    Returns:
         - return dict[str, float] - Raw feature payload for model service inference.
    """

    payload: dict[str, float] = {}

    # Write process_context FIRST so operational variables (HOT BLAST TEMP,
    # COKE_ASH%, PRODUCTIONTONNESPERHR, etc.) populate the payload. The
    # candidate-derived burden features below then OVERWRITE any colliding
    # keys (ORE_CALC_MT, ORE_1_PCT, ORE_FE(T)%, SINTER_SP_02_*, etc.). This
    # ordering matters: process_context is the LATEST history row, so without
    # this swap the model would see yesterday's blend regardless of which
    # candidate the optimizer proposed -- fuel cost would be invariant to the
    # blend and DE would never improve on the LP seed.
    if process_context:
        for key, value in process_context.items():
            if value is None:
                continue
            try:
                payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
            payload[normalize_feature_name(str(key))] = payload[str(key)]

    total_qty = float(sum(float(v) for v in quantities_mt.values()))
    sinter_qty = 0.0

    for ore_id, qty in quantities_mt.items():
        qty = float(qty)
        display = ore_display_name_by_id.get(ore_id, ore_id)
        if "sinter" in display.lower():
            sinter_qty += qty

        ore_slug = normalize_feature_name(display)
        payload[f"{ore_slug}_qty_mt"] = qty
        payload[f"{ore_slug}_share_pct"] = (
            (qty / total_qty * 100.0) if total_qty > 0 else 0.0
        )

    sinter_share_pct = (sinter_qty / total_qty * 100.0) if total_qty > 0 else 0.0
    payload["sinter_qty_mt"] = sinter_qty
    payload["ore_qty_mt"] = max(0.0, total_qty - sinter_qty)
    payload["sinter_share_pct"] = sinter_share_pct
    payload["ore_share_pct"] = max(0.0, 100.0 - sinter_share_pct)
    payload["total_burden_qty_mt"] = total_qty

    if ores:
        payload.update(
            _candidate_blend_features(
                quantities_mt, ores, hot_metal_target_mt=hot_metal_target_mt
            )
        )

    return payload


def _numeric_or_none(value: Any) -> float | None:
    """
    Convert a value into float when it is present and numeric.

    Feature resolution tries payload, history, and defaults in order. Returning
    ``None`` for invalid values lets the caller continue to the next source
    without raising on blanks or non-numeric labels.

    Args:
         - value: Any - Candidate value from payload or history.

    Returns:
         - return float | None - Numeric float value, or None when missing/invalid.
    """

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_lookup(
    payload: Mapping[str, Any], key: str, aliases: Mapping[str, str]
) -> tuple[float | None, str | None]:
    """
    Look up a feature value in the candidate payload using aliases and normalization.

    Candidate payload keys can be raw training names or normalized process names.
    This helper checks both forms and returns a source label for diagnostics
    when a value is resolved.

    Args:
         - payload: Mapping[str, Any] - Candidate blend and process feature payload.
         - key: str - Requested training feature name.
         - aliases: Mapping[str, str] - Training-name aliases to alternate names.

    Returns:
         - return tuple[float | None, str | None] - Resolved value and source label.
    """

    candidates = [key]
    if key in aliases:
        candidates.append(aliases[key])

    norm_payload = {
        normalize_feature_name(str(name)): value for name, value in payload.items()
    }
    for candidate in candidates:
        if candidate in payload:
            value = _numeric_or_none(payload[candidate])
            if value is not None:
                return value, f"payload:{candidate}"
        norm_name = normalize_feature_name(candidate)
        if norm_name in norm_payload:
            value = _numeric_or_none(norm_payload[norm_name])
            if value is not None:
                return value, f"payload_norm:{candidate}"
    value, source = _derived_thm_payload_lookup(payload, key)
    if value is not None:
        return value, source
    return None, None


def _mapping_lookup(mapping: Mapping[str, Any], key: str) -> Any | None:
    """
    Look up a value from a mapping using raw and normalized feature names.

    Training features can arrive as exact notebook names, YAML aliases, or
    normalized process keys. This helper keeps derived feature construction from
    duplicating that matching logic.

    Args:
         - mapping: Mapping[str, Any] - Raw payload or row values.
         - key: str - Feature name to resolve.

    Returns:
         - return Any | None - Matching value, or None when unavailable.
    """

    if key in mapping:
        return mapping[key]
    normalized = normalize_feature_name(key)
    for candidate_key, value in mapping.items():
        if normalize_feature_name(str(candidate_key)) == normalized:
            return value
    return None


def _derived_thm_payload_lookup(
    payload: Mapping[str, Any], key: str
) -> tuple[float | None, str | None]:
    """
    Derive ``*_THM`` payload features from matching ``*_MT`` and production.

    The updated V4 notebook converts material quantities to THM basis using
    ``quantity_mt / PRODUCTIONTONNESPERHR``. Candidate burden features supply
    direct THM values, but process materials such as coke, flux, nut coke, and
    PCI may only be available as MT columns in the history payload.

    Args:
         - payload: Mapping[str, Any] - Candidate and process feature payload.
         - key: str - Requested training feature name.

    Returns:
         - return tuple[float | None, str | None] - Derived THM value and source label.
    """

    source_keys = _THM_SOURCE_ALIASES.get(key)
    if not source_keys:
        return None, None

    production_value = None
    for production_key in _PRODUCTION_ALIASES:
        production_value = _numeric_or_none(_mapping_lookup(payload, production_key))
        if production_value and production_value > 0:
            break
    if not production_value or production_value <= 0:
        return None, None

    for source_key in source_keys:
        source_value = _numeric_or_none(_mapping_lookup(payload, source_key))
        if source_value is not None:
            return (
                float(source_value) / float(production_value),
                f"payload_derived:{source_key}/PRODUCTIONTONNESPERHR",
            )
    return None, None


def _history_lookup(
    history_df: pd.DataFrame | None,
    key: str,
    aliases: Mapping[str, str],
    *,
    lag_steps: int = 0,
) -> tuple[float | None, str | None]:
    """
    Look up a current or lagged feature value from historical process data.

    The model uses the static dataset for lagged operating context. This helper
    finds the requested column by raw or normalized name, drops missing values,
    and returns the row matching the requested lag step.

    Args:
         - history_df: pd.DataFrame | None - Historical process data frame.
         - key: str - Requested training feature name.
         - aliases: Mapping[str, str] - Training-name aliases to alternate names.
         - lag_steps: int - Number of rows to lag from the latest non-null value.

    Returns:
         - return tuple[float | None, str | None] - Resolved value and source label.
    """

    if history_df is None or history_df.empty:
        return None, None

    candidates = [key]
    if key in aliases:
        candidates.append(aliases[key])

    columns_norm = {normalize_feature_name(str(c)): str(c) for c in history_df.columns}
    for candidate in candidates:
        column = candidate if candidate in history_df.columns else None
        if column is None:
            column = columns_norm.get(normalize_feature_name(candidate))
        if not column:
            continue

        series = pd.to_numeric(history_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        idx = max(0, int(lag_steps)) + 1
        if len(series) < idx:
            continue
        return float(series.iloc[-idx]), f"history:{column}:lag{lag_steps}"

    value, source = _derived_thm_history_lookup(history_df, key, lag_steps=lag_steps)
    if value is not None:
        return value, source
    return None, None


def _derived_thm_history_lookup(
    history_df: pd.DataFrame | None, key: str, *, lag_steps: int = 0
) -> tuple[float | None, str | None]:
    """
    Derive ``*_THM`` history features from ``*_MT`` and production history.

    The static history currently stores several burden quantities on MT basis,
    while the updated notebook's scaler expects THM-basis columns. This resolver
    reproduces the notebook conversion for current and lagged history rows.

    Args:
         - history_df: pd.DataFrame | None - Historical process data frame.
         - key: str - Requested THM training feature.
         - lag_steps: int - Number of rows to lag from the latest non-null value.

    Returns:
         - return tuple[float | None, str | None] - Derived value and source label.
    """

    if history_df is None or history_df.empty:
        return None, None

    source_keys = _THM_SOURCE_ALIASES.get(key)
    if not source_keys:
        return None, None

    columns_norm = {normalize_feature_name(str(c)): str(c) for c in history_df.columns}
    production_col = None
    for production_key in _PRODUCTION_ALIASES:
        production_col = (
            production_key
            if production_key in history_df.columns
            else columns_norm.get(normalize_feature_name(production_key))
        )
        if production_col:
            break
    if not production_col:
        return None, None

    for source_key in source_keys:
        source_col = (
            source_key
            if source_key in history_df.columns
            else columns_norm.get(normalize_feature_name(source_key))
        )
        if not source_col:
            continue

        source = pd.to_numeric(history_df[source_col], errors="coerce")
        production = pd.to_numeric(history_df[production_col], errors="coerce")
        derived = (source / production.replace(0, pd.NA)).dropna()
        if derived.empty:
            continue
        idx = max(0, int(lag_steps)) + 1
        if len(derived) < idx:
            continue
        return (
            float(derived.iloc[-idx]),
            f"history_derived:{source_col}/{production_col}:lag{lag_steps}",
        )
    return None, None


def _latest_history_timestamp(history_df: pd.DataFrame | None) -> pd.Timestamp | None:
    """
    Return the latest valid timestamp from the history frame index.

    Temporal model features depend on the history index rather than a normal
    numeric column. Invalid or missing indexes return ``None`` so temporal
    values can fall through to configured defaults.

    Args:
         - history_df: pd.DataFrame | None - Historical process data frame.

    Returns:
         - return pd.Timestamp | None - Latest valid timestamp, or None.
    """

    if history_df is None or history_df.empty:
        return None
    try:
        index = pd.to_datetime(history_df.index, errors="coerce")
        index = index[~pd.isna(index)]
        if len(index) == 0:
            return None
        return pd.Timestamp(index[-1])
    except Exception:
        return None


def _temporal_feature_value(
    feature: str, history_df: pd.DataFrame | None
) -> tuple[float | None, str | None]:
    """
    Resolve engineered temporal features from the history frame.

    The trained model includes calendar and trend fields that are not explicit
    process columns. This resolver derives those values from the history index
    when enough timestamp context is available.

    Args:
         - feature: str - Requested temporal feature name.
         - history_df: pd.DataFrame | None - Historical process data frame.

    Returns:
         - return tuple[float | None, str | None] - Temporal value and source label.
    """

    timestamp = _latest_history_timestamp(history_df)
    if feature == "trend_index" and history_df is not None and not history_df.empty:
        return float(max(len(history_df) - 1, 0)), "temporal:history_length"
    if timestamp is None:
        return None, None
    if feature == "month":
        return float(timestamp.month), "temporal:month"
    if feature == "week_of_year":
        return float(timestamp.isocalendar().week), "temporal:week_of_year"
    if feature == "day_of_year":
        return float(timestamp.dayofyear), "temporal:day_of_year"
    return None, None


def default_candidate_lag_bases(feature_payload: Mapping[str, Any]) -> set[str]:
    """
    Return feature bases whose physical-lag columns should use the candidate.

    These are the burden/blend-dependent model inputs. Process and fuel
    operating variables still resolve from history for their lagged values.

    Args:
         - feature_payload: Mapping[str, Any] - Candidate blend and process features.

    Returns:
         - return set[str] - Feature base names eligible for candidate lag override.
    """

    bases: set[str] = set()
    for key in feature_payload:
        text = str(key)
        if (
            re.fullmatch(r"ORE_\d+_PCT", text)
            or text
            in {
                "ORE_CALC_MT",
                "SINTER_CALC_MT",
                "TOTAL_PELLET_CALC_MT",
                "ORE_CALC_THM",
                "TOTAL_CLO_THM",
                "SINTER_CALC_THM",
                "TOTAL_PELLET_CALC_THM",
            }
            or text in {"SINTER_CLO_RATIO", "PELLET_CLO_RATIO"}
            or text.startswith("ORE_")
            or text.startswith("SINTER_SP_02_")
            or text.startswith("PELLET_")
            or text.startswith("LLOYDS_PELLET_")
        ):
            bases.add(text)
    return bases


def build_bmo_v4_feature_frame(
    *,
    feature_payload: Mapping[str, Any],
    history_df: pd.DataFrame | None,
    expected_features: list[str],
    default_values: Mapping[str, float] | None = None,
    candidate_lag_bases: set[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> FeatureBuildResult:
    """
    Build one raw BMO V4 feature row in scaler order.

    The V4 fuel-cost model requires a raw notebook-feature row before scaling. This
    function resolves each feature from the candidate blend payload, historical
    data, temporal context, or scaler-mean defaults. Candidate-sensitive burden
    features can intentionally override their lagged ``GasImpact`` and
    ``MeltImpact`` columns so the optimizer evaluates planned blend effects.

    Args:
         - feature_payload: Mapping[str, Any] - Candidate blend and process features.
         - history_df: pd.DataFrame | None - Historical process data for lag features.
         - expected_features: list[str] - Raw feature order expected by the scaler.
         - default_values: Mapping[str, float] | None - Defaults for unresolved features.
         - candidate_lag_bases: set[str] | None - Bases whose lag columns use candidate values.
         - aliases: Mapping[str, str] | None - Optional training-name alias overrides.

    Returns:
         - return FeatureBuildResult - Feature frame, imputation lists, and source map.
    """

    aliases = dict(_TRAINING_ALIASES if aliases is None else aliases)
    defaults = dict(default_values or {})
    lag_override_bases = (
        set(candidate_lag_bases)
        if candidate_lag_bases is not None
        else default_candidate_lag_bases(feature_payload)
    )

    values: list[float] = []
    missing_features: list[str] = []
    imputed_features: list[str] = []
    source_map: dict[str, str] = {}

    for feature in expected_features:
        value: float | None = None
        source: str | None = None

        temporal_value, temporal_source = _temporal_feature_value(feature, history_df)
        if temporal_value is not None:
            value, source = temporal_value, temporal_source

        parsed = parse_bmo_lag_feature_name(feature)
        if value is None and parsed is not None:
            alias_base = aliases.get(parsed.base_name)
            if (
                parsed.base_name in lag_override_bases
                or alias_base in lag_override_bases
            ):
                value, source = _payload_lookup(
                    feature_payload, parsed.base_name, aliases
                )
                if value is not None:
                    source = f"candidate_lag_override:{parsed.base_name}"
            if value is None:
                value, source = _history_lookup(
                    history_df,
                    parsed.base_name,
                    aliases,
                    lag_steps=parsed.lag_steps,
                )

        if value is None and parsed is None:
            value, source = _payload_lookup(feature_payload, feature, aliases)
            if value is None:
                value, source = _history_lookup(
                    history_df, feature, aliases, lag_steps=0
                )

        if value is None:
            missing_features.append(feature)
            value = float(defaults.get(feature, 0.0))
            source = "default"
            imputed_features.append(feature)

        values.append(float(value))
        source_map[feature] = str(source or "unknown")

    vector_df = pd.DataFrame([values], columns=expected_features, dtype=float)
    return FeatureBuildResult(
        vector_df=vector_df,
        missing_features=missing_features,
        imputed_features=imputed_features,
        source_map=source_map,
    )


def build_dynamic_payload(
    context: PreBuiltFeatureContext,
    quantities_mt: Mapping[str, float],
) -> dict[str, float]:
    """
    Build only the burden-dependent payload fields for one DE candidate.

    The full payload dict produced by ``build_feature_payload`` contains many
    fields driven by static process context. Inside the DE loop only the burden
    quantities and weighted chemistry change. This helper produces exactly the
    subset that the prebuilt feature template needs to overwrite, skipping the
    static process-context block entirely.

    Args:
         - context: PreBuiltFeatureContext - Prebuilt context with ore snapshot/display names.
         - quantities_mt: Mapping[str, float] - Candidate wet ore quantities keyed by ore id.

    Returns:
         - return dict[str, float] - Burden-dependent training-feature names mapped to values.
    """

    ores = context.ores_snapshot
    ore_display_name_by_id = context.ore_display_name_by_id

    payload: dict[str, float] = {}
    total_qty = float(sum(float(v) for v in quantities_mt.values()))
    sinter_qty_alias = 0.0
    for ore_id, qty in quantities_mt.items():
        qty_f = float(qty)
        display = ore_display_name_by_id.get(ore_id, ore_id)
        if "sinter" in display.lower():
            sinter_qty_alias += qty_f
        ore_slug = normalize_feature_name(display)
        payload[f"{ore_slug}_qty_mt"] = qty_f
        payload[f"{ore_slug}_share_pct"] = (
            (qty_f / total_qty * 100.0) if total_qty > 0 else 0.0
        )

    sinter_share_pct = (sinter_qty_alias / total_qty * 100.0) if total_qty > 0 else 0.0
    payload["sinter_qty_mt"] = sinter_qty_alias
    payload["ore_qty_mt"] = max(0.0, total_qty - sinter_qty_alias)
    payload["sinter_share_pct"] = sinter_share_pct
    payload["ore_share_pct"] = max(0.0, 100.0 - sinter_share_pct)
    payload["total_burden_qty_mt"] = total_qty

    payload.update(
        _candidate_blend_features(
            quantities_mt,
            ores,
            hot_metal_target_mt=context.hot_metal_target_mt,
        )
    )
    return payload


def _resolve_dynamic_col_indexes(
    expected_features: list[str], dynamic_field_names: Iterable[str]
) -> dict[str, tuple[int, ...]]:
    """
    Map each dynamic payload key to the template column indexes it overwrites.

    Burden payload keys can drive both their direct training column and the
    physical-lag override columns (e.g. ``ORE_1_PCT`` also overwrites
    ``ORE_1_PCT_lag1_(GasImpact)``). This resolver inspects every expected
    feature once at setup so the inner loop can update template positions by
    array index instead of repeating the alias and lag parsing.

    Args:
         - expected_features: list[str] - Full ordered scaler feature column names.
         - dynamic_field_names: Iterable[str] - Payload keys produced by build_dynamic_payload.

    Returns:
         - return dict[str, tuple[int, ...]] - Payload key -> tuple of column indexes.
    """

    direct_idx: dict[str, int] = {}
    norm_idx: dict[str, int] = {}
    base_to_lag_idxs: dict[str, list[int]] = {}
    for i, col in enumerate(expected_features):
        direct_idx[col] = i
        norm_idx.setdefault(normalize_feature_name(col), i)
        parsed = parse_bmo_lag_feature_name(col)
        if parsed is not None:
            base_to_lag_idxs.setdefault(parsed.base_name, []).append(i)

    aliases = _TRAINING_ALIASES
    reverse_aliases = {v: k for k, v in aliases.items()}

    result: dict[str, tuple[int, ...]] = {}
    for key in dynamic_field_names:
        idxs: list[int] = []
        seen: set[int] = set()

        candidates = [key]
        if key in aliases:
            candidates.append(aliases[key])
        if key in reverse_aliases:
            candidates.append(reverse_aliases[key])

        for cand in candidates:
            if cand in direct_idx and direct_idx[cand] not in seen:
                idxs.append(direct_idx[cand])
                seen.add(direct_idx[cand])
            norm = normalize_feature_name(cand)
            if norm in norm_idx and norm_idx[norm] not in seen:
                idxs.append(norm_idx[norm])
                seen.add(norm_idx[norm])
            for lag_idx in base_to_lag_idxs.get(cand, ()):
                if lag_idx not in seen:
                    idxs.append(lag_idx)
                    seen.add(lag_idx)

        if idxs:
            result[key] = tuple(idxs)
    return result


def build_prebuilt_feature_context(
    *,
    ores: list[OreInput],
    process_context: Mapping[str, Any] | None,
    history_df: "pd.DataFrame | None",
    expected_features: list[str],
    default_values: Mapping[str, float] | None = None,
    selected_features: list[str] | None = None,
    hot_metal_target_mt: float | None = None,
) -> PreBuiltFeatureContext:
    """
    Prebuild the static portion of the BMO model feature vector for DE.

    Resolves every expected feature once using the static process context and
    history frame, captures the resulting numpy template, and pre-resolves which
    template columns each dynamic payload key needs to overwrite. The returned
    context is reused by ``FuelUnitCostModelService.predict_with_prebuilt`` so
    every candidate evaluation reduces to a small numpy mutate + scaler +
    XGBoost predict instead of re-resolving 150+ feature columns.

    Args:
         - ores: list[OreInput] - Ores selected for the optimization run.
         - process_context: Mapping[str, Any] | None - Latest process variables.
         - history_df: pd.DataFrame | None - Historical process data for lag features.
         - expected_features: list[str] - Scaler-ordered raw feature names.
         - default_values: Mapping[str, float] | None - Defaults for unresolved features.
         - selected_features: list[str] | None - Post-scaler selected feature names.

    Returns:
         - return PreBuiltFeatureContext - Cached state for fast inner-loop inference.
    """

    import numpy as np  # local import keeps module-load cost low

    ore_display_name_by_id = {ore.ore_id: ore.display_name for ore in ores}
    zero_quantities = {ore.ore_id: 0.0 for ore in ores}
    # Use unit sentinels so the probe surfaces every dynamic key (weighted-
    # chemistry and per-material-type fields are gated on qty > 0).
    probe_quantities = {ore.ore_id: 1.0 for ore in ores}

    static_payload: dict[str, float] = {}
    if process_context:
        for key, value in process_context.items():
            if value is None:
                continue
            try:
                static_payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
            static_payload[normalize_feature_name(str(key))] = static_payload[str(key)]

    # Use the natural default_candidate_lag_bases derived from the static
    # payload so lag columns whose base names appear in process_context get
    # overridden with the process_context value -- mirrors what the slow path
    # does when the same key appears in feature_payload from process_context.
    template_build = build_bmo_v4_feature_frame(
        feature_payload=static_payload,
        history_df=history_df,
        expected_features=list(expected_features),
        default_values=dict(default_values or {}),
    )
    template_x = np.asarray(template_build.vector_df.values, dtype=float)

    probe_context = PreBuiltFeatureContext(
        expected_features=list(expected_features),
        template_x=template_x,
        dynamic_col_indexes={},
        dynamic_field_names=frozenset(),
        selected_features=selected_features,
        selected_indexes=None,
        ores_snapshot=list(ores),
        ore_display_name_by_id=dict(ore_display_name_by_id),
        static_payload=dict(static_payload),
        default_values=dict(default_values or {}),
        history_present=history_df is not None and not history_df.empty,
        hot_metal_target_mt=hot_metal_target_mt,
    )
    probe_payload = build_dynamic_payload(probe_context, probe_quantities)
    _ = zero_quantities  # retained for reference; not used post-probe

    # In the slow-path build_feature_payload, candidate-derived burden fields
    # are written AFTER process_context so they overwrite the historical
    # values on collision (e.g. ORE_CALC_MT, ORE_1_PCT, ORE_FE(T)%). The
    # prebuilt fast path mirrors this by writing every dynamic key it sees
    # over the template; no exclusion based on process_context is needed.
    dynamic_field_names = frozenset(probe_payload.keys())

    dynamic_col_indexes = _resolve_dynamic_col_indexes(
        list(expected_features), dynamic_field_names
    )

    selected_indexes = None
    if selected_features:
        col_to_idx = {col: i for i, col in enumerate(expected_features)}
        try:
            selected_indexes = np.array(
                [col_to_idx[col] for col in selected_features], dtype=int
            )
        except KeyError:
            selected_indexes = None

    return PreBuiltFeatureContext(
        expected_features=list(expected_features),
        template_x=template_x,
        dynamic_col_indexes=dynamic_col_indexes,
        dynamic_field_names=dynamic_field_names,
        selected_features=list(selected_features) if selected_features else None,
        selected_indexes=selected_indexes,
        ores_snapshot=list(ores),
        ore_display_name_by_id=dict(ore_display_name_by_id),
        static_payload=dict(static_payload),
        default_values=dict(default_values or {}),
        history_present=history_df is not None and not history_df.empty,
        hot_metal_target_mt=hot_metal_target_mt,
    )
